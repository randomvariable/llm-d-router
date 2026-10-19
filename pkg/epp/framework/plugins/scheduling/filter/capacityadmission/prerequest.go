/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package capacityadmission

import (
	"context"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// PreRequest finalizes capacity accounting against the destination the
// scheduler actually picked, after every profile has run:
//
//   - Reference-only reads (Responses previous_response_id with empty
//     input) of an RTX-homed session are rejected with a clear typed error:
//     the engine cannot count that context, so it could only be admitted on
//     an unaccountable budget, and it must never be relocated.
//   - RTX destination: confirms the filter's reservation (idempotent
//     re-reserve as a safety net when the filter pass predates this
//     profile's outcome). For RTX-only (hardware alias) requests it takes
//     the reservation here and rejects request shapes the RTX engine cannot
//     account, with a clear typed error instead of silently placing them on
//     DGX.
//   - DGX destination: releases any reservation this request still holds
//     (e.g. a warm profile reserved RTX that the primary profile did not
//     select), and marks sessions that leave their RTX binding as relocated so
//     later turns stick to DGX rather than bouncing back.
//
// Reservations released here or at end-of-stream (see ResponseBody) are
// idempotent; the director guarantees the response hook runs on abort for any
// request that picked a pod, covering cancellation and failed forwarding.
func (p *CapacityAdmission) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) error {
	d := p.decisionFor(request)
	if d == nil || !d.governed {
		return nil
	}
	logger := log.FromContext(ctx)

	var primary *fwksched.ProfileRunResult
	if schedulingResult != nil {
		primary = schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	}
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		// No destination to confirm; a reservation this request took at the
		// filter stays held and is reclaimed by the request-context cancel
		// guard, the abort response hook, or idle-engine reconciliation.
		return nil
	}
	top := primary.TargetEndpoints[0]
	topIsRTX := p.isRTX(top)
	topID := p.rtxID(top)

	// Reference-only read of a session homed on RTX: neither force-admit
	// (unaccountable budget) nor relocate (context lives on RTX only). The
	// honest outcome is a clear rejection.
	if d.reference && d.wasRTX {
		p.release(d.reqKey)
		return errcommon.Error{
			Code: errcommon.PreconditionFailed,
			Msg:  fmt.Sprintf("capacity-admission: session is homed on RTX but the request only references stored state (%s); the RTX engine cannot count that context and the session must not be relocated", d.reason),
		}
	}

	if d.rtxOnly && !topIsRTX {
		// Defensive: the filter restricts alias traffic to RTX. Reaching a
		// DGX destination means a configuration fault; fail loudly rather
		// than silently violate the RTX-only policy.
		p.release(d.reqKey)
		return errcommon.Error{
			Code: errcommon.PreconditionFailed,
			Msg: fmt.Sprintf("capacity-admission: model %q is RTX-only but scheduling selected a non-RTX endpoint; refusing to fall back to DGX",
				request.TargetModel),
		}
	}

	if topIsRTX {
		if !d.countable {
			// No force-hold escape: a request whose budget the engine
			// cannot account must never forward to RTX, whether the shared
			// route reached this by misconfiguration or the alias pin has no
			// other home. Reference-only reads of RTX-homed sessions are
			// rejected earlier, independent of destination.
			p.release(d.reqKey)
			if d.reason.isCapacityExhaustion() {
				return errcommon.Error{
					Code: errcommon.ServiceUnavailable,
					Msg:  fmt.Sprintf("capacity-admission: RTX capacity unavailable for request %s (%s)", d.reqKey, d.reason),
				}
			}
			if d.rtxOnly {
				return errcommon.Error{
					Code: errcommon.BadRequest,
					Msg: fmt.Sprintf("capacity-admission: RTX cannot token-account this request (%s) and model %q is pinned to RTX; send this API shape to the common route instead",
						d.reason, request.TargetModel),
				}
			}
			return errcommon.Error{
				Code: errcommon.ServiceUnavailable,
				Msg:  fmt.Sprintf("capacity-admission: RTX cannot account this request (%s); it must not occupy the RTX budget", d.reason),
			}
		}

		if d.tooLarge {
			// Over the RTX per-request ceiling: for alias traffic this is
			// the honest rejection (there is no DGX escape); for shared
			// traffic the filter already dropped RTX, so reaching here is
			// defensive.
			p.release(d.reqKey)
			return errcommon.Error{
				Code: errcommon.BadRequest,
				Msg: fmt.Sprintf("capacity-admission: request %s reserves %d tokens above the RTX per-request ceiling %d",
					d.reqKey, d.tokens, p.params.MaxRequestTokens),
			}
		}

		res, holding := p.held(d.reqKey)
		if !holding || res.endpointID != topID {
			// Missing or misplaced hold (a lost race, a warm profile that
			// reserved another replica, or alias traffic the filter does not
			// reserve): re-book atomically against the final destination.
			if holding {
				p.release(d.reqKey)
			}
			if reason := p.reserve(d.reqKey, topID, d.tokens); reason != ReasonAllowed {
				return errcommon.Error{
					Code: errcommon.ServiceUnavailable,
					Msg: fmt.Sprintf("capacity-admission: RTX capacity exhausted for request %s (%s); no alternative endpoint applies for this route",
						d.reqKey, reason),
				}
			}
			d.reserved = true
			d.endpoint = topID
			p.guardCancel(ctx, d)
		}
		p.recordPlacement(d.sessionKey, topID, d.relocated, true)
		return nil
	}

	// --- DGX destination ---
	if released := p.release(d.reqKey); released {
		logger.V(logutil.DEBUG).Info("capacity-admission released RTX reservation for request routed to DGX",
			"requestID", d.reqKey, "model", request.TargetModel, "endpoint", topID)
	}
	if d.sessionKey != "" && !d.reference {
		if d.boundRTX && !d.relocated {
			// The session loses its RTX home this turn: make the move
			// sticky so later turns do not bounce back when capacity frees.
			exclusion := d.reason
			if exclusion == ReasonAllowed {
				exclusion = ReasonBudgetResident
			}
			logger.Info("capacity-admission relocated session to DGX",
				"requestID", d.reqKey, "model", request.TargetModel,
				"from", d.binding, "to", topID, "reason", exclusion)
		}
		p.recordPlacement(d.sessionKey, topID, d.relocated || (d.boundRTX && !d.reference), false)
	}
	return nil
}
