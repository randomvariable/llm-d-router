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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// Filter runs before the scoring plugins (prefix cache affinity and
// active-request scoring included), so an over-budget RTX endpoint is out of
// the candidate set before any affinity preference can pull traffic toward
// it.
//
// When the RTX endpoint passes the full policy the filter atomically
// reserves the request against the shared budget. Reservation is idempotent
// per request identity, so repeated profile runs within one scheduling cycle
// (warm/test profiles, iterative profile handlers) neither double count nor
// race. A request that loses the reservation race has RTX dropped from its
// candidates and proceeds on DGX through the unchanged scorers - the bounded
// reschedule that keeps races off the client as an avoidable 503. PreRequest
// confirms the hold against the final destination and releases it when the
// request lands on DGX anyway. A cancel guard registered with the request
// context releases the hold if the request dies before a destination is
// captured (no response hook would then ever fire).
func (p *CapacityAdmission) Filter(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	d := p.decide(ctx, request, endpoints)
	if !d.governed {
		return endpoints
	}

	// Hardware alias (RTX-only model): the request must reach RTX or fail -
	// never silently land on DGX.
	if d.rtxOnly {
		return p.rtxOnlyEndpoints(ctx, request, d, endpoints)
	}

	rtx := p.findRTX(endpoints)

	// A relocated session stays on the exact DGX endpoint it moved to, as
	// long as that endpoint is among the candidates (the approved
	// no-bounce mechanism: not merely "not RTX"). If the bound DGX replica
	// is gone, fall through to normal DGX selection.
	if d.relocated && !d.reference && d.binding != "" {
		for _, ep := range endpoints {
			if !p.isRTX(ep) && p.epID(ep) == d.binding {
				return []fwksched.Endpoint{ep}
			}
		}
	}

	if rtx == nil {
		// Nothing to decide about: a DGX-only candidate set proceeds
		// untouched. This is also the path that keeps DGX fully working
		// when RTX is scaled away, down, or misconfigured.
		return endpoints
	}

	// The startup gate is evaluated from the endpoint's live metrics on
	// every governed request: observing an idle, fresh snapshot opens it, and
	// a decision short-circuited while the gate was closed is recomputed so
	// the request can still reach RTX in this same cycle.
	if reason := p.checkRTXMetrics(ctx, rtx); reason != ReasonAllowed {
		return p.excludeRTX(ctx, request, d, rtx, reason, endpoints)
	}
	if d.reason == ReasonStartupGate {
		d = p.forceDecide(ctx, request, endpoints)
	}

	if reason := p.rtxAdmission(ctx, request, d, rtx); reason != ReasonAllowed {
		return p.excludeRTX(ctx, request, d, rtx, reason, endpoints)
	}
	return endpoints
}

// excludeRTX drops RTX endpoints from the candidate set, logging why.
func (p *CapacityAdmission) excludeRTX(ctx context.Context, request *fwksched.InferenceRequest, d *decision, rtx fwksched.Endpoint, reason Reason, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	log.FromContext(ctx).V(logutil.DEBUG).Info("capacity-admission excluded RTX endpoint",
		"requestID", d.reqKey, "model", request.TargetModel,
		"endpoint", p.rtxID(rtx), "reason", reason)
	filtered := make([]fwksched.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if !p.isRTX(ep) {
			filtered = append(filtered, ep)
		}
	}
	return filtered
}

// rtxOnlyEndpoints keeps only RTX candidates for a hardware-alias request
// and applies the same evidence gates as the shared route: the startup
// idle-observation (until RTX is proven drained since this process started)
// and the engine-reported running/waiting saturation. Alias token accounting
// is taken at PreRequest (no bypass); requests the engine cannot count are
// rejected there, not force-admitted.
func (p *CapacityAdmission) rtxOnlyEndpoints(ctx context.Context, request *fwksched.InferenceRequest, d *decision, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	var kept []fwksched.Endpoint
	var rtx fwksched.Endpoint
	for _, ep := range endpoints {
		if p.isRTX(ep) {
			kept = append(kept, ep)
			if rtx == nil {
				rtx = ep
			}
		}
	}
	if len(kept) == 0 {
		log.FromContext(ctx).Info("capacity-admission excluded all candidates: RTX-only model without an RTX endpoint",
			"requestID", d.reqKey, "model", request.TargetModel, "reason", string(ReasonRTXOnlyNoCandidate))
		return nil
	}

	// Same evidence gate as the shared route: alias traffic never lands on
	// an RTX engine this process has not observed drained (fresh idle
	// metrics), because pre-existing tokens would be unaccounted.
	if reason := p.checkRTXMetrics(ctx, rtx); reason != ReasonAllowed {
		log.FromContext(ctx).Info("capacity-admission excluded RTX for RTX-only request",
			"requestID", d.reqKey, "model", request.TargetModel,
			"endpoint", p.rtxID(rtx), "reason", reason)
		return nil
	}

	// Concurrency gate from the engine's own report: the reservation table
	// restarts empty, the engine's load does not.
	m := rtx.GetMetrics()
	if p.metricsUsable(m, time.Now()) && m.RunningRequestsSize+m.WaitingQueueSize >= p.params.MaxConcurrentRequests {
		log.FromContext(ctx).Info("capacity-admission excluded RTX for RTX-only request: engine saturated",
			"requestID", d.reqKey, "model", request.TargetModel,
			"endpoint", p.rtxID(rtx), "reason", string(ReasonBudgetConcurrency))
		return nil
	}
	return kept
}

// rtxAdmission applies the shared-route capacity policy for a request whose
// gate has already been checked: sticky relocation, countability, per-request
// ceiling, and the atomic budget reservation. ReasonAllowed means the hold
// is recorded for this request.
func (p *CapacityAdmission) rtxAdmission(ctx context.Context, _ *fwksched.InferenceRequest, d *decision, rtx fwksched.Endpoint) Reason {
	if d.relocated && !d.reference {
		return ReasonRelocated
	}
	if !d.countable {
		return d.reason
	}
	if d.tooLarge {
		return ReasonRequestTooLarge
	}
	return p.reserveFor(ctx, d, rtx)
}

// reserveFor takes (or idempotently confirms) the reservation for the
// request, registers the cancel guard, and returns the exclusion reason;
// ReasonAllowed means the hold is recorded.
func (p *CapacityAdmission) reserveFor(ctx context.Context, d *decision, rtx fwksched.Endpoint) Reason {
	epID := p.rtxID(rtx)
	if epID == "" {
		epID = d.endpoint
	}
	reason := p.reserve(d.reqKey, epID, d.tokens)
	if reason == ReasonAllowed {
		d.reserved = true
		d.endpoint = epID
		p.guardCancel(ctx, d)
	}
	return reason
}
