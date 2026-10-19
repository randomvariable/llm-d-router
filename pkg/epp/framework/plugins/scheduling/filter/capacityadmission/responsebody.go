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

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// ResponseBody releases the RTX reservation at end-of-stream. The director
// also forces it on abort (client disconnect, cancellation, failed
// forwarding) for every request that picked a pod, so the hold cannot leak
// past the request's lifecycle. Release is idempotent: repeated or racing
// end-of-stream calls free capacity at most once.
func (p *CapacityAdmission) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	if response == nil || !response.EndOfStream {
		return
	}
	d := p.decisionFor(request)
	if d == nil || !d.governed {
		return
	}
	if p.release(d.reqKey) {
		cause := response.TerminationCause
		if cause == "" {
			cause = fwkrc.TerminationCauseNatural
		}
		log.FromContext(ctx).V(logutil.DEBUG).Info("capacity-admission released RTX reservation",
			"requestID", d.reqKey, "endpoint", d.endpoint, "tokens", d.tokens,
			"termination", cause)
	}
	d.reserved = false
	if d.stopGuard != nil {
		// Completed (or aborted) request: stop the cancel guard so no
		// callback lingers on the request context.
		d.stopGuard()
		d.stopGuard = nil
	}
	// Keep the session binding warm for the next turn.
	p.touchPlacement(d.sessionKey)
}
