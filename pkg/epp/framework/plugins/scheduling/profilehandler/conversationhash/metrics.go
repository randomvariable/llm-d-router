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

package conversationhash

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var profileSelections = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
		Name:      "conversation_hash_profile_selections_total",
		Help: metricsutil.HelpMsgWithStability(
			"Scheduling profiles selected by the conversation hash profile handler, by profile.",
			compbasemetrics.ALPHA),
	},
	[]string{"profile"},
)

// registerMetrics registers the handler collectors. A nil registerer is a no-op,
// so the handler still runs when the handle supplies no recorder; repeated
// registration of the same collector is tolerated.
func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	if err := registerer.Register(profileSelections); err != nil {
		var alreadyRegistered prometheus.AlreadyRegisteredError
		if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == profileSelections {
			return nil
		}
		return fmt.Errorf("register conversation hash profile handler metric: %w", err)
	}
	return nil
}

func recordProfileSelection(profile string) {
	profileSelections.WithLabelValues(profile).Inc()
}

// resetMetrics clears the collector between tests.
func resetMetrics() {
	profileSelections.Reset()
}
