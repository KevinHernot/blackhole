from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _comparison_profiles as profiles
import _scenario_model as scenario

def test_token_merging_reduces_transport_volume_vs_q8_0():
    q8_0_transport = scenario.transport_metrics(profiles.Q8_0)
    tome_transport = scenario.transport_metrics(profiles.Q8_0_TOKEN_MERGING)
    blackhole_transport = scenario.transport_metrics(profiles.BLACKHOLE_ALL)

    assert tome_transport.transported_volume_gb < q8_0_transport.transported_volume_gb
    
    # Blackhole (all 5) should be even better than Token Merging alone
    assert blackhole_transport.transported_volume_gb < tome_transport.transported_volume_gb

def test_token_merging_reduces_active_tokens():
    q8_0_prefill = scenario.prefill_metrics(profiles.Q8_0)
    tome_prefill = scenario.prefill_metrics(profiles.Q8_0_TOKEN_MERGING)
    
    assert tome_prefill.active_kv_tokens < q8_0_prefill.active_kv_tokens
    assert tome_prefill.kv_reduction_fraction > q8_0_prefill.kv_reduction_fraction

def test_token_merging_retrieval_impact():
    tome_prob_single = scenario.retrieval_probability(profiles.Q8_0_TOKEN_MERGING, 8192, 50.0, "single")
    q8_0_prob_single = scenario.retrieval_probability(profiles.Q8_0, 8192, 50.0, "single")
    assert tome_prob_single < q8_0_prob_single

    tome_prob_multi = scenario.retrieval_probability(profiles.Q8_0_TOKEN_MERGING, 8192, 50.0, "multi-value")
    q8_0_prob_multi = scenario.retrieval_probability(profiles.Q8_0, 8192, 50.0, "multi-value")
    assert tome_prob_multi > q8_0_prob_multi
