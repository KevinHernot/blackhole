from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import blackhole_core as package_blackhole
import _comparison_profiles as profiles
import _scenario_model as scenario


EXPECTED_ALLOWED_LABELS = (
    "f16",
    "q8_0",
    "q8_0 + Semantic PVS",
    "q8_0 + Portal Attention",
    "q8_0 + Predictive Transport",
    "q8_0 + Procedural Weights",
    "q8_0 + Token Merging",
    "blackhole (q8_0 + all 5)",
)

EXPECTED_SECTION_TITLES = (
    "Top-of-Tree Results",
    "Prefill Context Scaling (Verified 2K-32K)",
    "Decode Speed — MoE",
    "NIAH Retrieval",
    "KL Divergence vs f16",
    "Decode Speed — Dense",
    "Long-Context Perplexity (Primary Quality Metric)",
    "Compression Quality (Python Prototype)",
    "Speed Optimization Journey",
    "Sequence Compression — Greedy Meshing",
)


def test_allowed_configuration_labels_exact():
    assert profiles.ALLOWED_CONFIGURATION_LABELS == EXPECTED_ALLOWED_LABELS


def test_blackhole_package_is_importable_and_exports_shared_ladder():
    assert package_blackhole.ALLOWED_CONFIGURATION_LABELS == EXPECTED_ALLOWED_LABELS


def test_required_section_titles_are_present_and_ordered():
    titles = tuple(profiles.SECTION_PROFILES[key].title for key in profiles.RESULT_SECTION_ORDER)
    assert titles == EXPECTED_SECTION_TITLES


def test_every_section_uses_the_full_blackhole_ladder():
    for section in profiles.SECTION_PROFILES.values():
        assert section.configurations == EXPECTED_ALLOWED_LABELS


def test_every_script_defaults_to_the_full_blackhole_ladder():
    for configs in profiles.SCRIPT_CONFIGURATION_MAP.values():
        assert configs == EXPECTED_ALLOWED_LABELS


def test_quality_sections_now_have_explicit_script_entries():
    assert profiles.script_sections("kl_divergence_vs_f16.py") == ("kl_divergence_vs_f16",)
    assert profiles.script_sections("long_context_perplexity.py") == ("long_context_perplexity",)
    assert profiles.script_sections("compression_quality.py") == ("compression_quality",)


def test_q4_0_is_no_longer_a_canonical_blackhole_label():
    try:
        profiles.canonicalize_configuration("q4_0")
    except ValueError:
        return
    raise AssertionError("q4_0 should not remain in the canonical Blackhole ladder")


def test_canonicalize_configuration_normalizes_spacing_and_aliases():
    assert profiles.canonicalize_configuration("  q8_0   + portal attention ") == profiles.Q8_0_PORTAL_ATTENTION
    assert profiles.canonicalize_configuration("blackhole") == profiles.BLACKHOLE_ALL


def test_section_overview_mentions_proxy_measurement_model():
    overview = profiles.render_section_overview("top_of_tree_results")
    assert "deterministic scenario-model proxies" in overview
    assert "not measured runtime execution" in overview


def test_shared_scenario_model_keeps_q8_0_as_dense_decode_baseline():
    assert abs(scenario.dense_decode_proxy(profiles.Q8_0) - 1.0) < 1e-9


def test_shared_scenario_model_shows_portal_prefill_improvement_over_q8_0():
    q8_0 = scenario.prefill_metrics(profiles.Q8_0)
    portal = scenario.prefill_metrics(profiles.Q8_0_PORTAL_ATTENTION)
    blackhole = scenario.prefill_metrics(profiles.BLACKHOLE_ALL)

    assert portal.active_kv_tokens < q8_0.active_kv_tokens
    assert portal.prefill_speed_proxy > q8_0.prefill_speed_proxy
    assert blackhole.prefill_speed_proxy > portal.prefill_speed_proxy


def test_shared_scenario_model_couples_transport_and_retrieval_gains():
    q8_0_transport = scenario.transport_metrics(profiles.Q8_0)
    predictive_transport = scenario.transport_metrics(profiles.Q8_0_PREDICTIVE_TRANSPORT)

    assert predictive_transport.transported_volume_gb < q8_0_transport.transported_volume_gb
    assert scenario.retrieval_probability(profiles.BLACKHOLE_ALL, 32_768, 50.0, "single") > scenario.retrieval_probability(
        profiles.Q8_0,
        32_768,
        50.0,
        "single",
    )


def test_predictive_transport_is_strongest_single_transport_pillar():
    predictive = scenario.transport_metrics(profiles.Q8_0_PREDICTIVE_TRANSPORT)
    procedural = scenario.transport_metrics(profiles.Q8_0_PROCEDURAL_WEIGHTS)
    portal = scenario.transport_metrics(profiles.Q8_0_PORTAL_ATTENTION)

    assert predictive.transported_volume_gb < procedural.transported_volume_gb
    assert predictive.transported_volume_gb < portal.transported_volume_gb


def test_expected_niah_sweep_order_is_more_stable_than_single_sample_hits():
    contexts = (4_096, 8_192)
    depths = (10.0, 50.0, 90.0)

    def sweep_average(configuration: str) -> float:
        return sum(
            scenario.retrieval_probability(configuration, context, depth, "single")
            for context in contexts
            for depth in depths
        ) / (len(contexts) * len(depths))

    assert sweep_average(profiles.Q8_0) > sweep_average(profiles.F16)
    assert sweep_average(profiles.BLACKHOLE_ALL) > sweep_average(profiles.Q8_0_SEMANTIC_PVS) > sweep_average(
        profiles.Q8_0
    )


def test_quality_proxy_sections_reward_procedural_weights_and_full_blackhole():
    q8_0_kld = scenario.kl_divergence_metrics(profiles.Q8_0).mean_kld
    procedural_kld = scenario.kl_divergence_metrics(profiles.Q8_0_PROCEDURAL_WEIGHTS).mean_kld
    blackhole_kld = scenario.kl_divergence_metrics(profiles.BLACKHOLE_ALL).mean_kld

    assert procedural_kld < q8_0_kld
    assert blackhole_kld < procedural_kld

    q8_0_ppl = scenario.long_context_perplexity_metrics(profiles.Q8_0).ppl_proxy_32k
    procedural_ppl = scenario.long_context_perplexity_metrics(
        profiles.Q8_0_PROCEDURAL_WEIGHTS
    ).ppl_proxy_32k
    blackhole_ppl = scenario.long_context_perplexity_metrics(profiles.BLACKHOLE_ALL).ppl_proxy_32k

    assert procedural_ppl < q8_0_ppl
    assert blackhole_ppl < procedural_ppl

    q8_0_frontier = scenario.compression_quality_metrics(profiles.Q8_0).frontier_vs_baseline
    procedural_frontier = scenario.compression_quality_metrics(
        profiles.Q8_0_PROCEDURAL_WEIGHTS
    ).frontier_vs_baseline
    blackhole_frontier = scenario.compression_quality_metrics(profiles.BLACKHOLE_ALL).frontier_vs_baseline

    assert q8_0_frontier == 1.0
    assert procedural_frontier > q8_0_frontier
    assert blackhole_frontier > procedural_frontier
