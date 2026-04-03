from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _comparison_profiles as profiles
import _scenario_model as scenario


def _load_script_module(module_name: str):
    module_path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"blackhole_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"blackhole_{module_name}"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPRESSION_QUALITY = _load_script_module("compression_quality")
KL_DIVERGENCE = _load_script_module("kl_divergence_vs_f16")
LONG_CONTEXT_PERPLEXITY = _load_script_module("long_context_perplexity")
MEASURE_SKIP_RATE = _load_script_module("measure_skip_rate")
PREDICTIVE_TRANSPORT = _load_script_module("predictive_transport")
UNIFIED_POC = _load_script_module("unified_poc")


def _capture_stdout(callback) -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        callback()
    return stream.getvalue()


def test_scenario_model_metrics_stay_within_expected_bounds():
    contexts = (512, 2_048, 8_192, 32_768)
    depths = (0.0, 10.0, 50.0, 90.0, 100.0)

    for configuration in profiles.ALLOWED_CONFIGURATION_LABELS:
        prefill = scenario.prefill_metrics(configuration)
        assert 64 <= prefill.active_kv_tokens <= scenario.PREFILL_TOTAL_TOKENS
        assert 0.0 <= prefill.kv_reduction_fraction <= 1.0

        moe = scenario.moe_decode_metrics(configuration, context_length=8_192)
        assert 1 <= moe.values_processed <= scenario.MOE_TOTAL_TOKENS
        assert 0.0 <= moe.compute_reduction_fraction <= 1.0

        retrieval = scenario.retrieval_metrics(configuration)
        assert 1 <= retrieval.blocks_scanned <= scenario.RETRIEVAL_TOTAL_BLOCKS
        assert retrieval.needles_routed == scenario.RETRIEVAL_NEEDLES
        assert 0.0 < retrieval.base_accuracy <= 1.0

        transport = scenario.transport_metrics(configuration)
        assert 0.0 < transport.transported_volume_gb <= scenario.TRANSPORT_FP16_VOLUME_GB
        assert 0.0 <= transport.reduction_vs_f16 <= 1.0

        quality_drift = scenario.kl_divergence_metrics(configuration)
        assert quality_drift.mean_kld >= 0.0
        assert 0.0 <= quality_drift.same_top_p_fraction <= 1.0

        long_context = scenario.long_context_perplexity_metrics(configuration)
        assert long_context.ppl_proxy_32k > 0.0
        assert 0.0 < long_context.stability_fraction <= 1.0

        compression_quality = scenario.compression_quality_metrics(configuration)
        assert compression_quality.compression_ratio >= 1.0
        assert 0.0 <= compression_quality.quality_cosine <= 1.0
        assert compression_quality.mse_proxy >= 0.0
        assert compression_quality.frontier_vs_baseline >= 0.0

        for context in contexts:
            skip_rate = scenario.dense_skip_rate(configuration, context)
            assert scenario.DENSE_SKIP_RATE_MIN <= skip_rate <= scenario.DENSE_SKIP_RATE_MAX

            for depth in depths:
                single_probability = scenario.retrieval_probability(configuration, context, depth, "single")
                multi_key_probability = scenario.retrieval_probability(
                    configuration,
                    context,
                    depth,
                    "multi-key",
                    num_distractors=3,
                )
                multi_value_probability = scenario.retrieval_probability(
                    configuration,
                    context,
                    depth,
                    "multi-value",
                    value_count=4,
                )

                assert scenario.RETRIEVAL_PROBABILITY_MIN <= single_probability <= scenario.RETRIEVAL_PROBABILITY_MAX
                assert scenario.RETRIEVAL_PROBABILITY_MIN <= multi_key_probability <= scenario.RETRIEVAL_PROBABILITY_MAX
                assert scenario.RETRIEVAL_PROBABILITY_MIN <= multi_value_probability <= scenario.RETRIEVAL_PROBABILITY_MAX


def test_measure_skip_rate_rejects_non_integer_contexts_with_friendly_error():
    try:
        MEASURE_SKIP_RATE.main(["1024", "2048", "foo"])
    except SystemExit as exc:
        message = str(exc)
        assert "integers" in message
        assert "foo" in message
    else:
        raise AssertionError("Expected measure_skip_rate.main() to reject non-integer contexts")


def test_predictive_transport_main_executes_without_crashing():
    output = _capture_stdout(PREDICTIVE_TRANSPORT.main)
    assert "Predictive Transport Proof of Concept" in output
    assert "q8_0 + Predictive Transport" in output
    assert "blackhole (q8_0 + all 5)" in output


def test_quality_script_entry_points_execute_without_crashing():
    kld_output = _capture_stdout(KL_DIVERGENCE.main)
    assert "KL Divergence Proof of Concept" in kld_output
    assert "q8_0 + Procedural Weights" in kld_output

    perplexity_output = _capture_stdout(LONG_CONTEXT_PERPLEXITY.main)
    assert "Long-Context Perplexity Proof of Concept" in perplexity_output
    assert "blackhole (q8_0 + all 5)" in perplexity_output

    compression_output = _capture_stdout(COMPRESSION_QUALITY.main)
    assert "Compression Quality Proof of Concept" in compression_output
    assert "Frontier vs q8_0" in compression_output


def test_unified_poc_main_executes_without_crashing():
    output = _capture_stdout(UNIFIED_POC.main)
    assert "Unified Blackhole Proof of Concept" in output
    assert "q8_0 + Semantic PVS" in output
    assert "Section coverage catalog" in output
    assert "deterministic scenario-model proxies" in output


def test_documented_python3_script_entry_points_work_from_repo_root():
    unified = subprocess.run(
        ["python3", str(SCRIPTS_DIR / "unified_poc.py")],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert "Unified Blackhole Proof of Concept" in unified.stdout
    assert "deterministic scenario-model proxies" in unified.stdout

    niah_help = subprocess.run(
        ["python3", str(SCRIPTS_DIR / "niah_test.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert "Blackhole NIAH proof-of-concept runner." in niah_help.stdout

    for script_name, heading in (
        ("kl_divergence_vs_f16.py", "KL Divergence Proof of Concept"),
        ("long_context_perplexity.py", "Long-Context Perplexity Proof of Concept"),
        ("compression_quality.py", "Compression Quality Proof of Concept"),
    ):
        result = subprocess.run(
            ["python3", str(SCRIPTS_DIR / script_name)],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )
        assert heading in result.stdout
