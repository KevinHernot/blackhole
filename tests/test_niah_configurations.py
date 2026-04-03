from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

MODULE_PATH = SCRIPTS_DIR / "niah_test.py"
SPEC = importlib.util.spec_from_file_location("blackhole_niah", MODULE_PATH)
NIAH = importlib.util.module_from_spec(SPEC)
sys.modules["blackhole_niah"] = NIAH
SPEC.loader.exec_module(NIAH)


EXPECTED_CONFIGS = (
    "f16",
    "q8_0",
    "q8_0 + Semantic PVS",
    "q8_0 + Portal Attention",
    "q8_0 + Predictive Transport",
    "q8_0 + Procedural Weights",
    "q8_0 + Token Merging",
    "blackhole (q8_0 + all 5)",
)


def test_parse_args_defaults_to_full_blackhole_ladder():
    args = NIAH.parse_args(["/tmp/llama.cpp", "/tmp/model.gguf"])
    assert args.configs == ",".join(EXPECTED_CONFIGS)
    assert args.cache_types == args.configs
    assert args.output_dir is None


def test_resolve_output_dir_defaults_to_repo_root():
    assert NIAH._resolve_output_dir(None) == PROJECT_ROOT / "niah_results_poc"


def test_resolve_output_dir_makes_relative_paths_repo_root_relative():
    assert NIAH._resolve_output_dir("custom-output") == PROJECT_ROOT / "custom-output"


def test_parse_requested_configurations_accepts_canonical_blackhole_labels():
    configs = NIAH._parse_requested_configurations("q8_0, q8_0 + Semantic PVS")
    assert configs == [
        "q8_0",
        "q8_0 + Semantic PVS",
    ]


def test_ordered_result_configurations_prefers_full_ladder_order():
    results = [
        NIAH.ConfigResult(mode="single", context_length=4096, cache_type="q8_0 + Portal Attention"),
        NIAH.ConfigResult(mode="single", context_length=4096, cache_type="q8_0"),
    ]
    assert NIAH._ordered_result_configurations(results) == [
        "q8_0",
        "q8_0 + Portal Attention",
    ]


def test_build_output_includes_niah_section_overview():
    result = NIAH.ConfigResult(mode="single", context_length=4096, cache_type="q8_0")
    result.trials.append(
        NIAH.TrialResult(
            expected="1234567",
            response="1234567",
            found=True,
            expected_probability=0.75,
        )
    )
    output = NIAH.build_output([result], "test-model", "single")
    assert "NIAH Retrieval" in output
    assert "Proof-of-concept ladder:" in output
    assert "Common baseline: q8_0" in output
    assert "Any hit/miss examples below are illustrative deterministic draws" in output
    assert "Expected retrieval summary" in output


def test_config_result_tracks_observed_and_expected_accuracy_separately():
    result = NIAH.ConfigResult(mode="multi-value", context_length=4096, cache_type="q8_0")
    result.trials.extend(
        [
            NIAH.TrialResult(expected="1111111", response="1111111", found=True, expected_probability=0.90),
            NIAH.TrialResult(expected="2222222", response="9999999", found=False, expected_probability=0.50),
        ]
    )

    assert result.accuracy_pct == 50.0
    assert result.expected_accuracy_pct == 70.0


def test_build_output_summary_uses_expected_accuracy_not_sample_hits():
    result = NIAH.ConfigResult(mode="multi-key", context_length=4096, cache_type="q8_0")
    result.trials.append(
        NIAH.TrialResult(
            expected="1234567",
            response="7654321",
            found=False,
            expected_probability=0.80,
        )
    )

    output = NIAH.build_output([result], "test-model", "multi-key")

    assert "Expected average accuracy" in output
    assert "| q8_0 | 80.0% | baseline | Quantized baseline before Blackhole-specific improvements. |" in output


def test_simulate_multi_value_result_keeps_per_trial_expected_probabilities():
    result = NIAH._simulate_multi_value_result("q8_0", 4096, 4)

    expected = [
        NIAH.retrieval_probability(
            "q8_0",
            4096,
            ((index + 1) / 5) * 100.0,
            "multi-value",
            value_count=4,
        )
        * 100.0
        for index in range(4)
    ]

    assert [round(trial.expected_probability * 100.0, 6) for trial in result.trials] == [
        round(value, 6) for value in expected
    ]
    assert round(result.expected_accuracy_pct, 6) == round(sum(expected) / len(expected), 6)


def test_wrong_number_fallback_returns_remaining_candidate_without_looping_forever():
    original_min = NIAH.SYNTHETIC_NUMBER_MIN
    original_max = NIAH.SYNTHETIC_NUMBER_MAX
    original_attempts = NIAH.WRONG_NUMBER_RANDOM_ATTEMPTS
    try:
        NIAH.SYNTHETIC_NUMBER_MIN = 10
        NIAH.SYNTHETIC_NUMBER_MAX = 12
        NIAH.WRONG_NUMBER_RANDOM_ATTEMPTS = 0

        assert NIAH._wrong_number({"10", "11"}, "forced-fallback") == "12"
    finally:
        NIAH.SYNTHETIC_NUMBER_MIN = original_min
        NIAH.SYNTHETIC_NUMBER_MAX = original_max
        NIAH.WRONG_NUMBER_RANDOM_ATTEMPTS = original_attempts


def test_wrong_number_raises_cleanly_when_number_space_is_exhausted():
    original_min = NIAH.SYNTHETIC_NUMBER_MIN
    original_max = NIAH.SYNTHETIC_NUMBER_MAX
    original_attempts = NIAH.WRONG_NUMBER_RANDOM_ATTEMPTS
    try:
        NIAH.SYNTHETIC_NUMBER_MIN = 10
        NIAH.SYNTHETIC_NUMBER_MAX = 12
        NIAH.WRONG_NUMBER_RANDOM_ATTEMPTS = 0

        try:
            NIAH._wrong_number({"10", "11", "12"}, "exhausted-space")
        except ValueError as exc:
            assert "exhaust" in str(exc).lower()
        else:
            raise AssertionError("Expected _wrong_number to fail cleanly when every synthetic value is forbidden")
    finally:
        NIAH.SYNTHETIC_NUMBER_MIN = original_min
        NIAH.SYNTHETIC_NUMBER_MAX = original_max
        NIAH.WRONG_NUMBER_RANDOM_ATTEMPTS = original_attempts
