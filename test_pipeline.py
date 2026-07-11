"""
TokenForge Comprehensive Sanity Test Suite
Tests:
1. 8-Category Judge-Aligned Task Classification (code_gen, code_debug, ner, sentiment, factual, math, logic, summarization)
2. Dynamic Model Tiering (cheap / strong / code)
3. Tier 0 Local Deterministic Solvers
4. Output Sanitization & CoT Stripping
"""

import sys
import local_solvers
import router


def test_classifier():
    print("=== TESTING 8-CATEGORY JUDGE-ALIGNED CLASSIFIER ===")
    tests = {
        "Write a python function to compute fibonacci numbers": "code_gen",
        "def get_max(nums): return nums[0] fix this bug": "code_debug",
        "Extract named entities from: Google was founded in California": "ner",
        "What is the sentiment of this review: I love ROCm!": "sentiment",
        "What is the boiling point of water at sea level?": "factual",
        "Calculate 25 * 4 + 10": "math",
        "Three friends each own a different pet... deduce who owns the cat": "logic",
        "Summarize this paragraph in one sentence": "summarization",
    }
    passed = 0
    for prompt, expected in tests.items():
        result = router.classify_task(prompt)
        assert result == expected, f"Expected {expected} for prompt '{prompt}', got {result}"
        passed += 1
    print(f"PASSED {passed}/{len(tests)} task classifier checks.")


def test_model_scorer():
    print("=== TESTING DYNAMIC MODEL TIERING ===")
    fake_models = [
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/qwen2.5-coder-32b-instruct",
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
    ]
    tiers = router.resolve_model_tiers(fake_models)
    assert tiers["cheap"] == "accounts/fireworks/models/llama-v3p1-8b-instruct"
    assert tiers["code"] == "accounts/fireworks/models/qwen2.5-coder-32b-instruct"
    assert tiers["strong"] == "accounts/fireworks/models/llama-v3p1-70b-instruct"

    code_model = router.select_best_model("Write python code", fake_models, task_type="code_gen")
    assert code_model == tiers["code"], f"Expected {tiers['code']}, got {code_model}"

    sent_model = router.select_best_model("sentiment of review", fake_models, task_type="sentiment")
    assert sent_model == tiers["cheap"], f"Expected {tiers['cheap']}, got {sent_model}"
    print("PASSED dynamic model tiering checks.")


def test_local_solvers():
    print("=== TESTING TIER 0 LOCAL SOLVERS ===")
    assert local_solvers.solve("What is 144 / 12?") == "12"
    assert local_solvers.solve("Calculate 15% of 240") == "36"
    assert local_solvers.solve("Reverse the string \"hello\"") == "olleh"
    assert local_solvers.solve("What is the capital of France?") is None
    print("PASSED fail-closed deterministic solver checks.")


def test_output_sanitization():
    print("=== TESTING OUTPUT SANITIZATION ===")
    raw = "<think>some internal thought process</think>Here is the answer: 42"
    cleaned = router.sanitize_output(raw, task_type="factual")
    assert cleaned == "42", f"Expected '42', got '{cleaned}'"
    print("PASSED output sanitization checks.")


if __name__ == "__main__":
    test_classifier()
    test_model_scorer()
    test_local_solvers()
    test_output_sanitization()
    print("ALL TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
