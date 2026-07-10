"""
TokenForge Comprehensive Sanity Test Suite
Tests:
1. Task classification across code, classification, short_fact, general
2. Generic model scoring against fake allowed_models
3. Fail-closed deterministic local solvers (Tier 0)
"""

import sys
import local_solvers
import router


def test_classifier():
    print("=== TESTING TASK CLASSIFIER ===")
    tests = {
        "Write a python function to compute fibonacci numbers": "code",
        "Explain the time complexity of this algorithm": "code",
        "Extract named entities from: Google was founded in California": "classification",
        "What is the sentiment of this review?": "classification",
        "True or false: AMD makes GPUs": "classification",
        "What is the boiling point of water at sea level?": "short_fact",
        "List the capital city of France": "short_fact",
        "Explain step-by-step why the sky is blue": "general",
    }
    passed = 0
    for prompt, expected in tests.items():
        result = router.classify_task(prompt)
        assert result == expected, f"Expected {expected} for prompt '{prompt}', got {result}"
        passed += 1
    print(f"PASSED {passed}/{len(tests)} task classifier checks.")


def test_model_scorer():
    print("=== TESTING GENERIC MODEL SCORER ===")
    fake_models = [
        "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "accounts/fireworks/models/qwen2.5-coder-32b-instruct",
        "accounts/fireworks/models/minimax-m3",
    ]
    code_model = router.select_best_model("Write python code", fake_models, task_type="code")
    assert "coder" in code_model or "code" in code_model, f"Expected coder model for code task, got {code_model}"

    fact_model = router.select_best_model("What is the capital?", fake_models, task_type="short_fact")
    assert fact_model in fake_models, f"Got unexpected model: {fact_model}"
    print("PASSED model scorer checks.")


def test_local_solvers():
    print("=== TESTING TIER 0 LOCAL SOLVERS ===")
    # Exact deterministic hits
    assert local_solvers.solve("What is 144 / 12?") == "12"
    assert local_solvers.solve("Calculate 15% of 240") == "36"
    assert local_solvers.solve("Reverse the string \"hello\"") == "olleh"

    # Fail-closed checks on ambiguous prompts
    assert local_solvers.solve("What is the capital of France?") is None
    assert local_solvers.solve("Solve equation: 2*x + 5 = 15") is None
    assert local_solvers.solve("Calculate roughly 10 percent") is None
    print("PASSED fail-closed deterministic solver checks.")


def test_output_sanitization():
    print("=== TESTING OUTPUT SANITIZATION ===")
    raw = "<think>some internal thought process</think>Here is the answer: 42"
    cleaned = router.sanitize_output(raw, task_type="short_fact")
    assert cleaned == "42", f"Expected '42', got '{cleaned}'"

    raw_class = "Positive.\nExtra note."
    cleaned_class = router.sanitize_output(raw_class, task_type="classification")
    assert cleaned_class == "Positive.", f"Expected 'Positive.', got '{cleaned_class}'"
    print("PASSED output sanitization checks.")


if __name__ == "__main__":
    test_classifier()
    test_model_scorer()
    test_local_solvers()
    test_output_sanitization()
    print("ALL TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
