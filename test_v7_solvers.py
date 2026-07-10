"""Test script for TokenForge v7.0 local solvers and classifier."""

from router import detect_category
from local_solvers import solve_math, solve_sentiment, solve_ner, solve_logic, solve_code_gen, solve_code_debug

# Test classifier
tests = {
    "What is the capital of France?": "factual",
    "Calculate 15% of 240": "math",
    "Classify the sentiment of this review": "sentiment",
    "Summarize the following paragraph": "summarization",
    "Extract all named entities from the text": "ner",
    "This function has a bug: def get_max": "code_debug",
    "Three friends each own a different pet. Sam does not own the bird": "logical",
    "Write a Python function that returns the second-largest number": "code_gen",
}
print("=== CLASSIFIER TEST ===")
passed = 0
for prompt, expected in tests.items():
    result = detect_category(prompt)
    status = "OK" if result == expected else "FAIL"
    if result == expected:
        passed += 1
    print(f"  {status}: '{prompt[:55]}' -> {result} (expected: {expected})")
print(f"  Result: {passed}/{len(tests)} passed")

# Test math solver
print("\n=== MATH SOLVER ===")
print(f"  144/12 = {solve_math('What is 144 / 12?')}")
print(f"  15% of 240 = {solve_math('Calculate 15% of 240')}")
print(f"  5 + 3 = {solve_math('5 + 3')}")

# Test sentiment solver
print("\n=== SENTIMENT SOLVER ===")
r = solve_sentiment("Classify the sentiment of this review: The battery life is great, but the screen scratches too easily.")
print(f"  Mixed review: {r}")
r2 = solve_sentiment("Classify the sentiment of this review: This product is absolutely amazing and wonderful!")
print(f"  Positive review: {r2}")
r3 = solve_sentiment("Classify the sentiment of this review: Terrible quality, worst purchase ever.")
print(f"  Negative review: {r3}")

# Test NER solver
print("\n=== NER SOLVER ===")
r = solve_ner("Extract all named entities and their types from: Maria Sanchez joined Fireworks AI in Berlin last March.")
print(f"  NER result:\n{r}")

# Test logic solver
print("\n=== LOGIC SOLVER ===")
r = solve_logic("Three friends, Sam, Jo, and Lee, each own a different pet: cat, dog, bird. Sam does not own the bird. Jo owns the dog. Who owns the cat?")
print(f"  Logic result: {r}")

# Test code gen solver
print("\n=== CODE GEN SOLVER ===")
r = solve_code_gen("Write a Python function that returns the second-largest number in a list, handling duplicates correctly.")
print(f"  Code gen result: {r}")

# Test code debug solver
print("\n=== CODE DEBUG SOLVER ===")
r = solve_code_debug("This function should return the max of a list but has a bug: def get_max(nums): return nums[0]. Find and fix it.")
print(f"  Code debug result: {r}")

# Test prompt compressor
print("\n=== PROMPT COMPRESSOR ===")
import prompt_compressor
original = "Could you please write a Python function that returns the second-largest number in a list, if possible? Thanks in advance."
compressed = prompt_compressor.compress(original)
print(f"  Original ({len(original)} chars): {original}")
print(f"  Compressed ({len(compressed)} chars): {compressed}")

print("\n=== ALL TESTS COMPLETE ===")
