"""
Test the modernized simulator implementation.
"""

import asyncio
from modern_simulator import (
    ModernNeuronSimulator,
    TokenByTokenSimulator,
    FewShotExample,
    simulate_and_score,
    compute_simulation_score
)


def test_basic_simulation():
    """Test basic synchronous simulation."""
    print("="*60)
    print("Test 1: Basic Synchronous Simulation")
    print("="*60)

    # Create a simulator for detecting numbers
    simulator = ModernNeuronSimulator(
        explanation="numeric values and digits",
        model_name="gpt-3.5-turbo"
    )

    # Test tokens
    tokens = ["The", "price", "is", "$", "42", "dollars"]

    # Simulate
    result = simulator.simulate(tokens)

    print(f"Tokens: {result.tokens}")
    print(f"Predicted activations: {result.expected_activations}")
    print(f"Activation scale: {result.activation_scale}")

    # Expected: high activation on "42", moderate on "$"
    print("\nExpected behavior: High activation on '42', moderate on '$'")


async def test_async_simulation():
    """Test asynchronous simulation."""
    print("\n" + "="*60)
    print("Test 2: Asynchronous Simulation")
    print("="*60)

    simulator = ModernNeuronSimulator(
        explanation="negative sentiment or words",
        model_name="gpt-3.5-turbo",
        use_async=True
    )

    tokens = ["I", "hate", "this", "bad", "movie"]

    result = await simulator.simulate_async(tokens)

    print(f"Tokens: {result.tokens}")
    print(f"Predicted activations: {result.expected_activations}")
    print("\nExpected behavior: High activations on 'hate' and 'bad'")


async def test_with_scoring():
    """Test simulation with scoring against actual activations."""
    print("\n" + "="*60)
    print("Test 3: Simulation with Scoring")
    print("="*60)

    explanation = "the token 'the'"
    tokens = ["In", "the", "beginning", "there", "was", "the", "word"]
    # Actual activations (high on "the")
    actual_activations = [0.1, 0.95, 0.05, 0.15, 0.05, 0.92, 0.08]

    simulation, score = await simulate_and_score(
        explanation=explanation,
        tokens=tokens,
        actual_activations=actual_activations,
        model_name="gpt-3.5-turbo"
    )

    print(f"Explanation: {explanation}")
    print(f"Tokens: {tokens}")
    print(f"Actual activations: {actual_activations}")
    print(f"Predicted activations: {simulation.expected_activations}")
    print(f"Correlation score: {score:.3f}")
    print("\nExpected: High score (>0.7) since explanation matches pattern")


async def test_token_by_token():
    """Test token-by-token simulation."""
    print("\n" + "="*60)
    print("Test 4: Token-by-Token Simulation")
    print("="*60)

    simulator = TokenByTokenSimulator(
        explanation="proper nouns and names",
        model_name="gpt-3.5-turbo"
    )

    tokens = ["John", "went", "to", "Paris", "yesterday"]

    result = await simulator.simulate(tokens)

    print(f"Tokens: {result.tokens}")
    print(f"Predicted activations: {result.expected_activations}")
    print("\nExpected behavior: High on 'John' and 'Paris'")


async def test_custom_examples():
    """Test with custom few-shot examples."""
    print("\n" + "="*60)
    print("Test 5: Custom Few-Shot Examples")
    print("="*60)

    # Create custom examples relevant to our domain
    custom_examples = [
        FewShotExample(
            explanation="mathematical operators",
            tokens=["+", "-", "*", "/", "="],
            activations=[10, 10, 10, 10, 8]
        ),
        FewShotExample(
            explanation="parentheses and brackets",
            tokens=["(", "x", "+", "y", ")"],
            activations=[10, 0, 0, 0, 10]
        ),
    ]

    simulator = ModernNeuronSimulator(
        explanation="equality and comparison operators",
        model_name="gpt-3.5-turbo",
        few_shot_examples=custom_examples,
        use_async=True
    )

    tokens = ["if", "x", "==", "y", "then"]

    result = await simulator.simulate_async(tokens)

    print(f"Tokens: {result.tokens}")
    print(f"Predicted activations: {result.expected_activations}")
    print("\nExpected behavior: High activation on '=='")


async def test_real_world_example():
    """
    Simulate a real-world scenario: evaluating if an explanation is good.
    """
    print("\n" + "="*60)
    print("Test 6: Real-World Evaluation Example")
    print("="*60)

    # Suppose we have a neuron that we think detects negation
    explanation = "negation words like 'not', 'never', 'no', etc."

    # Test cases
    test_cases = [
        {
            "tokens": ["I", "do", "not", "like", "this"],
            "actual": [0.1, 0.2, 0.95, 0.15, 0.1]
        },
        {
            "tokens": ["She", "never", "went", "there"],
            "actual": [0.1, 0.92, 0.15, 0.08]
        },
        {
            "tokens": ["The", "cat", "is", "sleeping"],
            "actual": [0.05, 0.08, 0.06, 0.1]  # No negation, low activation
        }
    ]

    scores = []
    for i, test_case in enumerate(test_cases):
        simulation, score = await simulate_and_score(
            explanation=explanation,
            tokens=test_case["tokens"],
            actual_activations=test_case["actual"],
            model_name="gpt-3.5-turbo"
        )

        print(f"\nTest case {i+1}:")
        print(f"  Tokens: {test_case['tokens']}")
        print(f"  Actual: {test_case['actual']}")
        print(f"  Predicted: {simulation.expected_activations}")
        print(f"  Score: {score:.3f}")

        scores.append(score)

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage score across all test cases: {avg_score:.3f}")
    print(f"Interpretation: {'Good' if avg_score > 0.7 else 'Moderate' if avg_score > 0.5 else 'Poor'} explanation quality")


async def test_comparison_with_simple_eval():
    """
    Compare with the simple_interp_eval approach.
    """
    print("\n" + "="*60)
    print("Test 7: Comparison with Simple Eval Approach")
    print("="*60)

    tokens = ["The", "number", "is", "42"]
    actual_activations = [0.1, 0.3, 0.1, 0.95]

    # Method 1: Using modern simulator (closer to original paper)
    simulator = ModernNeuronSimulator(
        explanation="numeric values",
        model_name="gpt-3.5-turbo",
        use_async=True
    )

    simulation = await simulator.simulate_async(tokens)
    score = compute_simulation_score(simulation, actual_activations)

    print("Method 1: Modern Simulator (Paper's approach)")
    print(f"  Predicted: {simulation.expected_activations}")
    print(f"  Score: {score:.3f}")

    # Method 2: Our simple approach (from simple_interp_eval.py)
    # Would need to import and run here for direct comparison
    print("\nMethod 2: Simple Eval Approach")
    print("  (Uses direct GPT prompting without few-shot examples)")
    print("  Both approaches should give similar scores for good explanations")


async def run_all_tests():
    """Run all tests."""
    print("Modern Neuron Simulator Test Suite")
    print("Using latest OpenAI Chat Completions API\n")

    # Synchronous test
    test_basic_simulation()

    # Async tests
    await test_async_simulation()
    await test_with_scoring()
    await test_token_by_token()
    await test_custom_examples()
    await test_real_world_example()
    await test_comparison_with_simple_eval()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
