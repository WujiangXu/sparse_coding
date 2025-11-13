"""
Test script to verify if neuron-explainer library works with current OpenAI API.

This script tests:
1. Basic neuron-explainer simulator functionality
2. OpenAI API compatibility with neuron-explainer

Run: python test_neuron_explainer_api.py
"""

import os
import sys
import json
import asyncio
from typing import List

# Load OpenAI API key
try:
    with open("secrets.json") as f:
        secrets = json.load(f)
        os.environ["OPENAI_API_KEY"] = secrets["openai_key"]
        print(f"✅ Loaded OpenAI API key from secrets.json")
except Exception as e:
    print(f"❌ Failed to load API key: {e}")
    sys.exit(1)

# Import neuron-explainer components
try:
    from neuron_explainer.activations.activations import ActivationRecord
    from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
    from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
    from neuron_explainer.explanations.prompt_builder import PromptFormat
    from neuron_explainer.explanations.scoring import simulate_and_score
    print(f"✅ Successfully imported neuron-explainer modules")
except ImportError as e:
    print(f"❌ Failed to import neuron-explainer: {e}")
    sys.exit(1)

# Check OpenAI version
try:
    import openai
    openai_version = openai.__version__
    print(f"✅ OpenAI library version: {openai_version}")
except Exception as e:
    print(f"⚠️  Could not determine OpenAI version: {e}")
    openai_version = "unknown"


async def test_neuron_explainer_simulator(model_name: str, explanation: str, test_tokens: List[str], test_activations: List[float]):
    """
    Test neuron-explainer simulator with given model.

    Args:
        model_name: Model to test (e.g., "gpt-3.5-turbo-instruct", "text-davinci-003")
        explanation: Feature explanation text
        test_tokens: List of test tokens
        test_activations: List of activation values (same length as test_tokens)
    """
    print(f"\n{'='*80}")
    print(f"Testing neuron-explainer simulator with model: {model_name}")
    print(f"{'='*80}")

    try:
        # Create activation records
        print(f"1. Creating test ActivationRecord...")

        if len(test_tokens) != len(test_activations):
            raise ValueError(f"Tokens and activations must have same length (got {len(test_tokens)} and {len(test_activations)})")

        activation_record = ActivationRecord(
            tokens=test_tokens,
            activations=test_activations
        )
        print(f"   ✅ Created ActivationRecord with {len(test_tokens)} tokens")
        print(f"      Tokens: {test_tokens}")
        print(f"      Activations: {test_activations}")

        # Determine prompt format
        if model_name == "gpt-3.5-turbo":
            prompt_format = PromptFormat.HARMONY_V4
        else:
            prompt_format = PromptFormat.INSTRUCTION_FOLLOWING

        print(f"\n2. Creating simulator...")
        print(f"   Model: {model_name}")
        print(f"   Prompt format: {prompt_format}")
        print(f"   Explanation: '{explanation}'")

        # Create simulator
        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                model_name,
                explanation,
                max_concurrent=None,
                prompt_format=prompt_format
            )
        )
        print(f"   ✅ Simulator created successfully")

        # Run simulation
        print(f"\n3. Running simulation...")
        print(f"   This will call OpenAI API with the neuron-explainer library...")

        scored_simulation = await simulate_and_score(
            simulator,
            [activation_record]
        )

        # Get results
        score = scored_simulation.get_preferred_score()
        print(f"   ✅ Simulation completed successfully!")
        print(f"      Pearson correlation score: {score:.4f}")

        # Additional info
        if hasattr(scored_simulation, 'ev_correlation_score'):
            ev_score = scored_simulation.ev_correlation_score
            print(f"      EV correlation score: {ev_score:.4f}")

        return True, score, None

    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__

        print(f"   ❌ Simulation failed!")
        print(f"      Error type: {error_type}")
        print(f"      Error message: {error_msg}")

        # Check for specific error patterns
        if "logprobs" in error_msg.lower() and "echo" in error_msg.lower():
            print(f"\n   💡 Analysis: This model doesn't support 'echo' + 'logprobs' combination")
            print(f"      This is REQUIRED for neuron-explainer to work")
        elif "deprecated" in error_msg.lower():
            print(f"\n   💡 Analysis: This model has been deprecated by OpenAI")
        elif "not found" in error_msg.lower():
            print(f"\n   💡 Analysis: This model is not available in your OpenAI account")

        # Print detailed traceback for debugging
        print(f"\n   Detailed traceback:")
        import traceback
        traceback.print_exc()

        return False, None, error_msg


async def main():
    """
    Main test function.
    """
    print(f"\n{'='*80}")
    print(f"NEURON-EXPLAINER COMPATIBILITY TEST")
    print(f"{'='*80}")
    print(f"\nThis script tests if the neuron-explainer library works with current OpenAI API.")
    print(f"Specifically, it tests the simulator used in the paper:")
    print(f"'Sparse Autoencoders Find Highly Interpretable Features in Language Models'")

    # Test data
    test_explanation = "This feature detects words starting with 'dis-' prefix."
    test_tokens = ["The", " word", " diss", "olve", " appears"]
    test_activations = [0.1, 0.2, 2.5, 1.8, 0.3]  # High activation on " diss"

    # Models to test (in order of preference)
    models_to_test = [
        ("gpt-3.5-turbo-instruct", "Currently the only OpenAI model supporting echo+logprobs"),
        ("text-davinci-003", "Original paper's model (likely deprecated)"),
        ("text-davinci-002", "Older model (likely deprecated)"),
    ]

    print(f"\n{'='*80}")
    print(f"TEST: Neuron-Explainer Simulator with Different Models")
    print(f"{'='*80}")

    results = {}

    for model_name, description in models_to_test:
        print(f"\nTesting: {model_name}")
        print(f"Description: {description}")

        success, score, error = await test_neuron_explainer_simulator(
            model_name,
            test_explanation,
            test_tokens,
            test_activations
        )

        results[model_name] = {
            'success': success,
            'score': score,
            'error': error,
            'description': description
        }

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\nOpenAI Library Version: {openai_version}")

    print(f"\nModel Test Results:")
    print(f"{'Model':<30} {'Status':<15} {'Score':<10} {'Notes'}")
    print(f"{'-'*80}")

    working_models = []

    for model_name, result in results.items():
        status = "✅ WORKS" if result['success'] else "❌ FAILED"
        score_str = f"{result['score']:.4f}" if result['score'] is not None else "N/A"

        # Truncate error message
        if result['error']:
            error_short = result['error'][:40] + "..." if len(result['error']) > 40 else result['error']
            notes = error_short
        else:
            notes = "Success"

        print(f"{model_name:<30} {status:<15} {score_str:<10} {notes}")

        if result['success']:
            working_models.append(model_name)

    # Conclusion
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")

    if working_models:
        print(f"\n✅ GOOD NEWS: Neuron-explainer works with the following model(s):")
        for model in working_models:
            print(f"   - {model}")
        print(f"\n   You CAN use the paper's scoring method!")
        print(f"\n   Recommendation:")
        print(f"   Use --simulator_model {working_models[0]} in your evaluation script")
    else:
        print(f"\n❌ BAD NEWS: Neuron-explainer does NOT work with any tested models.")
        print(f"\n   This confirms that the paper's scoring method cannot be used")
        print(f"   with the current OpenAI API.")
        print(f"\n   Possible reasons:")
        print(f"   1. OpenAI deprecated all models supporting echo+logprobs")
        print(f"   2. OpenAI API changed parameter requirements")
        print(f"   3. Incompatibility with OpenAI library version {openai_version}")
        print(f"\n   RECOMMENDATIONS:")
        print(f"   1. Use the alternative LLM-based prediction method in your code")
        print(f"   2. Only rely on Generation Evaluation (Method 1)")
        print(f"   3. Consider using other LLM providers that support logprobs")
        print(f"   4. Wait for neuron-explainer library updates")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with unexpected error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
