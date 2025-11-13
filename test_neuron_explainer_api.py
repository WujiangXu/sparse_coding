"""
Test script to verify if neuron-explainer library works with current OpenAI API.

This script tests:
1. Basic neuron-explainer simulator functionality
2. OpenAI API logprobs support
3. Different model compatibility

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

# Also test direct OpenAI API calls
try:
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    print(f"✅ OpenAI library imported")
except ImportError as e:
    print(f"⚠️  OpenAI library not available: {e}")


async def test_neuron_explainer_simulator(model_name: str, explanation: str, test_tokens: List[str]):
    """
    Test neuron-explainer simulator with given model.

    Args:
        model_name: Model to test (e.g., "gpt-3.5-turbo-instruct", "text-davinci-003")
        explanation: Feature explanation text
        test_tokens: List of test tokens
    """
    print(f"\n{'='*80}")
    print(f"Testing neuron-explainer simulator with model: {model_name}")
    print(f"{'='*80}")

    try:
        # Create activation records
        print(f"1. Creating test ActivationRecord...")
        test_activations = [1.0, 2.0, 1.5, 0.5, 0.0]  # Sample activation values
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
        print(f"   Explanation: {explanation[:100]}...")

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
        scored_simulation = await simulate_and_score(
            simulator,
            [activation_record]
        )

        # Get results
        score = scored_simulation.get_preferred_score()
        print(f"   ✅ Simulation completed successfully!")
        print(f"      Correlation score: {score:.4f}")

        return True, score

    except Exception as e:
        print(f"   ❌ Simulation failed with error:")
        print(f"      {type(e).__name__}: {str(e)}")

        # Print detailed error info
        import traceback
        print(f"\n   Detailed traceback:")
        traceback.print_exc()

        return False, None


def test_openai_logprobs_direct(model_name: str):
    """
    Test OpenAI API logprobs support directly (without neuron-explainer).

    Args:
        model_name: Model to test
    """
    print(f"\n{'='*80}")
    print(f"Testing OpenAI API logprobs directly with model: {model_name}")
    print(f"{'='*80}")

    try:
        import openai

        print(f"1. Testing basic completion (no logprobs)...")
        response = openai.Completion.create(
            model=model_name,
            prompt="Hello, world!",
            max_tokens=5,
            temperature=0
        )
        print(f"   ✅ Basic completion works")
        print(f"      Response: {response.choices[0].text.strip()}")

        print(f"\n2. Testing completion with logprobs...")
        response = openai.Completion.create(
            model=model_name,
            prompt="Hello, world!",
            max_tokens=5,
            temperature=0,
            logprobs=5
        )
        print(f"   ✅ Logprobs parameter works")
        print(f"      Has logprobs: {response.choices[0].logprobs is not None}")

        print(f"\n3. Testing completion with echo + logprobs...")
        response = openai.Completion.create(
            model=model_name,
            prompt="Hello, world!",
            max_tokens=5,
            temperature=0,
            echo=True,
            logprobs=5
        )
        print(f"   ✅ Echo + logprobs works!")
        print(f"      This is the required combination for neuron-explainer")

        return True

    except openai.error.InvalidRequestError as e:
        print(f"   ❌ Invalid request error:")
        print(f"      {str(e)}")
        return False

    except Exception as e:
        print(f"   ❌ Unexpected error:")
        print(f"      {type(e).__name__}: {str(e)}")
        return False


def test_available_models():
    """
    Test which models are available and support required features.
    """
    print(f"\n{'='*80}")
    print(f"Testing OpenAI Model Availability")
    print(f"{'='*80}")

    models_to_test = [
        "text-davinci-003",
        "text-davinci-002",
        "gpt-3.5-turbo-instruct",
        "gpt-3.5-turbo",
        "gpt-4",
    ]

    results = {}

    for model_name in models_to_test:
        print(f"\nTesting {model_name}...")
        try:
            import openai
            response = openai.Completion.create(
                model=model_name,
                prompt="Test",
                max_tokens=1,
                temperature=0
            )
            print(f"   ✅ Model available")
            results[model_name] = "available"
        except openai.error.InvalidRequestError as e:
            error_msg = str(e)
            if "deprecated" in error_msg.lower() or "not found" in error_msg.lower():
                print(f"   ❌ Model deprecated or not found")
                results[model_name] = "deprecated"
            else:
                print(f"   ❌ Error: {error_msg}")
                results[model_name] = f"error: {error_msg}"
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            results[model_name] = f"error: {type(e).__name__}"

    print(f"\n{'='*80}")
    print(f"Model Availability Summary:")
    print(f"{'='*80}")
    for model, status in results.items():
        print(f"  {model:<30} {status}")

    return results


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

    # Test 1: Check model availability
    print(f"\n{'='*80}")
    print(f"TEST 1: Check OpenAI Model Availability")
    print(f"{'='*80}")
    model_availability = test_available_models()

    # Test 2: Test direct API logprobs support
    print(f"\n{'='*80}")
    print(f"TEST 2: Test OpenAI API Logprobs Support")
    print(f"{'='*80}")

    models_to_test_logprobs = []
    if model_availability.get("gpt-3.5-turbo-instruct") == "available":
        models_to_test_logprobs.append("gpt-3.5-turbo-instruct")
    if model_availability.get("text-davinci-003") == "available":
        models_to_test_logprobs.append("text-davinci-003")

    if not models_to_test_logprobs:
        print(f"⚠️  No compatible models available for testing")
        models_to_test_logprobs = ["gpt-3.5-turbo-instruct"]  # Try anyway

    logprobs_results = {}
    for model in models_to_test_logprobs:
        logprobs_results[model] = test_openai_logprobs_direct(model)

    # Test 3: Test neuron-explainer simulator
    print(f"\n{'='*80}")
    print(f"TEST 3: Test Neuron-Explainer Simulator")
    print(f"{'='*80}")

    simulator_results = {}
    for model in models_to_test_logprobs:
        success, score = await test_neuron_explainer_simulator(
            model,
            test_explanation,
            test_tokens
        )
        simulator_results[model] = (success, score)

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\n1. Model Availability:")
    for model, status in model_availability.items():
        icon = "✅" if status == "available" else "❌"
        print(f"   {icon} {model:<30} {status}")

    print(f"\n2. Logprobs Support (echo + logprobs):")
    for model, success in logprobs_results.items():
        icon = "✅" if success else "❌"
        status = "Supported" if success else "Not supported"
        print(f"   {icon} {model:<30} {status}")

    print(f"\n3. Neuron-Explainer Simulator:")
    for model, (success, score) in simulator_results.items():
        icon = "✅" if success else "❌"
        status = f"Works (score: {score:.4f})" if success else "Failed"
        print(f"   {icon} {model:<30} {status}")

    # Conclusion
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")

    any_simulator_works = any(success for success, _ in simulator_results.values())

    if any_simulator_works:
        print(f"✅ GOOD NEWS: Neuron-explainer simulator works with at least one model!")
        working_models = [model for model, (success, _) in simulator_results.items() if success]
        print(f"   Working models: {', '.join(working_models)}")
        print(f"\n   You can use the paper's scoring method with these models.")
    else:
        print(f"❌ BAD NEWS: Neuron-explainer simulator does NOT work with current OpenAI API.")
        print(f"\n   Possible reasons:")
        print(f"   1. OpenAI has deprecated all models that support echo + logprobs")
        print(f"   2. OpenAI API has changed its parameter requirements")
        print(f"   3. The neuron-explainer library needs to be updated")
        print(f"\n   RECOMMENDATION:")
        print(f"   - Use alternative evaluation methods (e.g., LLM pattern matching)")
        print(f"   - Wait for neuron-explainer library updates")
        print(f"   - Consider using other model providers that support logprobs")

    print(f"{'='*80}\n")


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
