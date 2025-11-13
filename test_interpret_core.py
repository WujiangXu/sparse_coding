"""
Test if the core functionality of interpret.py works with current OpenAI API.

This tests the exact workflow from interpret.py:
1. Load OpenAI API key
2. Create ActivationRecord from sample data
3. Generate explanation using TokenActivationPairExplainer
4. Simulate and score using ExplanationNeuronSimulator

This is the same flow used in interpret.py lines 334-369.
"""

import os
import sys
import json
import asyncio

# Load OpenAI API key (same as interpret.py lines 28-32)
print("="*80)
print("Testing interpret.py core functionality")
print("="*80)

try:
    with open("secrets.json") as f:
        secrets = json.load(f)
        os.environ["OPENAI_API_KEY"] = secrets["openai_key"]
        print(f"✅ Loaded OpenAI API key from secrets.json")
except Exception as e:
    print(f"❌ Failed to load API key: {e}")
    sys.exit(1)

# Import neuron-explainer components (same as interpret.py lines 36-48)
try:
    from neuron_explainer.activations.activation_records import calculate_max_activation
    from neuron_explainer.activations.activations import (
        ActivationRecord, ActivationRecordSliceParams, NeuronId, NeuronRecord
    )
    from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
    from neuron_explainer.explanations.explainer import TokenActivationPairExplainer
    from neuron_explainer.explanations.prompt_builder import PromptFormat
    from neuron_explainer.explanations.scoring import (
        aggregate_scored_sequence_simulations, simulate_and_score
    )
    from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
    print(f"✅ Successfully imported all neuron-explainer modules")
except ImportError as e:
    print(f"❌ Failed to import neuron-explainer modules: {e}")
    sys.exit(1)

# Constants from interpret.py
EXPLAINER_MODEL_NAME = "gpt-4"  # Model for generating explanations
SIMULATOR_MODEL_NAME = "text-davinci-003"  # Model for simulating activations
OPENAI_FRAGMENT_LEN = 64
OPENAI_EXAMPLES_PER_SPLIT = 5
N_SPLITS = 4


async def test_interpret_workflow():
    """
    Test the exact workflow from interpret.py's interpret() function (lines 265-386).
    """
    print(f"\n{'='*80}")
    print(f"STEP 1: Create sample ActivationRecords")
    print(f"{'='*80}")

    # Create sample activation records (mimicking lines 284-291)
    # Example: tokens with high activation on "diss" prefix
    sample_tokens_1 = [
        "The", " research", " on", " diss", "ipation", " in", " physics", " is", " important", "."
    ]
    sample_activations_1 = [0.1, 0.2, 0.1, 2.5, 1.8, 0.2, 0.1, 0.2, 0.1, 0.1]

    sample_tokens_2 = [
        "We", " need", " to", " diss", "olve", " the", " compound", " carefully", "."
    ]
    sample_activations_2 = [0.1, 0.2, 0.1, 2.3, 1.6, 0.2, 0.1, 0.2, 0.1]

    # Pad to same length
    while len(sample_tokens_2) < len(sample_tokens_1):
        sample_tokens_2.append("")
        sample_activations_2.append(0.0)

    top_activation_records = [
        ActivationRecord(sample_tokens_1, sample_activations_1),
        ActivationRecord(sample_tokens_2, sample_activations_2),
    ]

    random_activation_records = [
        ActivationRecord(
            ["Random", " text", " with", " no", " pattern"],
            [0.1, 0.15, 0.12, 0.08, 0.1]
        )
    ]

    print(f"   ✅ Created {len(top_activation_records)} top activation records")
    print(f"   ✅ Created {len(random_activation_records)} random activation records")

    # Create NeuronRecord (mimicking lines 323-329)
    neuron_id = NeuronId(layer_index=0, neuron_index=0)
    neuron_record = NeuronRecord(
        neuron_id=neuron_id,
        random_sample=random_activation_records,
        most_positive_activation_records=top_activation_records,
    )

    # Split into train/valid (mimicking lines 330-332)
    slice_params = ActivationRecordSliceParams(n_examples_per_split=1)
    train_activation_records = neuron_record.train_activation_records(slice_params)
    valid_activation_records = neuron_record.valid_activation_records(slice_params)

    print(f"   ✅ Split into {len(train_activation_records)} train and {len(valid_activation_records)} valid records")

    # STEP 2: Generate explanation (mimicking lines 334-346)
    print(f"\n{'='*80}")
    print(f"STEP 2: Generate explanation using {EXPLAINER_MODEL_NAME}")
    print(f"{'='*80}")

    try:
        explainer = TokenActivationPairExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=None,
        )

        print(f"   ✅ Created TokenActivationPairExplainer")
        print(f"   🤖 Calling {EXPLAINER_MODEL_NAME} to generate explanation...")

        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
        )

        explanation = explanations[0]
        print(f"   ✅ Generated explanation: '{explanation}'")

    except Exception as e:
        print(f"   ❌ Failed to generate explanation: {e}")
        import traceback
        traceback.print_exc()
        return False, "explanation_generation_failed", str(e)

    # STEP 3: Simulate and score (mimicking lines 348-369)
    print(f"\n{'='*80}")
    print(f"STEP 3: Simulate and score using {SIMULATOR_MODEL_NAME}")
    print(f"{'='*80}")

    try:
        # Determine prompt format (mimicking line 349)
        format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING

        # Create simulator (mimicking lines 350-357)
        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                SIMULATOR_MODEL_NAME,
                explanation,
                max_concurrent=None,
                prompt_format=format,
            )
        )

        print(f"   ✅ Created simulator with model: {SIMULATOR_MODEL_NAME}")
        print(f"   🤖 Running simulation on {len(valid_activation_records)} validation records...")

        # Simulate and score (mimicking line 358)
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)
        score = scored_simulation.get_preferred_score()

        print(f"   ✅ Simulation completed successfully!")
        print(f"      Score: {score:.4f}")

        return True, "success", None

    except Exception as e:
        print(f"   ❌ Failed to simulate and score: {e}")

        # Analyze error
        error_msg = str(e)
        if "logprobs" in error_msg.lower() and "echo" in error_msg.lower():
            print(f"\n   💡 Error Analysis: Model doesn't support echo+logprobs combination")
            print(f"      This is REQUIRED for the simulator to work")
            error_type = "echo_logprobs_not_supported"
        elif "deprecated" in error_msg.lower():
            print(f"\n   💡 Error Analysis: Model has been deprecated")
            error_type = "model_deprecated"
        else:
            print(f"\n   💡 Error Analysis: Unknown error")
            error_type = "unknown_error"

        import traceback
        traceback.print_exc()
        return False, error_type, error_msg


async def test_with_different_simulator_models():
    """
    Test with different simulator models to see which ones work.
    """
    models_to_test = [
        "text-davinci-003",
        "gpt-3.5-turbo-instruct",
    ]

    print(f"\n{'='*80}")
    print(f"Testing with different simulator models")
    print(f"{'='*80}")

    results = {}

    for model in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing with simulator model: {model}")
        print(f"{'='*80}")

        # Temporarily change the global constant
        global SIMULATOR_MODEL_NAME
        original_model = SIMULATOR_MODEL_NAME
        SIMULATOR_MODEL_NAME = model

        success, error_type, error_msg = await test_interpret_workflow()

        results[model] = {
            'success': success,
            'error_type': error_type,
            'error_msg': error_msg
        }

        # Restore original model
        SIMULATOR_MODEL_NAME = original_model

    return results


async def main():
    """
    Main test function.
    """
    print(f"\n{'='*80}")
    print(f"Testing if interpret.py can work with current OpenAI API")
    print(f"{'='*80}")
    print(f"\nThis tests the exact workflow from interpret.py:")
    print(f"1. Create ActivationRecords (lines 284-329)")
    print(f"2. Generate explanation with GPT-4 (lines 334-346)")
    print(f"3. Simulate and score with text-davinci-003 (lines 348-369)")

    # Test with different simulator models
    results = await test_with_different_simulator_models()

    # Summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")

    print(f"\nSimulator Model Test Results:")
    print(f"{'Model':<30} {'Status':<15} {'Error Type'}")
    print(f"{'-'*70}")

    working_models = []
    for model, result in results.items():
        status = "✅ WORKS" if result['success'] else "❌ FAILED"
        error_type = result['error_type'] if not result['success'] else "N/A"
        print(f"{model:<30} {status:<15} {error_type}")

        if result['success']:
            working_models.append(model)

    # Conclusion
    print(f"\n{'='*80}")
    print(f"CONCLUSION")
    print(f"{'='*80}")

    if working_models:
        print(f"\n✅ GOOD NEWS: interpret.py CAN work with the following model(s):")
        for model in working_models:
            print(f"   - {model}")
        print(f"\n   The repository code is functional!")
        print(f"\n   To use interpret.py, update SIMULATOR_MODEL_NAME to: {working_models[0]}")
    else:
        print(f"\n❌ BAD NEWS: interpret.py CANNOT work with current OpenAI API")
        print(f"\n   The repository's automatic interpretation functionality is BROKEN.")
        print(f"\n   Reason: All tested simulator models fail due to:")
        for model, result in results.items():
            if not result['success']:
                print(f"   - {model}: {result['error_type']}")
        print(f"\n   This means:")
        print(f"   1. The paper's code is outdated and incompatible with current OpenAI API")
        print(f"   2. You cannot use the automatic interpretation feature (interpret.py)")
        print(f"   3. The repository needs updates to work with modern OpenAI API")
        print(f"\n   RECOMMENDATIONS:")
        print(f"   - Fork the repository and update to use alternative scoring methods")
        print(f"   - Use only the autoencoder training parts (not interpretation)")
        print(f"   - Wait for the authors to update the code")

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
