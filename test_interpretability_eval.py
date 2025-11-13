"""
Test script for interpretability evaluation with synthetic and real examples.
Demonstrates how to use the InterpretabilityEvaluator class.
"""

import asyncio
import json
import os
from interpretability_eval import ActivationExample, InterpretabilityEvaluator


def create_synthetic_examples():
    """
    Create synthetic test cases for common features.
    Returns dict mapping feature_name -> (train_examples, val_examples)
    """
    examples = {}

    # Feature 1: Numbers
    examples["numbers"] = (
        # Training examples (top activating)
        [
            ActivationExample(
                tokens=["The", "price", "is", "$", "42", "dollars"],
                activations=[0.1, 0.3, 0.1, 0.5, 0.95, 0.2]
            ),
            ActivationExample(
                tokens=["I", "bought", "3", "apples", "yesterday"],
                activations=[0.05, 0.2, 0.88, 0.15, 0.1]
            ),
            ActivationExample(
                tokens=["Chapter", "7", "discusses", "the", "theory"],
                activations=[0.3, 0.92, 0.1, 0.05, 0.15]
            ),
            ActivationExample(
                tokens=["There", "are", "100", "people", "here"],
                activations=[0.1, 0.15, 0.90, 0.2, 0.1]
            ),
            ActivationExample(
                tokens=["Version", "2", ".", "0", "is", "released"],
                activations=[0.2, 0.85, 0.4, 0.87, 0.1, 0.15]
            ),
        ],
        # Validation examples
        [
            ActivationExample(
                tokens=["He", "scored", "95", "points"],
                activations=[0.08, 0.2, 0.93, 0.15]
            ),
            ActivationExample(
                tokens=["The", "year", "2024", "was", "great"],
                activations=[0.1, 0.2, 0.89, 0.1, 0.12]
            ),
            ActivationExample(
                tokens=["She", "is", "tall", "and", "smart"],
                activations=[0.05, 0.05, 0.08, 0.05, 0.06]  # Low activation (no numbers)
            ),
        ]
    )

    # Feature 2: Negation words
    examples["negation"] = (
        [
            ActivationExample(
                tokens=["I", "do", "not", "like", "this"],
                activations=[0.1, 0.2, 0.95, 0.15, 0.1]
            ),
            ActivationExample(
                tokens=["She", "never", "went", "there"],
                activations=[0.1, 0.92, 0.15, 0.08]
            ),
            ActivationExample(
                tokens=["Nobody", "knows", "the", "answer"],
                activations=[0.88, 0.2, 0.1, 0.15]
            ),
            ActivationExample(
                tokens=["This", "is", "n't", "working"],
                activations=[0.1, 0.15, 0.90, 0.2]
            ),
            ActivationExample(
                tokens=["Nothing", "can", "stop", "us"],
                activations=[0.93, 0.1, 0.15, 0.08]
            ),
        ],
        [
            ActivationExample(
                tokens=["We", "cannot", "proceed", "further"],
                activations=[0.08, 0.91, 0.15, 0.1]
            ),
            ActivationExample(
                tokens=["No", "one", "was", "home"],
                activations=[0.89, 0.2, 0.1, 0.08]
            ),
            ActivationExample(
                tokens=["The", "cat", "is", "sleeping"],
                activations=[0.06, 0.08, 0.05, 0.1]  # Low activation
            ),
        ]
    )

    # Feature 3: Pronouns
    examples["pronouns"] = (
        [
            ActivationExample(
                tokens=["She", "went", "to", "the", "store"],
                activations=[0.92, 0.1, 0.05, 0.08, 0.12]
            ),
            ActivationExample(
                tokens=["He", "likes", "playing", "games"],
                activations=[0.90, 0.15, 0.08, 0.1]
            ),
            ActivationExample(
                tokens=["They", "are", "coming", "tomorrow"],
                activations=[0.88, 0.1, 0.12, 0.09]
            ),
            ActivationExample(
                tokens=["We", "should", "leave", "now"],
                activations=[0.91, 0.08, 0.1, 0.07]
            ),
            ActivationExample(
                tokens=["I", "think", "it", "works"],
                activations=[0.87, 0.1, 0.85, 0.08]
            ),
        ],
        [
            ActivationExample(
                tokens=["You", "can", "do", "it"],
                activations=[0.89, 0.1, 0.08, 0.86]
            ),
            ActivationExample(
                tokens=["The", "book", "is", "good"],
                activations=[0.08, 0.1, 0.12, 0.09]  # Low activation
            ),
        ]
    )

    return examples


async def test_single_feature():
    """Test evaluation on a single synthetic feature."""
    print("\n" + "="*70)
    print("TEST 1: Single Feature Evaluation (Numbers)")
    print("="*70)

    # Load API key
    try:
        with open("secrets.json") as f:
            secrets = json.load(f)
            api_key = secrets["openai_key"]
    except FileNotFoundError:
        print("ERROR: secrets.json not found. Please create it with your OpenAI API key.")
        print('Format: {"openai_key": "sk-..."}')
        return

    # Initialize evaluator
    evaluator = InterpretabilityEvaluator(
        api_key=api_key,
        explainer_model="gpt-4",
        simulator_model="gpt-3.5-turbo"
    )

    # Get synthetic examples
    examples = create_synthetic_examples()
    train_examples, val_examples = examples["numbers"]

    # Evaluate
    explanation, score, details = await evaluator.evaluate_feature(
        train_examples=train_examples,
        validation_examples=val_examples,
        n_train=5,
        n_val=3,
        verbose=True
    )

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Explanation: {explanation}")
    print(f"Overall Score: {score:.3f}")
    print(f"Score Statistics: mean={details['mean']:.3f}, std={details['std']:.3f}")
    print(f"Score Range: [{details['min']:.3f}, {details['max']:.3f}]")


async def test_multiple_features():
    """Test evaluation on multiple features."""
    print("\n" + "="*70)
    print("TEST 2: Multiple Feature Evaluation")
    print("="*70)

    # Load API key
    try:
        with open("secrets.json") as f:
            secrets = json.load(f)
            api_key = secrets["openai_key"]
    except FileNotFoundError:
        print("ERROR: secrets.json not found.")
        return

    # Initialize evaluator
    evaluator = InterpretabilityEvaluator(
        api_key=api_key,
        explainer_model="gpt-4",
        simulator_model="gpt-3.5-turbo"
    )

    # Get all synthetic examples
    examples = create_synthetic_examples()

    # Prepare data for batch evaluation
    features_data = [
        examples["numbers"],
        examples["negation"],
        examples["pronouns"]
    ]

    feature_names = ["numbers", "negation", "pronouns"]

    # Evaluate all features
    from interpretability_eval import evaluate_features_batch

    results = await evaluate_features_batch(
        evaluator=evaluator,
        features_data=features_data,
        max_concurrent=2  # Limit concurrent API calls
    )

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF ALL FEATURES")
    print("="*70)

    for name, (explanation, score, details) in zip(feature_names, results):
        print(f"\n{name.upper()}")
        print(f"  Explanation: {explanation}")
        print(f"  Score: {score:.3f} ± {details['std']:.3f}")
        print(f"  Range: [{details['min']:.3f}, {details['max']:.3f}]")

    # Rank features by interpretability
    ranked = sorted(
        zip(feature_names, [r[1] for r in results]),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "="*70)
    print("FEATURES RANKED BY INTERPRETABILITY SCORE")
    print("="*70)
    for i, (name, score) in enumerate(ranked, 1):
        print(f"{i}. {name}: {score:.3f}")


def test_activation_example():
    """Test the ActivationExample class."""
    print("\n" + "="*70)
    print("TEST 3: ActivationExample Class")
    print("="*70)

    example = ActivationExample(
        tokens=["The", "price", "is", "$", "42"],
        activations=[0.1, 0.3, 0.1, 0.5, 0.95]
    )

    print(f"Tokens: {example.tokens}")
    print(f"Activations: {example.activations}")
    print(f"Max activation: {example.max_activation}")
    print(f"Formatted: {example.format_for_prompt()}")
    print(f"Formatted (unnormalized): {example.format_for_prompt(normalize=False)}")


async def test_with_real_autoencoder():
    """
    Template for testing with a real sparse autoencoder.
    You would need to:
    1. Load your trained autoencoder
    2. Collect activations on real text
    3. Create ActivationExample objects
    """
    print("\n" + "="*70)
    print("TEST 4: Real Autoencoder Integration (Template)")
    print("="*70)

    print("""
To use with a real sparse autoencoder:

1. Load your model:
   ```python
   from autoencoders.learned_dict import LearnedDict
   autoencoder = torch.load("path/to/model.pt")
   ```

2. Collect activations:
   ```python
   # Get model activations
   _, cache = transformer.run_with_cache(tokens)
   mlp_acts = cache["blocks.6.mlp.hook_post"]

   # Encode through sparse autoencoder
   feature_acts = autoencoder.encode(mlp_acts)  # [batch, seq_len, n_features]
   ```

3. For each feature, find top-activating examples:
   ```python
   feature_idx = 42
   activations = feature_acts[:, :, feature_idx]

   # Get top examples
   top_indices = torch.topk(activations.max(dim=1).values, k=10).indices

   # Create ActivationExample objects
   train_examples = []
   for idx in top_indices[:5]:
       tokens = tokenizer.batch_decode(token_ids[idx])
       acts = activations[idx].tolist()
       train_examples.append(ActivationExample(tokens, acts))
   ```

4. Evaluate:
   ```python
   evaluator = InterpretabilityEvaluator(api_key=api_key)
   explanation, score, details = await evaluator.evaluate_feature(
       train_examples=train_examples,
       validation_examples=val_examples
   )
   ```
""")


if __name__ == "__main__":
    print("Sparse Autoencoder Interpretability Evaluation Tests")
    print("=" * 70)

    # Test 1: Basic ActivationExample class
    test_activation_example()

    # Test 2: Single feature evaluation
    print("\nRunning async test for single feature...")
    asyncio.run(test_single_feature())

    # Test 3: Multiple features
    print("\nRunning async test for multiple features...")
    asyncio.run(test_multiple_features())

    # Test 4: Show template for real usage
    asyncio.run(test_with_real_autoencoder())

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
