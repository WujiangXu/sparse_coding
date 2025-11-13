"""
Simple test for interpretability evaluation
"""

from simple_interp_eval import evaluate_feature_interpretability


def test_numbers_feature():
    """Test on a feature that detects numbers"""

    # Examples where a "numbers" feature activates
    tokens = [
        ["The", "price", "is", "$", "42"],
        ["I", "bought", "3", "apples"],
        ["Chapter", "7", "discusses", "theory"],
        ["Version", "2", ".", "0", "released"],
        ["There", "are", "100", "people"],
        ["He", "scored", "95", "points"],
        ["The", "year", "2024", "looks", "good"],
        ["She", "is", "tall", "and", "smart"],  # No number
    ]

    # Activation values (high where numbers appear)
    activations = [
        [0.1, 0.3, 0.1, 0.5, 0.95],  # High on "42"
        [0.05, 0.2, 0.88, 0.15],     # High on "3"
        [0.3, 0.92, 0.1, 0.15],      # High on "7"
        [0.2, 0.85, 0.4, 0.87, 0.15],# High on "2" and "0"
        [0.1, 0.15, 0.90, 0.2],      # High on "100"
        [0.08, 0.2, 0.93, 0.15],     # High on "95"
        [0.1, 0.2, 0.89, 0.1, 0.12], # High on "2024"
        [0.05, 0.05, 0.08, 0.05, 0.06], # Low (no numbers)
    ]

    print("="*60)
    print("Testing: Numbers Feature")
    print("="*60)

    result = evaluate_feature_interpretability(
        tokens_list=tokens,
        activations_list=activations,
        n_examples=5,
        model="gpt-4"
    )

    print(f"\nExplanation: {result['explanation']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Examples used: {result['examples_used']}")

    return result


def test_negation_feature():
    """Test on a feature that detects negation words"""

    tokens = [
        ["I", "do", "not", "like", "this"],
        ["She", "never", "went", "there"],
        ["Nobody", "knows", "the", "answer"],
        ["This", "isn't", "working", "well"],
        ["Nothing", "can", "stop", "us"],
        ["We", "cannot", "proceed", "now"],
        ["No", "one", "was", "home"],
        ["The", "cat", "is", "sleeping"],  # No negation
    ]

    activations = [
        [0.1, 0.2, 0.95, 0.15, 0.1],    # High on "not"
        [0.1, 0.92, 0.15, 0.08],        # High on "never"
        [0.88, 0.2, 0.1, 0.15],         # High on "Nobody"
        [0.1, 0.90, 0.2, 0.12],         # High on "isn't"
        [0.93, 0.1, 0.15, 0.08],        # High on "Nothing"
        [0.08, 0.91, 0.15, 0.1],        # High on "cannot"
        [0.89, 0.2, 0.1, 0.08],         # High on "No"
        [0.06, 0.08, 0.05, 0.1],        # Low everywhere
    ]

    print("\n" + "="*60)
    print("Testing: Negation Feature")
    print("="*60)

    result = evaluate_feature_interpretability(
        tokens_list=tokens,
        activations_list=activations,
        n_examples=5,
        model="gpt-4"
    )

    print(f"\nExplanation: {result['explanation']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Examples used: {result['examples_used']}")

    return result


if __name__ == "__main__":
    print("Sparse Autoencoder Interpretability Evaluation")
    print("Simple Test Suite\n")

    # Run tests
    result1 = test_numbers_feature()
    result2 = test_negation_feature()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Numbers feature score:  {result1['score']:.3f}")
    print(f"Negation feature score: {result2['score']:.3f}")

    avg_score = (result1['score'] + result2['score']) / 2
    print(f"\nAverage interpretability: {avg_score:.3f}")
