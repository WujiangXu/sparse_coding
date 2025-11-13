"""
Simplified Interpretability Evaluation for Sparse Autoencoders
Uses OpenAI's latest Chat Completions API (2025)

Based on the methodology from "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
Main idea:
1. Collect top-activating examples for each feature
2. Generate natural language explanations using GPT-4
3. Score explanations by predicting activations on validation set
"""

import asyncio
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from openai import AsyncOpenAI


@dataclass
class ActivationExample:
    """A text example with its activation values for a feature."""
    tokens: List[str]  # Token strings
    activations: List[float]  # Activation value per token

    @property
    def max_activation(self) -> float:
        return max(self.activations)

    def format_for_prompt(self, normalize: bool = True) -> str:
        """Format as: token1(0.5) token2(0.8) token3(0.2)"""
        if normalize and self.max_activation > 0:
            norm_acts = [a / self.max_activation for a in self.activations]
        else:
            norm_acts = self.activations

        return " ".join([f"{tok}({act:.2f})" for tok, act in zip(self.tokens, norm_acts)])


class InterpretabilityEvaluator:
    """Evaluates feature interpretability using OpenAI API."""

    def __init__(self, api_key: str, explainer_model: str = "gpt-4", simulator_model: str = "gpt-3.5-turbo"):
        """
        Args:
            api_key: OpenAI API key
            explainer_model: Model for generating explanations (e.g., "gpt-4", "gpt-4-turbo")
            simulator_model: Model for simulating/scoring (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.explainer_model = explainer_model
        self.simulator_model = simulator_model

    async def generate_explanation(
        self,
        examples: List[ActivationExample],
        max_examples: int = 5
    ) -> str:
        """
        Generate a natural language explanation for a feature based on its top activating examples.

        Args:
            examples: List of activation examples, should be sorted by activation strength
            max_examples: Maximum number of examples to show the model

        Returns:
            Natural language explanation string
        """
        # Format examples for the prompt
        example_texts = []
        for i, ex in enumerate(examples[:max_examples]):
            example_texts.append(f"Example {i+1}: {ex.format_for_prompt()}")

        prompt = f"""I'm analyzing a feature from a sparse autoencoder trained on language model activations. Below are text sequences where this feature activates most strongly. Each token is followed by its activation value in parentheses (normalized to max=1.0).

{chr(10).join(example_texts)}

Based on these examples, provide a concise explanation (1-2 sentences) of what pattern or concept this feature appears to detect. Focus on what makes the tokens with high activation values similar or related."""

        response = await self.client.chat.completions.create(
            model=self.explainer_model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing neural network features and identifying patterns in activations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        explanation = response.choices[0].message.content.strip()
        return explanation

    async def simulate_activation(
        self,
        explanation: str,
        example: ActivationExample
    ) -> List[float]:
        """
        Use the explanation to predict which tokens should activate the feature.

        Args:
            explanation: Natural language explanation of the feature
            example: An example to score

        Returns:
            List of predicted activation scores (0-10 scale) for each token
        """
        tokens_str = " ".join(example.tokens)

        prompt = f"""You are simulating a neural network feature with this behavior:
"{explanation}"

For each token in the following sequence, rate from 0-10 how much this feature should activate (0 = not at all, 10 = maximum activation).

Sequence: {tokens_str}

Respond with only a comma-separated list of numbers, one per token. For example: 2,5,8,3,0,1"""

        response = await self.client.chat.completions.create(
            model=self.simulator_model,
            messages=[
                {"role": "system", "content": "You are a neural network simulator. Respond only with comma-separated numbers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )

        # Parse the response
        try:
            scores_text = response.choices[0].message.content.strip()
            scores = [float(x.strip()) for x in scores_text.split(",")]

            # Normalize to 0-1 range
            if max(scores) > 0:
                scores = [s / 10.0 for s in scores]

            # Ensure we have the right number of scores
            if len(scores) != len(example.tokens):
                print(f"Warning: Expected {len(example.tokens)} scores, got {len(scores)}. Padding/truncating.")
                if len(scores) < len(example.tokens):
                    scores.extend([0.0] * (len(example.tokens) - len(scores)))
                else:
                    scores = scores[:len(example.tokens)]

            return scores
        except Exception as e:
            print(f"Error parsing simulation response: {e}")
            print(f"Response was: {response.choices[0].message.content}")
            # Return uniform low scores as fallback
            return [0.1] * len(example.tokens)

    def compute_score(
        self,
        predicted: List[float],
        actual: List[float]
    ) -> float:
        """
        Compute correlation between predicted and actual activations.
        Uses Pearson correlation coefficient.

        Args:
            predicted: Predicted activation values
            actual: Actual activation values

        Returns:
            Correlation score (0-1, higher is better)
        """
        if len(predicted) != len(actual):
            raise ValueError(f"Length mismatch: predicted {len(predicted)}, actual {len(actual)}")

        # Normalize both to have similar scale
        pred_array = np.array(predicted)
        actual_array = np.array(actual)

        # Handle edge cases
        if actual_array.std() == 0 or pred_array.std() == 0:
            return 0.0

        # Pearson correlation
        correlation = np.corrcoef(pred_array, actual_array)[0, 1]

        # Convert to 0-1 scale (from -1 to 1)
        score = (correlation + 1) / 2

        return float(score)

    async def evaluate_feature(
        self,
        train_examples: List[ActivationExample],
        validation_examples: List[ActivationExample],
        n_train: int = 5,
        n_val: int = 10,
        verbose: bool = True
    ) -> Tuple[str, float, dict]:
        """
        Complete evaluation pipeline for a single feature.

        Args:
            train_examples: Examples to generate explanation (sorted by activation)
            validation_examples: Examples to score the explanation
            n_train: Number of training examples to use
            n_val: Number of validation examples to score
            verbose: Whether to print progress

        Returns:
            Tuple of (explanation, overall_score, detailed_scores)
        """
        if verbose:
            print("Generating explanation...")

        # Generate explanation from top examples
        explanation = await self.generate_explanation(train_examples[:n_train])

        if verbose:
            print(f"Explanation: {explanation}")
            print(f"\nScoring on {n_val} validation examples...")

        # Score on validation set
        scores = []
        for i, example in enumerate(validation_examples[:n_val]):
            predicted = await self.simulate_activation(explanation, example)

            # Normalize actual activations to 0-1
            actual = example.activations
            if max(actual) > 0:
                actual = [a / max(actual) for a in actual]

            score = self.compute_score(predicted, actual)
            scores.append(score)

            if verbose:
                print(f"  Example {i+1}: score = {score:.3f}")

        overall_score = np.mean(scores)

        detailed_scores = {
            "overall": overall_score,
            "individual": scores,
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores)
        }

        if verbose:
            print(f"\nOverall Score: {overall_score:.3f} ± {np.std(scores):.3f}")

        return explanation, overall_score, detailed_scores


# Convenience function for batch evaluation
async def evaluate_features_batch(
    evaluator: InterpretabilityEvaluator,
    features_data: List[Tuple[List[ActivationExample], List[ActivationExample]]],
    max_concurrent: int = 5
) -> List[Tuple[str, float, dict]]:
    """
    Evaluate multiple features with rate limiting.

    Args:
        evaluator: InterpretabilityEvaluator instance
        features_data: List of (train_examples, val_examples) tuples
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of (explanation, score, details) for each feature
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def eval_with_limit(train, val, idx):
        async with semaphore:
            print(f"\n{'='*60}")
            print(f"Evaluating Feature {idx}")
            print(f"{'='*60}")
            return await evaluator.evaluate_feature(train, val)

    tasks = [
        eval_with_limit(train, val, i)
        for i, (train, val) in enumerate(features_data)
    ]

    results = await asyncio.gather(*tasks)
    return results
