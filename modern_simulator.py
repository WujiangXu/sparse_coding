"""
Modernized neuron simulator using latest OpenAI Chat Completions API.
Based on neuron_explainer's simulator.py but simplified and updated.

Key improvements:
- Uses Chat Completions API instead of deprecated Completions API
- No external dependencies beyond openai and numpy
- Cleaner, more maintainable code
- Supports both GPT-4 and GPT-3.5-turbo
"""

import asyncio
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

# Normalized activation values range from 0 to 10
MAX_NORMALIZED_ACTIVATION = 10
VALID_ACTIVATION_TOKENS = set(str(i) for i in range(MAX_NORMALIZED_ACTIVATION + 1))


class ActivationScale(str, Enum):
    """Scale for activation values."""
    SIMULATED_NORMALIZED_ACTIVATIONS = "simulated_normalized"
    NEURON_ACTIVATIONS = "neuron"


@dataclass
class SequenceSimulation:
    """Result of simulating neuron activations on a token sequence."""
    tokens: List[str]
    expected_activations: List[float]
    activation_scale: ActivationScale
    distribution_values: List[List[float]]  # For each token, possible activation values
    distribution_probabilities: List[List[float]]  # Probabilities for each value


@dataclass
class FewShotExample:
    """Example for few-shot prompting."""
    explanation: str
    tokens: List[str]
    activations: List[float]


def normalize_activations(
    activations: Sequence[float],
    max_activation: float
) -> List[int]:
    """
    Normalize activations to 0-10 integer scale.

    Args:
        activations: Raw activation values
        max_activation: Maximum activation value for normalization

    Returns:
        List of integers in range [0, 10]
    """
    if max_activation == 0:
        return [0] * len(activations)

    normalized = [
        min(10, max(0, int(round((act / max_activation) * MAX_NORMALIZED_ACTIVATION))))
        for act in activations
    ]
    return normalized


def format_token_activation_pair(token: str, activation: Union[int, str]) -> str:
    """Format a token-activation pair for the prompt."""
    return f"{token}\t{activation}"


def format_sequence_with_activations(
    tokens: Sequence[str],
    activations: Optional[Sequence[int]] = None
) -> str:
    """
    Format a sequence of tokens with their activations.

    Args:
        tokens: Token strings
        activations: Normalized activations (0-10), or None to use "unknown"
    """
    lines = []
    for i, token in enumerate(tokens):
        if activations is None:
            act = "unknown"
        else:
            act = str(activations[i])
        lines.append(format_token_activation_pair(token, act))

    return "<start>\n" + "\n".join(lines) + "\n<end>"


class ModernNeuronSimulator:
    """
    Simulate neuron activations based on an explanation using modern OpenAI API.

    This is a simplified version of ExplanationNeuronSimulator that:
    - Uses Chat Completions API
    - Works with GPT-4/GPT-3.5-turbo
    - Uses logprobs for probabilistic predictions
    """

    def __init__(
        self,
        explanation: str,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        few_shot_examples: Optional[List[FewShotExample]] = None,
        use_async: bool = False
    ):
        """
        Args:
            explanation: Natural language explanation of what the neuron detects
            api_key: OpenAI API key (or load from secrets.json)
            model_name: Model to use (gpt-4, gpt-3.5-turbo, gpt-4-turbo, etc.)
            few_shot_examples: Examples for few-shot prompting
            use_async: Whether to use async client
        """
        if api_key is None:
            with open("secrets.json") as f:
                api_key = json.load(f)["openai_key"]

        if use_async:
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key)

        self.explanation = explanation
        self.model_name = model_name
        self.few_shot_examples = few_shot_examples or self._get_default_examples()
        self.use_async = use_async

    def _get_default_examples(self) -> List[FewShotExample]:
        """Provide default few-shot examples."""
        return [
            FewShotExample(
                explanation="the token 'the'",
                tokens=["In", "the", "beginning", "there", "was", "the", "word"],
                activations=[0, 10, 0, 0, 0, 10, 0]
            ),
            FewShotExample(
                explanation="negative sentiment or words",
                tokens=["I", "hate", "this", "bad", "movie"],
                activations=[0, 9, 2, 8, 3]
            ),
            FewShotExample(
                explanation="proper nouns and names",
                tokens=["John", "went", "to", "Paris", "yesterday"],
                activations=[10, 0, 0, 10, 0]
            ),
        ]

    def _make_simulation_prompt(self, tokens: Sequence[str]) -> List[dict]:
        """
        Create a few-shot prompt for predicting neuron activations.

        Returns:
            List of message dicts for Chat Completions API
        """
        messages = [
            {
                "role": "system",
                "content": """We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at the summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, where activations range from 0 to 10.
"unknown" indicates an unknown activation. Most activations will be 0.
The sequence is wrapped in <start> and <end> markers."""
            }
        ]

        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples):
            messages.append({
                "role": "user",
                "content": f"\n\nNeuron {i + 1}\nExplanation of neuron {i + 1} behavior: "
                          f"this neuron activates for {example.explanation}"
            })

            formatted = format_sequence_with_activations(
                example.tokens,
                example.activations
            )
            messages.append({
                "role": "assistant",
                "content": f"\nActivations:\n{formatted}\n"
            })

        # Add the target neuron
        messages.append({
            "role": "user",
            "content": f"\n\nNeuron {len(self.few_shot_examples) + 1}\n"
                      f"Explanation of neuron {len(self.few_shot_examples) + 1} behavior: "
                      f"this neuron activates for {self.explanation.strip()}"
        })

        formatted_target = format_sequence_with_activations(tokens, None)
        messages.append({
            "role": "assistant",
            "content": f"\nActivations:\n{formatted_target}"
        })

        return messages

    async def simulate_async(self, tokens: Sequence[str]) -> SequenceSimulation:
        """
        Simulate activations asynchronously (requires use_async=True).

        Note: This is a simplified version. Full logprobs support requires
        using the Completions API or parsing chat completion tokens carefully.
        For now, we use a simpler approach without detailed probability distributions.
        """
        if not self.use_async:
            raise ValueError("Must initialize with use_async=True for async simulation")

        # For chat completions, we get the predicted sequence
        messages = self._make_simulation_prompt(tokens)

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.0,  # Deterministic
        )

        completion = response.choices[0].message.content
        expected_activations = self._parse_completion(completion, tokens)

        return SequenceSimulation(
            tokens=list(tokens),
            expected_activations=expected_activations,
            activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
            distribution_values=[],  # Not available without logprobs
            distribution_probabilities=[]
        )

    def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        """
        Simulate activations synchronously.

        Args:
            tokens: Sequence of token strings to simulate activations for

        Returns:
            SequenceSimulation with predicted activations
        """
        messages = self._make_simulation_prompt(tokens)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.0,
        )

        completion = response.choices[0].message.content
        expected_activations = self._parse_completion(completion, tokens)

        return SequenceSimulation(
            tokens=list(tokens),
            expected_activations=expected_activations,
            activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
            distribution_values=[],
            distribution_probabilities=[]
        )

    def _parse_completion(
        self,
        completion: str,
        expected_tokens: Sequence[str]
    ) -> List[float]:
        """
        Parse the completion to extract predicted activations.

        Args:
            completion: Model's completion text
            expected_tokens: Expected token sequence

        Returns:
            List of predicted activation values (0-10 scale)
        """
        # Find the activation sequence between <start> and <end>
        start_idx = completion.find("<start>")
        end_idx = completion.find("<end>")

        if start_idx == -1 or end_idx == -1:
            logger.warning("Could not find <start> and <end> markers, returning zeros")
            return [0.0] * len(expected_tokens)

        activation_text = completion[start_idx + 7:end_idx].strip()
        lines = [line.strip() for line in activation_text.split("\n") if line.strip()]

        activations = []
        for i, line in enumerate(lines):
            if i >= len(expected_tokens):
                break

            parts = line.split("\t")
            if len(parts) != 2:
                logger.warning(f"Malformed line: {line}")
                activations.append(0.0)
                continue

            token_part, activation_part = parts

            # Try to parse activation
            try:
                if activation_part in VALID_ACTIVATION_TOKENS:
                    activations.append(float(activation_part))
                else:
                    logger.warning(f"Invalid activation: {activation_part}")
                    activations.append(0.0)
            except ValueError:
                logger.warning(f"Could not parse activation: {activation_part}")
                activations.append(0.0)

        # Pad with zeros if we didn't get enough activations
        while len(activations) < len(expected_tokens):
            activations.append(0.0)

        return activations[:len(expected_tokens)]


class TokenByTokenSimulator:
    """
    Simulate activations one token at a time.

    This makes separate API calls for each token, which is slower but
    can be more accurate since each prediction is independent.
    """

    def __init__(
        self,
        explanation: str,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        few_shot_examples: Optional[List[FewShotExample]] = None
    ):
        if api_key is None:
            with open("secrets.json") as f:
                api_key = json.load(f)["openai_key"]

        self.client = AsyncOpenAI(api_key=api_key)
        self.explanation = explanation
        self.model_name = model_name
        self.few_shot_examples = few_shot_examples or self._get_default_examples()

    def _get_default_examples(self) -> List[FewShotExample]:
        """Same as ModernNeuronSimulator."""
        return ModernNeuronSimulator(self.explanation)._get_default_examples()

    async def _simulate_single_token(
        self,
        tokens: Sequence[str],
        token_index: int
    ) -> float:
        """Simulate activation for a single token given context."""
        # Truncate to current position
        tokens_so_far = tokens[:token_index + 1]

        messages = [
            {
                "role": "system",
                "content": """We're studying neurons in a neural network. Predict the activation (0-10)
of a neuron on the last token of a text sequence, given an explanation of what the neuron detects.
Most activations will be 0. Respond with only a single number from 0 to 10."""
            }
        ]

        # Add few-shot examples
        for i, example in enumerate(self.few_shot_examples):
            # Use first few tokens as context
            ex_tokens = example.tokens[:min(len(example.tokens), token_index + 1)]
            ex_activations = example.activations[:len(ex_tokens)]

            messages.append({
                "role": "user",
                "content": f"Neuron explanation: this neuron activates for {example.explanation}\n"
                          f"Text: {''.join(ex_tokens)}\n"
                          f"Last token: {ex_tokens[-1]}\n"
                          f"Last token activation:"
            })
            messages.append({
                "role": "assistant",
                "content": str(ex_activations[-1])
            })

        # Add target
        messages.append({
            "role": "user",
            "content": f"Neuron explanation: this neuron activates for {self.explanation}\n"
                      f"Text: {''.join(tokens_so_far)}\n"
                      f"Last token: {tokens_so_far[-1]}\n"
                      f"Last token activation:"
        })

        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=5,
            temperature=0.0
        )

        # Parse response
        try:
            activation = float(response.choices[0].message.content.strip())
            return max(0.0, min(10.0, activation))
        except ValueError:
            logger.warning(f"Could not parse activation, got: {response.choices[0].message.content}")
            return 0.0

    async def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        """Simulate activations for all tokens."""
        # Simulate all tokens concurrently
        tasks = [
            self._simulate_single_token(tokens, i)
            for i in range(len(tokens))
        ]
        expected_activations = await asyncio.gather(*tasks)

        return SequenceSimulation(
            tokens=list(tokens),
            expected_activations=list(expected_activations),
            activation_scale=ActivationScale.SIMULATED_NORMALIZED_ACTIVATIONS,
            distribution_values=[],
            distribution_probabilities=[]
        )


def compute_simulation_score(
    simulation: SequenceSimulation,
    actual_activations: Sequence[float]
) -> float:
    """
    Compute correlation score between simulated and actual activations.

    Args:
        simulation: Simulated activations
        actual_activations: Actual neuron activations

    Returns:
        Correlation score in range [0, 1]
    """
    if len(simulation.expected_activations) != len(actual_activations):
        raise ValueError("Simulation and actual activations must have same length")

    predicted = np.array(simulation.expected_activations)
    actual = np.array(actual_activations)

    # Normalize actual activations to 0-10 scale
    if actual.max() > 0:
        actual_normalized = (actual / actual.max()) * 10.0
    else:
        actual_normalized = actual

    # Compute Pearson correlation
    if predicted.std() == 0 or actual_normalized.std() == 0:
        return 0.0

    correlation = np.corrcoef(predicted, actual_normalized)[0, 1]

    # Convert to 0-1 range
    score = (correlation + 1) / 2

    return float(score)


# Convenience function
async def simulate_and_score(
    explanation: str,
    tokens: Sequence[str],
    actual_activations: Sequence[float],
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> Tuple[SequenceSimulation, float]:
    """
    Simulate activations and compute score in one call.

    Returns:
        Tuple of (simulation result, correlation score)
    """
    simulator = ModernNeuronSimulator(
        explanation=explanation,
        api_key=api_key,
        model_name=model_name,
        use_async=True
    )

    simulation = await simulator.simulate_async(tokens)
    score = compute_simulation_score(simulation, actual_activations)

    return simulation, score
