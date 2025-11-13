"""
Minimal Interpretability Evaluation for Sparse Autoencoders
Simple, clean implementation using OpenAI's latest API
"""

import json
import numpy as np
from openai import OpenAI


def evaluate_feature_interpretability(
    tokens_list,
    activations_list,
    api_key=None,
    n_examples=5,
    model="gpt-4"
):
    """
    Evaluate how interpretable a sparse autoencoder feature is.

    Args:
        tokens_list: List of token sequences, e.g., [["The", "cat", "sat"], ["I", "like", "dogs"], ...]
        activations_list: List of activation values per sequence, e.g., [[0.1, 0.9, 0.2], [0.8, 0.3, 0.1], ...]
        api_key: OpenAI API key (or loads from secrets.json)
        n_examples: Number of top examples to use for explanation
        model: OpenAI model to use

    Returns:
        dict: {
            "explanation": str,
            "score": float (0-1, higher = more interpretable),
            "examples_used": int
        }
    """

    # Load API key
    if api_key is None:
        with open("secrets.json") as f:
            api_key = json.load(f)["openai_key"]

    client = OpenAI(api_key=api_key)

    # Sort examples by max activation
    max_activations = [max(acts) for acts in activations_list]
    sorted_indices = np.argsort(max_activations)[::-1]  # Descending

    # Take top N examples
    top_indices = sorted_indices[:n_examples]

    # Format examples for prompt
    prompt_parts = []
    for i, idx in enumerate(top_indices):
        tokens = tokens_list[idx]
        acts = activations_list[idx]

        # Normalize activations
        if max(acts) > 0:
            acts = [a / max(acts) for a in acts]

        # Format as: token1(0.8) token2(0.3) token3(0.9)
        formatted = " ".join([f"{tok}({act:.2f})" for tok, act in zip(tokens, acts)])
        prompt_parts.append(f"Example {i+1}: {formatted}")

    prompt = f"""Below are text sequences where a neural network feature activates strongly. Numbers in parentheses show activation strength (0-1).

{chr(10).join(prompt_parts)}

What pattern does this feature detect? Answer in one sentence."""

    # Get explanation
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert at analyzing neural network features."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=100
    )

    explanation = response.choices[0].message.content.strip()

    # Score the explanation by testing on held-out examples
    # Use examples not in top N
    val_indices = sorted_indices[n_examples:n_examples+5]

    scores = []
    for idx in val_indices:
        if idx >= len(tokens_list):
            continue

        tokens = tokens_list[idx]
        actual_acts = activations_list[idx]

        # Normalize actual
        if max(actual_acts) > 0:
            actual_acts = [a / max(actual_acts) for a in actual_acts]
        else:
            continue

        # Ask model to predict activations
        tokens_str = " ".join(tokens)
        pred_prompt = f"""A neural network feature detects: "{explanation}"

Rate each token's activation (0-10) in: {tokens_str}

Respond with only comma-separated numbers."""

        pred_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": pred_prompt}],
            temperature=0.0,
            max_tokens=100
        )

        try:
            # Parse predictions
            pred_text = pred_response.choices[0].message.content.strip()
            predicted = [float(x.strip()) / 10.0 for x in pred_text.split(",")]

            # Ensure same length
            if len(predicted) != len(actual_acts):
                predicted = predicted[:len(actual_acts)] + [0.0] * max(0, len(actual_acts) - len(predicted))

            # Compute correlation
            if len(predicted) == len(actual_acts):
                correlation = np.corrcoef(predicted, actual_acts)[0, 1]
                if not np.isnan(correlation):
                    # Convert to 0-1 scale
                    score = (correlation + 1) / 2
                    scores.append(score)
        except:
            continue

    overall_score = np.mean(scores) if scores else 0.0

    return {
        "explanation": explanation,
        "score": float(overall_score),
        "examples_used": len(top_indices)
    }
