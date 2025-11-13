# Sparse Autoencoder Interpretability Evaluation

A simplified, clean implementation of interpretability evaluation for sparse autoencoders using the latest OpenAI Chat Completions API.

## Overview

This tool evaluates how interpretable sparse autoencoder features are by:
1. **Explaining**: Using GPT-4 to generate natural language explanations based on top-activating examples
2. **Simulating**: Using GPT-3.5-turbo to predict activations based on the explanation
3. **Scoring**: Computing correlation between predicted and actual activations

Based on the methodology from ["Sparse Autoencoders Find Highly Interpretable Features in Language Models"](https://arxiv.org/pdf/2309.08600.pdf).

## Files

- `interpretability_eval.py` - Main evaluation classes and functions
- `test_interpretability_eval.py` - Test suite with synthetic examples
- This README

## Setup

### 1. Install Dependencies

```bash
pip install openai numpy asyncio
```

### 2. Configure API Key

Create a `secrets.json` file in the repo root:

```json
{
  "openai_key": "sk-proj-your-api-key-here"
}
```

### 3. Run Tests

```bash
python test_interpretability_eval.py
```

## Usage

### Basic Example

```python
import asyncio
from interpretability_eval import ActivationExample, InterpretabilityEvaluator

# Create examples (tokens + activation values per token)
train_examples = [
    ActivationExample(
        tokens=["The", "price", "is", "$", "42"],
        activations=[0.1, 0.3, 0.1, 0.5, 0.95]
    ),
    # ... more examples
]

val_examples = [
    # Similar structure
]

# Initialize evaluator
evaluator = InterpretabilityEvaluator(
    api_key="your-api-key",
    explainer_model="gpt-4",
    simulator_model="gpt-3.5-turbo"
)

# Evaluate
explanation, score, details = await evaluator.evaluate_feature(
    train_examples=train_examples,
    validation_examples=val_examples
)

print(f"Explanation: {explanation}")
print(f"Score: {score:.3f}")
```

### Integration with Real Sparse Autoencoder

```python
import torch
from transformer_lens import HookedTransformer
from autoencoders.learned_dict import LearnedDict
from interpretability_eval import ActivationExample, InterpretabilityEvaluator

# 1. Load models
autoencoder = torch.load("path/to/autoencoder.pt")
transformer = HookedTransformer.from_pretrained("gpt2")

# 2. Collect activations on dataset
tokens = transformer.to_tokens("Your text here")
_, cache = transformer.run_with_cache(tokens)
mlp_acts = cache["blocks.6.mlp.hook_post"]

# 3. Encode through sparse autoencoder
feature_acts = autoencoder.encode(mlp_acts)  # [batch, seq, n_features]

# 4. For each feature, find top-activating examples
feature_idx = 42
activations = feature_acts[:, :, feature_idx]

# Sort by max activation
max_acts = activations.max(dim=1).values
top_indices = torch.topk(max_acts, k=20).indices

# 5. Create ActivationExample objects
examples = []
for idx in top_indices:
    token_strs = transformer.to_str_tokens(tokens[idx])
    acts = activations[idx].tolist()
    examples.append(ActivationExample(token_strs, acts))

# 6. Split into train/validation
train_examples = examples[:10]
val_examples = examples[10:20]

# 7. Evaluate
evaluator = InterpretabilityEvaluator(api_key="your-key")
explanation, score, details = await evaluator.evaluate_feature(
    train_examples, val_examples
)
```

### Batch Evaluation

```python
from interpretability_eval import evaluate_features_batch

# Prepare data for multiple features
features_data = [
    (train_examples_f1, val_examples_f1),
    (train_examples_f2, val_examples_f2),
    # ...
]

# Evaluate all features with rate limiting
results = await evaluate_features_batch(
    evaluator=evaluator,
    features_data=features_data,
    max_concurrent=5  # Limit concurrent API calls
)

# results is a list of (explanation, score, details) tuples
for i, (explanation, score, details) in enumerate(results):
    print(f"Feature {i}: {score:.3f} - {explanation}")
```

## API Models

The evaluator uses two models:

1. **Explainer Model** (default: `gpt-4`)
   - Generates natural language explanations
   - More expensive but higher quality
   - Alternatives: `gpt-4-turbo`, `gpt-4o`

2. **Simulator Model** (default: `gpt-3.5-turbo`)
   - Predicts activations based on explanations
   - Faster and cheaper
   - Alternatives: `gpt-4`, `gpt-4o-mini`

## Scoring Metric

The evaluation uses **Pearson correlation** between predicted and actual activations:

- **Score = 1.0**: Perfect prediction (very interpretable)
- **Score = 0.5**: Random/no correlation
- **Score = 0.0**: Perfect anti-correlation

The score is normalized to [0, 1] range: `score = (correlation + 1) / 2`

## Output Format

The `evaluate_feature()` function returns:

```python
(explanation, overall_score, detailed_scores)
```

Where:
- `explanation`: String description of the feature
- `overall_score`: Mean correlation across validation examples
- `detailed_scores`: Dict with keys:
  - `overall`: Same as overall_score
  - `individual`: List of scores per validation example
  - `mean`, `std`, `min`, `max`: Statistics

## Comparison with Original Implementation

### Advantages of This Version

1. ✅ **Uses latest OpenAI API** (Chat Completions, not deprecated Completions)
2. ✅ **Clean, simple code** (~300 lines vs 800+ lines)
3. ✅ **Well-documented** with docstrings and examples
4. ✅ **Easy to modify** and extend
5. ✅ **No external dependencies** on neuron_explainer library

### Differences from Original

1. **Simplified scoring**: Uses Pearson correlation instead of complex simulation scoring
2. **Fewer parameters**: Focuses on core functionality
3. **No calibration**: Original used calibrated simulators (UncalibratedNeuronSimulator)
4. **Direct API calls**: Original used neuron_explainer abstractions

Both approaches follow the same core methodology: explain → simulate → score.

## Performance Tips

1. **Rate Limiting**: Use `max_concurrent` parameter to avoid API rate limits
2. **Model Selection**: Use `gpt-3.5-turbo` for simulator to reduce costs
3. **Batch Size**: Evaluate 5-10 validation examples per feature for balance of cost/accuracy
4. **Caching**: Cache results to avoid re-running expensive evaluations

## Troubleshooting

### API Key Issues

```
Error: Incorrect API key provided
```
→ Check your `secrets.json` file has the correct key

### Rate Limiting

```
Error: Rate limit exceeded
```
→ Reduce `max_concurrent` parameter or add delays between calls

### Parsing Errors

```
Error parsing simulation response
```
→ The simulator occasionally returns malformed output. The code handles this with fallback scores.

## Citation

If you use this tool, please cite the original paper:

```bibtex
@article{cunningham2023sparse,
  title={Sparse Autoencoders Find Highly Interpretable Features in Language Models},
  author={Cunningham, Hoagy and Ewart, Aidan and Riggs, Logan and Huben, Robert and Sharkey, Lee},
  journal={arXiv preprint arXiv:2309.08600},
  year={2023}
}
```

## License

MIT License - Same as the original sparse_coding repository
