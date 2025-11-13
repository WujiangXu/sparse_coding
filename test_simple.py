"""
Simple test: Can OpenAI API support echo + logprobs (required for neuron-explainer)?

This is what neuron-explainer needs to work.
"""

import os
import json
from openai import OpenAI

# Load API key
with open("secrets.json") as f:
    os.environ["OPENAI_API_KEY"] = json.load(f)["openai_key"]

client = OpenAI()

# Test models
models = ["gpt-3.5-turbo-instruct", "text-davinci-003"]

print("\n" + "="*60)
print("Testing: echo + logprobs support (needed for neuron-explainer)")
print("="*60)

for model in models:
    print(f"\nModel: {model}")
    try:
        response = client.completions.create(
            model=model,
            prompt="Test",
            max_tokens=1,
            echo=True,      # neuron-explainer needs this
            logprobs=5      # neuron-explainer needs this
        )
        print(f"  ✅ WORKS - echo + logprobs supported")
    except Exception as e:
        error = str(e)
        if "deprecated" in error.lower():
            print(f"  ❌ FAILED - Model deprecated")
        elif "echo" in error.lower() and "logprobs" in error.lower():
            print(f"  ❌ FAILED - echo + logprobs not supported")
        else:
            print(f"  ❌ FAILED - {error[:80]}")

print("\n" + "="*60)
print("RESULT:")
print("If all models failed → neuron-explainer CANNOT work")
print("If any model works → neuron-explainer CAN work with that model")
print("="*60 + "\n")
