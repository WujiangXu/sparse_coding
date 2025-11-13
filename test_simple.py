"""
Simple test: Can OpenAI API support echo + logprobs (required for neuron-explainer)?

Based on OpenAI API migration:
- Old Completions API (text-davinci-003) → DEPRECATED
- New Chat Completions API → echo parameter NOT AVAILABLE
- gpt-3.5-turbo-instruct → Still uses Completions API, but echo support LIMITED
"""

import os
import json
from openai import OpenAI

# Load API key
with open("secrets.json") as f:
    os.environ["OPENAI_API_KEY"] = json.load(f)["openai_key"]

client = OpenAI()

print("\n" + "="*70)
print("Testing: echo + logprobs (required for neuron-explainer)")
print("="*70)

# Test 1: gpt-3.5-turbo-instruct (only viable option)
print(f"\nTest 1: gpt-3.5-turbo-instruct (Completions API)")
print("-" * 70)
try:
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt="Test",
        max_tokens=1,
        echo=True,      # neuron-explainer needs this
        logprobs=5      # neuron-explainer needs this
    )
    print(f"  ✅ SUCCESS - echo + logprobs both work!")
    print(f"     This means neuron-explainer CAN work with this model")
except Exception as e:
    error = str(e)
    if "echo" in error.lower() and "logprobs" in error.lower():
        print(f"  ❌ FAILED - echo + logprobs combination not supported")
        print(f"     Error: {error[:100]}")
    elif "echo" in error.lower():
        print(f"  ❌ FAILED - echo parameter not supported")
        print(f"     Error: {error[:100]}")
    else:
        print(f"  ❌ FAILED - {error[:100]}")
    print(f"     This means neuron-explainer CANNOT work")

# Test 2: Chat Completions API (for reference)
print(f"\nTest 2: gpt-3.5-turbo (Chat Completions API)")
print("-" * 70)
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=1,
        logprobs=True,
        top_logprobs=5
    )
    print(f"  ✅ logprobs work in Chat API")
    print(f"  ⚠️  But echo parameter is NOT AVAILABLE in Chat API")
    print(f"     neuron-explainer CANNOT use Chat Completions API")
except Exception as e:
    print(f"  ❌ Chat API test failed: {str(e)[:100]}")

# Final conclusion
print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("neuron-explainer requires: echo=True + logprobs")
print("- Chat Completions API: NO echo support → CANNOT work")
print("- gpt-3.5-turbo-instruct: Test result above shows if it works")
print("\nFor sparse_coding repository (interpret.py):")
print("- If Test 1 succeeds → Code CAN work (update model name)")
print("- If Test 1 fails → Code is BROKEN (need alternative method)")
print("="*70 + "\n")
