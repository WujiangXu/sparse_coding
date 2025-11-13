"""
最简单的 neuron_explainer 库测试
假设你已经修复了源码的 API 问题
"""

import asyncio
import json

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.scoring import simulate_and_score, aggregate_scored_sequence_simulations
from neuron_explainer.explanations.prompt_builder import PromptFormat


# ========== 配置 ==========
with open("secrets.json") as f:
    OPENAI_API_KEY = json.load(f)["openai_key"]

# 你修复后的模型名
SIMULATOR_MODEL_NAME = "gpt-3.5-turbo"  # 或你使用的任何模型
MAX_CONCURRENT = 10


# ========== 测试1: 最基础的模拟 ==========
async def test_basic():
    print("="*60)
    print("Test 1: Basic Simulation")
    print("="*60)

    # 解释
    explanation = "this neuron activates for numeric values and digits"
    print(f"Explanation: {explanation}\n")

    # 验证数据（3个examples）
    validation_records = [
        ActivationRecord(
            tokens=["The", "price", "is", "$", "42"],
            activations=[0.1, 0.3, 0.1, 0.5, 0.95]
        ),
        ActivationRecord(
            tokens=["I", "bought", "3", "apples"],
            activations=[0.05, 0.2, 0.88, 0.15]
        ),
        ActivationRecord(
            tokens=["Walking", "in", "the", "park"],
            activations=[0.05, 0.08, 0.06, 0.1]
        ),
    ]

    print(f"Validation examples: {len(validation_records)}\n")

    # 创建 simulator
    format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING

    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=MAX_CONCURRENT,
            prompt_format=format,
        )
    )

    # 模拟和评分
    scored_simulation = await simulate_and_score(simulator, validation_records)
    score = scored_simulation.get_preferred_score()

    print(f"Overall Score: {score:.3f}\n")

    # 详细结果
    print("Detailed Results:")
    print("-" * 60)
    for i, seq_sim in enumerate(scored_simulation.scored_sequence_simulations):
        print(f"\nExample {i+1}: {' '.join(seq_sim.activation_record.tokens)}")
        print(f"  Actual:    {seq_sim.activation_record.activations}")
        print(f"  Predicted: {seq_sim.simulation.expected_activations}")
        print(f"  Score:     {seq_sim.score:.3f}")

    return scored_simulation


# ========== 测试2: 论文完整流程（10个examples）==========
async def test_paper_style():
    print("\n" + "="*60)
    print("Test 2: Paper-Style (10 examples)")
    print("="*60)

    explanation = "this neuron activates for the token 'the'"
    print(f"Explanation: {explanation}\n")

    # Top 5 examples
    top_records = [
        ActivationRecord(["In", "the", "beginning"], [0.1, 0.95, 0.05]),
        ActivationRecord(["At", "the", "store"], [0.08, 0.92, 0.1]),
        ActivationRecord(["From", "the", "top"], [0.12, 0.88, 0.15]),
        ActivationRecord(["To", "the", "moon"], [0.09, 0.91, 0.2]),
        ActivationRecord(["Under", "the", "sea"], [0.11, 0.89, 0.18]),
    ]

    # Random 5 examples
    random_records = [
        ActivationRecord(["I", "like", "cats"], [0.05, 0.08, 0.12]),
        ActivationRecord(["Running", "is", "fun"], [0.06, 0.09, 0.11]),
        ActivationRecord(["Blue", "sky", "today"], [0.07, 0.1, 0.08]),
        ActivationRecord(["Coffee", "time", "now"], [0.05, 0.11, 0.09]),
        ActivationRecord(["Happy", "coding", "day"], [0.08, 0.12, 0.1]),
    ]

    validation_records = top_records + random_records

    print(f"Top examples: {len(top_records)}")
    print(f"Random examples: {len(random_records)}")
    print(f"Total: {len(validation_records)}\n")

    # 创建 simulator
    format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING

    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=MAX_CONCURRENT,
            prompt_format=format,
        )
    )

    # 模拟和评分
    scored_simulation = await simulate_and_score(simulator, validation_records)

    # 计算三种分数（论文风格）
    score = scored_simulation.get_preferred_score()

    assert len(scored_simulation.scored_sequence_simulations) == 10

    top_only_score = aggregate_scored_sequence_simulations(
        scored_simulation.scored_sequence_simulations[:5]
    ).get_preferred_score()

    random_only_score = aggregate_scored_sequence_simulations(
        scored_simulation.scored_sequence_simulations[5:]
    ).get_preferred_score()

    # 打印结果（完全按照论文格式）
    print("="*60)
    print(f"score={score:.2f}, top_only_score={top_only_score:.2f}, random_only_score={random_only_score:.2f}")
    print("="*60)

    return scored_simulation


# ========== 测试3: 直接调用 simulate（不用 scoring）==========
async def test_direct_simulate():
    print("\n" + "="*60)
    print("Test 3: Direct Simulate (No Scoring)")
    print("="*60)

    explanation = "numeric values"
    tokens = ["The", "price", "is", "$", "42"]

    print(f"Explanation: {explanation}")
    print(f"Tokens: {tokens}\n")

    # 创建 simulator
    format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING

    base_simulator = ExplanationNeuronSimulator(
        SIMULATOR_MODEL_NAME,
        explanation,
        max_concurrent=MAX_CONCURRENT,
        prompt_format=format,
    )

    # 直接模拟（不需要 ActivationRecord）
    simulation = await base_simulator.simulate(tokens)

    print(f"Predicted activations: {simulation.expected_activations}")
    print(f"Activation scale: {simulation.activation_scale}")

    if simulation.distribution_values:
        print(f"\nDistribution info available:")
        print(f"  Values: {simulation.distribution_values[0][:5]}...")  # 第一个token的前5个值
        print(f"  Probs:  {simulation.distribution_probabilities[0][:5]}...")

    return simulation


# ========== 主函数 ==========
async def main():
    print("\n🧪 NEURON EXPLAINER LIBRARY TEST")
    print("="*60)
    print(f"Model: {SIMULATOR_MODEL_NAME}")
    print("="*60)

    # 测试1: 基础
    await test_basic()

    # 测试2: 论文风格
    await test_paper_style()

    # 测试3: 直接模拟
    await test_direct_simulate()

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
