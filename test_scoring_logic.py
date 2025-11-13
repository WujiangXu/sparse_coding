"""
单独测试 Simulator 的核心：预测激活 + 计算分数
不需要完整的 interpret 流程，只测试这个核心逻辑
"""

import asyncio
import json
import numpy as np
from typing import List

from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.explanations.prompt_builder import PromptFormat


# ========== 配置 ==========
with open("secrets.json") as f:
    API_KEY = json.load(f)["openai_key"]

SIMULATOR_MODEL_NAME = "gpt-3.5-turbo"  # 你修复后的模型
MAX_CONCURRENT = 10


# ========== 核心函数：模拟激活 ==========
async def simulate_activations(
    explanation: str,
    tokens: List[str],
    model_name: str = SIMULATOR_MODEL_NAME
) -> List[float]:
    """
    核心功能：给定解释和tokens，预测激活值

    这就是原始代码中 simulator.simulate() 的部分
    """
    # 创建 simulator
    format = PromptFormat.HARMONY_V4 if model_name == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING

    simulator = ExplanationNeuronSimulator(
        model_name,
        explanation,
        max_concurrent=MAX_CONCURRENT,
        prompt_format=format,
    )

    # 调用 LLM 预测激活
    simulation = await simulator.simulate(tokens)

    return simulation.expected_activations


# ========== 核心函数：计算分数 ==========
def compute_correlation_score(
    predicted: List[float],
    actual: List[float]
) -> float:
    """
    核心功能：计算预测和实际激活的相关性

    这就是原始代码中 scoring 的部分
    返回 0-1 的分数，越高越好
    """
    predicted_array = np.array(predicted)
    actual_array = np.array(actual)

    # 归一化 actual 到 0-10 scale（和 predicted 匹配）
    if actual_array.max() > 0:
        actual_normalized = (actual_array / actual_array.max()) * 10.0
    else:
        actual_normalized = actual_array

    # 检查标准差
    if predicted_array.std() == 0 or actual_normalized.std() == 0:
        return 0.0

    # 计算 Pearson 相关系数
    correlation = np.corrcoef(predicted_array, actual_normalized)[0, 1]

    # 转换到 0-1 范围
    score = (correlation + 1) / 2

    return float(score)


# ========== 测试1: 单个例子 ==========
async def test_single_example():
    """测试单个 token 序列"""
    print("="*60)
    print("Test 1: Single Example")
    print("="*60)

    # Mock: 已经有了解释
    explanation = "this neuron activates for numeric values and digits"

    # 测试数据
    tokens = ["The", "price", "is", "$", "42"]
    actual_activations = [0.1, 0.3, 0.1, 0.5, 0.95]

    print(f"\nExplanation: {explanation}")
    print(f"Tokens: {tokens}")
    print(f"Actual activations: {actual_activations}")

    # Step 1: 用 LLM 预测激活
    print("\n[Step 1] Calling LLM to predict activations...")
    predicted_activations = await simulate_activations(explanation, tokens)

    print(f"Predicted activations: {predicted_activations}")

    # Step 2: 计算分数
    print("\n[Step 2] Computing correlation score...")
    score = compute_correlation_score(predicted_activations, actual_activations)

    print(f"\n{'='*60}")
    print(f"SCORE: {score:.3f}")
    print(f"{'='*60}")

    if score > 0.7:
        print("✅ Excellent! High correlation.")
    elif score > 0.5:
        print("✓ Good. Moderate correlation.")
    else:
        print("⚠️ Low correlation. Explanation may be inaccurate.")

    return score


# ========== 测试2: 多个例子（平均分数）==========
async def test_multiple_examples():
    """测试多个例子，计算平均分数"""
    print("\n" + "="*60)
    print("Test 2: Multiple Examples")
    print("="*60)

    # Mock: 已经有了解释
    explanation = "this neuron activates for the token 'the'"

    # 多个测试案例
    test_cases = [
        {
            "tokens": ["In", "the", "beginning"],
            "actual": [0.1, 0.95, 0.05]
        },
        {
            "tokens": ["At", "the", "store"],
            "actual": [0.08, 0.92, 0.1]
        },
        {
            "tokens": ["I", "like", "cats"],
            "actual": [0.05, 0.08, 0.12]  # 无 'the'，低激活
        },
    ]

    print(f"\nExplanation: {explanation}")
    print(f"Test cases: {len(test_cases)}\n")

    scores = []

    for i, case in enumerate(test_cases, 1):
        print(f"Case {i}: {' '.join(case['tokens'])}")

        # 预测
        predicted = await simulate_activations(explanation, case["tokens"])

        # 计算分数
        score = compute_correlation_score(predicted, case["actual"])
        scores.append(score)

        print(f"  Actual:    {case['actual']}")
        print(f"  Predicted: {predicted}")
        print(f"  Score:     {score:.3f}\n")

    # 平均分数
    avg_score = np.mean(scores)

    print(f"{'='*60}")
    print(f"AVERAGE SCORE: {avg_score:.3f}")
    print(f"Individual scores: {[f'{s:.3f}' for s in scores]}")
    print(f"{'='*60}")

    return avg_score, scores


# ========== 测试3: 对比好坏解释 ==========
async def test_good_vs_bad():
    """对比好解释 vs 坏解释的分数差异"""
    print("\n" + "="*60)
    print("Test 3: Good vs Bad Explanations")
    print("="*60)

    # 测试数据（实际激活高在数字上）
    tokens = ["The", "price", "is", "$", "42"]
    actual = [0.1, 0.3, 0.1, 0.5, 0.95]

    print(f"Tokens: {tokens}")
    print(f"Actual activations: {actual}")
    print(f"(High on position 4: '42')\n")

    # 不同的解释
    explanations = [
        ("numeric values and digits", "Good"),
        ("the token 'the'", "Bad"),
        ("negative sentiment", "Bad"),
    ]

    print(f"{'Explanation':<40} {'Type':<10} {'Score'}")
    print("-" * 60)

    for explanation, exp_type in explanations:
        # 预测
        predicted = await simulate_activations(explanation, tokens)

        # 计算分数
        score = compute_correlation_score(predicted, actual)

        status = "✅" if score > 0.6 else "❌"
        print(f"{explanation:<40} {exp_type:<10} {score:.3f} {status}")

    print("\n注意: 好的解释 → 高分数，坏的解释 → 低分数")


# ========== 测试4: 逐步展示计算过程 ==========
async def test_detailed_computation():
    """展示详细的计算过程"""
    print("\n" + "="*60)
    print("Test 4: Detailed Computation Breakdown")
    print("="*60)

    explanation = "numeric values"
    tokens = ["The", "number", "is", "42"]
    actual = [0.1, 0.2, 0.1, 0.95]

    print(f"\nExplanation: {explanation}")
    print(f"Tokens: {tokens}")
    print(f"Actual: {actual}\n")

    # Step 1: 预测
    print("[Step 1] Predicting activations...")
    predicted = await simulate_activations(explanation, tokens)
    print(f"Predicted (0-10 scale): {predicted}\n")

    # Step 2: 归一化 actual
    print("[Step 2] Normalizing actual to 0-10 scale...")
    actual_array = np.array(actual)
    actual_normalized = (actual_array / actual_array.max()) * 10.0
    print(f"Actual normalized: {actual_normalized.tolist()}\n")

    # Step 3: 逐位置对比
    print("[Step 3] Position-by-position comparison:")
    print(f"{'Position':<10} {'Token':<10} {'Actual(0-10)':<15} {'Predicted':<12} {'Diff'}")
    print("-" * 60)
    for i, (token, act, pred) in enumerate(zip(tokens, actual_normalized, predicted)):
        diff = abs(act - pred)
        print(f"{i:<10} {token:<10} {act:<15.2f} {pred:<12.2f} {diff:.2f}")

    # Step 4: 计算相关性
    print("\n[Step 4] Computing Pearson correlation...")
    predicted_array = np.array(predicted)
    correlation = np.corrcoef(predicted_array, actual_normalized)[0, 1]
    print(f"Pearson correlation: {correlation:.3f}")

    # Step 5: 转换到 0-1
    print("\n[Step 5] Converting to 0-1 score...")
    score = (correlation + 1) / 2
    print(f"Final score: {score:.3f}")

    print(f"\n{'='*60}")
    print(f"RESULT: {score:.3f}")
    print(f"{'='*60}")

    return score


# ========== 测试5: 批量验证（论文风格）==========
async def test_batch_validation():
    """
    模拟论文中的验证流程：
    1. 有一个解释
    2. 在多个验证样本上测试
    3. 计算平均分数
    """
    print("\n" + "="*60)
    print("Test 5: Batch Validation (Paper Style)")
    print("="*60)

    # Mock: 已经生成的解释
    explanation = "this neuron activates for numeric values and digits"

    # 验证集（10个样本）
    validation_set = [
        # Top examples (高激活)
        (["The", "price", "is", "$", "42"], [0.1, 0.3, 0.1, 0.5, 0.95]),
        (["I", "bought", "3", "apples"], [0.05, 0.2, 0.88, 0.15]),
        (["Chapter", "7", "begins"], [0.1, 0.91, 0.15]),
        (["Version", "2", ".", "0"], [0.2, 0.85, 0.4, 0.87]),
        (["There", "are", "100", "people"], [0.1, 0.15, 0.90, 0.2]),
        # Random examples (一些激活)
        (["I", "like", "cats"], [0.05, 0.08, 0.12]),
        (["Running", "is", "fun"], [0.06, 0.09, 0.11]),
        (["Blue", "sky", "today"], [0.07, 0.1, 0.08]),
        (["Coffee", "time", "now"], [0.05, 0.11, 0.09]),
        (["Happy", "coding", "day"], [0.08, 0.12, 0.1]),
    ]

    print(f"Explanation: {explanation}")
    print(f"Validation samples: {len(validation_set)}\n")

    scores = []

    for i, (tokens, actual) in enumerate(validation_set, 1):
        # 预测
        predicted = await simulate_activations(explanation, tokens)

        # 计算分数
        score = compute_correlation_score(predicted, actual)
        scores.append(score)

        if i <= 5:
            category = "Top"
        else:
            category = "Random"

        print(f"{i:2d}. [{category:6}] {' '.join(tokens):30s} Score: {score:.3f}")

    # 统计
    overall_score = np.mean(scores)
    top_score = np.mean(scores[:5])
    random_score = np.mean(scores[5:])

    print(f"\n{'='*60}")
    print(f"RESULTS (Paper Style):")
    print(f"  Overall score:      {overall_score:.3f}")
    print(f"  Top-only score:     {top_score:.3f}")
    print(f"  Random-only score:  {random_score:.3f}")
    print(f"{'='*60}")

    return overall_score, top_score, random_score


# ========== 主函数 ==========
async def main():
    print("\n🧪 CORE SCORING LOGIC TEST")
    print("="*60)
    print("Testing: Explanation → LLM Prediction → Correlation Score")
    print("="*60)

    # 测试1: 单个例子
    await test_single_example()

    # 测试2: 多个例子
    await test_multiple_examples()

    # 测试3: 好坏对比
    await test_good_vs_bad()

    # 测试4: 详细过程
    await test_detailed_computation()

    # 测试5: 批量验证
    await test_batch_validation()

    print("\n" + "="*60)
    print("✅ All core tests completed!")
    print("="*60)

    print("\n💡 Summary:")
    print("1. simulate_activations() - 用LLM预测激活")
    print("2. compute_correlation_score() - 计算相关性分数")
    print("3. 这就是论文核心评估逻辑！")


if __name__ == "__main__":
    asyncio.run(main())
