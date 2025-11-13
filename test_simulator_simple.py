"""
最简单的 Simulator 测试代码
展示如何使用 ModernNeuronSimulator 模拟激活并评分
"""

import asyncio
from modern_simulator import ModernNeuronSimulator, compute_simulation_score


# ========== 测试1: 同步模式（最简单）==========
def test_sync_basic():
    """最基础的同步测试"""
    print("="*60)
    print("Test 1: Basic Synchronous Simulation")
    print("="*60)

    # 1. 创建 simulator
    simulator = ModernNeuronSimulator(
        explanation="numeric values and digits",
        model_name="gpt-3.5-turbo"
    )

    # 2. 准备测试数据
    tokens = ["The", "price", "is", "$", "42"]

    # 3. 模拟激活
    simulation = simulator.simulate(tokens)

    # 4. 打印结果
    print(f"\nTokens: {simulation.tokens}")
    print(f"Predicted activations: {simulation.expected_activations}")
    print("\nExpected: High activation on '42', moderate on '$'")

    return simulation


# ========== 测试2: 带评分（完整流程）==========
def test_with_scoring():
    """带实际激活值的评分测试"""
    print("\n" + "="*60)
    print("Test 2: Simulation with Scoring")
    print("="*60)

    # 1. 创建 simulator
    simulator = ModernNeuronSimulator(
        explanation="numeric values and digits",
        model_name="gpt-3.5-turbo"
    )

    # 2. 准备测试数据和真实激活
    tokens = ["The", "price", "is", "$", "42"]
    actual_activations = [0.1, 0.3, 0.1, 0.5, 0.95]  # 真实激活（高在"42"）

    # 3. 模拟
    simulation = simulator.simulate(tokens)

    # 4. 评分
    score = compute_simulation_score(simulation, actual_activations)

    # 5. 打印结果
    print(f"\nTokens: {tokens}")
    print(f"Actual activations:    {actual_activations}")
    print(f"Predicted activations: {simulation.expected_activations}")
    print(f"\nScore: {score:.3f} (0=bad, 1=perfect)")

    if score > 0.7:
        print("✅ Good explanation! High correlation.")
    elif score > 0.5:
        print("⚠️  Moderate explanation.")
    else:
        print("❌ Poor explanation. Low correlation.")

    return simulation, score


# ========== 测试3: 异步模式（推荐）==========
async def test_async():
    """异步测试（可以批量处理）"""
    print("\n" + "="*60)
    print("Test 3: Async Simulation")
    print("="*60)

    # 1. 创建 async simulator
    simulator = ModernNeuronSimulator(
        explanation="negation words like 'not', 'never', 'no'",
        model_name="gpt-3.5-turbo",
        use_async=True
    )

    # 2. 准备多个测试案例
    test_cases = [
        {
            "tokens": ["I", "do", "not", "like", "this"],
            "actual": [0.1, 0.2, 0.95, 0.15, 0.1]
        },
        {
            "tokens": ["She", "never", "went", "there"],
            "actual": [0.1, 0.92, 0.15, 0.08]
        },
        {
            "tokens": ["The", "cat", "is", "sleeping"],
            "actual": [0.05, 0.08, 0.06, 0.1]  # 无否定词，低激活
        }
    ]

    # 3. 批量模拟
    print("\nSimulating on multiple examples...")
    scores = []

    for i, case in enumerate(test_cases, 1):
        simulation = await simulator.simulate_async(case["tokens"])
        score = compute_simulation_score(simulation, case["actual"])
        scores.append(score)

        print(f"\nCase {i}: {' '.join(case['tokens'])}")
        print(f"  Actual:    {case['actual']}")
        print(f"  Predicted: {simulation.expected_activations}")
        print(f"  Score: {score:.3f}")

    # 4. 总结
    avg_score = sum(scores) / len(scores)
    print(f"\n{'='*60}")
    print(f"Average score: {avg_score:.3f}")
    print(f"Interpretation: {'Good' if avg_score > 0.7 else 'Moderate' if avg_score > 0.5 else 'Poor'} explanation")

    return scores


# ========== 测试4: 逐步展示（教学用）==========
def test_step_by_step():
    """逐步展示每个环节，帮助理解"""
    print("\n" + "="*60)
    print("Test 4: Step-by-Step Walkthrough")
    print("="*60)

    # Step 1
    print("\n[Step 1] Create Simulator")
    explanation = "the token 'the'"
    print(f"Explanation: {explanation}")

    simulator = ModernNeuronSimulator(
        explanation=explanation,
        model_name="gpt-3.5-turbo"
    )
    print("✓ Simulator created")

    # Step 2
    print("\n[Step 2] Prepare Test Data")
    tokens = ["In", "the", "beginning", "there", "was", "the", "word"]
    actual_activations = [0.1, 0.95, 0.05, 0.15, 0.05, 0.92, 0.08]
    print(f"Tokens: {tokens}")
    print(f"Actual activations: {actual_activations}")
    print(f"Expected: High on positions 1 ('the') and 5 ('the')")

    # Step 3
    print("\n[Step 3] Simulate Activations")
    print("Calling OpenAI API...")
    simulation = simulator.simulate(tokens)
    print(f"✓ Got predictions: {simulation.expected_activations}")

    # Step 4
    print("\n[Step 4] Compute Score")
    score = compute_simulation_score(simulation, actual_activations)
    print(f"Correlation score: {score:.3f}")

    # Step 5
    print("\n[Step 5] Analysis")
    print("\nPosition-by-position comparison:")
    print(f"{'Token':<15} {'Actual':<10} {'Predicted':<10} {'Match?'}")
    print("-" * 50)
    for i, (token, actual, pred) in enumerate(zip(tokens, actual_activations, simulation.expected_activations)):
        # 归一化到0-10 scale比较
        actual_norm = actual / max(actual_activations) * 10
        match = "✓" if abs(actual_norm - pred) < 3 else "✗"
        print(f"{token:<15} {actual:<10.2f} {pred:<10.2f} {match}")

    print(f"\n{'='*60}")
    print(f"Final Score: {score:.3f}")
    if score > 0.8:
        print("✅ Excellent! The explanation accurately predicts activations.")
    elif score > 0.6:
        print("✓ Good. Reasonably accurate predictions.")
    else:
        print("⚠️  Moderate. Explanation may need refinement.")

    return simulation, score


# ========== 测试5: 比较不同解释（展示评分区别）==========
def test_compare_explanations():
    """比较好解释 vs 坏解释的分数差异"""
    print("\n" + "="*60)
    print("Test 5: Compare Good vs Bad Explanations")
    print("="*60)

    tokens = ["The", "price", "is", "$", "42"]
    actual_activations = [0.1, 0.3, 0.1, 0.5, 0.95]  # 高激活在"42"

    explanations = [
        ("numeric values and digits", "Good - should work"),
        ("negative sentiment", "Bad - wrong pattern"),
        ("verbs and actions", "Bad - wrong pattern"),
    ]

    print(f"\nTest tokens: {tokens}")
    print(f"Actual activations: {actual_activations}")
    print(f"(High activation on '42')\n")

    for explanation, description in explanations:
        print(f"\n{'-'*60}")
        print(f"Explanation: '{explanation}'")
        print(f"Description: {description}")

        simulator = ModernNeuronSimulator(
            explanation=explanation,
            model_name="gpt-3.5-turbo"
        )

        simulation = simulator.simulate(tokens)
        score = compute_simulation_score(simulation, actual_activations)

        print(f"Predicted: {simulation.expected_activations}")
        print(f"Score: {score:.3f} {'✓' if score > 0.6 else '✗'}")

    print(f"\n{'='*60}")
    print("Notice: Good explanation → High score, Bad explanation → Low score")


# ========== 主函数 ==========
def main():
    """运行所有测试"""
    print("\n" + "🧪 MODERN SIMULATOR TEST SUITE")
    print("="*60)
    print("Testing the simulator component that predicts activations")
    print("based on natural language explanations.\n")

    # 同步测试
    test_sync_basic()

    # 带评分
    test_with_scoring()

    # 逐步展示
    test_step_by_step()

    # 比较解释
    test_compare_explanations()

    print("\n" + "="*60)
    print("🎉 All synchronous tests completed!")
    print("="*60)

    # 异步测试
    print("\nRunning async test...")
    asyncio.run(test_async())

    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Simulator predicts activations based on explanations")
    print("2. Good explanations → High scores (>0.7)")
    print("3. Bad explanations → Low scores (<0.5)")
    print("4. Use async mode for batch processing")


if __name__ == "__main__":
    main()
