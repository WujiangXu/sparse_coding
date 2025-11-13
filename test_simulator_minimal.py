"""
最精简的 Simulator 测试 - 只有核心代码
"""

from modern_simulator import ModernNeuronSimulator, compute_simulation_score


# ========== 最简单的例子 ==========
def minimal_test():
    """只用5行代码测试 simulator"""

    # 1. 创建 simulator
    simulator = ModernNeuronSimulator(
        explanation="numeric values and digits",
        model_name="gpt-3.5-turbo"
    )

    # 2. 准备数据
    tokens = ["The", "price", "is", "$", "42"]
    actual_activations = [0.1, 0.3, 0.1, 0.5, 0.95]

    # 3. 模拟 + 评分
    simulation = simulator.simulate(tokens)
    score = compute_simulation_score(simulation, actual_activations)

    # 4. 打印结果
    print(f"Tokens: {tokens}")
    print(f"Actual:    {actual_activations}")
    print(f"Predicted: {simulation.expected_activations}")
    print(f"Score: {score:.3f}")


if __name__ == "__main__":
    minimal_test()
