"""
测试原始论文的 Simulator (neuron_explainer 库)

注意：原始代码使用 text-davinci-003 和 Completions API (已废弃)
这个脚本展示如何使用原始的类结构，但需要修复 API 问题
"""

import asyncio
import json
import sys
from typing import List

# 导入原始 neuron_explainer 库的类
try:
    from neuron_explainer.activations.activations import ActivationRecord
    from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
    from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
    from neuron_explainer.explanations.scoring import simulate_and_score, aggregate_scored_sequence_simulations
    from neuron_explainer.explanations.prompt_builder import PromptFormat
except ImportError as e:
    print("Error: neuron_explainer library not found!")
    print("This test requires the original neuron_explainer library.")
    print(f"Error: {e}")
    sys.exit(1)


# ========== 配置 ==========
with open("secrets.json") as f:
    secrets = json.load(f)
    API_KEY = secrets["openai_key"]

# 原始使用 text-davinci-003，但该模型已废弃
# 替代方案：
# 1. gpt-3.5-turbo-instruct (保留 Completions API 格式)
# 2. gpt-3.5-turbo (需要改用 Chat Completions API)
SIMULATOR_MODEL_NAME = "gpt-3.5-turbo-instruct"  # 最接近原始的替代
# SIMULATOR_MODEL_NAME = "gpt-3.5-turbo"  # 需要 HARMONY_V4 格式

MAX_CONCURRENT = 10


# ========== 测试1: 基础模拟（原始API风格）==========
async def test_basic_simulation():
    """
    测试原始 ExplanationNeuronSimulator

    问题：text-davinci-003 已废弃
    解决：使用 gpt-3.5-turbo-instruct
    """
    print("="*60)
    print("Test 1: Basic Simulation (Original Style)")
    print("="*60)

    # 1. 准备数据 - 使用 ActivationRecord
    explanation = "this neuron activates for numeric values and digits"

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
            activations=[0.05, 0.08, 0.06, 0.1]  # 无数字，低激活
        ),
    ]

    # 2. 选择 prompt format
    if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo":
        format = PromptFormat.HARMONY_V4
    else:
        format = PromptFormat.INSTRUCTION_FOLLOWING

    print(f"\nExplanation: {explanation}")
    print(f"Simulator model: {SIMULATOR_MODEL_NAME}")
    print(f"Prompt format: {format}")
    print(f"Validation records: {len(validation_records)}")

    # 3. 创建 Simulator
    try:
        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                SIMULATOR_MODEL_NAME,
                explanation,
                max_concurrent=MAX_CONCURRENT,
                prompt_format=format,
            )
        )
        print("✓ Simulator created")
    except Exception as e:
        print(f"✗ Error creating simulator: {e}")
        return None

    # 4. 模拟和评分
    try:
        print("\nSimulating activations...")
        scored_simulation = await simulate_and_score(
            simulator,
            validation_records
        )

        score = scored_simulation.get_preferred_score()
        print(f"\n✓ Overall Score: {score:.3f}")

        # 5. 打印详细结果
        print("\nDetailed Results:")
        for i, seq_sim in enumerate(scored_simulation.scored_sequence_simulations):
            print(f"\nRecord {i+1}:")
            print(f"  Tokens: {seq_sim.activation_record.tokens}")
            print(f"  Actual:    {seq_sim.activation_record.activations}")
            print(f"  Predicted: {seq_sim.simulation.expected_activations}")
            print(f"  Score: {seq_sim.score:.3f}")

        return scored_simulation

    except Exception as e:
        print(f"\n✗ Error during simulation: {e}")
        print(f"\nPossible issues:")
        print("1. Model name not supported (try gpt-3.5-turbo-instruct)")
        print("2. API format changed (try different PromptFormat)")
        print("3. neuron_explainer library needs updating")
        return None


# ========== 测试2: 完整流程（论文风格）==========
async def test_paper_style():
    """
    完全按照论文的方式：
    - Top examples + Random examples
    - 分成训练集和验证集
    - 计算3种分数：overall, top_only, random_only
    """
    print("\n" + "="*60)
    print("Test 2: Paper-Style Evaluation")
    print("="*60)

    explanation = "this neuron activates for the token 'the'"

    # 模拟论文中的数据结构
    # 通常有 10 个 top examples + 10 个 random examples
    # 这里简化为 5 + 5

    # Top 5 examples (高激活)
    top_records = [
        ActivationRecord(
            tokens=["In", "the", "beginning"],
            activations=[0.1, 0.95, 0.05]
        ),
        ActivationRecord(
            tokens=["At", "the", "store"],
            activations=[0.08, 0.92, 0.1]
        ),
        ActivationRecord(
            tokens=["From", "the", "top"],
            activations=[0.12, 0.88, 0.15]
        ),
        ActivationRecord(
            tokens=["To", "the", "moon"],
            activations=[0.09, 0.91, 0.2]
        ),
        ActivationRecord(
            tokens=["Under", "the", "sea"],
            activations=[0.11, 0.89, 0.18]
        ),
    ]

    # Random 5 examples (一些激活)
    random_records = [
        ActivationRecord(
            tokens=["I", "like", "cats"],
            activations=[0.05, 0.08, 0.12]
        ),
        ActivationRecord(
            tokens=["Running", "is", "fun"],
            activations=[0.06, 0.09, 0.11]
        ),
        ActivationRecord(
            tokens=["Blue", "sky", "today"],
            activations=[0.07, 0.1, 0.08]
        ),
        ActivationRecord(
            tokens=["Coffee", "time", "now"],
            activations=[0.05, 0.11, 0.09]
        ),
        ActivationRecord(
            tokens=["Happy", "coding", "day"],
            activations=[0.08, 0.12, 0.1]
        ),
    ]

    # 合并：前5个是top，后5个是random
    validation_records = top_records + random_records

    print(f"\nExplanation: {explanation}")
    print(f"Top examples: {len(top_records)}")
    print(f"Random examples: {len(random_records)}")
    print(f"Total validation: {len(validation_records)}")

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
    try:
        print("\nSimulating...")
        scored_simulation = await simulate_and_score(simulator, validation_records)

        # 整体分数
        score = scored_simulation.get_preferred_score()

        # 只看 top examples 的分数
        assert len(scored_simulation.scored_sequence_simulations) == 10
        top_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[:5]
        ).get_preferred_score()

        # 只看 random examples 的分数
        random_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[5:]
        ).get_preferred_score()

        # 打印结果（论文风格）
        print(f"\n{'='*60}")
        print(f"Results (Paper Style):")
        print(f"  Overall score:       {score:.3f}")
        print(f"  Top-only score:      {top_only_score:.3f}")
        print(f"  Random-only score:   {random_only_score:.3f}")
        print(f"{'='*60}")

        print(f"\nInterpretation:")
        if top_only_score > 0.8:
            print("✓ Good: High score on top-activating examples")
        if random_only_score > 0.3:
            print("✓ Good: Generalizes to random examples")
        if score > 0.6:
            print("✓ Overall: Good explanation quality")

        return scored_simulation

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return None


# ========== 测试3: 调试模式（看实际API调用）==========
async def test_debug_mode():
    """
    调试模式：打印更多信息，帮助理解原始代码
    """
    print("\n" + "="*60)
    print("Test 3: Debug Mode")
    print("="*60)

    explanation = "numeric values"
    tokens = ["The", "price", "is", "$", "42"]
    activations = [0.1, 0.3, 0.1, 0.5, 0.95]

    print(f"\nExplanation: {explanation}")
    print(f"Tokens: {tokens}")
    print(f"Actual activations: {activations}")

    # 创建单个 record
    record = ActivationRecord(tokens=tokens, activations=activations)

    # 创建 simulator
    format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING

    print(f"\nCreating simulator...")
    print(f"  Model: {SIMULATOR_MODEL_NAME}")
    print(f"  Format: {format}")

    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            SIMULATOR_MODEL_NAME,
            explanation,
            max_concurrent=MAX_CONCURRENT,
            prompt_format=format,
        )
    )

    # 直接调用 simulate (不通过 simulate_and_score)
    try:
        print("\nCalling simulator.simulate()...")
        base_simulator = simulator.simulator  # 获取内部的 ExplanationNeuronSimulator

        simulation = await base_simulator.simulate(tokens)

        print("\n✓ Simulation completed")
        print(f"\nResults:")
        print(f"  Tokens:     {simulation.tokens}")
        print(f"  Expected:   {simulation.expected_activations}")
        print(f"  Scale:      {simulation.activation_scale}")

        # 手动计算相关性
        import numpy as np
        predicted = np.array(simulation.expected_activations)
        actual = np.array(activations)

        # 归一化 actual 到 0-10
        if actual.max() > 0:
            actual_norm = (actual / actual.max()) * 10
        else:
            actual_norm = actual

        correlation = np.corrcoef(predicted, actual_norm)[0, 1]
        score = (correlation + 1) / 2

        print(f"\n  Correlation: {correlation:.3f}")
        print(f"  Score:       {score:.3f}")

        return simulation

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ========== 主函数 ==========
async def main():
    """运行所有测试"""
    print("\n" + "🧪 ORIGINAL SIMULATOR TEST SUITE")
    print("="*60)
    print("Testing neuron_explainer library's Simulator classes")
    print(f"Using model: {SIMULATOR_MODEL_NAME}")
    print("="*60)

    # 检查 API key
    if not API_KEY:
        print("\n✗ Error: No API key found in secrets.json")
        return

    # 测试1: 基础模拟
    result1 = await test_basic_simulation()

    if result1 is None:
        print("\n⚠️  Test 1 failed. Common issues:")
        print("1. text-davinci-003 is deprecated")
        print("   → Try: SIMULATOR_MODEL_NAME = 'gpt-3.5-turbo-instruct'")
        print("2. Completions API format changed")
        print("   → Try: PromptFormat.HARMONY_V4 with gpt-3.5-turbo")
        print("3. neuron_explainer library may need updates")
        print("\nSkipping remaining tests...")
        return

    # 测试2: 论文风格
    result2 = await test_paper_style()

    # 测试3: 调试模式
    result3 = await test_debug_mode()

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)


# ========== 修复建议 ==========
def print_fix_suggestions():
    """打印修复原始代码的建议"""
    print("\n" + "="*60)
    print("💡 HOW TO FIX ORIGINAL CODE FOR MODERN API")
    print("="*60)

    print("""
The original code uses deprecated APIs. Here are fixes:

1. MODEL CHANGE:
   Old: SIMULATOR_MODEL_NAME = "text-davinci-003"
   New: SIMULATOR_MODEL_NAME = "gpt-3.5-turbo-instruct"

   Or for chat models:
   New: SIMULATOR_MODEL_NAME = "gpt-3.5-turbo"

2. PROMPT FORMAT:
   For gpt-3.5-turbo:
       format = PromptFormat.HARMONY_V4

   For gpt-3.5-turbo-instruct:
       format = PromptFormat.INSTRUCTION_FOLLOWING

3. API CLIENT (in neuron_explainer library):
   The ApiClient class needs updating:

   Old (Completions API):
       response = await client.completions.create(
           prompt=prompt,
           echo=True,
           logprobs=15
       )

   New (Chat Completions API for gpt-3.5-turbo):
       response = await client.chat.completions.create(
           model=model,
           messages=[{"role": "user", "content": prompt}],
           logprobs=True,
           top_logprobs=15
       )

4. ALTERNATIVE: Use our modern_simulator.py
   Already fixed and working with latest APIs!

   from modern_simulator import ModernNeuronSimulator
   simulator = ModernNeuronSimulator(explanation, model_name="gpt-3.5-turbo")
""")


if __name__ == "__main__":
    print_fix_suggestions()

    print("\n" + "="*60)
    print("Starting tests...")
    print("="*60)

    asyncio.run(main())
