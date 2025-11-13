"""
基于LogProbs的可解释性评估 - 演示版本

展示完整的评估流程，包含模拟数据演示和真实API调用两种模式
"""

import math
import numpy as np


def demonstrate_logprobs_concept():
    """
    演示LogProbs的核心概念（使用模拟数据）
    """

    print("=" * 80)
    print("LogProbs评估方法核心概念演示")
    print("=" * 80)

    print("\n真实数据：")
    print('文本: "The answer is 42"')
    tokens = ["The", "answer", "is", "42"]
    actual_activations = [0.1, 0.2, 0.15, 8.5]
    print(f"Tokens: {tokens}")
    print(f"真实激活值: {actual_activations}")

    print("\n步骤1：神经元解释（由GPT-4生成）")
    explanation = "这个神经元在遇到数字时激活"
    print(f"解释: {explanation}")

    print("\n" + "=" * 80)
    print("步骤2：对每个token使用LogProbs预测激活值")
    print("=" * 80)

    # Token 1: "The"
    print("\n【Token 1: 'The'】")
    print("  上下文: (开始)")
    print("  LLM返回的LogProbs (模拟数据):")

    # 模拟logprobs: "The" 不是数字，应该激活值很低
    logprobs_the = {
        "0": -0.5,   # log P(activation=0)
        "1": -3.2,   # log P(activation=1)
        "2": -5.1,
        "3": -6.8,
    }

    print("\n  原始LogProbs:")
    for token, logprob in logprobs_the.items():
        print(f"    激活值 '{token}': logprob = {logprob:.2f}")

    print("\n  转换为概率 (P = e^logprob):")
    probs_the = {}
    for token, logprob in logprobs_the.items():
        prob = math.exp(logprob)
        probs_the[int(token)] = prob
        print(f"    P(激活={token}) = e^{logprob:.2f} = {prob:.4f}")

    # 归一化
    total_prob = sum(probs_the.values())
    normalized_probs_the = {k: v/total_prob for k, v in probs_the.items()}

    print("\n  归一化后的概率分布:")
    for act_val, prob in sorted(normalized_probs_the.items()):
        print(f"    P(激活={act_val}) = {prob:.3f} ({prob*100:.1f}%)")

    # 计算期望值
    expected_the = sum(k * prob for k, prob in normalized_probs_the.items())
    print(f"\n  → 期望激活值 = Σ k·P(k) = {expected_the:.2f}")

    # Token 2: "answer"
    print("\n【Token 2: 'answer'】")
    print("  上下文: The")
    logprobs_answer = {"0": -1.2, "1": -2.0, "2": -3.5}
    probs_answer = {int(k): math.exp(v) for k, v in logprobs_answer.items()}
    total = sum(probs_answer.values())
    normalized_answer = {k: v/total for k, v in probs_answer.items()}
    expected_answer = sum(k * p for k, p in normalized_answer.items())
    print(f"  概率分布: {', '.join([f'P({k})={v:.2f}' for k, v in normalized_answer.items()])}")
    print(f"  → 期望激活值 = {expected_answer:.2f}")

    # Token 3: "is"
    print("\n【Token 3: 'is'】")
    print("  上下文: The answer")
    logprobs_is = {"0": -0.8, "1": -2.5, "2": -4.0}
    probs_is = {int(k): math.exp(v) for k, v in logprobs_is.items()}
    total = sum(probs_is.values())
    normalized_is = {k: v/total for k, v in probs_is.items()}
    expected_is = sum(k * p for k, p in normalized_is.items())
    print(f"  概率分布: {', '.join([f'P({k})={v:.2f}' for k, v in normalized_is.items()])}")
    print(f"  → 期望激活值 = {expected_is:.2f}")

    # Token 4: "42" - 这个是数字！
    print("\n【Token 4: '42'】⭐")
    print("  上下文: The answer is")
    print("  (这是一个数字，根据解释应该强激活！)")

    # 模拟logprobs: "42" 是数字，应该激活值很高
    logprobs_42 = {
        "0": -6.2,
        "1": -5.8,
        "2": -5.0,
        "3": -4.0,
        "4": -3.0,
        "5": -2.5,
        "6": -2.0,
        "7": -1.5,
        "8": -0.5,   # 最可能！
        "9": -1.2,
        "10": -2.8,
    }

    print("\n  原始LogProbs:")
    for token, logprob in sorted(logprobs_42.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    激活值 '{token}': logprob = {logprob:.2f}")

    probs_42 = {int(k): math.exp(v) for k, v in logprobs_42.items()}
    total_prob = sum(probs_42.values())
    normalized_probs_42 = {k: v/total_prob for k, v in probs_42.items()}

    print("\n  归一化后的概率分布 (Top 5):")
    sorted_probs = sorted(normalized_probs_42.items(), key=lambda x: x[1], reverse=True)
    for act_val, prob in sorted_probs[:5]:
        print(f"    P(激活={act_val}) = {prob:.3f} ({prob*100:.1f}%)")

    # 计算期望值
    expected_42 = sum(k * prob for k, prob in normalized_probs_42.items())
    print(f"\n  → 期望激活值 = Σ k·P(k)")
    print(f"    = 0×{normalized_probs_42[0]:.3f} + 1×{normalized_probs_42[1]:.3f} + ... + 8×{normalized_probs_42[8]:.3f} + 9×{normalized_probs_42[9]:.3f} + 10×{normalized_probs_42[10]:.3f}")
    print(f"    = {expected_42:.2f}")

    print("\n" + "=" * 80)
    print("步骤3：计算相关性得分")
    print("=" * 80)

    predicted_activations = [expected_the, expected_answer, expected_is, expected_42]

    print(f"\n预测激活值: {[f'{x:.2f}' for x in predicted_activations]}")
    print(f"真实激活值: {actual_activations}")

    # 归一化真实值到0-10
    actual_max = max(actual_activations)
    actual_normalized = [(x / actual_max) * 10.0 for x in actual_activations]
    print(f"\n归一化真实值 (到0-10范围): {[f'{x:.2f}' for x in actual_normalized]}")

    # 计算Pearson相关系数
    predicted_array = np.array(predicted_activations)
    actual_array = np.array(actual_normalized)

    print("\nPearson相关系数计算:")
    print(f"  预测值: {predicted_array}")
    print(f"  归一化真实值: {actual_array}")

    correlation = np.corrcoef(predicted_array, actual_array)[0, 1]
    print(f"\n  相关系数 r = {correlation:.4f}")

    # 转换到0-1范围
    score = (correlation + 1) / 2
    print(f"\n  最终得分 = (r + 1) / 2 = ({correlation:.4f} + 1) / 2 = {score:.4f}")

    if score > 0.8:
        print("\n  ✓ 得分 > 0.8，解释非常准确！")
    elif score > 0.6:
        print("\n  ○ 得分在0.6-0.8之间，解释部分准确")
    else:
        print("\n  ✗ 得分 < 0.6，解释可能不够准确")

    print("\n" + "=" * 80)

    return {
        "tokens": tokens,
        "actual": actual_activations,
        "predicted": predicted_activations,
        "score": score
    }


def demonstrate_good_vs_bad():
    """
    对比演示：好的解释 vs 坏的解释
    """

    print("\n\n" + "=" * 80)
    print("对比实验：好的解释 vs 坏的解释")
    print("=" * 80)

    tokens = ["There", "are", "100", "students"]
    actual = [0.1, 0.2, 9.5, 0.15]

    print(f"\n测试数据: {' '.join(tokens)}")
    print(f"真实激活值: {actual}")
    print("(注意：第3个token '100' 激活值最高)")

    # 好的解释
    print("\n" + "-" * 80)
    print("【场景1：好的解释】")
    good_explanation = "这个神经元在遇到数字时激活"
    print(f"解释: {good_explanation}")

    # 模拟好解释的预测（应该在"100"上预测高）
    predicted_good = [0.15, 0.25, 8.2, 0.18]
    print(f"预测激活值: {predicted_good}")

    actual_normalized = np.array(actual) / max(actual) * 10.0
    corr_good = np.corrcoef(predicted_good, actual_normalized)[0, 1]
    score_good = (corr_good + 1) / 2

    print(f"相关系数: {corr_good:.4f}")
    print(f"得分: {score_good:.4f} ✓")

    # 坏的解释
    print("\n" + "-" * 80)
    print("【场景2：坏的解释】")
    bad_explanation = "这个神经元在遇到动词时激活"
    print(f"解释: {bad_explanation}")

    # 模拟坏解释的预测（应该在"are"上预测高，但实际是"100"高）
    predicted_bad = [0.2, 7.5, 0.3, 0.15]
    print(f"预测激活值: {predicted_bad}")

    corr_bad = np.corrcoef(predicted_bad, actual_normalized)[0, 1]
    score_bad = (corr_bad + 1) / 2

    print(f"相关系数: {corr_bad:.4f}")
    print(f"得分: {score_bad:.4f} ✗")

    print("\n" + "=" * 80)
    print("【对比结果】")
    print(f"好的解释得分: {score_good:.4f}")
    print(f"坏的解释得分: {score_bad:.4f}")
    print(f"差异: {abs(score_good - score_bad):.4f}")
    print("\n✓ 好的解释得到更高分数，评估方法有效！")
    print("=" * 80)


def explain_why_logprobs():
    """
    解释为什么使用LogProbs而不是直接采样
    """

    print("\n\n" + "=" * 80)
    print("为什么使用LogProbs？")
    print("=" * 80)

    print("\n【问题】：为什么不直接让LLM输出一个数字？")

    print("\n方法1：直接采样（temperature=0）")
    print("  Prompt: '预测token \"42\" 的激活值（0-10）：'")
    print("  输出: '8'")
    print("  问题：")
    print("    - 单点预测，不稳定")
    print("    - 丢失不确定性信息")
    print("    - 如果LLM在8和9之间犹豫，只能选一个")

    print("\n方法2：使用LogProbs（论文方法）⭐")
    print("  Prompt: '预测token \"42\" 的激活值（0-10）：'")
    print("  获取LogProbs:")
    print("    P(激活=7) = 12%")
    print("    P(激活=8) = 61%  ← 最可能")
    print("    P(激活=9) = 30%  ← 也有可能！")
    print("    P(激活=10) = 3%")
    print("  计算期望值: 7×0.12 + 8×0.61 + 9×0.30 + 10×0.03 = 8.2")
    print("  优势：")
    print("    ✓ 更稳定（加权平均）")
    print("    ✓ 反映不确定性（如果分布很分散，说明LLM不确定）")
    print("    ✓ 更精细（可以是小数：8.2）")

    print("\n【示例对比】")
    print("\n情况A：LLM非常确定")
    print("  LogProbs: P(8)=95%, P(7)=3%, P(9)=2%")
    print("  期望值: 8.0")
    print("  直接采样: 8")
    print("  → 两种方法结果相近")

    print("\n情况B：LLM不确定")
    print("  LogProbs: P(3)=25%, P(4)=25%, P(5)=25%, P(6)=25%")
    print("  期望值: 4.5")
    print("  直接采样: 3或4或5或6（随机）")
    print("  → LogProbs更稳定！")

    print("\n" + "=" * 80)


def show_api_example():
    """
    展示真实的OpenAI API调用代码
    """

    print("\n\n" + "=" * 80)
    print("真实API调用示例代码")
    print("=" * 80)

    code = '''
from openai import OpenAI

client = OpenAI()  # 需要设置 OPENAI_API_KEY

# 构造prompt
explanation = "这个神经元在遇到数字时激活"
token = "42"

prompt = f"""你是一个神经元模拟器。
神经元行为：{explanation}

预测token的激活值（0-10的整数）：
Token: {token}
激活值："""

# 调用API获取logprobs
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    logprobs=True,      # 开启logprobs
    top_logprobs=20,    # 返回top 20个最可能的token
    max_tokens=1,       # 只生成第一个token
    temperature=1.0     # 使用默认温度获得真实概率分布
)

# 解析logprobs
logprobs_data = response.choices[0].logprobs.content[0].top_logprobs

print("LLM返回的LogProbs:")
for item in logprobs_data:
    token = item.token.strip()
    logprob = item.logprob
    probability = math.exp(logprob)
    print(f"  Token '{token}': logprob={logprob:.2f}, prob={probability:.3f}")

# 计算期望激活值
activation_probs = {}
for item in logprobs_data:
    token_str = item.token.strip()
    if token_str.isdigit():
        act_val = int(token_str)
        if 0 <= act_val <= 10:
            activation_probs[act_val] = math.exp(item.logprob)

# 归一化并计算期望
total = sum(activation_probs.values())
expected = sum(k * (v/total) for k, v in activation_probs.items())
print(f"\\n期望激活值: {expected:.2f}")
'''

    print(code)
    print("\n注意：运行此代码需要：")
    print("  1. 安装 openai 库: pip install openai")
    print("  2. 设置环境变量: export OPENAI_API_KEY='your-key'")
    print("  3. 有可用的API credits")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 主演示
    result = demonstrate_logprobs_concept()

    # 对比实验
    demonstrate_good_vs_bad()

    # 解释原理
    explain_why_logprobs()

    # API代码示例
    show_api_example()

    print("\n\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
论文的LogProbs评估方法核心流程：

1️⃣  数据收集：获取真实的token激活值
2️⃣  生成解释：用GPT-4总结神经元的行为模式
3️⃣  预测激活：对每个token：
    - 构造prompt（包含解释）
    - 调用LLM API，开启logprobs
    - 获取0-10每个激活值的概率分布
    - 计算期望值：E[X] = Σ k·P(X=k)
4️⃣  计算得分：Pearson相关系数（预测值 vs 真实值）

为什么用LogProbs？
  ✓ 更稳定（概率分布的期望值，而非单点采样）
  ✓ 反映不确定性（分布分散 = LLM不确定）
  ✓ 更精细（可以是小数，如8.4）

关键创新：
  不是让LLM直接预测数字，而是获取完整的概率分布！
""")
    print("=" * 80)
