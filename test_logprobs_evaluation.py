"""
基于Chat API LogProbs的可解释性评估完整实现

演示如何使用GPT-4的logprobs功能来预测激活值并计算评分
"""

import math
import numpy as np
from openai import OpenAI

client = OpenAI()


def get_activation_logprobs(explanation: str, token: str, context_before: str = "") -> dict:
    """
    获取单个token的激活值预测logprobs

    Args:
        explanation: 神经元行为的自然语言解释
        token: 要预测激活值的token
        context_before: token之前的上下文

    Returns:
        dict: {activation_value: probability}
    """

    # 构造prompt，让LLM预测激活值（0-10）
    prompt = f"""你是一个神经元模拟器。

神经元行为规则：{explanation}

任务：预测token的激活值（0-10的整数，0表示不激活，10表示最强激活）

上下文：{context_before if context_before else "无"}
当前token：{token}

请只输出一个0-10之间的数字，表示激活值："""

    # 调用Chat API，获取logprobs
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        logprobs=True,
        top_logprobs=20,  # 获取top 20个最可能的token
        max_tokens=1,      # 只需要生成第一个token
        temperature=1.0    # 用默认温度获得真实的概率分布
    )

    # 解析logprobs
    if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
        print(f"警告：没有获取到logprobs for token '{token}'")
        return {}

    # 获取第一个生成token的top logprobs
    first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs

    # 提取0-10的概率分布
    activation_probs = {}

    for logprob_item in first_token_logprobs:
        token_str = logprob_item.token.strip()
        logprob_value = logprob_item.logprob

        # 检查是否是0-10的数字
        if token_str.isdigit():
            activation_value = int(token_str)
            if 0 <= activation_value <= 10:
                # 从对数概率转换为概率: P = e^(log P)
                probability = math.exp(logprob_value)
                activation_probs[activation_value] = probability

    return activation_probs


def compute_expected_activation(activation_probs: dict) -> float:
    """
    从概率分布计算期望激活值

    Args:
        activation_probs: {activation_value: probability}

    Returns:
        期望激活值 E[X] = Σ k * P(X=k)
    """

    if not activation_probs:
        return 0.0

    # 归一化概率（因为只有top-k，不是完整分布）
    total_prob = sum(activation_probs.values())
    normalized_probs = {k: v / total_prob for k, v in activation_probs.items()}

    # 计算期望值
    expected_value = sum(k * prob for k, prob in normalized_probs.items())

    return expected_value


def predict_activations_with_logprobs(
    explanation: str,
    tokens: list[str],
    show_details: bool = True
) -> list[float]:
    """
    使用logprobs预测一系列tokens的激活值

    Args:
        explanation: 神经元行为解释
        tokens: token列表
        show_details: 是否打印详细信息

    Returns:
        预测的激活值列表
    """

    predicted_activations = []
    context = ""

    for i, token in enumerate(tokens):
        # 获取logprobs概率分布
        activation_probs = get_activation_logprobs(explanation, token, context)

        # 计算期望激活值
        expected_activation = compute_expected_activation(activation_probs)
        predicted_activations.append(expected_activation)

        if show_details:
            print(f"\nToken {i+1}: '{token}'")
            print(f"  上下文: {context if context else '(开始)'}")

            # 显示概率分布
            if activation_probs:
                print(f"  LogProbs分布:")
                total_prob = sum(activation_probs.values())
                sorted_probs = sorted(activation_probs.items(), key=lambda x: x[1], reverse=True)

                for act_val, prob in sorted_probs[:5]:  # 显示top 5
                    normalized_prob = prob / total_prob
                    print(f"    激活值 {act_val}: 概率 {normalized_prob:.3f} (logprob: {math.log(prob):.2f})")

                print(f"  → 期望激活值: {expected_activation:.2f}")
            else:
                print(f"  → 期望激活值: {expected_activation:.2f} (无logprobs)")

        # 更新上下文
        context += (" " if context else "") + token

    return predicted_activations


def compute_correlation_score(predicted: list[float], actual: list[float]) -> float:
    """
    计算预测值与真实值的相关性得分

    Args:
        predicted: 预测的激活值
        actual: 真实的激活值

    Returns:
        0-1之间的得分（越高越好）
    """

    predicted_array = np.array(predicted)
    actual_array = np.array(actual)

    # 归一化真实激活值到0-10范围（匹配预测范围）
    if actual_array.max() > 0:
        actual_normalized = (actual_array / actual_array.max()) * 10.0
    else:
        actual_normalized = actual_array

    # 计算Pearson相关系数
    if len(predicted) > 1:
        correlation = np.corrcoef(predicted_array, actual_normalized)[0, 1]
    else:
        correlation = 1.0 if predicted[0] == actual_normalized[0] else 0.0

    # 转换到0-1范围
    score = (correlation + 1) / 2

    return score


def evaluate_example():
    """
    完整示例：评估"数字检测器"神经元
    """

    print("=" * 80)
    print("基于LogProbs的可解释性评估 - 完整演示")
    print("=" * 80)

    # 真实数据
    tokens = ["The", "answer", "is", "42"]
    actual_activations = [0.1, 0.2, 0.15, 8.5]

    print("\n【输入数据】")
    print(f"文本: {' '.join(tokens)}")
    print(f"真实激活值: {actual_activations}")

    # 步骤1：神经元解释（假设已从GPT-4生成）
    explanation = "这个神经元在遇到数字时激活"

    print(f"\n【步骤1：神经元解释】")
    print(f"解释: {explanation}")

    # 步骤2：使用logprobs预测激活值
    print(f"\n【步骤2：使用LogProbs预测激活值】")
    predicted_activations = predict_activations_with_logprobs(
        explanation,
        tokens,
        show_details=True
    )

    # 步骤3：计算相关性得分
    print(f"\n{'=' * 80}")
    print("【步骤3：计算相关性得分】")
    print(f"\n预测激活值: {[f'{x:.2f}' for x in predicted_activations]}")
    print(f"真实激活值: {actual_activations}")

    # 归一化真实值到0-10
    actual_max = max(actual_activations)
    actual_normalized = [(x / actual_max) * 10.0 for x in actual_activations]
    print(f"归一化真实值 (0-10): {[f'{x:.2f}' for x in actual_normalized]}")

    # 计算相关系数
    correlation = np.corrcoef(predicted_activations, actual_normalized)[0, 1]
    score = compute_correlation_score(predicted_activations, actual_activations)

    print(f"\nPearson相关系数: {correlation:.4f}")
    print(f"最终得分 (0-1): {score:.4f}")

    if score > 0.8:
        print("✓ 得分很高！解释与实际行为高度一致")
    elif score > 0.6:
        print("○ 得分中等，解释部分符合实际行为")
    else:
        print("✗ 得分较低，解释可能不准确")

    print("=" * 80)

    return {
        "explanation": explanation,
        "tokens": tokens,
        "actual_activations": actual_activations,
        "predicted_activations": predicted_activations,
        "score": score
    }


def test_good_vs_bad_explanation():
    """
    对比测试：好的解释 vs 坏的解释
    """

    print("\n\n" + "=" * 80)
    print("对比实验：好的解释 vs 坏的解释")
    print("=" * 80)

    tokens = ["There", "are", "100", "students"]
    actual_activations = [0.1, 0.2, 9.5, 0.15]

    print(f"\n测试数据: {' '.join(tokens)}")
    print(f"真实激活值: {actual_activations}")
    print("(可以看到第3个token '100' 激活很强)")

    # 测试1：好的解释
    print("\n" + "-" * 80)
    print("【测试1：好的解释】")
    good_explanation = "这个神经元在遇到数字时激活"
    print(f"解释: {good_explanation}")

    predicted_good = predict_activations_with_logprobs(
        good_explanation,
        tokens,
        show_details=False
    )
    score_good = compute_correlation_score(predicted_good, actual_activations)

    print(f"\n预测: {[f'{x:.2f}' for x in predicted_good]}")
    print(f"真实: {actual_activations}")
    print(f"得分: {score_good:.4f}")

    # 测试2：坏的解释
    print("\n" + "-" * 80)
    print("【测试2：坏的解释】")
    bad_explanation = "这个神经元在遇到动词时激活"
    print(f"解释: {bad_explanation}")

    predicted_bad = predict_activations_with_logprobs(
        bad_explanation,
        tokens,
        show_details=False
    )
    score_bad = compute_correlation_score(predicted_bad, actual_activations)

    print(f"\n预测: {[f'{x:.2f}' for x in predicted_bad]}")
    print(f"真实: {actual_activations}")
    print(f"得分: {score_bad:.4f}")

    # 对比结果
    print("\n" + "=" * 80)
    print("【对比结果】")
    print(f"好的解释得分: {score_good:.4f}")
    print(f"坏的解释得分: {score_bad:.4f}")
    print(f"差异: {abs(score_good - score_bad):.4f}")

    if score_good > score_bad:
        print("✓ 好的解释得分更高，评估方法有效！")
    else:
        print("注意：结果可能受LLM随机性影响")

    print("=" * 80)


if __name__ == "__main__":
    # 运行主要示例
    result = evaluate_example()

    # 运行对比实验
    test_good_vs_bad_explanation()

    print("\n\n【总结】")
    print("这个实现展示了论文中基于LogProbs的评估方法：")
    print("1. 对每个token，获取LLM预测激活值（0-10）的概率分布")
    print("2. 从概率分布计算期望值（而不是单点预测）")
    print("3. 计算预测值与真实值的Pearson相关系数")
    print("4. 高相关性 → 解释准确；低相关性 → 解释不准确")
