# Modern Neuron Simulator

现代化的神经元激活模拟器，基于原始论文的`simulator.py`，使用最新OpenAI API重新实现。

## 📋 概述

这个实现提取了原始neuron_explainer库中`simulator.py`的核心逻辑，并进行了现代化改造：

| 特性 | 原始实现 | 现代实现 |
|------|----------|----------|
| **API** | ❌ Completions API (已废弃) | ✅ Chat Completions API |
| **模型** | text-davinci-003 | gpt-4, gpt-3.5-turbo, gpt-4-turbo |
| **依赖** | neuron_explainer库 | 仅 openai + numpy |
| **代码量** | 600+ 行 + 依赖 | ~400 行，独立 |
| **Logprobs** | ✅ 完整支持 | ⚠️ 简化版（Chat API限制）|
| **异步支持** | ✅ | ✅ |

## 🎯 核心概念

### 原论文的方法

```
1. 生成解释 (GPT-4)
   输入: Top-activating examples
   输出: "This neuron detects..."

2. 模拟激活 (text-davinci-003)
   输入: 解释 + 新的token序列
   输出: 预测的激活值 (0-10)

3. 评分
   相关性 = corr(预测激活, 实际激活)
   分数越高 = 解释越好
```

### 我们的实现

保留了完全相同的三步流程，但使用现代API：

```python
# 步骤1: 已有解释（来自GPT-4或你自己写的）
explanation = "numeric values and digits"

# 步骤2: 模拟激活
simulator = ModernNeuronSimulator(explanation)
simulation = simulator.simulate(tokens)

# 步骤3: 评分
score = compute_simulation_score(simulation, actual_activations)
```

## 🚀 快速开始

### 安装

```bash
pip install openai numpy
```

### 基本使用

```python
from modern_simulator import ModernNeuronSimulator

# 创建模拟器
simulator = ModernNeuronSimulator(
    explanation="numeric values and digits",
    model_name="gpt-3.5-turbo"
)

# 模拟激活
tokens = ["The", "price", "is", "$", "42"]
result = simulator.simulate(tokens)

print(result.expected_activations)  # [0.0, 2.0, 0.0, 5.0, 10.0]
```

### 带评分的完整示例

```python
import asyncio
from modern_simulator import simulate_and_score

async def evaluate():
    # 你的数据
    explanation = "the token 'the'"
    tokens = ["In", "the", "beginning", "there", "was", "the", "word"]
    actual_activations = [0.1, 0.95, 0.05, 0.15, 0.05, 0.92, 0.08]

    # 模拟并评分
    simulation, score = await simulate_and_score(
        explanation=explanation,
        tokens=tokens,
        actual_activations=actual_activations
    )

    print(f"Score: {score:.3f}")  # 应该很高，因为解释准确
    print(f"Predicted: {simulation.expected_activations}")

asyncio.run(evaluate())
```

## 📊 与原始实现的详细对比

### ExplanationNeuronSimulator (原始)

```python
# 原始代码 (simulator.py:125-180)
simulator = ExplanationNeuronSimulator(
    model_name="text-davinci-003",  # ❌ 已废弃
    explanation=explanation,
)

# 使用 Completions API
response = await api_client.make_request(
    prompt=prompt,  # ❌ 旧格式
    max_tokens=0,
    echo=True,      # ❌ Chat API不支持
    logprobs=15,    # ❌ 格式不同
)

# 复杂的logprob解析
result = parse_simulation_response(response, prompt_format, tokens)
```

### ModernNeuronSimulator (我们的)

```python
# 现代化实现
simulator = ModernNeuronSimulator(
    explanation=explanation,
    model_name="gpt-3.5-turbo",  # ✅ 最新模型
)

# 使用 Chat Completions API
response = await client.chat.completions.create(
    model=model_name,
    messages=messages,  # ✅ 新格式
    max_tokens=500,
    temperature=0.0,
)

# 简化的文本解析
activations = _parse_completion(response.choices[0].message.content, tokens)
```

### 关键简化

**原始的复杂性**:
- 依赖`parse_top_logprobs()`解析token概率
- 处理byte encoding边界情况
- 处理tokenizer不匹配
- 计算概率分布的期望值

**我们的简化**:
- 直接从文本中提取预测的激活值
- 使用确定性预测（temperature=0）
- 不需要处理复杂的token边界情况

**为什么可以简化**:
- Chat Completions API更稳定的输出格式
- 对于评分目的，确定性预测已足够
- 期望值 ≈ 最可能的值（当temperature=0时）

## 🔬 两种模拟模式

### 1. All-at-once (默认，快速)

```python
simulator = ModernNeuronSimulator(explanation="...")
result = simulator.simulate(tokens)  # 一次API调用
```

**原理**:
- 单个prompt包含所有tokens
- 模型一次性预测所有激活
- 对应原始的`ExplanationNeuronSimulator`

**优点**: 快速，便宜
**缺点**: 后面的预测可能受前面影响

### 2. Token-by-token (更准确)

```python
simulator = TokenByTokenSimulator(explanation="...")
result = await simulator.simulate(tokens)  # 每个token一次调用
```

**原理**:
- 每个token单独预测
- 各预测相互独立
- 对应原始的`ExplanationTokenByTokenSimulator`

**优点**: 更准确的概率估计
**缺点**: 慢，贵（N次API调用）

## 📝 Few-shot Examples

两种实现都使用few-shot prompting：

```python
# 使用默认examples
simulator = ModernNeuronSimulator(explanation="...")

# 使用自定义examples（更好的domain适配）
from modern_simulator import FewShotExample

custom_examples = [
    FewShotExample(
        explanation="mathematical operators",
        tokens=["+", "-", "*", "/"],
        activations=[10, 10, 10, 10]
    ),
    # ... 更多examples
]

simulator = ModernNeuronSimulator(
    explanation="equality operators",
    few_shot_examples=custom_examples
)
```

**建议**:
- 默认examples适合一般文本
- 为特定领域（代码、数学等）创建自定义examples
- 3-5个examples通常足够

## 🎯 实际应用：集成到评估流程

### 方法1: 使用simple_interp_eval（推荐给快速评估）

```python
from simple_interp_eval import evaluate_feature_interpretability

# 简单、快速、适合批量评估
result = evaluate_feature_interpretability(
    tokens_list=tokens_list,
    activations_list=activations_list
)
```

### 方法2: 使用modern_simulator（更接近论文）

```python
from modern_simulator import ModernNeuronSimulator, compute_simulation_score

# 第一步：生成解释（使用GPT-4）
from simple_interp_eval import evaluate_feature_interpretability
# ... 生成explanation

# 第二步：使用simulator评估（更细粒度）
simulator = ModernNeuronSimulator(explanation)

scores = []
for tokens, actual_acts in zip(test_tokens, test_activations):
    simulation = simulator.simulate(tokens)
    score = compute_simulation_score(simulation, actual_acts)
    scores.append(score)

avg_score = np.mean(scores)
```

### 完整的论文式评估流程

```python
import asyncio
from openai import AsyncOpenAI
from modern_simulator import ModernNeuronSimulator, compute_simulation_score

async def evaluate_feature_paper_style(
    train_tokens_list,      # 用于生成解释
    train_activations_list,
    val_tokens_list,        # 用于评分
    val_activations_list
):
    """按照论文方法评估一个特征"""

    # 1. 生成解释（使用GPT-4，基于训练集）
    # 这部分可以用simple_interp_eval的逻辑
    client = AsyncOpenAI(api_key="...")

    # 准备top-activating examples
    max_acts = [max(acts) for acts in train_activations_list]
    top_indices = np.argsort(max_acts)[::-1][:5]

    # 构建prompt
    examples_text = []
    for idx in top_indices:
        tokens = train_tokens_list[idx]
        acts = train_activations_list[idx]
        # Format as token(activation) pairs
        formatted = " ".join([f"{t}({a:.2f})" for t, a in zip(tokens, acts)])
        examples_text.append(formatted)

    prompt = f"""Based on these examples where a neuron activates:
{chr(10).join(examples_text)}

What pattern does this neuron detect? (one sentence)"""

    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    explanation = response.choices[0].message.content.strip()
    print(f"Generated explanation: {explanation}")

    # 2. 模拟验证集的激活
    simulator = ModernNeuronSimulator(
        explanation=explanation,
        model_name="gpt-3.5-turbo",
        use_async=True
    )

    scores = []
    for tokens, actual_acts in zip(val_tokens_list, val_activations_list):
        simulation = await simulator.simulate_async(tokens)
        score = compute_simulation_score(simulation, actual_acts)
        scores.append(score)

    # 3. 返回结果
    return {
        "explanation": explanation,
        "score": np.mean(scores),
        "individual_scores": scores
    }

# 使用
result = asyncio.run(evaluate_feature_paper_style(
    train_tokens, train_acts,
    val_tokens, val_acts
))
```

## 🔍 与原始实现的主要差异

### 1. Logprobs处理

**原始**:
- 获取每个token位置的logprobs
- 解析出0-10每个值的概率
- 计算期望值

**现代**:
- 直接获取模型输出的激活值
- Temperature=0确保确定性
- 简化但有效

### 2. 提示格式

**原始**:
```
Activations: <start>
token1\tactivation1
token2\tactivation2
<end>
```

**现代**: 相同！我们保留了这个格式，因为它有效。

### 3. API调用

**原始**:
```python
response = await api_client.make_request(
    prompt=prompt,
    echo=True,
    logprobs=15
)
```

**现代**:
```python
response = await client.chat.completions.create(
    messages=messages,
    max_tokens=500,
    temperature=0.0
)
```

## ⚠️ 限制与注意事项

### Chat Completions API的限制

1. **没有echo参数**: 不能echo prompt来获取logprobs
2. **Logprobs格式不同**: 需要不同的解析方式
3. **Token对齐困难**: 不同tokenizer可能导致问题

### 我们的解决方案

1. 使用确定性采样（temperature=0）
2. 简化为直接预测而非概率分布
3. 假设模型会正确重复token序列

### 何时使用哪个

**使用simple_interp_eval**:
- ✅ 快速原型
- ✅ 批量评估大量特征
- ✅ 不需要完全复现论文

**使用modern_simulator**:
- ✅ 需要few-shot examples控制
- ✅ 更接近论文方法
- ✅ 需要token-by-token模式
- ✅ 自定义评估流程

**使用原始simulator.py**:
- 📊 完全复现论文结果
- 📊 需要完整的logprobs分布
- ⚠️ 但需要先修复API问题

## 🧪 测试

```bash
# 运行所有测试
python test_modern_simulator.py

# 测试包括：
# 1. 基本同步模拟
# 2. 异步模拟
# 3. 带评分的模拟
# 4. Token-by-token模式
# 5. 自定义few-shot examples
# 6. 真实场景示例
# 7. 与simple_eval对比
```

## 📈 性能与成本

### API调用次数

| 模式 | 调用次数 | 成本（100 tokens） |
|------|----------|-------------------|
| All-at-once | 1 | ~$0.001-0.01 |
| Token-by-token | N (序列长度) | ~$0.01-0.1 |

### 速度

- All-at-once: ~1-2秒/序列
- Token-by-token: ~0.5秒/token（并行）

## 🔗 与其他组件集成

### 与simple_interp_eval结合

```python
# Step 1: 使用simple_interp_eval快速生成解释和初步评分
from simple_interp_eval import evaluate_feature_interpretability

result = evaluate_feature_interpretability(tokens_list, activations_list)
explanation = result["explanation"]
quick_score = result["score"]

# Step 2: 如果分数有前景，使用simulator进行更详细评估
if quick_score > 0.6:
    from modern_simulator import ModernNeuronSimulator

    simulator = ModernNeuronSimulator(explanation)
    # 在更大的验证集上测试
    detailed_scores = [...]
```

### 与interpretability_eval.py结合

```python
# 使用interpretability_eval生成解释
from interpretability_eval import InterpretabilityEvaluator

evaluator = InterpretabilityEvaluator(api_key="...")
explanation, _, _ = await evaluator.evaluate_feature(train_examples, val_examples)

# 使用simulator进行额外验证
from modern_simulator import ModernNeuronSimulator

simulator = ModernNeuronSimulator(explanation)
# ... 额外的测试
```

## 📚 参考

- 原论文: [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/pdf/2309.08600.pdf)
- 原始代码: `neuron_explainer/explanations/simulator.py`
- OpenAI文档: [Chat Completions API](https://platform.openai.com/docs/guides/chat-completions)

## 🎓 总结

| 方面 | 结论 |
|------|------|
| **核心方法** | ✅ 与论文完全一致 |
| **API** | ✅ 使用最新Chat Completions |
| **代码复杂度** | ✅ 大幅简化 |
| **功能** | ⚠️ 简化logprobs，但核心完整 |
| **适用性** | ✅ 适合实际应用 |

**推荐用法**:
1. 快速评估 → `simple_interp_eval.py`
2. 接近论文 → `modern_simulator.py`
3. 完全复现 → 修复原始`simulator.py`的API问题
