# 原论文基于LogProbs的可解释性评估算法详解

## 核心思想

原论文使用了一个非常巧妙的评估方法：**让LLM基于解释（explanation）来预测激活值，然后计算预测值与真实激活值的相关性**。

关键创新点是使用 **logprobs（对数概率）** 来获得LLM对激活值的概率分布，而不是直接让LLM输出一个数字。

---

## 完整算法流程

### 阶段1：数据收集 (Activation Collection)

```python
# 输入：50,000个文本片段
# 输出：每个token的真实激活值

for text_fragment in dataset:
    tokens = tokenize(text_fragment)

    # 通过模型获取中间层表示
    hidden_states = model.forward(tokens, output_hidden_states=True)
    layer_activations = hidden_states[layer_idx]  # 例如第6层

    # 通过稀疏自编码器
    sae_features = sparse_autoencoder(layer_activations)

    # 对于特定的特征k
    feature_k_activations = sae_features[:, k]  # 每个token的激活值

    # 保存：ActivationRecord(tokens, feature_k_activations)
```

**结果**：得到50,000条记录，每条包含 `(tokens, activations)`

---

### 阶段2：生成解释 (Explanation Generation)

```python
# 输入：top激活的样本（例如前5个最强激活的例子）
# 输出：自然语言解释

# 选择top例子
top_records = sorted(records, key=lambda r: max(r.activations), reverse=True)[:5]

# 构造prompt给GPT-4
prompt = """
以下是一个神经元在不同文本中的激活模式，激活值越高说明神经元越活跃：

例子1：
Text: "The price is $42"
Activations: [0.1, 0.2, 0.1, 0.3, 8.5]
激活最强的token: "$42" (8.5)

例子2：
Text: "There are 100 students"
Activations: [0.1, 0.1, 9.2, 0.2]
激活最强的token: "100" (9.2)

...更多例子...

请给出这个神经元激活的模式解释，用一句话描述它什么时候会激活。
"""

explanation = gpt4(prompt)
# 例如返回："这个神经元在遇到数字时激活"
```

**关键**：这一步用的是GPT-4（理解能力强），但不需要logprobs

---

### 阶段3：模拟预测 (Simulation with LogProbs) ⭐核心⭐

这是最关键的部分！

#### 3.1 构造预测prompt

```python
# 输入：
# - explanation: "这个神经元在遇到数字时激活"
# - test_tokens: ["The", "answer", "is", "42"]
# 输出：预测每个token的激活值（0-10）

# 构造few-shot prompt
prompt = f"""
你是一个神经元模拟器。神经元的行为：{explanation}

你的任务是预测每个token的激活值（0-10，0表示不激活，10表示最强激活）。

格式要求：
<start>token\tactivation<end>

现在预测以下tokens的激活值：
<start>The\t
"""

# 注意：prompt在 <start>The\t 处截断，让模型补全激活值
```

#### 3.2 调用API获取logprobs

```python
# 使用OpenAI Completions API (text-davinci-003)
response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=1,
    logprobs=100,  # 返回top 100个最可能的token及其概率
    echo=True,     # 返回完整prompt的logprobs
    temperature=1.0
)

# API返回的logprobs格式：
{
    "choices": [{
        "logprobs": {
            "tokens": ["The", "\t", "0", "\n", ...],
            "token_logprobs": [-0.5, -0.1, -2.3, ...],
            "top_logprobs": [
                {
                    "0": -2.3,    # log(P(token="0"))
                    "1": -3.1,    # log(P(token="1"))
                    "2": -4.5,
                    "3": -5.2,
                    ...
                    "10": -6.8
                },
                ...
            ]
        }
    }]
}
```

#### 3.3 从logprobs计算期望激活值 ⭐关键创新⭐

```python
# 对于每个token位置，LLM会输出0-10之间的数字
# 我们获得了每个数字的对数概率 log(P(activation=k))

def compute_expected_activation(top_logprobs_dict):
    """
    从logprobs计算期望值

    top_logprobs_dict = {
        "0": -2.3,   # log P(activation=0)
        "1": -3.1,   # log P(activation=1)
        "2": -4.0,
        ...
        "10": -5.5
    }
    """

    # 步骤1：从对数概率转换为概率
    probs = {}
    for token, logprob in top_logprobs_dict.items():
        if token.isdigit() and 0 <= int(token) <= 10:
            prob = math.exp(logprob)  # P = e^(log P)
            probs[int(token)] = prob

    # 步骤2：归一化（因为只有top-100，不是全部vocab）
    total_prob = sum(probs.values())
    normalized_probs = {k: v/total_prob for k, v in probs.items()}

    # 步骤3：计算期望值 E[X] = Σ k * P(X=k)
    expected_activation = sum(k * prob for k, prob in normalized_probs.items())

    return expected_activation

# 例如：
# P(0) = 0.05
# P(1) = 0.10
# P(2) = 0.15
# P(3) = 0.20
# ...
# P(8) = 0.25
# P(9) = 0.08
# P(10) = 0.02
# Expected = 0*0.05 + 1*0.10 + 2*0.15 + ... + 10*0.02 = 5.3
```

**为什么这样做？**

1. **更准确**：不是让LLM直接输出一个数字（容易不稳定），而是获得完整的概率分布
2. **不确定性建模**：如果LLM不确定，logprobs会反映这种不确定性
   - 确定的情况：P(8)=0.95, P(7)=0.03, P(9)=0.02 → Expected ≈ 8.0
   - 不确定的情况：P(3)=0.25, P(4)=0.25, P(5)=0.25, P(6)=0.25 → Expected ≈ 4.5
3. **平滑预测**：期望值可以是小数（如5.3），更精细

#### 3.4 对所有token重复

```python
predicted_activations = []

for token in test_tokens:
    # 构造prompt，让LLM预测这个token的激活值
    prompt = construct_prompt(explanation, token, context)

    # 获取logprobs
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        logprobs=100
    )

    # 从logprobs计算期望激活值
    expected_activation = compute_expected_activation(
        response["choices"][0]["logprobs"]["top_logprobs"][position]
    )

    predicted_activations.append(expected_activation)

# 得到：[0.3, 0.5, 0.4, 7.8] 对应 ["The", "answer", "is", "42"]
```

---

### 阶段4：计算相关性得分 (Scoring)

```python
# 输入：
# - predicted_activations: [0.3, 0.5, 0.4, 7.8] (从logprobs得到)
# - actual_activations: [0.1, 0.2, 0.15, 8.5] (真实SAE激活)

def compute_correlation_score(predicted, actual):
    """计算Pearson相关系数"""

    # 归一化真实激活值到0-10范围（匹配预测范围）
    actual_max = max(actual)
    actual_normalized = [(a / actual_max) * 10.0 for a in actual]

    # 计算Pearson相关系数
    # r = Cov(X,Y) / (σ_X * σ_Y)
    predicted_array = np.array(predicted)
    actual_array = np.array(actual_normalized)

    correlation = np.corrcoef(predicted_array, actual_array)[0, 1]

    # 转换到0-1范围 (原始相关系数是-1到1)
    score = (correlation + 1) / 2

    return score

# 例如：相关系数 = 0.85 → 得分 = 0.925
```

**为什么用相关系数而不是MSE？**
- 相关系数关注**模式匹配**，不关心绝对值
- 只要预测的高低趋势正确就行，不需要精确数值
- 例如：真实[1, 2, 9]，预测[0.5, 1.0, 8.5] → 相关性很高

---

## 完整示例演示

假设我们要评估一个"数字检测器"特征：

### 步骤1：收集数据
```
Record 1: "The price is $50"
Tokens:     ["The", "price", "is", "$", "50"]
Activations: [0.1,   0.2,   0.1,  0.5,  9.5]

Record 2: "I have 3 cats"
Tokens:     ["I", "have", "3", "cats"]
Activations: [0.1, 0.2,   8.8, 0.1]
```

### 步骤2：生成解释（用GPT-4）
```
解释："这个神经元在遇到数字字符时激活"
```

### 步骤3：预测激活值（用text-davinci-003 + logprobs）

测试Record 1的第5个token "50"：

```python
# Prompt:
"""
神经元行为：这个神经元在遇到数字字符时激活

预测激活值格式：
<start>50\t
"""

# API返回logprobs:
{
    "0": -6.2,   # P(activation=0) = e^(-6.2) = 0.002
    "1": -5.8,   # P(activation=1) = 0.003
    ...
    "8": -0.5,   # P(activation=8) = 0.606  ← 最可能
    "9": -1.2,   # P(activation=9) = 0.301
    "10": -2.8   # P(activation=10) = 0.061
}

# 计算期望值：
E[activation] = 0*0.002 + 1*0.003 + ... + 8*0.606 + 9*0.301 + 10*0.061
              ≈ 8.4
```

对所有tokens预测：
```
预测激活：[0.3, 0.4, 0.2, 0.6, 8.4]
真实激活：[0.1, 0.2, 0.1, 0.5, 9.5]
```

### 步骤4：计算得分
```python
# 归一化真实激活到0-10
actual_normalized = [0.1, 0.2, 0.1, 0.5, 9.5] / 9.5 * 10 = [0.11, 0.21, 0.11, 0.53, 10.0]

# Pearson相关系数
r = pearson_correlation([0.3, 0.4, 0.2, 0.6, 8.4], [0.11, 0.21, 0.11, 0.53, 10.0])
  ≈ 0.95

# 得分
score = (0.95 + 1) / 2 = 0.975  ← 很高！说明解释很好
```

---

## 关键技术细节

### 1. 为什么需要logprobs？

**没有logprobs的问题：**
```python
# 直接让LLM输出数字
response = gpt("预测激活值：50\t")
# 返回："8" 或 "9"
# 问题：不稳定，无法知道LLM的置信度
```

**使用logprobs的优势：**
```python
# 获得概率分布
logprobs = {
    "7": -2.1,  # 12%
    "8": -0.5,  # 61%  ← 最可能
    "9": -1.2,  # 30%
    "10": -3.5  # 3%
}
# 期望值 = 7*0.12 + 8*0.61 + 9*0.30 + 10*0.03 = 8.2
# 更稳定，更精确，反映不确定性
```

### 2. 为什么用Completions API而不是Chat API？

**Completions API (`text-davinci-003`)**:
```python
openai.Completion.create(
    prompt="<start>token\t",
    logprobs=100,  # ✓ 支持logprobs
    echo=True      # ✓ 支持echo
)
```

**Chat API (`gpt-3.5-turbo`)**:
```python
openai.ChatCompletion.create(
    messages=[{"role": "user", "content": "..."}],
    logprobs=True  # ✓ 新版支持，但格式不同
    # ✗ 不支持echo
)
```

**现代替代方案**：
- 使用 `gpt-3.5-turbo-instruct`（仍支持Completions API）
- 或者用Chat API的logprobs（需要适配格式）
- 或者用 `temperature=0` 直接采样（简化版，不用logprobs）

### 3. Few-shot examples的作用

```python
prompt = f"""
神经元行为：{explanation}

示例格式：
<start>example\t5<end>
<start>test\t2<end>

现在预测：
<start>{token}\t
"""
```

Few-shot examples帮助LLM：
1. 理解输出格式（\t分隔）
2. 理解激活值范围（0-10）
3. 提供上下文参考

---

## 论文方法 vs 简化方法对比

| 方面 | 原论文方法 | 简化方法 |
|------|-----------|---------|
| **预测模型** | text-davinci-003 | gpt-3.5-turbo |
| **API类型** | Completions API | Chat API |
| **使用logprobs** | ✓ 是 | ✗ 否 |
| **预测方式** | 期望值（概率分布） | 直接采样（temperature=0） |
| **优点** | 更准确，反映不确定性 | 更简单，API仍可用 |
| **缺点** | API已弃用 | 略微不够准确 |

---

## 代码中的体现

### 原论文的核心代码（neuron_explainer库）：

```python
# simulator.py
class ExplanationNeuronSimulator:
    def simulate(self, tokens: List[str]) -> SequenceSimulation:
        # 构造prompt
        prompt = self._make_simulation_prompt(tokens)

        # 调用Completions API获取logprobs
        response = await self.client.completions.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=0,
            logprobs=100,  # ← 关键！
            echo=True
        )

        # 解析logprobs，计算期望激活值
        expected_activations = self._calculate_expected_activations(
            response.choices[0].logprobs
        )

        return SequenceSimulation(
            tokens=tokens,
            expected_activations=expected_activations
        )

    def _calculate_expected_activations(self, logprobs):
        """从logprobs计算期望值"""
        expected = []
        for top_logprobs in logprobs.top_logprobs:
            # 获取0-10的概率分布
            probs = {}
            for token, logprob in top_logprobs.items():
                if token.isdigit():
                    probs[int(token)] = math.exp(logprob)

            # 归一化
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}

            # 期望值
            expected_val = sum(k * p for k, p in probs.items())
            expected.append(expected_val)

        return expected
```

### 我们的简化版本（test_scoring_logic.py）：

```python
# 不使用logprobs，直接采样
async def simulate_activations(explanation, tokens):
    simulator = ExplanationNeuronSimulator(
        model_name="gpt-3.5-turbo",  # Chat API
        explanation=explanation,
        prompt_format=PromptFormat.HARMONY_V4  # Chat格式
    )

    # 内部会用temperature=0直接采样，不用logprobs
    simulation = await simulator.simulate(tokens)
    return simulation.expected_activations
```

---

## 总结

原论文的评估方法核心是：

1. **输入**：一个解释（explanation）
2. **过程**：用LLM基于解释预测激活值，通过logprobs获得概率分布并计算期望值
3. **输出**：预测值与真实值的相关性得分（0-1）

**关键创新**：使用logprobs来获得LLM对激活值的**概率分布**，而不是单一预测，这样更稳定、更准确。

**实践挑战**：Completions API已弃用，需要用替代方案（gpt-3.5-turbo-instruct或简化的Chat API方法）。
