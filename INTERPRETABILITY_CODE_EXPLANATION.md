# 原始论文可解释性评估代码详解

## 📋 概览

论文使用 `interpret.py` 实现自动可解释性评估，核心是三个步骤：
1. **收集激活数据** - 在大规模文本上运行并记录特征激活
2. **生成解释** - 使用GPT-4基于top激活examples生成自然语言解释
3. **模拟和评分** - 使用text-davinci-003模拟激活并计算分数

---

## 🔍 Step 1: 收集激活数据

### 代码位置：`interpret.py:82-212` - `make_feature_activation_dataset()`

```python
def make_feature_activation_dataset(
    model: HookedTransformer,
    learned_dict: LearnedDict,
    layer: int,
    layer_loc: str,
    device: str = "cpu",
    n_fragments=OPENAI_MAX_FRAGMENTS,  # 50000
    max_features: int = 0,
):
    """
    在大规模文本上运行模型，收集每个特征的激活值
    """
```

### 详细流程：

#### 1.1 数据准备
```python
# Line 108: 加载OpenWebText数据集
sentence_dataset = load_dataset("openwebtext", split="train", streaming=True)
iter_dataset = iter(sentence_dataset)

# Line 130-131: 创建存储表
activation_maxes_table = np.zeros((n_fragments, feat_dim), dtype=np.float16)
activation_data_table = np.zeros((n_fragments, feat_dim * OPENAI_FRAGMENT_LEN), dtype=np.float16)
```

**作用**：
- `activation_maxes_table`: 存储每个fragment中每个特征的**最大激活值**
- `activation_data_table`: 存储每个fragment中每个特征的**逐token激活值**

#### 1.2 批处理收集
```python
# Line 133-157: 主循环
while n_added < n_fragments:
    # 获取文本
    sentence = next(iter_dataset)
    sentence_tokens = tokenizer_model.to_tokens(sentence["text"], prepend_bos=False)

    # 随机选择64-token fragment
    token_start = np.random.randint(0, n_tokens - OPENAI_FRAGMENT_LEN)
    fragment_tokens = sentence_tokens[:, token_start : token_start + OPENAI_FRAGMENT_LEN]
```

**关键参数**：
- `OPENAI_FRAGMENT_LEN = 64`: 每个fragment固定64个tokens
- `OPENAI_MAX_FRAGMENTS = 50000`: 总共收集50000个fragments

#### 1.3 获取激活
```python
# Line 163-184: 运行模型获取激活
if use_baukit:
    with Trace(model, tensor_name) as ret:
        _ = model(tokens)
        mlp_activation_data = ret.output.to(device)
        mlp_activation_data = nn.functional.gelu(mlp_activation_data)
else:
    _, cache = model.run_with_cache(tokens)
    mlp_activation_data = cache[tensor_name].to(device)

# 通过稀疏自编码器编码
feature_activation_data = learned_dict.encode(activation_data)  # [seq_len, n_features]
feature_activation_maxes = torch.max(feature_activation_data, dim=0)[0]  # [n_features]
```

**流程图**：
```
文本 → Tokenize → Language Model → MLP激活 → Sparse Autoencoder → 特征激活
                                    [B, L, D]                      [B, L, F]
```

#### 1.4 存储数据
```python
# Line 180-189: 存储到表中
activation_maxes_table[n_added, :] = feature_activation_maxes.cpu().numpy()[:feat_dim]
activation_data_table[n_added, :] = feature_activation_data.flatten()
fragment_token_ids_list.append(token_ids)
fragment_token_strs_list.append(fragment_strs[i])
```

#### 1.5 构建DataFrame
```python
# Line 196-210: 构建pandas DataFrame
df = pd.DataFrame()
df["fragment_token_ids"] = fragment_token_ids_list
df["fragment_token_strs"] = fragment_token_strs_list

# 添加最大激活列
maxes_column_names = [f"feature_{i}_max" for i in range(feat_dim)]
df = pd.concat([df, pd.DataFrame(activation_maxes_table, columns=maxes_column_names)], axis=1)

# 添加逐token激活列
activations_column_names = [
    f"feature_{i}_activation_{j}"
    for j in range(OPENAI_FRAGMENT_LEN)
    for i in range(feat_dim)
]
df = pd.concat([df, pd.DataFrame(activation_data_table, columns=activations_column_names)], axis=1)
```

**DataFrame结构**：
| fragment_token_strs | feature_0_max | feature_0_activation_0 | ... | feature_0_activation_63 |
|---------------------|---------------|------------------------|-----|-------------------------|
| ["The", "cat", ...] | 0.95 | 0.1 | ... | 0.05 |
| ["I", "like", ...] | 0.12 | 0.05 | ... | 0.02 |

---

## 🤖 Step 2: 生成解释

### 代码位置：`interpret.py:265-346` - `interpret()` 函数

```python
async def interpret(base_df: pd.DataFrame, save_folder: str, n_feats_to_explain: int):
    for feat_n in range(0, n_feats_to_explain):
        # ... 处理每个特征
```

### 详细流程：

#### 2.1 选择Top激活Examples
```python
# Line 281-291: 按最大激活排序，选择top examples
df = base_df[read_fields].copy()
sorted_df = df.sort_values(by=f"feature_{feat_n}_max", ascending=False)
sorted_df = sorted_df.head(TOTAL_EXAMPLES)  # TOTAL_EXAMPLES = 20

top_activation_records = []
for i, row in sorted_df.iterrows():
    top_activation_records.append(
        ActivationRecord(
            row["fragment_token_strs"],  # tokens
            [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)],  # activations
        )
    )
```

**ActivationRecord**：
```python
@dataclass
class ActivationRecord:
    tokens: List[str]           # ["The", "cat", "sat", ...]
    activations: List[float]    # [0.1, 0.9, 0.2, ...]
```

#### 2.2 选择Random Examples
```python
# Line 293-316: 选择有一定激活的随机examples
random_activation_records = []
random_ordering = torch.randperm(len(df)).tolist()

while len(random_activation_records) < TOTAL_EXAMPLES:
    i = random_ordering.pop()
    # 跳过完全不激活的例子
    if df.iloc[i][f"feature_{feat_n}_max"] == 0:
        continue
    random_activation_records.append(
        ActivationRecord(...)
    )
```

**为什么需要random examples？**
- 提供对比：特征在"不太激活"时的表现
- 避免过拟合：确保解释不只是描述top examples的共性

#### 2.3 构建NeuronRecord
```python
# Line 323-332: 组装数据
neuron_record = NeuronRecord(
    neuron_id=NeuronId(layer_index=2, neuron_index=feat_n),
    random_sample=random_activation_records,           # 20个random
    most_positive_activation_records=top_activation_records,  # 20个top
)

# 分割训练/验证集
slice_params = ActivationRecordSliceParams(n_examples_per_split=OPENAI_EXAMPLES_PER_SPLIT)  # 5
train_activation_records = neuron_record.train_activation_records(slice_params)  # 前10个 (5 top + 5 random)
valid_activation_records = neuron_record.valid_activation_records(slice_params)  # 后10个 (5 top + 5 random)
```

**数据分割**：
```
Top 20 examples:     [0, 1, 2, 3, 4] [5, 6, 7, 8, 9] [10-19]
                      ↑ Train (5)     ↑ Valid (5)    ↑ 未使用

Random 20 examples:  [0, 1, 2, 3, 4] [5, 6, 7, 8, 9] [10-19]
                      ↑ Train (5)     ↑ Valid (5)    ↑ 未使用

Total train: 10 (5 top + 5 random)
Total valid: 10 (5 top + 5 random)
```

#### 2.4 调用GPT-4生成解释
```python
# Line 334-346: 生成解释
explainer = TokenActivationPairExplainer(
    model_name=EXPLAINER_MODEL_NAME,  # "gpt-4"
    prompt_format=PromptFormat.HARMONY_V4,
    max_concurrent=MAX_CONCURRENT,
)

explanations = await explainer.generate_explanations(
    all_activation_records=train_activation_records,  # 10个训练examples
    max_activation=calculate_max_activation(train_activation_records),
    num_samples=1,
)

explanation = explanations[0]
print(f"Feature {feat_n}, {explanation=}")
```

**TokenActivationPairExplainer内部流程**（来自neuron_explainer库）：
1. 构建few-shot prompt，包含：
   - 系统指令："分析神经网络特征"
   - Few-shot examples：已知特征的解释示例
   - 当前特征的10个activation records
2. 调用GPT-4 API
3. 返回自然语言解释

**示例Prompt结构**：
```
System: We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.

[Few-shot Example 1]
Neuron 1 explanation: the token "the"
Activations:
In(0.1)    the(10.0)    beginning(0.0)    ...

[Few-shot Example 2]
Neuron 2 explanation: negative sentiment
Activations:
I(0.0)    hate(9.0)    this(2.0)    ...

[Current Neuron]
Neuron 3
Explain what this neuron activates for based on these examples:

[10个training examples的token-activation pairs]
```

---

## 📊 Step 3: 模拟和评分

### 代码位置：`interpret.py:348-369`

```python
# Line 348-357: 创建Simulator
format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING

simulator = UncalibratedNeuronSimulator(
    ExplanationNeuronSimulator(
        SIMULATOR_MODEL_NAME,  # "text-davinci-003"
        explanation,
        max_concurrent=MAX_CONCURRENT,
        prompt_format=format,
    )
)

# Line 358: 模拟并评分
scored_simulation = await simulate_and_score(simulator, valid_activation_records)
```

### 详细流程：

#### 3.1 ExplanationNeuronSimulator工作原理

**来自 `simulator.py:125-180`**:

```python
class ExplanationNeuronSimulator(NeuronSimulator):
    async def simulate(self, tokens: Sequence[str]) -> SequenceSimulation:
        # 1. 构建few-shot prompt
        prompt = self.make_simulation_prompt(tokens)

        # 2. 调用API (关键！)
        response = await self.api_client.make_request(
            prompt=prompt,
            max_tokens=0,    # 不生成新token
            echo=True,       # 回显prompt
            logprobs=15,     # 返回top-15 token的logprobs
        )

        # 3. 解析logprobs获取激活预测
        result = parse_simulation_response(response, tokens)
        return result
```

**核心技巧：使用Logprobs预测**

Prompt格式：
```
Explanation: this neuron activates for numeric values

Activations: <start>
The    unknown
price  unknown
is     unknown
$      unknown
42     unknown
<end>
```

API返回时，对于每个`unknown`位置，返回top-15 tokens的logprobs：
```json
{
  "0": -2.5,   // P(0) = exp(-2.5) = 0.082
  "1": -3.1,   // P(1) = exp(-3.1) = 0.045
  "2": -4.2,
  ...
  "10": -1.8,  // P(10) = exp(-1.8) = 0.165
}
```

计算期望激活：
```python
expected_activation = Σ(value * probability)
                    = 0 * 0.082 + 1 * 0.045 + ... + 10 * 0.165
```

#### 3.2 simulate_and_score实现

**来自 `neuron_explainer/explanations/scoring.py`**:

```python
async def simulate_and_score(
    simulator: NeuronSimulator,
    valid_activation_records: List[ActivationRecord]
) -> ScoredSimulation:
    """
    对验证集的每个record:
    1. 模拟激活
    2. 计算预测 vs 实际的相关性
    """
    scored_sequences = []

    for record in valid_activation_records:
        # 模拟
        simulation = await simulator.simulate(record.tokens)

        # 计算分数
        correlation = compute_correlation(
            simulation.expected_activations,  # 预测
            record.activations                # 实际
        )

        scored_sequences.append(ScoredSequenceSimulation(
            simulation=simulation,
            score=correlation
        ))

    return aggregate_scored_sequence_simulations(scored_sequences)
```

#### 3.3 三种分数

```python
# Line 359-366: 计算三种分数
score = scored_simulation.get_preferred_score()  # 总分（10个样本）

top_only_score = aggregate_scored_sequence_simulations(
    scored_simulation.scored_sequence_simulations[:5]  # 只用top 5
).get_preferred_score()

random_only_score = aggregate_scored_sequence_simulations(
    scored_simulation.scored_sequence_simulations[5:]  # 只用random 5
).get_preferred_score()
```

**三种分数的意义**：
- `score`: 整体可解释性（top + random）
- `top_only_score`: 对高激活examples的解释质量
- `random_only_score`: 对一般examples的泛化能力

---

## 📝 完整流程总结

### 整体Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 数据收集 (make_feature_activation_dataset)          │
├─────────────────────────────────────────────────────────────┤
│ OpenWebText (50000 fragments) → Language Model              │
│                                ↓                             │
│                          MLP Activations                     │
│                                ↓                             │
│                      Sparse Autoencoder                      │
│                                ↓                             │
│                Feature Activations (DataFrame)               │
│  [fragment_tokens, feature_0_max, feature_0_act_0, ...]     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 生成解释 (interpret)                                 │
├─────────────────────────────────────────────────────────────┤
│ For each feature:                                            │
│   1. 选择 Top 20 + Random 20 examples                       │
│   2. 分割为 Train (10) + Valid (10)                         │
│   3. Train → GPT-4 → Explanation                            │
│      "this neuron activates for ..."                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: 模拟和评分 (simulate_and_score)                      │
├─────────────────────────────────────────────────────────────┤
│ For each validation example:                                │
│   1. Explanation + Tokens → Simulator (text-davinci-003)    │
│   2. Simulator 使用 logprobs 预测激活                        │
│   3. 计算 correlation(predicted, actual)                    │
│                                                              │
│ Aggregate → Final Score                                     │
└─────────────────────────────────────────────────────────────┘
```

### 关键数字

| 参数 | 值 | 说明 |
|------|-----|------|
| `OPENAI_MAX_FRAGMENTS` | 50,000 | 收集的文本片段数 |
| `OPENAI_FRAGMENT_LEN` | 64 | 每个片段的token数 |
| `TOTAL_EXAMPLES` | 20 | Top + Random各20个 |
| `OPENAI_EXAMPLES_PER_SPLIT` | 5 | 每个split的样本数 |
| `N_SPLITS` | 4 | 总共4个splits |
| Train examples | 10 | 5 top + 5 random |
| Validation examples | 10 | 5 top + 5 random |

---

## 🆚 与我们现代化实现的对比

### evaluate_with_simulator.py

我们的实现遵循完全相同的三步流程：

#### Step 1: 数据收集
```python
# 我们的实现 (evaluate_with_simulator.py:244-276)
def collect_feature_activations(autoencoder, model, n_samples=100, layer=6):
    """
    相同的逻辑：
    - 从OpenWebText采样
    - 运行模型获取MLP激活
    - 通过autoencoder编码
    - 返回 tokens_list, feature_acts_list
    """
```

**差异**：
- ✅ 简化了DataFrame结构（直接用lists）
- ✅ 没有50000个fragments（默认100个，可配置）
- ✅ 使用更现代的API

#### Step 2: 生成解释
```python
# 我们的实现 (evaluate_with_simulator.py:38-98)
async def generate_explanation(self, tokens_list, activations_list, n_examples=5):
    """
    相同的逻辑：
    - 找出top-activating examples
    - 格式化为 token(activation) pairs
    - 调用GPT-4生成解释
    """
    # 区别：直接调用OpenAI API，不依赖neuron_explainer
```

**差异**：
- ✅ 使用Chat Completions API（vs Completions API）
- ✅ 简化的prompt（保留核心思想）
- ✅ 不使用neuron_explainer库

#### Step 3: 模拟和评分
```python
# 我们的实现 (evaluate_with_simulator.py:100-163)
async def simulate_and_score(self, explanation, tokens_list, activations_list):
    """
    相同的逻辑：
    - 使用ModernNeuronSimulator模拟
    - 计算correlation
    - 返回分数
    """
```

**差异**：
- ⚠️ 不使用logprobs（Chat API限制）
- ✅ 使用确定性预测（temperature=0）
- ✅ 简化但有效的评分

### modern_simulator.py

直接对应 `simulator.py` 的核心功能：

```python
# 原始: ExplanationNeuronSimulator
# 现代: ModernNeuronSimulator

# 相同点：
# - Few-shot prompting
# - <start> token \t activation <end> 格式
# - 归一化到0-10

# 不同点：
# - 原始使用 logprobs 计算期望值
# - 现代使用 temperature=0 确定性采样
```

---

## 💡 核心洞察

### 为什么这个方法有效？

1. **Few-shot Learning**: GPT-4看过其他特征的examples，学会了"解释神经元"这个任务

2. **Self-consistency**: 如果解释是对的，那么基于解释的模拟应该接近实际激活

3. **Logprobs技巧**: 不需要生成文本，直接从概率分布计算期望激活

### 论文的创新点

1. **自动化**: 不需要人工标注
2. **可扩展**: 可以评估成千上万个特征
3. **定量**: 给出0-1的分数，而非定性描述

### 我们的改进

1. **现代API**: 使用最新的OpenAI接口
2. **简化依赖**: 不需要neuron_explainer库
3. **易于修改**: 代码清晰，容易适配新模型

---

## 📊 实际示例

### 原始论文输出

```
Feature 42, explanation='this neuron activates for numeric values and digits'
Feature 42, score=0.85, top_only_score=0.92, random_only_score=0.78

保存到: feature_42/explanation.txt
this neuron activates for numeric values and digits
Score: 0.85
Explainer model: gpt-4
Simulator model: text-davinci-003
Top only score: 0.92
Random only score: 0.78
```

### 我们的实现输出

```
Step 1: Generating explanation...
Explanation: this neuron activates for numeric values and digits

Step 2: Simulating activations on validation set...
Average score: 0.850
Score range: [0.723, 0.921]
Score std: 0.089

RESULT
Explanation: this neuron activates for numeric values and digits
Score: 0.850 ± 0.089
Range: [0.723, 0.921]
```

**相同的核心结果，更清晰的输出！**

---

## 🚀 使用建议

### 何时使用原始实现？

- 需要完全复现论文结果
- 需要logprobs的完整概率分布
- 发表研究论文

### 何时使用我们的实现？

- 快速评估特征质量
- 不需要完全复现
- 使用最新模型（gpt-4-turbo等）
- 易于修改和扩展

### 混合策略

```python
# 1. 用我们的实现快速筛选
from evaluate_with_simulator import FeatureEvaluatorWithSimulator

evaluator = FeatureEvaluatorWithSimulator()
quick_results = await evaluator.evaluate_feature(...)

# 2. 对高分特征，用原始方法详细验证
if quick_results["average_score"] > 0.8:
    # 运行原始 interpret.py
    # 获取更详细的统计数据
```

---

## 📚 参考代码位置

| 功能 | 原始代码 | 现代化实现 |
|------|----------|-----------|
| 数据收集 | `interpret.py:82-212` | `evaluate_with_simulator.py:244-276` |
| 生成解释 | `interpret.py:334-346` | `evaluate_with_simulator.py:38-98` |
| 模拟激活 | `simulator.py:125-180` | `modern_simulator.py:100-180` |
| 评分 | `scoring.py` (neuron_explainer库) | `modern_simulator.py:246-270` |

---

## 🎓 总结

原始论文的可解释性评估是一个**精心设计的pipeline**：

1. ✅ 大规模数据收集（50k fragments）
2. ✅ 智能的few-shot prompting
3. ✅ 巧妙的logprobs技巧
4. ✅ 训练/验证分离
5. ✅ Top/Random对比

我们的现代化实现**保留了所有核心思想**，同时：

1. ✅ 使用最新API
2. ✅ 简化代码结构
3. ✅ 易于理解和修改
4. ✅ 实际效果相当

**两者都是有效的工具，选择取决于你的具体需求！**
