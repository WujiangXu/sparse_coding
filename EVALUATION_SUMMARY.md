# Sparse Autoencoder Evaluation - 完整总结

## 论文评估方法分析

论文 ["Sparse Autoencoders Find Highly Interpretable Features in Language Models"](https://arxiv.org/pdf/2309.08600.pdf) 使用的评估体系：

### 1. 训练时评估 (实时监控)

**损失函数** (`autoencoders/sae_ensemble.py:53-78`):
```python
loss = reconstruction_loss + l1_penalty + bias_decay
     = MSE(x_hat, x) + λ·||c||₁ + γ·||b||₂
```

**每batch记录** (`big_sweep.py:173-197`):
- `l_reconstruction`: 重构损失 (MSE)
- `l_l1`: L1稀疏性惩罚
- `l_bias_decay`: 偏置衰减
- `num_nonzero`: L0稀疏性（非零激活数）

### 2. 重构质量指标

| 指标 | 公式 | 代码位置 | 含义 |
|------|------|----------|------|
| **FVU** | `residuals / total_variance` | `standard_metrics.py:310` | 未解释方差比例（越低越好）|
| **R²** | `1 - FVU` | `standard_metrics.py:344` | 解释方差比例（越高越好）|
| **Perplexity** | Language model loss | `standard_metrics.py:224` | 重构对模型性能影响 |

### 3. 稀疏性指标

| 指标 | 代码位置 | 含义 |
|------|----------|------|
| **L0** (count nonzero) | `big_sweep.py:171` | 平均激活特征数 |
| **L1** (sum abs) | `sae_ensemble.py:64` | 激活值总和 |
| **Ever-active count** | `standard_metrics.py:446` | 曾激活过的特征总数 |
| **Mean nonzero activations** | `standard_metrics.py:305` | 平均非零激活比例 |

### 4. 特征对齐指标

**MMCS (Mean Max Cosine Similarity)** (`standard_metrics.py:276-297`):
- 测量学到的特征与ground truth的相似度
- 用于比较不同字典大小、不同L1值的模型
- 生成热图展示特征对齐情况

**Capacity per Feature** (`standard_metrics.py:356-362`):
- 来自 Scherlis et al. 2022
- 测量每个特征的"独立容量"
- 理想情况下总容量 ≈ 激活维度

### 5. 可解释性评估（论文核心）

**方法** (`interpret.py:265-386`):

1. **收集激活数据**:
   - 在大规模文本上运行模型
   - 记录每个特征的激活值
   - 选择top-activating examples

2. **生成解释** (使用GPT-4):
   ```python
   explainer = TokenActivationPairExplainer(
       model_name="gpt-4",
       prompt_format=PromptFormat.HARMONY_V4
   )
   explanation = await explainer.generate_explanations(
       all_activation_records=train_records,
       max_activation=max_activation
   )
   ```

3. **模拟和评分** (原使用text-davinci-003):
   ```python
   simulator = UncalibratedNeuronSimulator(
       ExplanationNeuronSimulator(
           "text-davinci-003",
           explanation
       )
   )
   scored_simulation = await simulate_and_score(
       simulator,
       valid_activation_records
   )
   score = scored_simulation.get_preferred_score()
   ```

4. **分数类型**:
   - `score`: 整体分数（train + validation）
   - `top_only_score`: 仅top-activating examples
   - `random_only_score`: 仅随机examples

### 6. 对比基线

论文中比较的基线方法：
- **PCA**: 主成分分析
- **ICA**: 独立成分分析
- **NMF**: 非负矩阵分解
- **Neuron basis**: 原始神经元激活
- **Random**: 随机投影

评估维度：
- 稀疏性 vs 重构质量 (trade-off curve)
- 可解释性得分
- 计算效率

---

## 新实现文件

我为你创建了两套实现：

### 完整版本（功能丰富）

| 文件 | 行数 | 说明 |
|------|------|------|
| `interpretability_eval.py` | ~300 | 完整的评估类，支持异步、批处理 |
| `test_interpretability_eval.py` | ~400 | 全面的测试套件，包括多种示例 |
| `INTERPRETABILITY_EVAL_README.md` | - | 详细使用文档 |

**特点**:
- 面向对象设计
- 异步API调用
- 批量评估支持
- 详细的统计信息

### 简化版本（推荐使用）

| 文件 | 行数 | 说明 |
|------|------|------|
| `simple_interp_eval.py` | ~100 | 单个函数，核心功能 |
| `test_simple.py` | ~100 | 简洁的测试代码 |
| `SIMPLE_EVAL_README.md` | - | 快速上手文档 |

**特点**:
- ✅ 代码简洁（~100行）
- ✅ 易于理解和修改
- ✅ 最新OpenAI API
- ✅ 即拿即用

---

## 使用建议

### 快速开始（推荐）

```bash
# 1. 配置API key
echo '{"openai_key": "sk-..."}' > secrets.json

# 2. 运行测试
python test_simple.py
```

### 评估真实模型

```python
from simple_interp_eval import evaluate_feature_interpretability

# 准备你的稀疏自编码器特征的激活数据
tokens_list = [...]      # List[List[str]]
activations_list = [...]  # List[List[float]]

# 评估
result = evaluate_feature_interpretability(
    tokens_list=tokens_list,
    activations_list=activations_list
)

print(f"特征解释: {result['explanation']}")
print(f"可解释性分数: {result['score']:.3f}")
```

### 关键差异说明

| 维度 | 原始实现 | 新实现 |
|------|----------|--------|
| **API** | Completions (已废弃) | Chat Completions (最新) |
| **模型** | text-davinci-003 | gpt-3.5-turbo / gpt-4 |
| **复杂度** | 800+ 行，多个依赖 | 100 行，无外部依赖 |
| **评分方法** | 复杂的simulation scoring | Pearson correlation |
| **使用难度** | 需要neuron_explainer库 | 直接调用OpenAI API |

**核心思想保持一致**: 解释 → 模拟 → 评分

---

## 评估指标解读

### 重构质量

- **FVU < 0.1**: 优秀（解释了90%+的方差）
- **FVU 0.1-0.3**: 良好
- **FVU > 0.5**: 较差

### 稀疏性

- **L0 < 10**: 非常稀疏
- **L0 10-50**: 稀疏
- **L0 > 100**: 不够稀疏

### 可解释性

- **Score > 0.7**: 高度可解释
- **Score 0.5-0.7**: 中等可解释
- **Score < 0.5**: 难以解释

---

## 完整评估流程示例

```python
import torch
from transformer_lens import HookedTransformer
from simple_interp_eval import evaluate_feature_interpretability

# 1. 加载模型
autoencoder = torch.load("autoencoder.pt")
transformer = HookedTransformer.from_pretrained("gpt2")

# 2. 准备数据集
texts = ["Your corpus here..."]
tokens = transformer.to_tokens(texts)

# 3. 获取激活
_, cache = transformer.run_with_cache(tokens)
mlp_acts = cache["blocks.6.mlp.hook_post"]

# 4. 稀疏编码
feature_acts = autoencoder.encode(mlp_acts)

# 5. 对每个特征评估
results = []
for feature_idx in range(feature_acts.shape[-1]):
    # 收集该特征的激活
    acts = feature_acts[:, :, feature_idx]

    tokens_list = [transformer.to_str_tokens(t) for t in tokens]
    activations_list = [acts[i].tolist() for i in range(len(acts))]

    # 评估可解释性
    result = evaluate_feature_interpretability(
        tokens_list,
        activations_list
    )

    results.append({
        "feature": feature_idx,
        "explanation": result["explanation"],
        "score": result["score"]
    })

# 6. 排序并输出最可解释的特征
results.sort(key=lambda x: x["score"], reverse=True)
for r in results[:10]:
    print(f"Feature {r['feature']}: {r['score']:.3f}")
    print(f"  {r['explanation']}\n")
```

---

## 论文关键发现

1. **稀疏自编码器 > PCA/ICA**: 在相同稀疏度下，重构质量更好
2. **高度可解释**: 许多特征对应清晰的语义概念
3. **特征复用**: 更大的字典能学到更细粒度的特征
4. **L1调节**: 增加L1惩罚提高稀疏性，但降低重构质量

## OpenAI API 更新说明

**已废弃**:
- `client.completions.create()`
- `model="text-davinci-003"`
- `echo=True` 参数

**新API**:
- `client.chat.completions.create()`
- `model="gpt-3.5-turbo"` 或 `"gpt-4"`
- `messages=[{"role": "user", "content": "..."}]`
- `logprobs=True, top_logprobs=N` (用于获取token概率)

新实现完全兼容最新API。

---

## 文件清单

**原始代码库**:
- `standard_metrics.py` (867行) - 40+评估指标
- `big_sweep.py` (386行) - 训练循环+实时评估
- `interpret.py` (800+行) - 可解释性评估（使用旧API）
- `autoencoders/` - 各种自编码器实现

**新增文件**:
- ✨ `simple_interp_eval.py` - 简化的评估函数
- ✨ `test_simple.py` - 简洁测试代码
- ✨ `SIMPLE_EVAL_README.md` - 快速文档
- `interpretability_eval.py` - 完整版评估类
- `test_interpretability_eval.py` - 完整测试套件
- `INTERPRETABILITY_EVAL_README.md` - 详细文档
- 📄 `EVALUATION_SUMMARY.md` - 本文档

**推荐使用**: `simple_interp_eval.py` + `test_simple.py`
