# Simple Interpretability Evaluation

最简化的稀疏自编码器可解释性评估工具。

## 文件

- `simple_interp_eval.py` - 核心评估函数（约100行）
- `test_simple.py` - 测试代码
- 本README

## 安装

```bash
pip install openai numpy
```

## 配置

创建 `secrets.json`:

```json
{
  "openai_key": "sk-your-api-key"
}
```

## 使用

### 基本用法

```python
from simple_interp_eval import evaluate_feature_interpretability

# 准备数据
tokens = [
    ["The", "price", "is", "$", "42"],
    ["I", "bought", "3", "apples"],
    # ... 更多例子
]

activations = [
    [0.1, 0.3, 0.1, 0.5, 0.95],  # 对应每个token的激活值
    [0.05, 0.2, 0.88, 0.15],
    # ... 更多激活值
]

# 评估
result = evaluate_feature_interpretability(
    tokens_list=tokens,
    activations_list=activations,
    n_examples=5
)

print(result["explanation"])  # 特征的自然语言解释
print(result["score"])        # 0-1的可解释性分数
```

### 运行测试

```bash
python test_simple.py
```

## 工作原理

1. **解释生成**: 使用GPT-4分析激活最强的例子，生成特征描述
2. **激活预测**: 在验证集上，让GPT-3.5根据解释预测激活值
3. **相关性评分**: 计算预测值与实际值的相关系数

**分数解读**:
- `1.0` = 完美可解释
- `0.5` = 无关联
- `0.0` = 完全反向

## 与真实自编码器集成

```python
# 1. 加载模型
autoencoder = torch.load("model.pt")
transformer = HookedTransformer.from_pretrained("gpt2")

# 2. 获取激活
tokens = transformer.to_tokens("Your text")
_, cache = transformer.run_with_cache(tokens)
mlp_acts = cache["blocks.6.mlp.hook_post"]

# 3. 编码
feature_acts = autoencoder.encode(mlp_acts)  # [batch, seq, n_features]

# 4. 准备某个特征的数据
feature_idx = 42
acts = feature_acts[:, :, feature_idx]  # [batch, seq]

tokens_list = []
activations_list = []
for i in range(acts.shape[0]):
    tokens_list.append(transformer.to_str_tokens(tokens[i]))
    activations_list.append(acts[i].tolist())

# 5. 评估
result = evaluate_feature_interpretability(tokens_list, activations_list)
```

## 代码特点

✅ **简洁**: 单个函数，约100行
✅ **易懂**: 清晰的流程，无复杂抽象
✅ **最新API**: 使用Chat Completions API
✅ **即用**: 无需额外依赖

## 引用

```bibtex
@article{cunningham2023sparse,
  title={Sparse Autoencoders Find Highly Interpretable Features in Language Models},
  author={Cunningham, Hoagy and Ewart, Aidan and Riggs, Logan and Huben, Robert and Sharkey, Lee},
  journal={arXiv preprint arXiv:2309.08600},
  year={2023}
}
```
