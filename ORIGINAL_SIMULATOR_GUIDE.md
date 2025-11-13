# 使用原始 Simulator 的指南

## 📋 概述

原始论文使用 `neuron_explainer` 库的 simulator，但该库使用已废弃的 OpenAI API。

## ⚠️ 问题

```python
# 原始代码 (interpret.py:348-358)
SIMULATOR_MODEL_NAME = "text-davinci-003"  # ❌ 已废弃

simulator = UncalibratedNeuronSimulator(
    ExplanationNeuronSimulator(
        SIMULATOR_MODEL_NAME,
        explanation,
        max_concurrent=MAX_CONCURRENT,
        prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
    )
)
```

**错误信息**：
```
Error: Model `text-davinci-003` not found
404: The model `text-davinci-003` does not exist
```

## 🔧 解决方案

### 方案1: 使用 gpt-3.5-turbo-instruct（推荐）

这是最接近 text-davinci-003 的替代品，仍然支持 Completions API。

```python
# 修改 interpret.py
SIMULATOR_MODEL_NAME = "gpt-3.5-turbo-instruct"  # ✅ 替代品
```

**无需其他修改**！neuron_explainer 的代码应该能直接工作。

### 方案2: 使用 gpt-3.5-turbo（需要修改）

Chat 模型，需要修改 prompt format。

```python
SIMULATOR_MODEL_NAME = "gpt-3.5-turbo"

# 修改 format 判断
format = PromptFormat.HARMONY_V4  # ✅ Chat API 需要这个格式
```

**但是**：neuron_explainer 库的 ApiClient 需要更新以支持 Chat Completions API。

### 方案3: 修改 neuron_explainer 库（高级）

如果方案1和2都不工作，需要修改库本身。

找到 `neuron_explainer/api_client.py`:

```python
# 原始代码 (已废弃)
async def make_request(self, prompt, max_tokens, echo=True, logprobs=15, **kwargs):
    response = await self.client.completions.create(
        model=self.model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        echo=echo,
        logprobs=logprobs,
        **kwargs
    )
    return response
```

修改为支持 Chat Completions:

```python
async def make_request(self, prompt=None, messages=None, max_tokens=100, **kwargs):
    if messages:
        # Chat Completions API
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs
        )
    else:
        # 传统 Completions API (gpt-3.5-turbo-instruct)
        response = await self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=max_tokens,
            **kwargs
        )
    return response
```

## 🧪 测试原始 Simulator

### 运行测试脚本

```bash
python test_original_simulator.py
```

这个脚本会：
1. 尝试使用原始的 neuron_explainer 类
2. 提供详细的错误信息
3. 建议修复方法

### 预期输出

**如果成功**：
```
Test 1: Basic Simulation (Original Style)
============================================================
Explanation: this neuron activates for numeric values and digits
Simulator model: gpt-3.5-turbo-instruct
Prompt format: PromptFormat.INSTRUCTION_FOLLOWING
Validation records: 3

✓ Simulator created
Simulating activations...

✓ Overall Score: 0.850

Detailed Results:
Record 1:
  Tokens: ['The', 'price', 'is', '$', '42']
  Actual:    [0.1, 0.3, 0.1, 0.5, 0.95]
  Predicted: [0.0, 2.0, 0.0, 5.0, 10.0]
  Score: 0.89
```

**如果失败**：
```
✗ Error creating simulator: Model `text-davinci-003` not found

Possible issues:
1. Model name not supported (try gpt-3.5-turbo-instruct)
2. API format changed (try different PromptFormat)
3. neuron_explainer library needs updating
```

## 📊 原始代码的核心类

### 1. ActivationRecord

```python
from neuron_explainer.activations.activations import ActivationRecord

# 存储一个序列的 tokens 和激活值
record = ActivationRecord(
    tokens=["The", "price", "is", "$", "42"],
    activations=[0.1, 0.3, 0.1, 0.5, 0.95]
)
```

### 2. ExplanationNeuronSimulator

```python
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator

# 基于解释模拟激活
simulator = ExplanationNeuronSimulator(
    model_name="gpt-3.5-turbo-instruct",
    explanation="this neuron activates for numeric values",
    max_concurrent=10,
    prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
)

# 模拟单个序列
simulation = await simulator.simulate(tokens)
```

### 3. UncalibratedNeuronSimulator

```python
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator

# 包装 ExplanationNeuronSimulator
uncalibrated = UncalibratedNeuronSimulator(
    ExplanationNeuronSimulator(...)
)
```

**作用**：提供统一接口，可以与 calibrated simulator 互换。

### 4. simulate_and_score

```python
from neuron_explainer.explanations.scoring import simulate_and_score

# 模拟并计算分数
scored_simulation = await simulate_and_score(
    simulator=uncalibrated,
    activation_records=validation_records
)

score = scored_simulation.get_preferred_score()
```

**内部流程**：
1. 对每个 record 调用 `simulator.simulate()`
2. 计算 predicted vs actual 的相关性
3. 返回 `ScoredSimulation` 对象

### 5. aggregate_scored_sequence_simulations

```python
from neuron_explainer.explanations.scoring import aggregate_scored_sequence_simulations

# 计算 top-only 分数
top_only_score = aggregate_scored_sequence_simulations(
    scored_simulation.scored_sequence_simulations[:5]
).get_preferred_score()
```

## 🆚 原始 vs 现代实现对比

| 特性 | 原始 (neuron_explainer) | 现代 (modern_simulator.py) |
|------|------------------------|---------------------------|
| **API** | Completions (废弃) | Chat Completions ✅ |
| **模型** | text-davinci-003 | gpt-3.5-turbo, gpt-4 ✅ |
| **依赖** | neuron_explainer库 | 仅 openai + numpy ✅ |
| **Logprobs** | ✅ 完整支持 | ⚠️ 简化 |
| **复杂度** | 高（多层抽象） | 低（直接调用）✅ |
| **可维护性** | 难（外部库） | 易（自己的代码）✅ |

## 💡 推荐策略

### 用于快速开发

```python
from modern_simulator import ModernNeuronSimulator

simulator = ModernNeuronSimulator(
    explanation="...",
    model_name="gpt-3.5-turbo"
)
simulation = simulator.simulate(tokens)
```

✅ 简单、快速、最新 API

### 用于论文复现

```python
# 修改 interpret.py 中的模型名
SIMULATOR_MODEL_NAME = "gpt-3.5-turbo-instruct"

# 其他代码不变
```

✅ 最接近原始方法

### 用于研究/调试

```python
# 同时运行两者，对比结果
original_score = ...  # 使用 neuron_explainer
modern_score = ...    # 使用 modern_simulator

print(f"Original: {original_score:.3f}")
print(f"Modern:   {modern_score:.3f}")
```

✅ 验证一致性

## 🔍 调试技巧

### 1. 检查 neuron_explainer 版本

```bash
pip show neuron-explainer
```

### 2. 测试 API 连接

```python
from openai import OpenAI

client = OpenAI()

# 测试 Completions API (gpt-3.5-turbo-instruct)
response = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Test",
    max_tokens=5
)
print(response.choices[0].text)
```

### 3. 启用详细日志

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("neuron_explainer")
```

### 4. 直接测试 simulate

```python
# 不通过 simulate_and_score，直接调用
simulator = ExplanationNeuronSimulator(...)
simulation = await simulator.simulate(["Test", "tokens"])

print(simulation.expected_activations)
```

## 📝 完整工作示例

```python
import asyncio
from neuron_explainer.activations.activations import ActivationRecord
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.explanations.calibrated_simulator import UncalibratedNeuronSimulator
from neuron_explainer.explanations.scoring import simulate_and_score
from neuron_explainer.explanations.prompt_builder import PromptFormat

async def main():
    # 1. 准备数据
    explanation = "numeric values"
    records = [
        ActivationRecord(
            tokens=["The", "price", "is", "$", "42"],
            activations=[0.1, 0.3, 0.1, 0.5, 0.95]
        )
    ]

    # 2. 创建 simulator（使用修复后的模型名）
    simulator = UncalibratedNeuronSimulator(
        ExplanationNeuronSimulator(
            "gpt-3.5-turbo-instruct",  # ✅ 修复
            explanation,
            max_concurrent=10,
            prompt_format=PromptFormat.INSTRUCTION_FOLLOWING,
        )
    )

    # 3. 模拟和评分
    scored = await simulate_and_score(simulator, records)
    score = scored.get_preferred_score()

    print(f"Score: {score:.3f}")

asyncio.run(main())
```

## ⚠️ 常见错误

### Error 1: Model not found

```
404: The model `text-davinci-003` does not exist
```

**解决**：改用 `gpt-3.5-turbo-instruct`

### Error 2: Invalid prompt format

```
Error: 'messages' is a required property
```

**解决**：使用 `PromptFormat.HARMONY_V4` for chat models

### Error 3: Echo parameter not supported

```
Error: 'echo' is not a valid parameter
```

**解决**：Chat API 不支持 echo，需要修改 neuron_explainer 库

### Error 4: Import error

```
ModuleNotFoundError: No module named 'neuron_explainer'
```

**解决**：
```bash
pip install git+https://github.com/openai/automated-interpretability.git
```

## 🎯 总结

| 场景 | 推荐方案 |
|------|---------|
| 快速原型 | `modern_simulator.py` ✅ |
| 批量评估 | `evaluate_with_simulator.py` ✅ |
| 论文复现 | 原始代码 + 修改模型名 |
| 学习原理 | 阅读 `INTERPRETABILITY_CODE_EXPLANATION.md` |

**最简单的方式**：使用我们的现代化实现！

**需要原始库**：修改 `SIMULATOR_MODEL_NAME = "gpt-3.5-turbo-instruct"`

**遇到问题**：运行 `python test_original_simulator.py` 获取详细错误信息
