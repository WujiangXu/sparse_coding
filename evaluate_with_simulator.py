"""
使用 modern_simulator 评估稀疏自编码器特征
更接近论文的方法，更准确
"""

import torch
import json
import asyncio
import time
import numpy as np
from typing import List, Tuple
from openai import AsyncOpenAI
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm

from modern_simulator import (
    ModernNeuronSimulator,
    TokenByTokenSimulator,
    FewShotExample,
    compute_simulation_score,
    SequenceSimulation
)


class FeatureEvaluatorWithSimulator:
    """使用modern_simulator进行特征评估的完整pipeline"""

    def __init__(
        self,
        api_key: str = None,
        explainer_model: str = "gpt-4",
        simulator_model: str = "gpt-3.5-turbo",
        use_token_by_token: bool = False
    ):
        """
        Args:
            api_key: OpenAI API key (默认从secrets.json读取)
            explainer_model: 生成解释的模型（推荐gpt-4）
            simulator_model: 模拟激活的模型（gpt-3.5-turbo足够）
            use_token_by_token: 是否使用token-by-token模式（更准确但慢）
        """
        if api_key is None:
            with open("secrets.json") as f:
                api_key = json.load(f)["openai_key"]

        self.client = AsyncOpenAI(api_key=api_key)
        self.explainer_model = explainer_model
        self.simulator_model = simulator_model
        self.use_token_by_token = use_token_by_token

    async def generate_explanation(
        self,
        tokens_list: List[List[str]],
        activations_list: List[List[float]],
        n_examples: int = 5
    ) -> str:
        """
        Step 1: 生成特征解释

        Args:
            tokens_list: 所有样本的tokens
            activations_list: 对应的激活值
            n_examples: 使用多少个top examples

        Returns:
            自然语言解释
        """
        # 找出激活最强的examples
        max_activations = [max(acts) for acts in activations_list]
        sorted_indices = np.argsort(max_activations)[::-1]
        top_indices = sorted_indices[:n_examples]

        # 格式化examples
        example_texts = []
        for idx in top_indices:
            tokens = tokens_list[idx]
            acts = activations_list[idx]

            # 归一化激活值
            if max(acts) > 0:
                norm_acts = [a / max(acts) for a in acts]
            else:
                norm_acts = acts

            # 格式化为 token(activation) 形式
            formatted = " ".join([
                f"{tok}({act:.2f})"
                for tok, act in zip(tokens, norm_acts)
            ])
            example_texts.append(f"Example {len(example_texts)+1}: {formatted}")

        # 构建prompt
        prompt = f"""I'm analyzing a feature from a sparse autoencoder trained on language model activations.

Below are text sequences where this feature activates most strongly. Each token is followed by its activation value in parentheses (normalized to max=1.0).

{chr(10).join(example_texts)}

Based on these examples, provide a concise explanation (1-2 sentences) of what pattern or concept this feature appears to detect. Focus on what makes the tokens with high activation values similar or related.

Respond with ONLY the explanation, starting with "this neuron activates for" or similar."""

        # 调用GPT-4生成解释
        response = await self.client.chat.completions.create(
            model=self.explainer_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing neural network features and identifying patterns in activations."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        explanation = response.choices[0].message.content.strip()

        # 清理解释格式
        if not explanation.lower().startswith("this neuron activates"):
            explanation = "this neuron activates for " + explanation

        return explanation

    async def simulate_and_score(
        self,
        explanation: str,
        tokens_list: List[List[str]],
        activations_list: List[List[float]],
        custom_examples: List[FewShotExample] = None
    ) -> Tuple[List[SequenceSimulation], float, List[float]]:
        """
        Step 2 & 3: 模拟激活并计算分数

        Args:
            explanation: 特征解释
            tokens_list: 验证集的tokens
            activations_list: 验证集的真实激活
            custom_examples: 可选的自定义few-shot examples

        Returns:
            (simulations, average_score, individual_scores)
        """
        # 创建simulator
        if self.use_token_by_token:
            simulator = TokenByTokenSimulator(
                explanation=explanation,
                model_name=self.simulator_model,
                few_shot_examples=custom_examples
            )
        else:
            simulator = ModernNeuronSimulator(
                explanation=explanation,
                model_name=self.simulator_model,
                few_shot_examples=custom_examples,
                use_async=True
            )

        # 对每个验证样本模拟激活
        simulations = []
        scores = []

        for tokens, actual_acts in zip(tokens_list, activations_list):
            # 模拟
            if self.use_token_by_token:
                simulation = await simulator.simulate(tokens)
            else:
                simulation = await simulator.simulate_async(tokens)

            # 评分
            score = compute_simulation_score(simulation, actual_acts)

            simulations.append(simulation)
            scores.append(score)

        average_score = np.mean(scores)

        return simulations, average_score, scores

    async def evaluate_feature(
        self,
        train_tokens_list: List[List[str]],
        train_activations_list: List[List[float]],
        val_tokens_list: List[List[str]],
        val_activations_list: List[List[float]],
        n_train_examples: int = 5,
        custom_examples: List[FewShotExample] = None,
        verbose: bool = True
    ) -> dict:
        """
        完整评估一个特征

        Args:
            train_tokens_list: 训练集tokens（用于生成解释）
            train_activations_list: 训练集激活
            val_tokens_list: 验证集tokens（用于评分）
            val_activations_list: 验证集激活
            n_train_examples: 用多少个训练样本生成解释
            custom_examples: 自定义few-shot examples
            verbose: 是否打印过程

        Returns:
            评估结果字典
        """
        if verbose:
            print("Step 1: Generating explanation...")

        # Step 1: 生成解释
        explanation = await self.generate_explanation(
            train_tokens_list,
            train_activations_list,
            n_examples=n_train_examples
        )

        if verbose:
            print(f"Explanation: {explanation}\n")
            print("Step 2: Simulating activations on validation set...")

        # Step 2 & 3: 模拟并评分
        simulations, avg_score, individual_scores = await self.simulate_and_score(
            explanation,
            val_tokens_list,
            val_activations_list,
            custom_examples=custom_examples
        )

        if verbose:
            print(f"Average score: {avg_score:.3f}")
            print(f"Score range: [{min(individual_scores):.3f}, {max(individual_scores):.3f}]")
            print(f"Score std: {np.std(individual_scores):.3f}")

        return {
            "explanation": explanation,
            "average_score": avg_score,
            "individual_scores": individual_scores,
            "score_std": np.std(individual_scores),
            "score_min": min(individual_scores),
            "score_max": max(individual_scores),
            "n_validation_samples": len(val_tokens_list)
        }


def collect_feature_activations(
    autoencoder,
    model: HookedTransformer,
    n_samples: int = 100,
    layer: int = 6,
    device: str = "cuda"
):
    """收集特征激活数据"""
    dataset = load_dataset("openwebtext", split="train", streaming=True)
    iter_dataset = iter(dataset)

    all_tokens = []
    all_feature_acts = []

    model.to(device)
    autoencoder.to_device(device)

    for _ in tqdm(range(n_samples), desc="Collecting activations"):
        try:
            text = next(iter_dataset)["text"]
            tokens = model.to_tokens(text[:500], prepend_bos=True).to(device)

            # 获取MLP激活
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
                mlp_acts = cache[f"blocks.{layer}.mlp.hook_post"]

                # 编码到特征空间
                feature_acts = autoencoder.encode(mlp_acts)  # [1, seq_len, n_features]

            all_tokens.append(model.to_str_tokens(tokens[0]))
            all_feature_acts.append(feature_acts[0].cpu())

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    return all_tokens, all_feature_acts


async def evaluate_single_feature_example():
    """示例：评估单个特征"""
    print("="*60)
    print("Example: Evaluating a Single Feature")
    print("="*60)

    # 创建评估器
    evaluator = FeatureEvaluatorWithSimulator(
        explainer_model="gpt-4",
        simulator_model="gpt-3.5-turbo",
        use_token_by_token=False  # 使用快速模式
    )

    # 假设你已经有了数据
    # 在实际使用中，这些来自 collect_feature_activations()

    # 训练集（用于生成解释）
    train_tokens = [
        ["The", "price", "is", "$", "42"],
        ["I", "bought", "3", "apples"],
        ["Chapter", "7", "is", "interesting"],
        ["Version", "2", ".", "0"],
        ["There", "are", "100", "people"],
        ["He", "scored", "95", "points"],
        ["In", "the", "year", "2024"],
        ["Add", "5", "plus", "10"],
    ]

    train_activations = [
        [0.1, 0.3, 0.1, 0.5, 0.95],
        [0.05, 0.2, 0.88, 0.15],
        [0.3, 0.92, 0.1, 0.15],
        [0.2, 0.85, 0.4, 0.87],
        [0.1, 0.15, 0.90, 0.2],
        [0.08, 0.2, 0.93, 0.15],
        [0.1, 0.2, 0.15, 0.89],
        [0.2, 0.91, 0.3, 0.85],
    ]

    # 验证集（用于评分）
    val_tokens = [
        ["The", "answer", "is", "42"],
        ["She", "is", "25", "years", "old"],
        ["Walking", "in", "the", "park"],  # 无数字，低激活
    ]

    val_activations = [
        [0.1, 0.2, 0.15, 0.92],
        [0.08, 0.12, 0.88, 0.15, 0.1],
        [0.05, 0.08, 0.06, 0.1],
    ]

    # 评估
    result = await evaluator.evaluate_feature(
        train_tokens_list=train_tokens,
        train_activations_list=train_activations,
        val_tokens_list=val_tokens,
        val_activations_list=val_activations,
        n_train_examples=5,
        verbose=True
    )

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(f"Explanation: {result['explanation']}")
    print(f"Score: {result['average_score']:.3f} ± {result['score_std']:.3f}")
    print(f"Range: [{result['score_min']:.3f}, {result['score_max']:.3f}]")

    return result


async def evaluate_all_features_from_autoencoder(
    autoencoder,
    model: HookedTransformer,
    layer: int = 6,
    n_train_samples: int = 50,
    n_val_samples: int = 30,
    output_file: str = "simulator_results.json",
    use_token_by_token: bool = False
):
    """评估自编码器的所有特征"""

    print("="*60)
    print("Evaluating All Features with Modern Simulator")
    print("="*60)

    # Step 1: 收集训练集数据（用于生成解释）
    print("\nCollecting training data...")
    train_tokens, train_feature_acts = collect_feature_activations(
        autoencoder, model, n_samples=n_train_samples, layer=layer
    )

    # Step 2: 收集验证集数据（用于评分）
    print("Collecting validation data...")
    val_tokens, val_feature_acts = collect_feature_activations(
        autoencoder, model, n_samples=n_val_samples, layer=layer
    )

    # Step 3: 创建评估器
    evaluator = FeatureEvaluatorWithSimulator(
        explainer_model="gpt-4",
        simulator_model="gpt-3.5-turbo",
        use_token_by_token=use_token_by_token
    )

    # Step 4: 评估每个特征
    n_features = train_feature_acts[0].shape[-1]
    results = []

    print(f"\nEvaluating {n_features} features...")

    for feature_idx in tqdm(range(n_features)):
        try:
            # 准备该特征的数据
            train_acts = [
                acts[:, feature_idx].tolist()
                for acts in train_feature_acts
            ]
            val_acts = [
                acts[:, feature_idx].tolist()
                for acts in val_feature_acts
            ]

            # 评估
            result = await evaluator.evaluate_feature(
                train_tokens_list=train_tokens,
                train_activations_list=train_acts,
                val_tokens_list=val_tokens,
                val_activations_list=val_acts,
                n_train_examples=5,
                verbose=False
            )

            results.append({
                "feature_idx": feature_idx,
                "explanation": result["explanation"],
                "score": result["average_score"],
                "score_std": result["score_std"],
                "score_min": result["score_min"],
                "score_max": result["score_max"]
            })

            print(f"\nFeature {feature_idx}: {result['average_score']:.3f}")
            print(f"  {result['explanation']}")

            # Rate limiting
            time.sleep(2)

        except Exception as e:
            print(f"\nError on feature {feature_idx}: {e}")
            results.append({
                "feature_idx": feature_idx,
                "error": str(e)
            })

    # Step 5: 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Step 6: 打印Top 10
    valid_results = [r for r in results if "score" in r]
    valid_results.sort(key=lambda x: x["score"], reverse=True)

    print("\n" + "="*60)
    print("TOP 10 MOST INTERPRETABLE FEATURES")
    print("="*60)
    for i, r in enumerate(valid_results[:10], 1):
        print(f"{i}. Feature {r['feature_idx']}: {r['score']:.3f} ± {r['score_std']:.3f}")
        print(f"   {r['explanation']}\n")

    return results


# 使用自定义few-shot examples（推荐！）
async def evaluate_with_custom_examples():
    """使用自定义few-shot examples进行评估"""

    # 为你的特定领域创建examples
    custom_examples = [
        FewShotExample(
            explanation="the token 'the'",
            tokens=["In", "the", "beginning", "there", "was", "the", "word"],
            activations=[0, 10, 0, 0, 0, 10, 0]
        ),
        FewShotExample(
            explanation="negative words and sentiment",
            tokens=["I", "hate", "this", "terrible", "movie"],
            activations=[0, 9, 2, 8, 3]
        ),
        FewShotExample(
            explanation="numeric values",
            tokens=["The", "price", "is", "$", "42", "dollars"],
            activations=[0, 2, 0, 5, 10, 3]
        ),
    ]

    evaluator = FeatureEvaluatorWithSimulator()

    # 你的数据...
    train_tokens = []
    train_activations = []
    val_tokens = []
    val_activations = []

    if not train_tokens:  # 如果没有数据，使用示例数据
        print("Using example data for demonstration...")
        return await evaluate_single_feature_example()

    result = await evaluator.evaluate_feature(
        train_tokens_list=train_tokens,
        train_activations_list=train_activations,
        val_tokens_list=val_tokens,
        val_activations_list=val_activations,
        custom_examples=custom_examples  # 使用自定义examples
    )

    return result


if __name__ == "__main__":
    # 运行示例
    print("Running example evaluation...\n")
    asyncio.run(evaluate_single_feature_example())

    # 实际使用时：
    # 1. 加载你的模型
    # autoencoder = torch.load("path/to/autoencoder.pt")
    # model = HookedTransformer.from_pretrained("gpt2")

    # 2. 评估所有特征
    # asyncio.run(evaluate_all_features_from_autoencoder(
    #     autoencoder, model, layer=6
    # ))
