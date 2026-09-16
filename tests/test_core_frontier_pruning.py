import torch
from torch import nn

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.model.dynamic_pruning import attach_dynamic_pruner, freeze_llava_train_pruner_only
from llava.model.dynamic_pruning.selector import select_visual_token_indices
from llava.model.dynamic_pruning.threshold_module import (
    DynamicPruningConfig,
    DynamicVisionTokenPruner,
    calibrate_global_threshold,
    core_frontier_utility_targets,
    mean_answer_cross_entropy,
)
from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM


class _CountingVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward_with_attention_scores(self, images):
        self.calls += 1
        features = images.new_tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ).unsqueeze(0).expand(images.shape[0], -1, -1)
        salience = images.new_tensor([[1.0, 0.9, 0.8, 0.1]]).expand(images.shape[0], -1)
        return features, salience


class _CountingProjector(nn.Linear):
    def __init__(self):
        super().__init__(6, 8, bias=False)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return super().forward(inputs)


def test_default_token_budget():
    config = DynamicPruningConfig()
    assert config.min_tokens == 40
    assert config.max_tokens == 80
    assert config.target_avg_tokens == 64


def test_score_ranking_combines_salience_and_coverage_once():
    tokens = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )
    salience = torch.tensor([1.0, 0.9, 0.8, 0.1])
    ranking = select_visual_token_indices(tokens, salience, target_size=3, alpha=0.8)

    assert ranking[0].item() == 0
    # Token 1 is highly salient but redundant with token 0.  SCoRe first
    # covers a distant semantic region instead.
    assert ranking[1].item() in {2, 3}
    assert ranking.unique().numel() == 3


def test_inference_always_keeps_core_and_thresholds_frontier():
    config = DynamicPruningConfig(
        min_tokens=2,
        max_tokens=4,
        target_avg_tokens=3,
        threshold=0.0,
    )
    pruner = DynamicVisionTokenPruner(vision_hidden_size=2, config=config)
    projected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    analysis = {
        "ranking": torch.tensor([2, 0, 3, 1]),
        "core_indices": torch.tensor([2, 0]),
        "frontier_indices": torch.tensor([3, 1]),
        "salience": torch.ones(4),
        "predicted_utilities": torch.tensor([-0.1, 0.5]),
    }

    selected, aux = pruner.select_from_analysis(projected, analysis)

    assert aux["keep_count"].item() == 3
    assert set(aux["selected_indices"].tolist()) == {0, 1, 2}
    assert torch.equal(selected, projected[[0, 1, 2]])


def test_teacher_utility_is_core_ce_minus_candidate_ce():
    variant_losses = torch.tensor([2.0, 1.25, 1.75, 3.0, 2.5])
    core_variants = torch.tensor([0, 0, 3])
    candidate_variants = torch.tensor([1, 2, 4])

    targets = core_frontier_utility_targets(variant_losses, core_variants, candidate_variants)

    assert torch.allclose(targets, torch.tensor([0.75, 0.25, 0.5]))


def test_mean_answer_ce_ignores_prompt_and_averages_answer_tokens():
    logits = torch.zeros(1, 4, 2)
    # Positions 2 and 3 are answer labels.  Causal shifting uses logits at
    # positions 1 and 2, both uniform => CE log(2).
    labels = torch.tensor([[-100, -100, 0, 1]])

    mean_ce = mean_answer_cross_entropy(logits, labels)

    assert torch.allclose(mean_ce, torch.tensor([torch.log(torch.tensor(2.0))]))


def test_global_threshold_hits_dataset_average_budget():
    utilities = [torch.tensor([0.9, 0.2]), torch.tensor([0.8, 0.1])]
    threshold, statistics = calibrate_global_threshold(
        utilities,
        core_counts=[2, 2],
        target_avg_tokens=3,
    )

    kept = sum(int((values > threshold).sum()) for values in utilities)
    assert kept == 2
    assert statistics["achieved_avg_tokens"] == 3


def test_teacher_batches_variants_into_one_frozen_llm_forward():
    config = LlavaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pretraining_tp=1,
    )
    config.tokenizer_padding_side = "right"
    config.tokenizer_model_max_length = 32
    config.tune_mm_mlp_adapter = False
    model = LlavaLlamaForCausalLM(config)
    vision_tower = _CountingVisionTower()
    projector = _CountingProjector()
    model.model.vision_tower = vision_tower
    model.model.mm_projector = projector
    pruner = attach_dynamic_pruner(
        model,
        DynamicPruningConfig(
            min_tokens=2,
            max_tokens=4,
            target_avg_tokens=3,
            hidden_size=4,
        ),
    )
    freeze_llava_train_pruner_only(model)
    model.eval()
    pruner.train()

    llm_calls = [0]
    hook = model.model.register_forward_pre_hook(
        lambda *args: llm_calls.__setitem__(0, llm_calls[0] + 1)
    )
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 2, 3, 4]])
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 3, 4]])
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        images=torch.zeros(1, 3, 2, 2),
        return_dict=True,
    )
    hook.remove()
    output.loss.backward()

    assert (vision_tower.calls, projector.calls, llm_calls[0]) == (1, 1, 1)
    assert all(
        parameter.grad is not None
        for parameter in model.model.dynamic_pruner.parameters()
        if parameter.requires_grad
    )
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "dynamic_pruner" not in name
    )
