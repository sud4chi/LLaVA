from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .d_prune_score import compute_d_prune_importance
from .selector import hard_prune_by_keep_count, keep_count_from_ratio


@dataclass
class DynamicPruningConfig:
    input_type: str = "scores"
    score_method: str = "attention"
    hidden_size: int = 128
    min_keep: int = 64
    max_keep: int | None = None
    min_keep_ratio: float = 0.05
    max_keep_ratio: float = 1.0
    target_keep_ratio: float = 0.25
    alpha: float = 1.0
    temperature: float = 0.1
    budget_loss_weight: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "DynamicPruningConfig":
        return cls(**values)


class DynamicVisionTokenPruner(nn.Module):
    """Predict an image-wise keep ratio and prune/mask LLaVA visual tokens."""

    def __init__(self, vision_hidden_size: int, config: DynamicPruningConfig | None = None):
        super().__init__()
        self.config = config or DynamicPruningConfig()
        self.vision_hidden_size = vision_hidden_size

        score_stats_dim = 8
        token_stats_dim = 0
        if self.config.input_type in {"tokens", "tokens_scores"}:
            token_stats_dim = vision_hidden_size * 2

        input_dim = score_stats_dim + token_stats_dim
        self.predictor = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, 1),
        )

    def _score_stats(self, scores: torch.Tensor) -> torch.Tensor:
        scores_f = scores.float()
        probs = scores_f.clamp_min(0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1, keepdim=True)
        quantiles = torch.quantile(scores_f, torch.tensor([0.25, 0.5, 0.75], device=scores.device), dim=-1).transpose(0, 1)
        return torch.cat(
            [
                scores_f.mean(dim=-1, keepdim=True),
                scores_f.std(dim=-1, keepdim=True, unbiased=False),
                scores_f.amin(dim=-1, keepdim=True),
                scores_f.amax(dim=-1, keepdim=True),
                quantiles,
                entropy,
            ],
            dim=-1,
        )

    def _features(self, visual_tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        features = [self._score_stats(scores)]
        if self.config.input_type in {"tokens", "tokens_scores"}:
            tokens_f = visual_tokens.float()
            features.append(tokens_f.mean(dim=1))
            features.append(tokens_f.std(dim=1, unbiased=False))
        param = next(self.predictor.parameters())
        return torch.cat(features, dim=-1).to(device=param.device, dtype=param.dtype)

    def predict_keep_ratio(self, visual_tokens: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        logits = self.predictor(self._features(visual_tokens, scores)).squeeze(-1)
        ratio = torch.sigmoid(logits)
        span = self.config.max_keep_ratio - self.config.min_keep_ratio
        return self.config.min_keep_ratio + span * ratio

    def forward(
        self,
        visual_tokens: torch.Tensor,
        scores: torch.Tensor | None = None,
        hard: bool | None = None,
    ) -> tuple[torch.Tensor | list[torch.Tensor], dict]:
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
            squeeze_output = True
        elif visual_tokens.ndim == 3:
            squeeze_output = False
        else:
            raise ValueError(f"visual_tokens must be [tokens, dim] or [batch, tokens, dim], got {tuple(visual_tokens.shape)}")

        if scores is None and self.config.score_method == "attention":
            raise ValueError("Dynamic pruning is configured for attention scores, but no scores were provided.")
        if scores is None:
            scores = compute_d_prune_importance(visual_tokens, method=self.config.score_method)
        elif scores.ndim == 1:
            scores = scores.unsqueeze(0)
        scores = scores.to(device=visual_tokens.device).float()
        score_min = scores.amin(dim=-1, keepdim=True)
        score_max = scores.amax(dim=-1, keepdim=True)
        scores = ((scores - score_min) / (score_max - score_min).clamp_min(1e-6)).to(dtype=visual_tokens.dtype)

        keep_ratio = self.predict_keep_ratio(visual_tokens, scores)
        use_hard = (not self.training) if hard is None else hard

        if use_hard:
            outputs = []
            keep_counts = []
            for sample_tokens, sample_scores, sample_ratio in zip(visual_tokens, scores, keep_ratio):
                keep_count = keep_count_from_ratio(
                    sample_tokens.shape[0],
                    sample_ratio,
                    min_keep=self.config.min_keep,
                    max_keep=self.config.max_keep,
                )
                outputs.append(hard_prune_by_keep_count(sample_tokens, sample_scores, keep_count, alpha=self.config.alpha))
                keep_counts.append(keep_count)
            aux = {
                "keep_ratio": keep_ratio.detach(),
                "keep_count": torch.tensor(keep_counts, device=visual_tokens.device, dtype=torch.long),
                "scores": scores.detach(),
            }
            if squeeze_output:
                return outputs[0], aux
            return outputs, aux

        score_min = scores.float().amin(dim=-1)
        score_max = scores.float().amax(dim=-1)
        thresholds = score_min + (score_max - score_min) * (1.0 - keep_ratio.float()).clamp(0.0, 1.0)
        masks = torch.sigmoid((scores.float() - thresholds.unsqueeze(-1)) / max(self.config.temperature, 1e-6))
        masked_tokens = visual_tokens * masks.to(dtype=visual_tokens.dtype).unsqueeze(-1)
        aux = {
            "keep_ratio": keep_ratio,
            "soft_keep_ratio": masks.mean(dim=-1),
            "scores": scores.detach(),
            "mask": masks,
        }
        if squeeze_output:
            return masked_tokens.squeeze(0), aux
        return masked_tokens, aux

    def budget_loss(self, aux: dict) -> torch.Tensor:
        if self.config.budget_loss_weight <= 0:
            keep_ratio = aux["keep_ratio"]
            return keep_ratio.new_zeros(())
        keep_ratio = aux.get("soft_keep_ratio", aux["keep_ratio"])
        target = keep_ratio.new_full(keep_ratio.shape, self.config.target_keep_ratio)
        return self.config.budget_loss_weight * (keep_ratio - target).pow(2).mean()
