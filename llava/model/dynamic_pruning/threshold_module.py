from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import inf
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .d_prune_score import compute_d_prune_importance
from .selector import (
    gather_tokens_in_original_order,
    select_visual_token_indices,
    split_core_frontier_indices,
)


@dataclass
class DynamicPruningConfig:
    """Configuration for SCoRe Core--Frontier dynamic pruning."""

    score_method: str = "attention"
    hidden_size: int = 128
    min_tokens: int = 40
    max_tokens: int = 80
    target_avg_tokens: float = 64.0
    alpha: float = 0.8
    threshold: float = 0.0
    huber_delta: float = 1.0

    def __post_init__(self) -> None:
        if self.min_tokens <= 0:
            raise ValueError(f"min_tokens must be positive, got {self.min_tokens}")
        if self.max_tokens < self.min_tokens:
            raise ValueError(
                f"max_tokens ({self.max_tokens}) must be >= min_tokens ({self.min_tokens})"
            )
        if not self.min_tokens <= self.target_avg_tokens <= self.max_tokens:
            raise ValueError(
                "target_avg_tokens must be within [min_tokens, max_tokens], got "
                f"{self.target_avg_tokens} for [{self.min_tokens}, {self.max_tokens}]"
            )
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {self.huber_delta}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "DynamicPruningConfig":
        # Accept checkpoints from the earlier image-wise keep-ratio prototype
        # where possible, while deliberately dropping obsolete fields.
        values = dict(values)
        if "min_tokens" not in values and "min_keep" in values:
            values["min_tokens"] = values["min_keep"]
        if "max_tokens" not in values and values.get("max_keep") is not None:
            values["max_tokens"] = values["max_keep"]
        valid_names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in valid_names})


class UtilityPredictor(nn.Module):
    """Predict one marginal utility for every frontier visual token."""

    def __init__(self, visual_hidden_size: int, hidden_size: int):
        super().__init__()
        self.frontier_projection = nn.Linear(visual_hidden_size, hidden_size)
        self.core_projection = nn.Linear(visual_hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, frontier_tokens: torch.Tensor, core_summary: torch.Tensor) -> torch.Tensor:
        if frontier_tokens.ndim != 2:
            raise ValueError(f"frontier_tokens must be [frontier, dim], got {tuple(frontier_tokens.shape)}")
        if core_summary.ndim != 1:
            raise ValueError(f"core_summary must be [dim], got {tuple(core_summary.shape)}")
        if frontier_tokens.shape[-1] != core_summary.shape[-1]:
            raise ValueError("frontier token and core summary dimensions must match")

        frontier_hidden = self.frontier_projection(frontier_tokens)
        core_hidden = self.core_projection(core_summary).unsqueeze(0).expand(frontier_hidden.shape[0], -1)
        return self.mlp(torch.cat([frontier_hidden, core_hidden], dim=-1)).squeeze(-1)


class DynamicVisionTokenPruner(nn.Module):
    """SCoRe ranking followed by learned Core--Frontier token selection."""

    def __init__(self, vision_hidden_size: int, config: DynamicPruningConfig | None = None):
        super().__init__()
        self.config = config or DynamicPruningConfig()
        self.vision_hidden_size = vision_hidden_size
        self.utility_predictor = UtilityPredictor(vision_hidden_size, self.config.hidden_size)

    @property
    def threshold(self) -> float:
        return float(self.config.threshold)

    def set_threshold(self, threshold: float) -> None:
        self.config.threshold = float(threshold)

    def _prepare_salience(
        self,
        ranking_tokens: torch.Tensor,
        scores: torch.Tensor | None,
    ) -> torch.Tensor:
        if scores is None and self.config.score_method == "attention":
            raise ValueError("SCoRe is configured for attention salience, but no CLS attention was provided.")
        if scores is None:
            scores = compute_d_prune_importance(ranking_tokens, method=self.config.score_method)
        if scores.ndim != 1:
            raise ValueError(f"one sample's salience must be [tokens], got {tuple(scores.shape)}")
        if scores.shape[0] != ranking_tokens.shape[0]:
            raise ValueError("salience and ranking token counts must match")

        scores = scores.to(device=ranking_tokens.device, dtype=torch.float32)
        score_min = scores.amin()
        score_max = scores.amax()
        score_range = score_max - score_min
        if score_range <= 1e-6:
            return torch.ones_like(scores)
        return (scores - score_min) / score_range

    def rank(
        self,
        ranking_tokens: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run SCoRe once and return ranking, core, frontier, normalized salience."""
        if ranking_tokens.ndim != 2:
            raise ValueError(f"ranking_tokens must be [tokens, dim], got {tuple(ranking_tokens.shape)}")
        if ranking_tokens.shape[0] < self.config.min_tokens:
            raise ValueError(
                f"The encoder produced {ranking_tokens.shape[0]} tokens, fewer than min_tokens={self.config.min_tokens}"
            )
        salience = self._prepare_salience(ranking_tokens, scores)
        ranking = select_visual_token_indices(
            ranking_tokens,
            salience,
            target_size=self.config.max_tokens,
            alpha=self.config.alpha,
        )
        core_indices, frontier_indices = split_core_frontier_indices(
            ranking,
            min_tokens=self.config.min_tokens,
            max_tokens=self.config.max_tokens,
        )
        return ranking, core_indices, frontier_indices, salience

    def predict_utilities(
        self,
        projected_tokens: torch.Tensor,
        core_indices: torch.Tensor,
        frontier_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict frontier utilities using projector embeddings only."""
        if projected_tokens.ndim != 2:
            raise ValueError(f"projected_tokens must be [tokens, dim], got {tuple(projected_tokens.shape)}")
        if core_indices.numel() == 0:
            raise ValueError("at least one core token is required")
        core_summary = projected_tokens.index_select(0, core_indices).mean(dim=0)
        frontier_tokens = projected_tokens.index_select(0, frontier_indices)
        if frontier_tokens.shape[0] == 0:
            return projected_tokens.new_empty((0,), dtype=next(self.parameters()).dtype)

        parameter = next(self.utility_predictor.parameters())
        return self.utility_predictor(
            frontier_tokens.to(device=parameter.device, dtype=parameter.dtype),
            core_summary.to(device=parameter.device, dtype=parameter.dtype),
        )

    def analyze_sample(
        self,
        projected_tokens: torch.Tensor,
        ranking_tokens: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> dict:
        """Compute and cache all values shared by training variants/inference."""
        if projected_tokens.shape[0] != ranking_tokens.shape[0]:
            raise ValueError("projected and ranking token counts must match")
        ranking, core_indices, frontier_indices, salience = self.rank(ranking_tokens, scores=scores)
        ranking = ranking.to(projected_tokens.device)
        core_indices = core_indices.to(projected_tokens.device)
        frontier_indices = frontier_indices.to(projected_tokens.device)
        utilities = self.predict_utilities(projected_tokens, core_indices, frontier_indices).to(projected_tokens.device)
        return {
            "ranking": ranking,
            "core_indices": core_indices,
            "frontier_indices": frontier_indices,
            "salience": salience,
            "predicted_utilities": utilities,
        }

    def select_from_analysis(
        self,
        projected_tokens: torch.Tensor,
        analysis: dict,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Always retain core and add frontier tokens above the global threshold."""
        threshold = self.threshold if threshold is None else float(threshold)
        core_indices = analysis["core_indices"]
        frontier_indices = analysis["frontier_indices"]
        utilities = analysis["predicted_utilities"]
        selected_frontier = frontier_indices[utilities > threshold]
        selected_indices = torch.cat([core_indices, selected_frontier], dim=0)

        selected_tokens = gather_tokens_in_original_order(projected_tokens, selected_indices)
        aux = dict(analysis)
        aux.update(
            {
                "selected_indices": selected_indices,
                "keep_count": selected_indices.new_tensor(selected_indices.numel()),
                "threshold": threshold,
            }
        )
        return selected_tokens, aux

    def forward(
        self,
        projected_tokens: torch.Tensor,
        ranking_tokens: torch.Tensor | None = None,
        scores: torch.Tensor | None = None,
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if projected_tokens.ndim != 2:
            raise ValueError("DynamicVisionTokenPruner.forward expects one [tokens, dim] sample")
        if ranking_tokens is None:
            ranking_tokens = projected_tokens
        analysis = self.analyze_sample(projected_tokens, ranking_tokens, scores=scores)
        return self.select_from_analysis(projected_tokens, analysis, threshold=threshold)

    def utility_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if predictions.shape != targets.shape:
            raise ValueError(f"utility prediction/target shapes differ: {predictions.shape} and {targets.shape}")
        targets = targets.detach().to(device=predictions.device, dtype=torch.float32)
        return F.huber_loss(predictions.float(), targets, delta=self.config.huber_delta)


def mean_answer_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return answer-token mean CE independently for every packed variant."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("logits [batch, seq, vocab] and labels [batch, seq] must align")
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid_labels = shift_labels.ne(ignore_index)
    token_counts = valid_labels.sum(dim=-1)
    if torch.any(token_counts == 0):
        raise ValueError("Every utility teacher variant must contain at least one answer label")
    token_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).view_as(shift_labels)
    return token_ce.sum(dim=-1) / token_counts


def core_frontier_utility_targets(
    variant_mean_ce: torch.Tensor,
    core_variant_for_candidate: torch.Tensor,
    candidate_variant_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute ``core_loss - candidate_loss`` for every frontier token."""
    return (
        variant_mean_ce.index_select(0, core_variant_for_candidate)
        - variant_mean_ce.index_select(0, candidate_variant_indices)
    )


def calibrate_global_threshold(
    predicted_utilities: Iterable[torch.Tensor | Iterable[float]],
    core_counts: Iterable[int],
    target_avg_tokens: float,
) -> tuple[float, dict]:
    """Choose a strict-``>`` threshold closest to the dataset token target."""
    utility_groups = []
    for values in predicted_utilities:
        if torch.is_tensor(values):
            utility_groups.append(values.detach().float().cpu().reshape(-1))
        else:
            utility_groups.append(torch.tensor(list(values), dtype=torch.float32))
    core_counts = [int(value) for value in core_counts]
    if len(utility_groups) != len(core_counts):
        raise ValueError("predicted_utilities and core_counts must contain the same number of samples")
    if not core_counts:
        raise ValueError("cannot calibrate a threshold without validation samples")

    flat = torch.cat(utility_groups) if utility_groups else torch.empty(0)
    if flat.numel() and not torch.isfinite(flat).all():
        raise ValueError("validation utilities must all be finite")
    sample_count = len(core_counts)
    core_total = sum(core_counts)
    target_total = float(target_avg_tokens) * sample_count

    candidates: list[tuple[float, int]] = []
    if flat.numel() == 0:
        candidates.append((0.0, 0))
    else:
        unique_values, counts = torch.unique(flat, sorted=True, return_counts=True)
        unique_values = unique_values.flip(0)
        counts = counts.flip(0)
        candidates.append((float(unique_values[0].item()), 0))
        kept = 0
        for index, (value, count) in enumerate(zip(unique_values, counts)):
            kept += int(count.item())
            if index + 1 < unique_values.numel():
                # Inference is strict `>`, so the next lower observed value is
                # an exact boundary even for adjacent floating-point values.
                threshold = float(unique_values[index + 1].item())
            else:
                threshold = float(torch.nextafter(value, value.new_tensor(-inf)).item())
            candidates.append((threshold, kept))

    # If two achievable averages are equally close, prefer fewer tokens.
    threshold, frontier_total = min(
        candidates,
        key=lambda item: (abs(core_total + item[1] - target_total), item[1]),
    )
    achieved_avg = (core_total + frontier_total) / sample_count
    return threshold, {
        "num_samples": sample_count,
        "core_tokens": core_total,
        "frontier_tokens": frontier_total,
        "target_avg_tokens": float(target_avg_tokens),
        "achieved_avg_tokens": float(achieved_avg),
    }
