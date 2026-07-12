from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batched_visual_tokens(visual_tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if visual_tokens.ndim == 2:
        return visual_tokens.unsqueeze(0), True
    if visual_tokens.ndim == 3:
        return visual_tokens, False
    raise ValueError(f"visual_tokens must be [tokens, dim] or [batch, tokens, dim], got {tuple(visual_tokens.shape)}")


def _as_batched_scores(scores: torch.Tensor, batch_size: int, num_tokens: int) -> torch.Tensor:
    if scores.ndim == 1:
        scores = scores.unsqueeze(0)
    if scores.ndim != 2:
        raise ValueError(f"scores must be [tokens] or [batch, tokens], got {tuple(scores.shape)}")
    if scores.shape[-1] != num_tokens:
        raise ValueError(f"scores token dimension must match visual_tokens, got {scores.shape[-1]} and {num_tokens}")
    if scores.shape[0] == 1 and batch_size != 1:
        scores = scores.expand(batch_size, -1)
    if scores.shape[0] != batch_size:
        raise ValueError(f"scores batch dimension must match visual_tokens, got {scores.shape[0]} and {batch_size}")
    return scores


def select_visual_token_indices(
    visual_tokens: torch.Tensor,
    scores: torch.Tensor,
    target_size: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Select visual-token indices with salience-coverage reduction."""
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    visual_tokens, was_unbatched = _as_batched_visual_tokens(visual_tokens)
    batch_size, num_tokens, _ = visual_tokens.shape
    keep_count = min(target_size, num_tokens)
    scores = _as_batched_scores(scores, batch_size, num_tokens).to(device=visual_tokens.device, dtype=visual_tokens.dtype)

    normalized_tokens = F.normalize(visual_tokens, p=2, dim=-1)
    selected_indices = torch.empty(batch_size, keep_count, device=visual_tokens.device, dtype=torch.long)
    selected_mask = torch.zeros(batch_size, num_tokens, device=visual_tokens.device, dtype=torch.bool)

    first_indices = scores.argmax(dim=-1)
    selected_indices[:, 0] = first_indices
    selected_mask.scatter_(1, first_indices.unsqueeze(1), True)

    first_tokens = normalized_tokens.gather(
        dim=1,
        index=first_indices.view(batch_size, 1, 1).expand(-1, 1, normalized_tokens.shape[-1]),
    )
    min_distances = 1.0 - torch.bmm(normalized_tokens, first_tokens.transpose(1, 2)).squeeze(-1)

    salience_weights = scores.clamp_min(0).pow(alpha)
    for select_pos in range(1, keep_count):
        combined_scores = min_distances * salience_weights
        combined_scores = combined_scores.masked_fill(selected_mask, -torch.inf)
        next_indices = combined_scores.argmax(dim=-1)

        selected_indices[:, select_pos] = next_indices
        selected_mask.scatter_(1, next_indices.unsqueeze(1), True)

        next_tokens = normalized_tokens.gather(
            dim=1,
            index=next_indices.view(batch_size, 1, 1).expand(-1, 1, normalized_tokens.shape[-1]),
        )
        new_distances = 1.0 - torch.bmm(normalized_tokens, next_tokens.transpose(1, 2)).squeeze(-1)
        min_distances = torch.minimum(min_distances, new_distances)

    if was_unbatched:
        return selected_indices.squeeze(0)
    return selected_indices


def hard_prune_by_keep_count(
    visual_tokens: torch.Tensor,
    scores: torch.Tensor,
    keep_count: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    keep_indices = select_visual_token_indices(visual_tokens, scores, keep_count, alpha=alpha)
    if visual_tokens.ndim == 2:
        return visual_tokens.gather(0, keep_indices.unsqueeze(-1).expand(-1, visual_tokens.shape[-1]))
    return visual_tokens.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, visual_tokens.shape[-1]))


def keep_count_from_ratio(num_tokens: int, keep_ratio: torch.Tensor | float, min_keep: int, max_keep: int | None = None) -> int:
    ratio_value = float(keep_ratio.detach().float().item()) if torch.is_tensor(keep_ratio) else float(keep_ratio)
    upper = num_tokens if max_keep is None else min(max_keep, num_tokens)
    keep_count = int(round(num_tokens * ratio_value))
    return max(min_keep, min(upper, keep_count))
