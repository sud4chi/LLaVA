from __future__ import annotations

import torch
import torch.nn.functional as F


def _as_batched_tokens(visual_tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if visual_tokens.ndim == 2:
        return visual_tokens.unsqueeze(0), True
    if visual_tokens.ndim == 3:
        return visual_tokens, False
    raise ValueError(f"visual_tokens must be [tokens, dim] or [batch, tokens, dim], got {tuple(visual_tokens.shape)}")


def _normalize_scores(scores: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scores = scores.float()
    scores = scores - scores.amin(dim=-1, keepdim=True)
    denom = scores.amax(dim=-1, keepdim=True).clamp_min(eps)
    return scores / denom


def compute_d_prune_importance(
    visual_tokens: torch.Tensor,
    cls_attention_scores: torch.Tensor | None = None,
    method: str = "norm_centrality",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-token importance for dynamic D-prune-style pruning.

    If attention scores are supplied, they can be used directly. Otherwise this
    falls back to token-only proxies that are available from LLaVA image features.
    """
    tokens, was_unbatched = _as_batched_tokens(visual_tokens)

    if cls_attention_scores is not None and method == "attention":
        scores = cls_attention_scores
        if scores.ndim == 1:
            scores = scores.unsqueeze(0)
        elif scores.ndim == 3:
            scores = scores.mean(dim=1)
        elif scores.ndim == 4 and scores.shape[-2] == 1:
            scores = scores.squeeze(-2).mean(dim=1)
        if scores.ndim != 2:
            raise ValueError(f"Unsupported cls_attention_scores shape: {tuple(scores.shape)}")
        scores = scores.to(device=tokens.device)
    else:
        normalized = F.normalize(tokens.float(), p=2, dim=-1)
        token_norm = tokens.float().norm(p=2, dim=-1)

        if method == "norm":
            scores = token_norm
        elif method == "centrality":
            centroid = F.normalize(normalized.mean(dim=1, keepdim=True), p=2, dim=-1)
            scores = torch.matmul(normalized, centroid.transpose(1, 2)).squeeze(-1)
        elif method == "norm_centrality":
            centroid = F.normalize(normalized.mean(dim=1, keepdim=True), p=2, dim=-1)
            centrality = torch.matmul(normalized, centroid.transpose(1, 2)).squeeze(-1)
            scores = _normalize_scores(token_norm, eps=eps) * _normalize_scores(centrality, eps=eps)
        else:
            raise ValueError(f"Unknown D-prune importance method: {method}")

    scores = _normalize_scores(scores, eps=eps).to(device=tokens.device, dtype=tokens.dtype)
    if was_unbatched:
        return scores.squeeze(0)
    return scores
