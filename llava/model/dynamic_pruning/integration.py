from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from .threshold_module import DynamicPruningConfig, DynamicVisionTokenPruner


def _model_body(model):
    return model.get_model() if hasattr(model, "get_model") else model


def attach_dynamic_pruner(
    model,
    config: DynamicPruningConfig | dict | None = None,
    checkpoint_path: str | None = None,
) -> DynamicVisionTokenPruner:
    body = _model_body(model)
    if not isinstance(config, DynamicPruningConfig):
        config = DynamicPruningConfig.from_dict(config or {})

    vision_hidden_size = getattr(model.config, "hidden_size", None)
    if vision_hidden_size is None:
        vision_hidden_size = getattr(body, "config").hidden_size

    pruner = DynamicVisionTokenPruner(vision_hidden_size=vision_hidden_size, config=config)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        pruner.load_state_dict(state)

    try:
        first_param = next(body.parameters())
        pruner.to(device=first_param.device)
    except StopIteration:
        pass

    body.dynamic_pruner = pruner
    model.config.use_dynamic_pruning = True
    model.config.dynamic_pruning_config = config.to_dict()
    return pruner


def freeze_llava_train_pruner_only(model) -> None:
    model.requires_grad_(False)
    body = _model_body(model)
    if not hasattr(body, "dynamic_pruner"):
        raise ValueError("dynamic_pruner is not attached to the model")
    body.dynamic_pruner.requires_grad_(True)
    body.dynamic_pruner.train()


def save_dynamic_pruner(model, output_dir: str) -> None:
    body = _model_body(model)
    if not hasattr(body, "dynamic_pruner"):
        raise ValueError("dynamic_pruner is not attached to the model")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pruner = body.dynamic_pruner
    torch.save({"state_dict": pruner.state_dict()}, output_path / "dynamic_pruner.bin")
    with open(output_path / "dynamic_pruner_config.json", "w", encoding="utf-8") as f:
        json.dump(pruner.config.to_dict(), f, indent=2, sort_keys=True)
    model.config.save_pretrained(output_path)


def load_dynamic_pruner(model, checkpoint_dir_or_file: str) -> DynamicVisionTokenPruner:
    path = Path(checkpoint_dir_or_file)
    if path.is_dir():
        config_path = path / "dynamic_pruner_config.json"
        weight_path = path / "dynamic_pruner.bin"
    else:
        config_path = path.with_name("dynamic_pruner_config.json")
        weight_path = path

    config = {}
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    return attach_dynamic_pruner(model, config=config, checkpoint_path=os.fspath(weight_path))
