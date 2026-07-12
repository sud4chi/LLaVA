import importlib
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

from llava.model import *  # noqa: F403
from llava.model.dynamic_pruning import (
    DynamicPruningConfig,
    attach_dynamic_pruner,
    freeze_llava_train_pruner_only,
    save_dynamic_pruner,
)
from llava.train.llava_trainer import LLaVATrainer

base_train = importlib.import_module("llava.train.train")


@dataclass
class DynamicPruningTrainingArguments(base_train.TrainingArguments):
    dynamic_prune_enabled: bool = field(default=True)
    dynamic_prune_freeze_llava: bool = field(default=True)
    dynamic_prune_checkpoint: Optional[str] = field(default=None)
    dynamic_prune_input_type: str = field(default="scores")
    dynamic_prune_score_method: str = field(default="attention")
    dynamic_prune_hidden_size: int = field(default=128)
    dynamic_prune_min_keep: int = field(default=64)
    dynamic_prune_max_keep: Optional[int] = field(default=None)
    dynamic_prune_min_keep_ratio: float = field(default=0.05)
    dynamic_prune_max_keep_ratio: float = field(default=1.0)
    dynamic_prune_target_keep_ratio: float = field(default=0.25)
    dynamic_prune_alpha: float = field(default=1.0)
    dynamic_prune_temperature: float = field(default=0.1)
    dynamic_prune_budget_loss_weight: float = field(default=0.0)


class DynamicPruningTrainer(LLaVATrainer):
    def _save(self, output_dir=None, state_dict=None):
        if getattr(self.args, "dynamic_prune_enabled", False):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                save_dynamic_pruner(self.model, output_dir)
            return
        return super()._save(output_dir=output_dir, state_dict=state_dict)


def _build_model(model_args, training_args, compute_dtype, attn_implementation):
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    if model_args.vision_tower is not None:
        if "mpt" in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config["attn_impl"] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(  # noqa: F405
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args,
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(  # noqa: F405
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args,
            )
    else:
        raise ValueError("Dynamic visual pruning requires --vision_tower.")

    model.config.use_cache = False
    return model


def _build_tokenizer(model_args, training_args):
    if "mpt" in model_args.model_name_or_path:
        return transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )
    return transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )


def _configure_tokenizer_and_conversation(tokenizer, model, model_args):
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            base_train.smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in base_train.conversation_lib.conv_templates:
            base_train.conversation_lib.default_conversation = base_train.conversation_lib.conv_templates[model_args.version]
        else:
            base_train.conversation_lib.default_conversation = base_train.conversation_lib.conv_templates["vicuna_v1"]


def _configure_dynamic_pruner(model, training_args):
    config = DynamicPruningConfig(
        input_type=training_args.dynamic_prune_input_type,
        score_method=training_args.dynamic_prune_score_method,
        hidden_size=training_args.dynamic_prune_hidden_size,
        min_keep=training_args.dynamic_prune_min_keep,
        max_keep=training_args.dynamic_prune_max_keep,
        min_keep_ratio=training_args.dynamic_prune_min_keep_ratio,
        max_keep_ratio=training_args.dynamic_prune_max_keep_ratio,
        target_keep_ratio=training_args.dynamic_prune_target_keep_ratio,
        alpha=training_args.dynamic_prune_alpha,
        temperature=training_args.dynamic_prune_temperature,
        budget_loss_weight=training_args.dynamic_prune_budget_loss_weight,
    )
    attach_dynamic_pruner(model, config=config, checkpoint_path=training_args.dynamic_prune_checkpoint)
    if training_args.dynamic_prune_freeze_llava:
        freeze_llava_train_pruner_only(model)


def train(attn_implementation=None):
    parser = transformers.HfArgumentParser(
        (base_train.ModelArguments, base_train.DataArguments, DynamicPruningTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    base_train.local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model = _build_model(model_args, training_args, compute_dtype, attn_implementation)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        raise ValueError("LoRA is intentionally disabled for dynamic-pruner-only training.")

    tokenizer = _build_tokenizer(model_args, training_args)
    _configure_tokenizer_and_conversation(tokenizer, model, model_args)

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = False
    model.config.freeze_mm_mlp_adapter = True
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.dynamic_prune_enabled:
        _configure_dynamic_pruner(model, training_args)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and training_args.bf16:
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight") and training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    base_train.rank0_print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    data_module = base_train.make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = DynamicPruningTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        save_dynamic_pruner(model, training_args.output_dir)


if __name__ == "__main__":
    train()
