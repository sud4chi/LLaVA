import importlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import DataLoader, Subset

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model import *  # noqa: F403
from llava.model.dynamic_pruning import (
    DynamicPruningConfig,
    attach_dynamic_pruner,
    calibrate_global_threshold,
    freeze_llava_train_pruner_only,
    save_dynamic_pruner,
)
from llava.train.llava_trainer import LLaVATrainer

base_train = importlib.import_module("llava.train.train")


@dataclass
class DynamicPruningDataArguments(base_train.DataArguments):
    validation_data_path: Optional[str] = field(default=None)
    validation_split_ratio: float = field(default=0.05)


@dataclass
class DynamicPruningTrainingArguments(base_train.TrainingArguments):
    dynamic_prune_enabled: bool = field(default=True)
    dynamic_prune_checkpoint: Optional[str] = field(default=None)
    score_method: str = field(default="attention")
    score_alpha: float = field(default=0.8)
    utility_hidden_size: int = field(default=128)
    utility_threshold: float = field(default=0.0)
    huber_delta: float = field(default=1.0)
    min_tokens: int = field(default=40)
    max_tokens: int = field(default=80)
    target_avg_tokens: float = field(default=64.0)


class _ImageOnlySubset(Subset):
    @property
    def lengths(self):
        return [self.dataset.lengths[index] for index in self.indices]

    @property
    def modality_lengths(self):
        return [self.dataset.modality_lengths[index] for index in self.indices]


def _image_only(dataset):
    indices = [index for index, sample in enumerate(dataset.list_data_dict) if "image" in sample]
    if not indices:
        raise ValueError("Core--Frontier utility training requires image samples")
    if len(indices) == len(dataset):
        return dataset
    return _ImageOnlySubset(dataset, indices)


class DynamicPruningTrainer(LLaVATrainer):
    def _unwrapped_model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self.model)
        return self.model.module if hasattr(self.model, "module") else self.model

    def compute_loss(self, model, inputs, return_outputs=False):
        # Trainer calls model.train(), which would enable dropout in the frozen
        # teacher.  Keep the teacher deterministic while leaving only the
        # utility predictor in train mode.
        model.eval()
        unwrapped = self.accelerator.unwrap_model(model) if hasattr(self, "accelerator") else model
        unwrapped.get_model().dynamic_pruner.train()
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def _save(self, output_dir=None, state_dict=None):
        if getattr(self.args, "dynamic_prune_enabled", False):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            if self.is_world_process_zero():
                save_dynamic_pruner(self._unwrapped_model(), output_dir)
            return
        return super()._save(output_dir=output_dir, state_dict=state_dict)

    @torch.no_grad()
    def calibrate_utility_threshold(self) -> dict:
        if self.eval_dataset is None:
            raise ValueError("A validation dataset is required to calibrate the global utility threshold")

        model = self._unwrapped_model()
        model.eval()
        pruner = model.get_model().dynamic_pruner
        distributed = dist.is_available() and dist.is_initialized()
        if distributed and dist.get_rank() != 0:
            result = [None]
            dist.broadcast_object_list(result, src=0)
            threshold, statistics = result[0]
            pruner.set_threshold(threshold)
            return statistics

        # Rank 0 deliberately iterates the unsharded dataset.  Distributed
        # evaluation samplers pad their shards, which would duplicate samples
        # and bias the requested dataset-wide average token count.
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        all_utilities = []
        all_core_counts = []
        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            image_mask = (inputs["input_ids"] == IMAGE_TOKEN_INDEX).any(dim=-1)
            if not image_mask.any():
                continue
            images = inputs.get("images")
            if not torch.is_tensor(images):
                raise NotImplementedError("Validation calibration requires equally-sized image tensors")
            analyses = model.predict_core_frontier_utilities(images[image_mask])
            for analysis in analyses:
                all_utilities.append(analysis["predicted_utilities"].float().cpu())
                all_core_counts.append(int(analysis["core_indices"].numel()))

        threshold, statistics = calibrate_global_threshold(
            all_utilities,
            all_core_counts,
            target_avg_tokens=pruner.config.target_avg_tokens,
        )
        pruner.set_threshold(threshold)
        statistics["threshold"] = threshold
        if distributed:
            dist.broadcast_object_list([(threshold, statistics)], src=0)
        self.log({f"calibration_{key}": value for key, value in statistics.items()})
        return statistics


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
            raise NotImplementedError(
                "Core--Frontier utility teacher generation is currently implemented for LLaVA-Llama models"
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
        score_method=training_args.score_method,
        hidden_size=training_args.utility_hidden_size,
        min_tokens=training_args.min_tokens,
        max_tokens=training_args.max_tokens,
        target_avg_tokens=training_args.target_avg_tokens,
        alpha=training_args.score_alpha,
        threshold=training_args.utility_threshold,
        huber_delta=training_args.huber_delta,
    )
    attach_dynamic_pruner(model, config=config, checkpoint_path=training_args.dynamic_prune_checkpoint)
    freeze_llava_train_pruner_only(model)


def _build_data_module(tokenizer, data_args, seed):
    train_dataset = _image_only(
        base_train.LazySupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args,
        )
    )
    if data_args.validation_data_path is not None:
        eval_dataset = _image_only(
            base_train.LazySupervisedDataset(
                tokenizer=tokenizer,
                data_path=data_args.validation_data_path,
                data_args=data_args,
            )
        )
    else:
        if not 0.0 < data_args.validation_split_ratio < 1.0:
            raise ValueError("validation_split_ratio must be in (0, 1) when validation_data_path is omitted")
        if len(train_dataset) < 2:
            raise ValueError("At least two samples are required for an automatic train/validation split")
        validation_size = max(1, int(round(len(train_dataset) * data_args.validation_split_ratio)))
        validation_size = min(validation_size, len(train_dataset) - 1)
        train_size = len(train_dataset) - validation_size
        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(seed)).tolist()
        eval_dataset = _ImageOnlySubset(train_dataset, indices[train_size:])
        train_dataset = _ImageOnlySubset(train_dataset, indices[:train_size])

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": base_train.DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    }


def train(attn_implementation=None):
    parser = transformers.HfArgumentParser(
        (base_train.ModelArguments, DynamicPruningDataArguments, DynamicPruningTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if not training_args.dynamic_prune_enabled:
        raise ValueError("train_dynamic_prune requires --dynamic_prune_enabled True")
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

    data_module = _build_data_module(tokenizer=tokenizer, data_args=data_args, seed=training_args.seed)
    trainer = DynamicPruningTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()
    calibration = trainer.calibrate_utility_threshold()
    base_train.rank0_print(
        "Calibrated global threshold "
        f"{calibration['threshold']:.6g}: average retained tokens "
        f"{calibration['achieved_avg_tokens']:.3f} "
        f"(target {calibration['target_avg_tokens']:.3f})"
    )

    model.config.use_cache = True
    if trainer.is_world_process_zero():
        save_dynamic_pruner(trainer._unwrapped_model(), training_args.output_dir)


if __name__ == "__main__":
    train()
