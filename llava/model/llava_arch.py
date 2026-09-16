#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        vision_tower = self.get_model().get_vision_tower()
        image_features = vision_tower(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_for_dynamic_pruning(self, images):
        """Run vision encoder/projector once and retain both feature spaces."""
        vision_tower = self.get_model().get_vision_tower()
        if hasattr(vision_tower, "forward_with_attention_scores"):
            ranking_features, attention_scores = vision_tower.forward_with_attention_scores(images)
        else:
            ranking_features = vision_tower(images)
            attention_scores = None
        projector = self.get_model().mm_projector
        try:
            projector_parameter = next(projector.parameters())
            projector_inputs = ranking_features.to(
                device=projector_parameter.device,
                dtype=projector_parameter.dtype,
            )
        except StopIteration:
            projector_inputs = ranking_features
        projected_features = projector(projector_inputs)
        return projected_features, ranking_features, attention_scores

    def apply_dynamic_pruning(self, image_features, ranking_features=None, importance_scores=None):
        dynamic_pruner = getattr(self.get_model(), "dynamic_pruner", None)
        if dynamic_pruner is None:
            return image_features

        if isinstance(image_features, list):
            pruned_features = []
            aux_values = []
            if ranking_features is None:
                ranking_features = image_features
            if importance_scores is None:
                importance_scores = [None] * len(image_features)
            for cur_features, cur_ranking_features, cur_scores in zip(
                image_features, ranking_features, importance_scores
            ):
                cur_pruned, cur_aux = dynamic_pruner(
                    cur_features,
                    ranking_tokens=cur_ranking_features,
                    scores=cur_scores,
                )
                pruned_features.append(cur_pruned)
                aux_values.append(cur_aux)
            self.dynamic_pruning_aux = aux_values
            return pruned_features

        if image_features.ndim == 3:
            ranking_features = image_features if ranking_features is None else ranking_features
            scores_per_sample = importance_scores
            if scores_per_sample is None:
                scores_per_sample = [None] * image_features.shape[0]
            outputs = []
            aux_values = []
            for cur_features, cur_ranking_features, cur_scores in zip(
                image_features, ranking_features, scores_per_sample
            ):
                cur_pruned, cur_aux = dynamic_pruner(
                    cur_features,
                    ranking_tokens=cur_ranking_features,
                    scores=cur_scores,
                )
                outputs.append(cur_pruned)
                aux_values.append(cur_aux)
            self.dynamic_pruning_aux = aux_values
            return outputs

        ranking_features = image_features if ranking_features is None else ranking_features
        pruned_features, aux = dynamic_pruner(
            image_features,
            ranking_tokens=ranking_features,
            scores=importance_scores,
        )
        self.dynamic_pruning_aux = aux
        return pruned_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            if getattr(self.get_model(), "dynamic_pruner", None) is not None:
                image_features, ranking_features, importance_scores = self.encode_images_for_dynamic_pruning(concat_images)
            else:
                image_features = self.encode_images(concat_images)
                ranking_features, importance_scores = None, None
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            if ranking_features is not None:
                ranking_features = torch.split(ranking_features, split_sizes, dim=0)
            if importance_scores is not None:
                importance_scores = torch.split(importance_scores, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
                if ranking_features is not None:
                    ranking_features = [x.flatten(0, 1) for x in ranking_features]
                if importance_scores is not None:
                    importance_scores = [x.flatten(0, 1) for x in importance_scores]
            elif mm_patch_merge_type.startswith('spatial'):
                if getattr(self.get_model(), "dynamic_pruner", None) is not None:
                    raise NotImplementedError("Dynamic pruning with CLIP attention scores currently supports mm_patch_merge_type='flat' only.")
                ranking_features = None
                importance_scores = None
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if getattr(self.get_model(), "dynamic_pruner", None) is not None:
                image_features, ranking_features, importance_scores = self.encode_images_for_dynamic_pruning(images)
            else:
                image_features = self.encode_images(images)
                ranking_features, importance_scores = None, None

        image_features = self.apply_dynamic_pruning(
            image_features,
            ranking_features=ranking_features,
            importance_scores=importance_scores,
        )

        return self._prepare_multimodal_embeddings_from_features(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            image_features=image_features,
        )

    def _prepare_multimodal_embeddings_from_features(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        image_features,
    ):
        """Insert precomputed image embeddings into text sequences."""

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_core_frontier_teacher_batch(
        self,
        input_ids,
        attention_mask,
        labels,
        images,
    ):
        """Build all core/candidate variants while encoding each image once."""
        dynamic_pruner = getattr(self.get_model(), "dynamic_pruner", None)
        if dynamic_pruner is None:
            raise ValueError("Core--Frontier teacher generation requires an attached dynamic_pruner")
        if labels is None:
            raise ValueError("Core--Frontier teacher generation requires answer labels")
        if not torch.is_tensor(images) or images.ndim != 4:
            raise NotImplementedError(
                "Core--Frontier utility training currently requires one equally-sized image per sample"
            )
        if images.shape[0] != input_ids.shape[0]:
            raise ValueError("image and text batch sizes must match")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Vision encoder and projector are frozen.  The output tensors are
        # reused by all 1 + |frontier| variants for a sample.
        with torch.no_grad():
            projected_features, ranking_features, importance_scores = self.encode_images_for_dynamic_pruning(images)

        expanded_input_ids = []
        expanded_attention_mask = []
        expanded_labels = []
        variant_image_features = []
        predicted_utilities = []
        core_variant_for_candidate = []
        candidate_variant_indices = []
        analyses = []

        for sample_index in range(input_ids.shape[0]):
            valid_ids = input_ids[sample_index][attention_mask[sample_index].bool()]
            if int((valid_ids == IMAGE_TOKEN_INDEX).sum().item()) != 1:
                raise NotImplementedError(
                    "Core--Frontier utility training currently supports exactly one image token per sample"
                )

            sample_scores = None if importance_scores is None else importance_scores[sample_index]
            analysis = dynamic_pruner.analyze_sample(
                projected_features[sample_index],
                ranking_features[sample_index],
                scores=sample_scores,
            )
            core_indices = analysis["core_indices"]
            frontier_indices = analysis["frontier_indices"]
            if frontier_indices.numel() == 0:
                raise ValueError(
                    "No frontier tokens are available; max_tokens must exceed min_tokens and the encoder must provide enough tokens"
                )

            core_variant_index = len(variant_image_features)
            core_spatial_indices = core_indices.sort().values
            variant_image_features.append(
                projected_features[sample_index].index_select(0, core_spatial_indices)
            )
            expanded_input_ids.append(input_ids[sample_index])
            expanded_attention_mask.append(attention_mask[sample_index])
            expanded_labels.append(labels[sample_index])

            for frontier_index in frontier_indices:
                selected_indices = torch.cat([core_indices, frontier_index.view(1)]).sort().values
                candidate_variant_indices.append(len(variant_image_features))
                core_variant_for_candidate.append(core_variant_index)
                variant_image_features.append(
                    projected_features[sample_index].index_select(0, selected_indices)
                )
                expanded_input_ids.append(input_ids[sample_index])
                expanded_attention_mask.append(attention_mask[sample_index])
                expanded_labels.append(labels[sample_index])

            predicted_utilities.append(analysis["predicted_utilities"])
            analyses.append(analysis)

        expanded_input_ids = torch.stack(expanded_input_ids)
        expanded_attention_mask = torch.stack(expanded_attention_mask)
        expanded_labels = torch.stack(expanded_labels)
        prepared = self._prepare_multimodal_embeddings_from_features(
            input_ids=expanded_input_ids,
            position_ids=None,
            attention_mask=expanded_attention_mask,
            past_key_values=None,
            labels=expanded_labels,
            image_features=variant_image_features,
        )
        return {
            "position_ids": prepared[1],
            "attention_mask": prepared[2],
            "inputs_embeds": prepared[4],
            "labels": prepared[5],
            "predicted_utilities": torch.cat(predicted_utilities),
            "core_variant_for_candidate": torch.tensor(
                core_variant_for_candidate,
                device=expanded_input_ids.device,
                dtype=torch.long,
            ),
            "candidate_variant_indices": torch.tensor(
                candidate_variant_indices,
                device=expanded_input_ids.device,
                dtype=torch.long,
            ),
            "analyses": analyses,
        }

    @torch.no_grad()
    def predict_core_frontier_utilities(self, images):
        """Run the calibration/inference feature path without invoking the LLM."""
        dynamic_pruner = getattr(self.get_model(), "dynamic_pruner", None)
        if dynamic_pruner is None:
            raise ValueError("threshold calibration requires an attached dynamic_pruner")
        if not torch.is_tensor(images) or images.ndim != 4:
            raise NotImplementedError("threshold calibration requires a batch of equally-sized images")

        projected_features, ranking_features, importance_scores = self.encode_images_for_dynamic_pruning(images)
        analyses = []
        for sample_index in range(images.shape[0]):
            sample_scores = None if importance_scores is None else importance_scores[sample_index]
            analyses.append(
                dynamic_pruner.analyze_sample(
                    projected_features[sample_index],
                    ranking_features[sample_index],
                    scores=sample_scores,
                )
            )
        return analyses

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
