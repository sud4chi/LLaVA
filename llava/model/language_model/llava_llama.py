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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..dynamic_pruning import core_frontier_utility_targets


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        dynamic_pruner = getattr(self.get_model(), "dynamic_pruner", None)
        utility_teacher_mode = (
            inputs_embeds is None
            and labels is not None
            and images is not None
            and dynamic_pruner is not None
            and dynamic_pruner.training
        )
        if utility_teacher_mode:
            teacher_batch = self.prepare_core_frontier_teacher_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images,
            )
            # All core-only and core+one-frontier variants are packed into this
            # single LLM invocation.  The frozen teacher never builds a graph.
            with torch.no_grad():
                teacher_outputs = self.model(
                    input_ids=None,
                    attention_mask=teacher_batch["attention_mask"],
                    position_ids=teacher_batch["position_ids"],
                    past_key_values=None,
                    inputs_embeds=teacher_batch["inputs_embeds"],
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    cache_position=cache_position,
                )
                shift_labels = teacher_batch["labels"][:, 1:]
                valid_labels = shift_labels.ne(-100)
                answer_token_counts = valid_labels.sum(dim=-1)
                if torch.any(answer_token_counts == 0):
                    raise ValueError("Every utility teacher variant must contain at least one answer label")

                # Project only answer-predicting hidden states through the LM
                # head.  This is numerically equivalent to materializing full
                # [variant, sequence, vocabulary] logits, but avoids a very
                # large tensor for ignored prompt/image positions.
                answer_hidden = teacher_outputs.last_hidden_state[:, :-1][valid_labels]
                answer_targets = shift_labels[valid_labels]
                if self.config.pretraining_tp > 1:
                    lm_head_slices = self.lm_head.weight.split(
                        self.vocab_size // self.config.pretraining_tp,
                        dim=0,
                    )
                    answer_logits = torch.cat(
                        [F.linear(answer_hidden, weight) for weight in lm_head_slices],
                        dim=-1,
                    )
                else:
                    answer_logits = self.lm_head(answer_hidden)
                answer_ce = F.cross_entropy(answer_logits.float(), answer_targets, reduction="none")
                variant_indices = torch.where(valid_labels)[0]
                variant_ce_sums = answer_ce.new_zeros(shift_labels.shape[0])
                variant_ce_sums.scatter_add_(0, variant_indices, answer_ce)
                variant_mean_ce = variant_ce_sums / answer_token_counts
                utility_targets = core_frontier_utility_targets(
                    variant_mean_ce,
                    teacher_batch["core_variant_for_candidate"],
                    teacher_batch["candidate_variant_indices"],
                )

            utility_loss = dynamic_pruner.utility_loss(
                teacher_batch["predicted_utilities"],
                utility_targets,
            )
            if return_dict is False:
                return (utility_loss,)
            return CausalLMOutputWithPast(loss=utility_loss, logits=None)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
