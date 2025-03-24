import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional, List, Union, Tuple
import transformers
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLModel

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from transformers.cache_utils import Cache, SlidingWindowCache, StaticCache
from transformers.image_utils import ImageInput, VideoInput
from transformers import AutoConfig, AutoModel

from .vision_encoder import VISION_TRANSFORMER_CLASSES
from .language_model import LANGUAGE_MODEL_CLASSES
from .utils_vl import process_vision_info

from transformers.utils import logging

import random

logger = logging.get_logger(__name__)

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class SliMMForConditionalGeneration(Qwen2VLForConditionalGeneration):
    # replace qwen2vl with custom slimm
    def __init__(self, config,
                    loss_weight=0.05,
                    use_alignment_loss=False,
                    word_dist_path='checkpoints/word_dist.pth',
                    embed_temp=0.005,
                    word_temp=0.05,
                    word_eps=0.005,
                    use_rope=True,
                    use_native_attn=False,
                    custom_visual_model=None,
                    use_deepstack=False,
                    visual_special_tokens_dict=None,
                    spatial_merge_size=2,
                    temporal_patch_size=1,
                    use_mrope=True, # inconsistent with previous ckpts
                    deepstack_type='v1',
                    local_features_interval=1,
                    local_features_start=1,
                    global_interval=None,
                    local_splits_per_side=4,
                    **kwargs
                ):

        self.use_deepstack = getattr(config, 'use_deepstack', use_deepstack)
        self.deepstack_type = getattr(config, 'deepstack_type', deepstack_type)
        self.spatial_merge_size = getattr(config, 'spatial_merge_size', spatial_merge_size)
        self.temporal_patch_size = getattr(config, 'temporal_patch_size', temporal_patch_size)

        self.local_features_start = getattr(config, 'local_features_start', local_features_start)
        self.local_features_interval = getattr(config, 'local_features_interval', local_features_interval)

        self.global_interval = getattr(config, 'global_interval', global_interval)
        self.local_splits_per_side = getattr(config, 'local_splits_per_side', local_splits_per_side)

        self.use_alignment_loss = use_alignment_loss
        self.use_rope = use_rope
        self.loss_weight = loss_weight
        self.embed_temp = embed_temp
        self.word_temp = word_temp
        self.word_eps = word_eps
        self.word_dist_path = word_dist_path

        config.use_deepstack = self.use_deepstack
        config.deepstack_type = self.deepstack_type
        config.global_interval = self.global_interval
        config.use_rope = self.use_rope


        if self.global_interval:
            config.local_splits_per_side = self.local_splits_per_side

        if not config.model_type == 'qwen2_vl':
            rope_scaling = {'type': 'default', 'mrope_section': [16, 24, 24], 'rope_type': 'default'}
            if sum(rope_scaling['mrope_section']) * 2 > config.hidden_size // config.num_attention_heads:
                rope_scaling['mrope_section'] = [8 ,12, 12]

            if visual_special_tokens_dict is None:
                visual_special_tokens_dict = {
                    "vision_start_token_id": 151652,
                    "vision_end_token_id": 151653,
                    "vision_token_id": 151654,
                    "image_token_id": 151655,
                    "video_token_id": 151656,
                }

            # update special tokens not exists in previous config
            visual_special_tokens_dict = {k:v for k,v in visual_special_tokens_dict.items() if not hasattr(config, k)}
            config.update(visual_special_tokens_dict)
            config.rope_scaling = rope_scaling

        super(Qwen2VLForConditionalGeneration, self).__init__(config)

        if self.use_alignment_loss:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                config._name_or_path,
                model_max_length=2048,
                padding_side="right",
                use_fast=False,
            )
            vocab_dist = torch.load(self.word_dist_path)
            vocab_dist = torch.FloatTensor(vocab_dist).unsqueeze(1)
            vocab_dist = vocab_dist.clamp(min=1) / vocab_dist.sum(dim=0, keepdim=True)
            self.register_buffer('vocab_dist', vocab_dist)

        use_mrope = getattr(config, 'use_mrope', use_mrope)

        self.use_mrope = config.use_mrope = use_mrope

        self.model = LANGUAGE_MODEL_CLASSES[config.model_type](config)

        if custom_visual_model is None:
            custom_visual_model = getattr(config, 'custom_visual_model', None)

        if custom_visual_model is not None:
            vision_config = AutoConfig.from_pretrained(custom_visual_model)
            config.custom_visual_model = custom_visual_model
        else:
            vision_config = config.vision_config

        # update config for vision encoder
        vision_config.llm_hidden_size = config.hidden_size
        vision_config.use_deepstack = self.use_deepstack
        vision_config.spatial_merge_size = self.spatial_merge_size
        vision_config.temporal_patch_size = self.temporal_patch_size
        vision_config.deepstack_type = self.deepstack_type
        vision_config.global_interval = self.global_interval

        # comp-mm
        vision_config.use_rope = self.use_rope

        if self.global_interval is not None:
            vision_config.local_splits_per_side= self.local_splits_per_side
        
        if custom_visual_model is not None:
            self.visual = VISION_TRANSFORMER_CLASSES[vision_config.model_type].from_pretrained(custom_visual_model, attn_implementation=config._attn_implementation, config=vision_config)
        else:
            self.visual = VISION_TRANSFORMER_CLASSES[vision_config.model_type]._from_config(
                vision_config, attn_implementation=config._attn_implementation
            )

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self, name=None):
        if name is None:
            return self
        sub_model_list = name.split('.')
        model = self
        for sub_model in sub_model_list:
            model = getattr(model, sub_model)
        return model

    @property
    def patch_size(self):
        # FIXME: need to support other vision encoders
        return self.visual.patch_embed.patch_size

    def process_visual_embeddings(self, pixel_values, image_grid_thw, image_mask):
        image_grid_thw_visual = image_grid_thw.clone()
        image_embeds_deepstack = None

        if self.use_deepstack and self.deepstack_type.startswith('efficient'):
            image_grid_thw_visual[:,1:] = image_grid_thw_visual[:,1:] * 2
            patch_size = self.patch_size
            h_stack, w_stack = 2, 2
            seq_lens = image_grid_thw[:, 1] * image_grid_thw[:, 2]
            pixel_values = pixel_values.split(seq_lens.tolist())
            merge_size = self.spatial_merge_size
            pixel_values = [x.view(t, h//merge_size, w//merge_size, merge_size, merge_size, -1).permute(0,1,3,2,4,5).flatten(0,4).contiguous() for x, (t,h,w) in zip(pixel_values, image_grid_thw)]
            pixel_values = [x.view(t, h, w, x.shape[1]//(patch_size*patch_size*h_stack*w_stack), h_stack, patch_size, w_stack, patch_size).permute(0,1,2,4,6,3,5,7).flatten(0,4).flatten(1) \
                                        for x, (t,h,w) in zip(pixel_values, image_grid_thw)]

            pixel_values = torch.cat(pixel_values)

        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw_visual)

        if self.use_deepstack:
            image_embeds, image_embeds_deepstack = image_embeds
            l, d = image_embeds.shape
            image_embeds_deepstack = image_embeds_deepstack.view(l, -1, d).permute(1,0,2).contiguous()

        return image_embeds, image_embeds_deepstack

    def get_prompt(self, sources):
        prompts = []
        for sou in sources:
            sou = sou['conversations']
            answers = [x['value'] for x in sou if x['from'] == 'gpt']
            random.shuffle(answers)
            prompt = ' '.join(answers)
            prompts.append(prompt)
        return prompts

    def encode_prompt(self, prompts, language_condition_select_layer=None):
        text_inputs = self.get_model().tokenizer(
            prompts,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if language_condition_select_layer is None:
            language_condition_select_layer = self.get_vision_tower().language_condition_select_layer

        dtype, device = self.dtype, self.device
        input_ids = text_inputs.input_ids.to(device=device)
        attention_mask=text_inputs.attention_mask.to(device=device)
        with torch.no_grad():
            use_the_first_n_layer = language_condition_select_layer if isinstance(language_condition_select_layer, int) else None
            language_output = self.get_model()(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        if language_condition_select_layer == 'all':
            return language_output.hidden_states, attention_mask
        text_condition = language_output.hidden_states[language_condition_select_layer]
        return text_condition, attention_mask

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
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        pixel_values_deepstack: Optional[torch.Tensor] = None,
        image_grid_thw_deepstack: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sources=None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embeds_deepstack = []
        mask_deepstack = []

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                image_mask = input_ids == self.config.image_token_id
                pixel_values = pixel_values.type(self.visual.get_dtype())
                
                image_grid_thw_visual = image_grid_thw.clone()

                image_embeds, image_embeds_deepstack = self.process_visual_embeddings(pixel_values, image_grid_thw, image_mask)
                if image_embeds_deepstack is not None:
                    embeds_deepstack.append(image_embeds_deepstack)
                    mask_deepstack.append(image_mask)

                image_embeds = image_embeds.to(inputs_embeds.device)
                if self.training:
                    inputs_embeds = inputs_embeds.clone()

                inputs_embeds[image_mask] = image_embeds

            if pixel_values_videos is not None:
                video_mask = input_ids == self.config.video_token_id
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds, video_embeds_deepstack = self.process_visual_embeddings(pixel_values_videos, video_grid_thw, video_mask)
                inputs_embeds[video_mask] = video_embeds
                if video_embeds_deepstack is not None:
                    embeds_deepstack.append(video_embeds_deepstack)
                    mask_deepstack.append(video_mask)

            # process pure text in mm training
            if pixel_values is None and pixel_values_videos is None and self.training:
                pixel_values_dummy = inputs_embeds.new_zeros([16, 14*14*1*3])
                image_grid_thw_dummy = torch.tensor([[1,4,4]]).to(inputs_embeds.device)
                image_embeds_dummy = self.visual(pixel_values_dummy, grid_thw=image_grid_thw_dummy)
                if self.use_deepstack:
                    image_embeds_dummy, embeds_deepstack = image_embeds_dummy
                    # FIXME
                    l, d = image_embeds_dummy.shape
                    embeds_deepstack = embeds_deepstack.view(l, -1, d).permute(1,0,2).contiguous()
                    mask_deepstack = input_ids == self.config.image_token_id # all-zero matrix
                    embeds_deepstack = embeds_deepstack[:,:0]

                inputs_embeds = torch.cat([inputs_embeds, image_embeds_dummy[:0][None].expand(inputs_embeds.shape[0], -1, -1)], dim=1)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)


        if self.use_mrope and position_ids is None and input_ids is not None:
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)

        if isinstance(embeds_deepstack, list) and len(embeds_deepstack):
            # FIXME: tmp implent
            embeds_deepstack = torch.cat(embeds_deepstack)
            mask_deepstack, _ = torch.stack(mask_deepstack, dim=0).max(0)

        kwargs_deepstack = {}
        if self.use_deepstack:
            kwargs_deepstack=dict(
                image_embeds_deepstack=embeds_deepstack,
                image_mask_deepstack=mask_deepstack,
                local_features_interval=self.local_features_interval,
                local_features_start=self.local_features_start
            )

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs_deepstack
        )

        hidden_states = outputs[0]

        # to avoid oom caused by the large vocab
        if labels is not None and self.training:
            valid_mask = labels[..., 1:] != -100

            def efficient_loss(hidden_states, valid_mask, labels):
                shift_logits = self.lm_head(hidden_states[:,:-1][valid_mask].contiguous())
                shift_logits = shift_logits.view(-1, self.config.vocab_size).float()
                logits = shift_logits # dummy logits
                shift_labels = labels[..., 1:][valid_mask].contiguous()
                shift_labels = shift_labels.to(shift_logits.device)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)
                return loss, logits

            def efficient_klloss(hidden_states, valid_mask, target_logprob, aux_head=None):
                hidden_states = hidden_states[:,:-1][valid_mask].contiguous()
                if aux_head is not None:
                    hidden_states = aux_head(hidden_states)
                shift_logits = self.lm_head(hidden_states)
                shift_logits = shift_logits.view(-1, self.config.vocab_size).float()
                logits_logprob = shift_logits.log_softmax(-1) # dummy logits

                loss_fct = nn.KLDivLoss(reduction='batchmean', log_target=True)
                loss = loss_fct(logits_logprob, target_logprob)
                return loss, logits_logprob.detach()

            def alignment_loss(hidden_states, sources, image_embeds):
                @torch.no_grad()
                def sinkhorn(out, vocab_dist, sinkhorn_iterations=3, epsilon=self.word_eps, temperature=self.word_temp):
                    Q = torch.exp(out / epsilon).t()
                    B = Q.shape[1]
                    K = Q.shape[0] 
                    vocab_dist_prob = (vocab_dist / temperature).softmax(dim=0)
                    for it in range(sinkhorn_iterations):
                        Q *= vocab_dist_prob
                        Q /= torch.sum(Q, dim=0, keepdim=True)
                        Q /= B
                    Q *= B # the colomns must sum to 1 so that Q is an assignment Q: K x B
                    return Q.t()    # B x K

                prompts = self.get_prompt(sources)
                language_embed, attn_mask = self.encode_prompt(prompts, -1)
                language_embed = (language_embed * attn_mask[...,None]).sum(1) / attn_mask[...,None].sum(1).clamp(min=1)
                caption_features = nn.functional.normalize(language_embed, dim=-1, p=2).to(torch.bfloat16)

                if pixel_values is None: # pure text
                    image_features = torch.zeros([caption_features.shape[0], hidden_states.shape[-1]], dtype=torch.bfloat16).to(hidden_states.device)
                else:
                    cu_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
                    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]) // 4

                    image_embeds_split = image_embeds.split(seqlens.tolist())
                    image_features_pad = torch.nn.utils.rnn.pad_sequence(image_embeds_split, batch_first=True, padding_value=0)

                    image_mask = image_features_pad.ne(-1)
                    image_features = (image_features_pad * image_mask).sum(1) / image_mask.sum(1).clamp(min=1)
                    image_features = nn.functional.normalize(image_features, dim=-1, p=2)

                prototypes = nn.functional.normalize(self.lm_head.weight, dim=-1, p=2)
                code_v = torch.mm(image_features.float(), prototypes.float().detach().t())
                code_t = torch.mm(caption_features.float(), prototypes.float().detach().t())

                vocab_dist = self.vocab_dist.to(code_v)
                q_t = sinkhorn(code_t, vocab_dist)

                return -torch.mean(torch.sum(q_t * F.log_softmax(code_v.float() / self.embed_temp, dim=-1), dim=-1))

            loss, logits = efficient_loss(hidden_states, valid_mask, labels)

            if self.use_alignment_loss:
                loss_alg = alignment_loss(hidden_states, sources, image_embeds)
                loss = loss + self.loss_weight * loss_alg

        else:
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # modify how to fetch spatial_merge_size
        spatial_merge_size = self.config.vision_config.spatial_merge_size if hasattr(self.config, 'vision_config') \
                                else self.spatial_merge_size

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        model_inputs = {}
        if attention_mask is not None and position_ids is None:
            if not self.use_mrope:
                if self._supports_cache_class:
                    model_inputs["cache_position"] = cache_position
                # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
                #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
                #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
                elif cache_position is None:
                    past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
                    cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

                position_ids=None
            else:
                if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids, image_grid_thw, video_grid_thw, attention_mask
                    )
                else:
                    batch_size, seq_length = input_ids.shape
                    delta = (
                        cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                    )
                    position_ids = torch.arange(seq_length, device=input_ids.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs.update({"inputs_embeds": inputs_embeds, "input_ids": None})
        else:
            model_inputs.update({"input_ids": input_ids, "inputs_embeds": None})

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs
