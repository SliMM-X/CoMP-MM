# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import ast
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
import deepspeed

from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import Dataset

IGNORE_INDEX=-100

from transformers import BatchFeature

from slimm.model.processor import SliMMQwen2VLProcessor
from slimm.model.slimm import SliMMForConditionalGeneration
from slimm.model.utils_vl import process_vision_info, smart_resize
from slimm.train.trainer import SliMMTrainer
from slimm.utils import rank0_print

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )

    # FIXME: unused params
    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    # args for visual encoder
    max_num_vistoken: Optional[int] = field(default=1024)
    min_num_vistoken: Optional[int] = field(default=4)
    use_rope: Optional[bool] = field(default=True)
    use_alignment_loss: Optional[bool] = field(default=False)
    custom_visual_model: Optional[str] = field(default=None)
    spatial_merge_size: Optional[int] = field(default=2)
    temporal_patch_size: Optional[int] = field(default=2)
    
    loss_weight: Optional[float] = field(default=0.05)
    word_dist_path: Optional[str] = field(default='checkpoints/word_dist.pth')
    embed_temp: Optional[float] = field(default=0.005)
    word_temp: Optional[float] = field(default=0.05)
    word_eps: Optional[float] = field(default=0.005)

    # args for deepstack
    use_deepstack: Optional[bool] = field(default=False)
    deepstack_type: Optional[str] = field(default='v1')
    # global_interval: Optional[int] = field(default=None)
    # local_splits_per_side: Optional[int] = field(default=4)

    # args for lmms
    disable_mrope: Optional[bool] = field(default=False)
    left_pad_training: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    # is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    # data augs (commonly used in object detection)
    use_scale_jitting: Optional[bool] = field(default=False)
    force_resize: Optional[int] = field(default=0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

    # args for lora tuning
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear

    lora_module_names = set()
    multimodal_keywords = ["visual", "aux_heads", "aux_teacher"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess_qwen2vl(data, image_folder, video_folder, data_args=None):
    use_scale_jitting= data_args.use_scale_jitting if data_args is not None else False
    force_resize = data_args.force_resize if data_args is not None else False

    messages = []
    roles = {"human": "user", "gpt": "assistant"}

    if isinstance(data, list):
        data = data[0]
    source = data['conversations']
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    image_list = data.get('image', [])
    # FIXME:
    if isinstance(image_list, str):
        image_list = [image_list]
    img_idx = 0

    for conv in source:
        try:
            role = conv["role"]
            content = conv["content"]
        except:
            role = conv["from"]
            content = conv["value"]

        role =  roles.get(role, role)
        content_list = []
        if '<image>' in content:
            for _ in range(content.count('<image>')):
                image = Image.open(os.path.join(image_folder, image_list[img_idx]))

                if force_resize != 0:
                    height, width = force_resize, force_resize
                    image = image.resize((height, width), Image.Resampling.BICUBIC)
                if use_scale_jitting:
                    random_scale = np.random.uniform(0.1, 2)
                    height, width = [int(x*random_scale) for x in image.size]
                    image = image.resize((height, width), Image.Resampling.BICUBIC)
                content_list.append({
                        "type": "image",
                        "image": image,
                    })
                img_idx +=1
            content = content.replace('<image>\n', '').replace('<image>', '')
        content_list.append( {
                    "type": "text",
                    "text": content,
                })
        messages.append({"role": role, "content": content_list})

    # # FIXME: check
    if img_idx == 0 and len(image_list):
        for mess in messages:
            if mess['role'] == 'user':
                mess['content'] = [{
                            "type": "image",
                            "image": os.path.join(image_folder, image_list[img_idx]),
                        }] + mess['content']
                break

    return messages


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, processor: transformers.AutoProcessor, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = copy.deepcopy(self.list_data_dict[i])
        image_folder = self.data_args.image_folder
        video_folder = self.data_args.video_folder
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        processor = self.processor

        messages = preprocess_qwen2vl(sources, image_folder, video_folder, data_args=self.data_args)

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        image_inputs, video_inputs = process_vision_info(messages,image_factor=self.processor.image_processor.patch_size*2)
        if self.data_args.deepstack_type in ('hires', 'lowres'):
            # hw_list = [x.size for x in images] if images is not None else None
            # FIXME: support videos
            if image_inputs is not None:
                if self.data_args.deepstack_type == 'hires':
                    image_inputs = [x.resize((max(14, x.size[0]//14*14), max(x.size[1]//14*14, 14))) for x in image_inputs]
                    image_inputs_deepstack = [x.resize((x.size[0]*2, x.size[1]*2)) for x in image_inputs]
                elif self.data_args.deepstack_type == 'lowres':
                    image_inputs_deepstack = [x.resize((x.size[0]//2*2, x.size[1]//2*2)) for x in image_inputs]
                    image_inputs = [x.resize((x.size[0]//2, x.size[1]//2)) for x in image_inputs_deepstack]
            else:
                image_inputs_deepstack = None

            if video_inputs is not None:
                assert NotImplementedError
            else:
                video_inputs_deepstack = None

            inputs_deepstack = processor_hires(text=None, images=image_inputs_deepstack, videos=video_inputs_deepstack)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            truncation=True
        )
        inputs_dict = {k:v for k,v in inputs.items()}

        if self.data_args.deepstack_type in ('hires', 'lowres'):
            if 'pixel_values' in inputs_deepstack:
                inputs_dict['pixel_values_deepstack'] = inputs_deepstack.pixel_values
                inputs_dict['image_grid_thw_deepstack'] = inputs_deepstack.image_grid_thw
            if 'pixel_values_video' in inputs_deepstack:
                raise NotImplementedError

        labels = inputs.input_ids.clone()[0]
        mes_start = self.tokenizer("<|im_start|>").input_ids[0]
        start_list = torch.where(labels==mes_start)[0]
        start_list = torch.cat([start_list, torch.tensor([len(labels)])])
        
        seq_list = start_list[1:] - start_list[:-1]
        sub_label_list = labels.split(seq_list.tolist())
        
        labels = []
        if len(sub_label_list) > len(messages):
            assert len(sub_label_list) == len(messages) + 1
            sub_label_list[0][:] = IGNORE_INDEX
            labels.append(sub_label_list[0])
            sub_label_list = sub_label_list[1:]
            
        for i, sub_label in enumerate(sub_label_list):
            if messages[i]['role'] == 'user':
                sub_label[:] = IGNORE_INDEX
            elif messages[i]['role'] == 'assistant':
                sub_label[:3] = IGNORE_INDEX
                sub_label[-1:] = IGNORE_INDEX
            else:
                raise NotImplementedError
            labels.append(sub_label)
        
        labels = torch.cat(labels)[None]
        
        inputs_dict['labels'] = labels
        inputs_dict['sources'] = sources[0]

        # return BatchFeature(data=inputs_dict, tensor_type='pt')
        return inputs_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, sources = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "sources"))

        input_ids = [_input_ids[0, :] for _input_ids in input_ids]
        labels = [_labels[0, :] for _labels in labels]

        pixel_values = [x['pixel_values'] for x in instances if 'pixel_values' in x]
        image_grid_thw = [x['image_grid_thw'] for x in instances if 'image_grid_thw' in x]

        if len(pixel_values) == 0:
            pixel_values, image_grid_thw = None, None
            pixel_values_deepstack, image_grid_thw_deepstack = None, None
        else:
            pixel_values = torch.cat(pixel_values)
            image_grid_thw = torch.cat(image_grid_thw)

            pixel_values_deepstack = [x['pixel_values_deepstack'] for x in instances if 'pixel_values_deepstack' in x]
            image_grid_thw_deepstack = [x['image_grid_thw_deepstack'] for x in instances if 'image_grid_thw_deepstack' in x]
            if len(pixel_values_deepstack):
                pixel_values_deepstack = torch.cat(pixel_values_deepstack)
                image_grid_thw_deepstack = torch.cat(image_grid_thw_deepstack)
            else:
                pixel_values_deepstack, image_grid_thw_deepstack = None, None

        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.

        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, 
                        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                        pixel_values_deepstack=pixel_values_deepstack, image_grid_thw_deepstack=image_grid_thw_deepstack,
                        sources=sources,
                        )

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, processor=processor, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    # import pdb; pdb.set_trace()
    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )

    print('overwrite_config', overwrite_config )

    customized_kwargs["config"] = cfg_pretrained
    customized_kwargs['use_rope'] = model_args.use_rope
    customized_kwargs['use_alignment_loss'] = model_args.use_alignment_loss
    customized_kwargs['loss_weight'] = model_args.loss_weight
    customized_kwargs['word_dist_path'] = model_args.word_dist_path
    customized_kwargs['embed_temp'] = model_args.embed_temp
    customized_kwargs['word_temp'] = model_args.word_temp
    customized_kwargs['word_eps'] = model_args.word_eps

    customized_kwargs['custom_visual_model'] = model_args.custom_visual_model
    customized_kwargs['use_deepstack'] = model_args.use_deepstack
    customized_kwargs['deepstack_type'] = model_args.deepstack_type

    customized_kwargs['spatial_merge_size'] = model_args.spatial_merge_size
    customized_kwargs['temporal_patch_size'] = model_args.temporal_patch_size

    if model_args.disable_mrope:
        customized_kwargs['use_mrope'] = False
    else:
        customized_kwargs['use_mrope'] = True

    model = SliMMForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs,
    )

    return model


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    data_args.deepstack_type = model_args.deepstack_type if model_args.use_deepstack else None

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

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
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

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

    # lora init
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    training_padding_side = 'left' if model_args.left_pad_training else 'right'

    custom_config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if custom_config.model_type == 'qwen2_vl':
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side=training_padding_side)
        processor = SliMMQwen2VLProcessor.from_pretrained(model_args.model_name_or_path, 
                                                           max_pixels=14 * 14 * 4 * model_args.max_num_vistoken, min_pixels=14 * 14 * 4 * model_args.min_num_vistoken,
                                                           deepstack_type=model_args.deepstack_type)
    else:
        processor_template = model_args.custom_visual_model
        tokenizer = transformers.AutoTokenizer.from_pretrained(processor_template, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side=training_padding_side)
        processor = SliMMQwen2VLProcessor.from_pretrained(processor_template, 
                                                           max_pixels=14 * 14 * 4 * model_args.max_num_vistoken, min_pixels=14 * 14 * 4 * model_args.min_num_vistoken,
                                                           tokenizer=tokenizer, deepstack_type=model_args.deepstack_type)
        if model_args.custom_visual_model is not None and 'video' in model_args.custom_visual_model:
            processor.image_processor.temporal_patch_size = model_args.temporal_patch_size
        elif model_args.custom_visual_model is not None and 'DFN' in model_args.custom_visual_model:
            processor.image_processor.temporal_patch_size = model_args.temporal_patch_size
        else:
            processor.image_processor.temporal_patch_size = 1

    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token

    # resize token embeddings if add new tokens
    model.resize_token_embeddings(len(tokenizer))

    if model_args.custom_visual_model is not None and 'patch16' in model_args.custom_visual_model:
        processor.image_processor.patch_size = 16

    if model_args.use_deepstack and model_args.deepstack_type.startswith('efficient'):
        processor.image_processor.patch_size = processor.image_processor.patch_size * 2
    
    # FIXME
    if model_args.use_deepstack and model_args.deepstack_type.startswith('efficient'):
        processor.image_processor.patch_size = processor.image_processor.patch_size * 2

    if model_args.mm_tunable_parts is not None:
        rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts

        # Set the entire model to not require gradients by default
        if not training_args.lora_enable:
            model.requires_grad_(False)

        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_mlp_adapter" in tunable_parts:
            for param in model.get_model('visual').merger.parameters():
                param.requires_grad = True

        if "mm_vision_tower" in tunable_parts:
            for name, param in model.get_model('visual').named_parameters():
                if "merger" not in name:
                    param.requires_grad_(True)

        if "mm_language_model" in tunable_parts:
            for name, param in model.get_model('model').named_parameters():
                param.requires_grad_(True)

        if "mm_aux_heads" in tunable_parts:
            for name, param in model.get_model('aux_heads').named_parameters():
                param.requires_grad_(True)

        if "mm_lm_head" in tunable_parts:
            for name, param in model.get_model('lm_head').named_parameters():
                param.requires_grad_(True)

        trainable_params, all_param = get_nb_trainable_parameters(model)

        rank0_print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, processor=processor, data_args=data_args)
    trainer = SliMMTrainer(model=model, processing_class=processor, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
