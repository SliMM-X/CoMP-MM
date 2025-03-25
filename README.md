<div align="center">
<h2><font size=3>CoMP: Continual Multimodal Pre-training for Vision Foundation Models</h2>
<h4>

Yitong Chen<sup>1,2*</sup>, [Lingchen Meng](https://menglcool.github.io/)<sup>1*</sup>, [Wujian Peng](https://scholar.google.com/citations?user=GTuWk9YAAAAJ&hl)<sup>1,2</sup>, [Zuxuan Wu](https://zxwu.azurewebsites.net/)<sup>1,2&dagger;</sup>, [Yu-Gang Jiang](https://scholar.google.com/citations?user=f3_FP8AAAAAJ&hl=en)<sup>1</sup>

<sup>1</sup> Shanghai Key Lab of Intell. Info. Processing, School of CS, Fudan University <br>
<sup>2</sup> Shanghai Innovation Institute

 <sup>*</sup> Equal contributions; <sup>&dagger;</sup> Corresponding author.

[[`Paper`](https://arxiv.org/abs/2503.18931)] 
[[`Website`](https://slimm-x.github.io/comp/)] 
[[`Model`](https://huggingface.co/SliMM-X)] 
</div>

## Installation

1. Clone this repository and navigate to CoMP-SliMM folder
```bash
git https://github.com/SliMM-X/CoMP-MM.git
cd CoMP-MM
```

2. Install Package
```Shell
conda create -n comp-slimm python=3.10 -y
conda activate comp-slimm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# additional packages for training cases
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Quick Start With HuggingFace
Example Code of CoMP-VFMs:
```Python
import torch
from slimm.model.processor import SliMMQwen2VLProcessor
from slimm.model.utils_vl import process_vision_info
from slimm.model.vision_encoder import CoMPSiglipVisionModel, CoMPDinov2Model
from PIL import Image

model_path = "SliMM-X/CoMP-SigLIP-So400M"
# model_path = "SliMM-X/CoMP-DINOv2-Large"

model = CoMPSiglipVisionModel.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda", w_merger=False
).to(torch.bfloat16)

# model = CoMPDinov2Model.from_pretrained(
#     model_path, torch_dtype="auto", device_map="cuda", w_merger=False
# ).to(torch.bfloat16)

processor = SliMMQwen2VLProcessor.from_pretrained(model_path)

image_input = Image.open("https://slimm-x.github.io/comp/figs/teaser.png")
inputs = processor(
    images=image_input,
    return_tensors="pt",
)

inputs = inputs.to("cuda")
output_feat = model(inputs.pixel_values.to(torch.bfloat16), inputs.image_grid_thw)
print(output_feat)
```
</details>

Example Code of CoMP-MM:
```Python
# this is very similar to qwen2-vl
from slimm.model.processor import SliMMQwen2VLProcessor
from slimm.model.slimm import SliMMForConditionalGeneration
from slimm.model.utils_vl import process_vision_info

model_path = "SliMM-X/CoMP-MM-1B"

model = SliMMForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda"
)
processor = SliMMQwen2VLProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://slimm-x.github.io/comp/figs/teaser.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
</details>

## Train
We provide two scripts for reproduction about
(1) CoMP-MM-1B w/ SigLIP:
```Shell
bash scripts/comp/comp_1b_siglip.sh
```
and (2) CoMP-MM-1B w/ DINOv2:
```Shell
bash scripts/comp/comp_1b_dinov2.sh
```
For data preparation, please refer to [scripts/comp/README.md](scripts/comp/README.md).
## Evaluation
We provide an evaluation script for multimodal understanding based on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) locally. First, you need to install lmms-eval:
```Shell
cd lmms-eval
pip install -e .
cd ..
```
And then, run:
```Shell
bash scripts/comp/eval.sh
```
## 🔗 Citation
If you find our work helpful, please consider citing our paper :paperclip: and starring our repo :star2: :

```
@article{comp2025,
      title={CoMP: Continual Multimodal Pre-training for Vision Foundation Models}, 
      author={Yitong Chen and Lingchen Meng and Wujian Peng and Zuxuan Wu and Yu-Gang Jiang},
      year={2025},
      journal={arXiv preprint arXiv:2503.18931}, 
}
```

## Acknowledgement
Our work is built upon [SliMM](https://github.com/MengLcool/SliMM), [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), [LLaVA](https://github.com/haotian-liu/LLaVA) and [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).

Feel free to contribute and reach out if you have any questions!
