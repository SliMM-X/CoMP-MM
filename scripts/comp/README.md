# Training Scripts
We provide two scripts for reproduction about
(1) CoMP-MM-1B w/ SigLIP and Qwen2.5:
```Shell
bash scripts/comp/comp_1b_siglip.sh
```
and (2) CoMP-MM-1B w/ DINOv2 and Qwen2.5:
```Shell
bash scripts/comp/comp_1b_dinov2.sh
```
Alternatively, you can use our pre-trained CoMP-SigLIP (obtained by `bash scripts/comp/comp_1b_siglip.sh` ) to replace original SigLIP for more powerful and efficient training:
```Shell
bash scripts/comp/comp_1b_comp_siglip.sh
```

# Training Data
## Stage-I
```yaml
datasets:
  - json_path: data/LLaVA-Pretrain/blip_laion_cc_sbu_558k_mixin.json
    sampling_strategy: all
```
We use [BLIP558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/tree/main) (same as previous llava 1.5 series), and we modify the image path prefix to a mixin for compatibility with other datasets.

## Stage-II
```yaml
datasets:
  - json_path: data/LLaVA-OneVision-Mid-Data/synthdog_en_processed.json
    sampling_strategy: all
  - json_path: data/LLaVA-OneVision-Mid-Data/synthdog_zh_processed.json
    sampling_strategy: all
  - json_path: data/LLaVA-OneVision-Mid-Data/ureader_tr_processed.json
    sampling_strategy: all
  - json_path: data/cc3m_llava_recap/cc3m_llava_recap_2m_subset.json
    sampling_strategy: all
  - json_path: data/LLaVA-ReCap-118K/LLaVA-ReCap-118K_all.json
    sampling_strategy: all
  - json_path: data/LLaVA-ReCap-558K/LLaVA-ReCap-558K_all.json
    sampling_strategy: all
  - json_path: data/DenseFusion/DenseFusion-1M.json
    sampling_strategy: all
  # Due to the Alignment Loss, we removed all text-only training data.
  # - json_path: data/LLaVA-OneVision-Mid-Data/evol_instruct/evol_instruct_processed.json
  #   sampling_strategy: all
```
We use LLaVA-Mid-Stage data, including [
LLaVA-OneVision-Mid-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data) and [LLaVA-Recap-Data](https://huggingface.co/collections/lmms-lab/llava-next-6623288e2d61edba3ddbf5ff). To support high resolution inputs, we replace 1M data in CC3M with [Densefusion-1M](https://huggingface.co/datasets/BAAI/DenseFusion-1M).

## Stage-III
```yaml
datasets:
  # Due to the Alignment Loss, we removed all text-only training data.
  - json_path: data/LLaVA-NeXT-Data/llava_next_748k_mixin.json
    sampling_strategy: all
  - json_path: data/llava_ov_si_only_image.json
    sampling_strategy: all
```
We use LLaVA-OV-SI data, including [
LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) and images from [LLaVA-OneVision-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data).