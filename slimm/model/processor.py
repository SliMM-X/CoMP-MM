from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor, Qwen2VLProcessorKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.image_utils import ImageInput, VideoInput
from transformers.feature_extraction_utils import BatchFeature
from typing import Optional, List, Union, Tuple

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


class SliMMQwen2VLProcessor(Qwen2VLProcessor):
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, deepstack_type=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.deepstack_type = deepstack_type

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        if text is None:
            return BatchFeature(data={**image_inputs, **videos_inputs})

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2

            index = 0
            for i in range(len(text)):
                while "<|image_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|image_pad|>", "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while "<|video_pad|>" in text[i]:
                    text[i] = text[i].replace(
                        "<|video_pad|>", "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", "<|video_pad|>")

        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})
