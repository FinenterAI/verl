# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for tokenization."""

import types
import warnings

__all__ = [
    "hf_tokenizer",
    "hf_processor",
    "normalize_token_ids",
    "build_multimodal_processor_inputs",
    "get_processor_token_id",
]


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


def get_processor_token_id(processor, token_name: str) -> int | None:
    """Resolve a multimodal special token id from a processor.

    Newer processors may expose ``image_token``/``video_token`` strings instead
    of ``image_token_id``/``video_token_id`` integers. Fall back to tokenizer
    conversion so rollout code can stay processor-agnostic.
    """

    if processor is None:
        return None

    token_id_attr = f"{token_name}_token_id"
    token_id = getattr(processor, token_id_attr, None)
    if token_id is not None:
        return int(token_id)

    token_attr = f"{token_name}_token"
    token = getattr(processor, token_attr, None)
    tokenizer = getattr(processor, "tokenizer", None)
    if token is not None and tokenizer is not None:
        converted = tokenizer.convert_tokens_to_ids(token)
        if converted is not None:
            return int(converted)

    return None


def _split_videos_and_metadata(videos):
    if videos is None:
        return None, None
    videos = list(videos)
    if len(videos) > 0 and isinstance(videos[0], tuple):
        video_values, video_metadata = zip(*videos, strict=False)
        return list(video_values), list(video_metadata)
    return videos, None


def build_multimodal_processor_inputs(
    processor,
    *,
    text,
    images=None,
    videos=None,
    audio=None,
    mm_processor_kwargs=None,
    return_tensors: str = "pt",
):
    """Build kwargs for multimodal processor calls.

    This keeps the existing VL flow intact while extending it with audio-aware
    paths for processors that accept audio inputs.
    """
    processor_kwargs = dict(mm_processor_kwargs or {})
    if audio is not None and "sampling_rate" not in processor_kwargs:
        sampling_rate = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
        if sampling_rate is None:
            # MiniCPM-o Whisper feature extractor
            sampling_rate = getattr(getattr(processor, "audio_feature_extractor", None), "sampling_rate", None)
        if sampling_rate is not None:
            processor_kwargs["sampling_rate"] = int(sampling_rate)

    videos, video_metadata = _split_videos_and_metadata(videos)
    processor_kwargs.setdefault("return_tensors", return_tensors)

    if video_metadata is not None:
        processor_kwargs.setdefault("video_metadata", video_metadata)
        processor_kwargs.setdefault("do_sample_frames", False)

    processor_inputs = {"text": text, "images": images, "videos": videos, **processor_kwargs}
    if audio is not None:
        # MiniCPMOProcessor 签名是 audios=；其它 processor 多用 audio=
        if processor is not None and processor.__class__.__name__ == "MiniCPMOProcessor":
            processor_inputs["audios"] = audio
        else:
            processor_inputs["audio"] = audio

    return processor(**processor_inputs)


_MINICPM_AUDIO_PH = "(<audio>./</audio>)"


def is_minicpmo_processor(processor) -> bool:
    return processor is not None and processor.__class__.__name__ == "MiniCPMOProcessor"


def ensure_minicpm_audio_placeholder_bpe_safe(text: str) -> str:
    """Keep ``(<audio>./</audio>)`` as a contiguous token span for vLLM.

    vLLM MiniCPMO looks up the token ids of ``(<audio>./</audio>)`` inside
    ``prompt_token_ids``. If the placeholder is immediately followed by ``\\n``,
    BPE merges ``>)`` with the newline and the lookup fails with:
    ``Failed to apply prompt replacement for mm_items['audio'][0]``.
    """
    if not text:
        return text
    # Protect full MiniCPM placeholder, then map VeRL bare ``<audio>`` tags.
    sentinel = "\0MINICPM_AUDIO_PH\0"
    text = text.replace(_MINICPM_AUDIO_PH, sentinel)
    text = text.replace("<audio>", _MINICPM_AUDIO_PH)
    text = text.replace(sentinel, _MINICPM_AUDIO_PH)
    # Do not glue ``>)`` to a following newline.
    while f"{_MINICPM_AUDIO_PH}\n" in text:
        text = text.replace(f"{_MINICPM_AUDIO_PH}\n", _MINICPM_AUDIO_PH)
    return text


def _join_minicpm_content_parts(parts: list[str]) -> str:
    if not parts:
        return ""
    out = parts[0]
    for part in parts[1:]:
        if out.endswith(_MINICPM_AUDIO_PH):
            out = out + part.lstrip("\n")
        elif part.startswith(_MINICPM_AUDIO_PH):
            out = out.rstrip("\n") + part
        else:
            out = f"{out}\n{part}"
    return ensure_minicpm_audio_placeholder_bpe_safe(out.strip())


def _flatten_minicpm_messages_for_template(messages: list) -> list:
    """MiniCPM chat_template 只吃 str content；把 structured parts 拍平并插入音频占位符。"""
    flat = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            flat.append({"role": role, "content": ensure_minicpm_audio_placeholder_bpe_safe(content)})
            continue
        parts: list[str] = []
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    if p:
                        parts.append(str(p))
                    continue
                ptype = p.get("type")
                if ptype == "audio":
                    parts.append(_MINICPM_AUDIO_PH)
                else:
                    t = p.get("text") or ""
                    if t:
                        parts.append(str(t))
        flat.append({"role": role, "content": _join_minicpm_content_parts(parts)})
    return flat


def _install_minicpmo_chat_template_shim(processor) -> None:
    if getattr(processor, "_minicpm_chat_template_shim", False):
        return
    tok = getattr(processor, "tokenizer", None)
    if tok is None or not hasattr(tok, "apply_chat_template"):
        return
    orig = processor.apply_chat_template if hasattr(processor, "apply_chat_template") else None

    def apply_chat_template(conversation, *args, **kwargs):
        flat = _flatten_minicpm_messages_for_template(conversation)
        # prefer tokenizer; processor.apply_chat_template 在部分版本会走结构化路径
        return tok.apply_chat_template(flat, *args, **kwargs)

    processor.apply_chat_template = apply_chat_template  # type: ignore[method-assign]
    processor._minicpm_chat_template_shim = True  # type: ignore[attr-defined]
    if orig is not None:
        processor._orig_apply_chat_template = orig  # type: ignore[attr-defined]


def set_pad_token_id(tokenizer):
    """Set pad_token_id to eos_token_id if it is None.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to be set.

    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}", stacklevel=1)


def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    """Create a huggingface pretrained tokenizer which correctness handles eos and pad tokens.

    Args:

        name (str): The name of the tokenizer.
        correct_pad_token (bool): Whether to correct the pad token id.
        correct_gemma2 (bool): Whether to correct the gemma2 tokenizer.

    Returns:

        transformers.PreTrainedTokenizer: The pretrained tokenizer.

    """
    from transformers import AutoTokenizer

    if correct_gemma2 and isinstance(name_or_path, str) and "gemma-2-2b-it" in name_or_path:
        # the EOS token in gemma2 is ambiguious, which may worsen RL performance.
        # https://huggingface.co/google/gemma-2-2b-it/commit/17a01657f5c87135bcdd0ec7abb4b2dece04408a
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.", stacklevel=1
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer


def hf_processor(name_or_path, **kwargs):
    """Create a huggingface processor to process multimodal data.

    Args:
        name_or_path (str): The name of the processor.

    Returns:
        Optional[transformers.ProcessorMixin]: The pretrained multimodal processor.
        Returns ``None`` for text-only models (including AutoProcessor fallbacks to
        tokenizer backends such as ``TokenizersBackend``).
    """
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
        # In newer transformers, AutoProcessor may legitimately fall back to a
        # tokenizer backend (e.g. TokenizersBackend) for text-only models.
        # Treat it as "no multimodal processor" and let callers use hf_tokenizer.
        if isinstance(processor, PreTrainedTokenizerBase):
            return None

        config = AutoConfig.from_pretrained(name_or_path, **kwargs)

        # Bind vlm model's get_rope_index method to processor.
        processor.config = config
        model_class = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                model_class = Qwen2VLModel
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                model_class = Qwen2_5_VLModel
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                model_class = Qwen3VLModel
            case "Glm4vProcessor":
                from transformers.models.glm4v import Glm4vModel

                model_class = Glm4vModel
            case "MllamaProcessor":
                pass  # MllamaProcessor and MllamaModel doesn't have get_rope_index property
            case "Gemma4Processor":
                # Gemma4 uses standard 1D RoPE -> no get_rope_index to bind. Disable its strict
                # per-image-token check (which Qwen's processor lacks).
                processor.validate_inputs = lambda *args, **kwargs: None
            case "MiniCPMOProcessor":
                # MiniCPM-o：无 get_rope_index；chat_template 在 tokenizer 上，结构化 content 需先拍平。
                _install_minicpmo_chat_template_shim(processor)
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if model_class is not None:
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            if hasattr(model_class, "get_vision_position_ids"):
                processor.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, processor)
    except Exception as e:
        processor = None
        # TODO(haibin.lin): try-catch should be removed after adding transformer version req to setup.py to avoid
        # silent failure
        warnings.warn(f"Failed to create processor: {e}. This may affect multimodal processing", stacklevel=1)
    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/auto/processing_auto.py#L344
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
