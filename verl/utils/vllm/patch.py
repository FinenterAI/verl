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

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3_5MoeForCausalLM)
except ImportError:
    pass


def _as_1d_long_tensor(x):
    import torch

    if x is None:
        return torch.tensor([], dtype=torch.long)
    if torch.is_tensor(x):
        return x.detach().reshape(-1).long()
    if isinstance(x, (list, tuple)):
        if not x:
            return torch.tensor([], dtype=torch.long)
        if all(isinstance(v, (int, float)) for v in x):
            return torch.tensor([int(v) for v in x], dtype=torch.long)
        parts = [_as_1d_long_tensor(v) for v in x if v is not None]
        return torch.cat(parts, dim=0) if parts else torch.tensor([], dtype=torch.long)
    return torch.tensor([int(x)], dtype=torch.long)


def _stack_unpadded_audio_chunks(chunks: list):
    """Pad-stack chunk tensors ``[80, L_i]`` → ``[N, 80, max_L]``."""
    import torch

    if not chunks:
        return torch.zeros(0, 80, 0)
    if len(chunks) == 1:
        c = chunks[0]
        return c.unsqueeze(0) if c.dim() == 2 else c
    max_l = max(int(c.shape[-1]) for c in chunks)
    c0 = chunks[0]
    out = c0.new_zeros((len(chunks), c0.shape[-2], max_l))
    for i, c in enumerate(chunks):
        if c.dim() == 3:
            c = c.squeeze(0)
        out[i, :, : c.shape[-1]] = c
    return out


def _normalize_minicpmo_audio_hf_outputs(audio_features, audio_feature_lens):
    """Normalize HF MiniCPM-o audio outputs to per-audio lists.

    HF often returns:
      - ``audio_features``: ``Tensor[num_chunks, 80, T]`` for a single audio
      - ``audio_feature_lens``: ``[tensor([L1, L2, ...])]`` (one entry per audio)

    vLLM ``MultiModalBatchedField`` requires matching first-dim / list lengths
    (= number of audios). We return:
      - features: list[Tensor[num_chunks_i, 80, max_L_i]]
      - lenses: list[Tensor[num_chunks_i]]
    """
    import torch

    # Normalize lenses → list[1d long tensor] (one per audio)
    if torch.is_tensor(audio_feature_lens):
        if audio_feature_lens.dim() == 0:
            lens_list = [audio_feature_lens.reshape(1).long()]
        elif audio_feature_lens.dim() == 1:
            # Ambiguous; resolve with features below. Start as one-audio.
            lens_list = [audio_feature_lens.long()]
        else:
            lens_list = [t.long() for t in audio_feature_lens.unbind(0)]
    else:
        lens_list = [_as_1d_long_tensor(x) for x in audio_feature_lens]

    total_chunks = sum(int(x.numel()) for x in lens_list)

    if torch.is_tensor(audio_features):
        feats = audio_features
        if feats.dim() == 2:  # [80, T] single chunk
            feats = feats.unsqueeze(0)
        if feats.dim() != 3:
            raise ValueError(f"Unexpected audio_features dim={feats.dim()} shape={tuple(feats.shape)}")

        # Common MiniCPM-o case: one audio, all chunks stacked on dim0.
        if len(lens_list) == 1 and feats.shape[0] == int(lens_list[0].numel()):
            fl = lens_list[0]
            chunks = [feats[i, :, : int(fl[i].item())] for i in range(fl.numel())]
            return [_stack_unpadded_audio_chunks(chunks)], lens_list

        # One chunk per list entry (lens may be scalar-per-audio).
        if feats.shape[0] == len(lens_list) and all(int(x.numel()) == 1 for x in lens_list):
            out_feats = []
            for i, fl in enumerate(lens_list):
                L = int(fl[0].item())
                out_feats.append(feats[i, :, :L].unsqueeze(0))
            return out_feats, lens_list

        if feats.shape[0] != total_chunks:
            raise ValueError(
                f"audio_features rows ({feats.shape[0]}) != total chunks ({total_chunks}); "
                f"lens={lens_list}"
            )
        out_feats = []
        idx = 0
        for fl in lens_list:
            chunks = []
            for j in range(fl.numel()):
                L = int(fl[j].item())
                chunks.append(feats[idx, :, :L])
                idx += 1
            out_feats.append(_stack_unpadded_audio_chunks(chunks))
        return out_feats, lens_list

    # features already a sequence
    feat_seq = list(audio_features)
    if len(feat_seq) == len(lens_list):
        out_feats = []
        for feat, fl in zip(feat_seq, lens_list):
            if torch.is_tensor(feat) and feat.dim() == 2 and fl.numel() == 1:
                out_feats.append(feat[:, : int(fl[0].item())].unsqueeze(0))
            elif torch.is_tensor(feat) and feat.dim() == 3 and feat.shape[0] == fl.numel():
                chunks = [feat[i, :, : int(fl[i].item())] for i in range(fl.numel())]
                out_feats.append(_stack_unpadded_audio_chunks(chunks))
            else:
                out_feats.append(feat)
        return out_feats, lens_list

    raise ValueError(
        f"Cannot align audio_features(len={len(feat_seq)}) with audio_feature_lens(len={len(lens_list)})"
    )


def _coerce_minicpmo_audio_mm_tensors(audio_features, audio_feature_lens):
    """Fix shapes before MiniCPMOAudioFeatureInputs TensorSchema validation.

    Safety net for BatchedField.reduce which unsqueezes ``[S,80,L]`` → ``[1,S,80,L]``.
    Preferred path uses flat_from_sizes so this is usually a no-op.
    """
    import torch

    feats = audio_features
    lens = audio_feature_lens

    if torch.is_tensor(feats) and feats.dim() == 4:
        bn, s, c, l = feats.shape
        feats = feats.reshape(bn * s, c, l)
    elif isinstance(feats, list) and feats:
        if isinstance(feats[0], list):
            flat = []
            for part in feats:
                flat.extend(part)
            feats = flat
        elif torch.is_tensor(feats[0]) and feats[0].dim() == 3:
            feats = torch.cat(list(feats), dim=0)
        elif torch.is_tensor(feats[0]) and feats[0].dim() == 2:
            pass

    if torch.is_tensor(lens):
        if lens.dim() == 1:
            lens = [lens]
        elif lens.dim() >= 2:
            lens = list(lens.unbind(0))
    elif isinstance(lens, list) and lens and isinstance(lens[0], list):
        lens = [torch.as_tensor(x).reshape(-1) for x in lens]

    return feats, lens


def _patch_minicpmo_audio_feature_lens_schema(MiniCPMOAudioFeatureInputs) -> None:
    """Allow jagged per-audio chunk counts in one encoder batch.

    vLLM batches multiple audios' ``audio_feature_lens`` into a list like
    ``[tensor([3000,500]), tensor([3000]), tensor([3000,3000])]``. Stock
    ``TensorShape("bn","s")`` requires identical ``s``, which crashes (or, under
    async CUDA, surfaces later as embedding ``indexSelect`` OOB / EngineDead).
    """
    from typing import Annotated

    import torch
    from vllm.utils.tensor_schema import TensorShape

    if getattr(MiniCPMOAudioFeatureInputs, "_minicpm_lens_schema_dynamic_v6", False):
        return

    MiniCPMOAudioFeatureInputs.__annotations__["audio_feature_lens"] = Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("bn", "s", dynamic_dims={"s"}),
    ]
    MiniCPMOAudioFeatureInputs._minicpm_lens_schema_dynamic_v6 = True  # type: ignore[attr-defined]


def patch_vllm_minicpmo_process_audios() -> bool:
    """Fix MiniCPMO multi-chunk audio packing for vLLM mm encoder.

    Root cause: ``MultiModalBatchedField.reduce`` unsqueezes a single audio's
    ``[S,80,L]`` chunks into invalid rank-4 ``[1,S,80,L]`` on the GPU worker
    (separate process — model monkeypatches do not apply there).

    Fix: pack all chunks into one ``[total_S,80,L]`` tensor and declare
    ``audio_features`` with ``flat_from_sizes`` so reduce concatenates without
    an extra batch dim. Also coerce in ``_parse_and_validate_audio_input`` as
    a safety net when the base model class is importable in-process.

    v6: jagged ``audio_feature_lens`` across a multi-audio encoder batch
    (different chunk counts) must use ``dynamic_dims={"s"}`` / ``validate=False``.
    """
    try:
        from vllm.model_executor.models.minicpmo import (
            MiniCPMOAudioEmbeddingItems,
            MiniCPMOAudioFeatureInputs,
            MiniCPMOMultiModalProcessor,
        )
        import vllm.model_executor.models.minicpmo as minicpmo_mod
        from vllm.multimodal.inputs import MultiModalFieldConfig
    except ImportError:
        return False

    _patch_minicpmo_audio_feature_lens_schema(MiniCPMOAudioFeatureInputs)

    processor_done = getattr(MiniCPMOMultiModalProcessor, "_minicpm_audio_chunk_patched_v6", False)
    base_cls = getattr(minicpmo_mod, "MiniCPMOBaseModel", None)
    parse_done = base_cls is None or getattr(base_cls, "_minicpm_audio_parse_patched_v6", False)
    if processor_done and parse_done:
        return True

    try:
        from vllm.multimodal.parse import AudioProcessorItems
    except ImportError:
        try:
            from vllm.multimodal.inputs import AudioProcessorItems  # type: ignore
        except ImportError:
            return False

    if not processor_done:

        def process_audios(self, mm_data, mm_kwargs, tok_kwargs):
            import torch

            if (audios := mm_data.get("audios")) is None:
                return {}

            mm_items = self.info.parse_mm_data({"audio": audios}, validate=False)
            parsed_audios = mm_items.get_items(
                "audio", (MiniCPMOAudioEmbeddingItems, AudioProcessorItems)
            )

            if isinstance(parsed_audios, MiniCPMOAudioEmbeddingItems):
                return {}

            audio_inputs = self._base_call_hf_processor(
                prompts=[self.info.audio_pattern] * len(parsed_audios),
                mm_data={"audios": [[audio] for audio in parsed_audios]},
                mm_kwargs={**mm_kwargs, "chunk_input": True},
                tok_kwargs=tok_kwargs,
                out_keys={"audio_features", "audio_feature_lens"},
            )
            feats, lens = _normalize_minicpmo_audio_hf_outputs(
                audio_inputs["audio_features"],
                audio_inputs["audio_feature_lens"],
            )
            # Flatten all chunks into one tensor for flat_from_sizes field config.
            flat_chunks = []
            for feat, fl in zip(feats, lens):
                if feat.dim() == 3:
                    for i in range(feat.shape[0]):
                        flat_chunks.append(feat[i, :, : int(fl[i].item())])
                elif feat.dim() == 2:
                    flat_chunks.append(feat[:, : int(fl.reshape(-1)[0].item())])
                else:
                    raise ValueError(f"Unexpected per-audio feature shape {tuple(feat.shape)}")
            audio_inputs["audio_features"] = _stack_unpadded_audio_chunks(flat_chunks)
            audio_inputs["audio_feature_lens"] = lens
            audio_inputs["_audio_chunk_sizes"] = torch.tensor(
                [int(x.numel()) for x in lens], dtype=torch.long
            )
            return audio_inputs

        def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
            import torch

            # Keep vision configs from the stock helper.
            try:
                from vllm.model_executor.models.minicpmo import _minicpmo_field_config
            except ImportError:
                from vllm.model_executor.models.minicpmv import _minicpmv_field_config as _minicpmo_field_config  # type: ignore

            cfg = dict(_minicpmo_field_config(hf_inputs))
            lens = hf_inputs.get("audio_feature_lens")
            feats = hf_inputs.get("audio_features")
            if lens is not None and feats is not None:
                if "_audio_chunk_sizes" in hf_inputs:
                    sizes = hf_inputs["_audio_chunk_sizes"]
                elif isinstance(lens, list):
                    sizes = torch.tensor([int(torch.as_tensor(x).numel()) for x in lens], dtype=torch.long)
                elif torch.is_tensor(lens) and lens.dim() >= 2:
                    sizes = torch.tensor([int(row.numel()) for row in lens.unbind(0)], dtype=torch.long)
                elif torch.is_tensor(feats) and feats.dim() == 3 and isinstance(lens, list) is False:
                    # fallback: treat whole tensor as one audio
                    sizes = torch.tensor([int(feats.shape[0])], dtype=torch.long)
                else:
                    sizes = torch.tensor([1], dtype=torch.long)
                cfg["audio_features"] = MultiModalFieldConfig.flat_from_sizes("audio", sizes)
                cfg["audio_feature_lens"] = MultiModalFieldConfig.batched("audio")
                cfg.pop("_audio_chunk_sizes", None)
            return cfg

        MiniCPMOMultiModalProcessor.process_audios = process_audios  # type: ignore[method-assign]
        MiniCPMOMultiModalProcessor._get_mm_fields_config = _get_mm_fields_config  # type: ignore[method-assign]
        MiniCPMOMultiModalProcessor._minicpm_audio_chunk_patched_v6 = True  # type: ignore[attr-defined]
        MiniCPMOMultiModalProcessor._minicpm_audio_chunk_patched_v5 = True  # type: ignore[attr-defined]
        MiniCPMOMultiModalProcessor._minicpm_audio_chunk_patched_v4 = True  # type: ignore[attr-defined]
        MiniCPMOMultiModalProcessor._minicpm_audio_chunk_patched_v3 = True  # type: ignore[attr-defined]
        MiniCPMOMultiModalProcessor._minicpm_audio_chunk_patched_v2 = True  # type: ignore[attr-defined]
        MiniCPMOMultiModalProcessor._minicpm_feat_len_patched = True  # type: ignore[attr-defined]

    # In-process + worker safety net (workers import this via weight-loader hook).
    if base_cls is not None and not getattr(base_cls, "_minicpm_audio_parse_patched_v6", False):
        # Capture the true original once, before any of our wrappers overwrite it.
        if not hasattr(base_cls, "_minicpm_orig_get_audio_hidden_states"):
            raw = minicpmo_mod.MiniCPMOBaseModel.__dict__.get("get_audio_hidden_states")
            # If an older wrapper already overwrote the class dict, fall back carefully:
            # only accept functions that are not our previous wrappers.
            if raw is not None and getattr(raw, "__name__", "") == "get_audio_hidden_states":
                # May already be a wrapper named get_audio_hidden_states — still better than dict().
                base_cls._minicpm_orig_get_audio_hidden_states = raw  # type: ignore[attr-defined]
            else:
                base_cls._minicpm_orig_get_audio_hidden_states = base_cls.get_audio_hidden_states  # type: ignore[attr-defined]
        orig_hidden = base_cls._minicpm_orig_get_audio_hidden_states

        def _parse_and_validate_audio_input(self, **kwargs):
            audio_features = kwargs.get("audio_features", None)
            audio_embeds = kwargs.get("audio_embeds", None)
            if audio_features is None and audio_embeds is None:
                return None
            if audio_embeds is not None:
                from vllm.model_executor.models.minicpmo import MiniCPMOAudioEmbeddingInputs

                return MiniCPMOAudioEmbeddingInputs(
                    type="audio_embeds",
                    audio_embeds=kwargs.pop("audio_embeds"),
                )

            feats = kwargs.pop("audio_features")
            lens = kwargs.pop("audio_feature_lens")
            feats, lens = _coerce_minicpmo_audio_mm_tensors(feats, lens)
            # validate=False: stock schema rejects jagged chunk counts; we also
            # mark ``s`` dynamic above, but skip validation as belt-and-suspenders
            # for workers that imported the class before the annotation patch.
            return MiniCPMOAudioFeatureInputs(
                type="audio_features",
                audio_features=feats,
                audio_feature_lens=lens,
                validate=False,
            )

        def get_audio_hidden_states(self, data):
            import torch

            # TensorSchema is not a Mapping — dict(data) raises
            # TypeError: attribute name must be string, not 'int'.
            feats = data["audio_features"]
            lens = data["audio_feature_lens"]
            feats2, lens2 = _coerce_minicpmo_audio_mm_tensors(feats, lens)
            if isinstance(feats2, list) and feats2 and torch.is_tensor(feats2[0]) and feats2[0].dim() == 2:
                feats2 = _stack_unpadded_audio_chunks(feats2)
            if feats2 is feats and lens2 is lens:
                return orig_hidden(self, data)
            return orig_hidden(
                self,
                MiniCPMOAudioFeatureInputs(
                    type="audio_features",
                    audio_features=feats2,
                    audio_feature_lens=lens2,
                    validate=False,
                ),
            )

        base_cls._parse_and_validate_audio_input = _parse_and_validate_audio_input  # type: ignore[method-assign]
        base_cls.get_audio_hidden_states = get_audio_hidden_states  # type: ignore[method-assign]
        base_cls._minicpm_audio_parse_patched_v6 = True  # type: ignore[attr-defined]
        base_cls._minicpm_audio_parse_patched_v5 = True  # type: ignore[attr-defined]
        base_cls._minicpm_audio_parse_patched_v4 = True  # type: ignore[attr-defined]

    return True


def patch_vllm_moe_model_weight_loader(model):
    # Ensure MiniCPM-o multi-chunk audio packing works (also needed for non-MoE MiniCPM-o).
    patch_vllm_minicpmo_process_audios()
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (model / language_model / llm for MiniCPM-V/O wrappers)
    inner_model = (
        getattr(model, "model", None)
        or getattr(model, "language_model", None)
        or getattr(model, "llm", None)
    )
    # Non-MoE / non-standard wrappers (e.g. MiniCPM-o): nothing to patch.
    if inner_model is None:
        return

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS)):
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    if type(inner_model).__name__ in ("Qwen3MoeLLMForCausalLM", "Qwen3_5MoeForCausalLM"):
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts:
            continue

        # verl syncs plain HF names, not PEFT checkpoint names, so the LoRA MoE
        # ``lora_base_layer_prefix`` (folded into a dotted param_name that
        # getattr can't resolve) must be cleared. Runs before the
        # ``weight_loader`` guard so LoRA-wrapped MoE (no weight_loader) is covered.
        routed_experts = getattr(getattr(experts, "base_layer", None), "routed_experts", None)
        if routed_experts is not None and getattr(routed_experts, "lora_base_layer_prefix", ""):
            routed_experts.lora_base_layer_prefix = ""

        if not hasattr(experts, "weight_loader"):
            continue

        # Patch the weight loaders
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader
