from transformers.cache_utils import Cache
from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import repeat_kv, Phi3Attention, Phi3Model, Phi3ForCausalLM, apply_rotary_pos_emb, Phi3FlashAttention2
from model.configuation_miniPhi3 import MiniPhiConfig
from typing import List, Optional, Tuple, Union
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
import warnings

import inspect
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters)

import math
logger = logging.get_logger(__name__)
import torch
import torch.nn as nn

from einops import einsum


class CoPE(nn.Module):
    def __init__(self, npos_max, head_dim):
        super().__init__()
        self.npos_max = npos_max
        self.pos_emb = nn.parameter.Parameter(
            torch.zeros(1, head_dim, npos_max))

    def forward(self, query, attn_logits):
        # compute positions
        gates = torch.sigmoid(attn_logits)
        pos = gates.flip(-1).cumsum(dim=-1).flip(-1)
        pos = pos.clamp(max=self.npos_max - 1)
        # interpolate from integer positions
        pos_ceil = pos.ceil().long()
        pos_floor = pos.floor().long()
        logits_int = torch.matmul(query, self.pos_emb)
        logits_ceil = logits_int.gather(-1, pos_ceil)
        logits_floor = logits_int.gather(-1, pos_floor)
        w = pos - pos_floor
        return logits_ceil * w + logits_floor * (1 - w)


class MiniPhi3Attention(Phi3Attention):
    def __init__(self, config: MiniPhiConfig, origin_params):
        super().__init__(config, layer_idx=0)
        self.__replace_param(origin_params)
        self.cope = CoPE(self.max_position_embeddings, self.head_dim)

    def __replace_param(self, origin_params: dict):
        self.__dict__.update(origin_params)
        del self.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos: query_pos +
                         self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos +
                           self.num_key_value_heads * self.head_dim:]

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)
        # cos, sin = self.rotary_emb(
        #     value_states, position_ids, seq_len=kv_seq_len)

        # query_states, key_states = apply_rotary_pos_emb(
        #     query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            # key_states, value_states = past_key_value.update(
            #     key_states, value_states, self.layer_idx, cache_kwargs)
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        attn_weights = self.cope(query_states, attn_weights)
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniPhi3FlashAttention2(Phi3FlashAttention2):
    def __init__(self, config: MiniPhiConfig, origin_params):
        super().__init__(config, layer_idx=0)
        self.__replace_param(origin_params)
        "Flash attention does not support cope"
        self.cope = CoPE(self.max_position_embeddings, self.head_dim)

    def __replace_param(self, origin_params: dict):
        self.__dict__.update(origin_params)
        del self.rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Phi3FlashAttention2 attention does not support output_attentions

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention. Please use `attn_implementation='eager'` or upgrade flash-attn library."
            )
            raise ValueError(
                "The current flash attention version does not support sliding window attention.")

        output_attentions = False

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos: query_pos +
                         self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos +
                           self.num_key_value_heads * self.head_dim:]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        # cos, sin = self.rotary_emb(
        #     value_states, position_ids, seq_len=rotary_seq_len)

        # query_states, key_states = apply_rotary_pos_emb(
        #     query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(
                self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            # cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(
            bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class MiniPhi3(Phi3ForCausalLM):
    """
    参数量约0.13B
    MiniPhi3(
        (embed_tokens): Embedding(32000, 768, padding_idx=0)
        (embed_dropout): Dropout(p=0.0, inplace=False)
        (layers): ModuleList(
            (0-11): 12 x Phi3DecoderLayer(
            (self_attn): Phi3Attention(
                (o_proj): Linear(in_features=768, out_features=768, bias=False)
                (qkv_proj): Linear(in_features=768, out_features=2304, bias=False)
                (rotary_emb): Phi3RotaryEmbedding()
            )
            (mlp): Phi3MLP(
                (gate_up_proj): Linear(in_features=768, out_features=4096, bias=False)
                (down_proj): Linear(in_features=2048, out_features=768, bias=False)
                (activation_fn): SiLU()
            )
            (input_layernorm): Phi3RMSNorm()
            (resid_attn_dropout): Dropout(p=0.0, inplace=False)
            (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
            (post_attention_layernorm): Phi3RMSNorm()
            )
        )
        (norm): Phi3RMSNorm()
        )
    """

    def __init__(self, config: MiniPhiConfig):
        super().__init__(config)
        "原计划将CoPE加入Phi3，但是因为其暂时不支持Flash Attention，因此暂时搁置"
        if config.use_cope:
            ATTN_CLS = MiniPhi3FlashAttention2 if config._attn_implementation == "flash_attention_2" else MiniPhi3Attention
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = ATTN_CLS(
                    config, layer.self_attn.__dict__)
