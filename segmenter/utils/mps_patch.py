"""
MPS Compatibility Patch for HuggingFace SegFormer

On Apple MPS the C++ autograd backward pass of transpose / permute
internally calls view() on non-contiguous gradient tensors, which fails.
Python-level monkey-patching of Tensor.view cannot intercept these C++
calls.

The fix is to insert .contiguous() after every transpose / permute in
the forward pass so that the backward graph never hands the C++ engine
a non-contiguous tensor.

Usage:
    from segmenter.utils.mps_patch import patch_segformer_for_mps
    patch_segformer_for_mps()   # call once before model creation

This is a no-op when the MPS backend is not available.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_patched = False


def _patched_overlap_patch_embeddings_forward(self, pixel_values):
    embeddings = self.proj(pixel_values)
    _, _, height, width = embeddings.shape
    embeddings = embeddings.flatten(2).transpose(1, 2).contiguous()
    embeddings = self.layer_norm(embeddings)
    return embeddings, height, width


def _patched_efficient_self_attention_forward(
    self, hidden_states, height, width, output_attentions=False,
):
    batch_size, seq_length, _ = hidden_states.shape

    query_layer = (
        self.query(hidden_states)
        .reshape(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
        .contiguous()
    )

    if self.sr_ratio > 1:
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = (
            hidden_states
            .permute(0, 2, 1)
            .contiguous()
            .reshape(batch_size, num_channels, height, width)
        )
        hidden_states = self.sr(hidden_states)
        hidden_states = (
            hidden_states
            .reshape(batch_size, num_channels, -1)
            .permute(0, 2, 1)
            .contiguous()
        )
        hidden_states = self.layer_norm(hidden_states)

    key_layer = (
        self.key(hidden_states)
        .reshape(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
        .contiguous()
    )
    value_layer = (
        self.value(hidden_states)
        .reshape(batch_size, -1, self.num_attention_heads, self.attention_head_size)
        .transpose(1, 2)
        .contiguous()
    )

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2).contiguous())
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.reshape(new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs


def _patched_dwconv_forward(self, hidden_states, height, width):
    batch_size, seq_len, num_channels = hidden_states.shape
    hidden_states = (
        hidden_states
        .transpose(1, 2)
        .contiguous()
        .reshape(batch_size, num_channels, height, width)
    )
    hidden_states = self.dwconv(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
    return hidden_states


def _patched_mlp_forward(self, hidden_states):
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
    hidden_states = self.proj(hidden_states)
    return hidden_states


def _patched_decode_head_forward(self, encoder_hidden_states, **kwargs):
    batch_size = encoder_hidden_states[-1].shape[0]

    all_hidden_states = ()
    for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
        if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
            height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
            encoder_hidden_state = (
                encoder_hidden_state
                .reshape(batch_size, height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
        encoder_hidden_state = mlp(encoder_hidden_state)
        encoder_hidden_state = (
            encoder_hidden_state
            .permute(0, 2, 1)
            .contiguous()
            .reshape(batch_size, -1, height, width)
        )
        encoder_hidden_state = nn.functional.interpolate(
            encoder_hidden_state,
            size=encoder_hidden_states[0].size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        all_hidden_states += (encoder_hidden_state,)

    hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
    hidden_states = self.batch_norm(hidden_states)
    hidden_states = self.activation(hidden_states)
    hidden_states = self.dropout(hidden_states)
    logits = self.classifier(hidden_states)
    return logits


def _patched_encoder_forward(
    self,
    pixel_values,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=True,
):
    from transformers.modeling_outputs import BaseModelOutput

    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    batch_size = pixel_values.shape[0]
    hidden_states = pixel_values

    for idx, x in enumerate(zip(self.patch_embeddings, self.block, self.layer_norm)):
        embedding_layer, block_layer, norm_layer = x
        hidden_states, height, width = embedding_layer(hidden_states)
        for i, blk in enumerate(block_layer):
            layer_outputs = blk(hidden_states, height, width, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        hidden_states = norm_layer(hidden_states)
        if idx != len(self.patch_embeddings) - 1 or (
            idx == len(self.patch_embeddings) - 1 and self.config.reshape_last_stage
        ):
            hidden_states = (
                hidden_states
                .reshape(batch_size, height, width, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v for v in [hidden_states, all_hidden_states, all_self_attentions]
            if v is not None
        )
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )


def patch_segformer_for_mps():
    """Monkey-patch all SegFormer layers for MPS compatibility.

    Inserts .contiguous() after every transpose / permute so that
    the C++ autograd backward pass never encounters non-contiguous
    tensors (which would trigger an unsupported .view() call on MPS).
    """
    global _patched
    if _patched:
        return
    _patched = True

    if not torch.backends.mps.is_available():
        return

    try:
        from transformers.models.segformer.modeling_segformer import (
            SegformerDecodeHead,
            SegformerDWConv,
            SegformerEfficientSelfAttention,
            SegformerEncoder,
            SegformerMLP,
            SegformerOverlapPatchEmbeddings,
        )
    except ImportError:
        print("[mps_patch] Could not import SegFormer classes — patch not applied")
        return

    SegformerOverlapPatchEmbeddings.forward = _patched_overlap_patch_embeddings_forward
    SegformerEfficientSelfAttention.forward = _patched_efficient_self_attention_forward
    SegformerDWConv.forward = _patched_dwconv_forward
    SegformerMLP.forward = _patched_mlp_forward
    SegformerDecodeHead.forward = _patched_decode_head_forward
    SegformerEncoder.forward = _patched_encoder_forward

    print("[mps_patch] Patched all SegFormer layers for MPS compatibility")
