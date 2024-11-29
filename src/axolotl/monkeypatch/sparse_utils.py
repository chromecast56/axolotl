from transformers import (
    PretrainedConfig,
    TrainerCallback,
)
import logging
import warnings
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
import axolotl
from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import types
import math
import wandb
import transformers


from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaAttention
from einops import rearrange

from axolotl.monkeypatch.llama_attn_hijack_flash import generate_qkv

try:
    from flash_attn.flash_attn_interface import (  # pylint: disable=ungrouped-imports
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
except ImportError:
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_kvpacked_func as flash_attn_varlen_kvpacked_func,
    )
    from flash_attn.flash_attn_interface import (
        flash_attn_unpadded_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    )


logger = LOG = logging.getLogger("axolotl.monkeypatch.sparse")

def get_module_device(module):
    return next(module.parameters()).device
def interp(x, xp, fp):
    """Custom interpolation function for PyTorch tensors."""
    i = torch.searchsorted(xp, x)
    i = torch.clamp(i, 1, len(xp) - 1)
    
    xp_left = xp[i - 1]
    xp_right = xp[i]
    fp_left = fp[i - 1]
    fp_right = fp[i]
    
    t = (x - xp_left) / (xp_right - xp_left)
    return fp_left + t * (fp_right - fp_left)
class Distribution:
    def __init__(self, file_path, hidden_type):
        self.file_path = file_path
        self.hidden_type = hidden_type # h1 or h2
        
        histogram = torch.load(f"{self.file_path}/histograms.pt")

        self.bin_centers, self.counts = histogram[f"{self.hidden_type}_centers"], histogram[self.hidden_type]

        self.total_count = self.counts.sum()
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    # kernel smoothing
    def pdf(self, x, bandwidth=None):
        if bandwidth is None:
            bandwidth =  1.06 * torch.std(self.bin_centers[1:-1]) * (self.total_count-2)**(-1/5)
        
        bin_centers = self.bin_centers.unsqueeze(1)
        
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor([x])
        else:
            x = x.unsqueeze(0)
        
        kernel = torch.exp(-0.5 * ((x - bin_centers) / bandwidth)**2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))
        pdf = torch.sum(kernel * self.counts.unsqueeze(1), dim=0) / self.total_count
        
        return pdf
    
    def cdf(self, x):
        return interp(x, self.bin_centers, self.cumulative_counts / self.total_count)
    
    def icdf(self, q):
        # if q < 0.01 or q > 0.99:
        #     print(f"WARNING: All outliers clip to the most extreme bin")

        target_count = q * self.total_count
        idx = torch.searchsorted(self.cumulative_counts, target_count)
        
        if idx == 0:
            return self.bin_centers[0]
        elif idx == len(self.bin_centers):
            return self.bin_centers[-1]
        else:
            lower_count = self.cumulative_counts[idx - 1]
            upper_count = self.cumulative_counts[idx]
            lower_value = self.bin_centers[idx - 1]
            upper_value = self.bin_centers[idx]
            
            fraction = (target_count - lower_count) / (upper_count - lower_count)
            return lower_value + fraction * (upper_value - lower_value)

# TODO: for now sparsify entire prefill, modify later if needed
# def sparsify(hidden_states, thresh):
#     return (hidden_states.abs() > thresh).to(hidden_states.dtype) * hidden_states


class SparsifyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, thresh):
        # mask = (hidden_states.abs() > thresh).to(hidden_states.dtype)
        # ctx.save_for_backward(hidden_states)
        # return mask * hidden_states

        return (hidden_states.abs() > thresh).to(hidden_states.dtype) * hidden_states

    @staticmethod
    def backward(ctx, grad_output):
        # hidden_states, = ctx.saved_tensors
        # Straight-through estimator: pass gradients straight-through
        return grad_output, None  # None for thresh gradient

def sparsify(hidden_states, thresh):
    return SparsifyFunction.apply(hidden_states, thresh)


from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.cache_utils import StaticCache, Cache
def _self_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if isinstance(past_key_value, StaticCache):
        raise ValueError(
            "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
        )

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    # hidden_states = sparsify(hidden_states, self.thresh1)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = sparsify(query_states, self.thresh_q)
    key_states = sparsify(key_states, self.thresh_k)
    value_states = sparsify(value_states, self.thresh_v)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        # logger.warning_once(
        #     "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
        #     "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
        #     "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
        #     "removed and `position_embeddings` will be mandatory."
        # )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        # logger.warning_once(
        #     f"The input hidden states seems to be silently casted in float32, this might be related to"
        #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        #     f" {target_dtype}."
        # )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        position_ids=position_ids,
        dropout=dropout_rate,
        sliding_window=getattr(self, "sliding_window", None),
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
        is_causal=self.is_causal,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

    attn_output = sparsify(attn_output, self.thresh_o)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _mlp_forward(self, x, activation_module=None):
    if self.config.pretraining_tp > 1:
        # TODO: UNTESTED

        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        # TODO: modify if we're targetting different sparsities per proj

        x = sparsify(x, self.thresh1)
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)

        gate_proj = torch.cat(
            [F.linear(x_gate, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x_up, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        # MONKEYPATCH HERE
        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)

        # if self.grabbing_mode:
        #     self.activation_module.grab_activations(intermediate_states, 'h2')
        #     x_down = self.down_proj(intermediate_states)
        # else:
        #     x_down = self.sparse_fns['down'](intermediate_states)

        # TODO: modify if we're targetting different sparsities per proj
        intermediate_states = sparsify(intermediate_states, self.thresh2)
        x_down = self.down_proj(intermediate_states)

        
        down_proj = [
            F.linear(x_down, down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        
        x_gate = sparsify(x, self.thresh_gate)
        x_up = sparsify(x, self.thresh_up)

        intermediate_states = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
        x_down = sparsify(intermediate_states, self.thresh_down)

        down_proj = self.down_proj(x_down)

        # # TODO: modify if we're targetting different sparsities per proj
        # x = sparsify(x, self.thresh1)
        # intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)

        # # TODO: modify if we're targetting different sparsities per proj
        # intermediate_states = sparsify(intermediate_states, self.thresh2)
        # down_proj = self.down_proj(intermediate_states)

    return down_proj


def monkeypatch_attn(self_attn, hist_path, sparsity_q, sparsity_k, sparsity_v, sparsity_o):
    assert self_attn.q_proj is not None, "q_proj is None"

    distr1 = Distribution(f"{hist_path}/self_attn", "h1")
    distr2 = Distribution(f"{hist_path}/self_attn", "h2")

    thresh_q = distr1.icdf(0.5 + sparsity_q/2).to(get_module_device(self_attn))
    thresh_k = distr1.icdf(0.5 + sparsity_k/2).to(get_module_device(self_attn))
    thresh_v = distr1.icdf(0.5 + sparsity_v/2).to(get_module_device(self_attn))
    thresh_o = distr2.icdf(0.5 + sparsity_o/2).to(get_module_device(self_attn))

    self_attn.forward_old = self_attn.forward
    self_attn.forward = types.MethodType(_self_attn_forward, self_attn)
    self_attn.thresh_q = thresh_q
    self_attn.thresh_k = thresh_k
    self_attn.thresh_v = thresh_v
    self_attn.thresh_o = thresh_o


def monkeypatch_mlp(mlp, hist_path, sparsity_gate, sparsity_up, sparsity_down):
    distr1 = Distribution(f"{hist_path}/mlp", "h1")
    distr2 = Distribution(f"{hist_path}/mlp", "h2")

    thresh_gate = distr1.icdf(0.5 + sparsity_gate/2).to(get_module_device(mlp))
    thresh_up = distr1.icdf(0.5 + sparsity_up/2).to(get_module_device(mlp))
    thresh_down = distr2.icdf(0.5 + sparsity_down/2).to(get_module_device(mlp))

    mlp.forward_old = mlp.forward
    mlp.forward = types.MethodType(_mlp_forward, mlp)
    mlp.thresh_gate = thresh_gate
    mlp.thresh_up = thresh_up
    mlp.thresh_down = thresh_down

def monkeypatch_sparse_forward(model, sparse_hist_path, sparsity_level, lookup_path):

    sparse_levels = [sparsity_level] * len(model.model.layers)
    sparsities = get_layer_greedy_sparsities(sparse_levels, lookup_path)

    for layer_idx, layer in enumerate(model.model.layers):
        hist_path = f"{sparse_hist_path}/layer-{layer_idx}"
        monkeypatch_attn(layer.self_attn, hist_path, sparsities['q'][layer_idx], sparsities['k'][layer_idx], sparsities['v'][layer_idx], sparsities['o'][layer_idx])
        monkeypatch_mlp(layer.mlp, hist_path, sparsities['gate'][layer_idx], sparsities['up'][layer_idx], sparsities['down'][layer_idx])
    


import pandas as pd
import os
def get_layer_greedy_sparsities(layer_sparsities, results_dir):
    num_layers = len(layer_sparsities)
    projs = ['q', 'k', 'v', 'o', 'gate', 'up', 'down']
    sparsities = {proj: [0.0] * num_layers for proj in projs}
    
    for layer, target_sparsity in enumerate(layer_sparsities):
        file_path = os.path.join(results_dir, f'layer-{layer}', 'results.csv')
        df = pd.read_csv(file_path)
        
        # Find the row with the closest effective sparsity
        closest_row = df.iloc[(df['Effective Sparsity'] - target_sparsity).abs().argsort()[:1]]
        
        for proj in projs:
            sparsities[proj][layer] = closest_row[proj].values[0]
    
    return sparsities