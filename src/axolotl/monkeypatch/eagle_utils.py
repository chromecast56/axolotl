import torch
from torch import nn
from typing import Optional, Union, List, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers import PreTrainedModel, LlamaPreTrainedModel

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


from dataclasses import dataclass
@dataclass
# adding custom metrics
class EagleCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Custom output class that adds top1 metric to CausalLMOutputWithPast
    """
    top1: Optional[float] = None

class UniformNoise:
    def __init__(self, std=0.2):
        self.std = std

    def __call__(self, hidden_states):
        # NOTE: original repo scales the noise inversely by sequence length ???
        # noise = (torch.rand_like(hidden_states) - 0.5) * self.std * 512 / hidden_states.shape[1]
        noise = (torch.rand_like(hidden_states) - 0.5) * self.std
        return hidden_states + noise



class EagleHead(nn.Module):
    def __init__(self, model: PreTrainedModel, num_layers: int):
        super().__init__()

        self.register_buffer("embed_tokens", model.model.embed_tokens.weight, persistent=False)


        LlamaDecoderLayer = type(model.model.layers[0])

        hidden_dim = model.config.hidden_size

        # bias is True in official imp
        self.fc = nn.Linear(2*hidden_dim, hidden_dim, bias=True)
        
        # TODO: does initializing from a pretrained layer make sense?
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(model.config, layer_idx=idx) for idx in range(num_layers)
        ])

        self.noise = UniformNoise()


        # NOTE: EAGLE imp doesn't include norm
        # self.norm = type(model.model.norm)(model.config.hidden_size, eps=model.config.rms_norm_eps)


    def forward(
            self,
            hidden_states,
            input_ids,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,

    ):
        # NOTE: EAGLE adds noise, but is an artifact of bad methodology.
        hidden_states = self.noise(hidden_states)
        batch_size, seq_length, hidden_dim = hidden_states.shape

        # Shift tokens forward by one position
        # NOTE: See section 4.3.2 in EAGLE for why. For given hidden state, we technically know what the next token is by just doing lm head. 
        # So, we're allowed to be forward looking by 1 token (this results in +10% acceptance rate).

        with torch.no_grad():
            # Instead of shifting, we'll use the next tokens directly
            next_input_ids = input_ids[:, 1:]  # Remove first token
            pad_column = torch.full((batch_size, 1), IGNORE_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)
            next_input_ids = torch.cat([next_input_ids, pad_column], dim=1)  # Add padding at end
            inputs_embeds = torch.nn.functional.embedding(next_input_ids, self.embed_tokens)

        position_ids = position_ids.view(-1, seq_length).long()
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        hidden_states = self.fc(torch.cat((inputs_embeds, hidden_states), dim=-1))

        all_hidden_states = () if output_hidden_states else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )


            hidden_states = layer_outputs[0]

        # hidden_states = self.norm(hidden_states)

        return hidden_states

def compute_loss(hidden_states, eagle_hidden_states, eagle_logits, labels):
    # 1. Hidden state representation loss (L2 loss between predicted and actual next hidden states)
    hidden_state_loss = None
    if hidden_states is not None:
        # Shift hidden states for comparison
        shift_eagle = eagle_hidden_states[..., :-1, :].contiguous()
        shift_target = hidden_states[..., 1:, :].contiguous()

        # Create mask for ignore tokens
        not_ignore = labels[..., 1:].ne(IGNORE_TOKEN_ID)
        
        # Apply mask to hidden states
        shift_eagle = shift_eagle[not_ignore.unsqueeze(-1).expand_as(shift_eagle)]
        shift_target = shift_target[not_ignore.unsqueeze(-1).expand_as(shift_target)]

        hidden_state_loss = nn.SmoothL1Loss()(shift_eagle, shift_target)

    # 2. Language modeling loss
    lm_loss = None
    if labels is not None:
        vocab_size = eagle_logits.size(-1)

        # Shift for causal LM: predict next token
        shift_logits = eagle_logits[..., :-2, :].contiguous()
        # Shift labels by 2 since eagle is predicting next-next token
        shift_labels = labels[..., 2:].contiguous()
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        lm_loss = loss_fct(shift_logits, shift_labels)

    return hidden_state_loss, lm_loss


def compute_topk_acc(logits, labels, k=1):
    # Shift logits and labels similar to loss computation
    # Eagle predicts next-next token, so shift logits by -2 and compare with labels
    shift_logits = logits[..., :-2, :].contiguous()
    shift_labels = labels[..., 2:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Create mask for non-ignored tokens
    not_ignore = shift_labels.ne(IGNORE_TOKEN_ID)
    
    # Get topk predictions
    _, pred = shift_logits.topk(k, dim=-1)
    correct = pred.eq(shift_labels.unsqueeze(-1)).any(-1)
    
    # Calculate accuracy only on non-ignored tokens
    correct = correct[not_ignore]
    
    return correct.float().mean().item() if correct.numel() > 0 else 0.0

## adapted from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/models/llama/modeling_llama.py#L879
class EagleAugmentedModel(LlamaPreTrainedModel):
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(
        self,
        model: PreTrainedModel,
        freeze_model: bool = True,
        num_layers: int = 1,
        hidden_state_loss_coef: float = 0.1,
        lm_loss_coef: float = 1.0,
    ):
        super().__init__(config=model.config)
        self.model = model.model # use the decoder model w/o LM head
        hidden_dim = model.config.hidden_size

        self.lm_head = model.lm_head

        self.eagle_head = EagleHead(model, num_layers)

        self.hidden_state_loss_coef = hidden_state_loss_coef
        self.lm_loss_coef = lm_loss_coef


        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        hidden_states = outputs[0]

        eagle_hidden_states = self.eagle_head(
            hidden_states,
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


        if self.config.pretraining_tp > 1:
            raise NotImplementedError
        else:
            logits = self.lm_head(hidden_states)

            eagle_logits = self.lm_head(eagle_hidden_states)

            # TODO: HASS introduces distillation wrt base model, instead of LM loss


        loss = None
        top1 = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            hidden_states = hidden_states.float()
            eagle_hidden_states = eagle_hidden_states.float()
            eagle_logits = eagle_logits.float()

            hidden_state_loss, lm_loss = compute_loss(hidden_states, eagle_hidden_states, eagle_logits, labels)

            loss = self.hidden_state_loss_coef * hidden_state_loss + self.lm_loss_coef * lm_loss


            if not self.training:
                top1 = compute_topk_acc(eagle_logits, labels, k=1)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return EagleCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            top1 = top1,
        )

    @classmethod
    def _autoset_attn_implementation(cls, config, *args, **kwargs):
        # Skip torch dtype checking for attention implementation.
        return config

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model