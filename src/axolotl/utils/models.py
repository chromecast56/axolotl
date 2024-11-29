"""Module for models and model loading"""

# pylint: disable=too-many-lines

import logging
import math
import os
import types
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

import addict
import bitsandbytes as bnb
import torch
import transformers
import transformers.modeling_utils
from accelerate import init_empty_weights
from bitsandbytes.nn import Params4bit
from peft import (
    LoftQConfig,
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import QuantLinear
from torch import nn
from transformers import (  # noqa: F401
    AddedToken,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPTQConfig,
    LlavaForConditionalGeneration,
    MllamaForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.integrations.deepspeed import (
    HfTrainerDeepSpeedConfig,
    is_deepspeed_zero3_enabled,
)

from axolotl.models.mamba import fix_mamba_attn_for_loss
from axolotl.monkeypatch.multipack import (
    SUPPORTED_MULTIPACK_MODEL_TYPES,
    patch_for_multipack,
)
from axolotl.prompt_tokenizers import LLAMA_DEFAULT_EOS_TOKEN
from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import get_device_count, get_device_type, zero_only
from axolotl.utils.gradient_checkpointing import hf_grad_checkpoint_unsloth_wrapper
from axolotl.utils.lora_embeddings import get_linear_embedding_layers
from axolotl.utils.model_shard_quant import load_sharded_model, load_sharded_model_quant

LOG = logging.getLogger("axolotl")


# copied from accelerator.FullyShardedDataParallelPlugin
def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__

    if len(modules_children) == 0:
        return None

    for child_module in modules_children:
        module_class = get_module_class_from_name(child_module, name)
        if module_class is not None:
            return module_class

    return None


def check_model_config(cfg: DictDefault, model_config: Union[AutoConfig, DictDefault]):
    if cfg.is_multimodal:
        model_config = model_config.text_config

    quant_config_exists = (
        hasattr(model_config, "quantization_config")
        and model_config.quantization_config
    )
    quant_config_method_is_gptq = (
        quant_config_exists
        and "quant_method" in model_config.quantization_config
        and model_config.quantization_config["quant_method"] == "gptq"
    )

    if cfg.gptq and not quant_config_method_is_gptq:
        raise ValueError(
            "model_config.quantization_config is not set or quant_method is not set to gptq. "
            "Please make sure to point to a GPTQ model."
        )

    if not cfg.gptq and quant_config_exists:
        raise ValueError(
            "model_config.quantization_config is set but `gptq` flag is not. "
            "Please use the `gptq` flag to train quantized model or point to a non-quantized model."
        )

    lora_modules_to_save = get_linear_embedding_layers(model_config.model_type)
    if (
        cfg.adapter
        and cfg.tokens
        and (
            not cfg.lora_modules_to_save
            or not all(x in cfg.lora_modules_to_save for x in lora_modules_to_save)
        )
    ):
        lora_modules_to_save = ", ".join(map(lambda x: f"`{x}`", lora_modules_to_save))
        raise ValueError(
            f"`lora_modules_to_save` not properly set when adding new tokens. Please include [{lora_modules_to_save}] in `lora_modules_to_save`."
        )


def load_model_config(cfg):
    model_config_name = cfg.base_model_config or cfg.base_model
    if not model_config_name and cfg.tokenizer_config:
        model_config_name = cfg.tokenizer_config
    trust_remote_code = cfg.trust_remote_code is True
    config_kwargs = {}
    if cfg.revision_of_model:
        config_kwargs["revision"] = cfg.revision_of_model

    try:
        model_config = AutoConfig.from_pretrained(
            model_config_name,
            trust_remote_code=trust_remote_code,
            **config_kwargs,
        )
    except ValueError as err:
        if "mamba" in model_config_name:
            return addict.Dict(
                {
                    "model_type": "mamba",
                }
            )
        raise err

    if cfg.overrides_of_model_config:
        for key, val in cfg.overrides_of_model_config.items():
            setattr(model_config, key, val)

    check_model_config(cfg, model_config)

    return model_config


def load_tokenizer(cfg):
    model_config = load_model_config(cfg)
    tokenizer_kwargs = {}
    use_fast = True  # this is the default

    if cfg.tokenizer_use_fast is not None:
        use_fast = cfg.tokenizer_use_fast
    if cfg.tokenizer_legacy is not None:
        # True is the default w/ https://github.com/huggingface/transformers/pull/25224
        tokenizer_kwargs["legacy"] = cfg.tokenizer_legacy

    tokenizer_cls = AutoTokenizer
    if cfg.tokenizer_type:
        tokenizer_cls = getattr(transformers, cfg.tokenizer_type)

    tokenizer = tokenizer_cls.from_pretrained(
        cfg.tokenizer_config,
        trust_remote_code=cfg.trust_remote_code or False,
        use_fast=use_fast,
        **tokenizer_kwargs,
    )

    if (
        tokenizer.__class__.__name__
        in [
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
            "CodeLlamaTokenizerFast",
        ]
        and hasattr(tokenizer, "pad_token")
        and not tokenizer.pad_token
    ):
        # set a pad_token, but use eos_token so we don't add a new token
        tokenizer.pad_token = LLAMA_DEFAULT_EOS_TOKEN

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Mistral's official FA implementation requires left padding
    if cfg.is_mistral_derived_model and cfg.flash_attention and not cfg.sample_packing:
        tokenizer.padding_side = "left"

    # Qwen base only has single token, so we need to set the special tokens
    if cfg.is_qwen_derived_model:
        token_ids = ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]
        for attr_name in token_ids:
            if getattr(tokenizer, attr_name) is None:
                setattr(tokenizer, attr_name, tokenizer.eod_id)

        token_names = ["bos_token", "eos_token", "pad_token", "unk_token"]
        for attr_name in token_names:
            if getattr(tokenizer, attr_name) is None:
                setattr(tokenizer, attr_name, "<|endoftext|>")

    additional_special_tokens = None
    if cfg.special_tokens:
        special_tokens = cfg.special_tokens.to_dict()
        additional_special_tokens = special_tokens.pop(
            "additional_special_tokens", None
        )
        lora_modules_to_save = get_linear_embedding_layers(model_config.model_type)
        for k, val in special_tokens.items():
            # check if new special token is not already in tokenizer and
            # is adapter training to make sure lora_modules_to_save is set
            # pylint: disable=too-many-boolean-expressions
            if (
                (getattr(tokenizer, k) is None or getattr(tokenizer, k) != val)
                and (len(tokenizer.encode(val, add_special_tokens=False)) > 2)
                and cfg.adapter
                and (
                    not cfg.lora_modules_to_save
                    or not all(
                        x in cfg.lora_modules_to_save for x in lora_modules_to_save
                    )
                )
                and k != "pad_token"
            ):
                lora_modules_to_save = ", ".join(
                    [f"`{x}`" for x in lora_modules_to_save]
                )
                raise ValueError(
                    f"Please set lora_modules_to_save to [{lora_modules_to_save}] when using an adapter and changing the special tokens."
                )

            tokenizer.add_special_tokens(
                {k: AddedToken(val, rstrip=False, lstrip=False, normalized=False)}
            )

        # If we add bos_token and eos_token, we need to update the post processor to
        # handle them correctly.
        # https://github.com/huggingface/transformers/pull/24132
        bos_or_eos_in_special_tokens = (
            "bos_token" in cfg.special_tokens and "eos_token" in cfg.special_tokens
        )
        if (
            tokenizer.__class__.__name__
            in (
                "LlamaTokenizerFast",
                "CodeLlamaTokenizerFast",
            )
            and bos_or_eos_in_special_tokens
        ):
            tokenizer.update_post_processor()

    if cfg.tokens:
        tokenizer.add_tokens(
            [
                AddedToken(token, rstrip=False, lstrip=False, normalized=False)
                for token in cfg.tokens
            ]
        )

    # Additional special tokens are a List, and need to be treated differently than regular special
    # tokens. We add them after we have called `add_tokens` in case these additional special tokens
    # are new tokens.
    #
    # Usage:
    #
    # ```py
    # special_tokens:
    #   additional_special_tokens: ["<|im_start|>", "<|im_end|>"]
    # ```
    if additional_special_tokens is not None:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": additional_special_tokens}
        )

    with zero_only():
        LOG.debug(f"EOS: {tokenizer.eos_token_id} / {tokenizer.eos_token}")
        LOG.debug(f"BOS: {tokenizer.bos_token_id} / {tokenizer.bos_token}")
        LOG.debug(f"PAD: {tokenizer.pad_token_id} / {tokenizer.pad_token}")
        LOG.debug(f"UNK: {tokenizer.unk_token_id} / {tokenizer.unk_token}")

    if cfg.chat_template:
        chat_template_string = get_chat_template_from_config(
            cfg=cfg,
            tokenizer=tokenizer,
        )
        if cfg.default_system_message and cfg.chat_template == "chatml":
            chat_template_string = chat_template_string.replace(
                "You are a helpful assistant.", cfg.default_system_message
            )

        tokenizer.chat_template = chat_template_string
    else:
        LOG.info(
            "No Chat template selected. Consider adding a chat template for easier inference."
        )
    return tokenizer


def load_processor(cfg: DictDefault, tokenizer: PreTrainedTokenizerBase):
    processor_kwargs: Dict[str, Any] = {}  # do we actually need this?

    processor_cls = AutoProcessor
    if cfg.processor_type:
        processor_cls = getattr(transformers, cfg.processor_type)

    processor = processor_cls.from_pretrained(
        cfg.processor_config,
        trust_remote_code=cfg.trust_remote_code or False,
        tokenizer=tokenizer,
        **processor_kwargs,
    )

    return processor


class ModelLoader:
    """
    ModelLoader: managing all the config and monkey patches while loading model
    """

    def __init__(
        self,
        cfg: DictDefault,
        tokenizer: PreTrainedTokenizerBase,
        *,
        processor: ProcessorMixin = None,  # pylint: disable=unused-argument
        inference: bool = False,
        reference_model: bool = False,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.inference: bool = inference
        self.reference_model: bool = reference_model

        # init model kwargs
        self.model_kwargs: Dict[str, Any] = {}
        if cfg.model_kwargs:
            for key, val in cfg.model_kwargs.items():
                self.model_kwargs[key] = val

        # init model
        self.model: PreTrainedModel
        self.base_model = cfg.base_model
        self.model_type = cfg.type_of_model

        # init model config
        self.model_config = load_model_config(cfg)
        if cfg.is_multimodal:
            self.text_model_config = self.model_config.text_config
        else:
            self.text_model_config = self.model_config

        self.AutoModelLoader = AutoModelForCausalLM  # pylint: disable=invalid-name

    def apply_patches(self) -> None:
        # load any patches from plugins
        from axolotl.integrations.base import PluginManager

        plugin_manager = PluginManager.get_instance()
        plugin_manager.pre_model_load(self.cfg)

        if self.cfg.gradient_checkpointing == "unsloth":
            transformers.modeling_utils.checkpoint = hf_grad_checkpoint_unsloth_wrapper

        if self.cfg.flash_attention:
            self.patch_attention()

        if self.cfg.sample_packing and self.cfg.s2_attention:
            raise ValueError(
                "Received `sample_packing=true` and `s2_attention=true`; however, \
            shifted-sparse attention does not currently support sample packing."
            )

        if (
            self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
            and self.cfg.flash_attention
            and self.cfg.sample_packing
        ):
            has_remote_code = (
                "auto_map" in self.model_config
                and "AutoModelForCausalLM" in self.model_config["auto_map"]
            )
            if has_remote_code and self.cfg.trust_remote_code is False:
                # if explicitly set in the YAML, we should prefer that, for example if explicitly disabled
                has_remote_code = self.cfg.trust_remote_code
            patch_for_multipack(
                self.cfg.model_config_type,
                model_name=self.cfg.base_model,
                has_remote_code=has_remote_code,
            )

            if self.cfg.is_llama_derived_model:
                self.patch_loss()
                if self.cfg.unsloth_lora_qkv or self.cfg.unsloth_lora_o:
                    from axolotl.monkeypatch.unsloth_ import patch_self_attn_lora

                    patch_self_attn_lora()
        elif self.cfg.is_llama_derived_model:
            self.patch_llama_derived_model()

        if (
            self.cfg.model_config_type == "mistral"
            and self.cfg.flash_attn_cross_entropy_loss
        ):
            from axolotl.monkeypatch.mistral_attn_hijack_flash import (
                patch_mistral_cross_entropy,
            )

            replace_stablelm_attn_with_flash_attn(cfg.base_model)

    if cfg.sample_packing and cfg.s2_attention:
        raise ValueError(
            "Received `sample_packing=true` and `s2_attention=true`; however, \
        shifted-sparse attention does not currently support sample packing."
        )

    if (
        cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
        and cfg.flash_attention
        and cfg.sample_packing
    ):
        patch_for_multipack(cfg.model_config_type, model_name=cfg.base_model)
    elif cfg.is_llama_derived_model:
        # Modify all llama derived models in one block

        if self.cfg.flash_attention:
            from axolotl.monkeypatch.llama_attn_hijack_flash import (
                replace_llama_attn_with_flash_attn,
            )

            if self.cfg.sample_packing:
                if self.cfg.device not in ["mps", "cpu"] and not self.inference:
                    LOG.info("patching with flash attention for sample packing")
                    replace_llama_attn_with_flash_attn(
                        packed=True,
                        cross_entropy=self.cfg.flash_attn_cross_entropy,
                        rms_norm=self.cfg.flash_attn_rms_norm,
                    )
            elif self.cfg.s2_attention:
                LOG.info("patching w/ flash-enabled, shifted-sparse attention")
                replace_llama_attn_with_flash_attn(
                    packed=False,
                    cross_entropy=self.cfg.flash_attn_cross_entropy,
                    rms_norm=self.cfg.flash_attn_rms_norm,
                    use_shifted_sparse_attn=True,
                )
        elif cfg.xformers_attention:
            from axolotl.monkeypatch.llama_attn_hijack_xformers import (
                hijack_llama_attention,
            )

            LOG.info("patching with xformers attention")
            hijack_llama_attention()
        elif self.cfg.sample_packing:
            from axolotl.monkeypatch.llama_patch_multipack import (
                hijack_llama_prepare_4d_mask,
            )

            LOG.info("patching llama _prepare_4d_causal_attention_mask*")
            hijack_llama_prepare_4d_mask()
        elif self.cfg.s2_attention:
            raise NotImplementedError(
                "Shifted-sparse attention not currently implemented without flash attention."
            )

        if self.cfg.unsloth_cross_entropy_loss:
            from axolotl.monkeypatch.unsloth_ import integrate_cross_entropy_loss_patch

            integrate_cross_entropy_loss_patch()

        if self.cfg.unsloth_lora_qkv or self.cfg.unsloth_lora_o:
            from axolotl.monkeypatch.unsloth_ import patch_self_attn_lora

            patch_self_attn_lora()

    # Modify mistral derived models
    if (
        cfg.model_config_type == "mistral"
        and cfg.flash_attention
        and cfg.sample_packing
    ):
        from axolotl.monkeypatch.mistral_attn_hijack_flash import (
            replace_mistral_attn_with_flash_attn,
        )

        LOG.info("patching mistral with flash attention")
        replace_mistral_attn_with_flash_attn(packed=cfg.sample_packing)

    if cfg.is_llama_derived_model and cfg.sample_packing and not inference:
        from axolotl.monkeypatch.llama_expand_mask import hijack_expand_mask

        LOG.info("patching _expand_mask")
        hijack_expand_mask()

    model_kwargs: Dict[str, Any] = {}

    if cfg.model_kwargs:
        for key, val in cfg.model_kwargs.items():
            model_kwargs[key] = val

    max_memory = cfg.max_memory
    device_map = cfg.device_map

    if cfg.gpu_memory_limit:
        gpu_memory_limit = (
            str(cfg.gpu_memory_limit) + "GiB"
            if isinstance(cfg.gpu_memory_limit, int)
            else cfg.gpu_memory_limit
        )

            max_memory = {}
            num_device = get_device_count()
            for i in range(num_device):
                max_memory[i] = gpu_memory_limit
            max_memory["cpu"] = "256GiB"  # something sufficiently large to fit anything

        if max_memory is not None:
            # Based on https://github.com/togethercomputer/OpenChatKit/blob/main/inference/bot.py
            from accelerate import infer_auto_device_map

            with init_empty_weights():
                model_canvas = self.AutoModelLoader.from_config(
                    self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                )
            model_canvas.tie_weights()
            device_map = infer_auto_device_map(
                model_canvas,
                max_memory=max_memory,
                dtype=self.cfg.torch_dtype,
            )
            # We can discard max_memory now as we have a device map set up for us
            max_memory = None

        self.model_kwargs["device_map"] = device_map
        self.model_kwargs["torch_dtype"] = self.cfg.torch_dtype

        cur_device = get_device_type()
        if "mps" in str(cur_device):
            self.model_kwargs["device_map"] = "mps:0"
        elif "npu" in str(cur_device):
            self.model_kwargs["device_map"] = "npu:0"

        # TODO can we put the reference model on it's own gpu? I think we have to move logits around to calculate loss
        # if cfg.rl:
        #     if torch.cuda.device_count() > 1:
        #         if reference_model:
        #             model_kwargs["device_map"] = "cuda:" + str(
        #                 torch.cuda.current_device() + 1
        #             )
        #         else:
        #             model_kwargs["device_map"] = "cuda:" + str(torch.cuda.current_device())

        if is_deepspeed_zero3_enabled():
            del self.model_kwargs["device_map"]

    if cfg.revision_of_model:
        model_kwargs["revision"] = cfg.revision_of_model

    if cfg.gptq:
        if not hasattr(model_config, "quantization_config"):
            LOG.warning("model config does not contain quantization_config information")
        else:
            if cfg.gptq_disable_exllama is not None:
                model_config.quantization_config[
                    "disable_exllama"
                ] = cfg.gptq_disable_exllama
            model_kwargs["quantization_config"] = GPTQConfig(
                **model_config.quantization_config
            )
    if cfg.adapter == "qlora" and cfg.load_in_4bit:
        bnb_config = {
            "load_in_4bit": True,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
            "bnb_4bit_compute_dtype": cfg.torch_dtype,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_quant_storage": torch.bfloat16,
        }
        if cfg.model_config_type in ["jamba", "qwen2_moe"] and not cfg.deepspeed:
            # for some reason, this causes the loss to be off by an order of magnitude
            # but deepspeed needs this still in bfloat16
            bnb_config["bnb_4bit_quant_storage"] = torch.float32

            if self.cfg.bnb_config_kwargs:
                bnb_config.update(self.cfg.bnb_config_kwargs)

            self.model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **bnb_config,
            )
        elif self.cfg.adapter == "lora" and self.model_kwargs["load_in_8bit"]:
            bnb_config = {
                "load_in_8bit": True,
            }
            # Exclude mamba blocks from int8 quantization for jamba
            if self.cfg.model_config_type == "jamba":
                bnb_config["llm_int8_skip_modules"] = ["mamba"]
            self.model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **bnb_config,
            )

        # no longer needed per https://github.com/huggingface/transformers/pull/26610
        if "quantization_config" in self.model_kwargs or self.cfg.gptq:
            self.model_kwargs.pop("load_in_8bit", None)
            self.model_kwargs.pop("load_in_4bit", None)

    def set_attention_config(self) -> None:
        """
        sample packing uses custom FA2 patch
        """
        if self.cfg.flash_attention:
            if not self.cfg.sample_packing and self.cfg.s2_attention:
                pass
            self.model_kwargs["attn_implementation"] = "flash_attention_2"
            self.model_config._attn_implementation = (  # pylint: disable=protected-access
                "flash_attention_2"
            )
        elif self.cfg.sdp_attention:
            self.model_kwargs["attn_implementation"] = "sdpa"
            self.model_config._attn_implementation = (  # pylint: disable=protected-access
                "sdpa"
            )
        elif self.cfg.eager_attention:
            self.model_kwargs["attn_implementation"] = "eager"
            self.model_config._attn_implementation = (  # pylint: disable=protected-access
                "eager"
            )

        if self.cfg.low_cpu_mem_usage:
            self.model_kwargs["low_cpu_mem_usage"] = True

    def build_model(self, qlora_fsdp) -> bool:
        def _configure_zero3_memory_efficient_loading():
            """
            Set the deepspeed config to load the model into RAM first before moving to VRAM.

            We need to return hf_ds_cfg as it needs to exist before model loading.
            """
            hf_ds_cfg = None

            if os.getenv("ACCELERATE_DEEPSPEED_ZERO_STAGE") == "3":
                hf_ds_cfg = HfTrainerDeepSpeedConfig(self.cfg.deepspeed)
                hf_ds_cfg.fill_match(
                    "train_micro_batch_size_per_gpu", self.cfg.micro_batch_size
                )
                hf_ds_cfg.fill_match(
                    "gradient_accumulation_steps", self.cfg.gradient_accumulation_steps
                )
                hf_ds_cfg.fill_match(
                    "train_batch_size",
                    int(os.getenv("WORLD_SIZE", "1"))
                    * self.cfg.micro_batch_size
                    * self.cfg.gradient_accumulation_steps,
                )
                if "device_map" in self.model_kwargs:
                    del self.model_kwargs["device_map"]

                transformers.modeling_utils.is_deepspeed_zero3_enabled = lambda: True
                transformers.integrations.deepspeed.is_deepspeed_zero3_enabled = (
                    lambda: True
                )

            return hf_ds_cfg

        skip_move_to_device = False
        if (  # pylint: disable=condition-evals-to-constant)
            (self.cfg.fsdp and self.cfg.fsdp_config.fsdp_cpu_ram_efficient_loading)
            and not qlora_fsdp
            and False
        ):
            self.model = load_sharded_model(
                self.base_model,
                self.model_config,
                self.cfg,
                torch_dtype=self.cfg.torch_dtype,
            )
            skip_move_to_device = True
        elif (
            qlora_fsdp
            and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
            and cfg.model_config_type == "dbrx"
        ):
            quant_storage = cfg.torch_dtype
            model = load_sharded_model_quant(
                base_model,
                model_config,
                cfg,
                quant_storage=quant_storage,
            )
            skip_move_to_device = True
        elif (
            self.model_config.model_type == "llama"
            and not self.cfg.trust_remote_code
            and not self.cfg.gptq
        ):
            if qlora_fsdp and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading:
                skip_move_to_device = True
                if "device_map" in self.model_kwargs:
                    del self.model_kwargs["device_map"]

            _ = _configure_zero3_memory_efficient_loading()

            if self.cfg.is_multimodal:
                self.model_config.text_config = self.text_model_config
            self.model = self.AutoModelLoader.from_pretrained(
                self.base_model,
                config=self.model_config,
                **self.model_kwargs,
            )

            #  TODO (MengqingCao) split these patches seperately
            if self.cfg.flash_attention and not self.inference:
                from axolotl.monkeypatch.llama_attn_hijack_flash import (
                    is_xformers_swiglu_available,
                    replace_llama_mlp_with_swiglu,
                    replace_llama_qkv_with_fused,
                )

                if self.cfg.flash_attn_fuse_mlp and is_xformers_swiglu_available():
                    LOG.info("patching with SwiGLU")
                    replace_llama_mlp_with_swiglu(self.model)

                if self.cfg.flash_attn_fuse_qkv:
                    LOG.info("patching with fused QKV")
                    replace_llama_qkv_with_fused(self.model)
        elif self.model_type == "MambaLMHeadModel":
            # FIXME this is janky at best and hacked together to make it work
            MambaLMHeadModel = fix_mamba_attn_for_loss()  # pylint: disable=invalid-name

            self.model_kwargs["dtype"] = self.model_kwargs["torch_dtype"]
            self.model_kwargs["device"] = torch.cuda.current_device()
            del self.model_kwargs["torch_dtype"]
            del self.model_kwargs["device_map"]

            self.model = MambaLMHeadModel.from_pretrained(
                self.base_model,
                **self.model_kwargs,
            )
        elif (
            self.model_type
            and self.model_type != "AutoModelForCausalLM"
            and not self.cfg.trust_remote_code
        ):
            if self.cfg.is_multimodal:
                self.model_config.text_config = self.text_model_config
            if self.cfg.gptq:
                self.model = self.AutoModelLoader.from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
            else:
                self.model = getattr(transformers, self.model_type).from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
        else:
            # Shouldn't be a problem most of the time. will obviously error if the model doesn't support this
            # when training starts
            if (
                hasattr(self.text_model_config, "max_seq_len")
                and self.text_model_config.max_seq_len
                and self.cfg.sequence_len > self.text_model_config.max_seq_len
            ):
                self.text_model_config.max_seq_len = self.cfg.sequence_len
                LOG.warning(f"increasing context length to {self.cfg.sequence_len}")
            elif (
                hasattr(self.text_model_config, "max_sequence_length")
                and self.text_model_config.max_sequence_length
                and self.cfg.sequence_len > self.text_model_config.max_sequence_length
            ):
                self.text_model_config.max_sequence_length = self.cfg.sequence_len
                LOG.warning(f"increasing context length to {self.cfg.sequence_len}")
            if self.cfg.gptq:
                if self.cfg.is_multimodal:
                    self.model_config.text_config = self.text_model_config
                self.model = self.AutoModelLoader.from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
            else:
                if qlora_fsdp and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading:
                    # disabling either of these two still leads to VRAM spike before setting back down
                    skip_move_to_device = True
                    if "device_map" in self.model_kwargs:
                        del self.model_kwargs["device_map"]

                _ = _configure_zero3_memory_efficient_loading()

                if self.cfg.is_multimodal:
                    self.model_config.text_config = self.text_model_config
                self.model = self.AutoModelLoader.from_pretrained(
                    self.base_model,
                    config=self.model_config,
                    trust_remote_code=self.cfg.trust_remote_code or False,
                    **self.model_kwargs,
                )
        if is_deepspeed_zero3_enabled():
            skip_move_to_device = True

        return skip_move_to_device

    def ajust_model_config(self) -> None:
        if (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "max_position_embeddings")
            and self.model.config.max_position_embeddings
            and self.cfg.sequence_len > self.model.config.max_position_embeddings
        ):
            LOG.warning(
                f"increasing model.config.max_position_embeddings from {self.model.config.max_position_embeddings} to {self.cfg.sequence_len}"
            )
            self.model.config.max_position_embeddings = self.cfg.sequence_len

        if (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "bos_token_id")
            and self.model.config.bos_token_id
            and self.model.config.bos_token_id != self.tokenizer.bos_token_id
        ):
            self.model.config.bos_token_id = self.tokenizer.bos_token_id

        if (
            hasattr(self.model, "config")
            and hasattr(self.model.config, "eos_token_id")
            and self.model.config.eos_token_id
            and self.model.config.eos_token_id != self.tokenizer.eos_token_id
        ):
            self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def set_z3_leaf_modules(self) -> None:
        from deepspeed.utils import (  # pylint: disable=no-name-in-module
            set_z3_leaf_modules,
        )

        if cfg.model_config_type == "mixtral":
            moe_block = get_module_class_from_name(model, "MixtralSparseMoeBlock")
            set_z3_leaf_modules(model, [moe_block])
        elif cfg.model_config_type == "dbrx":
            moe_block = get_module_class_from_name(model, "DbrxFFN")
            set_z3_leaf_modules(model, [moe_block])

    def prepare_model(self, qlora_fsdp) -> None:
        skip_prepare_model_for_kbit_training = False
        if self.cfg.model_config_type == "qwen" and self.cfg.adapter == "lora":
            # Qwen doesn't play nicely with LoRA if this is enabled
            skip_prepare_model_for_kbit_training = True

        loftq_bits = (
            self.cfg.peft
            and self.cfg.peft.loftq_config
            and self.cfg.peft.loftq_config.loftq_bits
        )
        if self.cfg.adapter == "lora" and loftq_bits:
            skip_prepare_model_for_kbit_training = True

        if qlora_fsdp or (
            self.cfg.fsdp and self.cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        ):
            # make sure everything is in the same dtype
            skip_prepare_model_for_kbit_training = True

    if cfg.adapter in ["lora", "qlora"]:
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=cfg.gradient_checkpointing_kwargs
            )
        if (
            cfg.load_in_8bit or cfg.load_in_4bit
        ) and not skip_prepare_model_for_kbit_training:
            LOG.info("converting PEFT model w/ prepare_model_for_kbit_training")
            self.model = prepare_model_for_kbit_training(
                self.model, use_gradient_checkpointing=self.cfg.gradient_checkpointing
            )

    def convert_embedding_modules_dtype(
        self, embedding_modules, dist_dtype, before_kbit_train_or_finetune
    ) -> None:
        for name, module in self.model.named_modules():
            if "norm" in name:
                module.to(dist_dtype)
            if before_kbit_train_or_finetune:
                if name.endswith(".gate"):
                    module.to(dist_dtype)
                if self.model_config.model_type == "btlm":
                    # don't upcast lm_head for btlm
                    continue
            if any(m in name for m in embedding_modules):
                if hasattr(module, "weight"):
                    module.to(cfg.torch_dtype)


    LOG.info("HELLO!!!!!!")
    LOG.info(f"Sparse training: {cfg.sparse_training}")
    LOG.info(f"Sparsity level: {cfg.sparsity_level}")
    LOG.info(cfg)
    # TODO: JAMES SPARSE MONKEYPATCH
    if cfg.sparse_training is not None and cfg.sparse_training > 0:
        from transformers import LlamaForCausalLM
        assert isinstance(model, LlamaForCausalLM), "Only Llama models supported for sparse training for now"

        # TODO: update later with relevant hyperparams once I finalize
        LOG.info(f"Sparse training with sparsity level {cfg.sparsity_level}, using distributions in {cfg.histogram_path}")

        from axolotl.monkeypatch.sparse_utils import monkeypatch_sparse_forward
        monkeypatch_sparse_forward(model, cfg.histogram_path,cfg.sparsity_level, cfg.lookup_path)


    lora_config = None
    if not reference_model or cfg.lora_model_dir:
        # if we're not loading the reference model, then we're loading the model for training
        # then the dpo trainer doesn't want the peft model loaded over it, it just wants the lora/peft config
        if cfg.adapter and cfg.rl in ["dpo", "ipo", "kto"] and not cfg.merge_lora:
            _, lora_config = load_lora(model, cfg, inference=False, config_only=True)
        else:
            model, lora_config = load_adapter(model, cfg, cfg.adapter)

    if (
        cfg.ddp
        and not load_in_8bit
        and not (cfg.rl and cfg.load_in_4bit)
        and not skip_move_to_device
    ):
        # TODO revaldate this conditional
        model.to(f"cuda:{cfg.local_rank}")

        if get_device_count() > 1 and int(os.getenv("WORLD_SIZE", "1")) == 1:
            setattr(self.model, "is_parallelizable", True)
            setattr(self.model, "model_parallel", True)

        # ---------------------------------------------------------
        #  parameters that require gradient updates
        # ---------------------------------------------------------
        requires_grad = []
        for name, param in self.model.named_parameters(recurse=True):
            if param.requires_grad:
                requires_grad.append(f"{name}: {param.requires_grad}")
        if len(requires_grad) == 0:
            LOG.warning("there are no parameters that require gradient updates")
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        if self.cfg.flash_optimum:
            from optimum.bettertransformer import BetterTransformer

            self.model = BetterTransformer.transform(self.model)

        if self.cfg.adapter is not None:
            log_gpu_memory_usage(LOG, "after adapters", self.model.device)

    if cfg.unsloth_lora_mlp:
        from axolotl.monkeypatch.unsloth_ import integrate_lora_mlp_patch

        integrate_lora_mlp_patch(model)
    if cfg.unsloth_lora_qkv or cfg.unsloth_lora_o:
        from axolotl.monkeypatch.unsloth_ import integrate_lora_patch

        integrate_lora_patch(model, cfg)

    # TODO resume_from_checkpoint handling
    return model, lora_config


def load_adapter(model, cfg, adapter, inference=False):
    # type: (PreTrainedModel, DictDefault, Optional[str], bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    if adapter is None:
        return model, None
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if adapter in ["lora", "qlora"]:
        return load_lora(model, cfg, inference=inference)
    if adapter == "llama-adapter":
        return load_llama_adapter(model, cfg)

    raise NotImplementedError(f"{adapter} peft adapter not available")


def load_llama_adapter(model, cfg):
    # type: (PreTrainedModel, DictDefault) -> Tuple[PreTrainedModel, Optional[PeftConfig]]
    from peft import AdaptionPromptConfig, get_peft_model

    peft_config = AdaptionPromptConfig(
        adapter_layers=cfg.peft_adapter.layers,  # layers (L)
        adapter_len=cfg.peft_adapter.len,  # prompt length (K)
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - llama_adapter")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            torch_dtype=torch.float16,
        )
    else:
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    embedding_modules = get_linear_embedding_layers(model.config.model_type)
    output_embedding = embedding_modules[1]
    if output_embedding in lora_module_names:  # needed for 16-bit
        lora_module_names.remove(output_embedding)

    return list(lora_module_names)


def setup_quantized_meta_for_peft(model: nn.Module):
    """Replaces `quant_state.to` with a dummy function to prevent PEFT from moving `quant_state` to meta device"""

    def temp_to_method(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self

    for param in model.parameters():
        if isinstance(param, Params4bit):
            param.quant_state._orig_to = (  # pylint: disable=protected-access
                param.quant_state.to
            )
            param.quant_state.to = types.MethodType(temp_to_method, param.quant_state)


def setup_quantized_peft_meta_for_training(model: nn.Module):
    """Replaces dummy `quant_state.to` method with the original function to allow training to continue"""
    for param in model.parameters():
        if isinstance(param, Params4bit) and hasattr(param.quant_state, "_orig_to"):
            param.quant_state.to = (
                param.quant_state._orig_to  # pylint: disable=protected-access
            )
            param.quant_state._orig_to = None  # pylint: disable=protected-access


def load_lora(model, cfg, inference=False, config_only=False):
    # type: (PreTrainedModel, DictDefault, bool, bool) -> Tuple[Optional[PreTrainedModel], Optional[PeftConfig]]

    from peft import LoraConfig, get_peft_model

    lora_target_modules = cfg.lora_target_modules or []

    if cfg.lora_target_linear:
        linear_names = find_all_linear_names(model)
        LOG.info(f"found linear modules: {repr(sorted(linear_names))}")
        lora_target_modules_as_list = (
            lora_target_modules
            if isinstance(lora_target_modules, list)
            else [lora_target_modules]
        )
        lora_target_modules = list(set(lora_target_modules_as_list + linear_names))

    lora_config_kwargs = {}
    loftq_bits = cfg.peft and cfg.peft.loftq_config and cfg.peft.loftq_config.loftq_bits
    if loftq_bits:
        lora_config_kwargs["loftq_config"] = LoftQConfig(loftq_bits=loftq_bits)
        lora_config_kwargs["init_lora_weights"] = "loftq"
    if cfg.peft_use_dora:
        lora_config_kwargs["use_dora"] = cfg.peft_use_dora
    if cfg.peft_use_rslora:
        lora_config_kwargs["use_rslora"] = cfg.peft_use_rslora
    if cfg.peft_layer_replication:
        lora_config_kwargs["layer_replication"] = cfg.peft_layer_replication

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        layers_to_transform=cfg.peft_layers_to_transform,
        layers_pattern=cfg.peft_layers_pattern,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
        **lora_config_kwargs,
    )

    if config_only:
        return None, lora_config

    rank = int(os.environ.get("LOCAL_RANK", 0))

    if (
        cfg.fsdp
        and cfg.adapter
        and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_meta_for_peft(model)

    if cfg.lora_model_dir:
        LOG.debug("Loading pretrained PEFT - LoRA")
        model_kwargs: Any = {}
        if cfg.lora_on_cpu:
            model_kwargs["max_memory"] = {"cpu": "256GiB"}
            model_kwargs["device_map"] = {"": "cpu"}
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            is_trainable=(not inference),
            **model_kwargs,
        )
    else:
        model = get_peft_model(model, lora_config)

    if rank == 0:
        try:
            model.print_trainable_parameters()
        except AttributeError as exc:
            LOG.warning(
                "Exception caught during model.print_trainable_parameters(): %s", exc
            )
    elif (
        cfg.fsdp
        and cfg.adapter
        and cfg.fsdp_config.fsdp_cpu_ram_efficient_loading
        and rank != 0
    ):
        setup_quantized_peft_meta_for_training(model)

    return model, lora_config


def ensure_dtype(model, dtype=torch.bfloat16):
    for name, module in model.named_modules():
        weight_mismatch = False
        bias_mismatch = False
        try:
            weight_mismatch = module.weight.dtype != dtype
        except AttributeError:
            pass
        try:
            bias_mismatch = module.bias.dtype != dtype
        except AttributeError:
            pass

        if weight_mismatch:
            print(f"Converting module {name}.weight: {module.weight.dtype} -> {dtype}")
        if bias_mismatch:
            print(f"Converting module {name}.bias: {module.bias.dtype} -> {dtype}")
        if weight_mismatch or bias_mismatch:
            module.to(dtype)
