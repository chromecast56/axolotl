"""
E2E tests for lora llama
"""

import logging
import os
import unittest
from pathlib import Path

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config
from axolotl.utils.dict import DictDefault

from .utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestLlamaVision(unittest.TestCase):
    """
    Test case for Llama Vision models
    """

    @with_temp_dir
    def test_llama_vision_text_only_dataset(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "alpindale/Llama-3.2-11B-Vision-Instruct",
                "processor_type": "AutoProcessor",
                "skip_prepare_dataset": True,
                "remove_unused_columns": False,
                "sample_packing": False,
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": r"language_model.model.layers.[\d]+.(mlp|cross_attn|self_attn).(up|down|gate|q|k|v|o)_proj",
                "val_set_size": 0,
                "chat_template": "llama3_2_vision",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model-00001-of-00005.safetensors").exists()

    @with_temp_dir
    def test_llama_vision_multimodal_dataset(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "alpindale/Llama-3.2-11B-Vision-Instruct",
                "processor_type": "AutoProcessor",
                "skip_prepare_dataset": True,
                "remove_unused_columns": False,
                "sample_packing": False,
                "sequence_len": 1024,
                "load_in_8bit": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": r"language_model.model.layers.[\d]+.(mlp|cross_attn|self_attn).(up|down|gate|q|k|v|o)_proj",
                "val_set_size": 0,
                "chat_template": "llama3_2_vision",
                "datasets": [
                    {
                        "path": "HuggingFaceH4/llava-instruct-mix-vsft",
                        "type": "chat_template",
                        "split": "train[:1%]",
                        "field_messages": "messages",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
            }
        )
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model-00001-of-00005.safetensors").exists()
