# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
from transformers import GenerationConfig
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        self._debug_input_data = []  # 初始化一个用于调试的列表
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
            self,
            model: "torch.nn.Module",
            inputs: dict[str, Union["torch.Tensor", Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        # ****************** 在这里添加你的修改 ******************

        # 1. 确保 max_new_tokens 被正确传递
        # 尝试从 self.args (SFTTrainingArguments 实例) 中直接获取 max_new_tokens
        # LLaMA-Factory 的命令行参数最终会聚合到这个 args 对象中
        if "max_new_tokens" not in gen_kwargs:
            # 使用 getattr 安全地获取 max_new_tokens，如果不存在则默认为40
            gen_kwargs["max_new_tokens"] = getattr(self.args, "max_new_tokens", 25)

        # 获取实际的 max_new_tokens 值，用于计算 max_length
        effective_max_new_tokens = gen_kwargs.get("max_new_tokens", 0)

        # 2. 动态调整 max_length (增强鲁棒性)
        # 尝试从 self.args 中直接获取 cutoff_len
        # 如果 self.args.cutoff_len 仍然报错，请提供 LLaMA-Factory 的具体版本
        # LLaMA-Factory 0.5.0 及更高版本通常会将 cutoff_len 直接放在 SFTTrainingArguments 中
        current_cutoff_len = getattr(self.args, "cutoff_len", 1024)  # 从 self.args 获取 cutoff_len

        if "max_length" not in gen_kwargs or \
                gen_kwargs["max_length"] < inputs["input_ids"].size(-1) + effective_max_new_tokens:
            calculated_max_length = current_cutoff_len + effective_max_new_tokens + 10  # 10 是一个安全余量
            gen_kwargs["max_length"] = max(gen_kwargs.get("max_length", 0), calculated_max_length)

        # 3. 消除 "Setting `pad_token_id` to `eos_token_id`" 警告
        # 确保 model.config.pad_token_id 被设置
        # self.tokenizer 应该在 Trainer 实例中是可用的
        if model.config.pad_token_id is None and hasattr(self, 'tokenizer') and self.tokenizer.pad_token_id is not None:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        elif model.config.pad_token_id is None and hasattr(self,
                                                           'tokenizer') and self.tokenizer.eos_token_id is not None:
            model.config.pad_token_id = self.tokenizer.eos_token_id

        # ****************** 修改结束 ******************

        # =====================> 在这里插入代码 <=====================
        # 解码 prompt 文本，用于和 input_ids 一起保存
        decoded_batch_prompts = self.processing_class.batch_decode(
            inputs["input_ids"], skip_special_tokens=False
        )
        # 遍历当前批次中的每一个样本
        for i in range(len(decoded_batch_prompts)):
            self._debug_input_data.append({
                "prompt": decoded_batch_prompts[i],
                # 将 PyTorch 张量转换为 Python 列表
                "input_ids": inputs["input_ids"][i].tolist()
            })
        # =====================> 插入代码结束 <=====================
        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            # 使用 getattr 安全地获取 pad_token_id
            pad_id_for_masking = getattr(self.tokenizer, 'pad_token_id', -1)  # 如果没有pad_token_id，用一个不可能的ID
            generated_tokens[:, : inputs["input_ids"].size(-1)] = pad_id_for_masking
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
            self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        # ------------------ 第一部分：计算解码后的文本（保持不变） ------------------
        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        # 写入第一个文件 (generated_predictions.jsonl)
        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")

        # ------------------ 第二部分：保存原始ID和解码文本（核心修改） ------------------
        output_ids_file = os.path.join(self.args.output_dir, "raw_predictions.jsonl")
        logger.info_rank0(f"Saving raw prediction IDs with decoded text to {output_ids_file}")

        raw_input_ids = dataset["input_ids"]

        with open(output_ids_file, "w", encoding="utf-8") as f:
            # 遍历所有样本
            for i in range(len(raw_input_ids)):
                # 准备ID列表
                input_id_list = raw_input_ids[i]
                predict_id_list = preds[i].tolist()
                label_id_list = labels[i].tolist()

                # 【【【关键修改】】】
                # 创建一个包含所有需要信息的字典
                record = {
                    "input_ids": input_id_list,
                    "predict_ids": predict_id_list,
                    "label_ids": label_id_list,
                    "decoded_input": decoded_inputs[i],  # 新增解码后的输入文本
                    "decoded_predict": decoded_preds[i]  # 新增解码后的预测文本
                }

                # 将记录写入JSONL文件
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
