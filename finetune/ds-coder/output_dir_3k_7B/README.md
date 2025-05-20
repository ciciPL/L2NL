---
library_name: peft
license: other
base_model: ../../model/deepseek-coder-6.7b-instruct
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: output_dir_3k_7B
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output_dir_3k_7B

This model is a fine-tuned version of [../../model/deepseek-coder-6.7b-instruct](https://huggingface.co/../../model/deepseek-coder-6.7b-instruct) on the train_fixed dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2401

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.898         | 0.5229 | 50   | 1.7155          |
| 1.6158        | 1.0523 | 100  | 1.4485          |
| 1.4576        | 1.5752 | 150  | 1.3375          |
| 1.0714        | 2.1046 | 200  | 1.2693          |
| 1.2081        | 2.6275 | 250  | 1.2401          |


### Framework versions

- PEFT 0.15.1
- Transformers 4.51.0
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1