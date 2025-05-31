## How to Fine-tune DeepSeek-Coder

We provide script `finetune_deepseekcoder.py` for users to finetune our models on downstream tasks.

The script supports the training with [DeepSpeed](https://github.com/microsoft/DeepSpeed). You need install required packages by:

```bash
pip install -r requirements.txt
```

Please follow [Sample Dataset Format](https://huggingface.co/datasets/nickrosh/Evol-Instruct-Code-80k-v1) to prepare your training data.
Each line is a json-serialized string with two required fields `instruction` and `output`.

After data preparation, you can use the sample shell script to finetune `deepseek-ai/deepseek-coder-6.7b-instruct`.
Remember to specify `DATA_PATH`, `OUTPUT_PATH`.
And please choose appropriate hyper-parameters(e.g., `learning_rate`, `per_device_train_batch_size`) according to your scenario.

```bash
DATA_PATH="<your_data_path>"
OUTPUT_PATH="<your_output_path>"
MODEL_PATH="deepseek-ai/deepseek-coder-6.7b-instruct"

deepspeed finetune_deepseekcoder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 3 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True
```

tensorboard --logdir runs --port 6007

```bash
# 1. 设置环境变量 (根据您的实际情况修改)
export MODEL_PATH="../../model/deepseek-ai/deepseek-coder-1.3b-instruct"
export DATASET_PATH="train_fixed_3.6v6.4"
export OUTPUT_PATH="../output_fixed_3.6v6.4"
export DS_CONFIG_PATH="../ds_config_single_gpu.json"

# 2. 计算总步数并调整保存策略 (可选，但推荐)
# num_samples=3000
# batch_size_per_device=4
# grad_accum_steps=4
# epochs=3
# total_steps=$(( (num_samples / (batch_size_per_device * grad_accum_steps)) * epochs ))
# echo "Total training steps: $total_steps"
# save_steps_value=200 # 例如每200步保存一次

# 3. 运行训练命令
# 使用 torchrun 启动单GPU训练 (NPROC_PER_NODE=1)
    --deepspeed $DS_CONFIG_PATH \
torchrun --nproc_per_node=1 src/train.py \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn fa2\
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_PATH \
    --template deepseek \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 1024 \
    --save_steps 40 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16
```

