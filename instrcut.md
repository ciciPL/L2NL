```bash
export MODEL_PATH="../../model/deepseek-ai/deepseek-coder-1.3b-instruct"
export DATASET_PATH="train_fixed_3.6v6.4_without_sentence"
export OUTPUT_PATH="../output_fixed_3.6v6.4"

torchrun --nproc_per_node=1 src/train.py \
    --stage sft \
    --do_train \
    --do_eval \
    --use_fast_tokenizer \
    --flash_attn fa2 \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_PATH \
    --template deepseek \
    --finetuning_type lora \
    --lora_target all \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --cutoff_len 1024 \
    --save_steps 200 \
    --plot_loss \
    --num_train_epochs 2 \
    --bf16 \
    --val_size 0.1 \
    --eval_strategy="steps" \
    --eval_steps 200 \
    --load_best_model_at_end
    
    # 可选：根据需要调整lora_rank和lora_alpha
    # --lora_rank 32 \
    # --lora_alpha 64