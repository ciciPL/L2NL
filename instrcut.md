```bash
export MODEL_PATH="../../model/deepseek-ai/deepseek-coder-1.3b-instruct"
export DATASET_PATH="train_fixed_3.6v6.4_with_sentence"
export EVAL_PATH="val_fixed_3.6v6.4_with_sentence"
export OUTPUT_PATH="../output_train_fixed_3.6v6.4_with_sentence"

torchrun --nproc_per_node=1 src/train.py \
    --stage sft \
    --do_train \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --use_fast_tokenizer \
    --metric_for_best_model eval_rouge-l \
    --greater_is_better True \
    --flash_attn fa2 \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATASET_PATH \
    --eval_dataset $EVAL_PATH \
    --template deepseek \
    --finetuning_type lora \
    --lora_target "q_proj","v_proj" \
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --weight_decay 0.005 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --cutoff_len 1024 \
    --save_steps 400 \
    --plot_loss \
    --num_train_epochs 8 \
    --bf16 \
    --eval_strategy="steps" \
    --eval_steps 400 \
    --max_new_tokens 40 \
    --temperature 0.1 \
    --top_p 0.9 \
    --repetition_penalty 1.5 \
    --load_best_model_at_end
    
        --early_stopping_patience \
    --resume_from_checkpoint  '../output_6k_sentence/checkpoint-700' \
        --report_to none \
    --val_size 0.1 \    
      
    # 可选：根据需要调整lora_rank和lora_alpha
    # --lora_rank 32 \
    # --lora_alpha 64