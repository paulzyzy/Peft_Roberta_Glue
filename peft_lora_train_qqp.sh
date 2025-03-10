python3 peft_lora_train.py \
  --task_name "qqp" \
  --max_seq_length 512 \
  --padding "max_length" \
  --model_name "roberta-base" \
  --learning_rate 5e-4 \
  --num_train_epochs 25 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --weight_decay 0.1 \
  --logging_steps 10 \
  --fp16 True \
  --lora_r 8 \
  --lora_alpha 16 \
  --output_dir "./lora_results" \
  --logging_dir "./lora_logs" \
  --eval_strategy "epoch" \
  --save_strategy "no" \