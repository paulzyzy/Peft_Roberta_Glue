python3 "./peft_xlora_train.py" \
  --task_name "mrpc" \
  --max_seq_length 512 \
  --padding "max_length" \
  --model_name "roberta-base" \
  --learning_rate 4e-4 \
  --num_train_epochs 30 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --weight_decay 0.1 \
  --logging_steps 10 \
  --fp16 True \
  --output_dir "./lora_results" \
  --logging_dir "./lora_logs" \
  --eval_strategy "epoch" \
  --save_strategy "no" \
  --xlora_depth 8 \