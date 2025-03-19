python3 peft_lora_eval.py \
    --task_name mrpc \
    --max_seq_length 512 \
    --padding max_length \
    --model_name roberta-base \
    --lora_model_dir "./lora_finetuned_model_stsb" \