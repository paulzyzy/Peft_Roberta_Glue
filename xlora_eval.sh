python3 peft_xlora_eval.py \
    --task_name mrpc \
    --max_seq_length 512 \
    --padding max_length \
    --model_name roberta-base \
    --lora_model_dir "./xlora_finetuned_model_4adapters/xlora_finetuned_model_1" \