#!/usr/bin/env python
import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
import evaluate
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    default_data_collator,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    AutoConfig,
)
from peft import LoraConfig, get_peft_model
from Xlora.xlora import add_xlora_to_model
from Xlora.xlora_config import xLoRAConfig

# Setup logging
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA fine-tuned model on a GLUE task")
    parser.add_argument("--task_name", type=str, default="cola", help="GLUE task name (e.g. cola, qnli, etc.)")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--padding", type=str, default="max_length", help="Padding strategy")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model name or path")
    parser.add_argument("--learning_rate", type=float, default=1.6e-3, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Train batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Eval batch size per device")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for regularization")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--fp16", type=bool, default=True, help="Enable mixed precision training")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, default="./lora_results", help="Output directory for checkpoints")
    parser.add_argument("--logging_dir", type=str, default="./lora_logs", help="Logging directory")
    parser.add_argument("--eval_strategy", type=str, default="epoch", help="Evaluation strategy")
    parser.add_argument("--save_strategy", type=str, default="no", help="Save strategy")
    parser.add_argument("--xlora_depth", type=int, default=4, help="xLoRA depth")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine if this is a regression task (e.g. stsb)
    is_regression = args.task_name == "stsb"

    # Load the dataset and evaluation metric
    datasets = load_dataset("glue", args.task_name)
    label_list = datasets["train"].features["label"].names
    num_labels = len(label_list)
    metric = evaluate.load("accuracy")

    # Define task-to-keys mapping
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    # Load base model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels,use_cache=False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    args.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Handle label mapping if needed
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Model labels do not match dataset labels. "
                f"Model labels: {list(sorted(label_name_to_id.keys()))}, Dataset labels: {list(sorted(label_list))}."
                " Ignoring model labels."
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    # Preprocessing function for tokenization
    def preprocess_function(examples):
        tokens = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*tokens, padding=args.padding, max_length=args.max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] if l != -1 else -1 for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if args.task_name == "mnli" else "validation"]
    test_dataset = datasets["test_matched" if args.task_name == "mnli" else "test"]
    data_collator = default_data_collator

    # Setup LoRA configuration and freeze base model parameters
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        trust_remote_code=True,
        use_flash_attention_2=False,
    )
    for param in model.parameters():
        param.requires_grad = False

    xlora_model = add_xlora_to_model(
    model=model,
    xlora_config=xLoRAConfig(
        config.hidden_size,
        base_model_id=args.model_name,
        xlora_depth=args.xlora_depth,
        device=torch.device("cuda"),
        adapters={
            "adapter_1": "./lora_finetuned_model_cola",
            "adapter_2": "./lora_finetuned_model_mrpc",
            "adapter_3": "./lora_finetuned_model_qnli",
            "adapter_4": "./lora_finetuned_model_sst2",
        },
    ),
    verbose=True,
).to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,  # We only save the final model
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        # fp16=args.fp16,
        report_to="none",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=xlora_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Save final model and tokenizer
    xlora_model.save_pretrained("./lora_finetuned_model")
    tokenizer.save_pretrained("./lora_finetuned_model")

if __name__ == "__main__":
    main()
