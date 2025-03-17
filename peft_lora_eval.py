#!/usr/bin/env python
import argparse
import torch
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    default_data_collator,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
import logging
from peft import PeftModel, AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
import random
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned LoRA model on a GLUE task")
    parser.add_argument("--task_name", type=str, default="qnli", help="GLUE task name (e.g. cola, qnli, etc.)")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--padding", type=str, default="max_length", help="Padding strategy")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Base model name or path")
    parser.add_argument("--lora_model_dir", type=str, default="./lora_finetuned_model", help="LoRA fine-tuned model directory")
    return parser.parse_args()

def main():
    args = parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Determine if this is a regression task (e.g. stsb)
    is_regression = args.task_name == "stsb"

    # Load the dataset and evaluation metric
    datasets = load_dataset("glue", args.task_name)
    if not is_regression:
        label_list = datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    print(num_labels)
    metric = evaluate.load("glue", args.task_name)

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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    args.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        print(preds)
        print(p.label_ids)
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}

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

    tasks = [args.task_name]
    eval_datasets = [eval_dataset]
    if args.task_name == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(datasets["validation_mismatched"])

    # # If you're using a fine-tuned LoRA model, load it like this:
    # base_model = model  # the base model we loaded earlier
    # # Note: if you applied LoRA during training, you can load the LoRA weights like:
    # peft_model = PeftModel.from_pretrained(base_model, args.lora_model_dir)
    # peft_model.eval()
    peft_model = AutoPeftModelForSequenceClassification.from_pretrained(args.lora_model_dir)
    peft_model.eval()

    trainer = Trainer(
        model=peft_model,
        args=TrainingArguments(
            output_dir="./peft_test_results",
            report_to="none",
        ),
        data_collator=data_collator,           
        compute_metrics=compute_metrics
    )
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        print(metrics)

    # Select only the first example from the evaluation dataset
    first_sample = eval_dataset.select([0])

    # Run prediction on the single sample
    output = trainer.predict(first_sample)

    # Print predictions and labels
    print("Predictions:", output.predictions)
    print("Labels:", output.label_ids)

if __name__ == "__main__":
    main()
