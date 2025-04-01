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
    AutoConfig,
)
import logging
from peft import PeftModel, AutoPeftModelForSequenceClassification
from torch.utils.data import DataLoader
import random
from Xlora.xlora import add_xlora_to_model, from_pretrained
from Xlora.xlora_config import xLoRAConfig
from Xlora.xlora_utils import load_model
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)

def classifier_hook(module, inputs, output):
    print("[HOOK] Internal xLoRA classifier called.")
    print("Input shape:", inputs[0].shape)
    try:
        print("Output shape:", output.shape)
    except Exception:
        print("Output:", output)

# Define a hook function that prints out info from the module forward pass.
def debug_hook(module, input, output):
    # Here you can customize what you want to print.
    # For instance, print the module's class name and the shape of its output.
    try:
        out_shape = output.shape
    except Exception:
        out_shape = "N/A"
    print(f"[DEBUG] Module {module.__class__.__name__} called; output shape: {out_shape}")
    return output

def register_hooks(model):
    hook_handles = []
    # For example, if you want to hook every module:
    for name, module in model.named_modules():
        # If you want to filter and only hook certain modules, you could do so here.
        handle = module.register_forward_hook(debug_hook)
        hook_handles.append(handle)
    return hook_handles

def remove_hooks(hook_handles):
    for handle in hook_handles:
        handle.remove()

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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels,use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    args.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    sentence1_key, sentence2_key = task_to_keys[args.task_name]

    def compute_metrics(p: EvalPrediction):
        # print("Computing metrics...")
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        # print(preds)
        # print(p.label_ids)
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

    XLoRA_model_name = args.lora_model_dir
    XLoRA_model, tokenizer = load_model(
        model_name=XLoRA_model_name,
        device="cuda:0",
        dtype=torch.float32,
        load_xlora=True,
        adapters={
            "adapter_1": "./lora_finetuned_model_cola",
             "adapter_2": "./lora_finetuned_model_mrpc",
            "adapter_3": "./lora_finetuned_model_qnli",
            "adapter_4": "./lora_finetuned_model_sst2",
        },
    )
    # print(XLoRA_model)
    # print(type(XLoRA_model.forward))
    XLoRA_model.eval()
    hook_handle = XLoRA_model.internal_xlora_classifier.register_forward_hook(classifier_hook)
    # # hook_handles = register_hooks(XLoRA_model)

    trainer = Trainer(
        model=XLoRA_model,
        args=TrainingArguments(
            output_dir="./peft_test_results",
            report_to="none",
        ),
        data_collator=data_collator,           
        compute_metrics=compute_metrics
    )
    for dataset, task in zip(eval_datasets, tasks):
        print(f"Evaluating task: {task}")
        # Use trainer.predict to get model outputs and labels
        # dataset=dataset.select([0])
        eval_output = trainer.predict(dataset)
        # If the model output is a tuple, we assume logits are in index 1.
        logits = eval_output.predictions[1] if isinstance(eval_output.predictions, tuple) else eval_output.predictions
        # For classification, take argmax over logits to get predicted class indices.
        preds = np.argmax(logits, axis=1)
        labels = dataset["label"]
        # print(f"Predictions: {logits}")  
        # print(f"Labels: {labels}")

        # Compute accuracy and F1 score manually using sklearn
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        combined_score = (acc + f1) / 2
        print(f"Metrics for task {task}:")
        print({"accuracy": acc, "f1": f1, "combined_score": combined_score})
        print("results for xlora eval")
    # remove_hooks(hook_handle)
    hook_handle.remove()

if __name__ == "__main__":
    main()
