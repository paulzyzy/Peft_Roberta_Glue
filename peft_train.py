import random
import numpy as np
import torch
from datasets import load_dataset
import evaluate
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig,default_data_collator,TrainingArguments, Trainer, EvalPrediction
from peft import LoraModel, LoraConfig, TaskType, get_peft_model
import logging
# Set random seed for reproducibility
logger = logging.getLogger(__name__)
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
task_name = "qnli"  # Set the task name

is_regression = False
is_regression = task_name == "stsb"
# If using GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

max_seq_length = 512  # Set the maximum sequence length
padding = "max_length"  # Set the padding method
# Load the dataset
datasets = load_dataset("glue", task_name)
label_list = datasets["train"].features["label"].names
num_labels = len(label_list)
metric = evaluate.load("accuracy")

# Training configuration
lora_training_args = TrainingArguments(
    output_dir="./lora_results",   # Directory to save model checkpoints
    eval_strategy="epoch",          # Evaluate after each epoch
    save_strategy="no",          # Save model after each epoch
    learning_rate=4e-4,             # Learning rate for QLoRA
    per_device_train_batch_size=16, # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=25,             # Number of epochs
    weight_decay=0.1,              # Weight decay for regularization
    logging_dir="./lora_logs",     # Directory for training logs
    logging_steps=10,               # Log training metrics every 10 steps
    fp16=True,                      # Enable mixed precision
    report_to="none"
)

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
model_name = "roberta-base"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=None,
    use_fast=True,
    revision="main",
    use_auth_token= None,
)
max_seq_length = min(max_seq_length, tokenizer.model_max_length)

sentence1_key, sentence2_key = task_to_keys[task_name]

def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result


# print(model)
label_to_id = None
if (
    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    and task_name is not None
    and not is_regression
):
    # Some have all caps in their config, some don't.
    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    else:
        logger.warn(
            "Your model seems to have been trained with labels, but they don't match the dataset: ",
            f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
            "\nIgnoring the model labels as a result.",
        )
elif task_name is None and not is_regression:
    label_to_id = {v: i for i, v in enumerate(label_list)}

def preprocess_function(examples):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
    return result

datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=True)
train_dataset = datasets["train"]
eval_dataset = datasets["validation_matched" if task_name == "mnli" else "validation"]
test_dataset = datasets["test_matched" if task_name == "mnli" else "test"]
data_collator = default_data_collator
print(datasets)
print(datasets["train"].features["label"].names)





lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
)
# lora_model = LoraModel(model, config, "default")
# print(lora_model)
# Freeze model parameters to prevent weight updates
for param in model.parameters():
    param.requires_grad = False

peft_lora_model = get_peft_model(model, lora_config)
# print(peft_lora_model)

# Initialize Trainer
lora_trainer = Trainer(
    model=peft_lora_model,            # The QLoRA model
    args=lora_training_args,          # Training arguments
    train_dataset=train_dataset,       # Training dataset
    eval_dataset=eval_dataset,         # Validation dataset
    data_collator=data_collator,       # Data collator for batching
    compute_metrics=compute_metrics,   # Metrics for evaluation
)
lora_trainer.train()
# Save the final fine-tuned LoRA model
peft_lora_model.save_pretrained("./lora_finetuned_model")

# Save the tokenizer
tokenizer.save_pretrained("./lora_finetuned_model")


# config = AutoConfig.from_pretrained(
#         model_name,
#         num_labels=num_labels,
#         finetuning_task="cola",
#         cache_dir=None,
#         revision="main",
#         use_auth_token=None,
#         apply_lora=True,
#         lora_alpha=16,
#         lora_r=8,
#         apply_adapter=False,
#         adapter_type='houlsby',
#         adapter_size=64,
#         reg_loss_wgt=0.0,
#         masking_prob=0.0,
#     )