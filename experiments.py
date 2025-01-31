#!/usr/bin/env python
# coding=utf-8

"""
Replicate experiments from:
"How Many Data Points is a Prompt Worth?"

This script demonstrates two fine-tuning approaches:
  1) Head-based classification (standard).
  2) Prompt-based classification with pattern + verbalizer.

We evaluate with subsets of the training data of varying size,
run multiple seeds, and evaluate on the dev/validation set.

You should adapt the prompts/verbalizers for each dataset
(CB, COPA, MultiRC, BoolQ, WSC, WiC, RTE, MNLI) to exactly
match those in the paper and references therein.
"""

import os
import random
import logging
import argparse
import utils 

import torch
import numpy as np

# Hugging Face libraries
from datasets import load_dataset
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          AutoModelForMaskedLM,
                          DataCollatorWithPadding,
                          Trainer, 
                          TrainingArguments)



# A dictionary to specify how many examples to hold out as dev from training,
# for each dataset as described in the paper. Adjust as you replicate all tasks.
DEV_SPLITS = {
    "boolq": 500,
    "multirc": 50,
    "copa": 50,
    "cb": 50,
    "wic": 50,
    "wsc": 50,
    "rte": 50,
    "mnli": 1000  # Example; you'd verify your exact scheme
    # ... etc. for each dataset
}


# --------------------------
# Utility functions
# --------------------------
def set_seed(seed: int):
    """Ensure reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_superglue_dataset(task_name):
    """
    Load the dataset from Hugging Face 'super_glue' or 'glue' (for MNLI) 
    or custom tasks. 
    Return the raw dataset dict with splits: train, validation, test (if available).
    """
    if task_name.lower() == "mnli":
        # Load the GLUE MNLI dataset
        raw_datasets = load_dataset("glue", "mnli")
    else:
        # Load from 'super_glue'
        raw_datasets = load_dataset("super_glue", task_name.lower())
    return raw_datasets


# Example PVP placeholders for BoolQ.
# You must replicate the exact patterns/verbalizers from the paper or from 
# Schick & Schütze (2020) for full fidelity.
# This is just a single example demonstrating the structure.
BOOLQ_PATTERNS = [
    {
        "pattern": (
            "{passage} "
            "Based on the previous passage, can you answer the question: "
            "{question} <mask>."
        ),
        "verbalizer": {"Yes": "Yes", "No": "No"}
    },
    # You could add more patterns from the paper if they used an ensemble of prompts.
]


def create_boolq_prompt(example):
    """
    Convert a BoolQ example into a cloze-style string using the first pattern
    from BOOLQ_PATTERNS. Return the string and the correct verbalization label.
    """
    pattern_config = BOOLQ_PATTERNS[0]  # Here we pick the first pattern
    pattern = pattern_config["pattern"]
    
    # Insert actual text into the pattern
    text = pattern.format(
        passage=example["passage"],
        question=example["question"]
    )
    # The label is "Yes" or "No", so map example["label"] into the verbalizer tokens
    label_id = example["label"]  # 1 = True, 0 = False for BoolQ
    label_str = "Yes" if label_id == 1 else "No"
    
    return text, label_str


# --------------------------
# Dataset Preprocessing
# --------------------------
def prepare_dataset_for_head(task_name, tokenizer, raw_train, raw_dev, raw_test, max_samples=None):
    """
    Standard text classification approach.
    This function:
      1) Prepares the input fields (e.g., 'premise', 'hypothesis', etc. for tasks).
      2) Tokenizes.
      3) Returns processed Dataset objects.
    """
    # For simplicity, assume each dataset is either
    #  - single-sentence classification (BoolQ, WiC, etc.), or
    #  - pair classification (RTE, MNLI, CB, etc.).
    # Below is an *example* for BoolQ.
    # Extend for each dataset’s structure in a real replication.
    
    def preprocess_boolq(example):
        # For head-based approach, we treat passage and question as two separate sequences
        # for a text-classification model.
        # Input could be: "[CLS] passage [SEP] question [SEP]"
        return tokenizer(
            example["passage"],
            example["question"],
            truncation=True,
            max_length=512
        )
    
    def preprocess_labels(example):
        # Keep label as-is
        example["labels"] = example["label"]
        return example
    
    if task_name.lower() == "boolq":
        # Tokenize the dataset
        train_dataset = raw_train.map(preprocess_boolq, batched=False)
        dev_dataset   = raw_dev.map(preprocess_boolq, batched=False)
        test_dataset  = raw_test.map(preprocess_boolq, batched=False)

        # Convert label field
        train_dataset = train_dataset.map(preprocess_labels, batched=False)
        dev_dataset   = dev_dataset.map(preprocess_labels, batched=False)
        test_dataset  = test_dataset.map(preprocess_labels, batched=False)
    else:
        # Implement logic for other tasks as needed
        raise NotImplementedError(f"Head-based prep not implemented for task {task_name}")
    
    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), max_samples)))
        dev_dataset   = dev_dataset.select(range(min(len(dev_dataset), max_samples)))
        test_dataset  = test_dataset.select(range(min(len(test_dataset), max_samples)))
    
    return train_dataset, dev_dataset, test_dataset


def prepare_dataset_for_prompt(task_name, tokenizer, raw_train, raw_dev, raw_test, max_samples=None):
    """
    Prompt-based approach: convert each example into a single string
    containing the cloze pattern, plus we keep track of the *correct verbalization*.
    We'll handle this by storing the entire input text as 'input_ids'
    and the correct label token as something we can compute the cross-entropy over.
    """
    # Example for BoolQ using the single PVP in BOOLQ_PATTERNS[0].
    # For a real replication, you’d implement each pattern or an ensemble of them.
    
    # We treat this as a masked LM problem: the prompt includes a <mask> token,
    # and we want the model to predict "Yes" or "No" at that mask position.
    
    verbalizer_map = BOOLQ_PATTERNS[0]["verbalizer"]  # {"Yes": "Yes", "No": "No"}
    
    def tokenize_boolq_prompt(example):
        # Convert example to prompt
        prompt_text, label_str = create_boolq_prompt(example)
        # Insert the special <mask> token that the model expects for mask filling
        # For RoBERTa, the mask token is "<mask>" by default in Hugging Face tokenizers.
        
        # Tokenize
        tokenized = tokenizer(
            prompt_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # We need the ID of the correct verbalization token in the vocab.
        # e.g., if label_str = "Yes", we get the ID with tokenizer.encode(" Yes", add_special_tokens=False)
        # But be careful about spacing. For simplicity, let's do:
        verbalizer_ids = tokenizer.encode(" " + label_str, add_special_tokens=False)
        # The last token in that sequence is presumably the subword for the actual label (like "Yes", "No").
        # We'll store that as "labels" for the masked token. 
        correct_label_id = verbalizer_ids[-1]
        
        # We'll also store the correct_label_id for computing the MLM loss at the <mask> position.
        # The Trainer can compute MLM cross-entropy if we arrange the data correctly.
        
        # Convert to standard python ints
        input_ids = tokenized["input_ids"][0].tolist()
        attention_mask = tokenized["attention_mask"][0].tolist()
        
        # Identify the <mask> token position
        mask_token_id = tokenizer.mask_token_id
        try:
            mask_index = input_ids.index(mask_token_id)
        except ValueError:
            mask_index = -1  # In case there's an error
        
        # Create labels: all -100 except for the <mask> position
        labels = [-100] * len(input_ids)
        if mask_index >= 0:
            labels[mask_index] = correct_label_id
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    if task_name.lower() == "boolq":
        train_dataset = raw_train.map(tokenize_boolq_prompt)
        dev_dataset   = raw_dev.map(tokenize_boolq_prompt)
        test_dataset  = raw_test.map(tokenize_boolq_prompt)
    else:
        raise NotImplementedError(f"Prompt-based prep not implemented for task {task_name}")
    
    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), max_samples)))
        dev_dataset   = dev_dataset.select(range(min(len(dev_dataset), max_samples)))
        test_dataset  = test_dataset.select(range(min(len(test_dataset), max_samples)))
    
    return train_dataset, dev_dataset, test_dataset


# --------------------------
# Training & Evaluation
# --------------------------
def main():
    # 1. Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="boolq")
    parser.add_argument("--approach", type=str, default="head", help="'head' or 'prompt'")
    parser.add_argument("--subset_sizes", nargs="+", default=[50,100,250,500,1000,1500], type=int)
    parser.add_argument("--model_name_or_path", type=str, default="roberta-large")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--min_train_steps", type=int, default=250)
    parser.add_argument("--seeds", nargs="+", default=[0,1,2], type=int)
    args = parser.parse_args()
    
    exp_args = utils.ExperimentArguments(
        model_name_or_path=args.model_name_or_path,
        task_name=args.task_name,
        approach=args.approach,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        min_train_steps=args.min_train_steps,
        output_dir=args.output_dir,
        seed=args.seed,
        subset_sizes=args.subset_sizes
    )
    
    # 2. Set seed
    set_seed(exp_args.seed)
    
    # 3. Load dataset
    raw_datasets = load_superglue_dataset(exp_args.task_name)
    
    # For SuperGLUE, we typically have splits: train, validation, test
    # For some tasks (like MNLI), we have validation_mismatched, etc.
    # The paper mentions they do not use the official test set for SuperGLUE tasks
    # (since it's not public), but set aside part of train for dev, and use the official
    # validation as "test." We'll replicate that approach.
    
    # 4. Create custom dev (split from train)
    dev_size = DEV_SPLITS.get(exp_args.task_name.lower(), 50)
    full_train = raw_datasets["train"]
    
    # Shuffle train set
    full_train = full_train.shuffle(seed=exp_args.seed)
    
    # Reserve dev_size for dev, rest is new_train
    dev_dataset = full_train.select(range(dev_size))
    new_train = full_train.select(range(dev_size, len(full_train)))
    
    # Use the official validation as "test"
    test_dataset = raw_datasets["validation"]
    
    # 5. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(exp_args.model_name_or_path, use_fast=True)
    
    # 6. For each approach, prepare the data accordingly
    if exp_args.approach == "head":
        # We'll create a classification head on top
        # So we will use `AutoModelForSequenceClassification`
        # and treat examples as standard classification input
        # (e.g. input = [CLS] text [SEP] text [SEP]).
        
        def train_and_evaluate_head(train_subset):
            model = AutoModelForSequenceClassification.from_pretrained(
                exp_args.model_name_or_path,
                num_labels=2  # e.g., for BoolQ yes/no
            )
            
            # Preprocess
            train_dataset_prep, dev_dataset_prep, test_dataset_prep = prepare_dataset_for_head(
                exp_args.task_name,
                tokenizer,
                train_subset,
                dev_dataset,
                test_dataset
            )
            
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
            
            # Setup Trainer
            training_args = TrainingArguments(
                output_dir=os.path.join(exp_args.output_dir, f"{exp_args.task_name}_head_{len(train_subset)}"),
                learning_rate=exp_args.learning_rate,
                per_device_train_batch_size=exp_args.train_batch_size,
                per_device_eval_batch_size=exp_args.eval_batch_size,
                num_train_epochs=exp_args.num_train_epochs,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir=os.path.join(exp_args.output_dir, "logs"),
                logging_steps=50,
                seed=exp_args.seed
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset_prep,
                eval_dataset=dev_dataset_prep,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            
            # We replicate: train for up to N epochs, using a very low LR,
            # ensuring at least 250 steps
            step_per_epoch = len(train_dataset_prep) // training_args.per_device_train_batch_size
            total_steps = step_per_epoch * training_args.num_train_epochs
            if total_steps < exp_args.min_train_steps:
                # we can override the num_train_epochs or switch to max_steps if needed
                # for simplicity, let's just forcibly set max_steps = exp_args.min_train_steps
                trainer.args.max_steps = exp_args.min_train_steps
                trainer.args.num_train_epochs = 9999  # effectively ignore epoch-based termination
                trainer.args.evaluation_strategy="steps"
                trainer.args.save_strategy="steps"
            
            trainer.train()
            # Evaluate on dev, test
            dev_metrics = trainer.evaluate(eval_dataset=dev_dataset_prep)
            test_metrics = trainer.evaluate(eval_dataset=test_dataset_prep)
            
            return dev_metrics, test_metrics
        
        # For each subset size
        for size in exp_args.subset_sizes:
            # If subset size is larger than new_train, skip
            if size > len(new_train):
                continue
            train_subset = new_train.select(range(size))
            dev_metrics, test_metrics = train_and_evaluate_head(train_subset)
            print(f"[HEAD] Subset={size}, Dev={dev_metrics}, Test={test_metrics}")
    
    elif exp_args.approach == "prompt":
        # We'll treat this as a masked language modeling problem
        # with custom patterns and a restricted vocabulary for classification
        # but we can implement a simple approach using `AutoModelForMaskedLM`.
        
        def train_and_evaluate_prompt(train_subset):
            model = AutoModelForMaskedLM.from_pretrained(exp_args.model_name_or_path)
            
            # Preprocess
            train_dataset_prep, dev_dataset_prep, test_dataset_prep = prepare_dataset_for_prompt(
                exp_args.task_name,
                tokenizer,
                train_subset,
                dev_dataset,
                test_dataset
            )
            
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
            
            training_args = TrainingArguments(
                output_dir=os.path.join(exp_args.output_dir, f"{exp_args.task_name}_prompt_{len(train_subset)}"),
                learning_rate=exp_args.learning_rate,
                per_device_train_batch_size=exp_args.train_batch_size,
                per_device_eval_batch_size=exp_args.eval_batch_size,
                num_train_epochs=exp_args.num_train_epochs,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_dir=os.path.join(exp_args.output_dir, "logs"),
                logging_steps=50,
                seed=exp_args.seed
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset_prep,
                eval_dataset=dev_dataset_prep,
                tokenizer=tokenizer,
                data_collator=data_collator
            )
            
            # Adjust steps vs epochs
            step_per_epoch = len(train_dataset_prep) // training_args.per_device_train_batch_size
            total_steps = step_per_epoch * training_args.num_train_epochs
            if total_steps < exp_args.min_train_steps:
                trainer.args.max_steps = exp_args.min_train_steps
                trainer.args.num_train_epochs = 9999
                trainer.args.evaluation_strategy="steps"
                trainer.args.save_strategy="steps"
            
            trainer.train()
            
            dev_metrics = trainer.evaluate(eval_dataset=dev_dataset_prep)
            test_metrics = trainer.evaluate(eval_dataset=test_dataset_prep)
            
            return dev_metrics, test_metrics
        
        for size in exp_args.subset_sizes:
            if size > len(new_train):
                continue
            train_subset = new_train.select(range(size))
            dev_metrics, test_metrics = train_and_evaluate_prompt(train_subset)
            print(f"[PROMPT] Subset={size}, Dev={dev_metrics}, Test={test_metrics}")
    
    else:
        raise ValueError("approach must be either 'head' or 'prompt'.")


if __name__ == "__main__":
    main()