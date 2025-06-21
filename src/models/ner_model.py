import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class AmharicNERModel:
    def __init__(self, model_name: str = "xlm-roberta-base"):
        self.model_name = model_name
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = None
        self.label2id = {
            "O": 0,
            "B-PRODUCT": 1,
            "I-PRODUCT": 2,
            "B-PRICE": 3,
            "I-PRICE": 4,
            "B-LOC": 5,
            "I-LOC": 6
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=512
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label_ids[-1])
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = precision_recall_fscore_support(
            y_true=true_labels,
            y_pred=true_predictions,
            average="weighted",
            labels=list(self.id2label.values())
        )
        
        return {
            "precision": results[0],
            "recall": results[1],
            "f1": results[2]
        }

    def train(self, train_dataset, val_dataset, output_dir: str):
        self.model = XLMRobertaForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            data_collator=lambda x: {"input_ids": torch.stack([f["input_ids"] for f in x]),
                                    "attention_mask": torch.stack([f["attention_mask"] for f in x]),
                                    "labels": torch.stack([f["labels"] for f in x])}
        )

        trainer.train()
        return trainer
