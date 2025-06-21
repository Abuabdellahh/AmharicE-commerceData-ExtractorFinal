import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import logging
from pathlib import Path

class AmharicNERModel:
    def __init__(self, model_name: str = "xlm-roberta-base"):
        """
        Initialize the NER model
        
        Args:
            model_name: Name of the base model to use
        """
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
            "I-LOC": 6,
            "B-VENDOR": 7,
            "I-VENDOR": 8,
            "B-BRAND": 9,
            "I-BRAND": 10
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def tokenize_and_align_labels(self, examples):
        """
        Tokenize inputs and align labels with tokens
        
        Args:
            examples: Dictionary containing tokens and labels
        
        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=self.config.get("max_length", 512)
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
                    label_ids.append(self.label2id[label[word_idx]])
                else:
                    label_ids.append(self.label2id[label[word_idx]])
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_dataset(self, data: List[Dict]) -> Dataset:
        """
        Prepare dataset for training/evaluation
        
        Args:
            data: List of dictionaries containing tokens and labels
        
        Returns:
            HuggingFace Dataset object
        """
        dataset = Dataset.from_dict({
            "tokens": [d["tokens"] for d in data],
            "ner_tags": [d["labels"] for d in data]
        })
        
        return dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names
        )

    def train(
        self, 
        train_data: List[Dict], 
        val_data: Optional[List[Dict]] = None,
        config: Optional[Dict] = None
    ) -> None:
        """
        Train the NER model
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            config: Training configuration
        """
        if config:
            self.config = config
        
        train_dataset = self.prepare_dataset(train_data)
        
        if val_data:
            val_dataset = self.prepare_dataset(val_data)
            dataset_dict = DatasetDict({
                "train": train_dataset,
                "validation": val_dataset
            })
        else:
            dataset_dict = DatasetDict({"train": train_dataset})
        
        self.model = XLMRobertaForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        training_args = TrainingArguments(
            output_dir=str(Path("./results") / "model"),
            evaluation_strategy="epoch",
            learning_rate=config.get("learning_rate", 2e-5),
            per_device_train_batch_size=config.get("batch_size", 8),
            per_device_eval_batch_size=config.get("batch_size", 8),
            num_train_epochs=config.get("epochs", 5),
            weight_decay=config.get("weight_decay", 0.01),
            warmup_steps=config.get("warmup_steps", 100),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            fp16=config.get("fp16", True),
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=config.get("save_total_limit", 3),
            seed=config.get("seed", 42),
            metric_for_best_model=config.get("metric_for_best_model", "f1"),
            load_best_model_at_end=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict.get("validation"),
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        self.logger.info("Starting model training...")
        trainer.train()
        
        self.logger.info("Saving best model...")
        self.model.save_pretrained(str(Path("./results") / "model"))
        self.tokenizer.save_pretrained(str(Path("./results") / "model"))

    def compute_metrics(self, p):
        """
        Compute evaluation metrics
        
        Args:
            p: Prediction output from Trainer
        
        Returns:
            Dictionary of metrics
        """
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
        
        results = {
            "precision": precision_recall_fscore_support(
                true_labels, true_predictions, average="weighted"
            )[0],
            "recall": precision_recall_fscore_support(
                true_labels, true_predictions, average="weighted"
            )[1],
            "f1": precision_recall_fscore_support(
                true_labels, true_predictions, average="weighted"
            )[2],
            "accuracy": sum(1 for pred, gold in zip(true_predictions, true_labels) 
                           if pred == gold) / len(true_predictions)
        }
        
        return results

    def evaluate(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate the model on test data
        
        Args:
            test_data: Test data
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        test_dataset = self.prepare_dataset(test_data)
        
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        return trainer.evaluate(test_dataset)

    def predict(self, text: str) -> List[Dict]:
        """
        Make predictions on new text
        
        Args:
            text: Input text to predict
        
        Returns:
            List of predicted entities
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        tokens = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.get("max_length", 512)
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        entities = []
        current_entity = None
        
        for idx, (token, pred) in enumerate(zip(tokens, predictions[0])):
            label = self.id2label[pred.item()]
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "entity": label[2:],
                    "start": idx,
                    "end": idx + 1
                }
            elif label.startswith("I-"):
                if current_entity and current_entity["entity"] == label[2:]:
                    current_entity["text"] += " " + token
                    current_entity["end"] = idx + 1
            elif current_entity:
                entities.append(current_entity)
                current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained model
        
        Args:
            model_path: Path to the saved model
        """
        self.model = XLMRobertaForTokenClassification.from_pretrained(model_path)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
