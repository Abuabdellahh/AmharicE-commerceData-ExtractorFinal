import pandas as pd
from ner_model import AmharicNERModel
from sklearn.model_selection import train_test_split
import json
from typing import Dict, Any
import numpy as np

class ModelComparator:
    def __init__(self):
        self.models = {
            "xlm-roberta": "xlm-roberta-base",
            "distilbert": "distilbert-base-multilingual-cased",
            "mbert": "bert-base-multilingual-cased"
        }
        self.results = {}

    def load_dataset(self, dataset_path: str):
        """Load and preprocess dataset"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokens = []
        ner_tags = []
        
        for sentence in data:
            tokens.append(sentence["tokens"])
            ner_tags.append(sentence["ner_tags"])
            
        return {
            "tokens": tokens,
            "ner_tags": ner_tags
        }

    def evaluate(self, test_dataset):
        """
        Evaluate all models on the test dataset
        
        Args:
            test_dataset (list): List of test examples with text and labels
        """
        for model_type in self.models:
            config = self.model_configs[model_type]
            
            print(f"Evaluating {model_type}...")
            tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
            model = AutoModelForTokenClassification.from_pretrained(config['model_name'])
            
            # Process dataset and get predictions
            all_predictions = []
            all_true_labels = []
            
            for example in test_dataset:
                inputs = tokenizer(
                    example['text'],
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=config['max_length']
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                predicted_labels = torch.argmax(outputs.logits, dim=2)
                
                # Convert to lists and handle padding
                predicted = predicted_labels[0].tolist()[:len(example['text'])]
                true = example['labels'][:len(example['text'])]
                
                all_predictions.extend(predicted)
                all_true_labels.extend(true)
            
            # Calculate detailed metrics
            metrics = {
                'weighted': {},
                'micro': {},
                'macro': {}
            }
            
            # Weighted metrics
            report = classification_report(all_true_labels, all_predictions, output_dict=True)
            metrics['weighted'] = {
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            }
            
            # Micro metrics
            metrics['micro'] = {
                'precision': precision_score(all_true_labels, all_predictions, average='micro'),
                'recall': recall_score(all_true_labels, all_predictions, average='micro'),
                'f1': f1_score(all_true_labels, all_predictions, average='micro')
            }
            
            # Macro metrics
            metrics['macro'] = {
                'precision': precision_score(all_true_labels, all_predictions, average='macro'),
                'recall': recall_score(all_true_labels, all_predictions, average='macro'),
                'f1': f1_score(all_true_labels, all_predictions, average='macro')
            }
            
            self.results[model_type] = metrics

    def get_comparison_report(self):
        """
        Generate detailed comparison report
        
        Returns:
            pd.DataFrame: Comparison report with detailed metrics
        """
        metrics = ['precision', 'recall', 'f1']
        comparison_dfs = {}
        
        # Create separate dataframes for each metric type
        for metric_type in ['weighted', 'micro', 'macro']:
            df = pd.DataFrame()
            for model_name, results in self.results.items():
                model_metrics = {}
                for metric in metrics:
                    model_metrics[metric] = results[metric_type][metric]
                df[model_name] = pd.Series(model_metrics)
            comparison_dfs[metric_type] = df
        
        return comparison_dfs

    def get_best_model(self, metric_type='weighted', metric='f1'):
        """
        Get the best performing model based on specified metric
        
        Args:
            metric_type (str): Type of metric (weighted, micro, macro)
            metric (str): Specific metric (precision, recall, f1)
            
        Returns:
            str: Name of best performing model
        """
        best_model = None
        best_score = -1
        
        for model_name, results in self.results.items():
            score = results[metric_type][metric]
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score

    def generate_analysis_report(self):
        """
        Generate a comprehensive analysis report
        
        Returns:
            dict: Detailed analysis report
        """
        report = {
            'model_comparison': self.get_comparison_report(),
            'best_models': {
                'weighted': self.get_best_model('weighted'),
                'micro': self.get_best_model('micro'),
                'macro': self.get_best_model('macro')
            },
            'model_configs': self.model_configs
        }
        
        return report

    def compare_models(self, dataset_path: str, output_dir: str):
        """Compare different NER models"""
        dataset = self.load_dataset(dataset_path)
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

        self.evaluate(test_dataset)

        print("Comparison Report:")
        print(self.get_comparison_report())

        best_model, best_score = self.get_best_model()
        print(f"\nBest model: {best_model} with score {best_score}")

        analysis_report = self.generate_analysis_report()
        print("Analysis Report:")
        print(analysis_report)

    def save_comparison_results(self, output_path: str):
        """Save comparison results to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.compare_models(
        dataset_path="path/to/your/conll/dataset.json",
        output_dir="model_comparison_results"
    )
    best_model = comparator.get_best_model()
    print(f"\nBest model: {best_model}")
    comparator.save_comparison_results("model_comparison_results.json")
