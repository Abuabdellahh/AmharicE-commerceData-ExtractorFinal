import numpy as np
import shap
import lime
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
from typing import List, Dict, Any
import matplotlib.pyplot as plt

class ModelInterpreter:
    def __init__(self, model_path: str):
        self.model = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )
        
    def get_shap_explanation(self, text: str, num_samples: int = 100):
        """Get SHAP explanation for a text input"""
        # Create a SHAP explainer
        explainer = shap.Explainer(self.model)
        
        # Get SHAP values
        shap_values = explainer([text], max_evals=num_samples)
        
        # Visualize the explanation
        shap.plots.text(shap_values)
        plt.show()
        
        return shap_values

    def get_lime_explanation(self, text: str, num_features: int = 6):
        """Get LIME explanation for a text input"""
        # Create a LIME explainer
        explainer = LimeTextExplainer(class_names=["O", "B-PRODUCT", "I-PRODUCT", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"])
        
        # Define prediction function for LIME
        def predict_fn(texts: List[str]):
            predictions = self.model(texts)
            return np.array([
                [pred["score"] for pred in preds if pred["entity"] != "O"]
                for preds in predictions
            ])
        
        # Get explanation
        exp = explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            labels=(1,)
        )
        
        # Visualize the explanation
        exp.show_in_notebook(text=True)
        
        return exp

    def analyze_difficult_cases(self, texts: List[str], labels: List[List[str]]):
        """Analyze cases where the model struggles"""
        difficult_cases = []
        
        for text, true_labels in zip(texts, labels):
            predictions = self.model(text)
            
            # Check for mismatches
            if not self._check_prediction_accuracy(predictions, true_labels):
                difficult_cases.append({
                    "text": text,
                    "true_labels": true_labels,
                    "predicted": predictions,
                    "shap_explanation": self.get_shap_explanation(text),
                    "lime_explanation": self.get_lime_explanation(text)
                })
        
        return difficult_cases

    def _check_prediction_accuracy(self, predictions: List[Dict], true_labels: List[str]) -> bool:
        """Check if predictions match true labels"""
        pred_labels = [pred["entity"] for pred in predictions]
        return pred_labels == true_labels

    def generate_interpretability_report(self, texts: List[str], labels: List[List[str]], output_path: str):
        """Generate a comprehensive interpretability report"""
        report = {
            "model_performance": {},
            "difficult_cases": [],
            "feature_importance": {}
        }
        
        # Analyze difficult cases
        difficult_cases = self.analyze_difficult_cases(texts, labels)
        report["difficult_cases"] = difficult_cases
        
        # Calculate feature importance
        feature_importance = {}
        for case in difficult_cases:
            for word, importance in case["shap_explanation"].values:
                if word not in feature_importance:
                    feature_importance[word] = []
                feature_importance[word].append(importance)
        
        # Average importance scores
        for word, scores in feature_importance.items():
            feature_importance[word] = np.mean(scores)
        
        report["feature_importance"] = feature_importance
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    interpreter = ModelInterpreter("path/to/best/model")
    
    # Example usage
    example_text = "አዲስ አበባ በ 1000 ብር የሚገኙ አልባ አልባ አልባ"
    
    # Get SHAP explanation
    shap_values = interpreter.get_shap_explanation(example_text)
    
    # Get LIME explanation
    lime_exp = interpreter.get_lime_explanation(example_text)
    
    # Generate interpretability report
    interpreter.generate_interpretability_report(
        texts=[example_text],
        labels=[["B-LOC", "I-LOC", "O", "B-PRICE", "I-PRICE", "I-PRICE", "O", "B-PRODUCT", "I-PRODUCT", "I-PRODUCT"]],
        output_path="interpretability_report.json"
    )
