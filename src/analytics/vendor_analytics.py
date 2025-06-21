import pandas as pd
from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timedelta
import json

class VendorAnalyticsEngine:
    def __init__(self, model_path: str):
        self.model = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple"
        )
        
    def calculate_vendor_metrics(self, vendor_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key metrics for a vendor"""
        metrics = {
            "avg_views_per_post": self._calculate_avg_views(vendor_data),
            "posting_frequency": self._calculate_posting_frequency(vendor_data),
            "avg_price_point": self._calculate_avg_price(vendor_data),
            "top_performing_post": self._find_top_performing_post(vendor_data)
        }
        
        # Calculate lending score
        metrics["lending_score"] = self._calculate_lending_score(metrics)
        
        return metrics

    def _calculate_avg_views(self, data: pd.DataFrame) -> float:
        """Calculate average views per post"""
        return data["views"].mean()

    def _calculate_posting_frequency(self, data: pd.DataFrame) -> float:
        """Calculate posting frequency (posts per week)"""
        date_range = pd.to_datetime(data["date"]).max() - pd.to_datetime(data["date"]).min()
        weeks = date_range.total_seconds() / (7 * 24 * 60 * 60)
        return len(data) / weeks if weeks > 0 else 0

    def _calculate_avg_price(self, data: pd.DataFrame) -> float:
        """Calculate average price point"""
        prices = []
        for text in data["text"]:
            predictions = self.model(text)
            for pred in predictions:
                if pred["entity_group"] in ["PRICE"]:
                    try:
                        price = float(pred["word"])
                        prices.append(price)
                    except ValueError:
                        continue
        
        return np.mean(prices) if prices else 0

    def _find_top_performing_post(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Find the post with highest views"""
        top_post = data.loc[data["views"].idxmax()]
        return {
            "views": top_post["views"],
            "text": top_post["text"],
            "date": top_post["date"],
            "product": self._extract_product_from_text(top_post["text"])
        }

    def _extract_product_from_text(self, text: str) -> str:
        """Extract product name from text using NER"""
        predictions = self.model(text)
        product_words = []
        
        for pred in predictions:
            if pred["entity_group"] in ["PRODUCT"]:
                product_words.append(pred["word"])
        
        return " ".join(product_words) if product_words else "Unknown"

    def _calculate_lending_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate vendor's lending score"""
        # Weighted score calculation
        view_weight = 0.5
        frequency_weight = 0.3
        price_weight = 0.2
        
        # Normalize metrics
        normalized_views = metrics["avg_views_per_post"] / 10000  # Normalize to 10,000 views
        normalized_frequency = metrics["posting_frequency"] / 10  # Normalize to 10 posts/week
        normalized_price = metrics["avg_price_point"] / 10000  # Normalize to 10,000 ETB
        
        # Calculate weighted score
        score = (
            (normalized_views * view_weight) +
            (normalized_frequency * frequency_weight) +
            (normalized_price * price_weight)
        )
        
        return round(score * 100, 2)  # Scale to 0-100

    def generate_vendor_scorecard(self, vendors_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate a scorecard comparing multiple vendors"""
        vendor_metrics = []
        
        for vendor_id, data in vendors_data.items():
            metrics = self.calculate_vendor_metrics(data)
            vendor_metrics.append({
                "vendor_id": vendor_id,
                **metrics
            })
        
        scorecard = pd.DataFrame(vendor_metrics)
        scorecard = scorecard.sort_values("lending_score", ascending=False)
        
        return scorecard

if __name__ == "__main__":
    # Example usage
    engine = VendorAnalyticsEngine("path/to/best/model")
    
    # Example vendor data (in practice this would come from your database)
    vendor_data = {
        "vendor1": pd.read_csv("vendor1_data.csv"),
        "vendor2": pd.read_csv("vendor2_data.csv"),
        "vendor3": pd.read_csv("vendor3_data.csv")
    }
    
    # Generate scorecard
    scorecard = engine.generate_vendor_scorecard(vendor_data)
    
    # Save scorecard to CSV
    scorecard.to_csv("vendor_scorecard.csv", index=False)
    
    # Print top vendors
    print("Top Vendors:")
    print(scorecard.head())
