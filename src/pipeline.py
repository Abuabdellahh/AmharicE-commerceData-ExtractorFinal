import os
import json
from typing import List, Dict, Any
from models.ner_model import AmharicNERModel
from models.model_comparison import ModelComparator
from models.model_interpretability import ModelInterpreter
from labeling.conll_labeler import CoNLLLabeler
from data.telegram_scraper import TelegramScraper
from data.text_preprocessor import TextPreprocessor

class AmharicECommercePipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.labeler = CoNLLLabeler()
        self.scraper = TelegramScraper()
        self.preprocessor = TextPreprocessor()
        self.model = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            import yaml
            return yaml.safe_load(f)
    
    def scrape_data(self) -> List[Dict]:
        """Scrape data from Telegram channels"""
        channels = self.config['telegram']['channels']
        data = []
        for channel in channels:
            posts = self.scraper.scrape_channel(channel)
            data.extend(posts)
        return data
    
    def preprocess_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Preprocess scraped data"""
        processed_data = []
        for post in raw_data:
            text = post['text']
            clean_text = self.preprocessor.clean_text(text)
            tokens = self.preprocessor.tokenize(clean_text)
            
            # Auto-label entities
            labels = self.labeler.auto_label_entities(tokens)
            
            processed_data.append({
                'text': text,
                'clean_text': clean_text,
                'tokens': tokens,
                'labels': labels,
                'metadata': post.get('metadata', {})
            })
        return processed_data
    
    def train_model(self, train_data: List[Dict], model_name: str = None) -> None:
        """Train NER model"""
        if model_name is None:
            model_name = self.config['model']['base_model']
        self.model = AmharicNERModel(model_name)
        self.model.train(
            train_data,
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size']
        )
    
    def evaluate_models(self, test_data: List[Dict]) -> Dict:
        """Compare different models and select the best one"""
        comparator = ModelComparator()
        results = comparator.compare_models(
            test_data,
            output_dir=self.config['output']['comparison_dir']
        )
        return results
    
    def generate_interpretability_report(self, test_data: List[Dict]) -> None:
        """Generate model interpretability report"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        interpreter = ModelInterpreter(self.model.model_name)
        interpreter.generate_interpretability_report(
            texts=[d['text'] for d in test_data],
            labels=[d['labels'] for d in test_data],
            output_path=self.config['output']['interpretability_report']
        )
    
    def process_vendor_data(self, vendor_data: List[Dict]) -> Dict:
        """Process vendor data and calculate lending scores"""
        results = {}
        for vendor in self.config['vendors']:
            vendor_posts = [p for p in vendor_data if vendor in p.get('metadata', {}).get('vendor', '')]
            if vendor_posts:
                vendor_metrics = self._analyze_vendor(vendor_posts)
                results[vendor] = vendor_metrics
        return results
    
    def _analyze_vendor(self, posts: List[Dict]) -> Dict:
        """Analyze a single vendor's posts"""
        metrics = {
            'post_count': len(posts),
            'product_count': 0,
            'price_range': {'min': float('inf'), 'max': 0},
            'location_coverage': set(),
            'activity_score': 0
        }
        
        for post in posts:
            # Count products
            metrics['product_count'] += len([l for l in post['labels'] if 'PRODUCT' in l])
            
            # Track price range
            prices = [float(p['text']) for p in post['tokens'] if 'B-PRICE' in p['labels']]
            if prices:
                min_price = min(prices)
                max_price = max(prices)
                metrics['price_range']['min'] = min(metrics['price_range']['min'], min_price)
                metrics['price_range']['max'] = max(metrics['price_range']['max'], max_price)
            
            # Track locations
            locations = [t['text'] for t, l in zip(post['tokens'], post['labels']) if 'LOC' in l]
            metrics['location_coverage'].update(locations)
            
        # Calculate activity score
        metrics['activity_score'] = self._calculate_activity_score(metrics)
        return metrics
    
    def _calculate_activity_score(self, metrics: Dict) -> float:
        """Calculate vendor activity score"""
        score = 0
        
        # Post frequency score (30%)
        post_freq = metrics['post_count'] / self.config['vendor']['max_posts']
        score += min(post_freq, 1) * 30
        
        # Product diversity score (30%)
        product_div = metrics['product_count'] / self.config['vendor']['max_products']
        score += min(product_div, 1) * 30
        
        # Location coverage score (20%)
        loc_coverage = len(metrics['location_coverage']) / self.config['vendor']['max_locations']
        score += min(loc_coverage, 1) * 20
        
        # Price range score (20%)
        price_range = metrics['price_range']['max'] - metrics['price_range']['min']
        price_score = price_range / self.config['vendor']['max_price_range']
        score += min(price_score, 1) * 20
        
        return round(score, 2)
    
    def save_results(self, results: Dict, output_path: str) -> None:
        """Save results to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """Main pipeline execution"""
    pipeline = AmharicECommercePipeline()
    
    # Step 1: Data Collection
    print("Scraping data from Telegram...")
    raw_data = pipeline.scrape_data()
    
    # Step 2: Data Preprocessing
    print("\nPreprocessing data...")
    processed_data = pipeline.preprocess_data(raw_data)
    
    # Step 3: Train/Test Split
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    
    # Step 4: Model Training
    print("\nTraining model...")
    pipeline.train_model(train_data)
    
    # Step 5: Model Evaluation
    print("\nEvaluating models...")
    eval_results = pipeline.evaluate_models(test_data)
    pipeline.save_results(eval_results, pipeline.config['output']['evaluation_results'])
    
    # Step 6: Model Interpretability
    print("\nGenerating interpretability report...")
    pipeline.generate_interpretability_report(test_data)
    
    # Step 7: Vendor Analysis
    print("\nAnalyzing vendors...")
    vendor_results = pipeline.process_vendor_data(processed_data)
    pipeline.save_results(vendor_results, pipeline.config['output']['vendor_analysis'])
    
    print("\nPipeline completed successfully!")
    print(f"Results saved to: {pipeline.config['output']['base_dir']}")

if __name__ == "__main__":
    main()
