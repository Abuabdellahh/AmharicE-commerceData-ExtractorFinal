import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class VendorAnalyzer:
    def __init__(self):
        self.vendor_metrics = {}
        self.metrics_weights = {
            'posting_frequency': 0.3,
            'view_stats': 0.3,
            'price_consistency': 0.2,
            'engagement_score': 0.2
        }

    def calculate_posting_frequency(self, posts: List[Dict]) -> float:
        """
        Calculate posting frequency score for a vendor
        
        Args:
            posts (List[Dict]): List of posts with timestamps
            
        Returns:
            float: Posting frequency score (0-1)
        """
        if not posts:
            return 0
            
        timestamps = [post['timestamp'] for post in posts]
        time_range = (max(timestamps) - min(timestamps)).days + 1
        
        if time_range == 0:
            return 0
            
        post_frequency = len(posts) / time_range
        # Normalize to 0-1 range
        return min(post_frequency / 5, 1)  # Assuming 5 posts/day is maximum

    def calculate_view_stats(self, posts: List[Dict]) -> Tuple[float, float]:
        """
        Calculate view statistics for a vendor
        
        Args:
            posts (List[Dict]): List of posts with view counts
            
        Returns:
            Tuple[float, float]: (average_views, view_consistency_score)
        """
        if not posts:
            return (0, 0)
            
        view_counts = [post['views'] for post in posts if 'views' in post]
        if not view_counts:
            return (0, 0)
            
        avg_views = np.mean(view_counts)
        std_dev = np.std(view_counts)
        
        # Calculate consistency score (lower std_dev is better)
        consistency_score = max(0, 1 - (std_dev / avg_views) if avg_views > 0 else 0)
        
        return avg_views, consistency_score

    def calculate_price_metrics(self, posts: List[Dict]) -> Dict:
        """
        Calculate price-related metrics for a vendor
        
        Args:
            posts (List[Dict]): List of posts with price information
            
        Returns:
            Dict: Price metrics including average price and price consistency
        """
        if not posts:
            return {'avg_price': 0, 'price_consistency': 0}
            
        prices = [post['price'] for post in posts if 'price' in post]
        if not prices:
            return {'avg_price': 0, 'price_consistency': 0}
            
        avg_price = np.mean(prices)
        price_std = np.std(prices)
        
        # Calculate price consistency (lower std_dev is better)
        price_consistency = max(0, 1 - (price_std / avg_price) if avg_price > 0 else 0)
        
        return {
            'avg_price': avg_price,
            'price_consistency': price_consistency
        }

    def calculate_engagement_score(self, posts: List[Dict]) -> float:
        """
        Calculate engagement score based on likes, comments, and shares
        
        Args:
            posts (List[Dict]): List of posts with engagement data
            
        Returns:
            float: Engagement score (0-1)
        """
        if not posts:
            return 0
            
        total_engagement = 0
        total_possible = 0
        
        for post in posts:
            engagement = 0
            possible = 0
            
            if 'likes' in post:
                engagement += post['likes']
                possible += 1
            if 'comments' in post:
                engagement += post['comments']
                possible += 1
            if 'shares' in post:
                engagement += post['shares']
                possible += 1
            
            if possible > 0:
                total_engagement += engagement
                total_possible += possible
                
        if total_possible == 0:
            return 0
            
        return min(total_engagement / total_possible, 1)

    def calculate_lending_score(self, vendor_metrics: Dict) -> float:
        """
        Calculate overall lending score for a vendor
        
        Args:
            vendor_metrics (Dict): Dictionary of vendor metrics
            
        Returns:
            float: Lending score (0-1)
        """
        # Normalize all metrics to 0-1 range
        normalized_metrics = {
            'posting_frequency': min(vendor_metrics['posting_frequency'] / 5, 1),
            'view_consistency': vendor_metrics['view_consistency'],
            'price_consistency': vendor_metrics['price_consistency'],
            'engagement_score': vendor_metrics['engagement_score']
        }
        
        # Calculate weighted score
        score = sum(
            normalized_metrics[metric] * weight 
            for metric, weight in self.metrics_weights.items()
        )
        
        return score

    def analyze_vendor(self, vendor_id: str, posts: List[Dict]) -> Dict:
        """
        Analyze a vendor's performance and calculate lending score
        
        Args:
            vendor_id (str): Unique vendor identifier
            posts (List[Dict]): List of posts by the vendor
            
        Returns:
            Dict: Vendor analysis report including metrics and lending score
        """
        # Calculate all metrics
        posting_frequency = self.calculate_posting_frequency(posts)
        avg_views, view_consistency = self.calculate_view_stats(posts)
        price_metrics = self.calculate_price_metrics(posts)
        engagement_score = self.calculate_engagement_score(posts)
        
        # Calculate lending score
        metrics = {
            'posting_frequency': posting_frequency,
            'avg_views': avg_views,
            'view_consistency': view_consistency,
            'avg_price': price_metrics['avg_price'],
            'price_consistency': price_metrics['price_consistency'],
            'engagement_score': engagement_score
        }
        
        lending_score = self.calculate_lending_score(metrics)
        
        # Store metrics
        self.vendor_metrics[vendor_id] = {
            'metrics': metrics,
            'lending_score': lending_score,
            'total_posts': len(posts),
            'last_active': max(post['timestamp'] for post in posts).strftime('%Y-%m-%d')
        }
        
        return self.vendor_metrics[vendor_id]

    def get_vendor_ranking(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N vendors by lending score
        
        Args:
            n (int): Number of top vendors to return
            
        Returns:
            pd.DataFrame: DataFrame of top vendors with their metrics
        """
        if not self.vendor_metrics:
            return pd.DataFrame()
            
        metrics_df = pd.DataFrame.from_dict(self.vendor_metrics, orient='index')
        metrics_df = metrics_df.sort_values('lending_score', ascending=False)
        
        return metrics_df.head(n)

    def generate_vendor_report(self, vendor_id: str) -> Dict:
        """
        Generate detailed report for a specific vendor
        
        Args:
            vendor_id (str): Vendor identifier
            
        Returns:
            Dict: Comprehensive vendor report
        """
        if vendor_id not in self.vendor_metrics:
            return {}
            
        vendor_data = self.vendor_metrics[vendor_id]
        metrics = vendor_data['metrics']
        
        report = {
            'vendor_id': vendor_id,
            'lending_score': vendor_data['lending_score'],
            'total_posts': vendor_data['total_posts'],
            'last_active': vendor_data['last_active'],
            'metrics': {
                'posting_frequency': {
                    'score': metrics['posting_frequency'],
                    'weight': self.metrics_weights['posting_frequency']
                },
                'views': {
                    'average': metrics['avg_views'],
                    'consistency': metrics['view_consistency'],
                    'weight': self.metrics_weights['view_stats']
                },
                'prices': {
                    'average': metrics['avg_price'],
                    'consistency': metrics['price_consistency'],
                    'weight': self.metrics_weights['price_consistency']
                },
                'engagement': {
                    'score': metrics['engagement_score'],
                    'weight': self.metrics_weights['engagement_score']
                }
            }
        }
        
        return report
