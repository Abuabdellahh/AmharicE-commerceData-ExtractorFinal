import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import unicodedata
from datetime import datetime

class AmharicTextPreprocessor:
    def __init__(self):
        # Amharic Unicode range
        self.amharic_range = r'[\u1200-\u137F]'
        
        # Common Amharic stopwords (basic set)
        self.amharic_stopwords = {
            'እና', 'ወይም', 'ነው', 'ናት', 'ናቸው', 'አለ', 'አላት', 'አላቸው',
            'ይህ', 'ይህች', 'ይህን', 'እዚህ', 'እዚያ', 'እነዚህ', 'እነዚያ'
        }
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize Amharic text"""
        if not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '...', text)
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        return text.strip()
    
    def extract_entities_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entities using regex patterns"""
        entities = {
            'prices': [],
            'phone_numbers': [],
            'locations': [],
            'products': []
        }
        
        # Price patterns (Ethiopian Birr)
        price_patterns = [
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር|birr|ETB)',
            r'(?:ዋጋ|ዋጋው|price)\s*[:-]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:ብር|birr)'
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['prices'].extend(matches)
        
        # Phone number patterns
        phone_patterns = [
            r'(?:\+251|0)?[97]\d{8}',
            r'(?:\+251|0)?[97]\d{2}[-\s]?\d{3}[-\s]?\d{3}'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            entities['phone_numbers'].extend(matches)
        
        # Location patterns (Ethiopian cities/areas)
        location_keywords = [
            'አዲስ አበባ', 'አዲስ', 'አበባ', 'ቦሌ', 'ፒያሳ', 'መርካቶ', 'ካዛንቺስ',
            'ጎንደር', 'ባህር ዳር', 'ሐዋሳ', 'ጅማ', 'ደሴ', 'አዋሳ', 'ሞጆ',
            'addis ababa', 'bole', 'piassa', 'merkato', 'kazanchis'
        ]
        
        for keyword in location_keywords:
            if keyword.lower() in text.lower():
                entities['locations'].append(keyword)
        
        return entities
    
    def tokenize_amharic(self, text: str) -> List[str]:
        """Tokenize Amharic text"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        
        # Further split on punctuation while keeping Amharic characters together
        refined_tokens = []
        for token in tokens:
            # Split punctuation from words
            parts = re.findall(r'[\u1200-\u137F]+|[a-zA-Z0-9]+|[^\s\u1200-\u137Fa-zA-Z0-9]', token)
            refined_tokens.extend([part for part in parts if part.strip()])
        
        return refined_tokens
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the entire dataframe"""
        df_processed = df.copy()
        
        # Clean text
        df_processed['cleaned_text'] = df_processed['text'].apply(self.clean_text)
        
        # Extract entities
        df_processed['extracted_entities'] = df_processed['cleaned_text'].apply(
            self.extract_entities_patterns
        )
        
        # Tokenize
        df_processed['tokens'] = df_processed['cleaned_text'].apply(self.tokenize_amharic)
        
        # Add text statistics
        df_processed['text_length'] = df_processed['cleaned_text'].str.len()
        df_processed['token_count'] = df_processed['tokens'].apply(len)
        df_processed['has_amharic'] = df_processed['cleaned_text'].str.contains(
            self.amharic_range, regex=True
        )
        
        # Convert date strings to datetime
        if 'date' in df_processed.columns:
            df_processed['date'] = pd.to_datetime(df_processed['date'])
            df_processed['hour'] = df_processed['date'].dt.hour
            df_processed['day_of_week'] = df_processed['date'].dt.day_name()
        
        return df_processed

def main():
    # Load raw data
    df = pd.read_csv('data/raw/telegram_messages.csv')
    
    # Initialize preprocessor
    preprocessor = AmharicTextPreprocessor()
    
    # Preprocess data
    df_processed = preprocessor.preprocess_dataframe(df)
    
    # Save processed data
    df_processed.to_csv('data/processed/preprocessed_messages.csv', index=False)
    
    # Print statistics
    print(f"Total messages: {len(df_processed)}")
    print(f"Messages with Amharic text: {df_processed['has_amharic'].sum()}")
    print(f"Average text length: {df_processed['text_length'].mean():.2f}")
    print(f"Average token count: {df_processed['token_count'].mean():.2f}")

if __name__ == "__main__":
    main()
