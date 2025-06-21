import pandas as pd
import re
from typing import List, Tuple, Dict
import json

class CoNLLLabeler:
    def __init__(self):
        self.entity_patterns = {
            'PRODUCT': [
                r'(?:ሻይ|ቡና|coffee|tea)',
                r'(?:ሞባይል|phone|mobile)',
                r'(?:ልብስ|clothes|shirt|dress)',
                r'(?:ጫማ|shoes|boot)',
                r'(?:መጽሐፍ|book|kitab)',
                r'(?:መኪና|car|vehicle)',
                r'(?:ቤት|house|home)',
                r'(?:ኮምፒውተር|computer|laptop)'
            ],
            'PRICE': [
                r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:ብር|birr|ETB)',
                r'(?:ዋጋ|ዋጋው|price)\s*[:-]?\s*\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\d+\s*(?:ብር|birr)'
            ],
            'LOC': [
                r'(?:አዲስ\s*አበባ|addis\s*ababa)',
                r'(?:ቦሌ|bole)',
                r'(?:ፒያሳ|piassa)',
                r'(?:መርካቶ|merkato)',
                r'(?:ካዛንቺስ|kazanchis)',
                r'(?:ጎንደር|gondar)',
                r'(?:ባህር\s*ዳር|bahir\s*dar)',
                r'(?:ሐዋሳ|hawassa)',
                r'(?:ጅማ|jimma)'
            ]
        }
    
    def tokenize_for_conll(self, text: str) -> List[str]:
        """Tokenize text for CoNLL format"""
        # Split on whitespace and punctuation, keeping them separate
        tokens = re.findall(r'\S+', text)
        
        refined_tokens = []
        for token in tokens:
            # Further split punctuation
            parts = re.findall(r'[\u1200-\u137F]+|[a-zA-Z0-9]+|[^\s]', token)
            refined_tokens.extend([part for part in parts if part.strip()])
        
        return refined_tokens
    
    def auto_label_entities(self, tokens: List[str]) -> List[str]:
        """Automatically label entities using patterns"""
        labels = ['O'] * len(tokens)
        text = ' '.join(tokens)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
