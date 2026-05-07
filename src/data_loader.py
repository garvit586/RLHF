"""
Data loading utilities for RLHF pipeline
"""
import json
from typing import List, Dict, Tuple
from torch.utils.data import Dataset
import torch


class PreferenceDataset(Dataset):
    """Dataset for preference/ranking data"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(file_path)
    
    def _load_data(self, file_path: str) -> List[Dict]:
        """Load preference data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text'].replace('[summary]:', '').strip()
        
        # Tokenize input text
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get chosen and rejected summaries
        choice = item['choice']
        chosen_text = item[f'candidate_{choice}']
        rejected_text = item[f'candidate_{1-choice}']
        
        # Tokenize chosen and rejected summaries
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(),
        }


class PromptDataset(Dataset):
    """Dataset for prompt data (for inference and RL training)"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(file_path)
    
    def _load_data(self, file_path: str) -> List[Dict]:
        """Load prompt data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text'].replace('[summary]:', '').strip()
        
        # Tokenize input text
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': input_text
        }


def load_preference_data(file_path: str) -> List[Dict]:
    """Load preference data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_prompt_data(file_path: str) -> List[Dict]:
    """Load prompt data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

