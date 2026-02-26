import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
from src.utils.config import config

class MyDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, max_length=128):
        """
        Args:
            data_path (str): Path to the csv file (train/val/test).
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer.
            max_length (int): Max token length.
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        text = str(row['title']) + " [SEP] " + str(row['text'])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'item_idx': torch.tensor(row['item_idx'], dtype=torch.long),
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'rating': torch.tensor(row['rating'], dtype=torch.float)
        }
