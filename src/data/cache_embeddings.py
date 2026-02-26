import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import json
from src.utils.config import config
from src.data.dataset import MyDataset
from src.models.bert_encoder import BertEncoder

def cache_embeddings():
    csv_path = os.path.join(config.PROCESSED_DATA_DIR, 'train.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    dataset = MyDataset(csv_path)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertEncoder(freeze=True).to(device)
    model.eval()
    
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            embeddings = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            all_embeddings.append(embeddings.cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    with open(os.path.join(config.DATA_DIR, 'user2id.json')) as f:
        num_users = len(json.load(f))
    with open(os.path.join(config.DATA_DIR, 'item2id.json')) as f:
        num_items = len(json.load(f))

    user_emb_sum = np.zeros((num_users, config.EMBEDDING_DIM), dtype=np.float32)
    user_emb_count = np.zeros((num_users, 1), dtype=np.float32)
    item_emb_sum = np.zeros((num_items, config.EMBEDDING_DIM), dtype=np.float32)
    item_emb_count = np.zeros((num_items, 1), dtype=np.float32)
    
    current_idx = 0
    for batch in tqdm(dataloader, desc="Aggregating"):
        u_idx = batch['user_idx'].numpy()
        i_idx = batch['item_idx'].numpy()
        batch_emb = all_embeddings[current_idx : current_idx + len(u_idx)]
        
        np.add.at(user_emb_sum, u_idx, batch_emb)
        np.add.at(user_emb_count, u_idx, 1)
        np.add.at(item_emb_sum, i_idx, batch_emb)
        np.add.at(item_emb_count, i_idx, 1)
        current_idx += len(u_idx)

    user_final = user_emb_sum / np.maximum(user_emb_count, 1)
    item_final = item_emb_sum / np.maximum(item_emb_count, 1)
    
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.save(os.path.join(config.CACHE_DIR, 'user_text_emb.npy'), user_final)
    np.save(os.path.join(config.CACHE_DIR, 'item_text_emb.npy'), item_final)

if __name__ == "__main__":
    cache_embeddings()