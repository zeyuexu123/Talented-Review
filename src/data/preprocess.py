import pandas as pd
import numpy as np
import os
from src.utils.config import config
import json

def load_and_filter_data(filepath=None, sample_size=None):

    filepath = filepath or config.RAW_DATA_PATH

    print(f"Reading from {filepath}...")
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
                
    df = pd.DataFrame(data)

    if 'parent_asin' in df.columns:
        df['item_id'] = df['parent_asin'].fillna(df['asin'])
    else:
        df['item_id'] = df['asin']
        
    if 'sort_timestamp' in df.columns:
        df['timestamp'] = df['sort_timestamp']
    
    required_cols = ['user_id', 'item_id', 'rating', 'timestamp', 'title', 'text', 'verified_purchase']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Missing column {col}")
                
    df = df[required_cols]
    
    return process_dataframe(df)

def process_dataframe(df):
    """
    Applies filtering and 5-core logic.
    """
    print(f"Initial shape: {df.shape}")
    
    # (Optional) Filter verified purchases only
    # df = df[df['verified_purchase'] == True]
    
    
    # (Optional) K-core filtering
    # user_counts = df['user_id'].value_counts()
    # item_counts = df['item_id'].value_counts()
    
    # valid_users = user_counts[user_counts >= config.MIN_USER_INTERACTIONS].index
    # valid_items = item_counts[item_counts >= config.MIN_ITEM_INTERACTIONS].index

    # df = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
    
    print(f"Post-filtering shape: {df.shape}")
    
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    
    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {i: j for j, i in enumerate(item_ids)}
    
    df['user_idx'] = df['user_id'].map(user2id)
    df['item_idx'] = df['item_id'].map(item2id)
    
    import json
    with open(os.path.join(config.DATA_DIR, 'user2id.json'), 'w', encoding='utf-8') as f:
        json.dump({str(k): int(v) for k, v in user2id.items()}, f)
    with open(os.path.join(config.DATA_DIR, 'item2id.json'), 'w', encoding='utf-8') as f:
        json.dump({str(k): int(v) for k, v in item2id.items()}, f)

    return df

def temporal_split(df):

    df = df.sort_values('timestamp')
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

def save_splits(train, val, test):
    train.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'train.csv'), index=False)
    val.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'val.csv'), index=False)
    test.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'test.csv'), index=False)
    print("Splits saved.")

if __name__ == "__main__":
    df = load_and_filter_data(None)
    
    if df is not None:
        train, val, test = temporal_split(df)
        save_splits(train, val, test)
