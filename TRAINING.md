# Training Guide

This guide details the steps to prepare data and cache embeddings

## 1. Environment Setup

Ensure you have the required dependencies.

Note: It's HIGHLY RECOMMENDED to run the project on GPUs. 

Windows:

```bash
pip install -r requirements.txt
```

## 2. Data Preparation

### 2.1 Download Data
Download any arbitary category dataset from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
Place the `.jsonl` file in `data/raw/`.

### 2.2 Preprocess
Modify RAW_DATA_PATH in src/utils/config.py. Then, run the preprocessing script.

```bash
python -m src.data.preprocess
```

*Outputs:* `data/processed/{train,val,test}.csv`, `data/user2id.json`, `data/item2id.json`.

## 3. Embedding Caching

This script runs BERT inference on all reviews and aggregates them into **User Profiles** and **Item Profiles** (mean).

```bash
python -m src.data.cache_embeddings
```

*Outputs:* `data/cache/user_text_emb.npy`, `data/cache/item_text_emb.npy`.