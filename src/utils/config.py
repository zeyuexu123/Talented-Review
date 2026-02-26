import os

class Config:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'All_Beauty.jsonl') # Your dataset name
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    CACHE_DIR = os.path.join(DATA_DIR, 'cache')
    
    MIN_USER_INTERACTIONS = 5
    MIN_ITEM_INTERACTIONS = 5
    MAX_TEXT_LENGTH = 256
    SAMPLE_SIZE = 5000000
    
    MODEL_NAME = 'bert-base-uncased'
    EMBEDDING_DIM = 768
    
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    SEED = 42

    def __init__(self):
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)

config = Config()
