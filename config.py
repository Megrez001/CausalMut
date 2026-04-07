import os
import torch

class Config:
    DATA_PATH = "data/kcat.xlsx"
    CACHE_DIR = "cache"
    
    AA_VOCAB_SIZE = 21
    SMILES_VOCAB_SIZE = 50
    ESM_FEATURE_DIM = 1280
    MOLT5_FEATURE_DIM = 768
    MACCS_FEATURE_DIM = 167
    
    EMBEDDING_DIM = 128
    NUM_FILTERS = 128
    KERNEL_SIZES = [3, 5, 7]
    
    TRANSFORMER_D_MODEL = 384
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_DIM_FEEDFORWARD = 2048
    TRANSFORMER_DROPOUT = 0.1
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 300
    LEARNING_RATE = 0.001
    SEED = 3407
    
    FOCAL_ALPHA = 1.0
    FOCAL_BETA = 1.0
    FOCAL_GAMMA = 0.8
    
    USE_CACHED_FEATURES = True
    MOLT5_MODEL_PATH = "molt5-base"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()
os.makedirs(config.CACHE_DIR, exist_ok=True)