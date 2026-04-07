import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader as TorchDataLoader

from config import config
from utils import set_seed
from feature_extractor import FeatureExtractor
from data_loader import DataLoader
from preprocess import Preprocessor
from dataset import EnzymePairDataset
from models import EnzymeModel
from loss import FocalIPSLoss
from train import Trainer

def main():
    set_seed(config.SEED)
    print(f"Using device: {config.DEVICE}")
    os.makedirs("data", exist_ok=True)
    
    feature_extractor = FeatureExtractor(config.DEVICE)
    data_loader = DataLoader(feature_extractor)
    preprocessor = Preprocessor()
    
    print("Loading data...")
    train_data, test_data, test_ips_values = data_loader.load_excel_data_split(
        config.DATA_PATH, use_cached_features=config.USE_CACHED_FEATURES
    )
    
    print("Preprocessing data...")
    processed_train_data, aa_to_idx, smiles_to_idx = preprocessor.preprocess_pair_data(train_data)
    processed_test_data, _, _ = preprocessor.preprocess_pair_data(test_data)
    
    train_dataset = EnzymePairDataset(processed_train_data)
    test_dataset = EnzymePairDataset(processed_test_data)
    train_loader = TorchDataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = TorchDataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    esm_feature_dim = len(processed_train_data[0]['wild_esm_features'])
    molt5_feature_dim = len(processed_train_data[0]['molt5_features'])
    
    model = EnzymeModel(
        aa_vocab_size=len(aa_to_idx) + 1,
        smiles_vocab_size=len(smiles_to_idx) + 2,
        max_site_length=processed_train_data[0]['max_site_length'],
        max_aa_length=processed_train_data[0]['max_aa_length'],
        esm_feature_dim=esm_feature_dim,
        molt5_feature_dim=molt5_feature_dim
    ).to(config.DEVICE)
    
    criterion = FocalIPSLoss(
        alpha=config.FOCAL_ALPHA,
        beta=config.FOCAL_BETA,
        gamma=config.FOCAL_GAMMA
    ).to(config.DEVICE)
    
    trainer = Trainer(model, criterion, config.DEVICE)
    
    best_test_r2 = -float('inf')
    best_predictions = None
    best_targets = None
    best_test_data = None
    
    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_r2 = trainer.train_epoch(train_loader)
        test_loss, test_r2, test_targets, test_predictions = trainer.evaluate(test_loader)
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_predictions = test_predictions
            best_targets = test_targets
            best_test_data = processed_test_data
            
            trainer.save_model(
                "BEST_kcat.pth",
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_r2=train_r2,
                test_r2=test_r2,
                aa_vocab_size=len(aa_to_idx) + 1,
                smiles_vocab_size=len(smiles_to_idx) + 2,
                max_site_length=processed_train_data[0]['max_site_length'],
                max_aa_length=processed_train_data[0]['max_aa_length'],
                esm_feature_dim=esm_feature_dim,
                molt5_feature_dim=molt5_feature_dim
            )
            
            prediction_df = pd.DataFrame({
                'wild_sequence': [item['wild_sequence'] for item in best_test_data],
                'mutant_sequence': [item['mutant_sequence'] for item in best_test_data],
                'smiles': [item['smiles'] for item in best_test_data],
                'wild_log10_Kcat': [item['wild_log10_Kcat'] for item in best_test_data],
                'target_true': best_targets,
                'target_pred': best_predictions,
                'ips': [item['ips'] for item in best_test_data]
            })
            prediction_df.to_excel("data/best_predictions.xlsx", index=False)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/{config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train R²: {train_r2:.4f} | Test Loss: {test_loss:.4f} | Test R²: {test_r2:.4f}')
    
    print("Final Test Results:")
    test_rmse = np.sqrt(mean_squared_error(best_targets, best_predictions))
    test_mae = mean_absolute_error(best_targets, best_predictions)
    test_r2 = r2_score(best_targets, best_predictions)
    test_pearson = np.corrcoef(best_targets, best_predictions)[0, 1]
    
    print(f'RMSE:  {test_rmse:.4f}')
    print(f'MAE:   {test_mae:.4f}')
    print(f'R²:    {test_r2:.4f}')
    print(f'Pearson correlation: {test_pearson:.4f}')
    
    return model, group_results

if __name__ == "__main__":
    model, group_results = main()