import pandas as pd
import numpy as np
import os
from feature_extractor import FeatureExtractor
from utils import extract_ec_features, parse_mutant_sites, parse_aa_list
from config import config

class DataLoader:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def calculate_max_lengths(self, df):
        max_seq_len = 0
        max_smiles_len = 0
        max_site_length = 0
        max_aa_length = 0
        
        for _, row in df.iterrows():
            wild_seq = str(row['wild_Sequence'])
            mutant_seq = str(row['mutant_Sequence'])
            smiles = str(row['Smiles'])
            mutant_sites = parse_mutant_sites(str(row['Mutant Site']))
            wild_aas = parse_aa_list(str(row['wild_Amino Acid']))
            mutant_aas = parse_aa_list(str(row['mutant_Amino Acid']))
            
            max_seq_len = max(max_seq_len, len(wild_seq), len(mutant_seq))
            max_smiles_len = max(max_smiles_len, len(smiles))
            max_site_length = max(max_site_length, len(mutant_sites))
            max_aa_length = max(max_aa_length, len(wild_aas), len(mutant_aas))
        
        return max_seq_len, max_smiles_len, max_site_length, max_aa_length
    
    def load_excel_data_split(self, file_path, use_cached_features=True):
        train_df = pd.read_excel(file_path, sheet_name='Train_Data')
        test_df = pd.read_excel(file_path, sheet_name='Test_Data')
        
        all_df = pd.concat([train_df, test_df], ignore_index=True)
        max_seq_len, max_smiles_len, max_site_length, max_aa_length = self.calculate_max_lengths(all_df)
        
        train_data, train_ips = self._process_dataframe(
            train_df, 'train', max_seq_len, max_smiles_len, max_site_length, max_aa_length,
            use_cached_features, file_path
        )
        test_data, test_ips = self._process_dataframe(
            test_df, 'test', max_seq_len, max_smiles_len, max_site_length, max_aa_length,
            use_cached_features, file_path
        )
        
        return train_data, test_data, test_ips
    
    def _process_dataframe(self, df, data_type, max_seq_len, max_smiles_len, 
                          max_site_length, max_aa_length, use_cached_features, file_path):
        data = []
        all_wild_seqs = []
        all_mutant_seqs = []
        all_smiles = []
        ips_values = []
        ips_weights = []
        
        for _, row in df.iterrows():
            all_wild_seqs.append(str(row['wild_Sequence']))
            all_mutant_seqs.append(str(row['mutant_Sequence']))
            all_smiles.append(str(row['Smiles']))
            
            ips = float(row['IPS']) if 'IPS' in row and pd.notna(row['IPS']) else 1.0
            ips_weight = np.log10(ips) if ips > 0 else 0.0
            ips_values.append(ips)
            ips_weights.append(ips_weight)
        if use_cached_features:
            base_name = "kcat"
            wild_esm_cache = os.path.join(config.CACHE_DIR, f"{data_type}_wild_esm_features_{base_name}.npy")
            mutant_esm_cache = os.path.join(config.CACHE_DIR, f"{data_type}_mutant_esm_features_{base_name}.npy")
            maccs_cache = os.path.join(config.CACHE_DIR, f"{data_type}_maccs_features_{base_name}.npy")
            molt5_cache = os.path.join(config.CACHE_DIR, f"{data_type}_molt5_features_{base_name}.npy")
        else:
            wild_esm_cache = mutant_esm_cache = maccs_cache = molt5_cache = None

        print(f"Extracting ESM features for {data_type} wild sequences...")
        wild_esm_features = self.feature_extractor.extract_esm_features(all_wild_seqs, cache_file=wild_esm_cache)
        
        print(f"Extracting ESM features for {data_type} mutant sequences...")
        mutant_esm_features = self.feature_extractor.extract_esm_features(all_mutant_seqs, cache_file=mutant_esm_cache)
        
        print(f"Extracting MACCS features for {data_type} SMILES...")
        maccs_features = self.feature_extractor.extract_maccs_features(all_smiles, cache_file=maccs_cache)
        
        print(f"Extracting MolT5 features for {data_type} SMILES...")
        molt5_features = self.feature_extractor.extract_molt5_features(all_smiles, cache_file=molt5_cache)
        
        for idx, row in df.iterrows():
            mutant_sites = parse_mutant_sites(str(row['Mutant Site']))
            mutant_sites_padded = mutant_sites + [0] * (max_site_length - len(mutant_sites))
            
            wild_aas = parse_aa_list(str(row['wild_Amino Acid']))
            mutant_aas = parse_aa_list(str(row['mutant_Amino Acid']))
            
            data.append({
                'wild_sequence': str(row['wild_Sequence']),
                'mutant_sequence': str(row['mutant_Sequence']),
                'smiles': str(row['Smiles']),
                'wild_log10_Kcat': float(row['wild_log10_Kcat']),
                'ec_features': extract_ec_features(str(row['ECNumber'])),
                'target': float(row['mutant_log10_Kcat']),
                'mutant_sites': mutant_sites_padded,
                'wild_aas': wild_aas,
                'mutant_aas': mutant_aas,
                'wild_esm_features': wild_esm_features[idx].tolist(),
                'mutant_esm_features': mutant_esm_features[idx].tolist(),
                'maccs_features': maccs_features[idx].tolist(),
                'molt5_features': molt5_features[idx].tolist(),
                'ips': ips_values[idx],
                'ips_weight': ips_weights[idx],
                'max_seq_len': max_seq_len,
                'max_smiles_len': max_smiles_len,
                'max_site_length': max_site_length,
                'max_aa_length': max_aa_length,
            })
        
        return data, ips_values