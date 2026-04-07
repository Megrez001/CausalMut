import torch
import torch.nn as nn
from torch.utils.data import Dataset

class EnzymePairDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.max_seq_len = data[0]['max_seq_len']
        self.max_smiles_len = data[0]['max_smiles_len']
        self.max_site_length = data[0]['max_site_length']
        self.max_aa_length = data[0]['max_aa_length']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        wild_seq = torch.tensor(item['wild_sequence'][:self.max_seq_len], dtype=torch.long)
        wild_seq = nn.functional.pad(wild_seq, (0, max(0, self.max_seq_len - len(wild_seq))), value=20)
        
        mutant_seq = torch.tensor(item['mutant_sequence'][:self.max_seq_len], dtype=torch.long)
        mutant_seq = nn.functional.pad(mutant_seq, (0, max(0, self.max_seq_len - len(mutant_seq))), value=20)
        
        smiles = torch.tensor(item['smiles'][:self.max_smiles_len], dtype=torch.long)
        smiles = nn.functional.pad(smiles, (0, max(0, self.max_smiles_len - len(smiles))), value=0)
        
        mutant_sites = torch.tensor(item['mutant_sites'][:self.max_site_length], dtype=torch.float32)
        mutant_sites = nn.functional.pad(mutant_sites, (0, max(0, self.max_site_length - len(mutant_sites))), value=0.0)
        
        wild_aa_features = torch.tensor(item['wild_aa_features'][:self.max_aa_length], dtype=torch.long)
        wild_aa_features = nn.functional.pad(wild_aa_features, (0, max(0, self.max_aa_length - len(wild_aa_features))), value=20)
        
        mutant_aa_features = torch.tensor(item['mutant_aa_features'][:self.max_aa_length], dtype=torch.long)
        mutant_aa_features = nn.functional.pad(mutant_aa_features, (0, max(0, self.max_aa_length - len(mutant_aa_features))), value=20)
        
        return {
            'wild_sequence': wild_seq,
            'mutant_sequence': mutant_seq,
            'smiles': smiles,
            'mutant_sites': mutant_sites,
            'wild_aa_features': wild_aa_features,
            'mutant_aa_features': mutant_aa_features,
            'wild_esm_features': torch.tensor(item['wild_esm_features'], dtype=torch.float32),
            'mutant_esm_features': torch.tensor(item['mutant_esm_features'], dtype=torch.float32),
            'maccs_features': torch.tensor(item['maccs_features'], dtype=torch.float32),
            'molt5_features': torch.tensor(item['molt5_features'], dtype=torch.float32),
            'wild_log10_Kcat': torch.tensor([item['wild_log10_Kcat']], dtype=torch.float32),
            'ec_feats': torch.tensor(item['ec_features'], dtype=torch.float32),
            'target': torch.tensor(item['target'], dtype=torch.float32),
            'ips_weight': torch.tensor(item['ips_weight'], dtype=torch.float32)
        }