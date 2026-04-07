from utils import create_char_mappings

class Preprocessor:
    def __init__(self):
        self.aa_to_idx, self.smiles_to_idx = create_char_mappings()
    
    def preprocess_pair_data(self, data):
        processed = []
        max_seq_len = data[0]['max_seq_len']
        max_smiles_len = data[0]['max_smiles_len']
        max_site_length = data[0]['max_site_length']
        max_aa_length = data[0]['max_aa_length']
        
        for entry in data:
            wild_seq = [self.aa_to_idx.get(aa, 20) for aa in entry['wild_sequence']]
            mutant_seq = [self.aa_to_idx.get(aa, 20) for aa in entry['mutant_sequence']]
            smiles = [self.smiles_to_idx.get(c, 0) for c in entry['smiles']]
            wild_aa_features = [self.aa_to_idx.get(aa, 20) for aa in entry['wild_aas'] if aa]
            mutant_aa_features = [self.aa_to_idx.get(aa, 20) for aa in entry['mutant_aas'] if aa]
            
            processed.append({
                'wild_sequence': wild_seq,
                'mutant_sequence': mutant_seq,
                'smiles': smiles,
                'wild_log10_Kcat': entry['wild_log10_Kcat'],
                'ec_features': entry['ec_features'],
                'target': entry['target'],
                'mutant_sites': entry['mutant_sites'],
                'wild_aa_features': wild_aa_features,
                'mutant_aa_features': mutant_aa_features,
                'wild_esm_features': entry['wild_esm_features'],
                'mutant_esm_features': entry['mutant_esm_features'],
                'maccs_features': entry['maccs_features'],
                'molt5_features': entry['molt5_features'],
                'ips': entry['ips'],
                'ips_weight': entry['ips_weight'],
                'max_seq_len': max_seq_len,
                'max_smiles_len': max_smiles_len,
                'max_site_length': max_site_length,
                'max_aa_length': max_aa_length,
            })
        
        return processed, self.aa_to_idx, self.smiles_to_idx