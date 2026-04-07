import os
import numpy as np
import torch
import esm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from transformers import AutoTokenizer, AutoModel
from config import config

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        self.esm_model = None
        self.esm_alphabet = None
        self.esm_batch_converter = None
        self.molt5_model = None
        self.molt5_tokenizer = None
    
    def load_esm_model(self):
        if self.esm_model is None:
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_model = self.esm_model.to(self.device)
            self.esm_model.eval()
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        return self.esm_model, self.esm_alphabet, self.esm_batch_converter
    
    def load_molt5_model(self, model_path):
        if self.molt5_model is None:
            self.molt5_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.molt5_model = AutoModel.from_pretrained(model_path).to(self.device)
            self.molt5_model.eval()
        return self.molt5_model, self.molt5_tokenizer
    
    def extract_esm_features(self, sequences, batch_size=8, cache_file=None):
        if cache_file and os.path.exists(cache_file):
            return np.load(cache_file)
        
        self.load_esm_model()
        all_features = []
        esm_data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        
        for i in range(0, len(esm_data), batch_size):
            batch_data = esm_data[i:i+batch_size]
            batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(batch_data)
            batch_tokens = batch_tokens.to(self.device)
            
            with torch.no_grad():
                results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
                token_representations = results["representations"][33]
                sequence_representations = []
                for j, (_, seq) in enumerate(batch_data):
                    seq_len = len(seq)
                    seq_repr = token_representations[j, 1:seq_len+1].mean(dim=0)
                    sequence_representations.append(seq_repr.cpu().numpy())
                all_features.extend(sequence_representations)
        
        features_array = np.array(all_features)
        if cache_file:
            np.save(cache_file, features_array)
        return features_array
    
    def extract_molt5_features(self, smiles_list, batch_size=16, cache_file=None):
        if cache_file and os.path.exists(cache_file):
            return np.load(cache_file)
        
        self.load_molt5_model()
        all_features = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i+batch_size]
            inputs = self.molt5_tokenizer(
                batch_smiles, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                encoder_outputs = self.molt5_model.encoder(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                last_hidden_states = encoder_outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = last_hidden_states * attention_mask
                sequence_lengths = attention_mask.sum(dim=1)
                sequence_lengths = torch.where(
                    sequence_lengths == 0, torch.ones_like(sequence_lengths), sequence_lengths
                )
                pooled_features = masked_hidden.sum(dim=1) / sequence_lengths
                all_features.extend(pooled_features.cpu().numpy())
        
        features_array = np.array(all_features)
        if cache_file:
            np.save(cache_file, features_array)
        return features_array
    
    def extract_maccs_features(self, smiles_list, cache_file=None):
        if cache_file and os.path.exists(cache_file):
            return np.load(cache_file)
        
        maccs_features = []
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                    maccs_array = np.array(maccs_fp, dtype=np.float32)
                    maccs_features.append(maccs_array)
                else:
                    maccs_features.append(np.zeros(167, dtype=np.float32))
            except:
                maccs_features.append(np.zeros(167, dtype=np.float32))
        
        features_array = np.array(maccs_features)
        if cache_file:
            np.save(cache_file, features_array)
        return features_array