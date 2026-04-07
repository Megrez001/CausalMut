import random
import numpy as np
import torch
import ast

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_mutant_sites(sites_str):
    sites = ast.literal_eval(sites_str)
    if isinstance(sites, list):
        return [int(site) for site in sites if str(site).isdigit()]
    return []

def parse_aa_list(aa_str):
    aa_list = ast.literal_eval(aa_str)
    if isinstance(aa_list, list):
        return aa_list
    return []

def extract_ec_features(ec_number):
    parts = ec_number.split('.')
    ec_features = [int(part) if part.isdigit() else 0 for part in parts]
    while len(ec_features) < 4:
        ec_features.append(0)
    return ec_features[:4]

def create_char_mappings():
    aa_to_idx = {
        'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
        'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
        'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
        'N': 2, 'Y': 18, 'M': 12
    }
    
    smiles_chars = 'BrRrClLl0123456789()[]+-=#HBCNOPSFIK@\\/'
    smiles_to_idx = {c: i+1 for i, c in enumerate(smiles_chars)}
    
    return aa_to_idx, smiles_to_idx