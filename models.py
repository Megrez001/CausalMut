import torch
import torch.nn as nn

class SequenceCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=128, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, ks) for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = nn.functional.relu(conv_out)
            pooled = nn.functional.max_pool1d(conv_out, conv_out.shape[2])
            conv_outputs.append(pooled.squeeze(2))
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return x

class SmilesCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_filters=128, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, ks) for ks in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = nn.functional.relu(conv_out)
            pooled = nn.functional.max_pool1d(conv_out, conv_out.shape[2])
            conv_outputs.append(pooled.squeeze(2))
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return x

class FeatureTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, wild_seq_features, mutant_seq_features, smiles_features):
        wild_seq = wild_seq_features.unsqueeze(1)
        mutant_seq = mutant_seq_features.unsqueeze(1)
        smiles_seq = smiles_features.unsqueeze(1)
        
        sequence = torch.cat([wild_seq, mutant_seq, smiles_seq], dim=1)
        residual = sequence
        
        attn_output, _ = self.multihead_attn(sequence, sequence, sequence, need_weights=False)
        sequence = self.norm1(residual + self.dropout(attn_output))
        
        residual = sequence
        ff_output = self.feed_forward(sequence)
        sequence = self.norm2(residual + self.dropout(ff_output))
        
        return sequence.mean(dim=1)

class EnzymeModel(nn.Module):
    def __init__(self, aa_vocab_size, smiles_vocab_size, max_site_length, max_aa_length,
                 esm_feature_dim=1280, molt5_feature_dim=768):
        super().__init__()
        self.alpha = 0.3
        
        self.seq_cnn = SequenceCNN(aa_vocab_size)
        self.smiles_cnn = SmilesCNN(smiles_vocab_size)
        
        self.feature_transformer = FeatureTransformer(
            d_model=384, nhead=8, dim_feedforward=2048, dropout=0.1
        )
        
        self.aa_embedding = nn.Embedding(aa_vocab_size, 32)
        
        total_features = (
            128 * 3 * 3 +
            128 * 3 +
            max_site_length +
            32 * 2 * max_aa_length +
            esm_feature_dim * 3 +
            167 +
            molt5_feature_dim +
            1 + 4
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, batch):
        wild_seq_features = self.seq_cnn(batch['wild_sequence'])
        mutant_seq_features = self.seq_cnn(batch['mutant_sequence'])
        smiles_features = self.smiles_cnn(batch['smiles'])
        
        seq_cnn_delta = wild_seq_features - mutant_seq_features
        interaction_features = self.feature_transformer(
            wild_seq_features, mutant_seq_features, smiles_features
        )
        enhanced_delta = seq_cnn_delta + self.alpha * interaction_features
        
        wild_aa_features = self.aa_embedding(batch['wild_aa_features'])
        mutant_aa_features = self.aa_embedding(batch['mutant_aa_features'])
        aa_features = torch.cat([wild_aa_features, mutant_aa_features], dim=2)
        aa_features = aa_features.view(aa_features.size(0), -1)
        
        delta_esm_features = batch['wild_esm_features'] - batch['mutant_esm_features']
        
        combined = torch.cat([
            wild_seq_features, mutant_seq_features, enhanced_delta,
            smiles_features, batch['mutant_sites'], aa_features,
            batch['wild_esm_features'], batch['mutant_esm_features'],
            delta_esm_features, batch['maccs_features'],
            batch['molt5_features'], batch['wild_log10_Kcat'],
            batch['ec_feats']
        ], dim=1)
        
        return self.regressor(combined).squeeze(1)