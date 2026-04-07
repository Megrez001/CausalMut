import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from config import config

class Trainer:
    def __init__(self, model, criterion, device):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0.0
        all_targets = []
        all_predictions = []
        
        for batch in train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch['target'], batch['ips_weight'])
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item() * batch['target'].size(0)
            all_targets.extend(batch['target'].cpu().numpy())
            all_predictions.extend(outputs.detach().cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_r2 = r2_score(all_targets, all_predictions)
        
        return train_loss, train_r2
    
    def evaluate(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        test_targets = []
        test_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch)
                loss = self.criterion(outputs, batch['target'], batch['ips_weight'])
                test_loss += loss.item() * batch['target'].size(0)
                test_targets.extend(batch['target'].cpu().numpy())
                test_predictions.extend(outputs.cpu().numpy())
        
        test_loss /= len(test_loader.dataset)
        test_r2 = r2_score(test_targets, test_predictions)
        
        return test_loss, test_r2, test_targets, test_predictions
    
    def save_model(self, path, epoch, train_loss, test_loss, train_r2, test_r2, **kwargs):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_r2': train_r2,
            'test_r2': test_r2,
            **kwargs
        }, path)
