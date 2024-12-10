import os
import sys 
import torch
import numpy as np

sys.path.append(os.path.abspath('..'))
from base_trainer import Trainer
import torch.nn.functional as F

from omegaconf import OmegaConf
config = OmegaConf.load('config.yaml')

class MNLPTrainer(Trainer):
    def select_samples(self):
        self.model.eval()  # Set the model to evaluation mode
        all_scores = []  # List to store MNLP scores for each sample
        all_indices = []  # List to store indices of the samples
        current_index = 0  # Начальный индекс

        with torch.no_grad():
            for batch in self.pool_loader:
                inputs, labels = batch  # Assume loader returns data and their indices
                inputs = inputs.to(self.device)

                log_probs = torch.log_softmax(self.model(inputs), dim=-1) 
                #print(log_probs)
                # Для каждого примера суммируем логарифмы вероятностей по длине последовательности
                sequence_log_probs = (log_probs).sum(dim=1)
                # Нормализуем логарифмы вероятностей по длине последовательности
                normalized_log_probs = sequence_log_probs / config['batch_size']

                
                # Добавляем индексы текущего батча
                batch_size = inputs.size(0)
                batch_indices = torch.arange(current_index, current_index + batch_size)


                all_scores.append(normalized_log_probs.cpu())
                all_indices.append(batch_indices)
    
        #print(len(all_scores))
        all_scores = torch.cat(all_scores)
        all_indices = torch.cat(all_indices) 
        print(len(all_indices))
        print(len(all_scores))
    
        _, top_indices = torch.topk(-all_scores, config['num_samples'])  
        informative_indices = all_indices[top_indices].tolist()

        self.update_dataloader(informative_indices)

        return informative_indices

    def fit(self, num_epochs):
        """
        Полный цикл обучения.
        :param num_epochs: Количество эпох
        """
        for epoch in range(num_epochs):
            train_loss = self.train_step()

            val_loss, accuracy, f1  = self.val_step()
            self.select_samples()

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")

    