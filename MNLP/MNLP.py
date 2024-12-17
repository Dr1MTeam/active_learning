import os
import sys 
import torch
import numpy as np

sys.path.append(os.path.abspath('..'))
from base_trainer import Trainer
import torch.nn.functional as F

BATCH_ADD_SIZE = 500

class MNLPTrainer(Trainer):
    def select_samples(self, num_samples):
        self.model.eval()  # Set the model to evaluation mode
        all_scores = []  # List to store MNLP scores for each sample
        all_indices = []  # List to store indices of the samples
        current_index = 0  # Начальный индекс

        with torch.no_grad():
            for batch in self.pool_loader:
                inputs, _ = batch  
                inputs = inputs.to(self.device)

                log_probs = torch.log_softmax(self.model(inputs), dim=-1) 
                #print(log_probs)
                # Для каждого примера суммируем логарифмы вероятностей по длине последовательности
                sequence_log_probs = (log_probs).sum(dim=1)
                # Нормализуем логарифмы вероятностей по длине последовательности
                sequence_length = log_probs.size(1)  # Получаем длину последовательности
                normalized_log_probs = sequence_log_probs / sequence_length

                
                # Добавляем индексы текущего батча
                batch_size = inputs.size(0)
                batch_indices = torch.arange(current_index, current_index + batch_size)

                current_index += batch_size

                all_scores.append(normalized_log_probs.cpu())
                all_indices.append(batch_indices)
    
        #print(len(all_scores))
        all_scores = torch.cat(all_scores)
        all_indices = torch.cat(all_indices) 
        
        _, top_indices = torch.topk(all_scores, num_samples)  
        informative_indices = all_indices[top_indices].tolist()
        print(informative_indices)
        self.update_dataloader(informative_indices)

        return informative_indices
    
    def fit(self, end_data_amaunt = 10000, num_samples = BATCH_ADD_SIZE, epochs_for_batch = 5):

        """
        Полный цикл обучения.
        :param num_epochs: Количество эпох
        """
        num_epochs = epochs_for_batch  *end_data_amaunt // num_samples
        epoch = 1
        while (len(self.train_loader.dataset) <= end_data_amaunt):
            train_loss = self.train_step()
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
            print(f"Epoch {epoch}/{num_epochs} - Train data: {len(self.train_loader.dataset)}")
            print(f"Train Loss: {train_loss:.4f}")
            # print(f"Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
            if epoch % epochs_for_batch == 0:
                self.select_samples(num_samples)
                # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
            
            epoch+=1
        return self.val_step()

    def _fit(self, num_epochs):
        """
        Обучение на части данных
        """
        for epoch in range(num_epochs):
            train_loss = self.train_step()

            val_loss, accuracy, f1  = self.val_step()
            self.select_samples()

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")

    