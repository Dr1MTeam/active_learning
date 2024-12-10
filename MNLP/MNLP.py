import os
import sys 
import torch

sys.path.append(os.path.abspath('..'))
from base_trainer import Trainer
import torch.nn.functional as F



class MNLPTrainer(Trainer):
    def select_samples(self):
        self.model.eval()  # Set the model to evaluation mode
        all_scores = []  # List to store MNLP scores for each sample
        all_indices = []  # List to store indices of the samples

        with torch.no_grad():
            for batch in self.pool_loader:
                inputs, indices = batch  # Assume loader returns data and their indices
                inputs = inputs.to(self.device)

                log_probs = torch.log_softmax(self.model(inputs), dim=-1)  # (batch_size, seq_len, num_classes)

                # Для каждого примера суммируем логарифмы вероятностей по длине последовательности
                sequence_log_probs = log_probs.sum(dim=-1)  # (batch_size, seq_len)

                # Усредняем значения логарифмов вероятностей по длине последовательности
                mean_log_probs = sequence_log_probs.mean(axis=0)  # (batch_size)

                # Нормализуем логарифмы вероятностей по длине последовательности
                normalized_log_probs = mean_log_probs / inputs.shape[1]  # (batch_size)

                # Save results
                all_scores.append(normalized_log_probs.cpu())
                all_indices.append(indices)
    

        # Combine results across all batches
        all_scores = torch.stack(all_scores) # All MNLP scores
        all_indices = torch.cat(all_indices)  # All indices

        # Select top-K indices with the lowest MNLP scores (most uncertain samples)
        _, top_indices = torch.topk(-all_scores, 10)  # Negative for ascending sort
        informative_indices = all_indices[top_indices].tolist()

        # Update dataloaders for the next iteration
        self.update_dataloader(informative_indices)

        return informative_indices

    def fit(self, num_epochs):
        """
        Полный цикл обучения.
        :param num_epochs: Количество эпох
        """
        for epoch in range(num_epochs):
            train_loss = self.train_step()
            self.select_samples()
            val_loss, accuracy, f1  = self.val_step()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")

    