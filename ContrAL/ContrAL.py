import sys
import torch
import os
import numpy as np
sys.path.append(os.path.abspath('..'))

from sklearn.neighbors import NearestNeighbors
from torch.nn.functional import softmax

from base_trainer import Trainer
BATCH_ADD_SIZE = 500
class contrastive_AL(Trainer):

    def compute_encodings_and_probs(self, dataloader):

        self.model.eval()
        encodings, probabilities = [], []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)  
                outputs = self.model(inputs) # получили эмбеддинги
                probs = softmax(outputs, dim=-1).cpu().numpy()  # получили вероятности
                #print(probs)
                encodings.append(outputs.cpu().numpy())
                probabilities.append(probs)
        #print(encodings)
        return np.vstack(encodings), np.vstack(probabilities)

    def select_samples(self, num_samples):
        k = num_samples if num_samples < 128 else 128
        labeled_encodings, labeled_probs = self.compute_encodings_and_probs(self.train_loader)
        unlabeled_encodings, unlabeled_probs = self.compute_encodings_and_probs(self.pool_loader)

        knn = NearestNeighbors(n_neighbors=k).fit(labeled_encodings)
        neighbors_idx = knn.kneighbors(unlabeled_encodings, return_distance=False)
        print(len(neighbors_idx))
        # Compute contrastive scores
        cal_scores = []
        
        # Векторизованный вариант KL-дивергенции
        neighbor_probs = labeled_probs[neighbors_idx]  # [num_unlabeled, k, num_classes]
        unlabeled_probs_expanded = unlabeled_probs[:, np.newaxis, :]  # [num_unlabeled, 1, num_classes]

        kl_div = np.sum(neighbor_probs * np.log(neighbor_probs / unlabeled_probs_expanded), axis=-1)
        cal_scores = np.mean(kl_div, axis=1)  # Средний KL по k соседям
        top_indices = np.argsort(cal_scores)[-num_samples:]
        self.update_dataloader(top_indices)
        return top_indices
        for i, neighbors in enumerate(neighbors_idx):
            # Calculate KL divergence between unlabeled sample and its neighbors
            neighbor_probs = labeled_probs[neighbors]
            kl_scores = [
                np.sum(prob * np.log(prob / unlabeled_probs[i]))
                for prob in neighbor_probs
            ]
            # Average KL divergence over neighbors
            cal_scores.append(np.mean(kl_scores))

        # Select the indices of the top-k samples with the highest CAL scores
        top_indices = np.argsort(cal_scores)[:num_samples]

        self.update_dataloader(top_indices)

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
        for epoch in range(num_epochs):
            train_loss = self.train_step()
            val_loss, accuracy, f1 = self.val_step()
            self.select_samples()
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")
