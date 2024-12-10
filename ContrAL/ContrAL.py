import sys
import torch
import os
import numpy as np
from omegaconf import OmegaConf
sys.path.append(os.path.abspath('..'))

from sklearn.neighbors import NearestNeighbors
from torch.nn.functional import softmax

from base_trainer import Trainer
config = OmegaConf.load('config.yaml')
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

    def select_samples(self):
        num_samples = config['batch_size']
        k = config['batch_size']
        labeled_encodings, labeled_probs = self.compute_encodings_and_probs(self.train_loader)
        unlabeled_encodings, unlabeled_probs = self.compute_encodings_and_probs(self.pool_loader)

        knn = NearestNeighbors(n_neighbors=k).fit(labeled_encodings)
        neighbors_idx = knn.kneighbors(unlabeled_encodings, return_distance=False)
        print(len(neighbors_idx))
        # Compute contrastive scores
        cal_scores = []
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
        top_indices = np.argsort(cal_scores)[-num_samples:]

        self.update_dataloader(top_indices)

    

    def fit(self, num_epochs):
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
