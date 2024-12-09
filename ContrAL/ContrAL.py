from sklearn.metrics.pairwise import cosine_similarity
import sys
import torch
import os
import numpy as np

sys.path.append(os.path.abspath('..'))

from base_trainer import Trainer

class contrastive_AL(Trainer):
    def select_samples(self):
        self.model.eval()
        embeddings = []
        labels_list = []
        top_k=5
        # Проход по данным батчами для получения эмбеддингов
        with torch.no_grad():
            for inputs, batch_labels in self.pool_loader:
                inputs = inputs.to(self.device)
                batch_embeddings = self.model(inputs).cpu().numpy()  # Вычисляем эмбеддинги
                embeddings.append(batch_embeddings)
                labels_list.append(batch_labels.numpy())

        # Объединяем все эмбеддинги и метки
        embeddings = np.vstack(embeddings)
        labels_array = np.hstack(labels_list)

        # Выбор контрастных примеров
        contrastive_pairs = []
        for i in range(len(embeddings)):
            similarities = cosine_similarity([embeddings[i]], embeddings)
            sorted_indices = np.argsort(-similarities[0])

            # Выбираем примеры с разными метками
            for j in sorted_indices:
                if labels_array[j] != labels_array[i]:
                    contrastive_pairs.append((i, j))
                    if len(contrastive_pairs) >= top_k:
                        break
            if len(contrastive_pairs) >= top_k:
                break

        cont_p = [item for tup in contrastive_pairs for item in tup]

        self.update_dataloader(cont_p)

    

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
