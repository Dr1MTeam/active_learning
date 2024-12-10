import os
import sys 
import torch
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.optim as optim
sys.path.append(os.path.abspath('..'))
from base_trainer import Trainer
from model import SimpleCNN
from torch.utils.data import DataLoader, Subset, Dataset


class LeastConf(Trainer):
    def select_samples(self, dataloader, num_samples_to_select):
        """
        Выбираем образцы на основе наименьшей уверенности модели.
        
        :param dataloader: Загрузчик данных для выборки
        :param num_samples_to_select: Количество образцов для выбора
        :return: Список индексов выбранных образцов
        """
        self.model.eval()  # Устанавливаем модель в режим оценки
        all_confidences = []
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)  # Переносим данные на устройство модели
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                confidence_scores, _ = torch.max(probabilities, dim=1)  # Максимальная вероятность для каждого примера
                all_confidences.append(confidence_scores)

        # Конкатенируем все уверенности в один тензор
        confidence_scores = torch.cat(all_confidences)
        
        # Находим индексы образцов с наименьшей уверенностью
        least_confident_indices = torch.argsort(1 - confidence_scores)[:num_samples_to_select]
        self.update_dataloader(least_confident_indices.cpu().numpy())
        print("watch = ", least_confident_indices.cpu().numpy())
        return least_confident_indices.cpu().numpy()
    
    def fit(self, num_epochs):
        """
        Полный цикл обучения.
        :param num_epochs: Количество эпох
        """
        for epoch in range(num_epochs):
            train_loss = self.train_step()
            val_loss, accuracy, f1 = self.val_step()
            self.select_samples(dataloader = self.pool_loader, num_samples_to_select = num_epochs)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")