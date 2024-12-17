import os
import sys
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.append(os.path.abspath('..'))
from base_trainer import Trainer

from tqdm import tqdm

BATCH_ADD_SIZE = 500


class BALDTrainer(Trainer):

    def enable_dropout(self):
        """
        Включает Dropout в модели даже в режиме eval.
        """
        for module in self.model.modules():
            if module.__class__.__name__.startswith('Dropout'):
                module.train()

    def select_samples(self, num_samples):
        """
        Выбирает наиболее информативные примеры из пула данных, используя метод BALD.

        :param num_samples: Количество отбираемых примеров.
        :return: Список индексов информативных примеров.
        """
        self.model.eval()  # Переключаем модель в режим оценки
        self.enable_dropout() # Включаем Dropout для Monte Carlo Dropout
        all_entropies = []  # Список для хранения энтропий каждого примера
        all_indices = []  # Список для хранения индексов примеров

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.pool_loader, desc='BALD')):
                inputs, _ = batch  # Предполагается, что loader возвращает данные и их индексы
                inputs = inputs.to(self.device)

                batch_size = inputs.size(0)
                # Глобальные индексы батча
                batch_indices = torch.arange(i * batch_size, (i + 1) * batch_size)
                all_indices.append(batch_indices)

                # Monte Carlo Dropout: несколько проходов через модель
                mc_outputs = torch.stack([
                    torch.softmax(self.model(inputs), dim=-1) for _ in range(10)  # 10 итераций MC Dropout
                ])  # (mc_samples, batch_size, num_classes)

                # Среднее предсказание по всем MC итерациям
                mean_probs = mc_outputs.mean(dim=0)  # (batch_size, num_classes)

                # Общая энтропия
                total_entropy = -torch.sum(mean_probs * mean_probs.log(), dim=-1)  # (batch_size)

                # Условная энтропия
                conditional_entropy = -torch.mean(
                    torch.sum(mc_outputs * mc_outputs.log(), dim=-1), dim=0  # (batch_size)
                )

                # Взаимная информация (BALD score)
                bald_scores = total_entropy - conditional_entropy  # (batch_size)

                # Сохраняем результаты
                all_entropies.append(bald_scores.cpu())

        # Объединяем результаты для всех батчей
        all_entropies = torch.cat(all_entropies)  # Все BALD оценки
        all_indices = torch.cat(all_indices)  # Все индексы

        # Отбираем топ-K индексов с наибольшими BALD оценками
        _, top_indices = torch.topk(all_entropies, num_samples)
        informative_indices = all_indices[top_indices].tolist()

        # Обновляем dataloader для следующих итераций
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

    def _fit(self, num_epochs, num_samples = BATCH_ADD_SIZE):

        """
        Полный цикл обучения.
        :param num_epochs: Количество эпох
        """
        for epoch in range(num_epochs):

            train_loss = self.train_step()
            val_loss, accuracy, f1 = self.val_step()
            if len(self.pool_loader.dataset) > num_samples:
                self.select_samples(num_samples)
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")