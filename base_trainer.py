from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, Subset
import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader,  device):
        """
        :param model: Обучаемая модель
        :param optimizer: Оптимизатор
        :param criterion: Функция потерь
        :param device: Устройство ('cuda' или 'cpu')
        :train_loader: Загрузчик обучающего датасета
        :val_loader: Загрузчик тестового датасета
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_loader = train_loader
        self.pool_loader = None
        self.val_loader = val_loader
        self.acc = []
        self.f1 = []

    def train_step(self):
        """
        Один шаг обучения.
        :param train_loader: DataLoader для обучающей выборки
        :return: Средняя потеря за эпоху
        """
        self.model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(self.train_loader,desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Обнуление градиентов
            self.optimizer.zero_grad()

            # Прямой проход
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Обратное распространение ошибки
            loss.backward()

            # Обновление параметров модели
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def val_step(self):
        """
        Один шаг валидации.
        Вычисляет среднюю потерю, F1-score и точность.
        :return: Средняя потеря, F1-score и точность за эпоху
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Прямой проход
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Сохранение предсказаний и истинных меток
                predictions = outputs.argmax(dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(targets.cpu().numpy())

        # Рассчитываем метрики
        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average="weighted")
        self.acc.append(accuracy)
        self.f1.append(f1)

        self.val_losses.append(avg_loss)

        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        return avg_loss, accuracy, f1
    
    def select_samples(self):
        """
        Выбираем сэмплы на активное обучение

        """

        pass

    def update_dataloader(self, samples_index):
        """
        Обновляем даталоудеры
        """
        samples = [self.pool_loader.dataset[i] for i in samples_index]


        self.train_loader.dataset = ConcatDataset([self.train_loader.dataset, ConcatDataset(samples)])
        indices = list(set(range(len(self.pool_loader))) - set(samples_index))
        self.pool_loader.dataset = Subset(self.pool_loader.dataset, indices)



    
    def fit(self, num_epochs):
        """
        Полный цикл обучения.
        :param num_epochs: Количество эпох
        """
        for epoch in range(num_epochs):
            train_loss = self.train_step()
            val_loss, accuracy, f1 = self.val_step()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")

        self.plot_losses()

    def plot_losses(self):
        """
        Построение графиков потерь.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.show()