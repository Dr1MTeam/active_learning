from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, Subset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class Trainer:
    def __init__(self, model, optimizer,pool_loader, criterion, train_loader, val_loader,  device, scheduler = None):
        """
        :param model: Обучаемая модель
        :param optimizer: Оптимизатор
        :param scheduler: Оптимизатор
        :param criterion: Функция потерь
        :param device: Устройство ('cuda' или 'cpu')
        :param train_loader: Загрузчик обучающего датасета
        :param val_loader: Загрузчик тестового датасета
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        
        self.train_loader = train_loader
        self.pool_loader = pool_loader
        self.val_loader = val_loader

        self.train_losses = []
        self.val_losses = []

        self.train_acc = []
        self.val_acc = []

        self.train_f1 = []
        self.val_f1 = []

    def train_step(self):
        """
        Один шаг обучения.
        :param train_loader: DataLoader для обучающей выборки
        :return: Средняя потеря за эпоху
        """
        self.model.train()
        running_loss = 0.0

        all_targets = []
        all_predictions = []

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

            # Сохранение предсказаний и истинных меток
            predictions = outputs.argmax(dim=1).cpu().numpy()
                
            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)

        accuracy = accuracy_score(all_targets, all_predictions)
        self.train_acc.append(accuracy)

        f1 = f1_score(all_targets, all_predictions, average="weighted")
        self.train_f1.append(f1)

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
        self.val_losses.append(avg_loss)

        accuracy = accuracy_score(all_targets, all_predictions)
        self.val_acc.append(accuracy)

        f1 = f1_score(all_targets, all_predictions, average="weighted")
        self.val_f1.append(f1)

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


 


        new_train_data = ConcatDataset([self.train_loader.dataset, ConcatDataset(samples)])
        
        
        #####
        # Проблема -
        # Неравномерный батч, мб добавить паддинги???
       
       # padded_data = pad_sequence(new_data, batch_first=True)

        self.train_loader = DataLoader(new_train_data,
                                       batch_size=self.train_loader.batch_size,
                                       shuffle=True,
                                       num_workers=self.train_loader.num_workers)
        
        pool_indices = list(set(range(len(self.pool_loader))) - set(samples_index))

        new_pool_data = Subset(self.pool_loader.dataset, pool_indices)


        self.pool_loader = DataLoader(new_pool_data,
                                      batch_size=self.pool_loader.batch_size,
                                      shuffle=True,
                                      num_workers=self.pool_loader.num_workers)



    
    def fit(self, num_epochs):
        """
        Полный цикл обучения.
        :param num_epochs: Количество эпох
        """
        for epoch in range(num_epochs):
            train_loss = self.train_step()
            val_loss, accuracy, f1 = self.val_step()

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}")

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

    def plot_acc(self):
        """
        Построение графиков точности.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_acc, label="Train Accuracy")
        plt.plot(self.val_acc, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Acc")
        plt.title("Acc Curves")
        plt.legend()
        plt.show()

    def plot_f1(self):
        """
        Построение графиков f1.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_f1, label="Train F1")
        plt.plot(self.val_f1, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("F1 Curves")
        plt.legend()
        plt.show()