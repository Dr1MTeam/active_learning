import os
import sys
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from tqdm import tqdm

sys.path.append(os.path.abspath('..'))
from base_trainer import Trainer


BATCH_ADD_SIZE = 256

class EGLTrainer(Trainer):
    def select_samples(self, num_samples):
        # total_samples = len(self.pool_loader.dataset)
        # r = random.sample(range(total_samples), num_samples)
        # self.update_dataloader(r)
        # return r
        self.model.eval()
        self.pool_loader


        gradient_lengths = []
        ###
        ### Этот комментарий в память о тройном цикле for от чата gpt >_<
        ###
        for inputs, _ in tqdm(self.pool_loader, desc="EGL"):
            inputs = inputs.to(self.device)
            


            # Получаем выходы модели
            outputs = self.model(inputs)  # Tensor [batch_size, num_classes]
            probabilities = torch.softmax(outputs, dim=1)  # Tensor [batch_size, num_classes]

            # Создаем вектор one-hot меток для всех классов
            one_hot_labels = torch.eye(outputs.size(1), device=self.device)  # Tensor [num_classes, num_classes]

            # Расширяем one-hot метки для батча
            expanded_labels = one_hot_labels.unsqueeze(0).repeat(outputs.size(0), 1, 1)  # [batch_size, num_classes, num_classes]

            # Расширяем выходы модели для вычитания
            expanded_outputs = outputs.unsqueeze(1).repeat(1, outputs.size(1), 1)  # [batch_size, num_classes, num_classes]

            # Вычисляем потери для всех пар (объект, метка) в батче
            losses = self.criterion(expanded_outputs.view(-1, outputs.size(1)), expanded_labels.view(-1, outputs.size(1)))  # [batch_size * num_classes]

            # Считаем градиенты
            grads = torch.autograd.grad(
                losses.sum(),
                self.model.parameters(),
                retain_graph=True,
                create_graph=False
            )

            # Вычисляем длины градиентов
            grad_square_norms = torch.norm(torch.cat([g.view(-1) for g in grads])) # torch.cat([g.flatten(1).pow(2).sum(1) for g in grads]).sum(dim=0)  # [batch_size, num_classes]

            # Мат ожидание
            batch_gradient_expectations = (probabilities * grad_square_norms).sum(dim=1)  # [batch_size]

            gradient_lengths.extend(batch_gradient_expectations.cpu().detach().numpy())

        gradient_lengths = torch.tensor(gradient_lengths)

        # Выбираем `num_samples` с максимальной длиной градиента
        informative_indices = torch.topk(gradient_lengths, num_samples).indices
        
        self.update_dataloader(informative_indices)
        return informative_indices
    
    def update_dataloader(self, samples_index):
        """
        Обновляем даталоудеры
        """
        samples = [self.pool_loader.dataset[i] for i in samples_index]


        new_train_data = ConcatDataset([self.train_loader.dataset, samples])
        
        self.train_loader = DataLoader(new_train_data,
                                       batch_size=self.train_loader.batch_size,
                                       shuffle=True,
                                       num_workers=self.train_loader.num_workers)
        

        all_indexes = list(range(len(self.pool_loader.dataset)))
        
        filtered_indexes = [idx for idx in all_indexes if idx not in samples_index]

        new_pool_data = Subset(self.pool_loader.dataset, filtered_indexes)

        self.pool_loader = DataLoader(new_pool_data,
                                      batch_size=self.pool_loader.batch_size,
                                      shuffle=True,
                                      num_workers=self.pool_loader.num_workers)
    
    def fit(self, num_epochs, num_samples = BATCH_ADD_SIZE):

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


