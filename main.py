import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models
import numpy as np
from matplotlib import pyplot as plt
from model import SimpleCNN
from base_trainer import Trainer
from ContrAL.ContrAL import contrastive_AL
from omegaconf import OmegaConf  # Import OmegaConf

# Load configuration from YAML file
config = OmegaConf.load('config.yaml')  # Load config from config.yaml

NUM_CLASSES = config['NUM_CLASSES']           # CIFAR-10
NUM_EPOCH = config['NUM_EPOCH']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225])])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


model = SimpleCNN(NUM_CLASSES)
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
criterion = torch.nn.CrossEntropyLoss()

if config['split_label_unlabel']:
    initial_indices = np.random.choice(len(train_dataset), size=int(len(train_dataset)*config['split_label_unlabel']), replace=False) # 20%
    initial_data = Subset(train_dataset, initial_indices)

    unlabeled_indices = list(set(range(len(train_dataset))) - set(initial_indices))
    unlabeled_data = Subset(train_dataset, unlabeled_indices)

    train_dataloader = DataLoader(initial_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
    pool_dataloader = DataLoader(unlabeled_data, batch_size=config['batch_size'], shuffle=True)
    #print(f" Размер исходного датасета {len(train_dataset)}")
    
    print(f" Размер обучающего датасета {len(initial_data)}")
    print(f" Размер AL датасета {len(unlabeled_data)}")
    print(f" Размер тестового датасета {len(test_dataset)}")


match config['AL_Method']:
    case 'wo_AL':

        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
        pool_dataloader = None
        

    case 'ContrAL':
        Trainer = contrastive_AL


print(f" Используем метод { config['AL_Method'] }")




trainer = Trainer(model=model.to(DEVICE),
                optimizer=optimizer,
                criterion=criterion, 
                pool_loader=pool_dataloader, 
                train_loader=train_dataloader, 
                val_loader=val_dataloader, 
                device=DEVICE)



print('NUM_CLASSES: ', NUM_CLASSES)
print('NUM_EPOCH: ', NUM_EPOCH)
print('DEVICE: ', DEVICE)


trainer.fit(NUM_EPOCH)
trainer.plot_losses()
