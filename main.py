import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models
import numpy as np
from matplotlib import pyplot as plt
from model import SimpleCNN
from base_trainer import Trainer


NUM_CLASSES = 10           # CIFAR-10
NUM_EPOCH = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)



model = SimpleCNN(10)

optimizer = optim.AdamW(model.parameters(), lr=5e-4)
criterion = torch.nn.CrossEntropyLoss()
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print('NUM_CLASSES: ', NUM_CLASSES)
print('NUM_EPOCH: ', NUM_EPOCH)
print('DEVICE: ', DEVICE)

trainer = Trainer(model=model.to(DEVICE), optimizer=optimizer,criterion=criterion, train_loader=train_dataloader, val_loader=val_dataloader, device=DEVICE)

trainer.fit(NUM_EPOCH)
trainer.plot_losses()
