import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, SubsetRandomSampler

import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.models import resnet50

os.getcwd()

train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) ])

val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "dataset"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), val_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

train_dataset_len = len(train_dataset)
val_dataset_len = len(val_dataset)

class_names = train_dataset.classes

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    #plt.show()
    if title is not None:
        plt.title(title)

if __name__ == '__main__':
    inputs, classes = next(iter(train_dataloader))
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

model = resnet50(pretrained=True)
model.fc = nn.Linear(2048, 2)
criterion = nn.CrossEntropyLoss()
model.cuda()
optimizer = optim.AdamW(params=model.parameters())

train_losses = []
val_losses = []
num_epochs = 5
for epoch in range(0, num_epochs):
    epoch_train_losses = []
    epoch_val_losses = []

    model.train()
    if __name__ == '__main__':
        for images, labels in train_dataloader:
            images = images.cuda()
            labels = labels.cuda()

            model.zero_grad()
            pred = model(images)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            epoch_train_losses.append(loss.item())

        model.eval()
        for images, labels in val_dataloader:
            images = images.cuda()
            labels = labels.cuda()

            pred = model(images)
            loss = criterion(pred, labels)
            val_losses.append(loss.item())
            epoch_val_losses.append(loss.item())

        print("Epoch [{}/{}], Train Loss: {}, Valid Loss: {}".format(epoch + 1, num_epochs,
                                                                 sum(epoch_train_losses) / len(epoch_train_losses),
                                                                 sum(epoch_val_losses) / len(epoch_val_losses)))


model.eval()
result = 0

if __name__ == '__main__':
    for images, labels in val_dataloader:
        images = images.cuda()
        labels = labels.cuda()
        pred1 = model(images)
        prediction = torch.argmax(pred1,1)
        res = torch.eq(prediction.cpu(),labels.cpu()).sum()
        result+=res.item()
    print('Result: {}%'.format(result/len(val_dataset)*100))

    print(os.getcwd())
    torch.save(model, "D:/pyprojects/telegram_bot/required1")
    print("success")
