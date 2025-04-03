import os
import tarfile
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from classification import load_data, FlowerClassifier
def train_model(data_path, num_epochs=10, lr=0.001, save_path="~/flowerz/flower_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, dataloader = load_data(data_path)
    num_classes = len(dataset.classes)

    model = FlowerClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

if __name__=='__main__':
    train_model()