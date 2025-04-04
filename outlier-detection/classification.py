import os
import tarfile
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils.mapping import class_mapping
import scipy.io
import shutil
def download_oxford_102(data_dir="/home/ansh/flowerz/data"):
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    label_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    
    os.makedirs(data_dir, exist_ok=True)
    image_path = os.path.join(data_dir, "102flowers.tgz")
    label_path = os.path.join(data_dir, "imagelabels.mat")

    if not os.path.exists(image_path):
        print("Downloading Oxford 102 Flower Dataset...")
        urllib.request.urlretrieve(url, image_path)
    
    if not os.path.exists(label_path):
        print("Downloading image labels...")
        urllib.request.urlretrieve(label_url, label_path)
    
    if not os.path.exists(os.path.join(data_dir, "jpg")):
        print("Extracting images...")
        with tarfile.open(image_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    print("Dataset ready in:", data_dir)

def load_data(data_path, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

class FlowerClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FlowerClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

def organize_images(data_dir="/home/ansh/flowerz/data"):
    label_path = os.path.join(data_dir, "imagelabels.mat")
    image_dir = os.path.join(data_dir, "jpg")
    output_dir = os.path.join(data_dir, "flowers")

    os.makedirs(output_dir, exist_ok=True)
    labels = scipy.io.loadmat(label_path)['labels'][0]

    for idx, label in enumerate(labels, start=1):
        class_folder = os.path.join(output_dir, f"class_{label}")
        os.makedirs(class_folder, exist_ok=True)

        img_name = f"image_{idx:05d}.jpg"
        src_path = os.path.join(image_dir, img_name)
        dst_path = os.path.join(class_folder, img_name)

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

    print("Images organized into class folders!")

def classify_image(model_path, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, _ = load_data("/home/ansh/flowerz/data/flowers")
    num_classes = len(dataset.classes)
    
    model = FlowerClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        predicted_class = torch.argmax(output).item()
    predicted_folder = f"{dataset.classes[predicted_class]}"
    print(f"Predicted Index: {predicted_class}, Folder: {predicted_folder}")
    print(f"Flower Name: {class_mapping.get(predicted_folder, 'Unknown Flower')}")
    flowername = class_mapping.get(predicted_folder, 'Unknown Flower')
    return flowername


def train_model(data_path, num_epochs=10, lr=0.001, save_path="/home/ansh/flowerz/flower_model.pth"):
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
if __name__ == "__main__":
    download_oxford_102("/home/ansh/flowerz/data")
    organize_images()
    classify_image("/home/ansh/flowerz/flower_model.pth", "/home/ansh/flowerz/data/flowers/class_1/image_06734.jpg")