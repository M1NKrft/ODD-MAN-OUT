import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

# Load the pretrained ResNet-50 model
model = models.resnet50(pretrained=True)

# Remove the final classification layer (fc) to get features
model = torch.nn.Sequential(*list(model.children())[:-1])  # Removes the last layer
model.eval()  # Set to evaluation mode

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize as per ImageNet
])

def extract_features(image_path):
    """Extracts 2048-D feature vector from an image using ResNet-50."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():  # No gradient computation
        features = model(image)

    return features.flatten().numpy()  # Convert to 1D array

def extract_features_from_directory(image_dir):
    """Extracts features from all images in a directory."""
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(".jpg")]
    features = np.array([extract_features(img) for img in image_paths])
    return features, image_paths

