import os
import numpy as np
from feature_extraction import extract_features
from single_outlier import detect_outlier 
import sys
import os
import shutil
import scipy.io
sys.path.append(os.path.abspath("..")) 
from classification import classify_image      

def detect_and_classify(odd_data_folder, model_path):
    image_paths = [os.path.join(odd_data_folder, img) for img in os.listdir(odd_data_folder) if img.endswith(('jpg', 'png'))]
    features = np.array([extract_features(img_path) for img_path in image_paths])
    outlier_index = detect_outlier(features)
    odd_image_path = image_paths[outlier_index]
    
    print(f"Detected odd image: {odd_image_path}")
    predicted_class = classify_image(model_path, odd_image_path)
    print(f"Classified as: {predicted_class}")
    
    return odd_image_path, predicted_class

if __name__ == "__main__":
    odd_data_folder = "../input_data/odd-data"
    model_path = "flower_model.pth"
    detect_and_classify(odd_data_folder, model_path)