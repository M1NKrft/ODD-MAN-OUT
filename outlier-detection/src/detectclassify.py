import os
import numpy as np
from feature_extraction import extract_features
from single_outlier import detect_outlier_f
import sys
import os
sys.path.append(os.path.abspath("..")) 
from classification import classify_image      

def detect_and_classify(odd_data_folder='static/images', model_path='/home/ansh/flowerz/flower_model.pth'):
    image_paths = [os.path.join(odd_data_folder, img) for img in os.listdir(odd_data_folder) if img.endswith(('jpg', 'png'))]
    features = np.array([extract_features(img_path) for img_path in image_paths])
    outlier_index = detect_outlier_f(features)
    if(outlier_index == 0):
        other_ind = 1
    else:
        other_ind = 0
    odd_image_path = image_paths[outlier_index]
    other_img_path = image_paths[other_ind]
    #print(f"Detected odd image: {odd_image_path}")
    flowername = classify_image(model_path, odd_image_path)
    #print(f"Classified as: }")
    
    return odd_image_path, flowername, other_img_path