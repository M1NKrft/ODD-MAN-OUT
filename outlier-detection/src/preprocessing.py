import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):

    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Error: Cannot load image at {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, target_size)

    image = cv2.GaussianBlur(image, (5, 5), 0)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    _, binary = cv2.threshold(enhanced_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return image, enhanced_gray, binary

# Example Usage
#image_path = "/home/ansh/Downloads/5.jpg"
#processed_image, gray_image, binary_image = preprocess_image(image_path)

# Show Images
#cv2.imshow("Processed RGB Image", processed_image)
#cv2.imshow("Enhanced Grayscale Image", gray_image)
#cv2.imshow("Binary Image", binary_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
