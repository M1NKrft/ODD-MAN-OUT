import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess an image for feature extraction.
    
    Steps:
    1. Read image
    2. Convert to RGB
    3. Resize to standard size
    4. Denoise with Gaussian Blur
    5. Apply Adaptive Histogram Equalization (CLAHE)
    6. Convert to binary (optional)
    
    Args:
    - image_path: Path to the input image
    - target_size: Tuple (width, height) for resizing
    
    Returns:
    - Processed image (ready for feature extraction)
    """
    # 1. Read Image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Error: Cannot load image at {image_path}")
    
    # 2. Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to standard size
    image = cv2.resize(image, target_size)
    
    # 4. Denoising using Gaussian Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 5. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 6. Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    # 7. Convert to Binary Image (Thresholding)
    _, binary = cv2.threshold(enhanced_gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return image, enhanced_gray, binary

# Example Usage
image_path = "/home/ansh/Downloads/5.jpg"
processed_image, gray_image, binary_image = preprocess_image(image_path)

# Show Images
#cv2.imshow("Processed RGB Image", processed_image)
#cv2.imshow("Enhanced Grayscale Image", gray_image)
#cv2.imshow("Binary Image", binary_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
