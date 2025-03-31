import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

def create_cloud_mask(image: np.ndarray) -> np.ndarray:
    """Create a cloud mask using color and brightness information"""
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract channels
    h, s, v = cv2.split(hsv)
    l, a, b = cv2.split(lab)
    
    # Create masks based on different criteria
    # 1. High brightness in value channel
    value_mask = cv2.threshold(v, 200, 255, cv2.THRESH_BINARY)[1]
    
    # 2. Low saturation (whiteness)
    saturation_mask = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY_INV)[1]
    
    # 3. High brightness in L channel
    lightness_mask = cv2.threshold(l, 200, 255, cv2.THRESH_BINARY)[1]
    
    # 4. Neutral a and b channels (close to white/gray)
    a_mask = cv2.inRange(a, 120, 135)  # a channel is centered at 128
    b_mask = cv2.inRange(b, 120, 135)  # b channel is centered at 128
    
    # Combine masks
    combined_mask = cv2.bitwise_and(value_mask, saturation_mask)
    combined_mask = cv2.bitwise_and(combined_mask, lightness_mask)
    neutral_mask = cv2.bitwise_and(a_mask, b_mask)
    final_mask = cv2.bitwise_or(combined_mask, neutral_mask)
    
    # Clean up mask
    kernel = np.ones((5,5), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    return final_mask

def download_dataset():
    """Download the cloud dataset from Kaggle"""
    print("Downloading cloud dataset from Kaggle...")
    
    try:
        # Create dataset directories
        os.makedirs("dataset/images", exist_ok=True)
        os.makedirs("dataset/masks", exist_ok=True)
        
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        dataset_name = "nakendraprasathk/cloud-image-classification-dataset"
        api.dataset_download_files(dataset_name, path="dataset/download", unzip=True)
        
        print("\nProcessing downloaded images...")
        download_dir = "dataset/download"
        for filename in tqdm(os.listdir(download_dir)):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Read image
                image_path = os.path.join(download_dir, filename)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Save to our dataset directory
                save_path = os.path.join("dataset/images", filename)
                cv2.imwrite(save_path, image)
                
                # Create cloud mask
                mask = create_cloud_mask(image)
                
                # Save mask
                mask_filename = os.path.splitext(filename)[0] + "_mask.png"
                mask_path = os.path.join("dataset/masks", mask_filename)
                cv2.imwrite(mask_path, mask)
        
        print("\nDataset preparation completed!")
        print(f"Images saved in: {os.path.abspath('dataset/images')}")
        print(f"Masks saved in: {os.path.abspath('dataset/masks')}")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nTo use Kaggle datasets, make sure to:")
        print("1. Install kaggle: pip install kaggle")
        print("2. Set up your Kaggle API credentials in ~/.kaggle/kaggle.json")

def main():
    # Download and prepare dataset
    download_dataset()
    
    # If we have our own cloud image, add it to the dataset
    own_image = "istockphoto-466347707-612x612.jpg"
    if os.path.exists(own_image):
        print(f"\nAdding our own cloud image ({own_image}) to the dataset...")
        shutil.copy(own_image, "dataset/images/")
        
        # Create mask for our image (you'll need to create this manually)
        print("\nNote: You'll need to create a mask for your own image using:")
        print("python src/create_cloud_mask.py")

if __name__ == "__main__":
    main() 