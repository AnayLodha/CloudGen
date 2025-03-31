import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from cloud_detection import UNet
import os
from typing import Tuple
import shutil
import sys
from PIL import Image
from torchvision import transforms

# Cloud type descriptions
CLOUD_DESCRIPTIONS = {
    'Ac': 'Altocumulus - Mid-level, white/gray rounded masses',
    'As': 'Altostratus - Mid-level, gray/bluish cloud sheets',
    'Cb': 'Cumulonimbus - Thunderstorm clouds',
    'Cc': 'Cirrocumulus - High-level, small rippled elements',
    'Ci': 'Cirrus - High-level, thin wispy',
    'Cs': 'Cirrostratus - High-level, transparent sheets',
    'Ct': 'Contrails - Aircraft condensation trails',
    'Cu': 'Cumulus - Low-level, puffy cotton-like',
    'Ns': 'Nimbostratus - Low/mid-level, rain-bearing layers',
    'Sc': 'Stratocumulus - Low-level, gray/white patches',
    'St': 'Stratus - Low-level, uniform gray layer'
}

def load_model(model_path: str, device: torch.device) -> UNet:
    """Load the trained model"""
    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def enhance_image(image):
    """Enhance image for better cloud edge detection"""
    # Convert to float32
    image = image.astype(np.float32) / 255.0
    
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(np.uint8(l * 255)).astype(np.float32) / 255.0
    
    # Edge enhancement
    edges = cv2.Canny(np.uint8(l * 255), 50, 150).astype(np.float32) / 255.0
    l = l + 0.2 * edges
    l = np.clip(l, 0, 1)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced

def preprocess_image(image):
    """Enhanced preprocessing for better cloud structure detection"""
    processed_images = []
    
    # Convert to float and normalize
    img_float = image.astype(np.float32) / 255.0
    
    # Create multiple scales with structure preservation
    scales = [1.0, 0.75, 0.5]
    base_size = (300, 300)  # Fixed base size for consistency
    
    for scale in scales:
        # Resize to base size first, then apply scale
        img_resized = cv2.resize(img_float, base_size)
        if scale != 1.0:
            height = int(base_size[0] * scale)
            width = int(base_size[1] * scale)
            img_scaled = cv2.resize(img_resized, (width, height))
        else:
            img_scaled = img_resized
            
        # Enhance local contrast
        lab = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(np.uint8(l * 255)).astype(np.float32) / 255.0
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Convert to tensor
        tensor = torch.from_numpy(enhanced.transpose(2, 0, 1))
        processed_images.append(tensor)
    
    return processed_images

def predict_ensemble(model, processed_images, device):
    """Get ensemble prediction with enhanced edge detection"""
    model.eval()
    with torch.no_grad():
        predictions = []
        target_size = None
        
        for img in processed_images:
            img = img.unsqueeze(0).to(device)
            pred = model(img)
            pred = pred.squeeze().cpu().numpy()
            
            # Store first prediction size as target size
            if target_size is None:
                target_size = pred.shape
            
            # Resize prediction to match target size
            if pred.shape != target_size:
                pred = cv2.resize(pred, (target_size[1], target_size[0]))
            
            # Apply edge enhancement
            pred = cv2.GaussianBlur(pred, (3, 3), 0)
            edges = cv2.Canny(np.uint8(pred * 255), 50, 150).astype(np.float32) / 255.0
            pred = pred + 0.3 * edges
            pred = np.clip(pred, 0, 1)
            
            predictions.append(pred)
        
        # Stack predictions before averaging to ensure consistent shape
        predictions = np.stack(predictions, axis=0)
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Final edge enhancement
        ensemble_pred = cv2.GaussianBlur(ensemble_pred, (3, 3), 0)
        edges = cv2.Canny(np.uint8(ensemble_pred * 255), 50, 150).astype(np.float32) / 255.0
        final_pred = ensemble_pred + 0.3 * edges
        final_pred = np.clip(final_pred, 0, 1)
        
        return final_pred

def visualize_prediction(image, prediction, save_path):
    """Visualize prediction with cloud tracing"""
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Resize prediction to match original image size
    prediction_resized = cv2.resize(prediction, (image.shape[1], image.shape[0]))
    
    # Show probability map
    prob_map = ax2.imshow(prediction_resized, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Cloud Structure')
    ax2.axis('off')
    plt.colorbar(prob_map, ax=ax2)
    
    # Create binary mask with adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        np.uint8(prediction_resized * 255),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        -2
    )
    
    # Show thresholded result
    ax3.imshow(thresh, cmap='gray')
    ax3.set_title('Cloud Boundaries')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def test_model(model, image_path, save_path, device):
    """Test model with enhanced processing pipeline"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess image at multiple scales
    processed_images = preprocess_image(image)
    
    # Get ensemble prediction
    prediction = predict_ensemble(model, processed_images, device)
    
    # Print statistics
    print(f"Prediction stats - Min: {prediction.min():.4f}, Max: {prediction.max():.4f}, "
          f"Mean: {prediction.mean():.4f}, Std: {prediction.std():.4f}")
    
    # Visualize prediction
    visualize_prediction(image, prediction, save_path)

def process_frame(frame, model, device):
    """Enhanced cloud detection"""
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Enhance image
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    
    # Resize and prepare for model
    frame_resized = cv2.resize(enhanced, (296, 296))
    input_tensor = torch.from_numpy(frame_resized.transpose(2, 0, 1)).float() / 255.0
    input_tensor = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(input_tensor)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.squeeze().cpu().numpy()
    
    # Post-process prediction
    pred_resized = cv2.resize(prediction, (frame.shape[1], frame.shape[0]))
    
    # Multi-threshold detection
    thin_clouds = (pred_resized > 0.3)
    thick_clouds = (pred_resized > 0.5)
    
    # Create visualization
    output = frame.copy()
    
    # Add thin cloud overlay (yellow)
    output[thin_clouds] = cv2.addWeighted(
        output[thin_clouds], 0.7,
        np.full_like(output[thin_clouds], [0, 255, 255]), 0.3, 0
    )
    
    # Add thick cloud overlay (white)
    output[thick_clouds] = cv2.addWeighted(
        output[thick_clouds], 0.7,
        np.full_like(output[thick_clouds], [255, 255, 255]), 0.3, 0
    )
    
    # Add contours
    edges = cv2.Canny(np.uint8(pred_resized * 255), 30, 100)
    kernel = np.ones((2,2), np.uint8)
    edges = cv2.dilate(edges, kernel)
    output[edges > 0] = [0, 0, 255]  # Red boundaries
    
    return output

def main():
    # Load model
    device = torch.device('cpu')
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load('models/cloud_detector.pth', map_location=device))
    model.eval()
    
    print("Starting cloud detection...")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    
    # Open camera
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Could not open camera. Using image mode instead.")
        # Try processing a single image
        image_path = input("Enter path to image file: ")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not load image: {image_path}")
            return
            
        output = process_frame(frame, model, device)
        cv2.imwrite('cloud_detection_result.png', output)
        print("Result saved as cloud_detection_result.png")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Process frame
        output = process_frame(frame, model, device)
        
        # Show result
        cv2.imshow('Cloud Detection', output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('cloud_detection_capture.png', output)
            print("Saved current frame")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 