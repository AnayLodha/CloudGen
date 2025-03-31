import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import os

class ImageInput:
    def __init__(self, target_size: Tuple[int, int] = (640, 480)):
        """
        Initialize the ImageInput class
        
        Args:
            target_size (Tuple[int, int]): Target size for processed images (width, height)
        """
        self.target_size = target_size
        self.camera = None
        
    def start_camera(self, camera_id: int = 0) -> bool:
        """
        Start the camera for real-time capture
        
        Args:
            camera_id (int): Camera device ID
            
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            return False
        return True
    
    def stop_camera(self):
        """Stop the camera if it's running"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera
        
        Returns:
            Optional[np.ndarray]: Captured frame or None if capture failed
        """
        if self.camera is None:
            return None
            
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        return self.preprocess_image(frame)
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess an image from file
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[np.ndarray]: Processed image or None if loading failed
        """
        if not os.path.exists(image_path):
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            return self.preprocess_image(image)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def enhance_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance colors using histogram equalization and color correction
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Enhance saturation
        hsv[:, :, 1] *= 1.2  # Increase saturation by 20%
        
        # Enhance value (brightness)
        hsv[:, :, 2] *= 1.1  # Increase brightness by 10%
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Apply contrast enhancement
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def correct_color_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Correct color balance using multiple techniques
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Color corrected image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Adjust a and b channels to reduce color casts
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        
        # Reduce blue-yellow tint
        b = b - np.mean(b) * 0.5
        
        # Reduce green-red tint
        a = a - np.mean(a) * 0.3
        
        # Clip values to valid range
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        
        # Convert back to RGB
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Apply color correction
        image = self.correct_color_balance(image)
        
        # Enhance colors
        image = self.enhance_colors(image)
        
        return image
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate the input image
        
        Args:
            image (np.ndarray): Input image to validate
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        if image is None:
            return False
            
        if not isinstance(image, np.ndarray):
            return False
            
        if image.size == 0:
            return False
            
        if image.shape[:2] != self.target_size[::-1]:
            return False
            
        return True 