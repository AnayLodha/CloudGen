import cv2
import numpy as np
import os
from typing import Tuple, Optional

class CloudAnnotator:
    def __init__(self):
        self.drawing = False
        self.mask = None
        self.image = None
        self.window_name = "Cloud Annotation Tool"
        
    def start_drawing(self, event: int, x: int, y: int, flags: int, param: any):
        """Mouse callback function for drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.mask, (x, y), 2, (255), -1)
                # Show the mask overlay on the image
                overlay = self.image.copy()
                overlay[self.mask == 255] = [0, 255, 0]  # Green overlay for marked areas
                cv2.addWeighted(overlay, 0.3, self.image, 0.7, 0, self.image)
                cv2.imshow(self.window_name, self.image)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def create_mask(self, image_path: str, output_dir: str) -> bool:
        """
        Create a binary mask for cloud regions
        
        Args:
            image_path (str): Path to the input image
            output_dir (str): Directory to save the mask
            
        Returns:
            bool: True if mask creation was successful
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            print(f"Failed to load image: {image_path}")
            return False
        
        # Create empty mask
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.start_drawing)
        
        print("\nInstructions:")
        print("1. Left click and drag to mark cloud regions")
        print("2. Press 'c' to clear the mask")
        print("3. Press 's' to save the mask")
        print("4. Press 'q' to quit without saving")
        
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save the mask
                mask_path = os.path.join(output_dir, 
                                       os.path.splitext(os.path.basename(image_path))[0] + "_mask.png")
                cv2.imwrite(mask_path, self.mask)
                print(f"Mask saved to: {mask_path}")
                break
            elif key == ord('c'):
                # Clear the mask
                self.mask = np.zeros_like(self.mask)
                self.image = cv2.imread(image_path)
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return True

def main():
    # Create dataset directories
    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/masks", exist_ok=True)
    
    # Copy the cloud image to dataset
    image_path = "istockphoto-466347707-612x612.jpg"
    if os.path.exists(image_path):
        os.system(f'copy "{image_path}" "dataset/images/"')
        
        # Create annotator and start mask creation
        annotator = CloudAnnotator()
        annotator.create_mask("dataset/images/" + os.path.basename(image_path), "dataset/masks")
    else:
        print(f"Image not found: {image_path}")

if __name__ == "__main__":
    main() 