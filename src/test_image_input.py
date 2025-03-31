import cv2
import numpy as np
from image_input import ImageInput
import os

def test_image_input():
    # Initialize ImageInput
    image_input = ImageInput()
    
    # Test camera capture
    print("Testing camera capture...")
    if image_input.start_camera():
        print("Camera started successfully")
        
        # Capture a few frames
        for i in range(3):
            frame = image_input.capture_frame()
            if frame is not None:
                print(f"Frame {i+1} captured successfully")
                # Save frame for verification
                cv2.imwrite(f"test_frame_{i+1}.jpg", (frame * 255).astype(np.uint8))
            else:
                print(f"Failed to capture frame {i+1}")
        
        image_input.stop_camera()
        print("Camera stopped")
    else:
        print("Failed to start camera")
    
    # Test image loading
    print("\nTesting image loading...")
    # Create a test image if it doesn't exist
    if not os.path.exists("test_image.jpg"):
        test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite("test_image.jpg", test_img)
    
    image = image_input.load_image("test_image.jpg")
    if image is not None:
        print("Image loaded successfully")
        # Save processed image for verification
        cv2.imwrite("test_processed.jpg", (image * 255).astype(np.uint8))
    else:
        print("Failed to load image")

if __name__ == "__main__":
    test_image_input() 