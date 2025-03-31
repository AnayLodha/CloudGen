import cv2
import numpy as np
from cloud_detection import CloudDetector
from image_input import ImageInput
import os

def test_cloud_detection():
    # Initialize image input and cloud detector
    image_input = ImageInput()
    cloud_detector = CloudDetector()
    
    # Test with camera capture
    print("Testing cloud detection with camera capture...")
    if image_input.start_camera():
        print("Camera started successfully")
        
        # Capture and process a frame
        frame = image_input.capture_frame()
        if frame is not None:
            print("Frame captured successfully")
            
            # Process the frame
            image, mask, edges = cloud_detector.process_image(frame)
            
            # Save results
            cv2.imwrite("test_cloud_detection.jpg", (image * 255).astype(np.uint8))
            cv2.imwrite("test_cloud_mask.jpg", (mask * 255).astype(np.uint8))
            cv2.imwrite("test_cloud_edges.jpg", edges)
            
            print("Results saved successfully")
        else:
            print("Failed to capture frame")
        
        image_input.stop_camera()
        print("Camera stopped")
    else:
        print("Failed to start camera")
    
    # Test with image file
    print("\nTesting cloud detection with image file...")
    if os.path.exists("test_image.jpg"):
        image = image_input.load_image("test_image.jpg")
        if image is not None:
            print("Image loaded successfully")
            
            # Process the image
            image, mask, edges = cloud_detector.process_image(image)
            
            # Save results
            cv2.imwrite("test_file_cloud_detection.jpg", (image * 255).astype(np.uint8))
            cv2.imwrite("test_file_cloud_mask.jpg", (mask * 255).astype(np.uint8))
            cv2.imwrite("test_file_cloud_edges.jpg", edges)
            
            print("Results saved successfully")
        else:
            print("Failed to load image")
    else:
        print("Test image not found")

if __name__ == "__main__":
    test_cloud_detection() 