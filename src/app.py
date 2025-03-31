from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import torch
from cloud_detection import UNet, detect_clouds
import numpy as np
import random
import base64
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load model
device = torch.device('cpu')
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('models/cloud_detector.pth', map_location=device))
model.eval()

# Personal positive affirmations
personal_affirmations = [
    "You are as boundless as the sky above",
    "Your potential is limitless, reaching new heights every day",
    "Like clouds, you have the power to transform and grow",
    "You bring light and inspiration wherever you go",
    "Your spirit is free and your dreams are taking flight",
    "You have the strength to weather any storm",
    "Each day brings new opportunities for you to shine",
    "Your creativity flows as freely as the clouds above"
]

def process_image(image_path):
    """Process the image to detect clouds and enhance visualization"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
            
        # Convert to RGB for better processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image if too large
        max_dimension = 1024
        height, width = img_rgb.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img_rgb = cv2.resize(img_rgb, (int(width * scale), int(height * scale)))
        
        # Convert to float32 for processing
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Get cloud prediction
        cloud_mask = detect_clouds(img_float)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_OPEN, kernel)
        cloud_mask = cv2.morphologyEx(cloud_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create blurred background
        background = cv2.GaussianBlur(img_rgb, (31, 31), 0)
        
        # Combine original image with blurred background based on cloud mask
        cloud_mask_3d = np.stack([cloud_mask] * 3, axis=-1)
        result = np.where(cloud_mask_3d > 0.1, img_rgb, background)
        
        # Add nebula-pink boundaries around clouds
        edges = cv2.Canny(cloud_mask.astype(np.uint8) * 255, 100, 200)
        result[edges > 0] = [255, 20, 147]  # Nebula pink color
        
        # Convert back to BGR for saving
        result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Save the processed image
        output_path = image_path.replace('.jpg', '_processed.jpg')
        cv2.imwrite(output_path, result_bgr)
        
        return output_path
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle uploaded image and return processed result"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        processed_path = process_image(filepath)
        
        # Read the processed image and convert to base64
        with open(processed_path, 'rb') as img_file:
            processed_image = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up temporary files
        os.remove(filepath)
        os.remove(processed_path)
        
        affirmation = random.choice(personal_affirmations)
        return jsonify({
            'image': processed_image,
            'affirmation': affirmation
        })
        
    except Exception as e:
        print(f"Error in upload route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/manifest.webmanifest')
def serve_manifest():
    return send_file('static/manifest.webmanifest', mimetype='application/manifest+json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 