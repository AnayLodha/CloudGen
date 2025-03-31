import os
import requests
from PIL import Image
from io import BytesIO

def download_image(url: str, save_path: str) -> bool:
    """Download an image from URL and save it"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Open the image to verify it's valid
        img = Image.open(BytesIO(response.content))
        
        # Save the image
        img.save(save_path)
        print(f"Image downloaded successfully to {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        return False

def main():
    # Create test_images directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # List of cloud image URLs to try
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Cumulus_clouds_in_fair_weather.jpeg/1200px-Cumulus_clouds_in_fair_weather.jpeg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Altocumulus_clouds.jpg/1200px-Altocumulus_clouds.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Cirrus_clouds_mar13.jpg/1200px-Cirrus_clouds_mar13.jpg"
    ]
    
    # Try downloading each image until one succeeds
    for i, url in enumerate(image_urls):
        save_path = os.path.join("test_images", f"test_cloud_{i+1}.jpg")
        if download_image(url, save_path):
            break

if __name__ == "__main__":
    main() 