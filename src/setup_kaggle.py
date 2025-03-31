import os
import shutil
import json
from pathlib import Path

def setup_kaggle_credentials():
    """Set up Kaggle credentials from kaggle.json file"""
    # Get user's home directory
    home = str(Path.home())
    kaggle_dir = os.path.join(home, '.kaggle')
    
    # Create .kaggle directory if it doesn't exist
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Look for kaggle.json in current directory first
    current_dir_json = 'kaggle.json'
    downloads = os.path.join(home, 'Downloads')
    downloads_json = os.path.join(downloads, 'kaggle.json')
    
    # Try current directory first, then Downloads
    if os.path.exists(current_dir_json):
        kaggle_json = current_dir_json
    elif os.path.exists(downloads_json):
        kaggle_json = downloads_json
    else:
        kaggle_json = None
    
    if kaggle_json:
        # Copy the file to .kaggle directory
        target_path = os.path.join(kaggle_dir, 'kaggle.json')
        shutil.copy(kaggle_json, target_path)
        
        # Set proper permissions
        try:
            os.chmod(target_path, 0o600)
        except Exception as e:
            print(f"Note: Couldn't set file permissions, but this might be OK on Windows: {e}")
        
        print(f"Successfully copied kaggle.json to {kaggle_dir}")
        print("Credentials are now set up!")
        
        # Verify the JSON file
        try:
            with open(target_path, 'r') as f:
                credentials = json.load(f)
            if 'username' in credentials and 'key' in credentials:
                print("Credentials file format is valid!")
                print(f"Username: {credentials['username']}")
                print("API key: ****" + credentials['key'][-4:])
            else:
                print("Warning: Credentials file may not have the correct format")
        except json.JSONDecodeError:
            print("Warning: Could not verify credentials file format")
    else:
        print("\nCouldn't find kaggle.json in current directory or Downloads folder.")
        print("\nTo get your kaggle.json file:")
        print("1. Go to www.kaggle.com and sign in")
        print("2. Click on your profile picture → Account")
        print("3. Scroll to API section")
        print("4. Click 'Create New API Token'")
        print("5. Move the downloaded kaggle.json to this directory")
        print("6. Run this script again")

if __name__ == "__main__":
    setup_kaggle_credentials() 