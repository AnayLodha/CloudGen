import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import os
from cloud_detection import UNet

class CloudDataset(Dataset):
    def __init__(self, data_dir, image_size=(296, 296)):
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Get image files
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Found {len(self.image_files)} images for training")
        
        # Enhanced transforms for better cloud detection
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color augmentation
            transforms.ToTensor(),
        ])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load and process image
        img_name = self.image_files[idx]
        image = Image.open(os.path.join(self.data_dir, img_name)).convert('RGB')
        
        # Create better cloud mask
        img_np = np.array(image)
        
        # Convert to LAB color space for better cloud detection
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_chan = lab[:, :, 0]
        
        # Adaptive thresholding for cloud detection
        mask = cv2.adaptiveThreshold(
            l_chan,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -2
        )
        
        # Clean up mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Resize and normalize
        image = self.transform(image)
        image = self.normalize(image)
        mask = torch.from_numpy(mask).float() / 255.0
        mask = mask.view(1, self.image_size[0], self.image_size[1])
        
        return image, mask

def train_model():
    # Set device (CPU only)
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UNet(in_channels=3, out_channels=1)
    model.to(device)
    
    # Create dataset with fixed size
    dataset = CloudDataset('data/clouds', image_size=(296, 296))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    print("Starting training...")
    
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (images, masks) in enumerate(dataloader):
                images = images.to(device)
                masks = masks.unsqueeze(1).to(device)  # Add channel dimension
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                
                # Ensure outputs and masks are the same size
                if outputs.shape != masks.shape:
                    print(f"Size mismatch - Output: {outputs.shape}, Mask: {masks.shape}")
                    masks = torch.nn.functional.interpolate(masks, size=outputs.shape[2:])
                
                loss = criterion(outputs, masks)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 5 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            
            avg_loss = running_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
            
            # Save checkpoint
            torch.save(model.state_dict(), f'models/cloud_detector_epoch_{epoch+1}.pth')
        
        # Save final model
        torch.save(model.state_dict(), 'models/cloud_detector.pth')
        print("Training completed and model saved!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        torch.save(model.state_dict(), 'models/cloud_detector_backup.pth')
        print("Backup model saved!")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    train_model() 