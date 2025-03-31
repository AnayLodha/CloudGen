import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import os

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2)
        
        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder blocks
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec4 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Decoder path with size matching
        # Up-sample e4 and concatenate with e3
        d4_up = self.up(e4)
        if d4_up.shape[2] != e3.shape[2]:  # If sizes don't match
            e3 = e3[:, :, :d4_up.shape[2], :d4_up.shape[3]]
        d4 = self.dec4(torch.cat([d4_up, e3], dim=1))
        
        # Up-sample d4 and concatenate with e2
        d3_up = self.up(d4)
        if d3_up.shape[2] != e2.shape[2]:  # If sizes don't match
            e2 = e2[:, :, :d3_up.shape[2], :d3_up.shape[3]]
        d3 = self.dec3(torch.cat([d3_up, e2], dim=1))
        
        # Up-sample d3 and concatenate with e1
        d2_up = self.up(d3)
        if d2_up.shape[2] != e1.shape[2]:  # If sizes don't match
            e1 = e1[:, :, :d2_up.shape[2], :d2_up.shape[3]]
        d2 = self.dec2(torch.cat([d2_up, e1], dim=1))
        
        # Final 1x1 convolution
        out = self.dec1(d2)
        return torch.sigmoid(out)

class CloudDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the CloudDetector
        
        Args:
            model_path (Optional[str]): Path to pre-trained model weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet().to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
    
    def detect_clouds(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect clouds in the input image
        
        Args:
            image (np.ndarray): Input image (RGB, normalized to [0,1])
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Cloud mask and edge map
        """
        # Ensure input image has correct dimensions
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=2)
        
        # Prepare input tensor
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Get cloud mask
        with torch.no_grad():
            mask = torch.sigmoid(self.model(x))
            mask = mask.squeeze().cpu().numpy()
        
        # Apply threshold
        mask = (mask > 0.5).astype(np.uint8)
        
        # Get edge map
        edges = self.detect_edges(mask)
        
        return mask, edges
    
    def detect_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Detect edges in the cloud mask
        
        Args:
            mask (np.ndarray): Cloud mask
            
        Returns:
            np.ndarray: Edge map
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred.astype(np.uint8), 50, 150)
        
        # Apply morphological operations to clean up edges
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def refine_edges(self, edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Refine the edge map using the cloud mask
        
        Args:
            edges (np.ndarray): Initial edge map
            mask (np.ndarray): Cloud mask
            
        Returns:
            np.ndarray: Refined edge map
        """
        # Dilate the mask slightly
        kernel = np.ones((5,5), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Only keep edges that are near cloud regions
        refined_edges = cv2.bitwise_and(edges, edges, mask=dilated_mask)
        
        return refined_edges
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process an image to detect clouds and their edges
        
        Args:
            image (np.ndarray): Input image (RGB, normalized to [0,1])
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Original image, cloud mask, and edge map
        """
        # Detect clouds and edges
        mask, edges = self.detect_clouds(image)
        
        # Refine edges
        refined_edges = self.refine_edges(edges, mask)
        
        return image, mask, refined_edges 