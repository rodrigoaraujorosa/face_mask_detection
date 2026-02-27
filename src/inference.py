"""
Face Mask Detection - Inference Module

This module provides functions for loading the trained model and making predictions
on new images.

Usage:
    python src/inference.py --image path/to/image.jpg
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# Define model architectures (must match training)
class CustomCNN(nn.Module):
    """Custom CNN built from scratch."""
    
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_resnet18(num_classes=2, pretrained=False):
    """ResNet18 with transfer learning."""
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_mobilenet_v2(num_classes=2, pretrained=False):
    """MobileNetV2 with transfer learning."""
    model = models.mobilenet_v2(pretrained=pretrained)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


def get_vgg16(num_classes=2, pretrained=False):
    """VGG16 with transfer learning."""
    model = models.vgg16(pretrained=pretrained)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    return model


def get_efficientnet_b0(num_classes=2, pretrained=False):
    """EfficientNet-B0 with transfer learning."""
    model = models.efficientnet_b0(pretrained=pretrained)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


class FaceMaskDetector:
    """Face Mask Detection inference class."""
    
    def __init__(self, model_path, device='auto'):
        """
        Initialize the detector.
        
        Args:
            model_path (str): Path to the saved model (.pkl file)
            device (str): Device to use ('auto', 'cuda', or 'cpu')
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model, self.config = self._load_model(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Class names
        self.class_names = {0: 'With Mask', 1: 'Without Mask'}
        
        print(f"Model loaded: {self.config['model_name']}")
        print(f"Device: {self.device}")
    
    def _load_model(self, model_path):
        """Load the saved model."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model_name = model_data['model_name']
        num_classes = model_data['config']['num_classes']
        
        # Recreate model architecture
        if model_name == 'Custom_CNN':
            model = CustomCNN(num_classes=num_classes)
        elif model_name == 'ResNet18':
            model = get_resnet18(num_classes=num_classes, pretrained=False)
        elif model_name == 'MobileNetV2':
            model = get_mobilenet_v2(num_classes=num_classes, pretrained=False)
        elif model_name == 'VGG16':
            model = get_vgg16(num_classes=num_classes, pretrained=False)
        elif model_name == 'EfficientNet_B0':
            model = get_efficientnet_b0(num_classes=num_classes, pretrained=False)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load weights
        model.load_state_dict(model_data['model_state_dict'])
        
        return model, model_data
    
    def predict(self, image_path):
        """
        Make prediction on a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
        
        # Transform
        image_tensor = self.transform(image_pil).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Prepare results
        pred_class = predicted.item()
        pred_confidence = confidence.item()
        
        result = {
            'class': self.class_names[pred_class],
            'class_id': pred_class,
            'confidence': pred_confidence,
            'probabilities': {
                'with_mask': probabilities[0][0].item(),
                'without_mask': probabilities[0][1].item()
            }
        }
        
        return result
    
    def predict_with_visualization(self, image_path, output_path=None):
        """
        Make prediction and create visualization.
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the output image (optional)
            
        Returns:
            dict: Prediction results
        """
        # Get prediction
        result = self.predict(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        
        # Add text overlay
        label = result['class']
        confidence = result['confidence'] * 100
        text = f"{label}: {confidence:.1f}%"
        
        # Set color based on prediction
        color = (0, 255, 0) if result['class_id'] == 0 else (0, 0, 255)
        
        # Add text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )
        
        # Draw background rectangle
        cv2.rectangle(image, (10, 10), 
                     (20 + text_width, 20 + text_height + baseline),
                     color, -1)
        
        # Draw text
        cv2.putText(image, text, (15, 15 + text_height),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw border
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, 0), (w-1, h-1), color, 5)
        
        # Save or display
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Output saved to: {output_path}")
        else:
            cv2.imshow('Face Mask Detection', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Face Mask Detection - Inference'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.pkl',
        help='Path to model file (default: models/best_model.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output image (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please run the training notebook first to generate the model.")
        sys.exit(1)
    
    # Initialize detector
    print("Initializing Face Mask Detector...")
    detector = FaceMaskDetector(args.model, device=args.device)
    
    # Make prediction
    print(f"\nProcessing image: {args.image}")
    result = detector.predict_with_visualization(args.image, args.output)
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"Prediction: {result['class']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print("\nProbabilities:")
    print(f"  With Mask:    {result['probabilities']['with_mask']*100:.2f}%")
    print(f"  Without Mask: {result['probabilities']['without_mask']*100:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()
