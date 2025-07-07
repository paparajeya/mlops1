"""
MNIST CNN Model

This module contains the CNN architecture for MNIST digit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()


class MNISTCNN(nn.Module):
    """Convolutional Neural Network for MNIST digit classification."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the CNN model.
        
        Args:
            config: Model configuration dictionary
        """
        super(MNISTCNN, self).__init__()
        
        self.input_size = config.get('input_size', [28, 28])
        self.num_classes = config.get('num_classes', 10)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Calculate the size after convolutions and pooling
        # Input: 28x28 -> After 3 conv layers and 2 pooling: 7x7
        conv_output_size = 7 * 7 * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.num_classes)
        
        logger.info("Initialized MNIST CNN model",
                   input_size=self.input_size,
                   num_classes=self.num_classes,
                   dropout_rate=self.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "MNISTCNN",
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "CNN"
        }


class MNISTTrainer:
    """Trainer class for MNIST CNN model."""
    
    def __init__(self, model: MNISTCNN, config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Setup optimizer
        optimizer_name = config.get('optimizer', 'adam').lower()
        learning_rate = config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Setup loss function
        loss_function = config.get('loss_function', 'cross_entropy')
        if loss_function == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Setup scheduler
        scheduler_name = config.get('scheduler', 'step')
        if scheduler_name == 'step':
            step_size = config.get('scheduler_step_size', 7)
            gamma = config.get('scheduler_gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        else:
            self.scheduler = None
        
        logger.info("Initialized trainer",
                   optimizer=optimizer_name,
                   learning_rate=learning_rate,
                   loss_function=loss_function,
                   device=device)
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f"Training Epoch: {epoch}",
                           batch=batch_idx,
                           loss=loss.item(),
                           accuracy=100. * correct / total)
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def save_model(self, path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_info': self.model.get_model_info()
        }, path)
        logger.info("Saved model", path=path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Loaded model", path=path) 