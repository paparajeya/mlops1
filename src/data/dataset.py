"""
MNIST Dataset Module

This module provides data loading and preprocessing functionality for the MNIST dataset.
"""

import os
import json
from typing import Tuple, Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import structlog

logger = structlog.get_logger()


class MNISTDataset(Dataset):
    """Custom MNIST Dataset class for loading and preprocessing MNIST data."""
    
    def __init__(self, data_dir: str, train: bool = True, transform: Optional[transforms.Compose] = None):
        """
        Initialize MNIST dataset.
        
        Args:
            data_dir: Directory containing the MNIST data
            train: Whether to load training or test data
            transform: Optional transforms to apply to the data
        """
        self.data_dir = data_dir
        self.train = train
        self.transform = transform or self._get_default_transforms(train)
        
        # Load MNIST dataset
        self.dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            download=True,
            transform=self.transform
        )
        
        logger.info(f"Loaded MNIST dataset", 
                   split="train" if train else "test",
                   size=len(self.dataset))
    
    def _get_default_transforms(self, train: bool) -> transforms.Compose:
        """Get default transforms for MNIST data."""
        if train:
            return transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        class_counts = {}
        for _, label in self.dataset:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts


class DataManager:
    """Manages data loading, preprocessing, and splitting."""
    
    def __init__(self, data_dir: str, config: Dict[str, Any]):
        """
        Initialize DataManager.
        
        Args:
            data_dir: Directory containing the data
            config: Configuration dictionary
        """
        self.data_dir = data_dir
        self.config = config
        self.train_split = config.get('train_split', 0.8)
        self.val_split = config.get('val_split', 0.1)
        self.test_split = config.get('test_split', 0.1)
        
        # Ensure splits sum to 1
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total}")
    
    def create_data_loaders(self, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            batch_size: Batch size for the data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load full training dataset
        full_train_dataset = MNISTDataset(self.data_dir, train=True)
        
        # Calculate split sizes
        total_size = len(full_train_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset, test_dataset = random_split(
            full_train_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info("Created data loaders",
                   train_size=len(train_dataset),
                   val_size=len(val_dataset),
                   test_size=len(test_dataset),
                   batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    
    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Get test data loader using the official test set."""
        test_dataset = MNISTDataset(self.data_dir, train=False)
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def save_data_info(self, output_path: str):
        """Save dataset information and statistics."""
        train_dataset = MNISTDataset(self.data_dir, train=True)
        test_dataset = MNISTDataset(self.data_dir, train=False)
        
        data_info = {
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "num_classes": 10,
            "image_size": [28, 28],
            "train_class_distribution": train_dataset.get_class_distribution(),
            "test_class_distribution": test_dataset.get_class_distribution(),
            "splits": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data_info, f, indent=2)
        
        logger.info("Saved data information", output_path=output_path)
        return data_info 