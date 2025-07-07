#!/usr/bin/env python3
"""
Download and prepare MNIST dataset.

This script downloads the MNIST dataset and prepares it for training.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import structlog

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import MNISTDataset, DataManager

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def download_mnist_data(data_dir: str = "data/mnist", config: dict = None):
    """
    Download and prepare MNIST dataset.
    
    Args:
        data_dir: Directory to store the data
        config: Configuration dictionary
    """
    if config is None:
        config = {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info("Starting MNIST data download", data_dir=data_dir)
    
    try:
        # Download training data
        logger.info("Downloading training data")
        train_dataset = MNISTDataset(data_dir, train=True)
        
        # Download test data
        logger.info("Downloading test data")
        test_dataset = MNISTDataset(data_dir, train=False)
        
        # Create data manager
        data_manager = DataManager(data_dir, config)
        
        # Save data information
        data_info_path = os.path.join(data_dir, "data_info.json")
        data_info = data_manager.save_data_info(data_info_path)
        
        # Create metrics
        metrics = {
            "train_size": data_info["train_size"],
            "test_size": data_info["test_size"],
            "num_classes": data_info["num_classes"],
            "image_size": data_info["image_size"],
            "class_distribution": data_info["train_class_distribution"]
        }
        
        # Save metrics
        metrics_path = os.path.join(data_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("MNIST data download completed",
                   train_size=data_info["train_size"],
                   test_size=data_info["test_size"],
                   data_dir=data_dir)
        
        return data_info
        
    except Exception as e:
        logger.error("Failed to download MNIST data", error=str(e))
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download MNIST dataset")
    parser.add_argument("--data-dir", default="data/mnist", 
                       help="Directory to store the data")
    parser.add_argument("--config", default=None,
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Download data
    data_info = download_mnist_data(args.data_dir, config)
    
    print(f"✅ MNIST data downloaded successfully!")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📊 Training samples: {data_info['train_size']}")
    print(f"📊 Test samples: {data_info['test_size']}")
    print(f"🎯 Number of classes: {data_info['num_classes']}")


if __name__ == "__main__":
    main() 