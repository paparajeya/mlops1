#!/usr/bin/env python3
"""
Train MNIST CNN model with MLflow experiment tracking.

This script trains a CNN model on the MNIST dataset and tracks experiments with MLflow.
"""

import os
import sys
import json
import argparse
import yaml
import mlflow
import mlflow.pytorch
from pathlib import Path
import structlog
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import DataManager
from models.mnist_cnn import MNISTCNN, MNISTTrainer
from utils.metrics import ModelEvaluator

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


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_model(config: dict, data_dir: str = "data/mnist", 
                models_dir: str = "models", device: str = "cpu"):
    """
    Train the MNIST CNN model.
    
    Args:
        config: Training configuration
        data_dir: Directory containing the data
        models_dir: Directory to save models
        device: Device to train on ('cpu' or 'cuda')
    """
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # Set up MLflow
    mlflow.set_tracking_uri(config['experiment']['tracking_uri'])
    mlflow.set_experiment(config['experiment']['name'])
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "learning_rate": config['training']['learning_rate'],
            "batch_size": config['training']['batch_size'],
            "epochs": config['training']['epochs'],
            "optimizer": config['training']['optimizer'],
            "dropout_rate": config['model']['dropout_rate']
        })
        
        # Create data manager
        data_manager = DataManager(data_dir, config['data'])
        
        # Create data loaders
        train_loader, val_loader, test_loader = data_manager.create_data_loaders(
            config['training']['batch_size']
        )
        
        # Create model
        model = MNISTCNN(config['model'])
        
        # Create trainer
        trainer = MNISTTrainer(model, config['training'], device=device)
        
        # Training loop
        best_val_accuracy = 0.0
        early_stopping_counter = 0
        early_stopping_patience = config['validation']['early_stopping_patience']
        
        logger.info("Starting training", 
                   epochs=config['training']['epochs'],
                   device=device)
        
        for epoch in range(config['training']['epochs']):
            # Train epoch
            train_metrics = trainer.train_epoch(train_loader, epoch + 1)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Log metrics
            mlflow.log_metrics({
                f"train_loss": train_metrics['loss'],
                f"train_accuracy": train_metrics['accuracy'],
                f"val_loss": val_metrics['loss'],
                f"val_accuracy": val_metrics['accuracy'],
                f"learning_rate": train_metrics['learning_rate']
            }, step=epoch)
            
            logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}",
                       train_loss=train_metrics['loss'],
                       train_accuracy=train_metrics['accuracy'],
                       val_loss=val_metrics['loss'],
                       val_accuracy=val_metrics['accuracy'])
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                best_model_path = os.path.join(models_dir, "best_model.pth")
                trainer.save_model(best_model_path)
                
                # Log model artifact
                mlflow.log_artifact(best_model_path)
                
                early_stopping_counter = 0
                logger.info("New best model saved", 
                           accuracy=best_val_accuracy,
                           path=best_model_path)
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                logger.info("Early stopping triggered", 
                           patience=early_stopping_patience,
                           best_accuracy=best_val_accuracy)
                break
        
        # Evaluate on test set
        logger.info("Evaluating on test set")
        evaluator = ModelEvaluator(model, device=device)
        test_results = evaluator.evaluate_model(test_loader)
        
        # Log test metrics
        test_metrics = test_results['metrics']
        mlflow.log_metrics({
            "test_accuracy": test_metrics['accuracy'],
            "test_precision_macro": test_metrics['precision_macro'],
            "test_recall_macro": test_metrics['recall_macro'],
            "test_f1_macro": test_metrics['f1_macro']
        })
        
        # Save final model
        final_model_path = os.path.join(models_dir, "final_model.pth")
        trainer.save_model(final_model_path)
        mlflow.log_artifact(final_model_path)
        
        # Save metrics
        metrics_path = os.path.join(models_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)
        
        # Log model info
        model_info = model.get_model_info()
        model_info_path = os.path.join(models_dir, "model_info.json")
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        mlflow.log_artifact(model_info_path)
        
        logger.info("Training completed",
                   best_val_accuracy=best_val_accuracy,
                   test_accuracy=test_metrics['accuracy'])
        
        return {
            'best_val_accuracy': best_val_accuracy,
            'test_metrics': test_metrics,
            'model_info': model_info
        }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train MNIST CNN model")
    parser.add_argument("--config", default="configs/training.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--data-dir", default="data/mnist",
                       help="Directory containing the data")
    parser.add_argument("--models-dir", default="models",
                       help="Directory to save models")
    parser.add_argument("--device", default="cpu",
                       help="Device to train on ('cpu' or 'cuda')")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error("Configuration file not found", path=args.config)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Override parameters if provided
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    # Train model
    try:
        results = train_model(config, args.data_dir, args.models_dir, args.device)
        
        print(f"✅ Training completed successfully!")
        print(f"🏆 Best validation accuracy: {results['best_val_accuracy']:.2f}%")
        print(f"📊 Test accuracy: {results['test_metrics']['accuracy']:.2f}%")
        print(f"📁 Models saved to: {args.models_dir}")
        print(f"🔗 MLflow experiment: {config['experiment']['name']}")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main() 