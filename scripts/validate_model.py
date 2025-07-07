#!/usr/bin/env python3
"""
Model validation script.

This script validates trained models and checks their performance.
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
from typing import Dict, Any
import structlog

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import DataManager
from models.mnist_cnn import MNISTCNN
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
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_model(model_path: str, config: dict, data_dir: str = "data/mnist",
                  device: str = "cpu") -> Dict[str, Any]:
    """
    Validate a trained model.
    
    Args:
        model_path: Path to the trained model
        config: Validation configuration
        data_dir: Directory containing the data
        device: Device to run validation on
        
    Returns:
        Dictionary containing validation results
    """
    logger.info("Starting model validation", model_path=model_path)
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', {})
    
    model = MNISTCNN(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create data manager
    data_manager = DataManager(data_dir, config['data'])
    
    # Get test data loader
    test_loader = data_manager.get_test_loader(config['training']['batch_size'])
    
    # Create evaluator
    evaluator = ModelEvaluator(model, device=device)
    
    # Evaluate model
    results = evaluator.evaluate_model(test_loader)
    
    # Check validation thresholds
    validation_results = {
        'metrics': results['metrics'],
        'validation_passed': True,
        'issues': []
    }
    
    # Check accuracy threshold
    accuracy = results['metrics']['accuracy']
    threshold_accuracy = config['validation']['threshold_accuracy']
    if accuracy < threshold_accuracy:
        validation_results['validation_passed'] = False
        validation_results['issues'].append(
            f"Accuracy {accuracy:.2f}% below threshold {threshold_accuracy:.2f}%"
        )
    
    # Check loss threshold
    loss = results['metrics']['loss']
    threshold_loss = config['validation']['threshold_loss']
    if loss > threshold_loss:
        validation_results['validation_passed'] = False
        validation_results['issues'].append(
            f"Loss {loss:.4f} above threshold {threshold_loss:.4f}"
        )
    
    # Check class distribution
    class_distribution = results['class_distribution']
    expected_classes = 10
    if len(class_distribution) != expected_classes:
        validation_results['validation_passed'] = False
        validation_results['issues'].append(
            f"Expected {expected_classes} classes, got {len(class_distribution)}"
        )
    
    # Check for class imbalance
    min_samples = min(class_distribution.values())
    max_samples = max(class_distribution.values())
    imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    if imbalance_ratio > 2.0:  # More than 2:1 ratio
        validation_results['issues'].append(
            f"Class imbalance detected (ratio: {imbalance_ratio:.2f})"
        )
    
    logger.info("Model validation completed",
                accuracy=accuracy,
                loss=loss,
                validation_passed=validation_results['validation_passed'],
                issues_count=len(validation_results['issues']))
    
    return validation_results


def save_validation_results(results: Dict[str, Any], output_path: str):
    """Save validation results to file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Validation results saved", path=output_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate trained model")
    parser.add_argument("--model-path", default="models/best_model.pth",
                       help="Path to the trained model")
    parser.add_argument("--config", default="configs/training.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--data-dir", default="data/mnist",
                       help="Directory containing the data")
    parser.add_argument("--device", default="cpu",
                       help="Device to run validation on ('cpu' or 'cuda')")
    parser.add_argument("--output", default="models/validation_metrics.json",
                       help="Output file for validation results")
    
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        logger.error("Configuration file not found", path=args.config)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    
    # Validate model
    try:
        results = validate_model(args.model_path, config, args.data_dir, args.device)
        
        # Save results
        save_validation_results(results, args.output)
        
        # Print results
        print(f"\n📊 Model Validation Results:")
        print(f"   Accuracy: {results['metrics']['accuracy']:.2f}%")
        print(f"   Loss: {results['metrics']['loss']:.4f}")
        print(f"   Precision (macro): {results['metrics']['precision_macro']:.4f}")
        print(f"   Recall (macro): {results['metrics']['recall_macro']:.4f}")
        print(f"   F1-Score (macro): {results['metrics']['f1_macro']:.4f}")
        
        if results['validation_passed']:
            print(f"   ✅ Validation: PASSED")
        else:
            print(f"   ❌ Validation: FAILED")
            print(f"   Issues:")
            for issue in results['issues']:
                print(f"     - {issue}")
        
        print(f"\n💾 Results saved to: {args.output}")
        
    except Exception as e:
        logger.error("Model validation failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main() 