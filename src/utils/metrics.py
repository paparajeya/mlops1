"""
Metrics and Evaluation Utilities

This module provides functions for calculating and tracking ML performance metrics.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import structlog

logger = structlog.get_logger()


class MetricsCalculator:
    """Calculator for various ML metrics."""
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes in the dataset
        """
        self.num_classes = num_classes
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing various metrics
        """
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1-score for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        metrics['precision_macro'] = np.mean(precision)
        metrics['recall_macro'] = np.mean(recall)
        metrics['f1_macro'] = np.mean(f1)
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted
        
        # Per-class metrics
        for i in range(self.num_classes):
            metrics[f'precision_class_{i}'] = precision[i]
            metrics[f'recall_class_{i}'] = recall[i]
            metrics[f'f1_class_{i}'] = f1[i]
            metrics[f'support_class_{i}'] = support[i]
        
        # Additional metrics if probabilities are provided
        if y_proba is not None:
            metrics.update(self._calculate_probability_metrics(y_true, y_proba))
        
        return metrics
    
    def _calculate_probability_metrics(self, y_true: np.ndarray, 
                                     y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate metrics based on predicted probabilities."""
        metrics = {}
        
        # Log loss
        from sklearn.metrics import log_loss
        try:
            metrics['log_loss'] = log_loss(y_true, y_proba)
        except:
            metrics['log_loss'] = float('inf')
        
        # Average confidence for correct and incorrect predictions
        correct_mask = (y_true == np.argmax(y_proba, axis=1))
        if np.any(correct_mask):
            metrics['avg_confidence_correct'] = np.mean(
                np.max(y_proba[correct_mask], axis=1)
            )
        if np.any(~correct_mask):
            metrics['avg_confidence_incorrect'] = np.mean(
                np.max(y_proba[~correct_mask], axis=1)
            )
        
        return metrics
    
    def create_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: str = None) -> np.ndarray:
        """
        Create and optionally save confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the confusion matrix plot
            
        Returns:
            Confusion matrix array
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if save_path:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=range(self.num_classes),
                       yticklabels=range(self.num_classes))
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Saved confusion matrix", path=save_path)
        
        return cm
    
    def calculate_class_distribution(self, y_true: np.ndarray) -> Dict[int, int]:
        """Calculate the distribution of classes in the dataset."""
        unique, counts = np.unique(y_true, return_counts=True)
        return dict(zip(unique, counts))


class ModelEvaluator:
    """Evaluator for ML models with comprehensive metrics."""
    
    def __init__(self, model, device: str = "cpu"):
        """
        Initialize model evaluator.
        
        Args:
            model: PyTorch model
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_model(self, data_loader) -> Dict[str, Any]:
        """
        Evaluate model on a data loader.
        
        Args:
            data_loader: PyTorch DataLoader
            
        Returns:
            Dictionary containing evaluation results
        """
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        calculator = MetricsCalculator(num_classes=10)
        metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        # Add additional information
        results = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
            'probabilities': y_proba.tolist(),
            'targets': y_true.tolist(),
            'class_distribution': calculator.calculate_class_distribution(y_true)
        }
        
        logger.info("Model evaluation completed", 
                   accuracy=metrics['accuracy'],
                   f1_macro=metrics['f1_macro'])
        
        return results
    
    def evaluate_with_confidence_threshold(self, data_loader, 
                                        confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Evaluate model with confidence threshold filtering.
        
        Args:
            data_loader: PyTorch DataLoader
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Dictionary containing evaluation results
        """
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                probabilities = torch.softmax(output, dim=1)
                predictions = torch.argmax(output, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                # Filter by confidence threshold
                confident_mask = confidences >= confidence_threshold
                
                if torch.any(confident_mask):
                    all_predictions.extend(predictions[confident_mask].cpu().numpy())
                    all_probabilities.extend(probabilities[confident_mask].cpu().numpy())
                    all_targets.extend(target[confident_mask].cpu().numpy())
                    all_confidences.extend(confidences[confident_mask].cpu().numpy())
        
        if not all_predictions:
            logger.warning("No predictions met confidence threshold", 
                         threshold=confidence_threshold)
            return {'metrics': {}, 'coverage': 0.0}
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        calculator = MetricsCalculator(num_classes=10)
        metrics = calculator.calculate_metrics(y_true, y_pred, y_proba)
        
        # Calculate coverage
        total_samples = sum(len(batch[1]) for batch in data_loader)
        coverage = len(y_true) / total_samples
        
        results = {
            'metrics': metrics,
            'coverage': coverage,
            'avg_confidence': np.mean(all_confidences),
            'confidence_threshold': confidence_threshold
        }
        
        logger.info("Confidence-threshold evaluation completed",
                   accuracy=metrics.get('accuracy', 0),
                   coverage=coverage,
                   avg_confidence=np.mean(all_confidences))
        
        return results


def save_metrics(metrics: Dict[str, float], filepath: str):
    """Save metrics to a JSON file."""
    import json
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Saved metrics", path=filepath)


def load_metrics(filepath: str) -> Dict[str, float]:
    """Load metrics from a JSON file."""
    import json
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    logger.info("Loaded metrics", path=filepath)
    return metrics


def compare_models(metrics_dict: Dict[str, Dict[str, float]], 
                  primary_metric: str = 'accuracy') -> Dict[str, Any]:
    """
    Compare multiple models based on their metrics.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        primary_metric: Primary metric to compare on
        
    Returns:
        Comparison results
    """
    comparison = {
        'models': list(metrics_dict.keys()),
        'primary_metric': primary_metric,
        'rankings': {},
        'best_model': None,
        'best_score': -1
    }
    
    # Rank models by primary metric
    rankings = []
    for model_name, metrics in metrics_dict.items():
        score = metrics.get(primary_metric, 0)
        rankings.append((model_name, score))
        
        if score > comparison['best_score']:
            comparison['best_score'] = score
            comparison['best_model'] = model_name
    
    # Sort by score (descending)
    rankings.sort(key=lambda x: x[1], reverse=True)
    comparison['rankings'] = rankings
    
    logger.info("Model comparison completed",
               best_model=comparison['best_model'],
               best_score=comparison['best_score'])
    
    return comparison 