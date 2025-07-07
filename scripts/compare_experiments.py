#!/usr/bin/env python3
"""
Compare MLflow experiments.

This script compares different experiments and their results.
"""

import os
import sys
import json
import argparse
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import structlog

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


class ExperimentComparator:
    """Compare MLflow experiments."""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        """
        Initialize experiment comparator.
        
        Args:
            tracking_uri: MLflow tracking URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def get_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiments."""
        experiments = self.client.list_experiments()
        return [{'id': exp.experiment_id, 'name': exp.name} for exp in experiments]
    
    def get_runs(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Get all runs for an experiment."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return []
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"]
        )
        
        run_data = []
        for run in runs:
            run_info = {
                'run_id': run.info.run_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'params': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags
            }
            run_data.append(run_info)
        
        return run_data
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """Compare multiple experiments."""
        comparison = {
            'experiments': {},
            'summary': {}
        }
        
        for exp_name in experiment_names:
            runs = self.get_runs(exp_name)
            comparison['experiments'][exp_name] = runs
            
            if runs:
                # Calculate summary statistics
                successful_runs = [r for r in runs if r['status'] == 'FINISHED']
                if successful_runs:
                    test_accuracies = [r['metrics'].get('test_accuracy', 0) for r in successful_runs]
                    comparison['summary'][exp_name] = {
                        'total_runs': len(runs),
                        'successful_runs': len(successful_runs),
                        'avg_test_accuracy': sum(test_accuracies) / len(test_accuracies),
                        'max_test_accuracy': max(test_accuracies),
                        'min_test_accuracy': min(test_accuracies)
                    }
        
        return comparison
    
    def create_comparison_plots(self, comparison: Dict[str, Any], output_dir: str = "plots"):
        """Create comparison plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Accuracy comparison
        exp_names = list(comparison['summary'].keys())
        avg_accuracies = [comparison['summary'][exp]['avg_test_accuracy'] for exp in exp_names]
        max_accuracies = [comparison['summary'][exp]['max_test_accuracy'] for exp in exp_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average accuracy comparison
        bars1 = ax1.bar(exp_names, avg_accuracies, alpha=0.7, color='skyblue')
        ax1.set_title('Average Test Accuracy by Experiment')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, avg_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Max accuracy comparison
        bars2 = ax2.bar(exp_names, max_accuracies, alpha=0.7, color='lightgreen')
        ax2.set_title('Maximum Test Accuracy by Experiment')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars2, max_accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter comparison
        if exp_names:
            # Get parameters from the best run of each experiment
            best_runs = []
            for exp_name in exp_names:
                runs = comparison['experiments'][exp_name]
                if runs:
                    # Find run with highest test accuracy
                    best_run = max(runs, key=lambda x: x['metrics'].get('test_accuracy', 0))
                    best_runs.append({
                        'experiment': exp_name,
                        'accuracy': best_run['metrics'].get('test_accuracy', 0),
                        'learning_rate': float(best_run['params'].get('learning_rate', 0)),
                        'batch_size': int(best_run['params'].get('batch_size', 0)),
                        'epochs': int(best_run['params'].get('epochs', 0))
                    })
            
            if best_runs:
                df = pd.DataFrame(best_runs)
                
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                # Learning rate vs accuracy
                axes[0, 0].scatter(df['learning_rate'], df['accuracy'])
                axes[0, 0].set_xlabel('Learning Rate')
                axes[0, 0].set_ylabel('Test Accuracy (%)')
                axes[0, 0].set_title('Learning Rate vs Accuracy')
                
                # Batch size vs accuracy
                axes[0, 1].scatter(df['batch_size'], df['accuracy'])
                axes[0, 1].set_xlabel('Batch Size')
                axes[0, 1].set_ylabel('Test Accuracy (%)')
                axes[0, 1].set_title('Batch Size vs Accuracy')
                
                # Epochs vs accuracy
                axes[1, 0].scatter(df['epochs'], df['accuracy'])
                axes[1, 0].set_xlabel('Epochs')
                axes[1, 0].set_ylabel('Test Accuracy (%)')
                axes[1, 0].set_title('Epochs vs Accuracy')
                
                # Accuracy distribution
                axes[1, 1].hist(df['accuracy'], bins=10, alpha=0.7, color='orange')
                axes[1, 1].set_xlabel('Test Accuracy (%)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Accuracy Distribution')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'parameter_analysis.png'), dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info("Comparison plots created", output_dir=output_dir)
    
    def export_comparison(self, comparison: Dict[str, Any], output_file: str):
        """Export comparison results to JSON."""
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info("Comparison exported", output_file=output_file)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compare MLflow experiments")
    parser.add_argument("--tracking-uri", default="http://localhost:5000",
                       help="MLflow tracking URI")
    parser.add_argument("--experiments", nargs="+", default=["mnist_classification"],
                       help="Experiment names to compare")
    parser.add_argument("--output-dir", default="experiment_comparison",
                       help="Output directory for results")
    parser.add_argument("--plots", action="store_true",
                       help="Generate comparison plots")
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = ExperimentComparator(args.tracking_uri)
    
    # Get available experiments
    available_experiments = comparator.get_experiments()
    print(f"📊 Available experiments: {[exp['name'] for exp in available_experiments]}")
    
    # Compare experiments
    comparison = comparator.compare_experiments(args.experiments)
    
    # Print summary
    print("\n📈 Experiment Comparison Summary:")
    for exp_name, summary in comparison['summary'].items():
        print(f"\n🔬 {exp_name}:")
        print(f"   Total runs: {summary['total_runs']}")
        print(f"   Successful runs: {summary['successful_runs']}")
        print(f"   Average test accuracy: {summary['avg_test_accuracy']:.2f}%")
        print(f"   Maximum test accuracy: {summary['max_test_accuracy']:.2f}%")
        print(f"   Minimum test accuracy: {summary['min_test_accuracy']:.2f}%")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export results
    output_file = os.path.join(args.output_dir, "comparison_results.json")
    comparator.export_comparison(comparison, output_file)
    
    # Generate plots if requested
    if args.plots:
        plots_dir = os.path.join(args.output_dir, "plots")
        comparator.create_comparison_plots(comparison, plots_dir)
    
    print(f"\n✅ Comparison completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 