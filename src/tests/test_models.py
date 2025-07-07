"""
Unit tests for ML model components.

This module contains tests for the MNIST CNN model and related components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.mnist_cnn import MNISTCNN, MNISTTrainer
from data.dataset import MNISTDataset, DataManager


class TestMNISTCNN:
    """Test cases for MNIST CNN model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'input_size': [28, 28],
            'num_classes': 10,
            'dropout_rate': 0.2
        }
        self.model = MNISTCNN(self.config)
    
    def test_model_initialization(self):
        """Test model initialization."""
        assert self.model.input_size == [28, 28]
        assert self.model.num_classes == 10
        assert self.model.dropout_rate == 0.2
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        batch_size = 4
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        
        output = self.model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_info(self):
        """Test model information retrieval."""
        info = self.model.get_model_info()
        
        assert info['model_name'] == 'MNISTCNN'
        assert info['input_size'] == [28, 28]
        assert info['num_classes'] == 10
        assert info['architecture'] == 'CNN'
        assert 'total_parameters' in info
        assert 'trainable_parameters' in info
    
    def test_model_parameters(self):
        """Test that model has trainable parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
    
    def test_model_device_transfer(self):
        """Test model transfer to different devices."""
        if torch.cuda.is_available():
            self.model.cuda()
            assert next(self.model.parameters()).is_cuda
            
            self.model.cpu()
            assert not next(self.model.parameters()).is_cuda


class TestMNISTTrainer:
    """Test cases for MNIST trainer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_function': 'cross_entropy',
            'scheduler': 'step',
            'scheduler_step_size': 7,
            'scheduler_gamma': 0.1
        }
        self.model = MNISTCNN({'input_size': [28, 28], 'num_classes': 10})
        self.trainer = MNISTTrainer(self.model, self.config, device='cpu')
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        assert self.trainer.model == self.model
        assert self.trainer.config == self.config
        assert self.trainer.device == 'cpu'
        assert self.trainer.optimizer is not None
        assert self.trainer.criterion is not None
    
    def test_trainer_with_sgd_optimizer(self):
        """Test trainer with SGD optimizer."""
        config = self.config.copy()
        config['optimizer'] = 'sgd'
        
        trainer = MNISTTrainer(self.model, config, device='cpu')
        assert isinstance(trainer.optimizer, torch.optim.SGD)
    
    def test_trainer_invalid_optimizer(self):
        """Test trainer with invalid optimizer."""
        config = self.config.copy()
        config['optimizer'] = 'invalid'
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            MNISTTrainer(self.model, config, device='cpu')
    
    def test_trainer_invalid_loss_function(self):
        """Test trainer with invalid loss function."""
        config = self.config.copy()
        config['loss_function'] = 'invalid'
        
        with pytest.raises(ValueError, match="Unsupported loss function"):
            MNISTTrainer(self.model, config, device='cpu')
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        save_path = tmp_path / "test_model.pth"
        
        # Save model
        self.trainer.save_model(str(save_path))
        assert save_path.exists()
        
        # Load model
        new_trainer = MNISTTrainer(self.model, self.config, device='cpu')
        new_trainer.load_model(str(save_path))
        
        # Check that model parameters are the same
        for param1, param2 in zip(self.trainer.model.parameters(), 
                                 new_trainer.model.parameters()):
            assert torch.equal(param1, param2)


class TestMNISTDataset:
    """Test cases for MNIST dataset."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_dir = "test_data"
    
    @patch('torchvision.datasets.MNIST')
    def test_dataset_initialization(self, mock_mnist):
        """Test dataset initialization."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_mnist.return_value = mock_dataset
        
        dataset = MNISTDataset(self.data_dir, train=True)
        
        mock_mnist.assert_called_once_with(
            root=self.data_dir,
            train=True,
            download=True,
            transform=dataset.transform
        )
    
    @patch('torchvision.datasets.MNIST')
    def test_dataset_length(self, mock_mnist):
        """Test dataset length."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_mnist.return_value = mock_dataset
        
        dataset = MNISTDataset(self.data_dir, train=True)
        assert len(dataset) == 1000
    
    @patch('torchvision.datasets.MNIST')
    def test_dataset_getitem(self, mock_mnist):
        """Test dataset item retrieval."""
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=(torch.randn(1, 28, 28), 5))
        mock_mnist.return_value = mock_dataset
        
        dataset = MNISTDataset(self.data_dir, train=True)
        item = dataset[0]
        
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], torch.Tensor)
        assert isinstance(item[1], int)


class TestDataManager:
    """Test cases for data manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
        self.data_dir = "test_data"
        self.data_manager = DataManager(self.data_dir, self.config)
    
    def test_data_manager_initialization(self):
        """Test data manager initialization."""
        assert self.data_manager.data_dir == self.data_dir
        assert self.data_manager.config == self.config
        assert self.data_manager.train_split == 0.8
        assert self.data_manager.val_split == 0.1
        assert self.data_manager.test_split == 0.1
    
    def test_invalid_splits(self):
        """Test data manager with invalid splits."""
        invalid_config = {
            'train_split': 0.5,
            'val_split': 0.3,
            'test_split': 0.3  # Sum > 1
        }
        
        with pytest.raises(ValueError, match="Data splits must sum to 1.0"):
            DataManager(self.data_dir, invalid_config)
    
    @patch('data.dataset.MNISTDataset')
    def test_create_data_loaders(self, mock_dataset_class):
        """Test data loader creation."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset_class.return_value = mock_dataset
        
        train_loader, val_loader, test_loader = self.data_manager.create_data_loaders(32)
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_save_data_info(self, tmp_path):
        """Test data info saving."""
        output_path = tmp_path / "data_info.json"
        
        with patch('data.dataset.MNISTDataset') as mock_dataset_class:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=1000)
            mock_dataset.get_class_distribution = Mock(return_value={i: 100 for i in range(10)})
            mock_dataset_class.return_value = mock_dataset
            
            info = self.data_manager.save_data_info(str(output_path))
            
            assert output_path.exists()
            assert 'train_size' in info
            assert 'test_size' in info
            assert 'num_classes' in info


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_model_training_cycle(self):
        """Test complete model training cycle."""
        # Create model
        config = {
            'input_size': [28, 28],
            'num_classes': 10,
            'dropout_rate': 0.2
        }
        model = MNISTCNN(config)
        
        # Create trainer
        trainer_config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_function': 'cross_entropy'
        }
        trainer = MNISTTrainer(model, trainer_config, device='cpu')
        
        # Create dummy data
        batch_size = 4
        dummy_data = torch.randn(batch_size, 1, 28, 28)
        dummy_targets = torch.randint(0, 10, (batch_size,))
        
        # Create mock data loader
        mock_loader = [(dummy_data, dummy_targets)]
        
        # Test training epoch
        metrics = trainer.train_epoch(mock_loader, epoch=1)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert 'learning_rate' in metrics
        assert metrics['loss'] >= 0
        assert 0 <= metrics['accuracy'] <= 100
    
    def test_model_validation(self):
        """Test model validation."""
        # Create model
        config = {
            'input_size': [28, 28],
            'num_classes': 10,
            'dropout_rate': 0.2
        }
        model = MNISTCNN(config)
        
        # Create trainer
        trainer_config = {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss_function': 'cross_entropy'
        }
        trainer = MNISTTrainer(model, trainer_config, device='cpu')
        
        # Create dummy data
        batch_size = 4
        dummy_data = torch.randn(batch_size, 1, 28, 28)
        dummy_targets = torch.randint(0, 10, (batch_size,))
        
        # Create mock data loader
        mock_loader = [(dummy_data, dummy_targets)]
        
        # Test validation
        metrics = trainer.validate(mock_loader)
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics
        assert metrics['loss'] >= 0
        assert 0 <= metrics['accuracy'] <= 100


if __name__ == "__main__":
    pytest.main([__file__]) 