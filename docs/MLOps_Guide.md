# MLOps Complete Guide

This guide provides comprehensive instructions for setting up and using the MLOps project for MNIST digit classification.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Data Pipeline](#data-pipeline)
5. [Model Training](#model-training)
6. [Experiment Tracking](#experiment-tracking)
7. [Model Validation](#model-validation)
8. [API Deployment](#api-deployment)
9. [Testing](#testing)
10. [Monitoring](#monitoring)
11. [CI/CD Pipeline](#cicd-pipeline)
12. [Troubleshooting](#troubleshooting)

## Project Overview

This MLOps project demonstrates a complete machine learning pipeline for MNIST digit classification with the following components:

- **Data Versioning**: DVC for data and model versioning
- **Experiment Tracking**: MLflow for experiment management
- **Model Training**: PyTorch CNN for MNIST classification
- **API Service**: FastAPI for model inference
- **Testing**: Comprehensive unit and integration tests
- **Containerization**: Docker for reproducible environments
- **CI/CD**: GitHub Actions for automated workflows
- **Monitoring**: Prometheus and Grafana for observability

## Prerequisites

### System Requirements

- Python 3.12+
- Docker and Docker Compose
- Git
- At least 4GB RAM
- 10GB free disk space

### Required Software

1. **Python 3.12+**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   
   # macOS
   brew install python3
   
   # Windows
   # Download from python.org
   ```

2. **Docker**
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # macOS
   brew install --cask docker
   
   # Windows
   # Download Docker Desktop from docker.com
   ```

3. **DVC**
   ```bash
   pip install dvc
   ```

4. **Git**
   ```bash
   # Ubuntu/Debian
   sudo apt install git
   
   # macOS
   brew install git
   ```

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd mlops-mnist

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Start Services

```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000 &

# Start API service
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Access Services

- **MLflow UI**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Data Pipeline

### DVC Configuration

DVC is used for data versioning and pipeline orchestration.

```bash
# Initialize DVC (if not already done)
dvc init

# Add remote storage
dvc remote add -d myremote ./data

# Download data
python scripts/download_data.py

# Add data to DVC
dvc add data/mnist/
dvc push
```

### Data Structure

```
data/
├── mnist/
│   ├── MNIST/
│   │   ├── raw/
│   │   └── processed/
│   ├── data_info.json
│   └── metrics.json
└── processed/
    ├── train/
    ├── val/
    └── test/
```

### Data Pipeline Commands

```bash
# Download data
python scripts/download_data.py

# Run complete data pipeline
dvc repro

# Check pipeline status
dvc status

# View pipeline
dvc dag
```

## Model Training

### Training Configuration

The training configuration is defined in `configs/training.yaml`:

```yaml
model:
  name: "mnist_cnn"
  architecture: "cnn"
  input_size: [28, 28]
  num_classes: 10
  dropout_rate: 0.2

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 10
  optimizer: "adam"
  loss_function: "cross_entropy"
```

### Training Commands

```bash
# Train with default parameters
python scripts/train.py

# Train with custom parameters
python scripts/train.py --learning-rate 0.01 --epochs 20 --batch-size 32

# Train on GPU (if available)
python scripts/train.py --device cuda

# Train with specific config
python scripts/train.py --config configs/training.yaml
```

### Training Output

Training produces:
- Trained model files in `models/`
- MLflow experiment logs
- Training metrics and plots
- Model validation results

## Experiment Tracking

### MLflow Setup

MLflow is used for experiment tracking and model registry.

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Access MLflow UI
open http://localhost:5000
```

### Experiment Management

```bash
# List experiments
mlflow experiments list

# Compare experiments
python scripts/compare_experiments.py --experiments exp1 exp2

# Export experiment results
python scripts/compare_experiments.py --output-dir results --plots
```

### MLflow Features

- **Parameter Tracking**: Learning rate, batch size, epochs
- **Metric Tracking**: Accuracy, loss, precision, recall
- **Artifact Storage**: Models, plots, logs
- **Model Registry**: Version control for models
- **Experiment Comparison**: Side-by-side analysis

## Model Validation

### Validation Process

The validation pipeline checks:
- Model performance metrics
- Data quality issues
- Model drift detection
- Performance thresholds

```bash
# Validate trained model
python scripts/validate_model.py

# Validate specific model
python scripts/validate_model.py --model-path models/best_model.pth

# Validate with custom thresholds
python scripts/validate_model.py --config configs/training.yaml
```

### Validation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis

## API Deployment

### FastAPI Service

The API provides REST endpoints for model inference.

```bash
# Start API service
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test API endpoints
python scripts/test_api.py

# Performance test
python scripts/test_api.py --performance --num-requests 100
```

### API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `GET /model/info`: Model information
- `POST /predict`: Single prediction
- `POST /predict/batch`: Batch prediction
- `GET /metrics`: Prometheus metrics

### API Testing

```bash
# Test all endpoints
python scripts/test_api.py

# Test specific endpoint
curl -X GET http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [[0.1, 0.2, ...]], "confidence_threshold": 0.5}'
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest src/tests/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run with verbose output
pytest -v
```

### Integration Tests

```bash
# Test complete pipeline
python scripts/test_pipeline.py

# Test API integration
python scripts/test_api_integration.py

# Performance testing
python scripts/test_api.py --performance
```

### Test Coverage

The project includes tests for:
- Model components (CNN, trainer)
- Data processing (dataset, data manager)
- API endpoints (health, prediction)
- Utility functions (metrics, validation)
- Integration scenarios

## Monitoring

### Prometheus Setup

Prometheus collects metrics from the API service.

```bash
# Start Prometheus
docker run -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### Grafana Dashboard

Grafana provides visualization for metrics.

```bash
# Start Grafana
docker run -p 3000:3000 \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  grafana/grafana
```

### Key Metrics

- **API Performance**: Response time, throughput
- **Model Performance**: Prediction accuracy, confidence
- **System Metrics**: CPU, memory, disk usage
- **Business Metrics**: Request volume, error rates

## CI/CD Pipeline

### GitHub Actions

The CI/CD pipeline includes:

1. **Code Quality Checks**
   - Linting (flake8, black, isort)
   - Type checking (mypy)
   - Security scanning (bandit, safety)

2. **Testing**
   - Unit tests with pytest
   - Integration tests
   - Docker image testing

3. **Model Training**
   - Automated model training
   - Experiment tracking
   - Model validation

4. **Deployment**
   - Staging deployment
   - Production deployment
   - Performance testing

### Pipeline Triggers

- **Pull Request**: Runs tests and validation
- **Push to main**: Trains model and deploys to staging
- **Release tag**: Deploys to production

### Manual Triggers

```bash
# Trigger workflow manually
gh workflow run ci.yml

# View workflow status
gh run list
```

## Docker Deployment

### Single Service

```bash
# Build image
docker build -t mlops-mnist .

# Run container
docker run -p 8000:8000 mlops-mnist

# Run with volumes
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  mlops-mnist
```

### Multi-Service with Docker Compose

```bash
# Start all services
docker-compose up --build

# Start specific service
docker-compose up api

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Service Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │    │   MLflow    │    │  Prometheus │
│   (API)     │    │ (Tracking)  │    │ (Monitoring)│
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                    ┌─────────────┐
                    │   Grafana   │
                    │ (Dashboard) │
                    └─────────────┘
```

## Troubleshooting

### Common Issues

#### 1. DVC Issues

```bash
# Reset DVC cache
dvc checkout --force
dvc pull

# Clear DVC cache
dvc gc

# Reinitialize DVC
rm -rf .dvc
dvc init
```

#### 2. MLflow Connection Issues

```bash
# Check MLflow server
curl http://localhost:5000/health

# Restart MLflow
pkill -f mlflow
mlflow server --host 0.0.0.0 --port 5000 &
```

#### 3. Docker Issues

```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t mlops-mnist .

# Check container logs
docker logs <container_id>
```

#### 4. API Issues

```bash
# Check API health
curl http://localhost:8000/health

# Test API endpoints
python scripts/test_api.py

# Check API logs
tail -f logs/api.log
```

#### 5. Training Issues

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reduce batch size for memory issues
python scripts/train.py --batch-size 16

# Use CPU only
python scripts/train.py --device cpu
```

### Performance Optimization

#### 1. Model Optimization

```bash
# Use mixed precision training
python scripts/train.py --mixed-precision

# Optimize for inference
python scripts/optimize_model.py

# Quantize model
python scripts/quantize_model.py
```

#### 2. API Optimization

```bash
# Increase workers
uvicorn src.api.main:app --workers 4

# Use async processing
python scripts/train.py --async-processing

# Enable caching
python scripts/train.py --enable-cache
```

#### 3. Data Pipeline Optimization

```bash
# Use parallel processing
python scripts/download_data.py --parallel

# Optimize data loading
python scripts/train.py --num-workers 4

# Use memory mapping
python scripts/train.py --memory-mapped
```

## Best Practices

### 1. Code Organization

- Keep code modular and testable
- Use type hints and docstrings
- Follow PEP 8 style guidelines
- Implement proper error handling

### 2. Data Management

- Version control data with DVC
- Document data lineage
- Implement data validation
- Monitor data quality

### 3. Model Management

- Track experiments with MLflow
- Version control models
- Implement model validation
- Monitor model drift

### 4. Deployment

- Use containerization for consistency
- Implement health checks
- Monitor performance metrics
- Plan for rollbacks

### 5. Testing

- Write comprehensive unit tests
- Implement integration tests
- Use automated testing
- Monitor test coverage

## Conclusion

This MLOps project provides a complete framework for machine learning operations. It demonstrates best practices in:

- Data versioning and pipeline orchestration
- Experiment tracking and model management
- API development and deployment
- Testing and validation
- Monitoring and observability
- CI/CD automation

The project is designed to be extensible and can be adapted for different machine learning tasks and deployment scenarios.

For more information, refer to the individual component documentation and the main README file. 