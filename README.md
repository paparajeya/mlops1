# MLOps Project: MNIST Classification

A comprehensive MLOps project demonstrating best practices for machine learning operations, including data versioning, experiment tracking, testing, containerization, API services, and CI/CD pipelines.

## 🚀 Project Overview

This project implements a complete MLOps pipeline for MNIST digit classification using:
- **Data Versioning**: DVC for data and model versioning
- **Experiment Tracking**: MLflow for experiment management
- **Testing**: Unit tests and validation pipelines
- **Containerization**: Docker for reproducible environments
- **API Service**: FastAPI for model inference
- **CI/CD**: GitHub Actions for automated workflows
- **Monitoring**: Basic performance monitoring

## 📁 Project Structure

```
mlops-mnist/
├── data/                      # Data storage (gitignored)
├── models/                    # Trained models (gitignored)
├── notebooks/                 # Jupyter notebooks
├── src/
│   ├── data/                 # Data processing modules
│   ├── models/               # ML model components
│   ├── api/                  # FastAPI application
│   ├── utils/                # Utility functions
│   └── tests/                # Unit tests
├── configs/                  # Configuration files
├── scripts/                  # Utility scripts
├── .github/                  # GitHub Actions workflows
├── docker/                   # Docker configurations
├── docs/                     # Documentation
├── requirements.txt           # Python dependencies
├── dvc.yaml                  # DVC pipeline configuration
├── .dvcignore               # DVC ignore patterns
├── Dockerfile               # Main application Dockerfile
├── docker-compose.yml       # Multi-service orchestration
└── README.md               # This file
```

## 🛠️ Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Git
- DVC
- MLflow

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd mlops-mnist
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Initialize DVC
```bash
# Initialize DVC
dvc init

# Add remote storage (example with local storage)
dvc remote add -d myremote ./data

# Pull initial data
dvc pull
```

### 4. Set Up MLflow
```bash
# Start MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000
```

## 🚀 Quick Start

### 1. Data Pipeline
```bash
# Download and process MNIST data
python scripts/download_data.py
dvc add data/mnist/
dvc push
```

### 2. Training Pipeline
```bash
# Train model with default parameters
python scripts/train.py

# Train with custom parameters
python scripts/train.py --learning-rate 0.01 --epochs 10
```

### 3. Experiment Tracking
```bash
# View MLflow experiments
mlflow ui

# Compare experiments
python scripts/compare_experiments.py
```

### 4. Testing
```bash
# Run unit tests
pytest src/tests/

# Run validation pipeline
python scripts/validate_model.py
```

### 5. API Service
```bash
# Start FastAPI service
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Test API endpoints
python scripts/test_api.py
```

### 6. Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individual services
docker build -t mlops-mnist .
docker run -p 8000:8000 mlops-mnist
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file:
```bash
MLFLOW_TRACKING_URI=http://localhost:5000
DVC_REMOTE_URL=./data
MODEL_REGISTRY_PATH=./models
API_HOST=0.0.0.0
API_PORT=8000
```

### DVC Configuration
```bash
# Configure DVC remote
dvc remote add -d myremote ./data

# Set up data pipeline
dvc repro
```

## 📊 Monitoring and Logging

### MLflow Tracking
- Access MLflow UI: http://localhost:5000
- View experiments, metrics, and artifacts
- Compare model performances

### API Monitoring
- Health check: `GET /health`
- Model info: `GET /model/info`
- Prediction endpoint: `POST /predict`

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest src/tests/test_models.py
pytest src/tests/test_api.py
```

### Integration Tests
```bash
# Test complete pipeline
python scripts/test_pipeline.py

# Test API endpoints
python scripts/test_api_integration.py
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows for:
- Automated testing on pull requests
- Model training and evaluation
- Docker image building and pushing
- Deployment automation

### Workflow Triggers
- **Pull Request**: Runs tests and validation
- **Push to main**: Trains model and updates registry
- **Release**: Deploys to production

## 📈 Performance Metrics

The project tracks:
- **Model Performance**: Accuracy, precision, recall, F1-score
- **Training Metrics**: Loss curves, learning rate schedules
- **API Performance**: Response times, throughput
- **System Metrics**: Memory usage, CPU utilization

## 🔍 Troubleshooting

### Common Issues

1. **DVC Issues**
   ```bash
   # Reset DVC cache
   dvc checkout --force
   dvc pull
   ```

2. **MLflow Connection Issues**
   ```bash
   # Check MLflow server
   curl http://localhost:5000/health
   ```

3. **Docker Issues**
   ```bash
   # Clean Docker cache
   docker system prune -a
   ```

## 📚 Documentation

- [Data Pipeline Documentation](docs/data_pipeline.md)
- [Model Training Guide](docs/training.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Happy MLOps! 🚀**
