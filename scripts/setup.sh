#!/bin/bash

# MLOps Project Setup Script
# This script sets up the complete MLOps environment

set -e  # Exit on any error

echo "🚀 Setting up MLOps Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.12+ first."
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is not installed. Please install pip first."
        exit 1
    fi
}

# Check if Docker is installed
check_docker() {
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version 2>&1 | awk '{print $3}' | sed 's/,//')
        print_success "Docker $DOCKER_VERSION found"
    else
        print_warning "Docker is not installed. Some features may not work."
    fi
}

# Check if Docker Compose is installed
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version 2>&1 | awk '{print $3}' | sed 's/,//')
        print_success "Docker Compose $COMPOSE_VERSION found"
    else
        print_warning "Docker Compose is not installed. Some features may not work."
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Initialize DVC
init_dvc() {
    print_status "Initializing DVC..."
    if [ ! -d ".dvc" ]; then
        dvc init
        print_success "DVC initialized"
    else
        print_warning "DVC already initialized"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p data/mnist
    mkdir -p models
    mkdir -p logs
    mkdir -p notebooks
    mkdir -p docs
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    print_success "Directories created"
}

# Download data
download_data() {
    print_status "Downloading MNIST dataset..."
    python scripts/download_data.py
    print_success "Data downloaded"
}

# Initialize MLflow
init_mlflow() {
    print_status "Setting up MLflow..."
    # Create MLflow directories
    mkdir -p mlruns
    mkdir -p mlartifacts
    print_success "MLflow directories created"
}

# Build Docker image
build_docker() {
    if command -v docker &> /dev/null; then
        print_status "Building Docker image..."
        docker build -t mlops-mnist:latest .
        print_success "Docker image built"
    else
        print_warning "Docker not available, skipping Docker build"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    python -m pytest src/tests/ -v
    print_success "Tests completed"
}

# Create .env file
create_env_file() {
    print_status "Creating .env file..."
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000

# DVC Configuration
DVC_REMOTE_URL=./data

# Model Configuration
MODEL_REGISTRY_PATH=./models

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF
        print_success ".env file created"
    else
        print_warning ".env file already exists"
    fi
}

# Create .gitignore
create_gitignore() {
    print_status "Creating .gitignore file..."
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Data and models
data/
models/
mlruns/
mlartifacts/

# Environment variables
.env
.env.local
.env.production

# DVC
.dvc/cache/
.dvc/tmp/

# Testing
.coverage
.pytest_cache/
htmlcov/

# Jupyter
.ipynb_checkpoints/

# Docker
.dockerignore
EOF
        print_success ".gitignore file created"
    else
        print_warning ".gitignore file already exists"
    fi
}

# Main setup function
main() {
    echo "=========================================="
    echo "🚀 MLOps Project Setup"
    echo "=========================================="
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    check_python
    check_pip
    check_docker
    check_docker_compose
    
    # Create virtual environment
    create_venv
    
    # Activate virtual environment
    activate_venv
    
    # Install dependencies
    install_dependencies
    
    # Create directories
    create_directories
    
    # Initialize DVC
    init_dvc
    
    # Initialize MLflow
    init_mlflow
    
    # Create configuration files
    create_env_file
    create_gitignore
    
    # Download data
    download_data
    
    # Build Docker image
    build_docker
    
    # Run tests
    run_tests
    
    echo "=========================================="
    print_success "🎉 Setup completed successfully!"
    echo "=========================================="
    echo ""
    echo "📋 Next steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000"
    echo "3. Train model: python scripts/train.py"
    echo "4. Start API: uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    echo "5. Or use Docker: docker-compose up --build"
    echo ""
    echo "📚 Documentation: README.md"
    echo "🔗 MLflow UI: http://localhost:5000"
    echo "🔗 API Docs: http://localhost:8000/docs"
    echo "=========================================="
}

# Run main function
main "$@" 