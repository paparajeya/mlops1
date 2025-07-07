"""
FastAPI Application for MNIST Model Inference

This module provides a REST API for MNIST digit classification with monitoring and validation.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY
import yaml

# Import model components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mnist_cnn import MNISTCNN

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
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

# Load configuration
def load_config() -> Dict[str, Any]:
    """Load API configuration from YAML file."""
    config_path = "configs/api.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            "server": {"host": "0.0.0.0", "port": 8000},
            "model": {"path": "models/best_model.pth", "device": "cpu"},
            "monitoring": {"enable_metrics": True}
        }

config = load_config()

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Classification API",
    description="A REST API for MNIST digit classification using CNN",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("security", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
if config.get("monitoring", {}).get("enable_metrics", True):
    REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
    REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
    PREDICTION_COUNT = Counter('predictions_total', 'Total predictions made')
    PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')

# Global model instance
model = None
model_info = None

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    image: List[List[float]] = Field(..., description="28x28 grayscale image as 2D array")
    confidence_threshold: Optional[float] = Field(0.5, description="Minimum confidence threshold")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Predicted digit (0-9)")
    confidence: float = Field(..., description="Confidence score")
    probabilities: List[float] = Field(..., description="Class probabilities")
    processing_time: float = Field(..., description="Processing time in seconds")

class ModelInfo(BaseModel):
    model_name: str
    architecture: str
    input_size: List[int]
    num_classes: int
    total_parameters: int
    device: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

def load_model():
    """Load the trained model."""
    global model, model_info
    
    try:
        model_path = config["model"]["path"]
        device = config["model"]["device"]
        
        if not os.path.exists(model_path):
            logger.error("Model file not found", path=model_path)
            return False
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model instance
        model_config = checkpoint.get('config', {}).get('model', {})
        model = MNISTCNN(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Get model info
        model_info = checkpoint.get('model_info', {})
        model_info['device'] = device
        
        logger.info("Model loaded successfully", 
                   path=model_path, 
                   device=device,
                   model_info=model_info)
        return True
        
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        return False

def preprocess_image(image_data: List[List[float]]) -> torch.Tensor:
    """Preprocess image data for model inference."""
    try:
        # Convert to numpy array
        image_array = np.array(image_data, dtype=np.float32)
        
        # Ensure correct shape (28x28)
        if image_array.shape != (28, 28):
            raise ValueError(f"Expected image shape (28, 28), got {image_array.shape}")
        
        # Normalize to [0, 1] if not already
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Apply MNIST normalization
        image_array = (image_array - 0.1307) / 0.3081
        
        # Convert to tensor and add batch and channel dimensions
        tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
        
        return tensor
        
    except Exception as e:
        logger.error("Failed to preprocess image", error=str(e))
        raise ValueError(f"Invalid image data: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting MNIST Classification API")
    
    # Load model
    if not load_model():
        logger.warning("Model not loaded - some endpoints may not work")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "MNIST Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    import datetime
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.datetime.now().isoformat()
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if model_info is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(**model_info)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction on an MNIST image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(request.image)
        input_tensor = input_tensor.to(model.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        # Convert to Python types
        prediction = prediction.item()
        confidence = confidence.item()
        probabilities = probabilities.squeeze().cpu().numpy().tolist()
        
        processing_time = time.time() - start_time
        
        # Update metrics
        if config.get("monitoring", {}).get("enable_metrics", True):
            PREDICTION_COUNT.inc()
            PREDICTION_LATENCY.observe(processing_time)
        
        # Check confidence threshold
        if confidence < request.confidence_threshold:
            logger.warning("Low confidence prediction", 
                         prediction=prediction, 
                         confidence=confidence,
                         threshold=request.confidence_threshold)
        
        logger.info("Prediction made", 
                   prediction=prediction, 
                   confidence=confidence,
                   processing_time=processing_time)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make batch predictions on multiple MNIST images."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    results = []
    
    try:
        for i, request in enumerate(requests):
            # Preprocess image
            input_tensor = preprocess_image(request.image)
            input_tensor = input_tensor.to(model.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, prediction = torch.max(probabilities, 1)
            
            # Convert to Python types
            prediction = prediction.item()
            confidence = confidence.item()
            probabilities = probabilities.squeeze().cpu().numpy().tolist()
            
            results.append({
                "index": i,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities
            })
        
        processing_time = time.time() - start_time
        
        logger.info("Batch prediction completed", 
                   batch_size=len(requests),
                   processing_time=processing_time)
        
        return {
            "predictions": results,
            "batch_size": len(requests),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if not config.get("monitoring", {}).get("enable_metrics", True):
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    
    return JSONResponse(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )

@app.middleware("http")
async def add_process_time_header(request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Update metrics
    if config.get("monitoring", {}).get("enable_metrics", True):
        REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
        REQUEST_LATENCY.observe(process_time)
    
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", 
                error=str(exc), 
                path=request.url.path,
                method=request.method)
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config["server"]["host"],
        port=config["server"]["port"],
        reload=config["server"].get("reload", False)
    ) 