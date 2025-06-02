"""FastAPI application for biomedical image classification inference."""


import io
import base64
from typing import Dict, List
from contextlib import asynccontextmanager
import logging

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from inference import (build_and_load_model_from_state,
                       pre_process_image, predict,
                       LABELS,
                       PREDICTION_POWER_FILE_NAME
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class ImageRequest(BaseModel):
    """Request model for image prediction."""
    image: str  # base64 encoded image

class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    prediction: int
    confidence: float
    class_name: str

class ClassNamesResponse(BaseModel):
    """Response model for class names."""
    class_names: Dict[int, str]

class PredictivePowerResponse(BaseModel):
    """Response model for predictive power percentages."""
    predictive_power: List[float]

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool


# Global variables
MODEL = None
PREDICTIVE_POWER = None


def load_model():
    """Load the trained model."""
    global MODEL
    try:
        MODEL = build_and_load_model_from_state('model_cpu.pth')
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        return False


def load_predictive_power():
    """Load predictive power from numpy file."""
    global PREDICTIVE_POWER
    try:
        power_array = np.load(PREDICTION_POWER_FILE_NAME)
        # Convert to percentage and round to 2 decimal places
        PREDICTIVE_POWER = np.round(power_array * 100, 2).tolist()
        logger.info("Predictive power loaded successfully")
        return True
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load predictive power: %s", str(e))
        PREDICTIVE_POWER = [0.0, 0.0, 0.0, 0.0]  # Fallback
        return False


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        # Decode base64
        image_data = base64.b64decode(base64_string)
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        # Convert to numpy array
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting up the application...")
    # Load model
    model_loaded = load_model()
    if not model_loaded:
        logger.error("Failed to load model during startup")
    # Load predictive power
    load_predictive_power()
    logger.info("Application startup complete")
    logger.info("APP configuration: %s", app.openapi())
    yield
    # Shutdown
    logger.info("Shutting down the application...")


# Initialize FastAPI app
app = FastAPI(
    title="Biomedical Image Classification API",
    description="API for classifying biomedical images using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "unhealthy",
        model_loaded=MODEL is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(request: ImageRequest):
    """Predict the class of a base64 encoded image."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        # Decode image
        image_array = decode_base64_image(request.image)
        # Preprocess image
        processed_image = pre_process_image(image_array)
        # Make prediction
        prediction_result = predict(MODEL, processed_image)
        # Get prediction (single value)
        if isinstance(prediction_result, np.ndarray):
            predicted_class = int(prediction_result[0])
        else:
            predicted_class = int(prediction_result)
        # Get confidence from predictive power
        confidence = float(PREDICTIVE_POWER[predicted_class]) if PREDICTIVE_POWER else 0.0
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            class_name=LABELS[predicted_class]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}") from e


@app.get("/class-names", response_model=ClassNamesResponse)
async def get_class_names():
    """Get the mapping from class IDs to class names."""
    return ClassNamesResponse(class_names=LABELS)


@app.get("/predictive-power", response_model=PredictivePowerResponse)
async def get_predictive_power():
    """Get the predictive power percentages for each class."""
    return PredictivePowerResponse(predictive_power=PREDICTIVE_POWER or [0.0, 0.0, 0.0, 0.0])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Biomedical Image Classification API",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Image classification (POST)",
            "/class-names": "Get class name mappings",
            "/predictive-power": "Get predictive power percentages",
            "/docs": "API documentation"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
