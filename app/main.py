from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from . import schemas
from . import model_loader
from . import predictor
import logging
from .model_loader import PROJECT_NAME
from .model_loader import ARTIFACT_NAME
from .model_loader import ARTIFACT_ALIAS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application State ---
# A dictionary to hold our model and other state
ml_models = {}

# --- Lifespan Management (FastAPI's startup/shutdown events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    logger.info("Application startup: Loading model...")
    model, device = model_loader.load_model()
    ml_models["model"] = model
    ml_models["device"] = device
    logger.info("Model loaded and ready.")
    
    yield  # The application is now running
    
    # Shutdown: Clear the model from memory
    logger.info("Application shutdown: Clearing model...")
    ml_models.clear()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Cat-Dog Classifier API",
    description="A simple API to classify images as 'cat' or 'dog' using a ResNet50 model.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Cat-Dog Classifier API!"}

@app.get("/health", response_model=schemas.HealthCheckResponse, tags=["General"])
def health_check():
    """Health check endpoint to ensure the service is running."""
    return {"status": "ok"}

@app.post("/predict", response_model=schemas.PredictionResponse, tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, performs inference, and returns the prediction.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        # Read the image content as bytes
        image_bytes = await file.read()
        
        # Get the prediction
        result = predictor.predict_from_bytes(
            model=ml_models["model"],
            device=ml_models["device"],
            image_bytes=image_bytes
        )
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Error processing prediction request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.get("/info",response_model=schemas.ModelInfoResponse,tags=["General"])
def get_model_info():
    """
    Return: information about the deployed model
    """
    return {"project_name":PROJECT_NAME, "model_name":ARTIFACT_NAME, "model_alias":ARTIFACT_ALIAS}