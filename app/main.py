from fastapi import FastAPI, HTTPException, status
from .model import SentimentModel
from .schemas import SentimentRequest, SentimentResponse
import logging
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
from .logging_conf import configure_logging
from fastapi import BackgroundTasks
from typing import Optional
import shutil
import uuid,os

# Configure logging
configure_logging()

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using DistilBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize model
model = SentimentModel()

@app.get("/")
async def health_check():
    return b"WELCOME TO THE SENTIMENT ANALYSIS"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Predict sentiment for given text"""
    try:
        result = model.predict(request.text)
        return result
    except ValueError as e:
        logging.warning(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
        


@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks, dataset_name: Optional[str] = "imdb"):
    """
    Trigger model retraining
    - Uses background task to avoid blocking
    - Returns immediate response with job ID
    """
    job_id = str(uuid.uuid4())
    
    async def train_task():
        try:
            logging.info(f"Starting training job {job_id}")
            model.retrain(dataset_name=dataset_name)
            logging.info(f"Training job {job_id} completed")
        except Exception as e:
            logging.error(f"Training job {job_id} failed: {str(e)}")
    
    background_tasks.add_task(train_task)
    return {"status": "training_started", "job_id": job_id}

@app.post("/reset-model")
async def reset_model():
    """Revert to the original pretrained model"""
    try:
        if os.path.exists(model.retrained_model_path):
            shutil.rmtree(model.retrained_model_path)
            logging.info("Deleted retrained model")
        
        # Reload original model
        model.load_model()
        return {"status": "reset_complete"}
    except Exception as e:
        logging.error(f"Reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))