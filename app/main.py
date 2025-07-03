from fastapi import FastAPI, HTTPException, status
from .model import SentimentModel
from .schemas import SentimentRequest, SentimentResponse
import logging
from prometheus_client import make_asgi_app
from fastapi.middleware.cors import CORSMiddleware
from .logging_conf import configure_logging

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