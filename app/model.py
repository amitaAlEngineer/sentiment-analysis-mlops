from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram
import time
import torch


# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# tokenizer = AutoTokenizer.from_pretrained("DT12the/distilbert-sentiment-analysis")
# model = AutoModelForSequenceClassification.from_pretrained("DT12the/distilbert-sentiment-analysis")


# Set up logging
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNTER = Counter(
    'inference_requests_total', 
    'Total number of inference requests'
)
REQUEST_LATENCY = Histogram(
    'inference_request_latency_seconds',
    'Latency of inference requests'
)
PREDICTION_GAUGE = Histogram(
    'sentiment_prediction_score',
    'Sentiment prediction scores',
    ['sentiment']
)

class SentimentModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Load the DistilBERT model"""
        try:
            # Check GPU availability
            device = 0 if torch.cuda.is_available() else -1
            torch_dtype = torch.float16 if device == 0 else torch.float32
            
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch_dtype,
                device=device,
                truncation=True,
                padding=True
            )
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    @REQUEST_LATENCY.time()
    def predict(self, text: str) -> Dict[str, Any]:
        """Make a prediction using the loaded model"""
        REQUEST_COUNTER.inc()
        start_time = time.time()
        
        # Validate input
        if not text or not text.strip():
            logger.warning("Empty text input received")
            raise ValueError("Text cannot be empty")
        
        try:
            result = self.pipeline(text)[0]
            latency = time.time() - start_time
            
            # Log prediction metrics
            PREDICTION_GAUGE.labels(
                sentiment=result['label']
            ).observe(result['score'])
            
            logger.info(
                f"Prediction completed - Label: {result['label']}, "
                f"Score: {result['score']:.4f}, Latency: {latency:.4f}s"
            )
            
            return {
                "sentiment": result['label'],
                "confidence": float(result['score']),
                "model": "distilbert-base-uncased-finetuned-sst-2-english"
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise