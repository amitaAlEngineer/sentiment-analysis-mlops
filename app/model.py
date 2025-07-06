from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification,Trainer, TrainingArguments
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram
import time
import torch
from typing import Dict, Any
from prometheus_client import Counter, Histogram
import time
import torch
import datasets
import argparse
import os
from datasets import load_dataset


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

TRAINING_COUNTER = Counter(
    'model_retrainings_total',
    'Total number of model retrainings'
)

# Increment in retrain method
TRAINING_COUNTER.inc()

class SentimentModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.retrained_model_path = "./retrained_model"
        self.load_model()

    def load_model(self):
        """Load the DistilBERT model"""
        try:
            # Check GPU availability
            device = 0 if torch.cuda.is_available() else -1
            torch_dtype = torch.float16 if device == 0 else torch.float32
            
            if os.path.exists(self.retrained_model_path):
                model_name = self.retrained_model_path
                logger.info("Loading retrained model")
            else:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                logger.info("Loading base pretrained model")
            
            # model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            self.model_name = model_name
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
        
    def retrain(self, dataset_name="imdb", output_dir="./retrained_model"):
        """Retrain the model on new data"""
        try:
            logger.info("Starting retraining process")
            TRAINING_COUNTER.inc()

            # Load dataset
            dataset = load_dataset(dataset_name)
            
            # Tokenize (with progress bar)
            tokenized = dataset.map(
                lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True),
                batched=True,
                load_from_cache_file=False  # Disable caching to prevent hangs
            )
            # Limit to speed up retraining
            small_train_dataset = tokenized["train"].shuffle(seed=42).select(range(300))
            small_eval_dataset = tokenized["test"].shuffle(seed=42).select(range(100))
            
            # Training setup
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=8,
                eval_strategy="steps",  # Changed from "epoch"
                eval_steps=10,
                save_strategy="steps",
                logging_dir="./logs",
                logging_steps=10,
                disable_tqdm=True,  # Disable progress bars in CI
                report_to="none",    # Disable external reporting
                seed=42
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=small_train_dataset,
                eval_dataset=small_eval_dataset
            )
            
            # Train with explicit logging
            logger.info("Training started")
            trainer.train()
            logger.info("Training completed")
            
            # Save model
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info("Model saved")
            # Reload the retrained model
            self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
            self.pipeline = pipeline(
                task="sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Model retraining completed successfully")
            return True
        except Exception as e:
            logger.error(f"Retraining failed: {str(e)}")
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
                "model": self.model_name
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise