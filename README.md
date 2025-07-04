# Sentiment Analysis MLOps Pipeline

An end-to-end MLOps pipeline for sentiment analysis using DistilBERT.

## Features

- FastAPI REST API for model serving
- Docker containerization
- CI/CD pipeline with GitHub Actions
- Prometheus metrics endpoint
- Comprehensive logging
- Automated testing

## Getting Started

### Prerequisites

- Python 3.10
- Docker
- Git
- curl or Postman (for API testing)

### Installation Options

#### Option 1: Run from GitHub (Local Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/amitaAlEngineer/sentiment-analysis-mlops.git
   cd sentiment-analysis-mlops

2. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows

3. Install dependencies:
   pip install -r requirements.txt

4. Run the API server:
   uvicorn app.main:app --reload

   wait for some time, let the model download the hit the APIs 

### Option 2: Run with Docker
1. Pull the Docker image:
   - docker pull amitakri/sentiment-analysis
2. run using Docker
   - docker run -p 8000:8000 amitakri/sentiment-analysis:latest
3. For development with auto-reload:
   docker build -t sentiment-analysis .
   docker run -p 8000:8000 -v $(pwd):/app sentiment-analysis

### API Documentation
The API will be available at http://localhost:8000

## Endpoints
1. Health Check
- Endpoint: GET /health
- Response:
   ```bash
   {
      "status": "healthy"
   }

2. Predict Sentiment
- Endpoint: POST /predict

- Request Body:
   ```bash
   {
   "text": "Your text to analyze"
   }
- Response :
   ```bash
   {
      "sentiment": "POSITIVE/NEGATIVE",
      "confidence": 0.99,
      "model": "distilbert-base-uncased-finetuned-sst-2-english"
   }

- Error Response (empty text):
   ```bash
   {
      "detail": "Text cannot be empty"
   }