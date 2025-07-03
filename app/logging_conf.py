import logging
import sys
from logging.handlers import RotatingFileHandler

def configure_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                "app.log",
                maxBytes=1024 * 1024 * 5,  # 5MB
                backupCount=3
            ),
            logging.StreamHandler(sys.stdout)
        ]
    )