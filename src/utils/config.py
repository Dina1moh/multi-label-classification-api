from pathlib import Path
from dotenv import load_dotenv
import os 

load_dotenv(override=True)

# App configuration
APP_NAME = os.getenv("APP_NAME", "Multi-Label Classification")  
VERSION = os.getenv("VERSION", "1.0.0") 
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

if not API_SECRET_KEY:
    raise ValueError("API_SECRET_KEY environment variable is not set")


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_model.pth"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

VOC_CLASSES = [ "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"]
