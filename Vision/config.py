# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
LLM_MODEL      = "llama-3.3-70b-versatile"  # fast + free on Groq

# Camera
CAMERA_INDEX   = 0
CAMERA_WIDTH   = 640
CAMERA_HEIGHT  = 480

# Detection
YOLO_MODEL     = "yolov11n.pt"       
DETECT_CONF    = 0.1

# Server
HOST           = "0.0.0.0"
PORT           = 8000