from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from app.process_frame import ProcessFrame
from app.utils import get_mediapipe_pose
from app.thresholds import get_thresholds_beginner, get_thresholds_pro

# Initialize FastAPI
app = FastAPI(title="Pose Correction API (Webcam Mode)")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all (or restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pose model and processor
pose = get_mediapipe_pose()

@app.get("/")
def root():
    return {"message": "Pose Correction API is running in webcam mode!"}

@app.post("/predict_webcam/")
async def predict_webcam(request: Request):
    """Receives a base64 image (from webcam frame) and mode (beginner/pro)"""
    data = await request.json()
    frame_b64 = data.get("frame")
    mode = data.get("mode", "beginner").lower()

    if not frame_b64:
        return {"error": "No frame received"}

    # Convert base64 string to OpenCV frame
    frame_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Select thresholds based on mode
    thresholds = get_thresholds_pro() if mode == "pro" else get_thresholds_beginner()

    # Process frame
    processor = ProcessFrame(thresholds=thresholds)
    _, feedback = processor.process(frame, pose)

    return {"feedback": feedback or "No feedback detected"}
