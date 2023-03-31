import json
import numpy as np
from PIL import Image
import io
import cv2

import uvicorn
from fastapi import FastAPI, File, Response
from fastapi.responses import FileResponse

from src.racoon_detection import RacoonDetection

racoon_detection = RacoonDetection("src/racoon_detection.onnx")

app = FastAPI()

@app.get("/")
def detect():
    return "Racoon Detection With YOLOv8n"


@app.post("/detect")
def detect(file: bytes = File(...)):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = np.array(img)
    results = racoon_detection(img)
    return results


@app.post("/draw-box-detect")
def draw_box_detect(file: bytes = File(...)):
    img = Image.open(io.BytesIO(file)).convert("RGB")
    img = np.array(img)
    img = racoon_detection.draw_box_detect(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("response.png", img)
    return FileResponse("response.png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)