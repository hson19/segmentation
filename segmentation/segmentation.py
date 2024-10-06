import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import base64
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = FastAPI()


# https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
def create_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg), cfg

max_workers = 1 # You can adjust this value
executor = ProcessPoolExecutor(max_workers=max_workers, initializer=create_predictor)
# Create a thread pool

class ImageRequest(BaseModel):
    image: str

def process_single_image(image_data: str) -> str:
    predictor, cfg = create_predictor()
    # Decode base64 image
    image_data = base64.b64decode(image_data)
    
    # Convert to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Make prediction
    outputs = predictor(frame)

    # Visualize results
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Convert the result back to an image
    result_image = out.get_image()[:, :, ::-1]
    
    # Encode the result image to base64
    _, buffer = cv2.imencode('.jpg', result_image)
    result_image_data = base64.b64encode(buffer).decode('utf-8')

    return result_image_data

@app.post('/process_image')
async def process_image(request: ImageRequest):
    try:
        # Submit the task to the thread pool
        future = executor.submit(process_single_image, request.image)
        # Wait for the result
        result_image_data = future.result()
        return {'image': result_image_data}
    except Exception as e:
        import logging
        logging.error(f"Exception processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Add a new endpoint to set the number of workers
@app.post('/set_max_workers')
async def set_max_workers(max_workers: int):
    global executor
    executor.shutdown(wait=True)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    return {"message": f"Max workers set to {max_workers}"}
