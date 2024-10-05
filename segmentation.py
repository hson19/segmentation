import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import base64
import json
from flask import Flask, request, jsonify

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = Flask(__name__)

# Set up the predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image data from the request
    data = request.json
    image_data = base64.b64decode(data['image'])
    
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

    return jsonify({'image': result_image_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)