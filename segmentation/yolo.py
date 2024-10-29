import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
if torch.cuda.is_available():
    model.cuda()
model.eval()

class Yolo:
    def __init__(self):
        self.model = model

    def predict(self, image):
        return self.model(image)
