import sys
import os

import cv2
from segmentation import Yolo


yolo = Yolo()

image = cv2.imread('input/input.jpg')
predictions = yolo.predict(image)
cv2.imshow('Predictions', predictions)