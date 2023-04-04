from ultralytics import YOLO
import os
path = os.getcwd()

# Load a YOLOv8 model from a pre-trained weights file
model = YOLO('yolov8m.pt')

# Run MODE mode using the custom arguments ARGS (guess TASK)
model.train(data=f'{path}/ultralytics/data/hand.yaml', epochs=100, imgsz=640)