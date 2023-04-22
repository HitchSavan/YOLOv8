from ultralytics import YOLO
import os

if __name__ == '__main__':
    path = os.getcwd()

    print('Loading detection model...')
    nnsize = 'm' # 'n' for nano, 's' for small, 'm' for medium

    # Load a custom model
    model = YOLO(f'{path}/data/yolov8{nnsize}.pt')

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category