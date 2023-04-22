from ultralytics import YOLO
import os

if __name__ == "__main__":
    path = os.getcwd()
    nnsize = 'n' # 'n' for nano, 's' for small, 'm' for medium
    if nnsize == 'm':
        batch_size = 12
    else:
        batch_size = -1

    # Load a YOLOv8 model from a pre-trained weights file
    model = YOLO(f'yolov8{nnsize}.pt')

    # Run MODE mode using the custom arguments ARGS (guess TASK)
    model.train(data=f'{path}/data/hand.yaml', epochs=100, imgsz=640, batch=batch_size)