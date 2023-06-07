from ultralytics import YOLO
import os
import cv2
from time import time

def yolo_init(path):

    print('Loading YOLO detection model...')
    nnsize = 'm' # 'n' for nano, 's' for small, 'm' for medium

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (0, 255, 255)
    # Line thickness of 2 px
    thickness = 2

    # Load a custom model
    model = YOLO(f'{path}/hand_detector/yolov8{nnsize}.pt')

    return (model, org, font, fontScale, color, thickness)

def yolo(model, format, frame, prev_timestamp):
    # Predict with the model
    results = model(frame)  # predict on an image
    boxes = results[0].boxes
    crop_coords = []
    for box in boxes:
        xyxy = tuple([int(x) for x in box.xyxy[0]])
        frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=(255, 0, 0), thickness=2)
        crop_coords.append(xyxy)

    # FPS
    if prev_timestamp:
        timestamp = time()
        fps = '{:.0f}'.format(1/(timestamp - prev_timestamp))
        prev_timestamp = timestamp
        frame = cv2.putText(frame, fps, *format, cv2.LINE_AA)
    return (frame, prev_timestamp, crop_coords)

if __name__ == '__main__':

    path = os.getcwd()

    prev_timestamp = time()
    
    print('Opening camera...')
    vid = cv2.VideoCapture(0)

    model, *yolo_format = yolo_init(path)

    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()

        frame, prev_timestamp, crop_coords = yolo(model, yolo_format, frame, prev_timestamp)

        if crop_coords:
            for box in crop_coords:
                annotated_image = frame[box[1]:box[3], box[0]:box[2]]
                cv2.imshow('frame', annotated_image)
        else:
            # Display the resulting frame
            cv2.imshow('frame', frame)

        # the 'q' button is quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()