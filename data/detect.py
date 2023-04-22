from ultralytics import YOLO
import os
import cv2
from time import time

path = os.getcwd()
print('Opening camera...')
vid = cv2.VideoCapture(0)

print('Loading detection model...')
nnsize = 'm' # 'n' for nano, 's' for small, 'm' for medium

# Load a custom model
model = YOLO(f'{path}/data/yolov8{nnsize}.pt')

prev_timestamp = time()
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

while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Predict with the model
    results = model(frame)  # predict on an image
    boxes = results[0].boxes
    for box in boxes:
        xyxy = tuple([int(x) for x in box.xyxy[0]])
        frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=(255, 0, 0), thickness=2)

    # FPS
    timestamp = time()
    fps = '{:.0f}'.format(1/(timestamp - prev_timestamp))
    prev_timestamp = timestamp
    frame = cv2.putText(frame, fps, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()