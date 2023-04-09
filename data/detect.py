from ultralytics import YOLO
import os
import cv2

path = os.getcwd()
print('Opening camera...')
vid = cv2.VideoCapture(0)

print('Loading detection model...')
# Load a custom model
model = YOLO(f'{path}/runs/detect/train6/weights/best.pt')

while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Predict with the model
    results = model(frame)  # predict on an image
    boxes = results[0].boxes
    for box in boxes:
        xyxy = tuple([int(x) for x in box.xyxy[0]])
        frame = cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=(255, 0, 0), thickness=2)
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()