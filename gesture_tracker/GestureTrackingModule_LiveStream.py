import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import cv2
import time

path = os.getcwd()
print('Opening camera...')
vid = cv2.VideoCapture(0)
prev_timestamp = time.time()

model_path = f'{path}/gesture_tracker/hand_landmarker.task'

HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    
    annotated_frame = draw_landmarks_on_image(output_image.numpy_view(), result)
    
    # FPS
    global prev_timestamp
    fps = '{:.0f}'.format(1/((timestamp_ms - prev_timestamp)/1000))
    prev_timestamp = timestamp_ms
    print(f'FPS: {fps}')

    annotated_frame = cv2.putText(annotated_frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('frame', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return 0

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (0, 255, 255) # RGB

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

with open(model_path, 'rb') as f:
    model = f.read()

BaseOptions = python.BaseOptions(model_asset_buffer=model)
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions,
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        
        landmarker.detect_async(mp_image, timestamp)
        
        # the 'q' button is quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()