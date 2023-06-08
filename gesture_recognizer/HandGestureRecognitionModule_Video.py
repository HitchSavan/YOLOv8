import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import cv2
from time import time


def draw_landmarks_on_image(rgb_image, detection_result, MARGIN, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS):
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

def gesture_recognizer_init(path):
    print("Loading gesture detection model...")
    try:
        model_path = f'{path}/gesture_recognizer/model/gesture_recognizer.task'
    except:
        model_path = f'{path}\\gesture_recognizer\\model\\gesture_recognizer.task'

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (0, 255, 255) # RGB
    gest_format = (MARGIN, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS)

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)
    
    return (GestureRecognizer, options, gest_format)

def recognize_gesture(recognizer, gest_format, frame, timestamp):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    gesture_recognition_result = recognizer.recognize_for_video(mp_image, timestamp)

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), gesture_recognition_result, *gest_format)

    return annotated_image


if __name__ == '__main__':

    path = os.getcwd()
    video_name = 'gori.mp4'

    print('Opening video...')
    vidObj = cv2.VideoCapture(f'{path}/{video_name}')

    GestureRecognizer, options, gest_format = gesture_recognizer_init(path)

    frames = []

    with GestureRecognizer.create_from_options(options) as recognizer:
        if vidObj.isOpened():
            count = 0
            success = True
            while success:
                # Capture the video frame by frame
                success, image = vidObj.read()
                
                if success:

                    timestamp = vidObj.get(cv2.CAP_PROP_POS_MSEC)
                    count += 1

                    result = recognize_gesture(recognizer, gest_format, image, int(timestamp))
                    
                    frames.append(result)

                    # Display the resulting frame
                    # cv2.imshow('frame', result)
                    print(f'Processed frame {count}')

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break  
            
            h, w, layers = result.shape
            size = (w, h)

            out = cv2.VideoWriter(f'recognized_{video_name}',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
            for i in range(len(frames)):
                out.write(frames[i])
            out.release()

            vidObj.release()

            # Destroy all the windows
            cv2.destroyAllWindows()