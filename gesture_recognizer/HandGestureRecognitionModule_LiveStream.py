import mediapipe as mp
import os
import cv2
from time import time

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
    GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Create a gesture recognizer instance with the live stream mode:
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (50, 50)
        # fontScale
        fontScale = 1
        # Yellow color in BGR
        color = (0, 255, 255)
        # Line thickness of 2 px
        thickness = 2

        output_image = cv2.putText(output_image, result, org, font, fontScale, color, thickness, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', output_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 0   
        
        print('gesture recognition result: {}'.format(result))


    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    
    return (GestureRecognizer, options)

def recognize_gesture(recognizer, frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp = int(time() * 1000)
    
    recognizer.recognize_async(mp_image, timestamp)


if __name__ == '__main__':

    path = os.getcwd()

    print('Opening camera...')
    vid = cv2.VideoCapture(0)

    GestureRecognizer, options = gesture_recognizer_init(path)

    with GestureRecognizer.create_from_options(options) as recognizer:
        while(True):
            # Capture the video frame by frame
            ret, frame = vid.read()

            recognize_gesture(recognizer, frame)
            
            # the 'q' button is quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()