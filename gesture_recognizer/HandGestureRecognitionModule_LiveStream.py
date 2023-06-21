import math
import os
from time import sleep, time

import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2

plt.ion()

plt.switch_backend('agg')

fig, ax = plt.subplots()

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def display_image(data, result, fig, ax):
    ax.imshow(data)
    print('2')
    fig.canvas.draw()
    print('3')
    fig.canvas.flush_events()
    print('4')
    #sleep(0.01)
    print('5')
    plt.cla()

def display_image_with_gestures_and_hand_landmarks(image, result):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    image = image.numpy_view()

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = 1
    cols = 1

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows, cols, 1)
    
    dynamic_titlesize = FIGSIZE*SPACING/max(rows, cols) * 40 + 3

    fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))
    
    if result:
        gestures, multi_hand_landmarks_list = result
    else:
        title = 'No hand detected'
        titlesize = dynamic_titlesize

        """Displays one image along with the predicted category name and score."""
        plt.subplot(*subplot)
        
        if len(title) > 0:
            plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
         
        subplot = (subplot[0], subplot[1], subplot[2]+1)
        # Layout.
        plt.tight_layout()
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

        fig.canvas.draw()
        fig.canvas.flush_events()
        sleep(0.1)
        plt.cla()

        return 0


    # Display gestures and hand landmarks.
    title = f"{gestures.category_name} ({gestures.score:.2f})"
    annotated_image = image.copy()

    for hand_landmarks in multi_hand_landmarks_list[0]:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    image = annotated_image
    titlesize = dynamic_titlesize

    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
         
    subplot = (subplot[0], subplot[1], subplot[2]+1)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    fig.canvas.draw()
    fig.canvas.flush_events()
    sleep(0.1)
    plt.cla()

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

        #output_image = cv2.putText(output_image, str(result), org, font, fontScale, color, thickness, cv2.LINE_AA)

        try:
            print(f'gesture recognition result: Accuracy: {result.gestures[0][0].score}, letter: {result.gestures[0][0].category_name}')
            results = (result.gestures[0][0], result.hand_landmarks)
        except:
            print('No hands detected')
            results = ()

        global fig
        global ax

        # Display the result
        display_image_with_gestures_and_hand_landmarks(output_image, results)

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