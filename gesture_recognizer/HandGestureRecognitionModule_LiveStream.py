import mediapipe as mp
from matplotlib import pyplot as plt
import os
import cv2
from time import time, sleep
from mediapipe.framework.formats import landmark_pb2
import math

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


def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    # plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Images and labels.
    images = [image.numpy_view() for image in images]

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    
    dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3

    plt.ion()

    if rows < cols:
        fig = plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        fig = plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    if results:
        gestures = [top_gesture for (top_gesture, _) in results]
        multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]
    else:
        subplot = display_one_image(images[0], 'No hand detected', subplot, titlesize=dynamic_titlesize)

        # Layout.
        plt.tight_layout()
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

        fig.canvas.draw()
        fig.canvas.flush_events()
        sleep(1)
        plt.cla()

        return 0


    # Display gestures and hand landmarks.
    for i, (image, gestures) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gestures.category_name} ({gestures.score:.2f})"
        annotated_image = image.copy()

        for hand_landmarks in multi_hand_landmarks_list[i]:
          hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
          hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
          ])

          mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    fig.canvas.draw()
    fig.canvas.flush_events()
    sleep(1)
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

        #output_image = output_image.numpy_view()

        #output_image = cv2.putText(output_image, str(result), org, font, fontScale, color, thickness, cv2.LINE_AA)

        try:
            print(f'gesture recognition result: Accuracy: {result.gestures[0][0].score}, letter: {result.gestures[0][0].category_name}')
            results = [(result.gestures[0][0], result.hand_landmarks)]
        except:
            print('No hands detected')
            results = []

        images = [output_image]

        # Display the result
        display_batch_of_images_with_gestures_and_hand_landmarks(images, results)

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    
    return (GestureRecognizer, options)

def recognize_gesture(recognizer, frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp = int(time() * 1000)
    
    recognizer.reconize_async(mp_image, timestamp)


if __name__ == '__main__':

    path = os.getcwd()

    plt.ion()

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