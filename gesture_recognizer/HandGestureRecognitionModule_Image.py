from math import sqrt
import os
from time import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

class Point:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    def __add__(self, _pt):
        return Point(self.x + _pt.x, self.y + _pt.y)

    def __sub__(self, _pt):
        return Point(self.x - _pt.x, self.y - _pt.y)
    
    def __gt__(self, _pt):
        return True if self.x > _pt.x else (self.y > _pt.y if self.x == _pt.x else False)
    
    def __lt__(self, _pt):
        return self.__gt__(_pt, self)

class Vector:
    def __init__(self, _start, _end):
        self.start = _start
        self.end = _end
        self.coords = _end - _start

    def __init__(self, _start_x, _start_y, _end_x, _end_y):
        self.start = Point(_start_x, _start_y)
        self.end = Point(_end_x, _end_y)
        self.coords = self.end - self.start

    # codirectional vectors
    def __eq__(self, other):
        if ((self.coords.x / other.coords.x) == (self.coords.y / other.coords.y)):
            return 0 < (self.coords.x * other.coords.x + self.coords.y * other.coords.y)
        return False

    def distance(self, r):
        return sqrt(pow(self.coords.x - r.coords.x, 2) + pow(self.coords.y - r.coords.y, 2))

    def length(self):
        return sqrt(pow(self.coords.x, 2) + pow(self.coords.y, 2))

    # if positive --> <90 degrees
    def getScalarMult(self, r):
        return (self.coords.x * r.coords.x) + (self.coords.y * r.coords.y)

    # if cos > 0.7 --> <45 degrees
    # > 0,866 --> <30
    def getCos(self, r):
        return (self.getScalarMult(r) / (self.length() * r.length()))

    def update(self, newCoords):
        self.coords = newCoords
        self.end = self.start + newCoords

    def draw(self, image, colour, thickness=3):
        cv2.arrowedLine(image, (self.start.x, self.start.y), (self.end.x, self.end.y), colour, thickness)

def draw_landmarks_on_image(rgb_image, recogntion_result, MARGIN, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS):
    try:
        print(f'gesture recognition result: Accuracy: {recogntion_result.gestures[0][0].score}, letter: {recogntion_result.gestures[0][0].category_name}')
        letter = recogntion_result.gestures[0][0].category_name
        score = recogntion_result.gestures[0][0].score
    except:
        print('No hands detected')
        letter = ''
        score = 0
    
    hand_landmarks_list = recogntion_result.hand_landmarks
    handedness_list = recogntion_result.handedness
    annotated_image = np.copy(rgb_image)

    position = None

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
        
        position = (int(sum([coord * width for coord in x_coordinates])/len([coord * width for coord in x_coordinates])),
                    int(sum([coord * height for coord in y_coordinates])/len([coord * height for coord in y_coordinates])))

    return annotated_image, letter, position, score

def gesture_recognizer_init(path):

    try:
        model_path = f'{path}/gesture_recognizer/model/gesture_recognizer.task'
    except:
        model_path = f'{path}\\gesture_recognizer\\model\\gesture_recognizer.task'

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (0, 255, 255) # RGB
    gest_format = (MARGIN, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS)

    with open(model_path, 'rb') as f:
        model = f.read()

    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE)

    return (GestureRecognizer, options, gest_format)

def recognize_gesture(recognizer, gest_format, frame, prev_timestamp):

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    recognition_result = recognizer.recognize(mp_image)
    annotated_image, letter, position, score = draw_landmarks_on_image(mp_image.numpy_view(), recognition_result, *gest_format)
    
    # FPS
    timestamp = time()
    fps = f'{int(1/(timestamp - prev_timestamp))}'
    prev_timestamp = timestamp
    print(f'FPS: {fps}')
    annotated_image = cv2.putText(annotated_image, fps, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 
                    1, (0, 255, 255), 2, cv2.LINE_AA)
    
    return annotated_image, prev_timestamp, position, letter, score

if __name__ == '__main__':

    path = os.getcwd()

    prev_timestamp = time()

    print('Opening camera...')
    vid = cv2.VideoCapture(0)

    GestureRecognizer, options, gest_format = gesture_recognizer_init(path)

    buffer_size = 6
    movement_buffer = deque(maxlen=buffer_size)
    movement_frames_loss = 0
    Q = deque(maxlen=15)
    labels = deque(maxlen=Q.maxlen)
    labels_dict = {}
    labels_score_dict = {}
    vectors = deque(maxlen=Q.maxlen-1)
    argcoses_means = deque(maxlen=Q.maxlen)

    with GestureRecognizer.create_from_options(options) as recognizer:
        while(True):
            # Capture the video frame by frame
            ret, frame = vid.read()

            if not ret:
                continue

            annotated_image, prev_timestamp, position, letter, score = recognize_gesture(recognizer, gest_format, frame, prev_timestamp)

            labels.append(letter)
            if not labels_score_dict.__contains__(letter):
                labels_score_dict[letter] = score
            else:
                labels_score_dict[letter] += score

            unique, counts = np.unique(labels, return_counts=True)
            labels_dict = dict(zip(unique, counts))

            Q.append(labels_score_dict[letter] / labels_dict[letter])

            results = np.array(Q)
            i = np.argmax(results)
            
            result_letter = labels[i]

            if position is not None:
                movement_buffer.append(position)
                movement_frames_loss = 0

            else:
                movement_frames_loss += 1

            if movement_frames_loss > 3 and len(movement_buffer):
                movement_buffer.clear()
                vectors.clear()

            if len(movement_buffer) > 1:
                for pos in range(1, len(movement_buffer)):
                    vectors.append(Vector(movement_buffer[pos-1][0], movement_buffer[pos-1][1], movement_buffer[pos][0], movement_buffer[pos][1]))

            if len(argcoses_means):
                argcoses_means.clear()

            if len(vectors) > 0:
                for i, vector in enumerate(vectors):
                    vector.draw(annotated_image, (255, 0, 0), 3)
                    if i > 0 and vectors[i-1].length() > 4 and vector.length() > 4:
                        prev_vector = vectors[i-1]
                        argcoses_means.append(prev_vector.getCos(vector))

                np_argcoses = np.array(argcoses_means)
                
                movement_label = 'straight' if np_argcoses.mean() > 0.94 else ''
                movement_label = 'circle' if (np.any(np.where(np_argcoses > 0.7)) and movement_label == '') else movement_label

                annotated_image = cv2.putText(annotated_image, movement_label, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            annotated_image = cv2.putText(annotated_image, result_letter, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', annotated_image)

            # the 'q' button is quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()