import os
from time import time
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import sys

sys.path.append('../gesture_recognition/utils')

from movement_vector import Vector

class GestureRecognizerLiveStream():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles


    buffer_size = 6
    movement_buffer = deque(maxlen=buffer_size)
    movement_frames_loss = 0
    Q = deque(maxlen=15)
    labels = deque(maxlen=Q.maxlen)
    labels_dict = {}
    labels_score_dict = {}
    vectors = deque(maxlen=Q.maxlen-1)
    argcoses_means = deque(maxlen=Q.maxlen)
    prev_timestamp = 0.0

    def display_image_with_gestures_and_hand_landmarks(self, image, result):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (0, 255, 255) # RGB
        try:
            print(f'gesture recognition result: Accuracy: {result.gestures[0][0].score}, letter: {result.gestures[0][0].category_name}')
            letter = result.gestures[0][0].category_name
            score = result.gestures[0][0].score
        except:
            print('No hands detected')
            letter = ''
            score = 0

        
        hand_landmarks_list = result.hand_landmarks
        handedness_list = result.handedness
        
        annotated_image = np.copy(image)

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
            
        # FPS
        timestamp = time()
        fps = f'{int(1/(timestamp - self.prev_timestamp))}'
        self.prev_timestamp = timestamp
        print(f'FPS: {fps}')
        annotated_image = cv2.putText(annotated_image, fps, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 
                                    1, (0, 255, 255), 2, cv2.LINE_AA)

        self.labels.append(letter)

        if not self.labels_score_dict.__contains__(letter):
            self.labels_score_dict[letter] = score
        else:
            self.labels_score_dict[letter] += score

        unique, counts = np.unique(self.labels, return_counts=True)
        labels_dict = dict(zip(unique, counts))

        self.Q.append(self.labels_score_dict[letter] / labels_dict[letter])

        results = np.array(self.Q)
        i = np.argmax(results)
        
        result_letter = self.labels[i]

        if position is not None:
            self.movement_buffer.append(position)
            self.movement_frames_loss = 0

        else:
            self.movement_frames_loss += 1

        if self.movement_frames_loss > 3 and len(self.movement_buffer):
            self.movement_buffer.clear()
            self.vectors.clear()

        if len(self.movement_buffer) > 1:
            for pos in range(1, len(self.movement_buffer)):
                self.vectors.append(Vector(self.movement_buffer[pos-1][0], self.movement_buffer[pos-1][1],
                                           self.movement_buffer[pos][0], self.movement_buffer[pos][1]))

        if len(self.argcoses_means):
            self.argcoses_means.clear()

        if len(self.vectors) > 0:
            for i, vector in enumerate(self.vectors):
                vector.draw(annotated_image, (255, 0, 0), 3)
                if i > 0 and self.vectors[i-1].length() > 4 and vector.length() > 4:
                    prev_vector = self.vectors[i-1]
                    self.argcoses_means.append(prev_vector.getCos(vector))

            np_argcoses = np.array(self.argcoses_means)
            
            # movement_label = 'straight' if np_argcoses.mean() > 0.94 else ''
            # movement_label = 'circle' if (np.any(np.where(np_argcoses > 0.7)) and movement_label == '') else movement_label

            # annotated_image = cv2.putText(annotated_image, movement_label, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        annotated_image = cv2.putText(annotated_image, result_letter, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('frame', annotated_image)

    def gesture_recognizer_init(self, path):

        epochs = 500

        try:
            model_path = f'{path}/gesture_recognizer/model_{epochs}epochs/gesture_recognizer.task'
        except:
            model_path = f'{path}\\gesture_recognizer\\model_{epochs}epochs\\gesture_recognizer.task'

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
        def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
            # Display the result
            self.display_image_with_gestures_and_hand_landmarks(output_image.numpy_view(), result)

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result,
            num_hands=2)

        return (GestureRecognizer, options, gest_format)

if __name__ == '__main__':

    path = os.getcwd()

    prev_timestamp = time()

    print('Opening camera...')
    vid = cv2.VideoCapture(0)

    gest_recogn = GestureRecognizerLiveStream()

    GestureRecognizer, options, gest_format = gest_recogn.gesture_recognizer_init(path)

    timestamp = 0

    with GestureRecognizer.create_from_options(options) as recognizer:
        while(True):
            # Capture the video frame by frame
            ret, frame = vid.read()

            if not ret:
                continue

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            recognizer.recognize_async(mp_image, timestamp)
            timestamp += 1

            # the 'q' button is quitting button
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After the loop release the cap object
    vid.release()
    cv2.destroyAllWindows()