import mediapipe as mp
from pathlib import Path
import os
import cv2

if __name__ == '__main__':

    path = os.getcwd()
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = HandLandmarkerOptions(
        base_options=BaseOptions(delegate=BaseOptions.Delegate.GPU,
                                model_asset_path=os.path.join('dataset_utils', 'hand_landmarker.task')),
        running_mode=VisionRunningMode.IMAGE)

    dataset_src_folder = os.path.join(path, '..', 'Datasets', 'SLOVO_sign_dataset', 'slovo', 'letters')
    dataset_src_folder = Path(dataset_src_folder)
    dataset_parent_folder = dataset_src_folder.parent.parent

    with HandLandmarker.create_from_options(options) as landmarker:
        for dirpath, dirnames, filenames in os.walk(dataset_src_folder):
            for file in filenames:
                filepath = os.path.join(dirpath, file)
                img = cv2.imread(filepath)
                print(filepath)

                mp_image = mp.Image.create_from_file(filepath)
                hand_landmarker_result = landmarker.detect(mp_image)
                if len(hand_landmarker_result.handedness) == 0:
                    continue
                else:
                    
                    score = hand_landmarker_result.handedness[0][0].score
                    hand_landmarks_list = hand_landmarker_result.hand_landmarks

                    height, width, _ = img.shape
                    for idx in range(len(hand_landmarks_list)):
                        hand_landmarks = hand_landmarks_list[idx]
                        x_coordinates = [landmark.x for landmark in hand_landmarks]
                        y_coordinates = [landmark.y for landmark in hand_landmarks]
                        x_min = int(min(x_coordinates) * width)
                        y_min = int(min(y_coordinates) * height)
                        x_max = int(max(x_coordinates) * width)
                        y_max = int(max(y_coordinates) * height)

                        box_width = x_max - x_min
                        box_height = y_max - y_min

                        y_min = int(y_min - box_height/2.5)
                        y_min = y_min if y_min > 0 else 0

                        y_max = int(y_max + box_height/2.5)
                        y_max = y_max if y_max < height else height-1

                        x_min = int(x_min - box_width/2.5)
                        x_min = x_min if x_min > 0 else 0

                        x_max = int(x_max + box_width/2.5)
                        x_max = x_max if x_max < width else width-1

                        cropped_image = img[y_min:y_max, x_min:x_max]
                        new_filepath = os.path.join(dataset_parent_folder, 'slovo_cropped_mediapipe', 'letters',
                                                    Path(dirpath).name, Path(filepath).stem + '_' + str(idx) + '.jpg')
                        if not os.path.exists(Path(new_filepath).parent):
                            os.makedirs(Path(new_filepath).parent)
                        print(new_filepath)
                        cv2.imwrite(new_filepath, cropped_image)