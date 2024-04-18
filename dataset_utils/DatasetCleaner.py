import mediapipe as mp
import os
import cv2

path = os.getcwd()
dataset_path = os.path.join(path, '..', 'Datasets', 'SLOVO_sign_dataset', 'slovo', 'letters')

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(delegate=BaseOptions.Delegate.GPU,
                             model_asset_path=os.path.join('dataset_utils', 'hand_landmarker.task')),
    running_mode=VisionRunningMode.IMAGE)

with HandLandmarker.create_from_options(options) as landmarker:
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        if dirpath.endswith('none'):
            continue
        for file in filenames:
            filepath = os.path.join(dirpath, file)

            mp_image = mp.Image.create_from_file(filepath)
            hand_landmarker_result = landmarker.detect(mp_image)
            if len(hand_landmarker_result.handedness) == 0:
                if os.path.exists(filepath):
                    print(f'deleting {filepath}')
                    os.remove(filepath)
            else:
                continue
                score = hand_landmarker_result.handedness[0][0].score
                if score < 0.99:
                    img = cv2.imread(filepath)
                    cv2.imshow("window",
                               cv2.putText(cv2.resize(img, (0, 0), fx=0.3, fy=0.3), str(score), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 
                                           1, (0, 255, 255), 2, cv2.LINE_AA))
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()