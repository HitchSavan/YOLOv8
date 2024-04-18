from ultralytics import YOLO
from pathlib import Path
import os
import cv2

def yolo_init(path):
    sizedict = {
        'nano': 'n',
        'small': 's',
        'medium': 'm'
    }
    print('Loading YOLO detection model...')
    # 'n' for nano, 's' for small, 'm' for medium
    size = 'medium'
    # Load a custom model
    model = YOLO(os.path.join(path, 'hand_detector', f'yolov8{sizedict[size]}.pt'))
    return model

def yolo(model, img):
    # Predict with the model
    results = model(img)  # predict on an image
    boxes = results[0].boxes
    crop_coords = []
    for box in boxes:
        xyxy = tuple([int(x) for x in box.xyxy[0]])
        crop_coords.append(xyxy)
    
    return crop_coords

if __name__ == '__main__':

    path = os.getcwd()
    model = yolo_init(path)
    dataset_src_folder = os.path.join(path, '..', 'Datasets', 'SLOVO_sign_dataset', 'slovo', 'letters')
    dataset_src_folder = Path(dataset_src_folder)
    dataset_parent_folder = dataset_src_folder.parent.parent

    # get recursive
    # iterate over files
    for dirpath, dirnames, filenames in os.walk(dataset_src_folder):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            img = cv2.imread(filepath)
            print(filepath)

            crop_coords = yolo(model, img)

            if crop_coords:
                for i, box in enumerate(crop_coords):
                    width = box[2] - box[0]
                    height = box[3] - box[1]

                    y_min = int(box[1] - width/3)
                    y_min = y_min if y_min > 0 else 0

                    y_max = int(box[3] + width/3)
                    y_max = y_max if y_max < img.shape[0] else img.shape[0]-1

                    x_min = int(box[0] - height/3)
                    x_min = x_min if x_min > 0 else 0

                    x_max = int(box[2] + height/3)
                    x_max = x_max if x_max < img.shape[1] else img.shape[1]-1

                    cropped_image = img[y_min:y_max, x_min:x_max]
                    new_filepath = os.path.join(dataset_parent_folder, 'slovo_cropped', 'letters', Path(dirpath).name, Path(filepath).stem + '_' + str(i) + '.jpg')
                    if not os.path.exists(Path(new_filepath).parent):
                        os.makedirs(Path(new_filepath).parent)
                    print(new_filepath)
                    cv2.imwrite(new_filepath, cropped_image)