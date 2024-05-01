import os
import sys
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import gesture_recognizer
import matplotlib.pyplot as plt

import mediapipe as mp

path = os.getcwd()
try:
    dataset_path = path[:path.rindex('/')] + '/Datasets/SLOVO_sign_dataset/slovo_cropped_mediapipe/letters/'
except:
    dataset_path = path[:path.rindex('\\')] + '\\Datasets\\SLOVO_sign_dataset\\slovo_cropped_mediapipe\\letters\\'

data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams()
)


train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)
epochs=500
model_name = f"model_clean_cropped_mediapipe_{epochs}epochs"

orig_stdout = sys.stdout
f = open(f'{model_name}.txt', 'w')
sys.stdout = f

delegate=mp.tasks.BaseOptions.Delegate.GPU
hparams = gesture_recognizer.HParams(export_dir=os.path.join("gesture_recognizer", model_name), epochs=epochs)
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

model.export_model()

sys.stdout = orig_stdout
f.close()