import os
import pandas as pd
import cv2

def copy_frames(files, filename, text, path_to):
    converted = True
    file = next(item for item in files if item['name'][:-4] == filename)

    isExist = os.path.exists(f'{path_to}{text}')
    if not isExist:
        if any([x in text for x in ['?', '"']]):
            text.replace('?', '').replace('"', '')
        os.makedirs(f'{path_to}{text}')
    
    destination = f'{path_to}{text}' + '\\'
    orig_path = os.getcwd()
    os.chdir(destination)

    vidObj = cv2.VideoCapture(file['path'])
    if vidObj.isOpened():
        count = 0
        success = True

        while success:
            success, image = vidObj.read()
            if success:
                if not cv2.imwrite(f'{filename}{count}.jpg', image):
                    converted = False
            count += 1
        vidObj.release()
    else:
        print('failed')
        converted = False
    
    os.chdir(orig_path)
    return converted


path = os.path.join(os.path.dirname(os.getcwd()), 'Datasets', 'SLOVO_sign_dataset', 'slovo')
path_test = os.path.join(path, 'test')
path_train = os.path.join(path, 'train')
path_letters = os.path.join(path, 'letters')
path_words = os.path.join(path, 'words')

annotations = pd.read_csv(os.path.join(path, 'annotations.csv'), sep='\t')

print(annotations)

files = []

for folder in (path_test, path_train):
    for file in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file)):
            files.append({})
            files[-1]['path'] = os.path.join(folder, file)
            files[-1]['name'] = file

annotations = annotations.reset_index()  # make sure indexes pair with number of rows

annotations.loc[annotations['text'] == 'no_event', 'text'] = 'none'

fails = 0

for index, row in annotations.iterrows():

    if index < 6922:
        continue

    if row['text'].lower() != 'none':
        continue


    filename = row['attachment_id']
    text = row['text'].lower()
    is_train = row['train']

    if not copy_frames(files, filename, text, path_words):
        fails += 1

    if len(text) < 2 or text == 'none':
        if not copy_frames(files, filename, text, path_letters):
            fails += 1

    print(f'{index+1}/{len(annotations.index)}')

print(f'Total fails: {fails}')