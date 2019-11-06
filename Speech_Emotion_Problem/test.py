import sys
import os
import pandas as pd
from utils import *
from tqdm import tqdm

arguement_list = sys.argv
if len(arguement_list) <= 1:
    print("error : specify test folder path")
    exit()

classes = ['neutral', 'happy', 'disgust', 'sad', 'fear']

test_path = str(arguement_list[1])

files = os.listdir(test_path)
preds = list()

model = get_model()

for i in tqdm(range(len(files))):
    file = files[i]
    features = torch.tensor(get_mfcc_features(os.path.join(test_path, file))).float().unsqueeze(0)
    output = model(normalise(features))
    preds.append(classes[output.argmax()])

test_df = pd.DataFrame({'image_name': files, 'tags': preds})
test_df.to_csv('test.csv', index=False)
