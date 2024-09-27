
import os
import shutil
from tqdm import tqdm
import random


data = 'data'
train_data = 'train'
validation_data = 'validation'
test_data = 'test'

files = os.listdir(data)
data_files = []
for i in range(len(files)):
    if files[i].endswith('.xyz'):
        data_files.append(files[i])

# Split data_files into three different lists: one for training,
# validation, and testing
p_train = 0.7
p_val = 0.15
p_test = 0.15

n_train = int(p_train * len(data_files))
n_val = int(p_val * len(data_files))
n_test = int(p_test * len(data_files))

random.shuffle(data_files)

train = data_files[:n_train]
validation = data_files[n_train:(n_train+n_val)]
test = data_files[(n_train+n_val):]

new_folders = [train_data, validation_data, test_data]
new_files = [train, validation, test]

for folder in new_folders:
    if os.path.exists(folder):
        continue
    os.mkdir(folder)

for folder, files in zip(new_folders, new_files):
    for file in tqdm(files):
        shutil.copyfile(data + '/' + file, folder + '/' + file)
