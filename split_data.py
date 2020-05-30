import os
import random
import glob

random.seed(0)
path = './data_processed_vkt2'
file_list = os.listdir(path)
total_num = len(file_list)

val_num = int(total_num / 11)
train_num = total_num - val_num

train_list = random.sample(file_list, train_num)
val_list = [i for i in file_list if i not in train_list]
for fn in train_list:
    rela_fn = os.path.join(path, fn)
    new_fn = 'TRAIN_' + fn
    new_rela_fn = os.path.join(path, new_fn)
    os.rename(rela_fn, new_rela_fn)

for fn in val_list:
    rela_fn = os.path.join(path, fn)
    new_fn = 'VAL_' + fn
    new_rela_fn = os.path.join(path, new_fn)
    os.rename(rela_fn, new_rela_fn)

