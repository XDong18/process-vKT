import os
import random
import glob

random.seed(0)
path = './data_processed_vkt2'
file_list = os.listdir(path)
total_num = len(file_list)

val_num = int(total_num / 11)
train_num = total_num - val_num

# train_list = random.sample(file_list, train_num)
# val_list = random.sample(file_list, val_num)
val_list = glob.glob(os.path.join(path, 'clone' + '*' + '.npz'))
# for fn in train_list:
#     rela_fn = os.path.join(path, fn)
#     new_fn = 'TRAIN_' + fn
#     new_rela_fn = os.path.join(path, new_fn)
#     os.rename(rela_fn, new_rela_fn)

for fn in val_list:
    # rela_fn = os.path.join(path, fn)
    s_fn = os.path.split(fn)[-1]
    # print(s_fn)
    new_fn = 'VAL_' + s_fn
    new_rela_fn = os.path.join(path, new_fn)
    os.rename(fn, new_rela_fn)

