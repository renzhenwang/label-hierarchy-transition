# *_*coding: utf-8 *_*
 # author --liming--
 
import os
import shutil
import time
 
time_start = time.time()


class config(object):
    path = '../../data_source/cub_dataset/CUB_200_2011/'
    ROOT_TRAIN = path + 'images/train/'
    ROOT_TEST = path + 'images/test/'
    BATCH_SIZE = 16

path_images = config.path + 'images.txt'
path_split = config.path + 'train_test_split.txt'
trian_save_path = '../../data_target/CUB-200-2011/train/'
test_save_path = '../../data_target/CUB-200-2011/test/'
 

images = []
with open(path_images,'r') as f:
    for line in f:
        images.append(list(line.strip('\n').split(',')))
 
split = []
with open(path_split, 'r') as f_:
    for line in f_:
        split.append(list(line.strip('\n').split(',')))
 
num = len(images)
for k in range(num):
    file_name = images[k][0].split(' ')[1].split('/')[0]
    aaa = int(split[k][0][-1])
    if int(split[k][0][-1]) == 1:
        if os.path.isdir(trian_save_path + file_name):
            shutil.copy(config.path + 'images/' + images[k][0].split(' ')[1], trian_save_path+file_name+'/'+images[k][0].split(' ')[1].split('/')[1])
        else:
            os.makedirs(trian_save_path + file_name)
            shutil.copy(config.path + 'images/' + images[k][0].split(' ')[1], trian_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
        print('%s finished!' % images[k][0].split(' ')[1].split('/')[1])
    else:
         if os.path.isdir(test_save_path + file_name):
             aaaa = config.path + 'images/' + images[k][0].split(' ')[1]
             bbbb = test_save_path+file_name+'/'+images[k][0].split(' ')[1]
             shutil.copy(config.path + 'images/' + images[k][0].split(' ')[1], test_save_path+file_name+'/'+images[k][0].split(' ')[1].split('/')[1])
         else:
             os.makedirs(test_save_path + file_name)
             shutil.copy(config.path + 'images/' + images[k][0].split(' ')[1], test_save_path + file_name + '/' + images[k][0].split(' ')[1].split('/')[1])
         print('%s finished!' % images[k][0].split(' ')[1].split('/')[1])
 
time_end = time.time()
print('Finished! time: %s!!' % (time_end - time_start))