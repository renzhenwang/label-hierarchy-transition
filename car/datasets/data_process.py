# -*- coding: utf-8 -*-

import shutil
import sys
import scipy.io as scio
import os

import numpy as np
import pandas as pd


def df2dict(df_file):
    df_list = df_file.values.tolist()
    df_list = [[item[1].strip('\'').replace('/', ''), item[0]] for item in df_list]
    df_dict = dict(df_list)
    print(df_dict)
    return df_dict


def image2class_file(root_path, label_dict, save_path):
    data = scio.loadmat(root_path + './cars_annos.mat')
    base_path = root_path
    train_path = save_path + '/train'
    test_path = save_path + '/test'

    images = data['annotations'][0]
    classes = data['class_names'][0]
    num_images = images.size

    for i in range(num_images):
        image_path = os.path.join(base_path, images[i][0][0])
        print('*' * 10, image_path)

        file_name = images[i][0][0]  # fine name
        file_name = file_name.split('/')[1]
        print(file_name)

        classid = images[i][5][0][0]  # class id
        # classid=np.array2string(classid)
        classid = classid.astype(np.int32)
        print(classid)
        id = classes[classid - 1][0].replace('/', '')
        file_name_new = os.path.join(id, file_name)
        print(file_name_new)

        label_value = label_dict[id]
        id_new = str(label_value).zfill(3) + '.' + id
        file_name_new = str(label_value).zfill(3) + '.' + file_name_new
        print(file_name_new)

        istest = images[i][6][0]  # train/test
        if istest:
            if not os.path.exists(os.path.join(test_path, id_new)):
                os.makedirs(os.path.join(test_path, id_new))
            shutil.copy(image_path, os.path.join(test_path, file_name_new))
            with open(save_path + '/car_test.txt', 'a') as f:
                f.write('{} {}\n'.format(file_name_new, classid))

        if not istest:
            if not os.path.exists(os.path.join(train_path, id_new)):
                os.makedirs(os.path.join(train_path, id_new))
            shutil.copy(image_path, os.path.join(train_path, file_name_new))
            with open(save_path + '/car_train.txt', 'a') as f:
                f.write('{} {}\n'.format(file_name_new, classid))



if __name__ == "__main__":
    root_path = '../../data_source/car_dataset'
    save_path = '../../data_target/Cars'

    df_class = pd.read_excel("Cars.xlsx", usecols=[0, 1], names=None)
    label_dict = df2dict(df_class)

    image2class_file(root_path, label_dict, save_path)