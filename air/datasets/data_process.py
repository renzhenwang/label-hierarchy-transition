# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd
import numpy as np


def df2dict(df_file):
    df_list = df_file.values.tolist()
    df_list = [[item[0].strip(), item[1]] for item in df_list]
    df_dict = dict(df_list)
    print(df_dict)
    return df_dict


def image2class_file(root_path, label_dict, dataset='train', save_path=None):
    print('==> data processing...')
    image_path = os.path.join(root_path, 'images')
    image_list = os.listdir(image_path)
    print(image_list)

    if save_path is None:
        save_path = root_path

    _save_path = os.path.join(save_path, dataset)

    if dataset == 'train':
        f = open(root_path + '/images_variant_train.txt', 'r')
    elif dataset == 'val':
        f = open(root_path + '/images_variant_val.txt', 'r')
    elif dataset == 'trainval':
        f = open(root_path + '/images_variant_trainval.txt', 'r')
    else:
        f = open(root_path + '/images_variant_test.txt', 'r')

    dataset_list = list(f)

    count = 0
    for i in range(len(image_list)):
        for j in range(len(dataset_list)):
            if image_list[i][:7] == dataset_list[j][:7]:
                
                label_key = dataset_list[j][8:].strip()
                print(label_key)
                label_value = label_dict[label_key]

                class_name = str(label_value).zfill(3) + '.' + label_key
                class_name = class_name.replace('/', '')

                print(class_name)
                if os.path.isdir(_save_path + '/' + class_name):
                    shutil.copy(image_path + '/' + image_list[i], _save_path + '/' + class_name + '/' + image_list[i])
                else:
                    os.makedirs(_save_path + '/' + class_name)
                    shutil.copy(image_path + '/' + image_list[i], _save_path + '/' + class_name + '/' + image_list[i])
                count += 1
                print('Image %s belongs to %s dataset' % (count, dataset))
    print('Finished!!')
    return None


if __name__ == '__main__':
    root_path = '../../data_source/air_dataset/fgvc-aircraft-2013b/data'
    save_path = '../../data_target/Aircraft'

    df_class = pd.read_excel("Air.xls", usecols=[0, 1], names=None)

    label_dict = df2dict(df_class)
    image2class_file(root_path, label_dict, 'train', save_path)
    image2class_file(root_path, label_dict, 'val', save_path)
    image2class_file(root_path, label_dict, 'trainval', save_path)
    image2class_file(root_path, label_dict, 'test', save_path)

