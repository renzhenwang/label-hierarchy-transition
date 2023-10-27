import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.air_label_tree import Trees, Trees_Maker_to_Family, Trees_Family_to_Model


class AirDataset(Dataset):
    def __init__(self, image_dir, input_transform=None, semi_level=True, ratio=1.0, relabel='family'):
        super(AirDataset, self).__init__()

        self.ratio = ratio
        self.trees = Trees
        self.trees_maker_to_family = Trees_Maker_to_Family
        self.trees_family_to_model = Trees_Family_to_Model

        name_list = []
        label_list = []
        classes = os.listdir(image_dir)
        classes.sort()

        for cls in classes:
            tmp_name_list = []
            tmp_class_label_list = []
            cls_imgs = os.path.join(image_dir, cls)
            imgs = os.listdir(cls_imgs)

            random.seed(10)
            random.shuffle(imgs)

            # cls_name = cls.strip().split('_')[-1]
            model_label = int(cls.strip().split('.')[0])-1
            maker_label, family_label = self.get_maker_family_target(model_label)
            for img_idx in range(len(imgs)):
                tmp_name_list.append(os.path.join(image_dir, cls, imgs[img_idx]))
                if img_idx <= int(len(imgs) * self.ratio) or not semi_level:
                    tmp_class_label_list.append([maker_label, family_label, model_label])
                else:
                    if relabel == 'family':
                        tmp_class_label_list.append([maker_label, family_label, -1])
                    elif relabel == 'order':
                        tmp_class_label_list.append([maker_label, -1, -1])
                    else:
                        raise NotImplementedError

            name_list += tmp_name_list
            label_list += tmp_class_label_list

        self.input_transform = input_transform
        self.image_filenames = name_list
        self.labels = label_list

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index]
        return input, np.array(target)   # , imagename

    def __len__(self):
        return len(self.image_filenames)

    def get_maker_family_target(self, model):
        family = Trees[model][1] - 1
        maker = Trees[model][2] - 1
        return maker, family

    def get_transition_weights(self):
        W_s2f = np.zeros([100, 70])
        W_s2o = np.zeros([100, 30])
        W_f2o = np.zeros([70, 30])
        for i in range(100):
            family = Trees[i][1] -1
            order = Trees[i][2] - 1
            W_s2f[i, family] = 1
            W_s2o[i, order] = 1
            W_f2o[family, order] = 1
        np.savez('transition_weights', W_s2f=W_s2f, W_s2o=W_s2o, W_f2o=W_f2o)
        return W_s2f, W_s2o, W_f2o


if __name__ == '__main__':
    root = 'G:/dataset/Aircraft/train'
    ratio = 0.5
    dataset = AirDataset(root, semi_level=True, ratio=ratio)

    # for i, (image, label, name) in enumerate(dataset):
    #     print(name, label)

    dataset.get_transition_weights()

    x = np.load('transition_weights.npz', allow_pickle=True)
    print(x['W_s2f'])
    print(x['W_s2o'])
    print(x['W_f2o'])

