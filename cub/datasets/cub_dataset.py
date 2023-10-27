import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.cub_label_tree import Trees, Trees_Order_to_Family, Trees_Family_to_Species


class CubDataset(Dataset):
    def __init__(self, image_dir, input_transform=None, semi_level=True, ratio=1.0, relabel='family'):
        super(CubDataset, self).__init__()

        self.ratio = ratio
        self.trees = Trees
        self.trees_order_to_family = Trees_Order_to_Family
        self.trees_family_to_species = Trees_Family_to_Species

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
            species_label = int(cls.strip().split('.')[0])-1
            order_label, family_label = self.get_order_family_target(species_label)
            for img_idx in range(len(imgs)):
                tmp_name_list.append(os.path.join(image_dir, cls, imgs[img_idx]))
                if img_idx <= int(len(imgs) * self.ratio) or not semi_level:
                    tmp_class_label_list.append([order_label, family_label, species_label])
                else:
                    if relabel == 'family':
                        tmp_class_label_list.append([order_label, family_label, -1])
                    elif relabel == 'order':
                        tmp_class_label_list.append([order_label, -1, -1])
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

    def get_order_family_target(self, species):
        order = Trees[species][1] - 1
        family = Trees[species][2] - 1
        return order, family

    def get_transition_weights(self):
        W_s2f = np.zeros([200, 38])
        W_s2o = np.zeros([200, 13])
        W_f2o = np.zeros([38, 13])
        for i in range(200):
            order = Trees[i][1] -1
            family = Trees[i][2] - 1
            W_s2f[i, family] = 1
            W_s2o[i, order] = 1
            W_f2o[family, order] = 1
        np.savez('transition_weights', W_s2f=W_s2f, W_s2o=W_s2o, W_f2o=W_f2o)
        return W_s2f, W_s2o, W_f2o

    def get_hierarch_weights(self, beta=0.999):
        species_per_class_num = [0] * 200
        family_per_class_num = [0] * 38
        order_per_class_num = [0] * 13
        for label in self.labels:
            if label[-1] >= 0:
                species_per_class_num[label[-1]] += 1
            if label[1] >= 0:
                family_per_class_num[label[1]] += 1
            if label[0] >= 0:
                order_per_class_num[label[0]] += 1

        species_per_class_num = np.array(species_per_class_num)
        species_effective_num = 1.0 - np.power(beta, species_per_class_num)
        species_weights = (1.0 - beta) / np.array(species_effective_num)
        species_weights = species_weights / np.sum(species_weights) * 200
        print('class weight: ', species_weights)

        family_per_class_num = np.array(family_per_class_num)
        family_effective_num = 1.0 - np.power(beta, family_per_class_num)
        family_weights = (1.0 - beta) / np.array(family_effective_num)
        family_weights = family_weights / np.sum(family_weights) * 38
        print('class weight: ', family_weights)

        order_per_class_num = np.array(order_per_class_num)
        order_effective_num = 1.0 - np.power(beta, order_per_class_num)
        order_weights = (1.0 - beta) / np.array(order_effective_num)
        order_weights = order_weights / np.sum(order_weights) * 13
        print('class weight: ', order_weights)

        return order_weights, family_weights, species_weights


if __name__ == '__main__':
    root = 'D:/cub-200-2011/dataset/train'
    ratio = 0.5
    dataset = CubDataset(root, semi_level=True, ratio=ratio)

    for i, (image, label, name) in enumerate(dataset):
        print(name, label)

