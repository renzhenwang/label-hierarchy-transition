import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import copy

# from air_label_tree import Trees, Trees_Maker_to_Family, Trees_Family_to_Model


Trees = [
    [1, 1, 1],
    [2, 2, 1],
    [3, 22, 1],
    [4, 22, 1],
    [5, 22, 1],
    [6, 22, 1],
    [7, 7, 1],
    [8, 7, 1],
    [9, 40, 1],
    [10, 40, 1],
    [11, 40, 1],
    [12, 40, 1],
    [13, 6, 1],
    [14, 36, 2],
    [15, 4, 3],
    [16, 9, 3],
    [17, 10, 7],
    [18, 10, 7],
    [19, 11, 7],
    [20, 12, 4],
    [21, 13, 5],
    [22, 14, 5],
    [23, 15, 5],
    [24, 16, 5],
    [25, 16, 5],
    [26, 16, 5],
    [27, 16, 5],
    [28, 16, 5],
    [29, 16, 5],
    [30, 16, 5],
    [31, 16, 5],
    [32, 49, 5],
    [33, 49, 5],
    [34, 49, 5],
    [35, 49, 5],
    [36, 18, 5],
    [37, 18, 5],
    [38, 19, 5],
    [39, 19, 5],
    [40, 19, 5],
    [41, 20, 5],
    [42, 20, 5],
    [43, 21, 21],
    [44, 62, 14],
    [45, 23, 9],
    [46, 24, 9],
    [47, 25, 9],
    [48, 25, 9],
    [49, 26, 8],
    [50, 27, 8],
    [51, 28, 8],
    [52, 28, 8],
    [53, 29, 12],
    [54, 29, 12],
    [55, 48, 23],
    [56, 31, 14],
    [57, 32, 14],
    [58, 57, 14],
    [59, 17, 23],
    [60, 35, 12],
    [61, 51, 12],
    [62, 37, 12],
    [63, 38, 13],
    [64, 39, 26],
    [65, 64, 15],
    [66, 41, 15],
    [67, 41, 15],
    [68, 41, 15],
    [69, 59, 15],
    [70, 59, 15],
    [71, 58, 15],
    [72, 44, 16],
    [73, 33, 23],
    [74, 46, 22],
    [75, 47, 11],
    [76, 5, 11],
    [77, 42, 18],
    [78, 50, 18],
    [79, 8, 18],
    [80, 45, 6],
    [81, 53, 19],
    [82, 53, 19],
    [83, 54, 7],
    [84, 55, 20],
    [85, 56, 4],
    [86, 30, 21],
    [87, 3, 23],
    [88, 52, 23],
    [89, 52, 23],
    [90, 61, 23],
    [91, 34, 17],
    [92, 60, 25],
    [93, 63, 27],
    [94, 67, 27],
    [95, 65, 28],
    [96, 66, 10],
    [97, 43, 24],
    [98, 68, 29],
    [99, 69, 29],
    [100, 70, 30],
]


class AirDataset(Dataset):
    def __init__(self, image_dir, input_transform=None, semi_level=True, ratio=1.0, relabel='family'):
        super(AirDataset, self).__init__()

        self.ratio = ratio
        # self.trees = self.perturb_label_tree(Trees)
        self.trees = Trees

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

    def perturb_label_tree(self, label_tree, perturb_ratio=0.4):
        perturbed_tree = copy.deepcopy(label_tree)

        family_labels = [item[1] for item in perturbed_tree]
        unique_family_labels = list(set(family_labels))

        number_to_perturb = int(perturb_ratio * len(unique_family_labels))

        labels_to_perturb = random.sample(unique_family_labels, number_to_perturb)
        perturbed_labels = labels_to_perturb.copy()
        random.shuffle(perturbed_labels)

        label_mapping = {old: new for old, new in zip(labels_to_perturb, perturbed_labels)}
        
        for item in perturbed_tree:
            if item[1] in label_mapping:
                item[1] = label_mapping[item[1]]
        
        for i in range(len(label_tree)):
            # print("original label tree: ", label_tree[i])
            print(perturbed_tree[i])

        return perturbed_tree

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

