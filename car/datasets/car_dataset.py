import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from datasets.car_label_tree import Trees, Trees_Maker_to_Model


class CarDataset(Dataset):
    def __init__(self, image_dir, input_transform=None, semi_level=True, ratio=1.0):
        super(CarDataset, self).__init__()

        self.ratio = ratio
        self.trees = Trees
        self.trees_maker_to_model = Trees_Maker_to_Model

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
            maker_label = self.get_maker_target(model_label)
            for img_idx in range(len(imgs)):
                tmp_name_list.append(os.path.join(image_dir, cls, imgs[img_idx]))
                if img_idx <= int(len(imgs) * self.ratio) or not semi_level:
                    tmp_class_label_list.append([maker_label, model_label])
                else:
                    tmp_class_label_list.append([maker_label, -1])

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
        return input, np.array(target)  # , imagename

    def __len__(self):
        return len(self.image_filenames)

    def get_maker_target(self, model):
        maker = Trees[model][1] - 1
        return maker

    def get_transition_weights(self):
        W_s2f = np.zeros([196, 9])
        for i in range(196):
            family = Trees[i][1] -1
            W_s2f[i, family] = 1
        np.savez('transition_weights', W_s2f=W_s2f)
        return W_s2f


if __name__ == '__main__':
    root = 'G:/dataset/Cars/train'
    ratio = 0.5
    dataset = CarDataset(root, semi_level=True, ratio=ratio)

    # for i, (image, label, name) in enumerate(dataset):
    #     print(name, label)

    dataset.get_transition_weights()
    x = np.load('transition_weights.npz', allow_pickle=True)
    print(x['W_s2f'])

