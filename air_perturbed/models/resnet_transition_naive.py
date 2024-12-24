import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet50


class ResNetTransitionNaive(nn.Module):
    def __init__(self, label_hierarchy, hid_dim=600, pretrained=False, fine_to_coarse=False):

        super(ResNetTransitionNaive, self).__init__()

        assert type(label_hierarchy) == list
        print('Run the LHT with unlearned transition matrix!')

        self.fine_to_coarse = fine_to_coarse
        if self.fine_to_coarse:
            label_hierarchy.reverse()
        self.label_hierarchy = label_hierarchy
        self.split_dim = hid_dim // len(label_hierarchy)

        resnet = resnet50(pretrained)
        self.feature_dim = resnet.fc.in_features
        self.extractor = nn.Sequential(*list(resnet.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            # nn.Dropout(0.5),
            nn.Linear(self.feature_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ELU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(feature_size, classes_num),
        )

        self.classifier = nn.Linear(self.split_dim, label_hierarchy[0])
        # self.classifier = nn.Linear(hid_dim, label_hierarchy[0])
        
        W_s2f = np.load('models/transition_weights.npz')['W_s2f']
        W_f2o = np.load('models/transition_weights.npz')['W_f2o']
        self.W_s2f = torch.tensor(W_s2f, requires_grad=False).float().cuda()
        self.W_f2o = torch.tensor(W_f2o, requires_grad=False).float().cuda()


    def forward(self, x, targets=None):
        x = self.extractor(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # N * 512

        feat_splits = []
        for i in range(len(self.label_hierarchy)):
            feat_splits.append(x[:, self.split_dim * i:self.split_dim * (i + 1)])

        probs = []
        transition_matrices = []
        logit = self.classifier(feat_splits[0])
        # logit = self.classifier(x)
        probs.append(F.softmax(logit, dim=-1))

        for i in [0, 1]:
            if i == 0:
                tm = self.W_s2f.unsqueeze(0).expand(x.shape[0], -1, -1)
            elif i == 1:
                tm = self.W_f2o.unsqueeze(0).expand(x.shape[0], -1, -1)
            
            tm = tm.transpose(1, 2)
            
            prob = torch.matmul(tm, probs[-1].unsqueeze(-1)).squeeze(-1)
            probs.append(prob)
            transition_matrices.append(tm)

        if self.fine_to_coarse:
            probs.reverse()
            transition_matrices.reverse()

        return probs, transition_matrices


if __name__ == "__main__":
    label_hierarchy = [13, 38, 200]
    x = torch.rand(2, 3, 448, 448)
    y = [torch.ones(2, 13), torch.ones(2, 38), torch.ones(2, 200)]
    model = ResNetTransitionNaive(label_hierarchy, pretrained=False, fine_to_coarse=True)
    y_hat, t_mats = model(x, y)
    for i in range(len(y_hat)):
        print(y_hat[i].shape)
        # print(t_mats[i].shape)

    # for n, p in model.named_parameters():
    #     print(n, p.shape)