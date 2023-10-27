import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import resnet50


class ResNetTransition(nn.Module):
    def __init__(self, label_hierarchy, hid_dim=600, pretrained=False, fine_to_coarse=False, sprite_dim=2):

        super(ResNetTransition, self).__init__()

        assert type(label_hierarchy) == list

        self.fine_to_coarse = fine_to_coarse
        if self.fine_to_coarse:
            label_hierarchy.reverse()
        self.label_hierarchy = label_hierarchy
        self.split_dim = hid_dim // len(label_hierarchy)
        self.sprite_dim = sprite_dim

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
        self.transition_net_left = nn.ModuleList()
        self.transition_net_right = nn.ModuleList()
        for i in range(len(label_hierarchy)-1):
            self.transition_net_left.append(nn.Linear(self.split_dim, label_hierarchy[i] * self.sprite_dim))
            self.transition_net_right.append(nn.Linear(self.split_dim, self.sprite_dim * label_hierarchy[i+1]))

        W_s2f = np.load('models/transition_weights.npz')['W_s2f']
        self.W_s2f = torch.tensor(W_s2f, requires_grad=False).float().cuda()
        self.gamma_s2f = nn.Parameter(torch.ones(1))
        self.beta_s2f = nn.Parameter(torch.zeros(1))

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
        probs.append(F.softmax(logit, dim=-1))
        for i, (left_net, right_net) in enumerate(zip(self.transition_net_left, self.transition_net_right)):
            tm_left = left_net(feat_splits[i+1]).view(x.shape[0], self.label_hierarchy[i], self.sprite_dim)
            tm_right = right_net(feat_splits[i+1]).view(x.shape[0], self.sprite_dim, self.label_hierarchy[i+1])
            tm = torch.matmul(tm_left, tm_right)
            if i == 0:
                tm = tm + self.gamma_s2f*self.W_s2f + self.beta_s2f
            tm = torch.softmax(tm, dim=-1).transpose(1, 2)
            prob = torch.matmul(tm, probs[-1].unsqueeze(-1)).squeeze(-1)
            probs.append(prob)
            transition_matrices.append(tm)
        if self.fine_to_coarse:
            probs.reverse()
            transition_matrices.reverse()

        return probs, transition_matrices


if __name__ == "__main__":
    label_hierarchy = [9, 196]
    x = torch.rand(2, 3, 448, 448)
    y = [torch.ones(2, 9), torch.ones(2, 196)]
    model = ResNetTransition(label_hierarchy, pretrained=False, fine_to_coarse=True)
    y_hat, t_mats = model(x, y)
    for i in range(len(y_hat)):
        print(y_hat[i].shape)
        # print(t_mats[i].shape)

    # for n, p in model.named_parameters():
    #     print(n, p.shape)