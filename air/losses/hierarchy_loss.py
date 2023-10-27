import numpy as np
import torch
import torch.nn.functional as F


class Hierarchy_Loss(object):
    def __init__(self, beta=-2.0):
        self.beta = beta

    def __call__(self, output, target, t_mats):
        cls_loss = 0
        trst_loss = 0
        for i in range(len(output)):
            select_mask = target[i] > -1
            select_targets = target[i][select_mask]
            select_outputs = output[i][select_mask]
            # print(target[i])
            # print(select_mask)
            if select_mask.sum() > 0:
                cls_loss += F.nll_loss(torch.log(select_outputs), select_targets)
            if i < len(output) - 1:
                trst_loss += -torch.mean(torch.sum(torch.log(t_mats[i]+1e-10) * t_mats[i], dim=1))
        cls_loss = cls_loss / len(output)
        trst_loss = trst_loss / (len(output) - 1)

        return cls_loss + self.beta * trst_loss


class Hierarchy_KL_Loss(object):
    def __init__(self, beta=2.0):
        self.beta = beta

    def __call__(self, output, target, t_mats):
        cls_loss = 0
        trst_loss = 0
        for i in range(len(output)):
            cls_num = output[i].shape[-1]
            select_mask = target[i] > -1
            select_targets = target[i][select_mask]
            select_outputs = output[i][select_mask]
            if select_mask.sum() > 0:
                cls_loss += F.nll_loss(torch.log(select_outputs), select_targets)
            if i < len(output)-1:
                # trst_loss += -torch.mean(torch.sum(torch.log(t_mats[i]) * t_mats[i], dim=1))
                trst_loss += torch.mean(torch.sum(1.0/cls_num * (np.log(1.0 / cls_num) - (t_mats[i]+1e-10).log()), dim=1))
        cls_loss = cls_loss / len(output)
        trst_loss = trst_loss / (len(output)-1)

        return cls_loss + self.beta * trst_loss
