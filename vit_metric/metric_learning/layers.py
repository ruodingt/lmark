import math
import torch.nn as nn
import torch.nn.functional as F

import torch


class FocalLoss(nn.Module):
    """
    handling unbalanced dataset
    """

    def __init__(self, gamma=0.0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class ArcSubCentre(nn.Module):
    """
    Detail see:
    https://github.com/deepinsight/insightface/tree/master/recognition/SubCenter-ArcFace
    """

    def __init__(self, in_features, out_features, k=3, embedding_size=512, neck_type='A'):
        """

        Args:
            in_features: dim `F` of normalised feature vector
            out_features: dim `N` N_classes
            k:
        """
        super().__init__()
        if embedding_size == 0:
            raise NotImplementedError()

        self.neck_type = neck_type

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if neck_type == "LBA":
            self.neck = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=embedding_size, bias=True),
                nn.BatchNorm1d(embedding_size),
                torch.nn.PReLU()
            )
        elif neck_type == "DLBA":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features=in_features, out_features=embedding_size, bias=True),
                nn.BatchNorm1d(embedding_size),
                torch.nn.PReLU()
            )
        elif neck_type == "LB":
            self.neck = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=embedding_size, bias=False),
                nn.BatchNorm1d(embedding_size),
            )
        elif neck_type == "":
            self.dense_layer = torch.nn.Linear(in_features=in_features, out_features=embedding_size)

        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, embedding_size))  # (N*K)*F
        nn.init.xavier_uniform_(self.weight)

        self.k = k
        self.out_features = out_features

    def forward(self, features):
        """

        Args:
            features:

        Returns:
            cosine distance between feature and subclass center
        """

        if self.neck_type in ["LBA", "DLBA", "LB"]:
            features = self.neck(features)
        else:
            features = self.dense_layer(features)

        # FIXME: This normalisation does not look super correct when k>1, it's minor detail but worth a fix
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)  # (Batch, N, K)
        # find max cos similarities between feature vector and sub-center of each of N classes
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine




class ArcMarginModelLoss(nn.Module):
    """
    stealing implementation from
    https://github.com/foamliu/InsightFace-v2/blob/e7b6142875f3f6c65ce97dd1b2b58156c5f81a3d/models.py

    """
    def __init__(self, margin_m, feature_scaler, easy_margin, focal_loss, gamma):
        """

        Args:
            margin_m:
            feature_scaler:
            easy_margin:
            focal_loss:
            gamma:
        """
        super(ArcMarginModelLoss, self).__init__()

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = feature_scaler

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

        if focal_loss:
            self.criterion = FocalLoss(gamma=gamma)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, cosines, labels):
        sine = torch.sqrt(1.0 - torch.pow(cosines, 2))
        phi = cosines * self.cos_m - sine * self.sin_m  # cos(theta + m) such that avoid calculating arccos()
        if self.easy_margin:
            phi = torch.where(cosines > 0, phi, cosines)
        else:
            phi = torch.where(cosines > self.th, phi, cosines - self.mm)
        one_hot = torch.nn.functional.one_hot(labels, num_classes=cosines.size()[-1])
        # torch.zeros(cosine.size(), device=device)
        # one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosines)
        output *= self.s

        loss = self.criterion(input=output, target=labels)
        return loss


# class ArcFaceLossAdaptiveMargin(nn.modules.Module):
#     """
#     This is loss function no params
#     FIXME: It may not be necessary to put a pure loss in a module
#     """
#
#     def __init__(self, margins, s=30.0):
#         super().__init__()
#         self.crit = DenseCrossEntropy()
#         self.s = s
#         self.margins = margins
#
#     def forward(self, logits, labels):
#         ms = self.margins.cpu().numpy()[labels.cpu().numpy()]
#         cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
#         sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
#         th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
#         mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
#         labels = F.one_hot(labels, logits.size()).float()
#         logits = logits.float()
#         cosine = logits
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
#         phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
#         output = (labels * phi) + ((1.0 - labels) * cosine)
#         output *= self.s
#         loss = self.crit(output, labels)
#         return loss
#
# class DenseCrossEntropy(nn.Module):
#     def forward(self, x, target):
#         x = x.float()
#         target = target.float()
#         logprobs = torch.nn.functional.log_softmax(x, dim=-1)
#
#         loss = -logprobs * target
#         loss = loss.sum(-1)
#         return loss.mean()
