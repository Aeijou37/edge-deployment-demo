"""
PointNet 分类模型 — 内嵌版本（不依赖 pointcloud-demo 项目）

参考: Qi et al., PointNet: Deep Learning on Point Sets, CVPR 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformNet(nn.Module):
    """T-Net: 学习一个变换矩阵对输入点云做空间对齐"""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(B, -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        I = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(B, 1)
        x = x + I
        x = x.view(B, self.k, self.k)
        return x


class PointNetClassifier(nn.Module):
    """PointNet 分类网络"""

    def __init__(self, num_classes=40, num_points=1024):
        super().__init__()
        self.num_points = num_points
        self.input_transform = TransformNet(k=3)
        self.feature_transform = TransformNet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.transpose(2, 1)

        t3 = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), t3).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        t64 = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), t64).transpose(2, 1)

        point_feat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        global_feat = torch.max(x, 2, keepdim=True)[0].view(B, -1)

        x = F.relu(self.bn6(self.fc1(global_feat)))
        x = self.dropout(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x, global_feat
