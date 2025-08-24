import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义教师模型（简单示例，实际应用中可能更复杂）
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 定义学生模型
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 8 * 8 * 8)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 定义蒸馏损失函数（这里简单结合了交叉熵损失和特征图蒸馏损失）
def distillation_loss(logits_student, logits_teacher, feat_student, feat_teacher, alpha=0.5, temperature=10):
    criterion_ce = nn.CrossEntropyLoss()
    loss_ce = criterion_ce(logits_student, target)

    loss_kd = nn.MSELoss()(nn.functional.softmax(logits_teacher / temperature, dim=1),
                           nn.functional.softmax(logits_student / temperature, dim=1))
    loss_feat = nn.MSELoss()(feat_student, feat_teacher)

    return alpha * loss_ce + (1 - alpha) * (loss_kd + loss_feat)


