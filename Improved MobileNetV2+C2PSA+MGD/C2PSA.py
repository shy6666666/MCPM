import torch
import torch.nn as nn
import torch.nn.functional as F


class PSA(nn.Module):
    """金字塔注意力模块 (Pyramid Squeeze Attention)"""

    def __init__(self, c, hidden_dim=32, kernel_sizes=[3, 5, 7]):
        super().__init__()
        # 多尺度卷积提取特征
        self.convs = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, k, padding=k // 2, groups=hidden_dim)
            for k in kernel_sizes
        ])
        # 通道注意力（SE机制）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c + hidden_dim * len(kernel_sizes), c // 4, 1),
            nn.ReLU(),
            nn.Conv2d(c // 4, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 原始特征 + 多尺度特征融合
        conv_outs = [conv(x) for conv in self.convs]
        fused = torch.cat([x] + conv_outs, dim=1)  # 拼接原始特征和多尺度特征
        attn = self.se(fused)  # 生成通道注意力权重
        return x * attn  # 应用注意力


class Bottleneck(nn.Module):
    """CSP结构中的基础瓶颈模块"""

    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = nn.Conv2d(c, c // 2, 1, 1)
        self.cv2 = nn.Conv2d(c // 2, c, 3, 1, 1)
        self.act = nn.SiLU()
        self.shortcut = shortcut

    def forward(self, x):
        y = self.act(self.cv1(x))
        y = self.cv2(y)
        return x + y if self.shortcut else y


class C2PSA(nn.Module):
    """Cross Stage Partial with Pyramid Squeeze Attention模块"""

    def __init__(self, c, n=1, shortcut=True):
        super().__init__()
        self.c = c
        self.split = c // 2  # 按通道分成两部分
        self.cv1 = nn.Conv2d(c, self.split, 1, 1)  # 第一部分直接传递
        self.cv2 = nn.Conv2d(c, self.split, 1, 1)  # 第二部分进入处理链路

        # 处理链路：瓶颈层 + PSA注意力
        self.m = nn.Sequential(
            *[Bottleneck(self.split, shortcut) for _ in range(n)],
            PSA(self.split)  # 加入金字塔注意力
        )

        self.cv3 = nn.Conv2d(self.split * 2, c, 1, 1)  # 融合两部分特征

    def forward(self, x):
        # CSP结构：特征分路处理
        x1 = self.cv1(x)  # 第一路：直接传递
        x2 = self.cv2(x)  # 第二路：经瓶颈层和注意力处理
        x2 = self.m(x2)
        # 融合两路特征
        return self.cv3(torch.cat([x1, x2], dim=1))



