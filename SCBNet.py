import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    """
        Module for the Conv-BN-ReLU tuple.
    """

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation=1, group=1,
                 has_bn=True, has_relu=True, mode='2d'):
        super(ConvBNReLU, self).__init__()
        self.has_bn = has_bn
        self.has_relu = has_relu
        if mode == '2d':
            self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm2d
        elif mode == '1d':
            self.conv = nn.Conv1d(
                c_in, c_out, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False, groups=group)
            norm_layer = nn.BatchNorm1d
        if self.has_bn:
            self.bn = norm_layer(c_out)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

import torch
import torch.nn as nn
from torch.nn import functional as F


class SDTE(nn.Module):
    """
     SDTE operator
    """
    def __init__(self, L):
        super(SDTE, self).__init__()
        self.L = L 
        
        # 初始特征降维映射
        self.conv1 = nn.Sequential(ConvBNReLU(256, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)
        
        # 用于映射动态区间中心 Q_n 与统计描述符 H_n 的卷积层
        self.f1 = nn.Sequential(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
                                nn.LeakyReLU(inplace=True))
        self.f2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')
        
        # 最终输出深度纹理嵌入特征 T_n 的卷积层
        self.out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')

    def forward(self, X):
        """
        X: 输入浅层特征图，形状 (N, C, H, W)
        """
        # 特征预处理
        X = self.conv1(X)
        X = self.conv2(X)
        N, C, H, W = X.shape
        
        # 提取全局均值特征向量 X_bar
        X_bar = F.adaptive_avg_pool2d(X, (1, 1))
        
        # 计算余弦相似度谱 S
        S = (F.normalize(X_bar, dim=1) * F.normalize(X, dim=1)).sum(1)
        S = S.view(N, -1)  # 展平空间维度 p = H * W
        
        # 提取相似度谱的空间极值 S_max 与 S_min
        S_min, _ = S.min(-1)
        S_min = S_min.unsqueeze(-1)
        S_max, _ = S.max(-1)
        S_max = S_max.unsqueeze(-1)
        
        # 动态生成 L 个直方图区间中心 Q_l
        Q_l = torch.arange(self.L).float().cuda()
        Q_l = Q_l.expand(N, self.L)
        Q_l = (2 * Q_l + 1) / (2 * self.L) * (S_max - S_min) + S_min
        Q_l = Q_l.unsqueeze(1)
        
        # 计算区间跨度，用于软量化阈值截断
        Q_l_inter = Q_l[:, :, 1] - Q_l[:, :, 0]
        Q_l_inter = Q_l_inter.unsqueeze(-1)
        S = S.unsqueeze(-1)
        
        # 计算软分配权重 E
        E = 1 - torch.abs(Q_l - S)
        E = E * (E > (1 - Q_l_inter))  # 阈值截断等价于 ReLU 机制
        
        # 聚合与归一化，得到统计直方图描述符 H_n
        H_n = E.sum(1)
        H_n = H_n / (H_n.sum(-1).unsqueeze(-1))
        H_n = H_n.unsqueeze(1)
        
        # 拼接动态区间中心 Q_l 与统计描述符 H_n，经卷积映射
        QH_concat = torch.cat([Q_l, H_n], dim=1)
        mapped_QH = self.f1(QH_concat)
        mapped_QH = self.f2(mapped_QH)
        
        # 与全局均值特征 X_bar 沿通道维度拼接
        X_bar_flat = X_bar.squeeze(-1).squeeze(-1)
        X_bar_flat = X_bar_flat.expand(self.L, N, C).permute(1, 2, 0)
        
        T_n_concat = torch.cat([mapped_QH, X_bar_flat], dim=1)
        
        # 映射为深度嵌入的纹理特征
        T_n = self.out(T_n_concat)
        
        # 返回统计特征 T_n 与软分配权重矩阵 E
        return T_n, E


class HiSTE(nn.Module):
    """
    基于直方图的统计纹理增强模块
    """
    def __init__(self, L):
        super(HiSTE, self).__init__()
        self.L = L
        self.sdte = SDTE(L)
        self.k = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')

    def forward(self, X):
        N, C, H, W = X.shape
        T_n, E = self.sdte(X)
        K = self.k(T_n)
        Q = self.q(T_n)
        V = self.v(T_n)
        
        K = K.permute(0, 2, 1)
        Attention_map = torch.bmm(K, Q)
        Attention_map = F.softmax(Attention_map, dim=-1)
        
        V = V.permute(0, 2, 1)
        F_n = torch.bmm(Attention_map, V)
        F_n = F_n.permute(0, 2, 1)
        F_n = self.out(F_n)  # 经过注意力细化的特征 F_n
        
        E = E.permute(0, 2, 1)
        X_tex = torch.bmm(F_n, E)
        X_tex = X_tex.view(N, 256, H, W)
        
        return X_tex


class GTSF(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels (int): 输入每路特征的通道数
            reduction (int): 通道注意力中降维比例，默认为 16
        """
        super(GTSF, self).__init__()
        # 通道注意力模块：对拼接特征 (2C) 进行全局统计，输出 2C 的注意力权重，再拆分为两个 C 通道的权重
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2 // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2 // reduction, channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        # 空间注意力模块：利用拼接特征生成一个空间注意力图
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, f_ori, f_tex):
        """
        Args:
            f_ori: 原始特征，Tensor，形状 (N, C, H, W)
            f_tex: HiSTE 输出的纹理特征，Tensor，形状 (N, C, H, W)
        Returns:
            out: 融合后的特征，Tensor，形状 (N, C, H, W)
        """
        # 拼接两个特征： (N, 2C, H, W)
        fusion = torch.cat([f_ori, f_tex], dim=1)

        # --- 通道注意力 ---
        # 全局平均池化
        channel_attn = self.avg_pool(fusion)  # (N, 2C, 1, 1)
        # 生成注意力权重，形状仍为 (N, 2C, 1, 1)
        channel_attn = self.fc(channel_attn)
        # 将 2C 分成两部分，每部分 (N, C, 1, 1)
        channel_attn_ori, channel_attn_tex = torch.chunk(channel_attn, chunks=2, dim=1)

        # --- 空间注意力 ---
        spatial_attn = self.spatial_conv(fusion)  # (N, 1, H, W)

        # --- 应用注意力进行门控 ---
        # 调整 HiSTE 分支：先乘以对应的通道注意力，再乘以空间注意力
        f_tex_attn = f_tex * channel_attn_tex * spatial_attn
        # 对原始特征直接使用其通道权重
        f_ori_attn = f_ori * channel_attn_ori

        # 最后融合，两路特征相加
        out = f_ori_attn + f_tex_attn

        return out
