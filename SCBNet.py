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


class SDTE(nn.Module):
    """
    Args:
        level_num: The number of boxes in the histogram
    """
    def __init__(self, level_num):
        super(SDTE, self).__init__()
        self.conv1 = nn.Sequential(ConvBNReLU(256, 256, 3, 1, 1, has_relu=False), nn.LeakyReLU(inplace=True))
        self.conv2 = ConvBNReLU(256, 128, 1, 1, 0, has_bn=False, has_relu=False)

        self.encode_step1 = nn.Sequential(ConvBNReLU(2, 64, 1, 1, 0, has_bn=False, has_relu=False, mode='1d'),
                                          nn.LeakyReLU(inplace=True))
        self.encode_step2 = ConvBNReLU(64, 128, 1, 1, 0, has_bn=False, mode='1d')

        self.embed_out = ConvBNReLU(256, 128, 1, 1, 0, has_bn=True, mode='1d')
        self.level_num = level_num

    def forward(self, x):
        """
        Input:
            X
        Output:
            T: Deep texture embedding feature
            E: Soft quantitative weight
        """
        x = self.conv1(x)
        x = self.conv2(x)
        N, C, H, W = x.shape

        X_bar = F.adaptive_avg_pool2d(x, (1, 1))
        S = (F.normalize(X_bar, dim=1) * F.normalize(x, dim=1)).sum(1)
        S = S.view(N, -1)

        S_min, _ = S.min(-1)
        S_min = S_min.unsqueeze(-1)
        S_max, _ = S.max(-1)
        S_max = S_max.unsqueeze(-1)

        q_idx = torch.arange(self.level_num).float().cuda()
        Q = q_idx.expand(N, self.level_num)
        Q = (2 * Q + 1) / (2 * self.level_num) * (S_max - S_min) + S_min
        Q = Q.unsqueeze(1)

        Delta_Q = Q[:, :, 1] - Q[:, :, 0]
        Delta_Q = Delta_Q.unsqueeze(-1)

        S_flat = S.unsqueeze(-1)
        E = 1 - torch.abs(Q - S_flat)
        E = E * (E > (1 - Delta_Q))

        H_hist = quant.sum(1)
        H_hist = H_hist / (H_hist.sum(-1).unsqueeze(-1))

        H_for_conv1d = H_hist.unsqueeze(1)
        Q_for_conv1d = Q.squeeze(1).unsqueeze(1)
        H_concat = torch.cat([Q_for_conv1d, H_for_conv1d], dim=1)

        H_encoded = self.encode_step1(H_concat)
        H_encoded = self.encode_step2(H_encoded)

        X_bar_vec = X_bar.squeeze(-1).squeeze(-1)
        X_bar_seq = X_bar_vec.expand(self.level_num, N, C).permute(1, 2, 0)

        T_in = torch.cat([H_encoded, X_bar_seq], dim=1)
        T = self.embed_out(T_in)
        return T, E


class HiSTE(nn.Module):
    def __init__(self, level_num):
        super(HiSTE, self).__init__()
        self.level_num = level_num
        self.sdte = SDTE(level_num)
        self.k = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.q = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.v = ConvBNReLU(128, 128, 1, 1, 0, has_bn=False, has_relu=False, mode='1d')
        self.out = ConvBNReLU(128, 256, 1, 1, 0, mode='1d')

    def forward(self, x):
        N, C, H, W = x.shape
        T, E = self.sdte(X)

        K = self.proj_k(T)
        Q = self.proj_q(T)
        V = self.proj_v(T)

        Kt = K.permute(0, 2, 1)
        W_attn = torch.bmm(Kt, Q)
        W_attn = F.softmax(W_attn, dim=-1)

        Vt = V.permute(0, 2, 1)
        T_refined = torch.bmm(W_attn, Vt)
        T_refined = T_refined.permute(0, 2, 1)
        T_refined = self.proj_out(T_refined)

        E_perm = E.permute(0, 2, 1)
        X_tex_1d = torch.bmm(T_refined, E_perm)
        X_tex = X_tex_1d.view(N, 256, H, W)
        return X_tex


class GTSF(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels (int): Input the number of channels for each feature
            reduction (int): The dimensionality reduction ratio in channel attention, with a default of 16
        """
        super(GTSF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2 // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2 // reduction, channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
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
            f_ori: Original features
            f_tex: texture features output by HiSTE
        Returns:
            out: fused features
        """
        fusion = torch.cat([f_ori, f_tex], dim=1)

        channel_attn = self.avg_pool(fusion)
        channel_attn = self.fc(channel_attn)
        channel_attn_ori, channel_attn_tex = torch.chunk(channel_attn, chunks=2, dim=1)

        spatial_attn = self.spatial_conv(fusion)

        f_tex_attn = f_tex * channel_attn_tex * spatial_attn
        f_ori_attn = f_ori * channel_attn_ori

        out = f_ori_attn + f_tex_attn

        return out
