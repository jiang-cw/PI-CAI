""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
from .model_utils import *
from collections import OrderedDict
from skimage import measure
import scipy.stats
import cv2 as cv
from .components import *


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class CoAtten2(nn.Module):
    def __init__(self, in_dim):
        super(CoAtten2, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)  #

    # x_0 guide x_1
    def forward(self, x_f, x_m, x_l):
        """
           前后做K
        """
        batch_0, C_0, width_0, height_0 = x_f.size()
        batch_1, C_1, width_1, height_1 = x_m.size()

        proj_query_m = self.query_conv1(x_m).view(batch_0, C_0, -1).permute(0, 2, 1).squeeze(0)

        proj_key_f = self.key_conv1(x_f).view(batch_1, C_1, -1).squeeze(0)  # error cases---> K_number*(512*8*8)
        proj_key_l = self.key_conv2(x_l).view(batch_1, C_1, -1).squeeze(0)  # error cases---> K_number*(512*8*8)

        energy_f = torch.mm(proj_key_f,
                            proj_query_m)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(512*8*8)) * ((512*8*8)*N_number) == K_number * N_number
        energy_l = torch.mm(proj_key_l,
                            proj_query_m)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(512*8*8)) * ((512*8*8)*N_number) == K_number * N_number
        attention_f = self.softmax(energy_f)  # the shape are K_number * N_number
        attention_l = self.softmax(energy_l)  # the shape are K_number * N_number

        proj_value_m = self.value_conv1(x_m).view(batch_0, C_0, -1).squeeze(0)  # g

        out_f = torch.mm(attention_f,
                         proj_value_m)  # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out_l = torch.mm(attention_l,
                         proj_value_m)  # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out = out_f + out_l

        out = out.view(batch_1, C_1, width_1, height_1)  # output the shape is (K_number * 512 * 8 * 8)

        out = self.gamma * out + (x_l + x_f) / 2  # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out


class CoAtten(nn.Module):
    def __init__(self, in_dim):
        super(CoAtten, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)

        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    # x_0 guide x_1
    def forward(self, x_f, x_m, x_l):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batch_0, C_0, width_0, height_0 = x_f.size()
        batch_1, C_1, width_1, height_1 = x_m.size()

        proj_query_f = self.query_conv1(x_f).view(batch_0, C_0, -1).permute(0, 2, 1).squeeze(0)
        proj_query_l = self.query_conv2(x_l).view(batch_0, C_0, -1).permute(0, 2, 1).squeeze(0)

        proj_key = self.key_conv(x_m).view(batch_1, C_1, -1).squeeze(0)  # error cases---> K_number*(512*8*8)
        energy_f = torch.mm(proj_key,
                            proj_query_f)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(512*8*8)) * ((512*8*8)*N_number) == K_number * N_number
        energy_l = torch.mm(proj_key,
                            proj_query_l)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(512*8*8)) * ((512*8*8)*N_number) == K_number * N_number
        attention_f = self.softmax(energy_f)  # the shape are K_number * N_number
        attention_l = self.softmax(energy_l)  # the shape are K_number * N_number

        proj_value_f = self.value_conv1(x_f).view(batch_0, C_0, -1).squeeze(0)  # g
        proj_value_l = self.value_conv2(x_l).view(batch_0, C_0, -1).squeeze(0)  # g

        out_f = torch.mm(attention_f,
                         proj_value_f)  # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out_l = torch.mm(attention_l,
                         proj_value_l)  # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out = out_f + out_l

        out = out.view(batch_1, C_1, width_1, height_1)  # output the shape is (K_number * 512 * 8 * 8)

        out = self.gamma * out + x_m  # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out


class SpatialAtt(nn.Module):
    def __init__(self, in_dim):
        super(SpatialAtt, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    # x_0 guide x_1
    def forward(self, x_0, x_1):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batch_0, C_0, width_0, height_0 = x_0.size()
        batch_1, C_1, width_1, height_1 = x_1.size()
        proj_query = self.query_conv(x_0).view(batch_0, -1).permute(1,
                                                                    0)  # good cases---> N_number*(512*8*8)---> (512*8*8)*N_number
        proj_key = self.key_conv(x_1).view(batch_1, -1)  # error cases---> K_number*(512*8*8)
        energy = torch.mm(proj_key,
                          proj_query)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(512*8*8)) * ((512*8*8)*N_number) == K_number * N_number
        attention = self.softmax(energy)  # the shape are K_number * N_number
        proj_value = self.value_conv(x_0).view(batch_0,
                                               -1)  # good cases, the two conv process, output a N_number*(512*8*8)

        out = torch.mm(attention,
                       proj_value)  # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out = out.view(batch_1, C_1, width_1, height_1)  # output the shape is (K_number * 512 * 8 * 8)

        out = self.gamma * out + x_1  # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out


class BlockMaskGenerator:
    def __init__(self, mask_ratio=0.95,mask_block_size=8):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size
    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape
        # input_mask = torch.rand((B,1,1,1), device=imgs.device)
        # input_mask = (input_mask < self.mask_ratio).float()

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask < self.mask_ratio).float()
        input_mask = F.interpolate(input_mask, size=(H, W), mode='nearest')


        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask



class DEAttention_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(DEAttention_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.5)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X D)
            returns :
                out : attention value + input feature
                attention: B X (HxWXD) X (HxWXD)
        """
        m_batchsize, C, width, depth = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, depth * width, -1).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, depth * width, -1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, depth * width, -1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, depth)

        out = self.gamma * out + x
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
    # x_0 guide x_1
    def forward(self, x_0, x_1):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self self.attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        # from x get the x0 and x1, x0 is the 0 sign with low entropy, x1 is the 1 sign with high entropy.
        # print("x.shape, zero_low.shape:", x.shape, torch.tensor(zero_low).cuda(async=True).shape)
        # x_0 = torch.index_select(x, 0, torch.tensor(zero_low).cuda(async=True))     # good cases
        # x_1 = torch.index_select(x, 0, torch.tensor(one_high).cuda(async=True))     # error cases
        batch_0, C_0, width_0, height_0 = x_0.size()
        batch_1, C_1, width_1, height_1 = x_1.size()
        # print("x_0.shape, x_1.shape:", x_0.shape, x_1.shape)
        proj_query = self.query_conv(x_0).view(batch_0, -1).permute(1, 0)     # good cases---> N_number*(32*64*64)---> (32*64*64)*N_number

        # print("x_0.shape:", x_0.shape)
        # print("proj_query.shape:", proj_query.shape)
        proj_key = self.key_conv(x_1).view(batch_1, -1)  # error cases---> K_number*(512*8*8)
        # print("x_1.shape:", x_1.shape)
        # print("proj_key.shape:", proj_key.shape)
        energy = torch.mm(proj_key, proj_query)  # transpose check, good cases wise dot the error cases, so should be N*K / as, (K_number*(32*64*64)) * ((32*64*64)*N_number) == K_number * N_number
        attention = self.softmax(energy)  # the shape are K_number * N_number
        proj_value = self.value_conv(x_0).view(batch_0, -1)  # good cases, the two conv process, output a N_number*(512*8*8)
        # print("proj_value.shape:", proj_value.shape)

        out = torch.mm(attention, proj_value)     # (K_number * N_number) * (N_number*(512*8*8)) output a tensor, the shape is K_number*(512*8*8)
        out = out.view(batch_1, C_1, width_1, height_1)     # output the shape is (K_number * 512 * 8 * 8)

        out = self.gamma * out + x_1     # (K_number * 512 * 8 * 8) attention + x_1 (error cases)
        return out, attention


def safe_wasserstein(u, v):
    if np.isnan(u).any() or np.isnan(v).any():
        return 0.0  # 或者 np.nan，或 return None
    return scipy.stats.wasserstein_distance(u, v)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_t2w = DoubleConv(1, 16)
        self.inc_dwi = DoubleConv(1, 16)
        self.inc_adc = DoubleConv(1, 16)

        self.skip_conv = DoubleConv(48, 32)
        self.skip_fuse = SpatialAttentionModule()
        self.modality_attention2 = ChannelTransformer(vis=False, img_size=64, dim=32,
                                                      channel_num=[32, 32, 32],
                                                      patchSize=[1, 1, 1])

        self.feature_fusion2 = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(32 * 3, 64, kernel_size=3, stride=1, bias=False, padding=1)),
            ('gn', nn.GroupNorm(64, 64, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.down1_t = Down(16, 32)
        self.down1_d = Down(16, 32)
        self.down1_a = Down(16, 32)

        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)


        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

        self.attention = Self_Attn(64)

    def forward(self, x, train=False):



        t2w = x[:, 0, :, :].unsqueeze(1)
        dwi = x[:, 1, :, :].unsqueeze(1)
        adc = x[:, 2, :, :].unsqueeze(1)


        if train:
            t2w = BlockMaskGenerator().mask_image(t2w)
            dwi = BlockMaskGenerator().mask_image(dwi)
            adc = BlockMaskGenerator().mask_image(adc)


        t2w = self.inc_t2w(t2w)
        dwi = self.inc_dwi(dwi)
        adc = self.inc_adc(adc)


        skip = torch.cat([t2w, dwi, adc], dim=1)

        skip = self.skip_conv(skip)
        skip = self.skip_fuse(skip)


        t2w = self.down1_t(t2w)
        dwi = self.down1_d(dwi)
        adc = self.down1_a(adc)


        attention_features = self.modality_attention2(t2w, dwi, adc,train)

        attention_feature = torch.cat(attention_features, dim=2)
        attention_feature = attention_feature.transpose(-1, -2)
        attention_feature = attention_feature.reshape(
            (attention_feature.shape[0], attention_feature.shape[1], int(np.sqrt(attention_feature.shape[2])), -1))

        fusion1 = self.feature_fusion2(attention_feature)



        x3 = self.down2(fusion1)
        x4 = self.down3(x3)
        x5 = self.down4(x4)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)

        # x = self.up3(x, fusion1)


        c5d_ = x  # [b,32,64,64]
        heatmap = (F.relu(x.clone())).sum(dim=1).cpu().detach().numpy()
        # cam_img = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # Normalize
        denom = heatmap.max() - heatmap.min()
        cam_img = (heatmap - heatmap.min()) / (denom + 1e-8)
        

        new_cam_img = np.zeros((cam_img.shape[0], 128, 128))
        for slice_id in range(x.shape[0]):
            CAMs = cam_img[slice_id]
            CAMs = cv.resize(CAMs, (128, 128))
            new_cam_img[slice_id] = CAMs
        new_tensor_attentioned = []
        b, c, h, w = c5d_.shape
        if train:
            c5d_ = c5d_.reshape(-1, 8, c, h, w)
        else:
            c5d_ = c5d_.reshape(1, -1, c, h, w)
        with open('txt.txt', 'a', encoding='utf-8') as f:
        
            for bs in range(c5d_.shape[0]):
                c5d = c5d_[bs]
                for i in range(c5d_.shape[1]):
                    if i == 0:
                        d1 = safe_wasserstein(new_cam_img[0].flatten(), new_cam_img[1].flatten())
                        f.write('slice 0, dis:{:.5f}\n'.format(
                            safe_wasserstein(new_cam_img[0].flatten(), new_cam_img[1].flatten())))
                        if d1 > 0.005:
                            f.write('slice 0, help\n')
        
                            tem_tensor = torch.cat(
                                [torch.unsqueeze(c5d[i + 1], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                            # print("tem_tensor.shape, c5d[i].shape:", tem_tensor.shape, c5d[i].shape)
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
                        else:
                            tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                            # print("tem_tensor.shape, c5d[i].shape:", tem_tensor.shape, c5d[i].shape)
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
                    elif i == c5d_.shape[1] - 1:
                        d2 = safe_wasserstein(new_cam_img[-1].flatten(), new_cam_img[-2].flatten())
                        f.write('slice {}, dis:{:.5f}\n'.format((c5d_.shape[1] - 1),
                                                                safe_wasserstein(
                                                                    new_cam_img[0].flatten(),
                                                                    new_cam_img[1].flatten())))
                        if d2 > 0.005:
                            # print('help')
                            f.write('slice {}, help\n'.format(c5d_.shape[1] - 1))
                            tem_tensor = torch.cat(
                                [torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i - 1], dim=0)])
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
                        else:
                            tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
        
                    else:
                        d1 = safe_wasserstein(new_cam_img[i].flatten(), new_cam_img[i - 1].flatten())
                        d2 = safe_wasserstein(new_cam_img[i].flatten(), new_cam_img[i + 1].flatten())
                        f.write('slice {}, d1:{:.5f},d2:{:.5f}\n'.format(i, d1, d2))
                        if d1 > 0.007 and d2 > 0.007:
                            # print('help')
                            f.write('help\n')
                            tem_tensor = torch.cat(
                                [torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
                        elif d1 < 0.007 and d2 > 0.007:
                            # print('help')
                            f.write('help\n')
                            tem_tensor = torch.cat([torch.unsqueeze(c5d[i - 1], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
        
                        elif d1 > 0.007 and d2 < 0.007:
                            # print('help')
                            f.write('help\n')
                            tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i + 1], dim=0)])
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
        
                        elif d1 < 0.007 and d2 < 0.007:
                            tem_tensor = torch.cat([torch.unsqueeze(c5d[i], dim=0), torch.unsqueeze(c5d[i], dim=0)])
                            out_atte, attention = self.attention(tem_tensor, torch.unsqueeze(c5d[i], dim=0))
                            new_tensor_attentioned.append(out_atte)
                        else:
                            print("???????????????????????")

        tensor_attentioned = torch.stack(new_tensor_attentioned, dim=0)
        tensor_attentioned = torch.squeeze(tensor_attentioned)
        # tensor_attentioned = torch.stack(new_tensor_attentioned, dim=0)  # [N, C, H, W]
        # tensor_attentioned = tensor_attentioned.view(B, 32, 64, 64)
        tensor_attentioned = tensor_attentioned.view(*x.shape)       


        # print(tensor_attentioned.shape)
        # print(x.shape)
        x = self.up3(tensor_attentioned, fusion1)
        x = self.up4(x, skip)
        # x = self.up4(tensor_attentioned, skip)

        return self.outc(x),x
    


