from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import cv2
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(0.1)

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor,align_corners=True)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Modality_2_Attention(nn.Module):
    def __init__(self, vis ,channel_num):
        super(Modality_2_Attention, self).__init__()
        self.vis = vis
        self.KV_size = 32
        self.channel_num = channel_num
        self.num_attention_heads = 2

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()


        for _ in range(2):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)

            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)

        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)



    def forward(self, emb1,emb2):


        m1 = nn.AdaptiveMaxPool2d(32)(emb1.clone())
        m2 = nn.AdaptiveMaxPool2d(32)(emb2.clone())


        # [32,32]
        multi_head_Q1_list = []
        multi_head_Q2_list = []


        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(m1)
                multi_head_Q1_list.append(Q1)    #  是为multi-head
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(m2)
                multi_head_Q2_list.append(Q2)


        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None


        attention_scoresAB = torch.matmul(multi_head_Q1, multi_head_Q2.transpose(-1,-2))/((multi_head_Q1**2).sum().sqrt()+(multi_head_Q2**2).sum().sqrt()) if emb1 is not None else None
        attention_scoresBA = torch.matmul(multi_head_Q2, multi_head_Q1.transpose(-1,-2))/((multi_head_Q1**2).sum().sqrt()+(multi_head_Q2**2).sum().sqrt())  if emb2 is not None else None
        attention_scoresAA = torch.matmul(multi_head_Q1, multi_head_Q1.transpose(-1,-2))/((multi_head_Q1**2).sum().sqrt()+(multi_head_Q1**2).sum().sqrt())  if emb2 is not None else None

        attention_scoresAB = attention_scoresAB / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scoresBA = attention_scoresBA / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scoresAA = attention_scoresAA / math.sqrt(self.KV_size) if emb2 is not None else None

        attention_probsAB = self.softmax(self.psi(attention_scoresAB)) if emb1 is not None else None
        attention_probsBA = self.softmax(self.psi(attention_scoresBA)) if emb2 is not None else None
        attention_probsAA = self.softmax(self.psi(attention_scoresAA)) if emb2 is not None else None


        if self.vis:
            weights =  []
            weights.append(attention_probsAB.mean(1))
            weights.append(attention_probsBA.mean(1))
            weights.append(attention_scoresAA.mean(1))


        else: weights=None

        attention_probsAB = self.attn_dropout(attention_probsAB) if emb1 is not None else None
        attention_probsBA = self.attn_dropout(attention_probsBA) if emb2 is not None else None
        attention_probsAA = self.attn_dropout(attention_probsAA) if emb2 is not None else None


        attention_probsAB = attention_probsAB.mean(dim=1) if emb1 is not None else None
        attention_probsBA = attention_probsBA.mean(dim=1) if emb2 is not None else None
        attention_probsAA = attention_probsAA.mean(dim=1) if emb1 is not None else None


        context_layer1 = torch.matmul(emb1,attention_probsAB) if emb1 is not None else None
        context_layer2 = torch.matmul(emb2,attention_probsBA) if emb2 is not None else None
        context_layer3 = torch.matmul(emb1,attention_probsAA) if emb1 is not None else None

        attention_scoresAB = attention_scoresAB.abs().mean(dim=1) * 10
        attention_scoresBA = attention_scoresBA.abs().mean(dim=1) * 10
        attention_scoresAA = attention_scoresAA.abs().mean(dim=1) * 10

        # context_layer1 = context_layer1.permute(0, 2, 1).contiguous() if emb1 is not None else None
        # context_layer2 = context_layer2.permute(0, 2, 1).contiguous() if emb2 is not None else None
        # context_layer3 = context_layer3.permute(0, 2, 1).contiguous() if emb1 is not None else None

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out2(context_layer3) if emb1 is not None else None

        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb1 is not None else None


        return O1,O2,O3,attention_scoresAB,attention_scoresBA,attention_scoresAA,weights




class Attention_org(nn.Module):
    def __init__(self, vis ,channel_num,dim):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = 960
        self.channel_num = channel_num
        self.num_attention_heads = 4

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(4):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False)

            key = nn.Linear( dim*3,  dim*3, bias=False)
            value = nn.Linear(dim*3,  dim*3, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))

            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)

        self.attn_dropout = Dropout(0.1)
        self.proj_dropout = Dropout(0.1)



    def forward(self, emb1,emb2,emb3, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []

        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3)
                multi_head_Q3_list.append(Q3)

        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None

        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None


        # attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K)/((multi_head_Q1**2).sum().sqrt()+(multi_head_K**2).sum().sqrt())  if emb1 is not None else None
        # attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K)/((multi_head_Q2**2).sum().sqrt()+(multi_head_K**2).sum().sqrt())  if emb2 is not None else None
        # attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K)/((multi_head_Q3**2).sum().sqrt()+(multi_head_K**2).sum().sqrt())  if emb3 is not None else None


        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) / math.sqrt(self.KV_size) if emb3 is not None else None


        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None

        # print(attention_probs4.size())

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))

        else: weights=None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None


        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None


        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None

        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None


        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None

        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None

        return O1,O2,O3,weights




class Mlp(nn.Module):
    def __init__(self, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self, vis, channel_num,dim):
        super(Block_ViT, self).__init__()
        expand_ratio = 4
        self.attn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2],eps=1e-6)

        self.attn_norm =  LayerNorm(channel_num[2]*3,eps=1e-6)
        self.channel_attn = Attention_org(True, channel_num,dim)

        self.ffn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2],eps=1e-6)

        self.ffn1 = Mlp(channel_num[0],channel_num[0]*expand_ratio)
        self.ffn2 = Mlp(channel_num[1],channel_num[1]*expand_ratio)
        self.ffn3 = Mlp(channel_num[2],channel_num[2]*expand_ratio)



    def forward(self, emb1,emb2,emb3):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3

        for i in range(3):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None

        emb_all = self.attn_norm(emb_all)
        cx1,cx2,cx3, weights = self.channel_attn(cx1,cx2,cx3,emb_all)


        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None


        org1 = cx1
        org2 = cx2
        org3 = cx3

        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None

        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None

        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None


        return x1, x2, x3, weights


class Modality_Gate(nn.Module):
    def __init__(self, vis, channel_num, dim):
        super(Modality_Gate,self).__init__()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)

        for _ in range(4):
            layer = Modality_2_Attention(True, channel_num)
            self.layer.append(copy.deepcopy(layer))


    def forward(self, emb1, emb2):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, weights = layer_block(emb1, emb2)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None

        # emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1, emb2, attn_weights



class Encoder(nn.Module):
    def __init__(self, vis, channel_num,dim):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        # self.encoder_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        for _ in range(4):
            layer = Block_ViT(vis, channel_num,dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3):
        attn_weights = []
        for layer_block in self.layer:
            emb1,emb2,emb3, weights = layer_block(emb1,emb2,emb3)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        # emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        return emb1,emb2,emb3, attn_weights




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


class ChannelTransformer(nn.Module):
    def __init__(self, vis, img_size, dim,channel_num=[64, 128, 256], patchSize=[32, 16, 8],):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.embeddings_1 = Channel_Embeddings(self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(self.patchSize_2, img_size=img_size, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(self.patchSize_3, img_size=img_size, in_channels=channel_num[2])

        self.encoder = Encoder(vis, channel_num,dim)
        self.encoder2 = Encoder(vis, channel_num,dim)


    def forward(self,en1,en2,en3,train):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)


        emb1, emb2, emb3, attn_weights = self.encoder(emb1,emb2,emb3)  # (B, n_patch, hidden)

        encoded1, encoded2, encoded3, attn_weights = self.encoder2(emb1, emb2, emb3)

        return encoded1, encoded2, encoded3

