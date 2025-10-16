

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import math
class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x
def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(1024), nn.ReLU(True))
class FPN(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection [b, 768] --> [b, 1024]
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])  # linear + batch norm + relu

        # fusion 1: v5 & seq -> f_5: b, 1024, 24, 24
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)  # CBR
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 24, 24
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 24, 24
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 24, 24
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # imgs: 3 x [b, 576, 768] state: [b, 768]
        v3, v4, v5 = imgs
        B, C, H, W = v3.shape
        # v3=v3.view(b,-1,C)
        # v3 = d3_to_d4(self, v3) # [b, 768, 24, 24]
        # v4 = d3_to_d4(self, v4)
        # v5 = d3_to_d4(self, v5)
        # fusion 1: b, 768, 24, 24
        # text projection: b, 768 -> b, 1024
        # out: b, 1024, 24, 24
        # print(state.size())
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        # print(f5.size(),state.size())
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 768, 24, 24
        # out: b, 512, 24, 24
        f4 = self.f2_v_proj(v4)
        f4 = self.f2_cat(torch.cat([f4, f5], dim=1))
        # fusion 3: b, 768, 24, 24
        # out: b, 256, 24, 24
        f3 = self.f3_v_proj(v3)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: 3 * [b, 768, 24, 24]
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query

        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, out_channels[1], 24, 24
        return fq
class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis):
        '''
            vis: b, 512, h, w
            txt: b, L, 512

        '''
        B, C, H, W = vis.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, vis_pos)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, vis_pos):
        '''
            vis: 24*24, b, 512
            vis_pos: 24*24, 1, 512
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # FFN
        vis2 = self.norm2(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout2(vis2)
        return vis
class StructureEnhancementBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.q_norm = LayerNorm2d(dim)
        self.kv_norm = LayerNorm2d(dim)

        self.num_heads = num_heads
        self.attn_scale = nn.Parameter(torch.zeros(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.kv_proj = nn.Conv2d(dim, 2 * dim, 1, bias=False)
        self.o_proj = nn.Conv2d(dim, dim, 1, bias=True)

        self.norm_ffn = ConvMlp(dim, dim)

    def forward(self, img_feat, aux_feats=None):
        assert isinstance(aux_feats, (list, tuple))
        # B,C,H,W

        B, C, H, W = img_feat.shape
        # print(img_feat.size())
        q = self.q_norm(img_feat)
        kv = self.kv_norm(torch.cat(aux_feats, dim=0))  # 2B,C,H,W

        q = self.q_proj(q)
        kv = self.kv_proj(kv)
        q = rearrange(q, "b (nh hd) h w -> b nh hd (h w)", nh=self.num_heads)
        kv = rearrange(kv, "b (ng nh hd) h w -> ng b nh hd (h w)", ng=2, nh=self.num_heads)
        k, v = kv.unbind(0)

        hw_norm_q = F.normalize(q, dim=-1)  # b nh hd1 hw
        hw_norm_k = F.normalize(k, dim=-1)  # 2b nh hd2 hw
        if len(aux_feats) == 1:
            attn = hw_norm_q @ hw_norm_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
            qkv = attn.softmax(dim=-1) @ v  # b nh hd1 hw
        else:
            assert len(aux_feats) == 2, len(aux_feats)
            hw_norm_tex_k, hw_norm_dep_k = hw_norm_k.chunk(2, dim=0)  # b nh hd2 hw
            tex_v, dep_v = v.chunk(2, dim=0)  # b nh hd2 hw

            tex_attn = hw_norm_q @ hw_norm_tex_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
            dep_attn = hw_norm_q @ hw_norm_dep_k.transpose(-1, -2)  # b nh hd1 hw @ b nh hw hd2 => b nh hd1 hd2
            tex_qkv = tex_attn.softmax(dim=-1) @ tex_v  # b nh hd1 hw
            dep_qkv = dep_attn.softmax(dim=-1) @ dep_v  # b nh hd1 hw
            qkv = self.attn_scale.sigmoid() * tex_qkv + (1 - self.attn_scale.sigmoid()) * dep_qkv
        qkv = rearrange(qkv, "b nh hd (h w) -> b (nh hd) h w", h=H, w=W)

        img_feat = img_feat + self.o_proj(qkv)
        img_feat = img_feat + self.norm_ffn(img_feat)
        return img_feat
class MultiLevelVisualCollaborationModule(nn.Module):
    def __init__(self, in_channels):
        """
        :param in_channels: 通道数 (F_c, F2, F3 的通道数)
        """
        super().__init__()

        self.mhsa1 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)
        self.mhsa2 = nn.MultiheadAttention(embed_dim=in_channels, num_heads=8, batch_first=True)

        self.conv3x3 = lambda: nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.refine1 = nn.Sequential(self.conv3x3(), self.conv3x3())
        self.refine2 = nn.Sequential(self.conv3x3(), self.conv3x3())
        self.refine3 = nn.Sequential(self.conv3x3(), self.conv3x3())

        # Final fusion
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)

    def forward(self, F_c, F2, F3):

        B, C, H, W = F_c.shape

        # Alignment Stage
        Fc_flat = F_c.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        F2_flat = F2.flatten(2).permute(0, 2, 1)
        F3_flat = F3.flatten(2).permute(0, 2, 1)

        Fn1, _ = self.mhsa1(Fc_flat, F2_flat, F2_flat)  # MHSA(F_c, F2)
        Fn2, _ = self.mhsa2(Fc_flat, F3_flat, F3_flat)  # MHSA(F_c, F3)

        Fn = (Fn1 + Fn2).permute(0, 2, 1).reshape(B, C, H, W)

        # Enhancement Stage
        F1n = self.refine1(Fn) + Fn
        F2n = self.refine2(Fn * F1n) + Fn * F1n
        F3n = self.refine3(Fn * F2n) + Fn * F2n

        # Fusion
        F_cat = torch.cat([F1n, F2n, F3n], dim=1)  # [B, 3C, H, W]
        Fv = self.fusion_conv(F_cat)  # [B, C, H, W]
        return Fv

class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim=None):
        super(ProjectionNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + proj_dim) // 2  # A heuristic for hidden dimension size

        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class LNConvAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, k, s=1, p=0, d=1, g=1, bias=False, act_name="relu"):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module("ln", LayerNorm2d(in_planes))
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, k, s, p, d, g, bias=bias))
        if act_name is not None:
            self.add_module(name=act_name, module=get_act_fn(act_name=act_name))

class LayerNorm2d(nn.Module):
    """
    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "idy":
        return nn.Identity()
    else:
        raise NotImplementedError


class LNConvAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, k, s=1, p=0, d=1, g=1, bias=False, act_name="relu"):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module("ln", LayerNorm2d(in_planes))
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, k, s, p, d, g, bias=bias))
        if act_name is not None:
            self.add_module(name=act_name, module=get_act_fn(act_name=act_name))

class ConvMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        mlp_times: float = 1,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = LayerNorm2d,
        bias: bool = False,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(mlp_times * in_features)

        self.norm = norm_layer(in_features) if norm_layer else nn.Identity()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
class CrossAttentionBlock(nn.Module):
    def __init__(self, feat_dim, guide_dim, num_heads=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.num_heads = num_heads

        # Norm layers
        self.guide_norm = nn.LayerNorm(guide_dim)
        self.feat_norm = LayerNorm2d(feat_dim)

        # Projection layers for text and image features
        self.g_proj = nn.Linear(guide_dim, feat_dim, bias=False)  # Text projection
        self.qkv_proj_img = nn.Conv2d(feat_dim, 3 * feat_dim, 1, bias=False)  # Image QKV projection
        self.qkv_proj_txt = nn.Linear(guide_dim, 3 * feat_dim, bias=False)  # Text QKV projection

        # Output projection
        self.o_proj_img = nn.Conv2d(feat_dim, feat_dim, 1, bias=True)  # Image output projection
        self.o_proj_txt = nn.Linear(feat_dim, guide_dim, bias=True)  # Text output projection
        self.o_proj_img2 = nn.Linear(feat_dim // num_heads, 24 * 24)
        # Feed-forward layers
        self.norm_ffn_img = ConvMlp(feat_dim, feat_dim)
        self.norm_ffn_txt = nn.Linear(guide_dim, guide_dim)

    def forward(self, img_feats, txt_feats):
        """
        img_feats: B,C,H,W (Image features)
        txt_feats: B,D (Text features, N=sequence length, D=embedding dim)
        """

        # Normalize features
        normed_img_feats = self.feat_norm(img_feats)
        normed_txt_feats = self.guide_norm(txt_feats)

        # Image QKV projection
        qkv_img = self.qkv_proj_img(normed_img_feats)
        qkv_img = rearrange(qkv_img, "b (ng nh hd) h w -> ng b nh hd (h w)", ng=3, nh=self.num_heads)
        q_img, k_img, v_img = qkv_img.unbind(0)  # Image Q, K, V

        # Text QKV projection
        qkv_txt = self.qkv_proj_txt(normed_txt_feats)
        qkv_txt = rearrange(qkv_txt, "b (nh hd) -> b nh hd", nh=self.num_heads)
        qkv_txt = rearrange(qkv_txt, "b nh (ng hd) -> ng b nh hd", ng=3, nh=self.num_heads)
        q_txt, k_txt, v_txt = qkv_txt.unbind(0)  # Text Q, K, V

        # Cross-attention: text Q attends to image K, V
        q_txt_exp = q_txt.unsqueeze(-1)  # Add spatial dimension to text queries
        attn_txt_to_img = F.softmax(
            torch.einsum("bnhd,bnhf->bnhf", F.normalize(q_txt_exp, dim=-1), F.normalize(k_img, dim=-1)), dim=-1)
        txt_on_img = attn_txt_to_img @ v_img.transpose(-1, -2)  # B, nh, D, n (Text attends to Image)

        k_txt = k_txt.unsqueeze(-1)
        attn_txt_to_img2 = F.softmax(
            torch.einsum("bnhd,bnhf->bnhf", F.normalize(k_txt, dim=-1), F.normalize(q_img, dim=-1)), dim=-1)
        txt_to_img2 = attn_txt_to_img2 @ v_img.transpose(-1, -2)  # B, nh, D, n (Image attends to Text)

        # Reshape attended features
        # print(txt_on_img.size(),txt_to_img2.size())

        img = txt_on_img + txt_to_img2
        b, c, a, b = img.shape
        # img=img.reshape(b,c*a,-1)
        # print(img.size())
        # attn_txt_on_img = img.mean(dim=-1, keepdim=True)  # 对序列维度取平均 [B, nh, hd, 1]
        attn_txt_on_img = self.o_proj_img2(img.view(img.size(0), 768, -1))  # [B, (nh * hd), 1, 1]
        # print(attn_txt_on_img.size())

        txt_on_img2 = rearrange(attn_txt_on_img, "b x (h w) -> b x h w", h=img_feats.shape[2], w=img_feats.shape[3])

        # print(img_feats.size())
        img_feats = self.o_proj_img(txt_on_img2) + img_feats

        # Apply feed-forward networks
        img_feats = img_feats + self.norm_ffn_img(img_feats)

        return img_feats


class LayerNorm2d(nn.Module):
    """
    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



def get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "idy":
        return nn.Identity()
    else:
        raise NotImplementedError


class LNConvAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, k, s=1, p=0, d=1, g=1, bias=False, act_name="relu"):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module("ln", LayerNorm2d(in_planes))
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, k, s, p, d, g, bias=bias))
        if act_name is not None:
            self.add_module(name=act_name, module=get_act_fn(act_name=act_name))
class LayerNorm2d(nn.Module):
    """
    From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



def get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "idy":
        return nn.Identity()
    else:
        raise NotImplementedError


class LNConvAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, k, s=1, p=0, d=1, g=1, bias=False, act_name="relu"):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.add_module("ln", LayerNorm2d(in_planes))
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, k, s, p, d, g, bias=bias))
        if act_name is not None:
            self.add_module(name=act_name, module=get_act_fn(act_name=act_name))
