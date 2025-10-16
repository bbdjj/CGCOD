import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from .clip336 import  ViTCLIP

from .CGD import Network
from .pvt import pvt_v2_b2



def cosine_similarity_loss(text_features, visual_features):
    """Cosine Similarity Loss"""
    text_features = F.normalize(text_features, p=2, dim=1)
    visual_features = F.normalize(visual_features, p=2, dim=1)
    cosine_sim = torch.sum(text_features * visual_features, dim=1)
    return -cosine_sim.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

from  .lib import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextToPixelContrastive(nn.Module):
    def __init__(self, img_feat_dim, text_feat_dim, proj_dim):
        super().__init__()
        self.img_proj = nn.Linear(img_feat_dim, proj_dim
        self.text_proj = nn.Linear(text_feat_dim, proj_dim)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(self, Fc, Fs, gt_mask):

        B, C, H4, W4 = Fc.shape
        N = H4 * W4
        

        Fc_flat = Fc.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
        zv = self.img_proj(Fc_flat)                  # [B, N, D]
        zv = F.normalize(zv, dim=-1)                 # 单位化


        zt = self.text_proj(Fs)                      # [B, D]
        zt = F.normalize(zt, dim=-1)                 # 单位化
        zt = zt.unsqueeze(1)                         # [B, 1, D]


        sim = torch.sum(zt * zv, dim=-1)             # [B, N]
        sim_sigmoid = torch.sigmoid(sim)             # [B, N]

        gt_mask_down = F.interpolate(gt_mask.float(), size=(H4, W4), mode='nearest')
        gt_mask_flat = gt_mask_down.view(B, -1).long()  # [B, N], 二值化

        loss = 0
        for b in range(B):
            positive_idx = (gt_mask_flat[b] == 1)
            negative_idx = (gt_mask_flat[b] == 0)

            pos_sim = sim_sigmoid[b][positive_idx]
            neg_sim = sim_sigmoid[b][negative_idx]

            pos_loss = -torch.log(pos_sim + 1e-6).mean() if pos_sim.numel() > 0 else 0
            neg_loss = -torch.log(1 - neg_sim + 1e-6).mean() if neg_sim.numel() > 0 else 0
            loss += (pos_loss + neg_loss)

        loss = loss / B


        seg_map = sim_sigmoid.view(B, 1, H4, W4)     # [B, 1, H/4, W/4]
        seg_map_up = self.upsample(seg_map)          # [B, 1, H, W]

        return loss


class CGNet(nn.Module):
    """Cross-Modality Guided Network"""
    def __init__(self, encoder=None, feature_levels=[64, 128, 320, 512], Net=Network):
        super().__init__()
        self.clip =  ViTCLIP(
    model_name="ViT-L-14-336",
    pretrained="openai",
    use_dense_aligner=True,
    dense_layers=[16,20,24],
    # dense_args=dense_args
    
)
        # for p in self.clip.parameters():
        #  p.requires_grad = False

        self.encoder = encoder
        self.feature_levels = feature_levels
        self.hidden_dim = 768

        self.mlp_blocks = nn.ModuleList([ConvMlp(1024, self.hidden_dim) for _ in range(4)])
        self.cross_attention = CrossAttentionBlock(self.hidden_dim, guide_dim=self.hidden_dim)
        self.structure_merge_deep = StructureEnhancementBlock(512)
        self.segmentation_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.refinement_head = nn.Sequential(
            LNConvAct(512, 512, 3, 1, 1, act_name="relu"),
            nn.Conv2d(512, 1, 3, 1, 1)
        )
        self.vis_proj = ProjectionNetwork(input_dim=512, proj_dim=512)

        self.text_projection = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=512)
        self.visual_projection_mid = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=512)
        self.visual_projection_deep = ProjectionNetwork(input_dim=512, proj_dim=feature_levels[3])

        self.body_encoder = MultiLevelVisualCollaborationModule(self.hidden_dim)
        self.neck = FPN(in_channels=[self.hidden_dim]*3, out_channels=[256, 512, 1024])
        self.decoder = TransformerDecoder(num_layers=1, d_model=512)
        self.Net = Net
        self.c=TextToPixelContrastive(512,512,512)

    def get_visual_features(self, image, text_embeddings):
        visual_feats = self.clip.get_visual_feats_bchw(image,text_embeddings[0])
        visual_feats = [mlp(f) for mlp, f in zip(self.mlp_blocks, visual_feats)]
        fused_feats = self.neck(visual_feats[:-1], text_embeddings[0])
        return visual_feats, fused_feats

    def pool_features(self, features, pooling='avg'):
        return torch.mean(features, dim=1) if pooling == 'avg' else torch.max(features, dim=1)[0]

    def forward_pass(self, image, image_aux, text_embeddings,gt):
        res1, res2, res3, res_deep, fused = self.get_visual_features(image, text_embeddings)
        text_proj = self.text_projection(text_embeddings[0])

        b, c, h, w = fused.shape
        decoded = self.decoder(fused).view(b, c, h, w)
        refined = self.refinement_head(decoded)

        res1 = self.cross_attention(res1 * refined, text_embeddings[0])


        body_features = self.body_encoder(res1, res3, res2)
        segmentation_map = self.segmentation_head(body_features)
        # print(body_features.size(),decoded.size())
        

        vis_mid = F.interpolate(
            self.visual_projection_mid(body_features.reshape(b, -1, 768)).view(b, -1, h, w),
            size=decoded.shape[2:], mode='bilinear', align_corners=True
        )
        vis_deep = F.interpolate(
            self.visual_projection_deep(decoded.reshape(b, -1, 512)).view(b, -1, h, w),
            size=decoded.shape[2:], mode='bilinear', align_corners=True
        )

        enc1, enc2, enc3, enc4 = self.encoder(image_aux)
        
        vis_mid = F.interpolate(vis_mid, size=enc4.shape[2:], mode='bilinear', align_corners=True)
        vis_deep = F.interpolate(vis_deep, size=enc4.shape[2:], mode='bilinear', align_corners=True)

        # print(enc4.size(),vis_mid.size(),vis_deep.size())
        sls=[]
        sls.append(refined)
        sls.append(segmentation_map)

        merged_output = self.structure_merge_deep(enc4, [vis_mid, vis_deep])
        # print(enc4.size(),vis_mid.size(),vis_deep.size(),merged_output.size())

        final_segmentation = self.Net(enc1, enc2, enc3, enc4, merged_output)
        pred_proj=self.vis_proj(fused.view(b,-1,c))

        consistency = cosine_similarity_loss(self.pool_features(pred_proj), text_proj) * 0.2
        # c2=self.c(fused,text_proj,gt)

        return final_segmentation, sls, consistency

    def forward(self, image_aux, image, class_names,gt):
        class_embs = self.clip.get_text_embeddings(class_names)
        return self.forward_pass(image, image_aux, class_embs,gt)


def main():
    batch_size, channels, height, width = 2, 3, 336, 336
    input_image = torch.randn(batch_size, channels, height, width)
    aux_image = torch.randn(batch_size, channels, 448, 448)
    class_names = "cat"
    class_names = tokenize(class_names, 77, True)
    

    encoder_model = pvt_v2_b2()
    backbone_model = Network(fl=[64, 128, 320, 512])
    model = CGNet(encoder=encoder_model, backbone=backbone_model)

    model.train()
    final_seg, seg_map, loss = model(input_image, aux_image, class_names)

    print("Final Segmentation:", final_seg.shape)
    print("Segmentation Map:", seg_map.shape)
    print("Consistency Loss:", loss.item())


if __name__ == "__main__":
    main()
