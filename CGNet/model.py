import sys


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from clip336 import  ViTCLIP

from CGD import Network
from pvt import pvt_v2_b2



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

from  lib import *


class CGNet(nn.Module):
    """Cross-Modality Guided Network"""
    def __init__(self, encoder=None, feature_levels=[64, 128, 320, 512], backbone=Network):
        super().__init__()
        self.clip = ViTCLIP()
        self.encoder = encoder
        self.feature_levels = feature_levels
        self.hidden_dim = 768

        self.mlp_blocks = nn.ModuleList([ConvMlp(1024, self.hidden_dim) for _ in range(4)])
        self.cross_attention = CrossAttentionBlock(self.hidden_dim, guide_dim=self.hidden_dim)

        self.structure_merge_deep = StructureEnhancementBlock(self.hid_dim)
        self.segmentation_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.refinement_head = nn.Sequential(
            LNConvAct(512, 512, 3, 1, 1, act_name="relu"),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

        self.text_projection = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=512)
        self.visual_projection_mid = ProjectionNetwork(input_dim=self.hidden_dim, proj_dim=feature_levels[3])
        self.visual_projection_deep = ProjectionNetwork(input_dim=512, proj_dim=feature_levels[3])

        self.body_encoder = MultiLevelVisualCollaborationModule(self.hidden_dim)
        self.neck = FPN(in_channels=[self.hidden_dim]*3, out_channels=[256, 512, 1024])
        self.decoder = TransformerDecoder(num_layers=1, d_model=512)
        self.backbone = backbone

    def get_visual_features(self, image, text_embeddings):
        visual_feats = self.clip.get_visual_feats_bchw(image)
        visual_feats = [mlp(f) for mlp, f in zip(self.mlp_blocks, visual_feats)]
        fused_feats = self.neck(visual_feats[:-1], text_embeddings)
        return *visual_feats, fused_feats

    def pool_features(self, features, pooling='avg'):
        return torch.mean(features, dim=1) if pooling == 'avg' else torch.max(features, dim=1)[0]

    def forward_pass(self, image, image_aux, text_embeddings):
        res1, res2, res3, res_deep, fused = self.get_visual_features(image_aux, text_embeddings)
        text_proj = self.text_projection(text_embeddings)

        b, c, h, w = fused.shape
        decoded = self.decoder(fused).view(b, c, h, w)
        refined = self.refinement_head(decoded)

        res1 = self.cross_attention(res1 * refined, text_embeddings)


        body_features = self.body_encoder([res1, res3, res2])
        segmentation_map = self.segmentation_head(body_features)

        vis_mid = F.interpolate(
            self.visual_projection_mid(body_features.view(b, -1, c)).view(b, -1, h, w),
            size=decoded.shape[2:], mode='bilinear', align_corners=True
        )
        vis_deep = F.interpolate(
            self.visual_projection_deep(decoded.view(b, -1, c)).view(b, -1, h, w),
            size=decoded.shape[2:], mode='bilinear', align_corners=True
        )

        enc1, enc2, enc3, enc4 = self.encoder(image)
        merged_output = self.structure_merge_deep(enc4, [vis_mid, vis_deep])
        final_segmentation = self.backbone(enc1, enc2, enc3, enc4, merged_output)

        consistency = cosine_similarity_loss(self.pool_features(fused), text_proj) * 0.2

        return final_segmentation, segmentation_map, consistency

    def forward(self, image, image_aux, class_names):
        _, class_embs = self.clip.get_text_embs(class_names)
        return self.forward_pass(image, image_aux, class_embs)


def main():
    batch_size, channels, height, width = 2, 3, 336, 336
    input_image = torch.randn(batch_size, channels, height, width)
    aux_image = torch.randn(batch_size, channels, 448, 448)
    class_names = ["cat", "dog"]

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
