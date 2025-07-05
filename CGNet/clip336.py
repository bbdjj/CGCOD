import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F



from .lib import tokenize 
class ViTCLIP(nn.Module):
    def __init__(self, model_name="ViT-L-14-336", pretrained="openai"):
        super().__init__()
        self.clip_model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def device(self):
        return next(self.clip_model.parameters()).device
    @property
    def dtype(self):
        return self.clip_model.visual.conv1.weight.dtype

    @torch.no_grad()
    def get_text_embeddings(self, text_tokens, normalize=True):
        """Get text embeddings given tokenized input"""
        self.eval()
        
        
        cast_dtype = self.clip_model.transformer.get_cast_dtype()
        
        x = self.clip_model.token_embedding(text_tokens).to(self.dtype)
        x += self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x)
        text_embs = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(text_embs, dim=-1) if normalize else text_embs
    

    
    @torch.no_grad()
    def get_visual_feats_bchw(self, x):
        vit_model = self.clip_model.visual
        intermediate_features = []
        # 获取patch特征
        x = vit_model.conv1(x)  # 初始卷积，获取patch嵌入
        batch_size, num_channels, h_patches, w_patches = x.shape  # x shape = [B, C, H_patches, W_patches]

        # 将特征展平，准备输入到 Transformer
        x = x.reshape(batch_size, num_channels, -1)  # shape = [B, C, H_patches * W_patches]
        x = x.permute(0, 2, 1)  # shape = [B, H_patches * W_patches, C]

        # 添加分类 token
        x = torch.cat([vit_model.class_embedding.unsqueeze(0).expand(x.shape[0], 1, -1).to(x.dtype), x], dim=1)

        # 添加位置嵌入
        x = x + vit_model.positional_embedding.to(x.dtype)

        # 预处理归一化
        x = vit_model.ln_pre(x)
        x = x.permute(1, 0, 2)  # 转换为 LND 格式

        # 传入 Transformer 块
        # laion2b_s29b_b131k_ft_soup
        selected_layers = [8, 16, 24]
        xx = x
        for i, blk in enumerate(vit_model.transformer.resblocks):
            xx = blk(xx)

            x = xx.permute(1, 0, 2)  # [L, B, C] -> [B, L, C]
            patch_features = x[:, 1:, :]  # 忽略CLS token，保留patch特征 [B, H_patches * W_patches, C]

            # 将patch特征还原为图像形式
            if (i + 1) in selected_layers:
                patch_features = patch_features.permute(0, 2, 1)  # [B, C, H_patches * W_patches]
                patch_features = patch_features.reshape(batch_size, num_channels, h_patches,
                                                        w_patches)  # [B, C, H_patches, W_patches]
                intermediate_features.append(patch_features)  # 保存每层特征
            #  print(patch_features.size())

        x = xx.permute(1, 0, 2)  # [L, B, C] -> [B, L, C]
        patch_features = x[:, 1:, :]
        patch_features = patch_features.permute(0, 2, 1)  # [B, C, H_patches * W_patches]
        patch_features = patch_features.reshape(batch_size, num_channels, h_patches,
                                                w_patches)  # [B, C, H_patches, W_patches]
        intermediate_features.append(patch_features)  # 保存每层特征

        return intermediate_features  # [B, C, H, W] 格式

    @torch.no_grad()
    def get_visual_embedding(self, images, normalize=True):
        """Get final visual embedding from ViT"""
        self.eval()
        x = self.clip_model.visual(images)
        return F.normalize(x, dim=-1) if normalize else x


if __name__ == "__main__":
    # Test ViTCLIP visual feature extraction
    model = ViTCLIP()
    dummy_img = torch.randn(1, 3, 224, 224)
    features = model.get_visual_features_bchw(dummy_img)
    print(f"Extracted {len(features)} feature maps with shapes:", [f.shape for f in features])
