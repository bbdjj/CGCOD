import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from datetime import datetime
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from PIL import Image
import torchvision.utils as vutils
from CGNet.pvt import pvt_v2_b2
from CGNet.model import CGNet as Network # 自定义网络
from CGNet.CGD import Network as CGD
from data.dataloader import get_camo_loader  # 新版数据加载器
from utils import clip_gradient, poly_lr
from loss_f import seg_loss

tmp_path = '/home/q/ours/ZCX/our_SED11/mid'
best_mae = 1
best_epoch = 0
step = 0

def train(train_loader, model, optimizer, epoch, save_path, batch_size):
    global step
    model.train()
    epoch_loss = 0

    for i, (images, gts, sal_f, name, image2, edge) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, image2, mask, edge, sal_f = images.cuda(), image2.cuda(), gts.cuda(), edge.cuda(), sal_f.cuda()

        preds, pred2, cos = model(images, image2, sal_f)
        loss = (
            seg_loss(preds[3], mask) + seg_loss(preds[1], mask) +
            seg_loss(preds[2], mask) + seg_loss(pred2[0], mask) +
            seg_loss(preds[0], mask) * 2 +  # Final prediction weighted more
            seg_loss(pred2[1], mask) + cos
        )

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        epoch_loss += loss.item()
        step += 1

        if i % 60 == 0 or i == len(train_loader):
            print(f'{datetime.now()} Epoch [{epoch}/{opt.epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
            logging.info(f'Epoch [{epoch}/{opt.epoch}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')

        if i == 1:
            visualize_predictions(images, gts, preds, pred2, epoch)

    avg_loss = epoch_loss / len(train_loader)
    logging.info(f'Epoch [{epoch}] Average Loss: {avg_loss:.4f}')

    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f'{save_path}/Net_epoch_{epoch}.pth')


def visualize_predictions(images, gts, preds, pred2, epoch):
    os.makedirs(tmp_path, exist_ok=True)
    predx1 = F.interpolate(preds[1], size=gts.shape[2:], mode='bilinear', align_corners=False)
    predx4 = F.interpolate(preds[0], size=gts.shape[2:], mode='bilinear', align_corners=False)
    x1 = F.interpolate(pred2[0], size=gts.shape[2:], mode='bilinear', align_corners=False)
    x2 = F.interpolate(pred2[1], size=gts.shape[2:], mode='bilinear', align_corners=False)

    vutils.save_image(images, f'{tmp_path}/{epoch}_rgb.jpg', normalize=True)
    vutils.save_image(gts, f'{tmp_path}/{epoch}_gts.jpg')
    vutils.save_image(predx1, f'{tmp_path}/{epoch}_pred1.jpg')
    vutils.save_image(predx4, f'{tmp_path}/{epoch}_pred4.jpg')
    vutils.save_image(x1, f'{tmp_path}/{epoch}_x1.jpg')
    vutils.save_image(x2, f'{tmp_path}/{epoch}_x2.jpg')

    imgs = [Image.open(f'{tmp_path}/{epoch}_{suffix}.jpg') for suffix in ['rgb', 'gts', 'pred1', 'x1', 'pred4']]
    combined = Image.new('RGB', (imgs[0].width, imgs[0].height * len(imgs)))
    for idx, img in enumerate(imgs):
        combined.paste(img, (0, idx * imgs[0].height))
    combined.save(f'{tmp_path}/{epoch}_grid.jpg')


def validate(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    total_mae = 0

    with torch.no_grad():
        for image, gt, sal_f, name, img2, edge in test_loader:
            image, img2, sal_f = image.cuda(), img2.cuda(), sal_f.cuda()
            pred, _, _ = model(image, img2, sal_f)
            pred = F.interpolate(pred[0], size=gt.shape[2:], mode='bilinear', align_corners=False)
            pred = pred.sigmoid().cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

            gt_np = np.asarray(gt).squeeze()
            total_mae += np.abs(pred - gt_np).mean()

    avg_mae = total_mae / len(test_loader)
    logging.info(f'Validation Epoch {epoch}, MAE: {avg_mae:.4f}')

    if avg_mae < best_mae:
        best_mae = avg_mae
        best_epoch = epoch
        torch.save(model.state_dict(), f'{save_path}/Net_epoch_best.pth')
        print(f'Best model saved at epoch {epoch}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--trainsize', type=int, default=448)
    parser.add_argument('--clipsize', type=int, default=336)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--train_root', type=str, required=True)
    parser.add_argument('--val_root', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./home/q/ours/ZCX/our_SED11/checkpoints/')
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    os.makedirs(opt.save_path, exist_ok=True)
    cudnn.benchmark = True
    logging.basicConfig(filename=f'{opt.save_path}/train.log', level=logging.INFO)
    path = '/home/q/ours/ZYL/zyl/Five/pvt_v2_b2.pth' 
    save_model = torch.load(path)
    encoder=pvt_v2_b2()
    
    model_dict = encoder.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    encoder.load_state_dict(model_dict)

   
    fl = [64, 128, 320, 512]
    model = Network(encoder=encoder, Net=CGD(fl=fl)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    train_loader = get_camo_loader(opt.train_root, opt.batchsize, mode='train', image_size=opt.trainsize,clip_size=opt.clipsize)
    val_loader = get_camo_loader(opt.val_root, 1, mode='test/CAMO',image_size=opt.trainsize,clip_size=opt.clipsize)

    for epoch in range(1, opt.epoch + 1):
        poly_lr(optimizer, opt.lr, epoch, opt.epoch)
        train(train_loader, model, optimizer, epoch, opt.save_path, opt.batchsize)
        validate(val_loader, model, epoch, opt.save_path)
