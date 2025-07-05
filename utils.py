import os
import numpy as np
import random
from tqdm import tqdm
import json

from PIL import Image, ImageEnhance
import cv2
import torch


# read image
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

# several data augumentation strategies
def cv_random_flip(img, label, edge):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge


def randomCrop(image, label, edge):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge.crop(random_region)


def randomRotation(image, label, edge):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge = edge.rotate(random_angle, mode)
    return image, label, edge


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class Normalize(object):
    def __init__(self):
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])
    
    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        return image, mask/255

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # body  = cv2.resize( body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        return image, mask 
    

# -----------------------------------

def split_ref_data(data_root, record_file='./data/refsplits.json'):
    '''
    给定收集好的Ref图片，
    划分训练和测试集，记录划分情况
    '''
    assert os.path.exists(data_root)
    os.makedirs('/'.join(record_file.split('/')[:-1]), exist_ok=True)

    refsplits = {
        'train': {},
        "test": {}
    }

    ref_image_root = os.path.join(data_root, 'Ref', 'Images')
    assert os.path.exists(ref_image_root)
    cates = os.listdir(ref_image_root)

    for cate in cates:
        ref_cate_image_dir = os.path.join(ref_image_root, cate)
        ref_cate_image_names = os.listdir(ref_cate_image_dir)
        assert len(ref_cate_image_names) == 25
        random.shuffle(ref_cate_image_names)
        ref_cate_train_samples = [name[:-4] for name in ref_cate_image_names[:20]]
        ref_cate_test_samples = [name[:-4] for name in ref_cate_image_names[20:]]

        refsplits['train'][cate] = ref_cate_train_samples
        refsplits['test'][cate] = ref_cate_test_samples

    with open(record_file, 'w') as f:
        json.dump(refsplits, f, indent=4)


def collect_r2c_data(data_root, mode='train', record_file='./data/refsplits.json'):

    if not os.path.exists(record_file):
        split_ref_data(data_root, record_file) 
    # print(data_root)
    # assert os.path.exists(data_root)
    
    camo_Imgs_dir = os.path.join(data_root, 'Camo', mode if mode == 'train'  else 'test/'+mode, 'Imgs')
    # print(camo_Imgs_dir)
    # camo_Edge_dir = os.path.join(data_root, 'Camo', mode if mode == 'train' else 'test/'+mode, 'Edge' if mode == 'train' else 'GT')
    camo_Edge_dir=os.path.join(data_root, 'Camo', mode if mode == 'train' else 'test/'+mode, 'GT')
    camo_gts_dir = os.path.join(data_root, 'Camo', mode if mode == 'train' else 'test/'+mode, 'GT')






 #kind
    # camo_Imgs_dir = os.path.join(data_root, 'Camo', mode if mode == 'train'  else 'all/Imgs/'+mode)
    # print(camo_Imgs_dir)
    # camo_Edge_dir = os.path.join(data_root, 'Camo', mode if mode == 'train' else 'Imgs'+mode, 'Edge' if mode == 'train' else 'GT')
    # camo_gts_dir = os.path.join(data_root, 'Camo', mode if mode == 'train' else 'all/GT/'+mode)



    # print(camo_Imgs_dir,camo_gts_dir)
    assert os.path.exists(camo_Imgs_dir) and os.path.exists(camo_gts_dir)

    # ref_feats_dir = os.path.join(data_root, 'Ref', 'RefFeat_ICON-R')
    # assert os.path.exists(ref_feats_dir)
    
    camo_classes = os.listdir(camo_Imgs_dir)
    # ref_classes = os.listdir(ref_feats_dir)

    # assert len(camo_classes) == len(ref_classes) == 64

    with open(record_file, 'r') as f:
        splits = json.load(f)

    image_label_list = []
    class_file_list = {}
    for c_idx in tqdm(range(len(camo_classes))):
        cate = camo_classes[c_idx]

        camo_cate_Imgs_dir = os.path.join(camo_Imgs_dir, cate)
        camo_cate_gts_dir = os.path.join(camo_gts_dir, cate)
        
        camo_cate_Edge_dir = os.path.join(camo_Edge_dir, cate)
        
        # print(camo_Imgs_dir)
        camo_img_names = sorted(os.listdir(camo_cate_Imgs_dir))
        camo_gt_names = sorted(os.listdir(camo_cate_gts_dir))
        camo_Edge_names = sorted(os.listdir(camo_cate_Edge_dir))
        # print(len(camo_Edge_names),len(camo_gt_names),len(camo_img_names))
        # print(cate)
        #assert len(camo_img_names) == len(camo_gt_names)==len(camo_Edge_names)

        image_label_list += [(os.path.join(camo_cate_Imgs_dir, camo_img_names[f_idx]), os.path.join(camo_cate_gts_dir, camo_gt_names[f_idx]),os.path.join(camo_cate_Edge_dir, camo_Edge_names[f_idx])) for f_idx in range(len(camo_img_names))]

        # ref_cate_feats_dir = os.path.join(ref_feats_dir, cate)

        # ref_cate_split_names = splits[mode if mode != 'val' else 'test'][cate]
        # class_file_list[cate] = [ref_cate_split_names[f_idx]+'.npy') for f_idx in range(len(ref_cate_split_names))]

    print('>>> {}ing with {} r2c samples'.format(mode, len(image_label_list)))
    
    return image_label_list
# import torch
# import numpy as np
# # from thop import profile
# # from thop import clever_format
# import imageio
# import torch.nn as nn
from torch.nn import functional as F
# def save_img1(img,str):
#     res2 = F.upsample(img, size=(302,400), mode='bilinear', align_corners=False)
#     res2 = res2.sigmoid().data.cpu().numpy().squeeze()
#     res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
#     image= (res2 * 255).astype(np.uint8)
#     imageio.imwrite("/media/fiona/3bab5e8f-eee1-471f-9466-f383b18459a0/zhangchenxi/demo/BGNet-master/image/" +str+".jpg", (res2 * 255).astype(np.uint8))
#     imageio.imsave("/media/fiona/3bab5e8f-eee1-471f-9466-f383b18459a0/zhangchenxi/demo/BGNet-master/image/" +str+".jpg", res2)
# def save_img(img, str):
#     print(img.size())
#     img = img.squeeze()  # 压缩为3D张量
#     print(img.size())
#     img = img.sigmoid().data.cpu().numpy()

#     img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#     img = (img * 255).astype(np.uint8)
#     imageio.imwrite("/media/fiona/3bab5e8f-eee1-471f-9466-f383b18459a0/zhangchenxi/demo/BGNet-master/image/" +str+".jpg", (img * 255).astype(np.uint8))
#     imageio.imsave("/media/fiona/3bab5e8f-eee1-471f-9466-f383b18459a0/zhangchenxi/demo/BGNet-master/image/" +str+".jpg", img)

# def clip_gradient(optimizer, grad_clip):
#     """
#     For calibrating misalignment gradient via cliping gradient technique
#     :param optimizer:
#     :param grad_clip:
#     :return:
#     """
#     for group in optimizer.param_groups:
#         for param in group['params']:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)

# def get_coef(iter_percentage, method):
#     if method == "linear":
#         milestones = (0.3, 0.7)
#         coef_range = (0, 1)
#         min_point, max_point = min(milestones), max(milestones)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         if iter_percentage < min_point:
#             ual_coef = min_coef
#         elif iter_percentage > max_point:
#             ual_coef = max_coef
#         else:
#             ratio = (max_coef - min_coef) / (max_point - min_point)
#             ual_coef = ratio * (iter_percentage - min_point)
#     elif method == "cos":
#         coef_range = (0, 1)
#         min_coef, max_coef = min(coef_range), max(coef_range)
#         normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
#         ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
#     else:
#         ual_coef = 1.0
#     return ual_coef


# def cal_ual(seg_logits, seg_gts):
#     # print(seg_logits.size(),seg_gts.size())
#     seg_logits=F.upsample(seg_logits, size=seg_gts.size()[2:], mode='bilinear', align_corners=False)
    
#     assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
#     sigmoid_x = seg_logits.sigmoid()
#     loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
#     return loss_map.mean()

# def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
#     decay = decay_rate ** (epoch // decay_epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = init_lr * decay


def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def warmup_poly(optimizer, init_lr, curr_iter, max_iter):
    warm_start_lr = 1e-7
    warm_steps = 1000

    if curr_iter<= warm_steps:
        warm_factor = (init_lr / warm_start_lr) ** (1 / warm_steps)
        warm_lr = warm_start_lr * warm_factor ** curr_iter
        for param_group in optimizer.param_groups:
            param_group['lr'] = warm_lr
    else:
        lr = init_lr * (1 - (curr_iter - warm_steps) / (max_iter - warm_steps)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
import torch
import numpy as np
from thop import profile
from thop import clever_format

def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


def cal_ual(seg_logits, seg_gts):
    seg_logits=F.upsample(seg_logits, size=seg_gts.size()[2:], mode='bilinear', align_corners=False)
    assert seg_logits.shape == seg_gts.shape, (seg_logits.shape, seg_gts.shape)
    sigmoid_x = seg_logits.sigmoid()
    loss_map = 1 - (2 * sigmoid_x - 1).abs().pow(2)
    return loss_map.mean()

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

