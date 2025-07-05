import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import List, Union


from .simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """Tokenize text(s) for CLIP"""
    if isinstance(texts, str):
        texts = [texts]
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result



def random_flip(image, label, edge):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
    return image, label, edge

def random_crop(image, label, edge, border=30):
    w, h = image.size
    crop_w = np.random.randint(w - border, w)
    crop_h = np.random.randint(h - border, h)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    right = (w + crop_w) // 2
    bottom = (h + crop_h) // 2
    return image.crop((left, top, right, bottom)), label.crop((left, top, right, bottom)), edge.crop((left, top, right, bottom))

def random_rotate(image, label, edge):
    if random.random() > 0.8:
        angle = random.uniform(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.BICUBIC)
        edge = edge.rotate(angle, Image.BICUBIC)
    return image, label, edge

def color_enhance(image):
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color, ImageEnhance.Sharpness]
    factors = [random.uniform(0.5, 1.5) for _ in enhancers]
    for enhancer, factor in zip(enhancers, factors):
        image = enhancer(image).enhance(factor)
    return image

def random_pepper(img, noise_ratio=0.0015):
    img = np.array(img)
    noise_num = int(noise_ratio * img.shape[0] * img.shape[1])
    for _ in range(noise_num):
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        img[x, y] = random.choice([0, 255])
    return Image.fromarray(img)



class CamoClassDataset(Dataset):
    def __init__(self, data_root, mode='train', image_size=352, clip_size=336, shot=5):
        super().__init__()
        self.data_root = os.path.join(data_root, "Camo")
        self.mode = mode
        self.shot = shot
        self.image_size = image_size
        self.clip_size = clip_size

        self.data_list = self._collect_data()

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.clip_transform = transforms.Compose([
            transforms.Resize((clip_size, clip_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        self.edge_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def _collect_data(self):
        img_dir = os.path.join(self.data_root, self.mode, "Imgs")
        gt_dir = os.path.join(self.data_root, self.mode, "GT")
        assert os.path.exists(img_dir) and os.path.exists(gt_dir)

        categories = sorted(os.listdir(img_dir))
        data = []
        for category in categories:
            img_cat_dir = os.path.join(img_dir, category)
            gt_cat_dir = os.path.join(gt_dir, category)
            img_files = sorted(os.listdir(img_cat_dir))
            gt_files = sorted(os.listdir(gt_cat_dir))
            for img_file, gt_file in zip(img_files, gt_files):
                data.append((
                    os.path.join(img_cat_dir, img_file),
                    os.path.join(gt_cat_dir, gt_file),
                    category
                ))
        print(f">>> Loaded {len(data)} samples from {len(categories)} classes for {self.mode}")
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path, gt_path, category = self.data_list[index]
        name = os.path.basename(img_path).split('.')[0]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(gt_path).convert('L')

        edge = np.array(label)
        edge = cv2.Canny(edge, 100, 200)
        edge = Image.fromarray(edge)

        if self.mode == 'train':
            image, label, edge = random_flip(image, label, edge)
            image, label, edge = random_crop(image, label, edge)
            image, label, edge = random_rotate(image, label, edge)
            image = color_enhance(image)
            label = random_pepper(label)
            edge = random_pepper(edge)

        image1 = self.img_transform(image)
        image2 = self.clip_transform(image)
        label = self.label_transform(label)
        edge = self.edge_transform(edge)

        class_prompt = f"A photo of a camouflaged {category}"
        class_tokens = tokenize(class_prompt, 77, truncate=True).squeeze(0)

        return image1, label, class_tokens, name, image2, edge


def get_camo_loader(data_root, batch_size=4, mode='train', num_workers=4, image_size=352, clip_size=336, shot=5, shuffle=True):
    dataset = CamoClassDataset(data_root, mode, image_size, clip_size, shot)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


if __name__ == "__main__":
    root_dir = "/home/q/ours/ZCX/our_SED1/CODclass"
    loader = get_camo_loader(root_dir, batch_size=2, mode='train',image_size=448,clip_size=336)
    for i, (img1, label, tokens, name, img2, edge) in enumerate(loader):
        print(f"[{i}] Image1: {img1.shape}, Label: {label.shape}, Edge: {img2.shape}, Tokens: {tokens.shape}, Name: {name}")
        if i == 2: break
