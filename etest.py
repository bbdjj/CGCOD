import os
import cv2
import argparse
import concurrent.futures
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

# your imports for dataset and model
from data.dataloader import get_camo_loader
from CGNet.model import CGNet as Network
from CGNet.CGD import Network as CGD
from CGNet.pvt import pvt_v2_b2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name',    type=str,   default='CHAMELEON')
    parser.add_argument('--testsize',     type=int,   default=224)
    parser.add_argument('--batchsize',    type=int,   default=4)
    parser.add_argument('--pth_path',     type=str,   required=True)
    parser.add_argument('--mask_root_tpl',type=str,   default='/path/to/{data_name}/GT')
    parser.add_argument('--save_pred_tpl',type=str,   default='/path/to/{data_name}/pred')
    return parser.parse_args()

def init_metrics():
    return {
        'FM':  Fmeasure(),
        'WFM': WeightedFmeasure(),
        'SM':  Smeasure(),
        'EM':  Emeasure(),
        'MAE': MAE(),
    }

def predict_and_save(model, loader, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for images, _, _, names, img2, _ in tqdm(loader, desc='Inference'):
            images = images.to(device)
            img2    = img2.to(device)
            # forward your model, adjust this to your forward signature
            preds, _, _ = model(images, img2, None)
            # take the final mask prediction, e.g. preds[0]
            out = F.interpolate(preds[0], size=(loader.dataset.img_size, loader.dataset.img_size),
                                mode='bilinear', align_corners=False).sigmoid().cpu().numpy()
            for i, name in enumerate(names):
                mask = (out[i] > 0.5).astype('uint8') * 255
                cv2.imwrite(os.path.join(save_dir, name + '.png'), mask)

def eval_pair(mask_path, pred_path, meters):
    gt   = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    for m in meters.values():
        m.step(pred=pred, gt=gt)

def compute_results(meters):
    fm = meters['FM'].get_results()['fm']
    em = meters['EM'].get_results()['em']
    return {
        'MAE':       meters['MAE'].get_results()['mae'],
        'wFmeasure': meters['WFM'].get_results()['wfm'],
        'Smeasure':  meters['SM'].get_results()['sm'],
        'adpEm':     em['adp'],
        'meanEm':    em['curve'].mean(),
        'maxEm':     em['curve'].max(),
        'adpFm':     fm['adp'],
        'meanFm':    fm['curve'].mean(),
        'maxFm':     fm['curve'].max(),
    }

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model
    encoder = pvt_v2_b2()
    path = './pvt_v2_b2.pth' 
    save_model = torch.load(path)
    model_dict = encoder.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    encoder.load_state_dict(model_dict)

   
    fl = [64, 128, 320, 512]
    model = Network(encoder=encoder, Net=CGD(fl=fl)).cuda()
    model.load_state_dict(torch.load(args.pth_path))
    model.to(device)

    # prepare loader
    loader = get_camo_loader(
        data_root=f'/path/to/{args.data_name}',
        shot=1,
        image_size=args.testsize,
        batch_size=args.batchsize,
        mode='test/'+args.data_name
    )
   
    # inference
    pred_dir = args.save_pred_tpl.format(data_name=args.data_name)
    predict_and_save(model, loader, pred_dir, device)

    # evaluation
    mask_dir = args.mask_root_tpl.format(data_name=args.data_name)
    mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.png'))
    meters = init_metrics()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
        futures = []
        for fname in mask_files:
            mpath = os.path.join(mask_dir, fname)
            ppath = os.path.join(pred_dir, fname)
            if not os.path.isfile(ppath):
                print(f'Warning: missing pred {ppath}')
                continue
            futures.append(exe.submit(eval_pair, mpath, ppath, meters))
        for _ in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures),
                      desc='Evaluating'):
            pass

    results = compute_results(meters)
    print(f"Results for {args.data_name}:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

if __name__ == '__main__':
    main()
