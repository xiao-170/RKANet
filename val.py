import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd

import archs

from dataset import Dataset
from metrics import (
    iou_score,
    indicators,
    normalized_surface_dice,
    average_symmetric_surface_distance,
    boundary_f1,
)
from utils import AverageMeter
import numpy as np
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    # dataset override & split controls
    parser.add_argument('--split', choices=['val', 'all'], default='val', help='use train/val split or evaluate all images')
    parser.add_argument('--img_dir', default=None, help='override images directory')
    parser.add_argument('--mask_dir', default=None, help='override masks directory')
    parser.add_argument('--img_ext', default=None, help='override image extension, e.g., .png/.jpg')
    parser.add_argument('--mask_ext', default=None, help='override mask extension, e.g., .png/_mask.png')
    # outputs
    parser.add_argument('--output', default=None, help='Optional CSV path to write summary; default outputs/<name>/val_summary.csv')
    parser.add_argument('--per_case', action='store_true', help='Also write per-image metrics CSV next to summary')
    parser.add_argument('--nsd_tolerance', type=float, default=1.0, help='NSD tolerance in pixels (default: 1.0)')
    parser.add_argument('--bf_tolerance', type=float, default=1.0, help='Boundary F1 tolerance in pixels (default: 1.0)')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create model aligned with archs.RKANet and train.py
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config.get('no_kan', False),
        use_mamba=config.get('use_mamba', True),
        mamba_d_state=config.get('mamba_d_state', 16),
        mamba_kan_mode=config.get('mamba_kan_mode', 'kan_first'),
        edge_share_enc1=config.get('edge_share_enc1', False),
        drop_rate=config.get('drop_rate', 0.0),
        drop_path_rate=config.get('drop_path_rate', 0.0),
        use_checkpoint=config.get('use_checkpoint', False),
        bi_mamba=config.get('bi_mamba', False),
        use_edge_branch=config.get('use_edge_branch', False),
        # SBR removed
    )

    model = model.cuda()

    dataset_name = str(config['dataset'])
    # default image extension (can be auto-detected later)
    img_ext = args.img_ext if args.img_ext is not None else '.png'

    ds_norm = dataset_name.strip().lower().replace(' ', '').replace('_', '').replace('-', '')
    # resolve mask extension (overrides > config > heuristic)
    if args.mask_ext is not None:
        mask_ext = str(args.mask_ext)
    elif config.get('mask_ext'):
        mask_ext = str(config['mask_ext'])
    else:
        mask_ext = '.png'
        if ds_norm in ('busi', 'busidataset', 'breastus'):
            mask_ext = '_mask.png'
        elif ds_norm in ('glas', 'gland', 'cvc', 'cvcclinicdb', 'cvcdataset', 'clinicdb'):
            mask_ext = '.png'

    # Resolve data directories similar to train.py
    base1 = os.path.join(config['data_dir'], dataset_name)
    base2 = config['data_dir']
    def _has_im_mask(base):
        return os.path.isdir(os.path.join(base, 'images')) and os.path.isdir(os.path.join(base, 'masks'))
    if _has_im_mask(base1):
        images_dir = os.path.join(base1, 'images')
        masks_dir = os.path.join(base1, 'masks')
    elif _has_im_mask(base2):
        images_dir = os.path.join(base2, 'images')
        masks_dir = os.path.join(base2, 'masks')
    else:
        images_dir = os.path.join(base1, 'images')
        masks_dir = os.path.join(base1, 'masks')
    # apply user overrides if provided
    if args.img_dir is not None:
        images_dir = args.img_dir
    if args.mask_dir is not None:
        masks_dir = args.mask_dir

    # Data loading with extension auto-detect
    def collect_ids(img_dir: str, cand_exts):
        paths_all = []
        used_ext = None
        for e in cand_exts:
            paths = sorted(glob(os.path.join(img_dir, '*' + e)))
            if paths:
                paths_all = paths
                used_ext = e
                break
        return [os.path.splitext(os.path.basename(p))[0] for p in paths_all], used_ext

    cand_exts = [img_ext] if img_ext else []
    # add common types for auto-detect
    for e in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
        if e not in cand_exts:
            cand_exts.append(e)

    img_ids, detected = collect_ids(images_dir, cand_exts)
    if not img_ids:
        raise RuntimeError(f"No images found in '{images_dir}' with extensions {cand_exts}. Use --img_dir/--img_ext to override.")
    if detected and args.img_ext is None:
        img_ext = detected
    print(f"Using images from: {images_dir} (ext: {img_ext}), count={len(img_ids)}")

    if args.split == 'val':
        _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])
    else:
        val_img_ids = img_ids

    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')

    try:        
        model.load_state_dict(ckpt)
    except:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())

        pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
        current_dict = model.state_dict()
        diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

        print("Difference in model keys:")
        for key in diff_keys:
            print(f"Key: {key}")

        model.load_state_dict(ckpt, strict=False)
        
    model.eval()

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=images_dir,
        mask_dir=masks_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    f1_avg_meter = AverageMeter()
    dsc_avg_meter = AverageMeter()
    nsd_avg_meter = AverageMeter()
    assd_avg_meter = AverageMeter()
    bf1_avg_meter = AverageMeter()

    # dirs for probability heatmaps and overlays
    heat_dir = os.path.join(args.output_dir, config['name'], 'heatmaps')
    overlay_dir = os.path.join(args.output_dir, config['name'], 'overlays')
    os.makedirs(heat_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    per_rows = []

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou, dice, hd95_ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))

            # Additional metrics: F1 and DSC (Dice Similarity Coefficient)
            # Use medpy-based indicators for precision/recall and dc
            iou_m, dsc_m, hd_m, hd95_m, recall_m, specificity_m, precision_m, _ = indicators(output, target)
            eps = 1e-8
            f1_m = (2.0 * precision_m * recall_m) / (precision_m + recall_m + eps)
            f1_avg_meter.update(f1_m, input.size(0))
            dsc_avg_meter.update(dsc_m, input.size(0))

            # NSD using shared implementation with configurable tolerance
            out_bin = (torch.sigmoid(output).detach().cpu().numpy() > 0.5).astype(np.uint8)
            tgt_bin = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)
            # ensure (B,1,H,W)
            if out_bin.ndim == 3:
                out_bin = out_bin[:, None, ...]
            if tgt_bin.ndim == 3:
                tgt_bin = tgt_bin[:, None, ...]
            nsd_val = normalized_surface_dice(out_bin, tgt_bin, tolerance=args.nsd_tolerance)
            nsd_avg_meter.update(nsd_val, input.size(0))

            # ASSD and Boundary-F1
            assd_val = average_symmetric_surface_distance(out_bin, tgt_bin)
            assd_avg_meter.update(assd_val, input.size(0))
            _, _, bf1_val = boundary_f1(out_bin, tgt_bin, tolerance=args.bf_tolerance)
            bf1_avg_meter.update(bf1_val, input.size(0))

            # Save probability heatmaps and overlays
            prob = torch.sigmoid(output).detach().cpu().numpy()  # (B,1,H,W)
            for p, img_id in zip(prob, meta['img_id']):
                p2d = (p[0] * 255.0).astype(np.uint8)
                heat = cv2.applyColorMap(p2d, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(heat_dir, f"{img_id}.png"), heat)

                # overlay on the original image
                orig_path = os.path.join(images_dir, img_id + img_ext)
                orig = cv2.imread(orig_path)
                if orig is not None:
                    if (orig.shape[0] != p2d.shape[0]) or (orig.shape[1] != p2d.shape[1]):
                        orig = cv2.resize(orig, (p2d.shape[1], p2d.shape[0]), interpolation=cv2.INTER_LINEAR)
                    overlay = cv2.addWeighted(orig, 0.5, heat, 0.5, 0)
                    cv2.imwrite(os.path.join(overlay_dir, f"{img_id}.png"), overlay)

            # Also save thresholded predictions as before
            pred_bin = (prob >= 0.5).astype(np.uint8)
            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(pred_bin, meta['img_id']):
                pred_np = pred[0].astype(np.uint8) * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], f'out_val/{img_id}.jpg'))

            # Per-image metrics (optional)
            if args.per_case:
                for i, img_id in enumerate(meta['img_id']):
                    oi = output[i:i+1]
                    yi = target[i:i+1]
                    iou_i, dice_i, _ = iou_score(oi, yi)
                    outb_i = (torch.sigmoid(oi).detach().cpu().numpy() > 0.5).astype(np.uint8)
                    yb_i = (yi.detach().cpu().numpy() > 0.5).astype(np.uint8)
                    nsd_i = normalized_surface_dice(outb_i, yb_i, tolerance=args.nsd_tolerance)
                    assd_i = average_symmetric_surface_distance(outb_i, yb_i)
                    _, _, bf1_i = boundary_f1(outb_i, yb_i, tolerance=args.bf_tolerance)
                    per_rows.append({'img_id': img_id, 'iou': iou_i, 'dice': dice_i, 'nsd': nsd_i, 'assd': assd_i, 'bf1': bf1_i})

    
    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('HD95: %.4f' % hd95_avg_meter.avg)
    print('F1: %.4f' % f1_avg_meter.avg)
    print('DSC: %.4f' % dsc_avg_meter.avg)
    print('NSD: %.4f' % nsd_avg_meter.avg)
    print('ASSD: %.4f' % assd_avg_meter.avg)
    print('BoundaryF1: %.4f' % bf1_avg_meter.avg)

    # Write CSV summary (+ per-case if requested)
    summary = OrderedDict(
        dataset=config['dataset'],
        split=args.split,
        iou=float(iou_avg_meter.avg),
        dice=float(dice_avg_meter.avg),
        hd95=float(hd95_avg_meter.avg),
        f1=float(f1_avg_meter.avg),
        dsc=float(dsc_avg_meter.avg),
        nsd=float(nsd_avg_meter.avg),
        assd=float(assd_avg_meter.avg),
        boundary_f1=float(bf1_avg_meter.avg),
    )

    out_csv = args.output or os.path.join(args.output_dir, config['name'], 'val_summary.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_csv, index=False)
    print(f'Wrote summary: {out_csv}')
    if args.per_case:
        per_csv = out_csv.replace('.csv', '.cases.csv')
        pd.DataFrame(per_rows).to_csv(per_csv, index=False)
        print(f'Wrote per-case: {per_csv}')



if __name__ == '__main__':
    main()
