import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

import albumentations as A
import math

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm


import archs

import losses
from dataset import Dataset

from metrics import iou_score, indicators
from typing import Optional

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os
import subprocess

from pdb import set_trace as st


ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def count_params_m(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def try_compute_flops_g(model: nn.Module, c: int, h: int, w: int) -> Optional[float]:
    """Try compute GFLOPs using thop if available. Returns None if unavailable.

    Note: thop reports MACs; we report them as GFLOPs (MACs) for consistency.
    """
    try:
        from thop import profile
        device = next(model.parameters()).device
        dummy = torch.randn(1, c, h, w, device=device)
        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        if model_was_training:
            model.train()
        gflops = macs / 1e9
        return float(gflops)
    except Exception:
        return None


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    # resume options
    parser.add_argument('--resume', default='', type=str,
                        help='path to checkpoint to resume from (checkpoint.pth)')
    parser.add_argument('--resume_auto', default=False, type=str2bool,
                        help='if true, tries outputs/<name>/checkpoint.pth')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='RKANet')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])
    parser.add_argument('--use_msaa_skip', default=False, type=str2bool,
                        help='use MSAA fusion on decoder skip connections')
    parser.add_argument('--msaa_factor', default=6.0, type=float,
                        help='MSAA internal reduction factor (lower -> wider)')
    parser.add_argument('--msaa_sa_kernel', default=3, type=int,
                        help='MSAA spatial attention kernel size')
    parser.add_argument('--msaa_use_dec1', default=True, type=str2bool,
                        help='enable MSAA at first decoder fusion (dec1)')
    parser.add_argument('--msaa_use_dec2', default=True, type=str2bool,
                        help='enable MSAA at second decoder fusion (dec2)')

    # RKANet-specific toggles
    parser.add_argument('--use_mamba', default=True, type=str2bool,
                        help='enable Mamba blocks in RKANet')
    parser.add_argument('--bi_mamba', default=False, type=str2bool,
                        help='use bidirectional Mamba (if available)')
    parser.add_argument('--mamba_d_state', default=16, type=int,
                        help='Mamba state dimension')
    parser.add_argument('--mamba_kan_mode', default='kan_first', type=str,
                        choices=['kan_first', 'mamba_first', 'parallel'],
                        help='composition of Mamba and KAN blocks per stage')
    parser.add_argument('--drop_rate', default=0.0, type=float,
                        help='dropout rate inside blocks')
    parser.add_argument('--drop_path_rate', default=0.0, type=float,
                        help='stochastic depth drop path rate')
    parser.add_argument('--use_checkpoint', default=False, type=str2bool,
                        help='enable gradient checkpointing')
    parser.add_argument('--use_edge_branch', default=False, type=str2bool,
                        help='enable input edge extraction branch')
    parser.add_argument('--edge_share_enc1', default=False, type=str2bool,
                        help='edge branch reuses encoder stage-1 feature (pre-pooling) as stem to save compute')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')

    # (cleaned) legacy V2 / parallel Mamba options removed to match archs.RKANet



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, nsd_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, nsd_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'nsd': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
                # additional metrics
                try:
                    _, _, _, _, _, _, _, nsd_ = indicators(outputs[-1], target)
                except Exception:
                    nsd_ = 0.0
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)
                try:
                    _, _, _, _, _, _, _, nsd_ = indicators(output, target)
                except Exception:
                    nsd_ = 0.0

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['nsd'].update(nsd_, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('nsd', avg_meters['nsd'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('nsd', avg_meters['nsd'].avg)])

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
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model (aligned with archs.RKANet signature)
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan'],
        use_mamba=config['use_mamba'],
        mamba_d_state=config['mamba_d_state'],
        mamba_kan_mode=config.get('mamba_kan_mode', 'kan_first'),
        drop_rate=config['drop_rate'],
        drop_path_rate=config['drop_path_rate'],
        use_checkpoint=config['use_checkpoint'],
        bi_mamba=config['bi_mamba'],
        use_edge_branch=config['use_edge_branch'],
        edge_share_enc1=config.get('edge_share_enc1', False),
    
    )

    model = model.cuda()

    # ----- model stats -----
    params_M = count_params_m(model)
    flops_G = try_compute_flops_g(model, config['input_channels'], config['input_h'], config['input_w'])

    # Print model stats after computing them
    print(f"Model Params (M): {params_M:.3f}")
    if flops_G is not None:
        print(f"Model FLOPs (G, MACs): {flops_G:.3f}")
    else:
        print("Model FLOPs (G): N/A (install 'thop' to enable)")


    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  
    

    
    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'CVC-ClinicDB':
        mask_ext = '.png'
    else:
        # Fallback for arbitrary datasets: auto-detect mask suffix
        masks_dir = os.path.join(config['data_dir'], dataset_name, 'masks')
        # prefer *_mask.png if such files exist; otherwise assume same-name .png
        if any(glob(os.path.join(masks_dir, '*_mask.png'))):
            mask_ext = '_mask.png'
        else:
            mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),   # 50% 概率水平翻转
        A.VerticalFlip(p=0.5),     # 50% 概率垂直翻转
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_nsd', []),
        ('params_M', []),
        ('flops_G', []),
    ])


    # ----- resume support -----
    start_epoch = 0
    best_iou = 0.0
    best_dice = 0.0
    trigger = 0
    ckpt_path = config.get('resume','') or ''
    if not ckpt_path and config.get('resume_auto', False):
        ckpt_path = os.path.join(output_dir, exp_name, 'checkpoint.pth')
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state.get('model', {}), strict=False)
            if 'optimizer' in state:
                try:
                    optimizer.load_state_dict(state['optimizer'])
                except Exception:
                    pass
            if 'scheduler' in state and scheduler is not None:
                try:
                    scheduler.load_state_dict(state['scheduler'])
                except Exception:
                    pass
            start_epoch = int(state.get('epoch', 0))
            best_iou = float(state.get('best_iou', 0.0))
            best_dice = float(state.get('best_dice', 0.0))
            trigger = int(state.get('trigger', 0))
            print(f"[resume] loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")
        except Exception as e:
            print(f"[resume] failed to load checkpoint '{ckpt_path}': {e}")

    try:
        for epoch in range(start_epoch, config['epochs']):
            print('Epoch [%d/%d]' % (epoch, config['epochs']))

            # train for one epoch
            train_log = train(config, train_loader, model, criterion, optimizer)
            # evaluate on validation set
            val_log = validate(config, val_loader, model, criterion)

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler.step()
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])

            if flops_G is None:
                flops_str = 'N/A'
            else:
                flops_str = f"{flops_G:.3f}"
            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f - val_nsd %.4f - params(M) %.3f - flops(G) %s'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['dice'], val_log['nsd'], params_M, flops_str))

            log['epoch'].append(epoch)
            log['lr'].append(config['lr'])
            log['loss'].append(train_log['loss'])
            log['iou'].append(train_log['iou'])
            log['val_loss'].append(val_log['loss'])
            log['val_iou'].append(val_log['iou'])
            log['val_dice'].append(val_log['dice'])
            log['val_nsd'].append(val_log['nsd'])
            log['params_M'].append(params_M)
            log['flops_G'].append(0.0 if flops_G is None else flops_G)

            pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

            my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
            my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
            my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
            my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
            my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
            my_writer.add_scalar('val/nsd', val_log['nsd'], global_step=epoch)
            # record model stats at step 0
            if epoch == start_epoch:
                my_writer.add_scalar('model/params_M', params_M, global_step=epoch)
                if flops_G is not None:
                    my_writer.add_scalar('model/flops_G', flops_G, global_step=epoch)

            my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
            my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

            trigger += 1

            if val_log['iou'] > best_iou:
                torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
                best_iou = val_log['iou']
                best_dice = val_log['dice']
                print("=> saved best model")
                print('IoU: %.4f' % best_iou)
                print('Dice: %.4f' % best_dice)
                trigger = 0

            # save latest checkpoint for resume
            ckpt = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': (scheduler.state_dict() if scheduler is not None else None),
                'best_iou': best_iou,
                'best_dice': best_dice,
                'trigger': trigger,
                'config': config,
            }
            torch.save(ckpt, os.path.join(output_dir, exp_name, 'checkpoint.pth'))

            # early stopping
            if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                print("=> early stopping")
                break

            torch.cuda.empty_cache()
    except KeyboardInterrupt:
        # Save an interrupt checkpoint to resume later without losing progress
        interrupt_ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': (scheduler.state_dict() if scheduler is not None else None),
            'best_iou': best_iou,
            'best_dice': best_dice,
            'trigger': trigger,
            'config': config,
        }
        torch.save(interrupt_ckpt, os.path.join(output_dir, exp_name, 'checkpoint.interrupted.pth'))
        print("\n[interrupt] Caught KeyboardInterrupt. Saved checkpoint.interrupted.pth for resume.")

if __name__ == '__main__':
    main()
