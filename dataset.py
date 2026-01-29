#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import cv2
from PIL import Image
import re
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file suffix/extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Albumentations transforms. Defaults to None.

        Expected layout (examples):
            <dataset>/
              images/
                0001.png
                0002.png
                ...
              masks/
                0001_mask.png   (single-class, flat under masks/)
                0002_mask.png
            or (multi-class):
              masks/
                0/0001.png
                1/0001.png
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to PIL for formats OpenCV may not decode (e.g., some JPEG builds)
            try:
                pil = Image.open(img_path).convert('RGB')
                img = np.array(pil)[:, :, ::-1]  # RGB -> BGR to keep consistency with cv2
            except Exception as e:
                raise FileNotFoundError(f"Failed to read image: {img_path}. Error: {e}")

        # Flexible mask loading:
        # - If num_classes==1, prefer masks/<img_id><mask_ext>,
        #   else fallback to masks/0/<img_id><mask_ext>
        # - If num_classes>1, read from masks/{i}/<img_id><mask_ext>
        if self.num_classes == 1:
            # build candidate paths in a robust order, also try stripping known tokens (e.g., '_0000')
            id_variants = []

            def sanitise_variants(s: str):
                vars = [s]
                # normalize full-width parentheses to ASCII
                s_norm = s.replace('（', '(').replace('）', ')')
                if s_norm not in vars:
                    vars.append(s_norm)
                # drop common modality suffix
                if '_0000' in s_norm:
                    v = s_norm.replace('_0000', '')
                    if v not in vars:
                        vars.append(v)
                # drop trailing bracketed suffix like (1) or (copy)
                v2 = re.sub(r"\s*\([^)]*\)\s*$", "", s_norm)
                if v2 and v2 not in vars:
                    vars.append(v2)
                # also try removing any bracketed segments anywhere (aggressive)
                v3 = re.sub(r"\([^)]*\)", "", s_norm).strip('_- ')
                if v3 and v3 not in vars:
                    vars.append(v3)
                return vars

            id_variants.extend(sanitise_variants(img_id))

            cands = []
            for img_id_try in id_variants:
                # exact as configured
                cands.append(os.path.join(self.mask_dir, img_id_try + self.mask_ext))
                cands.append(os.path.join(self.mask_dir, '0', img_id_try + self.mask_ext))
                # If mask_ext is just an extension (e.g., '.png'), try common suffixes
                if self.mask_ext.startswith('.'):
                    for suf in ('_mask', '_gt', '_seg', '_label'):
                        alt = suf + self.mask_ext
                        cands.insert(0, os.path.join(self.mask_dir, img_id_try + alt))
                        cands.insert(1, os.path.join(self.mask_dir, '0', img_id_try + alt))
                else:
                    # If mask_ext already includes a suffix (e.g., '_mask.png'), also try bare ext
                    base, ext = os.path.splitext(self.mask_ext)
                    if ext:
                        cands.append(os.path.join(self.mask_dir, img_id_try + ext))
                        cands.append(os.path.join(self.mask_dir, '0', img_id_try + ext))

            m = None
            for p in cands:
                if os.path.exists(p):
                    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        try:
                            pilm = Image.open(p).convert('L')
                            m = np.array(pilm)
                        except Exception as e:
                            raise FileNotFoundError(f"Failed to read mask: {p}. Error: {e}")
                    if m is not None:
                        break
            if m is None:
                raise FileNotFoundError(f"Mask not found for {img_id}: tried {cands}")
            # ensure 2D
            if m.ndim == 3:
                m = m[..., 0]
            mask = m[..., None]
        else:
            mask_list = []
            for i in range(self.num_classes):
                p = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Mask class {i} not found: {p}")
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    try:
                        pilm = Image.open(p).convert('L')
                        m = np.array(pilm)
                    except Exception as e:
                        raise FileNotFoundError(f"Failed to read mask: {p}. Error: {e}")
                mask_list.append(m[..., None])
            mask = np.dstack(mask_list)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # Note: Albumentations Normalize() in the train/val pipelines already
        # performs scaling/standardization. Keep raw dynamic range here.
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        if mask.max()<1:
            mask[mask>0] = 1.0

        return img, mask, {'img_id': img_id}
