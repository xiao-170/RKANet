import numpy as np
import torch
import torch.nn.functional as F

from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision, assd as _assd
import cv2
import numpy as np



def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2* iou) / (iou+1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    nsd_ = normalized_surface_dice(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_, nsd_


def _binary_edges(mask: np.ndarray) -> np.ndarray:
    """Compute 4-connected boundary pixels of a binary mask (H,W)->bool."""
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, kernel, iterations=1)
    edges = (m ^ er).astype(bool)
    return edges


def _nsd_2d(pred2d: np.ndarray, gt2d: np.ndarray, tolerance: float = 1.0) -> float:
    """NSD for a single 2D mask pair (H,W)."""
    p = (pred2d.astype(np.uint8) > 0).astype(np.uint8)
    g = (gt2d.astype(np.uint8) > 0).astype(np.uint8)
    if p.max() == 0 and g.max() == 0:
        return 1.0

    sp = _binary_edges(p)
    sg = _binary_edges(g)

    np_sp = int(sp.sum())
    np_sg = int(sg.sum())
    denom = np_sp + np_sg
    if denom == 0:
        inter = int((p & g).sum())
        union = int(p.sum() + g.sum())
        return (2 * inter) / union if union > 0 else 1.0

    # distance to the other surface: make edges zeros in the distance input
    dt_g = cv2.distanceTransform((~sg).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((~sp).astype(np.uint8), cv2.DIST_L2, 3)

    tp_sp = int((dt_g[sp] <= tolerance).sum())
    tp_sg = int((dt_p[sg] <= tolerance).sum())

    nsd = (tp_sp + tp_sg) / float(denom)
    return float(max(0.0, min(1.0, nsd)))


def normalized_surface_dice(pred: np.ndarray, gt: np.ndarray, tolerance: float = 1.0) -> float:
    """Batch-aware NSD. Accepts shapes (H,W), (1,H,W), (B,1,H,W) or (B,H,W).

    Computes per-sample NSD and returns the mean over batch. For single 2D inputs,
    returns the scalar NSD.
    """
    try:
        a = np.asarray(pred)
        b = np.asarray(gt)
        # Align to (B,H,W)
        if a.ndim == 2:
            a = a[None, ...]
            b = b[None, ...]
        elif a.ndim == 3:
            # could be (H,W,1) or (B,H,W); if trailing dim==1, squeeze it
            if a.shape[-1] == 1 and b.shape[-1] == 1:
                a = a[..., 0]
                b = b[..., 0]
                a = a[None, ...] if a.ndim == 2 else a
                b = b[None, ...] if b.ndim == 2 else b
        elif a.ndim == 4:
            # assume (B,C,H,W); take channel 0
            a = a[:, 0]
            b = b[:, 0]

        # iterate per-sample
        nsds = []
        for i in range(a.shape[0]):
            nsds.append(_nsd_2d(a[i], b[i], tolerance=tolerance))
        return float(np.mean(nsds)) if nsds else 0.0
    except Exception:
        return 0.0


def average_symmetric_surface_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """Wrapper for medpy ASSD with safety for empty masks. Returns mean over batch.

    Accepts shapes (H,W), (1,H,W), (B,1,H,W) or (B,H,W). Returns float.
    """
    try:
        a = np.asarray(pred)
        b = np.asarray(gt)
        # Align to (B,H,W)
        if a.ndim == 2:
            a = a[None, ...]; b = b[None, ...]
        elif a.ndim == 3:
            if a.shape[-1] == 1 and b.shape[-1] == 1:
                a = a[..., 0]; b = b[..., 0]
                a = a[None, ...] if a.ndim == 2 else a
                b = b[None, ...] if b.ndim == 2 else b
        elif a.ndim == 4:
            a = a[:, 0]; b = b[:, 0]

        vals = []
        for i in range(a.shape[0]):
            p = (a[i] > 0).astype(np.uint8)
            g = (b[i] > 0).astype(np.uint8)
            if p.max() == 0 and g.max() == 0:
                vals.append(0.0)
                continue
            if p.max() == 0 or g.max() == 0:
                # One empty, assign a large penalty (use diagonal as proxy)
                h, w = p.shape
                diag = float(np.hypot(h, w))
                vals.append(diag)
                continue
            try:
                vals.append(float(_assd(p, g)))
            except Exception:
                vals.append(0.0)
        return float(np.mean(vals)) if vals else 0.0
    except Exception:
        return 0.0


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: float = 1.0):
    """Boundary Precision/Recall/F1 within tolerance band.

    Returns (precision, recall, f1). Accepts (H,W), (1,H,W), (B,1,H,W) or (B,H,W),
    returns batch-averaged metrics.
    """
    try:
        a = np.asarray(pred)
        b = np.asarray(gt)
        # Align to (B,H,W)
        if a.ndim == 2:
            a = a[None, ...]; b = b[None, ...]
        elif a.ndim == 3:
            if a.shape[-1] == 1 and b.shape[-1] == 1:
                a = a[..., 0]; b = b[..., 0]
                a = a[None, ...] if a.ndim == 2 else a
                b = b[None, ...] if b.ndim == 2 else b
        elif a.ndim == 4:
            a = a[:, 0]; b = b[:, 0]

        kernel3 = np.ones((3, 3), np.uint8)
        band_k = np.ones((2 * int(round(tolerance)) + 1, 2 * int(round(tolerance)) + 1), np.uint8)
        precisions, recalls, f1s = [], [], []
        eps = 1e-8
        for i in range(a.shape[0]):
            p = (a[i] > 0).astype(np.uint8)
            g = (b[i] > 0).astype(np.uint8)
            sp = cv2.morphologyEx(p, cv2.MORPH_GRADIENT, kernel3)
            sg = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel3)
            if sp.sum() == 0 and sg.sum() == 0:
                precisions.append(1.0); recalls.append(1.0); f1s.append(1.0)
                continue
            band_g = cv2.dilate(sg, band_k)
            band_p = cv2.dilate(sp, band_k)
            tp_p = (sp & band_g).sum()
            tp_g = (sg & band_p).sum()
            prec = float(tp_p / (sp.sum() + eps))
            rec = float(tp_g / (sg.sum() + eps))
            f1 = 2 * prec * rec / (prec + rec + eps)
            precisions.append(prec); recalls.append(rec); f1s.append(f1)
        return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))
    except Exception:
        return 0.0, 0.0, 0.0
