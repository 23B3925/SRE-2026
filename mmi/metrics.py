"""Evaluation metrics used in the paper: SSIM (Eq. 9) and threshold-segmentation IoU (Eq. 10),
plus lesion localisation error."""
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

C3 = 1e-6


def ssim(pred, truth, mask=None, data_range=70.0):
    """SSIM between two eps' images (same grid). Computed on the bounding box of `mask`
    (the head) so the empty background does not inflate the score."""
    if mask is not None:
        ys, xs = np.nonzero(mask)
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        pred, truth = pred[sl], truth[sl]
    return float(structural_similarity(truth, pred, data_range=data_range, win_size=7))


def iou(pred_mask, true_mask):
    p, r = pred_mask.astype(bool), true_mask.astype(bool)
    return float(((p & r).sum() + C3) / ((p | r).sum() + C3))


def threshold_mask(img, thr, region=None):
    m = img >= thr
    return m if region is None else (m & region)


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def centroid_mm(mask, h_mm):
    if mask.sum() == 0:
        return None
    y, x = ndi.center_of_mass(mask)
    return np.array([x * h_mm, y * h_mm])


def localisation_error_mm(pred_mask, true_mask, h_mm):
    a, b = centroid_mm(pred_mask, h_mm), centroid_mm(true_mask, h_mm)
    return float("nan") if a is None or b is None else float(np.hypot(*(a - b)))


def half_max_mask(d_img, region=None, frac=0.5):
    """Lesion estimate from a contrast image: connected blob above frac * peak."""
    img = d_img if region is None else np.where(region, d_img, 0)
    if img.max() <= 0:
        return np.zeros_like(img, dtype=bool)
    m = img >= frac * img.max()
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    peak = np.unravel_index(np.argmax(img), img.shape)
    return lab == lab[peak]
