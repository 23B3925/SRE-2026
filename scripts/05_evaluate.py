"""Evaluate reconstructions on the generated dataset: SSIM (vs ground-truth eps' map), IoU of the lesion
(threshold segmentation), centroid error.  Compares  (a) healthy-anatomy-only baseline,
(b) DBIM composite, (c) Pix2pix output if weights exist (needs torch).
usage: python scripts/05_evaluate.py"""
import sys, os, glob, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy import ndimage as ndi
from mmi import metrics

files = sorted(glob.glob(f"{ROOT}/outputs/dataset/case_*.npz"))
G = {}
try:
    import torch, importlib.util
    spec = importlib.util.spec_from_file_location("p2p", f"{ROOT}/scripts/04_train_pix2pix.py")
    p2p = importlib.util.module_from_spec(spec); sys.argv = sys.argv[:1]; spec.loader.exec_module(p2p)
    for lab in ("confused", "clear"):
        w = f"{ROOT}/outputs/weights/G_{lab}.pt"
        if os.path.exists(w):
            g = p2p.UNet(); g.load_state_dict(torch.load(w, map_location="cpu")); g.eval(); G[lab] = g
except Exception as e:
    print("Pix2pix stage skipped:", repr(e)[:100])

rows = []
for f in files:
    z = np.load(f, allow_pickle=True)
    inp, tgt, les, d, m = z["inp"], z["target"], z["lesion"], z["d_eps"], z["mask"]
    h = 330.0 / 256
    est = metrics.half_max_mask(d, m)
    base = tgt.copy(); base[les] = np.median(tgt[ndi.binary_dilation(les, iterations=3) & ~les & m])  # healthy-only proxy
    r = dict(file=os.path.basename(f), severity=str(z["severity"]), part=int(z["part"]), apk=str(z["apk"]),
             ssim_dbim=metrics.ssim(inp, tgt, m), ssim_healthy_only=metrics.ssim(base, tgt, m),
             iou=metrics.iou(est, les), loc_err_mm=metrics.localisation_error_mm(est, les, h))
    lab = str(z["apk"])
    if lab in G:
        with torch.no_grad():
            out = (G[lab](torch.tensor(inp / 35.0 - 1.0)[None, None].float())[0, 0].numpy() + 1.0) * 35.0
        r["ssim_pix2pix"] = metrics.ssim(out, tgt, m)
        r["iou_pix2pix"] = metrics.iou(out >= 0.5 * (out[les].max() + np.median(out[m])) , les)
    rows.append(r)
json.dump(rows, open(f"{ROOT}/outputs/eval_rows.json", "w"), indent=1)
keys = [k for k in rows[0] if k not in ("file", "severity", "part", "apk")] if rows else []
for grp in ("severity", "part"):
    for v in sorted({r[grp] for r in rows}):
        sub = [r for r in rows if r[grp] == v]
        print(grp, v, len(sub), {k: round(float(np.nanmean([r[k] for r in sub if k in r])), 3) for k in keys})
print("ALL", len(rows), {k: round(float(np.nanmean([r[k] for r in rows if k in r])), 3) for k in keys})
