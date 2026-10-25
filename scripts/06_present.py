"""Presentation view: shows the saved demo reconstructions (instant, no recomputation) as one figure.
Run scripts/02_run_demo.py first (or use the shipped outputs/demo_*.pkl).
usage: python scripts/06_present.py        -> opens a window and saves outputs/presentation_summary.png"""
import sys, os, pickle, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib.pyplot as plt
from mmi.pipeline import build_scene
from mmi.scene import COARSE
from mmi import render

sc = build_scene(f"{ROOT}/data")
e_b, _, _ = sc.ref_maps(COARSE, with_lesion=False)
box = render.crop_box(sc, COARSE)
cases = [("severe", 1), ("severe", 2), ("mild", 1), ("mild", 2)]
fig, ax = plt.subplots(2, 4, figsize=(18, 9.5))
for c, (sev, part) in enumerate(cases):
    f = f"{ROOT}/outputs/demo_{sev}_part{part}.pkl"
    if not os.path.exists(f): continue
    z = pickle.load(open(f, "rb")); res, L, row = z["res"], z["lesion"], z["row"]
    sc.lesions = [L]
    e_t, _, les = sc.ref_maps(COARSE, with_lesion=True)
    P, d = res["mask"], res["d_eps"]
    lim = max(5.0, float(np.percentile(np.abs(d[P]), 99.5)))
    p = render.eps_panel(ax[0, c], d, COARSE, box, f"{sev} / part {part}: DBIM δε′ (from S-params)", vmin=-lim, vmax=lim, cmap="RdBu_r", mask=P)
    render.outline(ax[0, c], les, COARSE, "k")
    render.eps_panel(ax[1, c], res["eps"], COARSE, box, f"composite ε′ (anatomy prior + δε′)\nloc. error {row['centroid_err_mm']:.1f} mm, IoU {row['iou_halfmax']:.2f}", mask=sc.mask_c)
    render.outline(ax[1, c], les, COARSE, "w")
fig.suptitle("Microwave stroke imaging: differential DBIM, 8 antennas, 3.0-3.5 GHz, 100 dB dynamic range\n(outline = true lesion; anatomy is from the phantom, only the lesion is reconstructed from data)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{ROOT}/outputs/presentation_summary.png", dpi=130)
out = f"{ROOT}/outputs/presentation_summary.png"
print("saved outputs/presentation_summary.png")
plt.show()
