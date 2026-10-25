"""Reconstruct from REAL CST S-parameter files (healthy + lesion scan).
usage: python scripts/07_reconstruct_from_s8p.py data/healthy.s8p data/mild.s8p [tag]

Calibration (relative-change transfer): the 3-D CST helmet cannot be matched to the 2-D model in absolute terms,
so the measured per-port-pair, per-frequency RELATIVE change  R = S_lesion / S_healthy  is applied to the modelled
healthy data:  D_lesion_equiv = D_model_healthy * R.  The same differential DBIM then inverts it.
A consistency check reports how well a localized lesion can explain the data; if the misfit is high the
files are NOT consistent with a localized scatterer and the image must not be interpreted as a lesion."""
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib.pyplot as plt
from mmi.pipeline import build_scene, DEFAULT_FREQS
from mmi.sparams import read_touchstone
from mmi.dbim import simulate_measurements, invert
from mmi.scene import COARSE
from mmi.render import eps_panel, crop_box

hf, lf = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(lf))[0]
f, H = read_touchstone(hf); _, L = read_touchstone(lf)
sc = build_scene(f"{ROOT}/data"); sc.lesions = []
Dh = simulate_measurements(sc, DEFAULT_FREQS, with_lesion=False)
Dl, off = {}, ~np.eye(8, dtype=bool)
for fr in DEFAULT_FREQS:
    i = int(round((fr - f[0]) * 1000)); Dl[fr] = Dh[fr] * (L[i] / H[i])
spread = np.mean([np.abs(L[int(round((fr - f[0]) * 1000))] / H[int(round((fr - f[0]) * 1000))])[off].std() for fr in DEFAULT_FREQS])
print(f"spatial spread of |ratio| over port pairs: {spread:.2e}  (0 = identical scaling for all pairs = no spatial information)")
t = time.time()
res = invert(sc, DEFAULT_FREQS, Dl, Dh, noise_std=0.0, mode="differential", n_iter=30, lam_rel=0.003, focus=0.5, ell_mm=8.0)
mis = res["misfit"][-1]
ok = mis < 0.3 and spread > 1e-3
print(f"final relative misfit {mis:.2f}  ->  {'consistent with a localized lesion' if ok else 'NOT consistent with a localized lesion'}  ({time.time()-t:.0f}s)")
d, P = res["d_eps"], res["mask"]
fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
lim = max(5.0, float(np.percentile(np.abs(d[P]), 99.5)))
p = eps_panel(ax[0], d, COARSE, crop_box(sc, COARSE), f"δε′ reconstructed from {os.path.basename(lf)}", vmin=-lim, vmax=lim, cmap="RdBu_r", mask=P)
fig.colorbar(p, ax=ax[0], fraction=0.046)
ax[1].axis("off")
msg = (f"spatial spread of |S_lesion/S_healthy| over port pairs: {spread:.1e}\nfinal relative misfit: {mis:.2f}\n\n" +
       ("Data are consistent with a localized lesion;\nimage can be interpreted." if ok else
        "WARNING\nThis file is the healthy scan multiplied by one\nconstant. A constant carries no position information,\nso no localized lesion can explain it (large misfit).\nThe image on the left is an artifact: do NOT read it as a lesion.\n\nExport a CST run with the lesion actually\nplaced in the head model."))
ax[1].text(0.02, 0.95, msg, va="top", fontsize=11, color="#b00020" if not ok else "k", family="monospace")
fig.tight_layout(); out = f"{ROOT}/outputs/real_{tag}.png"; fig.savefig(out, dpi=120); print("saved", out)
if sys.platform.startswith("win"): os.startfile(out)
