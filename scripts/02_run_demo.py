"""Reconstruct mild/severe stroke at the paper's two lesion locations.
usage: python scripts/02_run_demo.py [severity ...]   (default: mild severe)
Outputs outputs/demo_<severity>_part<k>.png and outputs/demo_metrics.json"""
import sys, os, json, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib.pyplot as plt
from mmi.pipeline import build_scene, paper_lesion, run_case, DEFAULT_FREQS
from mmi.scene import COARSE
from mmi import metrics, apk, render

DR_DB = float(os.environ.get("DR_DB", 100))
sevs = sys.argv[1:] or ["mild", "severe"]
sc = build_scene(f"{ROOT}/data")
out = f"{ROOT}/outputs"
os.makedirs(out, exist_ok=True)
hea = None
allm = {}
for sev in sevs:
    for part in (1, 2):
        L = paper_lesion(sc, part, sev)
        t = time.time()
        print(f"\n=== {sev} stroke, part {part}: simulating scans and running DBIM ===", flush=True)
        res = run_case(sc, [L], DEFAULT_FREQS, dr_db=DR_DB, seed=part, healthy_data=hea, verbose=True)
        hea = res["healthy_data"]
        e_t, _, les = sc.ref_maps(COARSE, with_lesion=True)
        e_b, _, _ = sc.ref_maps(COARSE, with_lesion=False)
        d = res["d_eps"]
        est = metrics.half_max_mask(d, res["mask"])
        h = COARSE.h * 1e3
        row = dict(severity=sev, part=part, dr_db=DR_DB, freqs=DEFAULT_FREQS,
                   true_peak_deps=float((e_t - e_b).max()), recon_peak_deps=float(d.max()),
                   centroid_err_mm=metrics.localisation_error_mm(est, les, h),
                   iou_halfmax=metrics.iou(est, les),
                   ssim_composite=metrics.ssim(res["eps"], e_t, sc.mask_c),
                   ssim_healthy_only=metrics.ssim(e_b, e_t, sc.mask_c),
                   apk=apk.classify(res["eps"], sc, COARSE)[0],
                   misfit=float(res["misfit"][-1]), seconds=time.time() - t)
        allm[f"{sev}_part{part}"] = row
        print(json.dumps(row), flush=True)
        render.case_figure(sc, res, f"{out}/demo_{sev}_part{part}.png",
                           f"{sev} stroke, part {part}  |  {DR_DB:.0f} dB dynamic range, 3.0-3.5 GHz", show=True)
        pickle.dump(dict(res={k: v for k, v in res.items() if k != "healthy_data"}, lesion=L, row=row),
                    open(f"{out}/demo_{sev}_part{part}.pkl", "wb"))
        json.dump(allm, open(f"{out}/demo_metrics_{'_'.join(sevs)}.json", "w"), indent=1)

print("\nDone. Close the figure windows to exit.")
plt.show()
