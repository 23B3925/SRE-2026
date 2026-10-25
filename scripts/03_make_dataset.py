"""Generate a training/test set of (DBIM image -> ground-truth eps' map) pairs.
Each sample: random severity / part / position / size / noise level / slice, forward-simulated on the
1.1 mm grid, reconstructed by differential DBIM on the 1.65 mm grid.
usage: python scripts/03_make_dataset.py N_CASES [seed]      -> outputs/dataset/case_XXXX.npz
Cost: ~1-2 min per case on a laptop CPU; run several processes with different seeds to parallelise."""
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from scipy import ndimage as ndi
from mmi.pipeline import build_scene, paper_lesion, run_case, DEFAULT_FREQS
from mmi.scene import COARSE
from mmi import apk

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
SLICES = [50, 53, 55, 58, 61]
OUT = f"{ROOT}/outputs/dataset"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(seed)
S = 256 / COARSE.n


def to256(a):
    return ndi.zoom(a.astype(np.float32), S, order=1)


scenes, healthy = {}, {}
for n in range(N):
    k = int(rng.choice(SLICES))
    if k not in scenes:
        scenes[k] = build_scene(f"{ROOT}/data", k)
    sc = scenes[k]
    sev = ["mild", "severe"][rng.integers(2)]
    part = int(rng.integers(1, 3))
    L = paper_lesion(sc, part, sev, size_scale=rng.uniform(0.8, 1.2), rotate_deg=rng.uniform(0, 360))
    if part == 1:
        L.cx_mm += rng.uniform(-5, 5); L.cy_mm += rng.uniform(-5, 5)
    dr = rng.uniform(90, 115)
    t = time.time()
    res = run_case(sc, [L], DEFAULT_FREQS, dr_db=dr, seed=int(rng.integers(1 << 30)), healthy_data=healthy.get(k))
    healthy[k] = res["healthy_data"]
    e_t, _, les = sc.ref_maps(COARSE, with_lesion=True)
    label, _ = apk.classify(res["eps"], sc, COARSE)
    np.savez_compressed(f"{OUT}/case_s{seed}_{n:04d}.npz", inp=to256(res["eps"]), target=to256(e_t),
                        lesion=to256(les) > 0.5, d_eps=to256(res["d_eps"]), mask=to256(sc.mask_c) > 0.5,
                        apk=label, severity=sev, part=part, slice=k, dr_db=dr)
    print(f"[{n+1}/{N}] slice {k} {sev} part{part} DR {dr:.0f} dB apk={label}  {time.time()-t:.0f}s", flush=True)
