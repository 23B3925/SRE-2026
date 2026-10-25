"""End-to-end helpers: build scene, place the paper's two lesion types, simulate, invert."""
import numpy as np
from scipy import ndimage as ndi

from .phantom import load_voxels, TissueDB, WHITE_MATTER, GRAY_MATTER, THALAMUS
from .scene import Scene, Lesion, LESION_PRESETS, FINE, COARSE, H_FINE
from .dbim import simulate_measurements, noise_std_for, add_noise, invert

DEFAULT_FREQS = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5]   # lesion signal is strongest at the low end of the band
DEFAULT_SLICE = 55


def build_scene(data_dir, k_slice=DEFAULT_SLICE, lesions=()):
    vox = load_voxels(f"{data_dir}/Head_voxel.dat")
    db = TissueDB(data_dir)
    return Scene(vox, db, k_slice, list(lesions))


def paper_lesion(scene, part, severity, size_scale=1.0, rotate_deg=0.0):
    """Lesion at one of the paper's two sensitive areas.
    part=1: ~2 cm disc in the thalamus region (centre of mass of thalamus voxels, else mid-brain)
    part=2: ~1 cm wide arc just inside the skull (ellipse, long axis tangential)"""
    pr = LESION_PRESETS[severity]
    ids = scene.ids_f
    brain = np.isin(ids, WHITE_MATTER + GRAY_MATTER)
    h = H_FINE * 1e3
    cx, cy = scene.centre_mm
    if part == 1:
        th = np.isin(ids, THALAMUS)
        if th.sum() > 20:
            yy, xx = ndi.center_of_mass(th)
        else:
            yy, xx = ndi.center_of_mass(brain)
        return Lesion(cx_mm=xx * h, cy_mm=yy * h, rx_mm=10 * size_scale, ry_mm=10 * size_scale,
                      label=f"{severity}-part1", **pr)
    # part 2: walk outward from the centre along rotate_deg to the brain edge, step 9 mm inward
    phi = np.deg2rad(rotate_deg)
    d = np.array([np.cos(phi), np.sin(phi)])
    r = np.arange(0, 150.0, 0.5)
    ix = np.clip(np.round((cx + r * d[0]) / h).astype(int), 0, brain.shape[1] - 1)
    iy = np.clip(np.round((cy + r * d[1]) / h).astype(int), 0, brain.shape[0] - 1)
    rb = r[np.nonzero(brain[iy, ix])[0].max()]
    rc = rb - 9.0
    return Lesion(cx_mm=cx + rc * d[0], cy_mm=cy + rc * d[1], rx_mm=15 * size_scale, ry_mm=5 * size_scale,
                  angle_deg=rotate_deg + 90.0, label=f"{severity}-part2", **pr)


def run_case(scene, lesions, freqs=DEFAULT_FREQS, dr_db=100.0, seed=0, healthy_data=None,
             n_iter=30, lam_rel=0.003, focus=0.5, ell_mm=8.0, verbose=False):
    """Simulate (fine grid, noisy) lesion + healthy scans and reconstruct the contrast on the coarse grid."""
    scene.lesions = list(lesions)
    D_les = simulate_measurements(scene, freqs, with_lesion=True)
    D_hea = healthy_data if healthy_data is not None else simulate_measurements(scene, freqs, with_lesion=False)
    rng = np.random.default_rng(seed)
    if dr_db is not None and dr_db < 900:
        std = noise_std_for(D_les, dr_db)
        Dl, Dh = add_noise(D_les, std, rng), add_noise(D_hea, std, rng)
    else:
        std, Dl, Dh = 0.0, D_les, D_hea
    res = invert(scene, freqs, Dl, Dh, noise_std=std, mode="differential", n_iter=n_iter,
                 lam_rel=lam_rel, focus=focus, ell_mm=ell_mm, verbose=verbose)
    res["noise_std"] = std
    res["healthy_data"] = D_hea
    return res
