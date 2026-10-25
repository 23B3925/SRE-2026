"""Synthetic measurements + Distorted Born Iterative Method (DBIM).

Data matrix D (n_ant x n_ant, complex) mimics a VNA measurement after an
air-only reflection calibration:
        D[i, j] = T_head[i, j]                      (i != j, total transfer)
        D[i, i] = T_head[i, i] - T_free[i, i]       (reflection, air-calibrated)
Subtracting the free-space self term removes the grid-dependent singular self-field
of the line sources, so the reflection terms can be used on a different grid.

Two inversion modes share one Gauss-Newton / Levenberg-Marquardt engine:

  mode='differential'  (recommended; uses a healthy reference scan)
        data   : D_les - D_healthy
        model  : T(bg + delta) - T(bg),  bg = healthy anatomy
        unknown: contrast  delta_eps', delta_sigma  inside the brain
        Modelling errors common to both scans (grid, antenna model, dispersion)
        cancel, which is what makes a 1-12 % lesion signal recoverable.

  mode='absolute'      (no reference; experimental at 3-4 GHz)
        data   : D
        model  : T(air + x)
        unknown: eps'(r), sigma(r) of the whole head, from a homogeneous start.

Unknowns are eps' and sigma at 3.5 GHz with a fixed power-law dispersion across the
sweep, so all frequencies are inverted jointly.

Update (data-space Gauss-Newton with a Gaussian-correlation prior W):
        x <- x + W J^T ( J W J^T + lam I )^-1 r
with the Jacobian from the CURRENT total fields (distorted Born):
        dT_ij / d eps_c(p) = k0^2 h^2 E_i(p) E_j(p)
"""
import time
import numpy as np
from scipy import ndimage as ndi

from .forward2d import Helmholtz2D, C0, EPS0
from .scene import (Scene, FINE, COARSE, NPML, F_REF, EXP_EPS, EXP_SIG)
from .phantom import WHITE_MATTER, GRAY_MATTER

_free_cache = {}


def free_space_T(grid, ant_ij, f_ghz):
    key = (grid.n, round(grid.h * 1e6), round(f_ghz * 1e3), tuple(ant_ij))
    if key not in _free_cache:
        ec = np.ones((grid.n, grid.n), dtype=complex)
        H = Helmholtz2D(ec, grid.h, f_ghz * 1e9, npml=NPML)
        _free_cache[key] = _T_from_fields(H.solve_points(ant_ij), ant_ij)
    return _free_cache[key]


def data_matrix(T_head, T_free):
    D = T_head.copy()
    d = np.arange(D.shape[0])
    D[d, d] = T_head[d, d] - T_free[d, d]
    return D


def _T_from_fields(F, ant_ij):
    iy = np.array([a[0] for a in ant_ij]); ix = np.array([a[1] for a in ant_ij])
    return F[:, iy, ix].T.copy()          # T[i, j] = field at antenna i from source j


def simulate_measurements(scene: Scene, freqs, with_lesion=True):
    """Noise-free data on the FINE grid: dict f -> D."""
    out = {}
    for f in freqs:
        ec = scene.eps_c_true(FINE, f, with_lesion=with_lesion)
        H = Helmholtz2D(ec, FINE.h, f * 1e9, npml=NPML)
        T = _T_from_fields(H.solve_points(scene.ant_f), scene.ant_f)
        out[f] = data_matrix(T, free_space_T(FINE, scene.ant_f, f))
    return out


def noise_std_for(D, dr_db):
    """Receiver noise std for a dynamic range of dr_db below the strongest entry."""
    return max(np.abs(M).max() for M in D.values()) * 10 ** (-dr_db / 20.0)


def add_noise(D, std, rng):
    """Add reciprocal complex white noise of the given std."""
    out = {}
    for f, M in D.items():
        n = rng.normal(size=M.shape) + 1j * rng.normal(size=M.shape)
        n = (n + n.T) / 2.0
        out[f] = M + std * n / np.sqrt(2)
    return out


def _pairs(n, use_diag=True):
    return [(i, j) for i in range(n) for j in range(i, n) if use_diag or i != j]


def _coarse_T(scene, ec, f):
    H = Helmholtz2D(ec, COARSE.h, f * 1e9, npml=NPML)
    F = H.solve_points(scene.ant_c)
    return data_matrix(_T_from_fields(F, scene.ant_c), free_space_T(COARSE, scene.ant_c, f)), F


def brain_unknown_mask(scene: Scene, grow=1):
    brain = np.isin(scene.ids_c, WHITE_MATTER + GRAY_MATTER)
    return ndi.binary_dilation(brain, iterations=grow) & scene.mask_c


def invert(scene: Scene, freqs, D_meas, D_ref=None, noise_std=0.0, mode="differential",
           n_iter=8, lam_rel=0.05, ell_mm=8.0, eps_scale=15.0, sig_scale=1.5,
           use_diag=True, unknown=None, verbose=False, tol=2e-3, init_hom=(45.0, 2.0),
           focus=0.0, focus_eps=0.05):
    n = len(scene.ant_c)
    pairs = _pairs(n, use_diag)
    diff = mode == "differential"
    P = (brain_unknown_mask(scene) if unknown is None else unknown) if diff else scene.unknown_c
    ys, xs = np.nonzero(P)
    y0, y1, x0, x1 = ys.min() - 4, ys.max() + 5, xs.min() - 4, xs.max() + 5
    Pc = P[y0:y1, x0:x1]
    pidx = np.nonzero(Pc.ravel())[0]
    nP = len(pidx)
    ell_px = ell_mm / (COARSE.h * 1e3)

    # ---- background, reference data, weights
    if diff:
        bg = {f: scene.eps_c_true(COARSE, f, with_lesion=False) for f in freqs}
        T_bg = {f: _coarse_T(scene, bg[f], f)[0] for f in freqs}
        target = {f: D_meas[f] - D_ref[f] for f in freqs}
        floor = max(noise_std, 1e-12)
        weights = {f: np.full((n, n), 1.0 / floor) for f in freqs}
        lo = np.array([-30.0, -2.5]); hi = np.array([60.0, 4.0])
        x_eps = np.zeros(P.shape); x_sig = np.zeros(P.shape)
    else:
        bg = {f: np.ones((COARSE.n, COARSE.n), dtype=complex) for f in freqs}
        T_bg = {f: np.zeros((n, n), dtype=complex) for f in freqs}
        target = D_meas
        floor = max(noise_std, 1e-6 * max(np.abs(D_meas[f]).max() for f in freqs))
        weights = {f: 1.0 / np.maximum(np.abs(D_meas[f]), floor) for f in freqs}
        x_eps = np.where(P, init_hom[0] - 1.0, 0.0)
        x_sig = np.where(P, init_hom[1], 0.0)

    def eps_model(f, xe, xs_):
        a_e = (f / F_REF) ** EXP_EPS
        a_s = (f / F_REF) ** EXP_SIG
        w = 2 * np.pi * f * 1e9
        return bg[f] + a_e * xe - 1j * a_s * xs_ / (w * EPS0)

    hist, lam, best = [], lam_rel, None
    wsp = np.ones(nP)           # spatial focusing weights (minimum-support style reweighting)
    for it in range(n_iter):
        t0 = time.time()
        fw = {f: _coarse_T(scene, eps_model(f, x_eps, x_sig), f) for f in freqs}
        mis = norm = 0.0
        for f in freqs:
            M = fw[f][0] - T_bg[f]
            for (i, j) in pairs:
                w = weights[f][i, j]
                mis += abs((target[f][i, j] - M[i, j]) * w) ** 2
                norm += abs(target[f][i, j] * w) ** 2
        rel = float(np.sqrt(mis / norm))
        hist.append(rel)
        accepted = best is None or rel < best["rel"]
        if verbose:
            print(f"  it {it:2d}  rel.misfit {rel:.4f}  lam {lam:.1e}  {'ok ' if accepted else 'REJ'} ({time.time()-t0:.1f}s)", flush=True)
        if accepted:
            if best is not None:
                lam = max(lam * 0.6, 1e-5)
                if best["rel"] - rel < tol * best["rel"]:
                    best = dict(rel=rel, xe=x_eps.copy(), xs=x_sig.copy())
                    break
            rows_e, rows_s, rr = [], [], []
            for f in freqs:
                T, F = fw[f]
                M = T - T_bg[f]
                k2h2 = (2 * np.pi * f * 1e9 / C0) ** 2 * COARSE.h ** 2
                Fc = F[:, y0:y1, x0:x1].reshape(n, -1)[:, pidx]
                a_e = (f / F_REF) ** EXP_EPS
                a_s = -1j / (2 * np.pi * f * 1e9 * EPS0) * (f / F_REF) ** EXP_SIG
                for (i, j) in pairs:
                    w = weights[f][i, j]
                    g = k2h2 * Fc[i] * Fc[j] * w
                    rows_e.append(g * a_e); rows_s.append(g * a_s)
                    rr.append((target[f][i, j] - M[i, j]) * w)
            Je = np.array(rows_e); Js = np.array(rows_s); rr = np.array(rr)
            Jr = np.block([[Je.real, Js.real], [Je.imag, Js.imag]])
            rvec = np.concatenate([rr.real, rr.imag])
            R = Jr.shape[0]
            if focus > 0 and it > 0:
                a = np.sqrt((x_eps[P] / eps_scale) ** 2 + (x_sig[P] / sig_scale) ** 2)
                if a.max() > 0:
                    a = a / a.max()
                    wsp = 0.5 * wsp + 0.5 * (a + focus_eps) ** focus
                    wsp = wsp / wsp.max()
            WJt = np.zeros((2 * nP, R))
            img = np.zeros(Pc.shape)
            for r in range(R):
                for part, sc_ in ((0, eps_scale), (1, sig_scale)):
                    img[:] = 0
                    img.ravel()[pidx] = Jr[r, part * nP:(part + 1) * nP] * sc_ * wsp
                    g = ndi.gaussian_filter(img, ell_px, mode="constant")
                    WJt[part * nP:(part + 1) * nP, r] = g.ravel()[pidx] * sc_ * wsp
            M_ = Jr @ WJt
            best = dict(rel=rel, xe=x_eps.copy(), xs=x_sig.copy(), WJt=WJt, M=M_, rvec=rvec, R=R)
        else:
            lam *= 4.0
            if lam > 10:
                break
        lamabs = lam * np.trace(best["M"]) / best["R"]
        y = np.linalg.solve(best["M"] + lamabs * np.eye(best["R"]), best["rvec"])
        dx = best["WJt"] @ y
        xe_n, xs_n = best["xe"].copy(), best["xs"].copy()
        ef = xe_n[y0:y1, x0:x1].ravel(); sf = xs_n[y0:y1, x0:x1].ravel()
        ef[pidx] += dx[:nP]; sf[pidx] += dx[nP:]
        xe_n[y0:y1, x0:x1] = ef.reshape(Pc.shape); xs_n[y0:y1, x0:x1] = sf.reshape(Pc.shape)
        if diff:
            x_eps = np.clip(xe_n, lo[0], hi[0]); x_sig = np.clip(xs_n, lo[1], hi[1])
        else:
            x_eps = np.clip(xe_n, 0.0, 79.0); x_sig = np.clip(xs_n, 0.0, 6.0)
        x_eps[~P] = 0.0; x_sig[~P] = 0.0

    xe, xs_ = best["xe"], best["xs"]
    if diff:
        e_bg, s_bg, _ = scene.ref_maps(COARSE, with_lesion=False)
        eps_img, sig_img = e_bg + xe, s_bg + xs_
    else:
        eps_img, sig_img = 1.0 + xe, xs_
    return dict(eps=eps_img, sig=sig_img, d_eps=xe, d_sig=xs_, misfit=hist, mask=P, mode=mode)
