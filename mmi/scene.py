"""Scene = one axial slice of the voxel head + 8 line-source antennas + optional lesion(s).

Two node-centred grids cover the same canvas so that data can be generated on a
FINE grid and inverted on a COARSE grid (avoids the 'inverse crime'):

    fine   h = 1.1 mm, 300 x 300 nodes   (native voxel pitch)
    coarse h = 1.65 mm, 200 x 200 nodes

Antennas sit on a 3.3 mm lattice (= 3 fine = 2 coarse pitches) so both grids
place every antenna at exactly the same physical point.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy import ndimage as ndi

from .phantom import (TissueDB, complex_permittivity, tissue_class_map,
                      WHITE_MATTER, GRAY_MATTER)

CANVAS_MM = 330.0
VOX_PITCH_MM = 1.1
VOX_N = 256
H_FINE = 1.1e-3
H_COARSE = 1.65e-3
N_FINE = 300
N_COARSE = 200
NPML = 12
F_REF = 3.5          # reference frequency (GHz) at which unknown maps are defined
EXP_EPS = -0.10      # assumed global dispersion exponents used by the inversion
EXP_SIG = 1.20
LATTICE_MM = 3.3


@dataclass
class Lesion:
    cx_mm: float            # centre in canvas coordinates (mm)
    cy_mm: float
    rx_mm: float = 10.0     # semi-axes (mm); 10 mm radius -> 2 cm diameter (paper "Part 1")
    ry_mm: float = 10.0
    angle_deg: float = 0.0
    eps_ref: float = 64.8   # relative permittivity at 3.5 GHz (user's severe target)
    sig_ref: float = 3.5    # S/m at 3.5 GHz (blood-like)
    label: str = "severe"


LESION_PRESETS = {
    "mild":   dict(eps_ref=52.4, sig_ref=2.5),
    "severe": dict(eps_ref=64.8, sig_ref=3.5),
}


@dataclass
class Grid:
    h: float
    n: int

    @property
    def x_mm(self):
        return np.arange(self.n) * self.h * 1e3


FINE = Grid(H_FINE, N_FINE)
COARSE = Grid(H_COARSE, N_COARSE)


class Scene:
    def __init__(self, vox, db: TissueDB, k_slice, lesions=(), standoff_mm=12.0, n_ant=8,
                 centre_shift_mm=(0.0, 0.0)):
        self.vox_slice = vox[k_slice]
        self.k = k_slice
        self.db = db
        self.lesions = list(lesions)
        self.n_ant = n_ant
        self.off_mm = (CANVAS_MM - VOX_N * VOX_PITCH_MM) / 2.0
        self.off_mm = np.array([self.off_mm + centre_shift_mm[0], self.off_mm + centre_shift_mm[1]])
        self._cache = {}
        self.ids_f = self._ids_on(FINE)
        self.ids_c = self._ids_on(COARSE)
        self.mask_f = ndi.binary_fill_holes(self.ids_f != 0)
        self.mask_c = ndi.binary_fill_holes(self.ids_c != 0)
        self.antennas_mm = self._place_antennas(standoff_mm)
        self.ant_f = [(int(round(y / (H_FINE * 1e3))), int(round(x / (H_FINE * 1e3)))) for x, y in self.antennas_mm]
        self.ant_c = [(int(round(y / (H_COARSE * 1e3))), int(round(x / (H_COARSE * 1e3)))) for x, y in self.antennas_mm]
        # unknown region for the inversion: head mask dilated by 2 coarse pixels
        self.unknown_c = ndi.binary_dilation(self.mask_c, iterations=2)

    # ---------------------------------------------------------------- geometry
    def _ids_on(self, g: Grid):
        x = g.x_mm
        ix = np.floor((x - self.off_mm[0]) / VOX_PITCH_MM).astype(int)
        iy = np.floor((x - self.off_mm[1]) / VOX_PITCH_MM).astype(int)
        okx = (ix >= 0) & (ix < VOX_N)
        oky = (iy >= 0) & (iy < VOX_N)
        out = np.zeros((g.n, g.n), dtype=np.uint8)
        sub = self.vox_slice[np.clip(iy, 0, VOX_N - 1)[:, None], np.clip(ix, 0, VOX_N - 1)[None, :]]
        out[:] = np.where(oky[:, None] & okx[None, :], sub, 0)
        return out

    def _place_antennas(self, standoff_mm):
        yy, xx = np.nonzero(self.mask_f)
        cx, cy = xx.mean() * H_FINE * 1e3, yy.mean() * H_FINE * 1e3
        self.centre_mm = (cx, cy)
        pts = []
        for m in range(self.n_ant):
            phi = 2 * np.pi * m / self.n_ant
            d = np.array([np.sin(phi), -np.cos(phi)])      # phi=0 -> front (-y)
            r = np.arange(0, 200.0, 0.5)
            px = cx + r * d[0]; py = cy + r * d[1]
            ix = np.clip(np.round(px / (H_FINE * 1e3)).astype(int), 0, N_FINE - 1)
            iy = np.clip(np.round(py / (H_FINE * 1e3)).astype(int), 0, N_FINE - 1)
            inside = self.mask_f[iy, ix]
            rb = r[np.nonzero(inside)[0].max()]
            ax = cx + (rb + standoff_mm) * d[0]
            ay = cy + (rb + standoff_mm) * d[1]
            ax = round(ax / LATTICE_MM) * LATTICE_MM
            ay = round(ay / LATTICE_MM) * LATTICE_MM
            pts.append((ax, ay))
        return pts

    # ---------------------------------------------------------------- materials
    def _grid_ids(self, g):
        return self.ids_f if g is FINE else self.ids_c

    def lesion_mask(self, g: Grid, which=None):
        x = g.x_mm
        X, Y = np.meshgrid(x, x)
        out = np.zeros((g.n, g.n), dtype=bool)
        for L in (self.lesions if which is None else which):
            c, s = np.cos(np.deg2rad(L.angle_deg)), np.sin(np.deg2rad(L.angle_deg))
            u = (X - L.cx_mm) * c + (Y - L.cy_mm) * s
            v = -(X - L.cx_mm) * s + (Y - L.cy_mm) * c
            out |= (u / L.rx_mm) ** 2 + (v / L.ry_mm) ** 2 <= 1.0
        return out

    def ref_maps(self, g: Grid, with_lesion=True):
        """eps' and sigma at F_REF (3.5 GHz) on grid g, plus lesion mask."""
        ids = self._grid_ids(g)
        e, s = self.db.lut(F_REF)
        eps, sig = e[ids].copy(), s[ids].copy()
        les = np.zeros(ids.shape, dtype=bool)
        if with_lesion:
            brain = np.isin(ids, WHITE_MATTER + GRAY_MATTER)
            for L in self.lesions:
                m = self.lesion_mask(g, [L]) & brain
                eps[m] = L.eps_ref
                sig[m] = L.sig_ref
                les |= m
        return eps, sig, les

    def eps_c_true(self, g: Grid, f_ghz, with_lesion=True):
        """'Real' (table-interpolated) complex permittivity at frequency f for the data side."""
        ids = self._grid_ids(g)
        e, s = self.db.lut(f_ghz)
        eps, sig = e[ids].copy(), s[ids].copy()
        if with_lesion:
            brain = np.isin(ids, WHITE_MATTER + GRAY_MATTER)
            for L in self.lesions:
                m = self.lesion_mask(g, [L]) & brain
                eps[m] = L.eps_ref * (f_ghz / F_REF) ** EXP_EPS
                sig[m] = L.sig_ref * (f_ghz / F_REF) ** EXP_SIG
        return complex_permittivity(eps, sig, f_ghz)

    def class_map(self, g: Grid):
        return tissue_class_map(self._grid_ids(g))


def eps_c_from_params(eps_ref, sig_ref, f_ghz):
    """Inversion-side model: fixed power-law dispersion about F_REF."""
    eps = eps_ref * (f_ghz / F_REF) ** EXP_EPS
    sig = sig_ref * (f_ghz / F_REF) ** EXP_SIG
    return complex_permittivity(eps, sig, f_ghz)
