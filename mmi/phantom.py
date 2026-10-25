"""Voxel head phantom -> complex permittivity maps (Gabriel-style tables, 3-4 GHz).

Voxel file layout (from Head_voxel_3500MHz.vox): int8, nx=256, ny=256, nz=128,
dx=dy=1.1 mm, dz=1.4 mm, stored x-fastest  ->  numpy array [z, y, x].
Tissue index == the `Num` column of the Tissue_*.txt tables; 0 = background (air).
"""
import glob
import os
import re
import numpy as np
from scipy.interpolate import PchipInterpolator

EPS0 = 8.854187817e-12
C0 = 299792458.0

VOX = dict(nx=256, ny=256, nz=128, dx=1.1e-3, dy=1.1e-3, dz=1.4e-3)

# tissue ids that are brain parenchyma / structures used for the paper's classes
WHITE_MATTER = [83, 111]                       # white matter, corpus callosum
GRAY_MATTER = [77, 85, 89, 91, 95, 96, 101, 103, 105, 107, 108, 109, 112, 114, 117, 118, 120, 124]
CSF = [2, 75, 92, 115, 122, 123]
BONE = [4, 5, 70, 71, 76, 81, 99, 125, 26]
SKIN = [1]
MUSCLE = [9, 102]
FAT = [22, 98, 116]
AIR = [0, 15, 84, 104]
THALAMUS = [109]


def load_voxels(path, shape=(128, 256, 256)):
    return np.fromfile(path, dtype=np.uint8).reshape(shape)


def parse_tissue_file(path):
    """Return {num: (name, eps, mu, sigma)}"""
    out = {}
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("#"):
                continue
            m = re.match(r"^(.*?)\s{2,}(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", s)
            if not m:
                continue
            name, num, eps, mu, kap = m.groups()
            out[int(num)] = (name.strip(), float(eps), float(mu), float(kap))
    return out


class TissueDB:
    """Frequency-interpolated (PCHIP) eps' and sigma for each tissue id."""

    def __init__(self, data_dir):
        files = sorted(glob.glob(os.path.join(data_dir, "Tissue_*GHz.txt")),
                       key=lambda p: float(re.search(r"Tissue_([\d.]+)GHz", p).group(1)))
        self.freqs = np.array([float(re.search(r"Tissue_([\d.]+)GHz", p).group(1)) for p in files])
        tabs = [parse_tissue_file(p) for p in files]
        self.ids = sorted(tabs[0].keys())
        self.names = {i: tabs[0][i][0] for i in self.ids}
        self._eps, self._sig = {}, {}
        for i in self.ids:
            e = np.array([t[i][1] for t in tabs])
            s = np.array([t[i][3] for t in tabs])
            self._eps[i] = PchipInterpolator(self.freqs, e)
            self._sig[i] = PchipInterpolator(self.freqs, s)

    def lut(self, f_ghz):
        """Return (eps_lut[256], sigma_lut[256]); id 0 and unknown ids -> air."""
        e = np.ones(256)
        s = np.zeros(256)
        for i in self.ids:
            e[i] = float(self._eps[i](f_ghz))
            s[i] = float(self._sig[i](f_ghz))
        return e, s


def complex_permittivity(eps, sigma, f_ghz):
    """Relative complex permittivity, e^{+j w t} convention: eps' - j sigma/(w eps0)."""
    w = 2 * np.pi * f_ghz * 1e9
    return eps - 1j * sigma / (w * EPS0)


def tissue_class_map(ids):
    """Coarse classes used for the paper's 4-layer view: 0 air, 1 skin/fat/muscle,
    2 skull(bone), 3 CSF, 4 gray matter, 5 white matter."""
    cls = np.zeros(ids.shape, dtype=np.uint8)
    for lst, c in ((SKIN + FAT + MUSCLE, 1), (BONE, 2), (CSF, 3), (GRAY_MATTER, 4), (WHITE_MATTER, 5)):
        cls[np.isin(ids, lst)] = c
    return cls


def get_slice(vox, k):
    """Axial slice k as [y, x] tissue ids."""
    return vox[k]


def brain_slice_range(vox, min_wm=800):
    wm = np.isin(vox, WHITE_MATTER).sum(axis=(1, 2))
    ks = np.where(wm >= min_wm)[0]
    return int(ks.min()), int(ks.max())
