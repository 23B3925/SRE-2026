"""2-D scalar (TM, E_z) frequency-domain Helmholtz forward solver with PML.

Convention: time dependence e^{+j w t}.  Complex relative permittivity
eps_c = eps' - j sigma/(w eps0).  Governing equation for a unit line current
(normalised so the free-space Green's function is  G = -(j/4) H0^(2)(k r)):

        ( laplacian + k0^2 eps_c(r) ) E(r) = - delta(r - r_s)

Antennas are modelled as ideal line sources/receivers at pixel positions
(reciprocal, so the transfer matrix  T[i, j] = E(r_i ; source j)  is symmetric).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import hankel2

C0 = 299792458.0
EPS0 = 8.854187817e-12


def pml_profiles(n, npml, smax=10.0, power=2):
    """Stretching factors at cell centres (len n) and at the n+1 cell edges."""
    def prof(x):  # x in cell units, positions along axis
        d = np.zeros_like(x, dtype=float)
        left = npml - x
        right = x - (n - npml)
        d = np.where(left > 0, left, 0.0) + np.where(right > 0, right, 0.0)
        return 1.0 - 1j * smax * (d / npml) ** power
    centres = prof(np.arange(n) + 0.5)
    edges = prof(np.arange(n + 1).astype(float))
    return centres, edges


class Helmholtz2D:
    def __init__(self, eps_c, h, f_hz, npml=10, smax=10.0, fourth=True):
        """eps_c: complex [ny, nx] relative permittivity (including PML region)."""
        self.ny, self.nx = eps_c.shape
        self.h = h
        self.f = f_hz
        self.k0 = 2 * np.pi * f_hz / C0
        self.eps_c = eps_c
        self.npml, self.smax = npml, smax
        self.fourth = fourth
        self._build()

    def _build(self):
        ny, nx, h = self.ny, self.nx, self.h
        scx, sex = pml_profiles(nx, self.npml, self.smax)
        scy, sey = pml_profiles(ny, self.npml, self.smax)
        N = nx * ny
        idx = np.arange(N).reshape(ny, nx)
        # coefficient of  d/dx( (1/sx) dE/dx ) /sx
        cxp = 1.0 / (sex[1:][None, :] * scx[None, :]) * np.ones((ny, 1))     # to +x neighbour
        cxm = 1.0 / (sex[:-1][None, :] * scx[None, :]) * np.ones((ny, 1))    # to -x neighbour
        cyp = 1.0 / (sey[1:][:, None] * scy[:, None]) * np.ones((1, nx))
        cym = 1.0 / (sey[:-1][:, None] * scy[:, None]) * np.ones((1, nx))
        k2 = self.k0 ** 2 * self.eps_c
        # interior mask: pixel and all 8 neighbours outside the PML
        inner = np.zeros((ny, nx), dtype=bool)
        m = self.npml + 1
        inner[m:ny - m, m:nx - m] = True
        self.inner = inner
        if self.fourth:
            # Mehrstellen (Collatz) 4th-order compact scheme in the interior
            k2eff = np.where(inner, k2 * (1.0 - h ** 2 * k2 / 12.0), k2)
        else:
            k2eff = k2
        diag = -(cxp + cxm + cyp + cym) / h ** 2 + k2eff
        if self.fourth:
            diag = diag + np.where(inner, 4.0 / (6 * h ** 2), 0.0)
        rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
        if self.fourth:
            ii = idx[m:ny - m, m:nx - m].ravel()
            wd = 1.0 / (6 * h ** 2)
            for dy, dx, w in ((-1, -1, wd), (-1, 1, wd), (1, -1, wd), (1, 1, wd)):
                rows.append(ii)
                cols.append(idx[m + dy:ny - m + dy, m + dx:nx - m + dx].ravel())
                vals.append(np.full(ii.size, w, dtype=complex))
            # -2*A_axis/(6h^2) contribution added to the axis couplings below
            self._axis_extra = -2.0 / (6 * h ** 2)
        # +x
        ex = np.where(inner, getattr(self, '_axis_extra', 0.0), 0.0)
        r = idx[:, :-1].ravel(); c = idx[:, 1:].ravel(); v = (cxp[:, :-1] / h ** 2 + ex[:, :-1]).ravel()
        rows += [r]; cols += [c]; vals += [v]
        # -x
        r = idx[:, 1:].ravel(); c = idx[:, :-1].ravel(); v = (cxm[:, 1:] / h ** 2 + ex[:, 1:]).ravel()
        rows += [r]; cols += [c]; vals += [v]
        # +y
        r = idx[:-1, :].ravel(); c = idx[1:, :].ravel(); v = (cyp[:-1, :] / h ** 2 + ex[:-1, :]).ravel()
        rows += [r]; cols += [c]; vals += [v]
        # -y
        r = idx[1:, :].ravel(); c = idx[:-1, :].ravel(); v = (cym[1:, :] / h ** 2 + ex[1:, :]).ravel()
        rows += [r]; cols += [c]; vals += [v]
        A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        self.A = A
        self.lu = spla.splu(A)

    def solve_points(self, src_ij):
        """Fields (nsrc, ny, nx) for unit line sources at pixel indices [(iy, ix), ...]."""
        N = self.nx * self.ny
        B = np.zeros((N, len(src_ij)), dtype=complex)
        h = self.h
        for m, (iy, ix) in enumerate(src_ij):
            if not self.fourth:
                B[iy * self.nx + ix, m] = -1.0 / h ** 2
                continue
            d = np.zeros((self.ny, self.nx), dtype=complex)
            d[iy, ix] = 1.0 / h ** 2
            lap = np.zeros_like(d)
            lap[1:-1, 1:-1] = (d[1:-1, 2:] + d[1:-1, :-2] + d[2:, 1:-1] + d[:-2, 1:-1] - 4 * d[1:-1, 1:-1]) / h ** 2
            k2s = self.k0 ** 2 * self.eps_c[iy, ix]
            rhs = -(d + h ** 2 / 12.0 * lap - h ** 2 * k2s / 12.0 * d)
            B[:, m] = rhs.ravel()
        X = self.lu.solve(B)
        return X.T.reshape(len(src_ij), self.ny, self.nx)


def free_space_G(k0, r):
    """Analytic 2-D free-space Green's function for e^{+jwt}: -(j/4) H0^(2)(k r)."""
    return -0.25j * hankel2(0, k0 * r)


def transfer_matrix(fields, ant_ij):
    """T[i, j] = field at antenna i from source j.   fields: (nsrc, ny, nx)."""
    n = len(ant_ij)
    T = np.zeros((n, n), dtype=complex)
    for j in range(n):
        for i in range(n):
            T[i, j] = fields[j, ant_ij[i][0], ant_ij[i][1]]
    return T
