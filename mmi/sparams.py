"""Touchstone (.s8p) reader for CST-exported multi-port S-parameters."""
import numpy as np


def read_touchstone(path, nports=8):
    """Return (freq_GHz[nf], S[nf, nports, nports]) as complex128.

    Handles the '# GHz S RI R 50' option line (also MA / DB formats) and the
    standard Touchstone layout where each frequency block holds
    nports*nports complex pairs spread over several lines.
    """
    fmt, unit_scale = "RI", 1.0
    toks = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!"):
                continue
            if line.startswith("#"):
                parts = line[1:].upper().split()
                unit = parts[0]
                unit_scale = {"HZ": 1e-9, "KHZ": 1e-6, "MHZ": 1e-3, "GHZ": 1.0}[unit]
                fmt = parts[parts.index("S") + 1]
                continue
            line = line.split("!")[0]
            toks.extend(line.split())
    vals = np.asarray(toks, dtype=float)
    per_block = 1 + 2 * nports * nports
    if vals.size % per_block != 0:
        raise ValueError(f"{path}: {vals.size} numbers is not a multiple of {per_block}")
    vals = vals.reshape(-1, per_block)
    freq = vals[:, 0] * unit_scale
    a, b = vals[:, 1::2], vals[:, 2::2]
    if fmt == "RI":
        s = a + 1j * b
    elif fmt == "MA":
        s = a * np.exp(1j * np.deg2rad(b))
    elif fmt == "DB":
        s = 10 ** (a / 20) * np.exp(1j * np.deg2rad(b))
    else:
        raise ValueError(fmt)
    # Touchstone for N>2 ports: row-major (S11 S12 S13 ...)
    S = s.reshape(-1, nports, nports)
    return freq, S


def audit(freq, S, name=""):
    """Basic sanity checks. Returns dict of scalars."""
    recip = np.abs(S - np.swapaxes(S, 1, 2)).max()
    # passivity: largest singular value <= 1
    smax = np.linalg.svd(S, compute_uv=False)[:, 0].max()
    d = np.diff(freq)
    return dict(name=name, nf=len(freq), f0=freq[0], f1=freq[-1],
                df_min=d.min(), df_max=d.max(),
                max_reciprocity_err=recip, max_singular_value=smax,
                mean_S11_dB=float(np.mean(20 * np.log10(np.abs(np.diagonal(S, axis1=1, axis2=2))))),
                mean_offdiag_dB=float(np.mean(20 * np.log10(np.abs(S[:, ~np.eye(S.shape[1], dtype=bool)]) + 1e-30))))
