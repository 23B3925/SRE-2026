"""Audit the CST-exported .s8p files and produce presentation figures.
Shows that mild/severe differ from healthy by a spatially uniform complex ratio
(i.e. they carry no lesion-position information)."""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib.pyplot as plt
from mmi.sparams import read_touchstone, audit

D = f"{ROOT}/data"
files = {"healthy": "healthy.s8p", "mild": "mild.s8p", "severe": "severe.s8p"}
S = {}
for k, fn in files.items():
    f, S[k] = read_touchstone(f"{D}/{fn}")
    audit(f, S[k], k)
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for k, c in zip(S, ("C0", "C1", "C3")):
    ax[0, 0].plot(f, 20 * np.log10(np.abs(S[k][:, 0, 0])), c, label=k)
    ax[0, 1].plot(f, 20 * np.log10(np.abs(S[k][:, 0, 1])), c, label=k)
    ax[0, 2].plot(f, 20 * np.log10(np.abs(S[k][:, 0, 4])), c, label=k)
for a, t in zip(ax[0], ("|S11| (dB)", "|S12| adjacent (dB)", "|S15| opposite (dB)")):
    a.set_title(t); a.set_xlabel("GHz"); a.legend(); a.grid(alpha=.3)
for r, k in enumerate(("mild", "severe")):
    ratio = S[k] / S["healthy"]
    R = ratio.mean(0); off = ~np.eye(8, dtype=bool)
    im = ax[1, r].imshow(np.abs(R), cmap="viridis"); fig.colorbar(im, ax=ax[1, r], fraction=0.046)
    ax[1, r].set_title(f"|{k}/healthy| per port pair (freq-averaged)")
    ax[1, r].set_xlabel("port j"); ax[1, r].set_ylabel("port i")
    print(f"{k}: off-diagonal |ratio| = {np.abs(R[off]).mean():.4f} +- {np.abs(R[off]).std():.1e} (same for ALL pairs); "
          f"drift over 3-4 GHz = {np.abs(ratio[:, off]).std(0).mean():.1e}; diagonal |ratio| = {np.abs(R[~off]).mean():.4f}")
ax[1, 2].plot(f, np.abs(S["mild"][:, 0, 0] / S["healthy"][:, 0, 0]), label="mild")
ax[1, 2].plot(f, np.abs(S["severe"][:, 0, 0] / S["healthy"][:, 0, 0]), label="severe")
ax[1, 2].set_title("|S11 ratio| vs frequency"); ax[1, 2].legend(); ax[1, 2].grid(alpha=.3)
fig.tight_layout(); os.makedirs(f"{ROOT}/outputs", exist_ok=True)
fig.savefig(f"{ROOT}/outputs/real_data_audit.png", dpi=120)
plt.show()
