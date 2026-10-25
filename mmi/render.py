"""Figures: anatomy classes, eps' maps with the paper's 0-70 viridis scale, DBIM contrast."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

CLASS_NAMES = None


def _extent(g):
    return [0, g.n * g.h * 1e3, g.n * g.h * 1e3, 0]


def crop_box(scene, g, pad_mm=20):
    m = scene.mask_f if g.n == 300 else scene.mask_c
    ys, xs = np.nonzero(m)
    h = g.h * 1e3
    return (xs.min() * h - pad_mm, xs.max() * h + pad_mm, ys.max() * h + pad_mm, ys.min() * h - pad_mm)


def eps_panel(ax, img, g, box, title, vmax=70, cmap="viridis", vmin=0, mask=None):
    im = np.ma.masked_where(~mask, img) if mask is not None else img
    p = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, extent=_extent(g), interpolation="nearest")
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return p


def antennas(ax, scene):
    a = np.array(scene.antennas_mm)
    ax.plot(a[:, 0], a[:, 1], "w^", ms=5, mec="k")


def outline(ax, mask, g, color="r"):
    ax.contour(np.arange(g.n) * g.h * 1e3, np.arange(g.n) * g.h * 1e3, mask.astype(float), [0.5],
               colors=color, linewidths=1.0)


def case_figure(scene, res, path, title="", show=False):
    """4 panels: true eps' | reconstructed contrast (data-driven) | composite image | tissue classes."""
    from .scene import COARSE, FINE
    e_t, _, les_c = scene.ref_maps(COARSE, with_lesion=True)
    e_b, _, _ = scene.ref_maps(COARSE, with_lesion=False)
    box = crop_box(scene, COARSE)
    P = res["mask"]
    d = res["d_eps"]
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
    m = scene.mask_c
    p0 = eps_panel(ax[0], e_t, COARSE, box, "Ground truth  ε′ @3.5 GHz\n(phantom + lesion)", mask=m)
    outline(ax[0], les_c, COARSE, "w")
    antennas(ax[0], scene)
    lim = max(5.0, float(np.percentile(np.abs(d[P]), 99.5)))
    p1 = eps_panel(ax[1], d, COARSE, box, "DBIM result: δε′ from S-parameters\n(data-driven, healthy scan as reference)",
                   vmin=-lim, vmax=lim, cmap="RdBu_r", mask=P)
    outline(ax[1], les_c, COARSE, "k")
    fig.colorbar(p1, ax=ax[1], fraction=0.046)
    p2 = eps_panel(ax[2], res["eps"], COARSE, box, "Composite image  ε′ = anatomy prior + δε′\n(anatomy NOT measured)", mask=m)
    outline(ax[2], les_c, COARSE, "w")
    fig.colorbar(p2, ax=ax[2], fraction=0.046)
    cls = scene.class_map(COARSE)
    ccmap = ListedColormap(["#ffffff", "#e8b4a0", "#d9d9d9", "#7fc4ff", "#8a8a8a", "#f5e6a3"])
    ax[3].imshow(np.ma.masked_where(~m, cls), cmap=ccmap, extent=_extent(COARSE), interpolation="nearest", vmin=0, vmax=5)
    ax[3].legend(handles=[Patch(color=c, label=n) for c, n in zip(["#e8b4a0", "#d9d9d9", "#7fc4ff", "#8a8a8a", "#f5e6a3"],
                 ["skin/fat/muscle", "skull", "CSF", "gray matter", "white matter"])], loc="lower center", fontsize=6, ncol=3, frameon=False)
    ax[3].set_xlim(box[0], box[1]); ax[3].set_ylim(box[2], box[3]); ax[3].set_xticks([]); ax[3].set_yticks([])
    ax[3].set_title("Tissue-class map (phantom prior)", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if show:
        fig.canvas.manager.set_window_title(title or "reconstruction")
        plt.show(block=False)
        plt.pause(0.5)          # draw the window now; the script keeps computing
        return fig
    plt.close(fig)
