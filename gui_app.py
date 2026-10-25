"""Interactive pop-up app for the presentation.  Run from the project folder:

        python gui_app.py

Buttons
  RUN LIVE            full simulate + DBIM reconstruction for the chosen severity / location / noise (~1.5-2 min)
  SHOW SAVED          instant: loads the pre-computed result for the chosen severity / location
  SUMMARY (all 4)     opens the 4-case overview window
  AUDIT REAL DATA     your .s8p files: mild/severe = healthy x constant (opens a window)
Only numpy, scipy, matplotlib, scikit-image are needed (no extra GUI package)."""
import os, sys, io, time, pickle, queue, threading, subprocess, contextlib

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.makedirs(f"{ROOT}/outputs", exist_ok=True)

import numpy as np
import matplotlib
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")                      # headless Linux only; Windows/macOS use their normal window backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.widgets import Button, RadioButtons, TextBox

from mmi.pipeline import build_scene, paper_lesion, run_case, DEFAULT_FREQS
from mmi.scene import COARSE
from mmi import metrics, apk
from mmi.render import eps_panel, outline, antennas, crop_box

CLS_COL = ["#ffffff", "#e8b4a0", "#d9d9d9", "#7fc4ff", "#8a8a8a", "#f5e6a3"]
CLS_NAMES = ["skin/fat/muscle", "skull", "CSF", "gray matter", "white matter"]
BG, FG, ACC = "#101820", "#e8eef4", "#2f80ed"


class App:
    def __init__(self):
        self.sc = build_scene(f"{ROOT}/data")
        self.box = crop_box(self.sc, COARSE)
        self.healthy = None
        self.busy = False
        self.q = queue.Queue()
        self.log = []
        self.t0 = 0.0
        self.sev, self.part, self.dr = "severe", 1, 100.0
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        fig = self.fig = plt.figure(figsize=(16, 8.6), facecolor=BG)
        try:
            fig.canvas.manager.set_window_title("Microwave stroke imaging - DBIM")
        except Exception:
            pass
        fig.text(0.02, 0.95, "Microwave head imaging  |  8 antennas, 3.0-3.5 GHz  |  differential DBIM",
                 color=FG, fontsize=15, weight="bold")
        fig.text(0.02, 0.918, "Lesion is reconstructed from S-parameters; anatomy is the phantom prior.",
                 color="#9fb3c8", fontsize=10)
        self.ax = [fig.add_axes([0.25 + 0.185 * i, 0.20, 0.165, 0.64]) for i in range(4)]
        self.cbs = []
        # control panel
        ax_sev = fig.add_axes([0.02, 0.74, 0.20, 0.12], facecolor="#1b2733")
        self.r_sev = RadioButtons(ax_sev, ("mild", "severe"), active=1, activecolor=ACC)
        ax_sev.set_title("Stroke severity", color=FG, fontsize=10)
        ax_part = fig.add_axes([0.02, 0.56, 0.20, 0.14], facecolor="#1b2733")
        self.r_part = RadioButtons(ax_part, ("1: thalamus (2 cm)", "2: near skull (1 cm arc)"), active=0, activecolor=ACC)
        ax_part.set_title("Lesion location (paper)", color=FG, fontsize=10)
        for r in (self.r_sev, self.r_part):
            for t in r.labels:
                t.set_color(FG); t.set_fontsize(10)
        self.r_sev.on_clicked(lambda v: setattr(self, "sev", v))
        self.r_part.on_clicked(lambda v: setattr(self, "part", int(v[0])))
        ax_dr = fig.add_axes([0.14, 0.50, 0.08, 0.04])
        self.tb = TextBox(ax_dr, "Dynamic range (dB)  ", initial="100", color="#1b2733", hovercolor="#26364a")
        self.tb.label.set_color(FG); self.tb.text_disp.set_color(FG)
        self.tb.on_submit(self._set_dr)
        specs = [("RUN LIVE", 0.40, self.on_run, ACC), ("SHOW SAVED (instant)", 0.33, self.on_saved, "#27ae60"),
                 ("SUMMARY (all 4)", 0.26, self.on_summary, "#8e44ad"), ("AUDIT REAL DATA", 0.19, self.on_audit, "#d35400")]
        self.btns = []
        for label, y, cb, col in specs:
            a = fig.add_axes([0.02, y, 0.20, 0.055])
            b = Button(a, label, color=col, hovercolor=col)
            b.label.set_color("white"); b.label.set_fontsize(11); b.label.set_weight("bold")
            b.on_clicked(cb); self.btns.append(b)
        self.status = fig.text(0.02, 0.02, "Ready. Pick a case, then RUN LIVE or SHOW SAVED.", color="#7ee0a6",
                               fontsize=11, family="monospace", va="bottom")
        self.metrics_txt = fig.text(0.02, 0.095, "", color=FG, fontsize=10, family="monospace", va="bottom")
        self._blank()
        self.timer = fig.canvas.new_timer(interval=400)
        self.timer.add_callback(self.poll)
        self.timer.start()

    def _blank(self):
        titles = ["Ground truth  ε′ (phantom + lesion)", "DBIM result δε′\n(from S-parameters)",
                  "Composite ε′\n(anatomy prior + δε′)", "Tissue classes\n(phantom prior)"]
        for a, t in zip(self.ax, titles):
            a.clear(); a.set_facecolor("#1b2733"); a.set_xticks([]); a.set_yticks([])
            a.set_title(t, color=FG, fontsize=8)
            for s in a.spines.values():
                s.set_color("#3a4b5e")

    def _set_dr(self, text):
        try:
            self.dr = float(text)
        except ValueError:
            self.say("Dynamic range must be a number (e.g. 100)", "#ff7675")

    def say(self, msg, col="#7ee0a6"):
        self.status.set_text(msg); self.status.set_color(col); self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ drawing
    def show_result(self, res, lesion, title_note=""):
        sc = self.sc
        sc.lesions = [lesion]
        e_t, _, les = sc.ref_maps(COARSE, with_lesion=True)
        P, d, m = res["mask"], res["d_eps"], sc.mask_c
        for cb in self.cbs:
            cb.remove()
        self.cbs = []
        for a in self.ax:
            a.clear()
        eps_panel(self.ax[0], e_t, COARSE, self.box, "Ground truth ε′\n(phantom + lesion)", mask=m)
        outline(self.ax[0], les, COARSE, "w"); antennas(self.ax[0], sc)
        lim = max(5.0, float(np.percentile(np.abs(d[P]), 99.5)))
        p1 = eps_panel(self.ax[1], d, COARSE, self.box, "DBIM result δε′\n(from S-parameters)", vmin=-lim, vmax=lim,
                       cmap="RdBu_r", mask=P)
        outline(self.ax[1], les, COARSE, "k")
        p2 = eps_panel(self.ax[2], res["eps"], COARSE, self.box, "Composite ε′\n(anatomy prior + δε′)", mask=m)
        outline(self.ax[2], les, COARSE, "w")
        for pp, a in ((p1, self.ax[1]), (p2, self.ax[2])):
            cb = self.fig.colorbar(pp, ax=a, orientation="horizontal", fraction=0.04, pad=0.03)
            cb.ax.tick_params(colors=FG, labelsize=7); self.cbs.append(cb)
        cls = sc.class_map(COARSE)
        h = COARSE.n * COARSE.h * 1e3
        self.ax[3].imshow(np.ma.masked_where(~m, cls), cmap=ListedColormap(CLS_COL), vmin=0, vmax=5,
                          extent=[0, h, h, 0], interpolation="nearest")
        self.ax[3].set_xlim(self.box[0], self.box[1]); self.ax[3].set_ylim(self.box[2], self.box[3])
        self.ax[3].set_xticks([]); self.ax[3].set_yticks([])
        self.ax[3].set_title("Tissue classes\n(phantom prior)", fontsize=8)
        self.ax[3].legend(handles=[Patch(color=c, label=n) for c, n in zip(CLS_COL[1:], CLS_NAMES)],
                          loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, ncol=2, frameon=False, labelcolor=FG)
        for a in self.ax:
            a.set_facecolor("#1b2733"); a.title.set_color(FG)
        est = metrics.half_max_mask(d, P)
        e_b, _, _ = sc.ref_maps(COARSE, with_lesion=False)
        info = (f"lesion: {self.sev_of(lesion)}   true peak δε′ {np.max(e_t - e_b):.1f}   recon peak δε′ {d.max():.1f}   "
                f"centroid error {metrics.localisation_error_mm(est, les, COARSE.h * 1e3):.1f} mm   "
                f"IoU {metrics.iou(est, les):.2f}   APK: {apk.classify(res['eps'], sc, COARSE)[0]} {title_note}")
        self.metrics_txt.set_text(info)
        self.fig.canvas.draw_idle()

    @staticmethod
    def sev_of(L):
        return L.label

    # ------------------------------------------------------------------ actions
    def on_saved(self, _evt=None):
        f = f"{ROOT}/outputs/demo_{self.sev}_part{self.part}.pkl"
        if not os.path.exists(f):
            return self.say("No saved result for this case - press RUN LIVE.", "#ff7675")
        z = pickle.load(open(f, "rb"))
        self.show_result(z["res"], z["lesion"], "(saved)")
        self.say(f"Loaded saved result: {self.sev}, part {self.part} (100 dB)")

    def on_run(self, _evt=None):
        if self.busy:
            return
        self.busy = True; self.log = []; self.t0 = time.time()
        self.say(f"Simulating {self.sev}, part {self.part}, {self.dr:.0f} dB ... (fine-grid forward model)", "#f9ca24")
        threading.Thread(target=self._worker, args=(self.sev, self.part, self.dr), daemon=True).start()

    def _worker(self, sev, part, dr):
        class W(io.TextIOBase):
            def __init__(s, lst): s.lst = lst
            def write(s, t):
                if t.strip(): s.lst.append(t.strip())
                return len(t)
        try:
            L = paper_lesion(self.sc, part, sev)
            with contextlib.redirect_stdout(W(self.log)):
                res = run_case(self.sc, [L], DEFAULT_FREQS, dr_db=dr, seed=part, healthy_data=self.healthy, verbose=True)
            self.healthy = res["healthy_data"]
            self.q.put(("done", res, L))
        except Exception as e:                      # surface errors in the window
            self.q.put(("err", repr(e)))

    def poll(self):
        if self.busy and self.q.empty():
            last = self.log[-1] if self.log else "simulating measurements ..."
            self.say(f"[{time.time() - self.t0:4.0f}s] reconstructing: {last}", "#f9ca24")
        while not self.q.empty():
            msg = self.q.get()
            self.busy = False
            if msg[0] == "done":
                _, res, L = msg
                self.show_result(res, L, f"({time.time() - self.t0:.0f}s)")
                self.say(f"Done in {time.time() - self.t0:.0f} s.  Saved outputs/live_{self.sev}_part{self.part}.png")
                self.fig.savefig(f"{ROOT}/outputs/live_{self.sev}_part{self.part}.png", dpi=110, facecolor=self.fig.get_facecolor())
            else:
                self.say("Error: " + msg[1], "#ff7675")

    def on_summary(self, _evt=None):
        cases = [("severe", 1), ("severe", 2), ("mild", 1), ("mild", 2)]
        f2, ax = plt.subplots(2, 4, figsize=(18, 9))
        try:
            f2.canvas.manager.set_window_title("Summary - four cases")
        except Exception:
            pass
        sc = self.sc
        for c, (sev, part) in enumerate(cases):
            f = f"{ROOT}/outputs/demo_{sev}_part{part}.pkl"
            if not os.path.exists(f):
                continue
            z = pickle.load(open(f, "rb")); res, L, row = z["res"], z["lesion"], z["row"]
            sc.lesions = [L]
            _, _, les = sc.ref_maps(COARSE, with_lesion=True)
            P, d = res["mask"], res["d_eps"]
            lim = max(5.0, float(np.percentile(np.abs(d[P]), 99.5)))
            eps_panel(ax[0, c], d, COARSE, self.box, f"{sev} / part {part}: DBIM δε′", vmin=-lim, vmax=lim, cmap="RdBu_r", mask=P)
            outline(ax[0, c], les, COARSE, "k")
            eps_panel(ax[1, c], res["eps"], COARSE, self.box,
                      f"composite ε′\nerror {row['centroid_err_mm']:.1f} mm, IoU {row['iou_halfmax']:.2f}", mask=sc.mask_c)
            outline(ax[1, c], les, COARSE, "w")
        f2.suptitle("Outline = true lesion.  Anatomy from phantom; only the lesion comes from the data.", fontsize=11)
        f2.tight_layout(); plt.show(block=False)

    def on_audit(self, _evt=None):
        self.say("Auditing your .s8p files ...", "#f9ca24")
        env = dict(os.environ, MPLBACKEND="Agg")
        r = subprocess.run([sys.executable, f"{ROOT}/scripts/01_audit_real_data.py"], capture_output=True, text=True, env=env)
        png = f"{ROOT}/outputs/real_data_audit.png"
        if not os.path.exists(png):
            return self.say("Audit failed: " + (r.stderr or r.stdout)[-200:], "#ff7675")
        f2 = plt.figure(figsize=(15, 8.2))
        try:
            f2.canvas.manager.set_window_title("Real data audit")
        except Exception:
            pass
        a = f2.add_axes([0, 0.08, 1, 0.92]); a.imshow(plt.imread(png)); a.axis("off")
        f2.text(0.5, 0.03, "  |  ".join(l for l in r.stdout.splitlines() if "ratio" in l), ha="center", fontsize=9)
        plt.show(block=False)
        self.say("Audit done: mild/severe = healthy x one constant (no lesion position in the files).")


if __name__ == "__main__":
    app = App()
    plt.show()
