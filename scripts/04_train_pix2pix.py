"""Pix2pix (cGAN + L1) refinement stage of the IPIGS pipeline.  *** NOT RUN in the authoring sandbox
(PyTorch could not be installed there) - treat as a reference implementation and expect to debug. ***

Two weight sets are trained, selected by the APK classifier (Eqs. 2-5):
    'confused'  samples (bright stroke-like spot in a sensitive area)  -> weights_confused
    'clear'     samples                                                 -> weights_clear
usage: python scripts/04_train_pix2pix.py [epochs=100] [lambda_l1=100]
needs: pip install torch   (GPU recommended)"""
import sys, os, glob
import numpy as np
import torch, torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
dev = "cuda" if torch.cuda.is_available() else "cpu"


def block(i, o, down=True, bn=True, drop=False):
    L = [nn.Conv2d(i, o, 4, 2, 1, bias=False)] if down else [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False)]
    if bn: L.append(nn.BatchNorm2d(o))
    L.append(nn.LeakyReLU(0.2) if down else nn.ReLU())
    if drop: L.append(nn.Dropout(0.5))
    return nn.Sequential(*L)


class UNet(nn.Module):                      # 256x256 -> 256x256, 8 levels
    def __init__(self):
        super().__init__()
        ch = [64, 128, 256, 512, 512, 512, 512, 512]
        self.d = nn.ModuleList([block(1 if i == 0 else ch[i - 1], ch[i], True, bn=(0 < i < 7)) for i in range(8)])
        self.u = nn.ModuleList([block(ch[7 - i] * (1 if i == 0 else 2), ch[6 - i] if i < 7 else 64, False, drop=(i < 3)) for i in range(7)])
        self.out = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        s = []
        for l in self.d:
            x = l(x); s.append(x)
        x = self.u[0](s[-1])
        for i in range(1, 7):
            x = self.u[i](torch.cat([x, s[6 - i]], 1))
        return self.out(torch.cat([x, s[0]], 1))


class PatchD(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = nn.Sequential(nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2), block(64, 128), block(128, 256),
                               nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), nn.Conv2d(512, 1, 4, 1, 1))

    def forward(self, a, b):
        return self.n(torch.cat([a, b], 1))


def load(label):
    X, Y = [], []
    for f in sorted(glob.glob(f"{ROOT}/outputs/dataset/case_*.npz")):
        z = np.load(f, allow_pickle=True)
        if str(z["apk"]) != label: continue
        X.append(z["inp"] / 35.0 - 1.0); Y.append(z["target"] / 35.0 - 1.0)      # eps' 0..70 -> -1..1
    return (torch.tensor(np.array(X))[:, None].float(), torch.tensor(np.array(Y))[:, None].float()) if X else (None, None)


def train(label):
    X, Y = load(label)
    if X is None or len(X) < 4:
        print(f"[{label}] too few samples ({0 if X is None else len(X)}), skipped"); return
    G, D = UNet().to(dev), PatchD().to(dev)
    oG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    for ep in range(EPOCHS):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 4):
            idx = perm[i:i + 4]; x, y = X[idx].to(dev), Y[idx].to(dev)
            fake = G(x)
            pr, pf = D(x, y), D(x, fake.detach())
            lD = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf)))
            oD.zero_grad(); lD.backward(); oD.step()
            pf = D(x, fake)
            lG = bce(pf, torch.ones_like(pf)) + LAM * l1(fake, y)
            oG.zero_grad(); lG.backward(); oG.step()
        if ep % 10 == 0: print(f"[{label}] epoch {ep}  D {lD.item():.3f}  G {lG.item():.3f}", flush=True)
    os.makedirs(f"{ROOT}/outputs/weights", exist_ok=True)
    torch.save(G.state_dict(), f"{ROOT}/outputs/weights/G_{label}.pt")


if __name__ == "__main__":
    for lab in ("confused", "clear"):
        train(lab)
