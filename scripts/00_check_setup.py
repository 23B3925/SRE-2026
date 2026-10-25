"""Quick sanity check: libraries, data files, and a 5-second solver test. Run this first."""
import sys, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ok = True
for m in ("numpy", "scipy", "matplotlib", "skimage"):
    try:
        __import__(m); print(f"[ok] {m}")
    except ImportError:
        ok = False; print(f"[MISSING] {m}  ->  pip install -r requirements.txt")
need = ["Head_voxel.dat", "healthy.s8p", "mild.s8p", "severe.s8p"] + [f"Tissue_{f}GHz.txt" for f in ("3.0", "3.5", "4.0")]
for n in need:
    p = f"{ROOT}/data/{n}"
    if os.path.exists(p): print(f"[ok] data/{n}")
    else: ok = False; print(f"[MISSING] data/{n}")
if ok:
    from mmi.pipeline import build_scene
    t = time.time(); sc = build_scene(f"{ROOT}/data")
    print(f"[ok] scene built, {len(sc.antennas_mm)} antennas ({time.time()-t:.1f}s)")
print("\nSETUP OK - you can run the demo." if ok else "\nFix the items above first.")
