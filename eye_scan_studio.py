#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eye Scan Studio  -  one click-through interface for every tool built in this project.

  python eye_scan_studio.py              # opens http://127.0.0.1:8765 in your browser
  python eye_scan_studio.py --port 9000 --no-browser

Needs only: numpy scipy matplotlib pillow  (already installed in the project venv)
and, for the 3D pages, the C++ tools built in  ~/Documents/ai_model/eye3d/build
(eyeview2, img2mesh).  Keep this file next to oct_bscan.py and image_to_bscan.py.
"""
import argparse
import glob
import gzip
import io
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oct_bscan as ob            # noqa: E402
import image_to_bscan as itb      # noqa: E402

HOME = os.path.expanduser("~")
CFG_PATH = os.path.join(HOME, ".eye_scan_studio.json")
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
TMP = tempfile.mkdtemp(prefix="eye_scan_studio_")
RLOCK = threading.RLock()          # oct_bscan keeps module-level globals, so renders are serialised


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
def _p(*a):
    return os.path.join(HOME, *a)


DEFAULTS = {
    "project_root": _p("Documents", "ai_model"),
    "build_dir": _p("Documents", "ai_model", "eye3d", "build"),
    "cmake_src": _p("Documents", "ai_model", "eye3d"),
    "depth_python": _p("Documents", "ai_model", "depth", ".venv", "bin", "python"),
    "depth_script": _p("Documents", "ai_model", "depth", "depth.py"),
    "data_dir": _p("Documents", "ai_model_data"),
    "out_dir": _p("Documents", "ai_model", "oct", "results"),
    "atlas_labels": _p("Downloads", "mreye", "C_eye_atlas", "colin27", "1_tpl-MNIColin27_T1w_cropped_labels.nii.gz"),
    "atlas_mri": _p("Downloads", "mreye", "C_eye_atlas", "colin27", "1_tpl-MNIColin27_T1w_cropped.nii.gz"),
}


def load_cfg():
    cfg = dict(DEFAULTS)
    try:
        with open(CFG_PATH, encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except Exception:
        pass
    return cfg


CFG = load_cfg()


def save_cfg():
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(CFG, f, indent=2, ensure_ascii=False)


def X(p):
    return os.path.expanduser(str(p)) if p else ""


class ApiError(Exception):
    pass


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def parse_box(s):
    if not s or not str(s).strip():
        return None
    v = [int(float(t)) for t in re.split(r'[,\s]+', str(s).strip()) if t]
    if len(v) != 4:
        raise ApiError("A box needs 4 numbers: left,top,right,bottom  (got '%s')" % s)
    return tuple(v)


def parse_boxes(s):
    if not s or not str(s).strip():
        return []
    return [parse_box(part) for part in re.split(r'[;\n]+', str(s)) if part.strip()]


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def build_lut256():
    lut = np.array(ob.LUT, float)                       # row 0 = 0 dB (white) ... row 80 = -80 dB (black)
    db = -np.arange(len(lut), dtype=float)
    xs = np.linspace(-80, 0, 256)
    out = np.stack([np.interp(xs, db[::-1], lut[::-1, c]) for c in range(3)], axis=1)
    return out.astype(np.uint8)


LUT256 = build_lut256()


def colorize_db(db_hw):
    idx = np.clip((db_hw + 80.0) / 80.0 * 255.0, 0, 255).astype(np.uint8)
    return LUT256[idx]


def need_bin(name):
    p = os.path.join(X(CFG["build_dir"]), name)
    if not (os.path.isfile(p) and os.access(p, os.X_OK)):
        raise ApiError("'%s' is not built yet (looked in %s). Open Settings -> 'Build / rebuild C++ tools'." % (name, X(CFG["build_dir"])))
    return p


def rnd(a, n=2):
    return [round(float(v), n) for v in a]


# ----------------------------------------------------------------------------
# background jobs (subprocesses with live log)
# ----------------------------------------------------------------------------
JOBS = {}


def start_job(title, cmd, cwd=None, shell=False):
    jid = uuid.uuid4().hex[:8]
    job = {"id": jid, "title": title, "cmd": cmd if isinstance(cmd, str) else " ".join(map(str, cmd)),
           "lines": [], "done": False, "rc": None, "t0": time.time()}
    JOBS[jid] = job

    def run():
        try:
            p = subprocess.Popen(cmd, cwd=cwd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
            for line in p.stdout:
                job["lines"].append(line.rstrip("\n"))
            p.wait()
            job["rc"] = p.returncode
        except Exception as e:           # noqa: BLE001
            job["lines"].append("ERROR: %s" % e)
            job["rc"] = -1
        job["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return {"job": jid, "cmd": job["cmd"]}


def job_status(q):
    job = JOBS.get(q.get("id"))
    if not job:
        raise ApiError("Unknown job")
    since = int(q.get("since", 0))
    return {"lines": job["lines"][since:], "next": len(job["lines"]), "done": job["done"], "rc": job["rc"],
            "title": job["title"], "elapsed": round(time.time() - job["t0"], 1)}


# ----------------------------------------------------------------------------
# page 1: OCT scans (A-scan / B-scan)
# ----------------------------------------------------------------------------
OCT = {"db": None, "meta": {}}


def reset_ob_defaults():
    ob.LAT_MM, ob.DEP_UM, ob.N_A, ob.N_Z = 5.0, 2000.0, 512, 1024


def oct_figure(scale=1.0, interp=None, marker=None, out=None):
    m, db = OCT["meta"], OCT["db"]
    if db is None:
        raise ApiError("Load or generate a scan first.")
    out = out or os.path.join(TMP, "bscan_%s.png" % uuid.uuid4().hex[:8])
    with RLOCK:
        ob.LAT_MM, ob.DEP_UM = float(m["width_mm"]), float(m["depth_um"])
        ob.N_A, ob.N_Z = db.shape
        ob.render(db, out, labels=m["labels"], info_lines=m["info"], inset=m["inset"], title=m["title"],
                  scale=scale, interp=interp or m["interp"], marker_x=marker)
    return out


def oct_public_meta():
    m, db = OCT["meta"], OCT["db"]
    l, t, r, b = ob.AX_PX
    return {"source": m["source"], "name": m["name"], "n_a": int(db.shape[0]), "n_z": int(db.shape[1]),
            "width_mm": m["width_mm"], "depth_um": m["depth_um"], "note": m.get("note", ""),
            "ax": {"l": l / ob.FIG_W, "t": t / ob.FIG_H, "r": r / ob.FIG_W, "b": b / ob.FIG_H}}


def oct_load(b):
    src = b.get("source")
    if src == "sim":
        seed = int(b.get("seed") or 1)
        with RLOCK:
            reset_ob_defaults()
            db = ob.simulate(seed)
        meta = dict(source="sim", name="phantom_seed%d" % seed, width_mm=5.0, depth_um=2000.0, labels=True,
                    inset=True, info=None, title=None, interp="nearest",
                    note="Synthetic phantom calibrated on one reference figure - not a real patient scan.")
    elif src == "image":
        path = X(b.get("path"))
        if not os.path.isfile(path):
            raise ApiError("Image not found: %s" % path)
        g, is_color = itb.load_gray(path, parse_box(b.get("crop")), bool(b.get("auto_crop")), parse_boxes(b.get("mask")))
        if is_color and not b.get("force"):
            raise ApiError("This looks like a colour fundus photo. Fundus images are surface photographs with no depth, "
                           "so A-scans cannot be extracted. Use a grayscale OCT B-scan (e.g. OCTID), or tick 'force'.")
        h, w = g.shape
        n_a = int(b.get("n_ascans") or 0) or w
        n_z = int(b.get("n_depth") or 0) or h
        floor = float(b.get("floor_db") or -74)
        db = itb.image_to_ascans(g, n_a, n_z, 80.0, 1.0, bool(b.get("invert")), True, floor)
        wm, dm = float(b.get("width_mm") or 6), float(b.get("depth_mm") or 2)
        name = os.path.splitext(os.path.basename(path))[0]
        info = []
        if b.get("show_info"):
            info = ["Source: %s" % name[:24], "Image: %d x %d px" % (w, h), "A-Scans: %d x %d pts" % (n_a, n_z),
                    "Assumed: %g x %g mm, floor %g dB" % (wm, dm, floor)]
        meta = dict(source="image", name=name, width_mm=wm, depth_um=dm * 1000.0, labels=False,
                    inset=bool(b.get("inset")), info=info,
                    title="B-Scan from OCT image\n%d A-Scans \u00d7 %d Depth Points" % (n_a, n_z), interp="lanczos",
                    note="Reconstructed from display pixels: dB values and the %g x %g mm size are assumptions." % (wm, dm))
    elif src == "data":
        path = X(b.get("path"))
        if not os.path.isfile(path):
            raise ApiError("Data file not found: %s" % path)
        try:
            db = ob.load_data(path, bool(b.get("linear")), bool(b.get("amplitude")))
        except ValueError as e:
            raise ApiError(str(e))
        wm, dm = float(b.get("width_mm") or 5), float(b.get("depth_mm") or 2)
        name = os.path.splitext(os.path.basename(path))[0]
        meta = dict(source="data", name=name, width_mm=wm, depth_um=dm * 1000.0, labels=False, inset=False,
                    info=["Source: %s" % name[:24], "A-Scans: %d x %d pts" % db.shape],
                    title="B-Scan from data file\n%d A-Scans \u00d7 %d Depth Points" % db.shape, interp="nearest",
                    note="Rendered directly from your data file.")
    else:
        raise ApiError("Unknown source")
    OCT["db"], OCT["meta"] = db, meta
    fig = oct_figure(scale=1.0)
    name = os.path.basename(fig)
    return {"meta": oct_public_meta(), "fig": "/img/" + name}


def oct_ascan(q):
    if OCT["db"] is None:
        raise ApiError("Load or generate a scan first.")
    P = itb.ascan_profile(OCT["db"], int(q.get("idx", 0)), OCT["meta"]["depth_um"], int(q.get("avg", 5)))
    n_a = OCT["db"].shape[0]
    return {"idx": P["idx"], "x_mm": (P["idx"] + 0.5) * OCT["meta"]["width_mm"] / n_a, "z": rnd(P["z"], 1),
            "raw": rnd(P["raw"]), "smooth": rnd(P["smooth"]), "lo": P["lo"], "hi": P["hi"], "dz": P["dz"],
            "peaks": [{"z": p[1], "db": p[2]} for p in P["peaks"]], "depth_um": OCT["meta"]["depth_um"]}


def oct_export(b):
    if OCT["db"] is None:
        raise ApiError("Load or generate a scan first.")
    out_dir = X(CFG["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    m, db = OCT["meta"], OCT["db"]
    stem = m["name"]
    kind = b.get("kind")
    idx = int(b.get("idx", db.shape[0] // 2))
    scale = float(np.clip(float(b.get("scale") or 2), 1, 4))
    if kind == "bscan":
        marker = (idx + 0.5) * m["width_mm"] / db.shape[0] if b.get("marker") else None
        path = os.path.join(out_dir, stem + "_bscan.png")
        with RLOCK:
            oct_figure(scale=scale, interp=b.get("interp") or None, marker=marker, out=path)
    elif kind == "ascan":
        path = os.path.join(out_dir, "%s_ascan_%d.png" % (stem, idx))
        with RLOCK:
            itb.plot_ascan(db, idx, m["width_mm"], m["depth_um"], path, avg=int(b.get("avg") or 5), scale=scale)
    elif kind == "npy":
        path = os.path.join(out_dir, stem + "_ascans.npy")
        np.save(path, db)
    else:
        raise ApiError("Unknown export kind")
    return {"path": path}


# ----------------------------------------------------------------------------
# page 2: OCT volume (stack of B-scans) -> B-scan browser, A-scan, C-scan (en-face)
# ----------------------------------------------------------------------------
VOL = {"g": None, "meta": {}, "cache": (None, None)}


def vol_load(b):
    folder = X(b.get("folder"))
    if not os.path.isdir(folder):
        raise ApiError("Folder not found: %s" % folder)
    files = sorted([f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(IMG_EXT)], key=natural_key)
    if len(files) < 2:
        raise ApiError("Need at least 2 images in the folder (found %d)." % len(files))
    masks, crop = parse_boxes(b.get("mask")), parse_box(b.get("crop"))
    g0, is_color = itb.load_gray(files[0], crop, False, masks)
    if is_color and not b.get("force"):
        raise ApiError("The first image is a colour photo (fundus-like). A volume needs grayscale OCT B-scans.")
    h, w = g0.shape
    max_w = int(b.get("max_w") or 512)
    f = min(1.0, max_w / float(w))
    n = len(files)
    while n * int(h * f) * int(w * f) > 600e6:
        f *= 0.8
    H, W = max(8, int(h * f)), max(8, int(w * f))
    vol = np.zeros((n, H, W), np.uint8)
    for i, fp in enumerate(files):
        g, _ = itb.load_gray(fp, crop, False, masks)
        im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        if im.size != (W, H):
            im = im.resize((W, H), Image.LANCZOS)
        vol[i] = np.asarray(im)
    VOL["g"] = vol
    VOL["cache"] = (None, None)
    VOL["meta"] = dict(n=n, h=H, w=W, folder=folder, first=os.path.basename(files[0]), last=os.path.basename(files[-1]),
                       width_mm=float(b.get("width_mm") or 6), depth_mm=float(b.get("depth_mm") or 2),
                       spacing_mm=float(b.get("spacing_mm") or 0.05))
    return dict(VOL["meta"], note="Slices are used in sorted order. If these are independent images (different "
                                  "patients/eyes) the C-scan will be meaningless - check with the Dataset Inspector.")


def vol_need():
    if VOL["g"] is None:
        raise ApiError("Load a folder of B-scans first.")
    return VOL["g"], VOL["meta"]


def vol_db(i):
    g, _ = vol_need()
    if VOL["cache"][0] == i:
        return VOL["cache"][1]
    H, W = g.shape[1:]
    db = itb.image_to_ascans(g[i].astype(float), W, H, 80.0, 1.0, False, True, -74.0)    # (W, H)
    VOL["cache"] = (i, db)
    return db


def vol_slice(q):
    g, m = vol_need()
    i = int(np.clip(int(q.get("i", 0)), 0, m["n"] - 1))
    if q.get("cmap") == "gray":
        arr = np.stack([g[i]] * 3, axis=2)
    else:
        arr = colorize_db(vol_db(i).T)
    return ("image/png", png_bytes(arr))


def vol_cscan(q):
    g, m = vol_need()
    z0 = int(np.clip(int(q.get("z0", 0)), 0, m["h"] - 1))
    z1 = int(np.clip(int(q.get("z1", m["h"] // 2)), z0 + 1, m["h"]))
    sub = g[:, z0:z1, :].astype(float)
    en = sub.max(axis=1) if q.get("mode") == "max" else sub.mean(axis=1)          # (N, W)
    lo, hi = np.percentile(en, 1), np.percentile(en, 99.5)
    en8 = np.clip((en - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    if q.get("cmap") == "gray":
        arr = np.stack([en8] * 3, axis=2)
    else:
        arr = LUT256[en8]
    px_h = int(np.clip(m["n"] * m["spacing_mm"] * m["w"] / max(m["width_mm"], 1e-6), max(m["n"], 160), 1400))
    im = Image.fromarray(arr).resize((m["w"], px_h), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return ("image/png", buf.getvalue())


def vol_ascan(q):
    g, m = vol_need()
    i = int(np.clip(int(q.get("i", 0)), 0, m["n"] - 1))
    P = itb.ascan_profile(vol_db(i), int(q.get("x", 0)), m["depth_mm"] * 1000.0, int(q.get("avg", 5)))
    return {"idx": P["idx"], "slice": i, "x_mm": (P["idx"] + 0.5) * m["width_mm"] / m["w"], "z": rnd(P["z"], 1),
            "raw": rnd(P["raw"]), "smooth": rnd(P["smooth"]), "lo": P["lo"], "hi": P["hi"], "dz": P["dz"],
            "peaks": [{"z": p[1], "db": p[2]} for p in P["peaks"]], "depth_um": m["depth_mm"] * 1000.0}


def write_nifti(path, vol, spacing):
    """minimal NIfTI-1 writer (uint8). vol shape (N, H, W) -> dims (W, H, N); gzip if path ends with .gz"""
    n, h, w = vol.shape
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)
    dim = [3, w, h, n, 1, 1, 1, 1]
    struct.pack_into("<8h", hdr, 40, *dim)
    struct.pack_into("<h", hdr, 70, 2)                  # datatype uint8
    struct.pack_into("<h", hdr, 72, 8)                  # bitpix
    pix = [1.0, spacing[0], spacing[1], spacing[2], 0, 0, 0, 0]
    struct.pack_into("<8f", hdr, 76, *pix)
    struct.pack_into("<f", hdr, 108, 352.0)             # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)               # scl_slope
    struct.pack_into("<f", hdr, 116, 0.0)               # scl_inter
    hdr[123] = 10                                       # xyzt_units = mm
    struct.pack_into("<h", hdr, 252, 0)                 # qform_code
    struct.pack_into("<h", hdr, 254, 1)                 # sform_code
    struct.pack_into("<4f", hdr, 280, spacing[0], 0, 0, 0)
    struct.pack_into("<4f", hdr, 296, 0, spacing[1], 0, 0)
    struct.pack_into("<4f", hdr, 312, 0, 0, spacing[2], 0)
    hdr[344:348] = b"n+1\0"
    data = bytes(hdr) + b"\0\0\0\0" + np.ascontiguousarray(vol).tobytes()
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wb") as f:
        f.write(data)


def vol_export_nii(b):
    g, m = vol_need()
    out_dir = X(CFG["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, os.path.basename(m["folder"].rstrip("/")) + "_volume.nii.gz")
    sp = (m["width_mm"] / m["w"], m["depth_mm"] / m["h"], m["spacing_mm"])
    write_nifti(path, g, sp)
    res = {"path": path}
    if b.get("open"):
        exe = need_bin("eyeview2")
        iso = float(b.get("iso") or 90)
        res.update(start_job("3D viewer (volume)", [exe, path, "--iso", str(iso), "--up", "1", "--out",
                                                    os.path.join(out_dir, "volume_mesh")]))
    return res


# ----------------------------------------------------------------------------
# page 3: fundus -> 3D (img2mesh) + AI depth
# ----------------------------------------------------------------------------
def fundus_depth(b):
    img = X(b.get("image"))
    if not os.path.isfile(img):
        raise ApiError("Image not found: %s" % img)
    py, sc = X(CFG["depth_python"]), X(CFG["depth_script"])
    if not os.path.isfile(py) or not os.path.isfile(sc):
        raise ApiError("Depth tool not found (%s, %s). Fix the paths in Settings." % (py, sc))
    out = os.path.join(X(CFG["out_dir"]), "fundus_depth")
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, os.path.splitext(os.path.basename(img))[0] + "_depth.png")
    res = start_job("AI depth map", [py, sc, img, dst])
    res["depth"] = dst
    return res


def fundus_mesh(b):
    img = X(b.get("image"))
    if not os.path.isfile(img):
        raise ApiError("Image not found: %s" % img)
    exe = need_bin("img2mesh")
    out = os.path.join(X(CFG["out_dir"]), "fundus_mesh", os.path.splitext(os.path.basename(img))[0])
    os.makedirs(out, exist_ok=True)
    cmd = [exe, img, "--mode", b.get("mode") or "sphere", "--relief", str(b.get("relief") or 0.06),
           "--step", str(int(b.get("step") or 1)), "--sigma", str(b.get("sigma") or 2),
           "--channel", str(int(b.get("channel") or 1)), "--cap", str(b.get("cap") or 60),
           "--thr", str(b.get("thr") or 25), "--out", out]
    if b.get("depth"):
        dp = X(b["depth"])
        if not os.path.isfile(dp):
            raise ApiError("Depth map not found: %s" % dp)
        cmd += ["--height", dp]
    if b.get("invert"):
        cmd.append("--invert")
    if b.get("convex"):
        cmd.append("--convex")
    if not b.get("interactive"):
        cmd.append("--no-window")
    res = start_job("Fundus 3D mesh", cmd)
    res["out"] = out
    return res


# ----------------------------------------------------------------------------
# page 4: MRI eye atlas (eyeview2)
# ----------------------------------------------------------------------------
def atlas_run(b):
    labels = X(b.get("labels"))
    if not os.path.isfile(labels):
        raise ApiError("Labels file not found: %s" % labels)
    exe = need_bin("eyeview2")
    out = os.path.join(X(CFG["out_dir"]), "atlas_mesh")
    os.makedirs(out, exist_ok=True)
    cmd = [exe, labels, "--up", str(int(b.get("up") or 3)), "--out", out]
    if b.get("mri") and os.path.isfile(X(b["mri"])):
        cmd += ["--mri", X(b["mri"])]
    if b.get("names"):
        cmd += ["--names", b["names"]]
    if b.get("hide"):
        cmd += ["--hide", b["hide"]]
    if b.get("iso"):
        cmd += ["--iso", str(b["iso"])]
    if not b.get("interactive"):
        cmd.append("--no-window")
    res = start_job("Eye atlas 3D", cmd)
    res["out"] = out
    return res


# ----------------------------------------------------------------------------
# page 5: dataset inspector
# ----------------------------------------------------------------------------
def thumb(path):
    im = Image.open(path).convert("L").resize((32, 32), Image.BILINEAR)
    a = np.asarray(im, float).ravel()
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-6 else a * 0


def classify_image(path):
    im = Image.open(path)
    rgb = np.asarray(im.convert("RGB"), float)
    sat = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
    kind = "grayscale (OCT-like)" if sat <= 12 else ("colour, reddish (fundus-like)" if rgb[..., 0].mean() > rgb[..., 2].mean() * 1.3 else "colour")
    return im.size, im.mode, kind


def inspect_folder(b):
    root = X(b.get("folder"))
    if not os.path.isdir(root):
        raise ApiError("Folder not found: %s" % root)
    ext_count, dir_count, dir_imgs, total = Counter(), Counter(), {}, 0
    for dp, _, fn in os.walk(root):
        for f in fn:
            total += 1
            if total > 60000:
                break
            e = os.path.splitext(f)[1].lower() or "(none)"
            ext_count[e] += 1
            dir_count[dp] += 1
            if e in IMG_EXT:
                dir_imgs.setdefault(dp, []).append(os.path.join(dp, f))
    allimgs = [p for v in dir_imgs.values() for p in v]
    samples = []
    if allimgs:
        pick = [allimgs[int(i)] for i in np.linspace(0, len(allimgs) - 1, min(12, len(allimgs)))]
        for p in pick:
            try:
                size, mode, kind = classify_image(p)
                samples.append({"file": os.path.relpath(p, root), "size": "%d x %d" % size, "mode": mode, "kind": kind})
            except Exception as e:          # noqa: BLE001
                samples.append({"file": os.path.relpath(p, root), "size": "?", "mode": "?", "kind": "unreadable: %s" % e})
    stacks = []
    for d, imgs in sorted(dir_imgs.items(), key=lambda kv: -len(kv[1]))[:6]:
        if len(imgs) < 8:
            continue
        imgs = sorted(imgs, key=natural_key)[:64]
        try:
            T = [thumb(p) for p in imgs]
        except Exception:                   # noqa: BLE001
            continue
        c1 = float(np.mean([np.mean(T[i] * T[i + 1]) for i in range(len(T) - 1)]))
        rng = random.Random(1)
        c0 = float(np.mean([np.mean(T[i] * T[j]) for i, j in (rng.sample(range(len(T)), 2) for _ in range(200))]))
        if c1 >= 0.97:
            v = "Neighbouring images are nearly identical -> probably augmented copies of one image, not slices."
        elif c0 >= 0.45 and c1 - c0 < 0.2:
            v = "ALL images look alike (even random pairs) -> probably variants/augmentations of one image, not a volume."
        elif c1 >= 0.7 and c1 - c0 >= 0.25:
            v = "Neighbouring images are much more similar than random pairs -> could be a real slice stack (verify by eye)."
        else:
            v = "Neighbouring images are unrelated -> independent images (e.g. different patients), not a volume."
        stacks.append({"folder": os.path.relpath(d, root) if d != root else ".", "images": len(dir_imgs[d]),
                       "consecutive_corr": round(c1, 2), "random_corr": round(c0, 2), "verdict": v})
    volumetric = [e for e in ext_count if e in (".dcm", ".nii", ".gz", ".mhd", ".nrrd", ".vol", ".e2e", ".fds", ".oct")]
    return {"total": total, "ext": ext_count.most_common(12), "dirs": [[os.path.relpath(d, root) or ".", c] for d, c in dir_count.most_common(10)],
            "samples": samples, "stacks": stacks, "volumetric_ext": volumetric}


# ----------------------------------------------------------------------------
# page 6: settings, environment, file browser
# ----------------------------------------------------------------------------
def env_check():
    rows = []
    for name in ("eyeview2", "img2mesh", "eyeview"):
        p = os.path.join(X(CFG["build_dir"]), name)
        rows.append({"item": "C++ tool: " + name, "ok": os.path.isfile(p) and os.access(p, os.X_OK), "detail": p})
    for k in ("depth_python", "depth_script", "atlas_labels", "atlas_mri", "data_dir", "cmake_src"):
        p = X(CFG[k])
        rows.append({"item": k, "ok": os.path.exists(p), "detail": p})
    for mod in ("numpy", "scipy", "matplotlib", "PIL"):
        try:
            __import__(mod)
            rows.append({"item": "python: " + mod, "ok": True, "detail": ""})
        except Exception as e:              # noqa: BLE001
            rows.append({"item": "python: " + mod, "ok": False, "detail": str(e)})
    for tool in ("cmake", "make", "xdg-open"):
        w = shutil.which(tool)
        rows.append({"item": "command: " + tool, "ok": bool(w), "detail": w or "not found"})
    return {"rows": rows}


def build_tools(_b):
    src, bd = X(CFG["cmake_src"]), X(CFG["build_dir"])
    if not os.path.isfile(os.path.join(src, "CMakeLists.txt")):
        raise ApiError("No CMakeLists.txt in %s" % src)
    cmd = 'cmake -S "%s" -B "%s" -DCMAKE_BUILD_TYPE=Release -Wno-dev && cmake --build "%s" -j %d' % (src, bd, bd, os.cpu_count() or 2)
    return start_job("Build C++ tools", cmd, shell=True)


def list_dir(q):
    path = X(q.get("path")) or HOME
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        path = HOME
    path = os.path.abspath(path)
    entries = []
    try:
        for name in sorted(os.listdir(path), key=natural_key):
            if name.startswith(".") and q.get("hidden") != "1":
                continue
            fp = os.path.join(path, name)
            try:
                isd = os.path.isdir(fp)
                size = 0 if isd else os.path.getsize(fp)
            except OSError:
                continue
            entries.append({"name": name, "dir": isd, "size": size})
    except PermissionError:
        raise ApiError("Permission denied: %s" % path)
    entries.sort(key=lambda e: (not e["dir"],))
    return {"path": path, "parent": os.path.dirname(path), "entries": entries[:3000]}


def open_path(b):
    p = X(b.get("path"))
    if not os.path.exists(p):
        raise ApiError("Path does not exist: %s" % p)
    w = shutil.which("xdg-open") or shutil.which("open")
    if not w:
        raise ApiError("xdg-open not found")
    subprocess.Popen([w, p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"ok": True}


def preview_image(q):
    p = X(q.get("path"))
    if not (os.path.isfile(p) and p.lower().endswith(IMG_EXT)):
        raise ApiError("Not an image file")
    im = Image.open(p).convert("RGB")
    im.thumbnail((int(q.get("max", 480)),) * 2)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return ("image/png", buf.getvalue())


# ----------------------------------------------------------------------------
# HTTP layer
# ----------------------------------------------------------------------------
def config_endpoint(b):
    if b:
        for k, v in b.items():
            if k in DEFAULTS:
                CFG[k] = str(v)
        save_cfg()
    return dict(CFG)


ROUTES = {
    ("GET", "/api/config"): lambda q, b: dict(CFG),
    ("POST", "/api/config"): lambda q, b: config_endpoint(b),
    ("GET", "/api/ls"): lambda q, b: list_dir(q),
    ("GET", "/api/job"): lambda q, b: job_status(q),
    ("GET", "/api/env"): lambda q, b: env_check(),
    ("POST", "/api/build"): lambda q, b: build_tools(b),
    ("POST", "/api/open"): lambda q, b: open_path(b),
    ("GET", "/api/preview"): lambda q, b: preview_image(q),
    ("POST", "/api/oct/load"): lambda q, b: oct_load(b),
    ("GET", "/api/oct/ascan"): lambda q, b: oct_ascan(q),
    ("POST", "/api/oct/export"): lambda q, b: oct_export(b),
    ("POST", "/api/volume/load"): lambda q, b: vol_load(b),
    ("GET", "/api/volume/slice"): lambda q, b: vol_slice(q),
    ("GET", "/api/volume/cscan"): lambda q, b: vol_cscan(q),
    ("GET", "/api/volume/ascan"): lambda q, b: vol_ascan(q),
    ("POST", "/api/volume/nifti"): lambda q, b: vol_export_nii(b),
    ("POST", "/api/fundus/depth"): lambda q, b: fundus_depth(b),
    ("POST", "/api/fundus/mesh"): lambda q, b: fundus_mesh(b),
    ("POST", "/api/atlas/run"): lambda q, b: atlas_run(b),
    ("POST", "/api/inspect"): lambda q, b: inspect_folder(b),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "EyeScanStudio/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, "application/json; charset=utf-8", json.dumps(obj).encode("utf-8"))

    def _handle(self, method):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if method == "GET" and u.path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", INDEX_HTML.encode("utf-8"))
            if method == "GET" and u.path.startswith("/img/"):
                fp = os.path.join(TMP, os.path.basename(u.path))
                if not os.path.isfile(fp):
                    return self._send(404, "text/plain", b"not found")
                with open(fp, "rb") as f:
                    return self._send(200, "image/png", f.read())
            body = {}
            if method == "POST":
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b"{}"
                body = json.loads(raw or b"{}")
            fn = ROUTES.get((method, u.path))
            if not fn:
                return self._send(404, "text/plain", b"unknown endpoint")
            res = fn(q, body)
            if isinstance(res, tuple):
                return self._send(200, res[0], res[1])
            return self._json(res)
        except ApiError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:              # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": "%s: %s" % (type(e).__name__, e)}, 500)

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")


# ----------------------------------------------------------------------------
# front-end (single page, no external libraries)
# ----------------------------------------------------------------------------
INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eye Scan Studio</title>
<style>
:root{--bg:#eef2f7;--card:#fff;--ink:#0f172a;--mut:#64748b;--line:#dbe3ee;--acc:#2563eb;--acc2:#1d4ed8;--ok:#16a34a;--bad:#dc2626;--side:#0b1220}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--ink);display:flex;min-height:100vh}
nav{width:236px;background:var(--side);color:#cbd5e1;padding:18px 12px;position:sticky;top:0;height:100vh;overflow:auto;flex:none}
nav h1{font-size:17px;color:#fff;margin:2px 8px 4px}nav .sub{font-size:11px;color:#64748b;margin:0 8px 16px}
nav button{display:flex;gap:10px;align-items:center;width:100%;text-align:left;background:none;border:0;color:inherit;padding:10px;border-radius:9px;font-size:14px;cursor:pointer;margin-bottom:3px}
nav button:hover{background:#16233b}nav button.on{background:var(--acc);color:#fff}nav .ic{width:22px;text-align:center}
main{flex:1;min-width:0;padding:22px 26px 260px}
h2{margin:0 0 4px;font-size:22px}.lead{color:var(--mut);margin:0 0 16px;font-size:14px}
.page{display:none}.page.on{display:block}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;cursor:pointer;transition:.15s;text-align:left}
.tile:hover{border-color:var(--acc);box-shadow:0 6px 22px rgba(37,99,235,.15);transform:translateY(-2px)}
.tile .big{font-size:30px}.tile h3{margin:8px 0 4px;font-size:16px}.tile p{margin:0;color:var(--mut);font-size:13px;line-height:1.45}
.tile .tag{display:inline-block;margin-top:10px;font-size:11px;padding:2px 8px;border-radius:99px;background:#e0e7ff;color:#3730a3}
.row{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:flex-end;margin-bottom:10px}
label{display:flex;flex-direction:column;font-size:12px;color:var(--mut);gap:4px}
input[type=text],input[type=number],select{padding:8px 10px;border:1px solid var(--line);border-radius:8px;font-size:14px;background:#fff;color:var(--ink);min-width:90px}
input.path{min-width:420px;width:min(560px,70vw)}
label.chk{flex-direction:row;align-items:center;gap:6px;color:var(--ink);font-size:13px}
button.btn{background:var(--acc);color:#fff;border:0;border-radius:9px;padding:9px 15px;font-size:14px;cursor:pointer;font-weight:600}
button.btn:hover{background:var(--acc2)}button.btn.alt{background:#e2e8f0;color:var(--ink)}button.btn.alt:hover{background:#cbd5e1}
button.btn:disabled{opacity:.5;cursor:wait}
.tabs{display:flex;gap:6px;margin-bottom:12px}.tabs button{padding:8px 14px;border-radius:9px;border:1px solid var(--line);background:#fff;cursor:pointer;font-size:14px}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.src{display:none}.src.on{display:block}
.viewer{position:relative;display:inline-block;width:100%;max-width:1100px;line-height:0}
.viewer img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);cursor:crosshair;background:#000}
.marker{position:absolute;width:0;border-left:2px dashed #fff;mix-blend-mode:difference;pointer-events:none}
canvas{width:100%;max-width:1100px;height:380px;background:#fff;border:1px solid var(--line);border-radius:8px;display:block}
.note{font-size:12.5px;color:var(--mut);margin:8px 0}.warn{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;padding:9px 12px;border-radius:8px;font-size:13px;margin:8px 0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:13px}.kv b{color:var(--mut);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}
.ok{color:var(--ok);font-weight:700}.no{color:var(--bad);font-weight:700}
.three{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.three img{width:100%;border:1px solid var(--line);border-radius:8px;cursor:crosshair;background:#000}
.pv{display:flex;gap:14px;flex-wrap:wrap}.pv img{max-height:230px;border-radius:8px;border:1px solid var(--line)}
#console{position:fixed;left:236px;right:0;bottom:0;height:200px;background:#0b1220;color:#d1fae5;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;display:flex;flex-direction:column;border-top:2px solid #1e293b;z-index:20}
#console.small{height:34px}#conbar{display:flex;justify-content:space-between;align-items:center;padding:6px 12px;background:#111b2e;color:#94a3b8;font-size:12px;cursor:pointer}
#conlog{flex:1;overflow:auto;padding:6px 12px;white-space:pre-wrap}#conlog .e{color:#fca5a5}#conlog .c{color:#93c5fd}
#busy{position:fixed;inset:0;background:rgba(15,23,42,.35);display:none;align-items:center;justify-content:center;z-index:50;color:#fff;font-size:16px}
#busy div{background:#0f172a;padding:18px 26px;border-radius:12px}
#modal{position:fixed;inset:0;background:rgba(15,23,42,.5);display:none;align-items:center;justify-content:center;z-index:60}
#mbox{background:#fff;border-radius:14px;width:min(760px,94vw);max-height:82vh;display:flex;flex-direction:column;overflow:hidden}
#mtop{display:flex;gap:8px;padding:12px;border-bottom:1px solid var(--line)}#mtop input{flex:1}
#mlist{overflow:auto;padding:6px;min-height:240px}.ent{display:flex;gap:10px;padding:7px 10px;border-radius:8px;cursor:pointer;font-size:14px}.ent:hover{background:#eef2ff}
.ent .sz{margin-left:auto;color:var(--mut);font-size:12px}#mbot{padding:10px 12px;border-top:1px solid var(--line);display:flex;gap:8px;justify-content:flex-end}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;font-size:12px;background:#e2e8f0;margin-left:8px}
.help{font-size:13px;line-height:1.7}.help kbd{background:#e2e8f0;border-radius:5px;padding:1px 6px;font-family:ui-monospace,monospace;font-size:12px}
@media(max-width:900px){nav{display:none}#console{left:0}}
</style></head><body>
<nav>
 <h1>👁 Eye Scan Studio</h1><div class="sub">OCT · Fundus · MRI atlas · 3D</div>
 <button data-page="home" class="on"><span class="ic">🏠</span>Home</button>
 <button data-page="oct"><span class="ic">📈</span>OCT A-scan / B-scan</button>
 <button data-page="vol"><span class="ic">🧊</span>OCT Volume / C-scan</button>
 <button data-page="fundus"><span class="ic">🔴</span>Fundus → 3D</button>
 <button data-page="atlas"><span class="ic">🧠</span>MRI Eye Atlas 3D</button>
 <button data-page="inspect"><span class="ic">🔎</span>Dataset Inspector</button>
 <button data-page="settings"><span class="ic">⚙️</span>Settings & Tools</button>
</nav>
<main>

<!-- HOME -->
<section class="page on" id="page-home">
 <h2>Choose a scan type</h2><p class="lead">Click a card. Every tool built in this project is here; each page tells you what data it needs.</p>
 <div class="grid">
  <div class="tile" data-go="oct" data-src="sim"><div class="big">🧪</div><h3>Simulated OCT B-scan</h3><p>The retina phantom (ILM → choroid, foveal pit, speckle) matching your reference figure, with clickable A-scans.</p><span class="tag">no data needed</span></div>
  <div class="tile" data-go="oct" data-src="image"><div class="big">📈</div><h3>OCT image → A-scan & B-scan</h3><p>Load a real OCT B-scan image (e.g. OCTID). Click anywhere to see that A-scan, then export hi-res figures.</p><span class="tag">grayscale OCT images</span></div>
  <div class="tile" data-go="oct" data-src="data"><div class="big">📄</div><h3>B-scan from data file</h3><p>Render an array exported from OptiSystem or another tool (.npy / .csv / .txt, 512×1024).</p><span class="tag">.npy · .csv</span></div>
  <div class="tile" data-go="vol"><div class="big">🧊</div><h3>OCT volume · C-scan (en-face)</h3><p>Stack a folder of B-scans: browse slices, pick A-scans, and view en-face C-scans at any depth. Exports NIfTI for 3D.</p><span class="tag">real slice stacks</span></div>
  <div class="tile" data-go="fundus"><div class="big">🔴</div><h3>Fundus photo → 3D</h3><p>Colour fundus image onto a retinal dome, optional AI depth relief, opened in the VTK 3D viewer.</p><span class="tag">estimated depth</span></div>
  <div class="tile" data-go="atlas"><div class="big">🧠</div><h3>MRI eye atlas → 3D</h3><p>Real MRI volume of the eye (lens, globe, optic nerve, muscles) with cut-away planes and MRI slices.</p><span class="tag">NIfTI atlas</span></div>
  <div class="tile" data-go="inspect"><div class="big">🔎</div><h3>Dataset inspector</h3><p>Find out what is in a folder: image types, sizes, and whether images form a real volume or are augmented copies.</p><span class="tag">check before you use</span></div>
  <div class="tile" data-go="settings"><div class="big">⚙️</div><h3>Settings & tools</h3><p>Paths, environment check, and a button that builds the C++ tools (eyeview2, img2mesh).</p><span class="tag">setup</span></div>
 </div>
 <div class="card" style="margin-top:16px"><b>Read this once.</b>
  <p class="note">Only <b>OCT B-scan images</b> contain depth, so only they can give A-scans. Fundus photos and STARE images are surface photographs: the 3D shown from them is an <b>estimate</b> for display. All outputs here are for visualisation and study, not diagnosis.</p></div>
</section>

<!-- OCT -->
<section class="page" id="page-oct">
 <h2>OCT A-scan / B-scan</h2><p class="lead">B-scan = many A-scans side by side. Click the B-scan (or use the slider) to inspect one A-scan.</p>
 <div class="tabs" id="octtabs"><button data-src="sim" class="on">Simulated phantom</button><button data-src="image">OCT image</button><button data-src="data">Data file</button></div>
 <div class="card">
  <div class="src on" id="src-sim"><div class="row"><label>Noise seed<input type="number" id="sim_seed" value="1" style="width:100px"></label><button class="btn" id="go-sim">Generate B-scan</button></div>
   <div class="note">Layers: ILM, NFL, GCL, IPL, OPL, ONL, ELM, IS/OS, RPE, Choroid — calibrated against one reference figure.</div></div>
  <div class="src" id="src-image">
   <div class="row"><label>OCT image<input type="text" class="path" id="im_path" placeholder="/path/to/CSR4.jpeg"></label><button class="btn alt" onclick="pick('im_path','file')">📂 Browse</button></div>
   <div class="row"><label>Width (mm)<input type="number" id="im_w" value="6" step="0.5" style="width:90px"></label><label>Depth (mm)<input type="number" id="im_d" value="2" step="0.5" style="width:90px"></label>
    <label>Noise floor (dB)<input type="number" id="im_floor" value="-74" style="width:100px"></label>
    <label>Hide boxes (l,t,r,b ; ...)<input type="text" id="im_mask" placeholder="18,378,126,492" style="width:200px"></label>
    <label>Crop (l,t,r,b)<input type="text" id="im_crop" placeholder="optional" style="width:150px"></label></div>
   <div class="row"><label class="chk"><input type="checkbox" id="im_auto">auto-crop</label><label class="chk"><input type="checkbox" id="im_inv">invert</label>
    <label class="chk"><input type="checkbox" id="im_info">info box</label><label class="chk"><input type="checkbox" id="im_inset">axes arrows</label><label class="chk"><input type="checkbox" id="im_force">allow colour image</label>
    <button class="btn" id="go-image">Convert to A-scans + B-scan</button></div>
   <div class="note">OCTID images have a scan-direction square at the bottom-left: hide it with <code>18,378,126,492</code>. Width/depth/dB are assumptions (the image carries no physical scale).</div></div>
  <div class="src" id="src-data">
   <div class="row"><label>Data file<input type="text" class="path" id="da_path" placeholder="/path/data.npy"></label><button class="btn alt" onclick="pick('da_path','file')">📂 Browse</button></div>
   <div class="row"><label>Width (mm)<input type="number" id="da_w" value="5" step="0.5" style="width:90px"></label><label>Depth (mm)<input type="number" id="da_d" value="2" step="0.5" style="width:90px"></label>
    <label class="chk"><input type="checkbox" id="da_lin">linear values</label><label class="chk"><input type="checkbox" id="da_amp">amplitude (20·log)</label><button class="btn" id="go-data">Load & render</button></div>
   <div class="note">Shape must be 512×1024 or 1024×512 (A-scans × depth points).</div></div>
 </div>
 <div id="oct-out" style="display:none">
  <div class="card"><div class="kv" id="oct-meta"></div><div class="warn" id="oct-note" style="display:none"></div></div>
  <div class="card"><b>B-scan</b> <span class="chip" id="oct-pos"></span>
   <div class="viewer" style="margin-top:8px"><img id="bscan-img" alt="B-scan"><div class="marker" id="bscan-marker"></div></div></div>
  <div class="card"><b>A-scan</b>
   <div class="row" style="margin:8px 0"><label>A-scan index <input type="number" id="oct_idx" value="0" min="0" style="width:100px"></label>
    <input type="range" id="oct_slider" min="0" max="511" value="0" style="width:min(520px,60vw)">
    <label>Average of neighbours<input type="number" id="oct_avg" value="5" min="1" max="31" style="width:80px"></label></div>
   <canvas id="oct-canvas"></canvas></div>
  <div class="card"><b>Export</b>
   <div class="row" style="margin-top:8px"><label>Resolution ×<select id="ex_scale"><option>1</option><option selected>2</option><option>3</option><option>4</option></select></label>
    <label>Interpolation<select id="ex_interp"><option value="">(default)</option><option>lanczos</option><option>bicubic</option><option>bilinear</option><option>nearest</option></select></label>
    <label class="chk"><input type="checkbox" id="ex_marker" checked>mark A-scan on B-scan</label></div>
   <div class="row"><button class="btn" onclick="octExport('bscan')">💾 B-scan PNG</button><button class="btn" onclick="octExport('ascan')">💾 A-scan PNG</button><button class="btn alt" onclick="octExport('npy')">💾 A-scans .npy</button>
    <button class="btn alt" onclick="openOut()">📂 Open results folder</button></div></div>
 </div>
</section>

<!-- VOLUME -->
<section class="page" id="page-vol">
 <h2>OCT volume · C-scan</h2><p class="lead">Load a folder of B-scan images that belong to ONE eye (a real slice stack). You then get B-scans, A-scans and en-face C-scans.</p>
 <div class="card">
  <div class="row"><label>Folder of B-scans<input type="text" class="path" id="vo_folder" placeholder="/path/to/stack"></label><button class="btn alt" onclick="pick('vo_folder','dir')">📂 Browse</button></div>
  <div class="row"><label>Width (mm)<input type="number" id="vo_w" value="6" step="0.5" style="width:90px"></label><label>Depth (mm)<input type="number" id="vo_d" value="2" step="0.5" style="width:90px"></label>
   <label>Slice spacing (mm)<input type="number" id="vo_sp" value="0.05" step="0.01" style="width:110px"></label><label>Max width (px)<input type="number" id="vo_maxw" value="512" style="width:100px"></label>
   <label>Hide boxes<input type="text" id="vo_mask" placeholder="18,378,126,492" style="width:190px"></label><label class="chk"><input type="checkbox" id="vo_force">allow colour</label>
   <button class="btn" id="go-vol">Load stack</button></div>
  <div class="warn">Folders like OCTID (one B-scan per patient) are <b>not</b> volumes: the C-scan would mix different people. Use the Dataset Inspector if unsure.</div>
 </div>
 <div id="vol-out" style="display:none">
  <div class="card"><div class="kv" id="vol-meta"></div></div>
  <div class="row"><label>Colour<select id="vo_cmap"><option value="oct">OCT colours</option><option value="gray">Gray</option></select></label>
   <label>Depth range from (px)<input type="number" id="vo_z0" value="0" style="width:100px"></label><label>to (px)<input type="number" id="vo_z1" value="50" style="width:100px"></label>
   <label>Projection<select id="vo_mode"><option value="mean">mean</option><option value="max">max</option></select></label>
   <label>A-scan avg<input type="number" id="vo_avg" value="5" min="1" style="width:80px"></label></div>
  <div class="three">
   <div class="card"><b>B-scan</b> <span class="chip" id="vo-pos"></span><input type="range" id="vo_slider" min="0" max="0" value="0" style="width:100%"><div style="position:relative;line-height:0"><img id="vo-bscan"><div class="marker" id="vo-bmark"></div></div><div class="note">Click to choose the A-scan column.</div></div>
   <div class="card"><b>C-scan (en-face)</b><div class="note" style="margin-top:2px">Vertical = slice number, horizontal = lateral. Click to jump to a slice/column.</div><div style="position:relative;line-height:0"><img id="vo-cscan"><div class="marker" id="vo-cmark"></div></div></div>
  </div>
  <div class="card"><b>A-scan</b> <span class="chip" id="vo-ascan-pos"></span><canvas id="vo-canvas" style="margin-top:8px"></canvas></div>
  <div class="card"><b>3D</b><div class="row" style="margin-top:8px"><label>Surface threshold (0–255)<input type="number" id="vo_iso" value="90" style="width:110px"></label>
   <button class="btn" id="go-nii">Export NIfTI + open 3D viewer</button><button class="btn alt" id="go-nii2">Export NIfTI only</button></div>
   <div class="note">Uses the C++ viewer (eyeview2 --iso). Experimental for OCT: voxel spacing comes from the values above.</div></div>
 </div>
</section>

<!-- FUNDUS -->
<section class="page" id="page-fundus">
 <h2>Fundus photo → 3D</h2><p class="lead">A fundus photo has no depth. The dome shape and relief shown here are an estimate for display only.</p>
 <div class="card">
  <div class="row"><label>Fundus image<input type="text" class="path" id="fu_img" placeholder="/path/to/fundus.png"></label><button class="btn alt" onclick="pick('fu_img','file')">📂 Browse</button></div>
  <div class="row"><label>Depth map (optional)<input type="text" class="path" id="fu_depth" placeholder="generate with the AI button, or browse"></label><button class="btn alt" onclick="pick('fu_depth','file')">📂 Browse</button>
   <button class="btn alt" id="go-depth">🤖 Generate AI depth map</button></div>
  <div class="pv" id="fu-prev"></div>
  <div class="row" style="margin-top:10px"><label>Shape<select id="fu_mode"><option>sphere</option><option>flat</option></select></label><label>Relief<input type="number" id="fu_relief" value="0.06" step="0.01" style="width:90px"></label>
   <label>Step (1 = finest)<input type="number" id="fu_step" value="1" min="1" style="width:90px"></label><label>Smooth σ<input type="number" id="fu_sigma" value="2" step="0.5" style="width:80px"></label>
   <label>Height channel<select id="fu_ch"><option value="0">R</option><option value="1" selected>G</option><option value="2">B</option></select></label>
   <label class="chk"><input type="checkbox" id="fu_inv">invert relief</label><label class="chk"><input type="checkbox" id="fu_conv">convex</label></div>
  <div class="row"><button class="btn" id="go-fu">Build mesh + open 3D viewer</button><button class="btn alt" id="go-fu2">Build mesh only (save files)</button><button class="btn alt" onclick="openOut('fundus_mesh')">📂 Open output</button></div>
  <div class="note">Outputs: .ply (coloured), .stl, .obj. Drag to rotate, wheel to zoom in the viewer window.</div>
 </div>
</section>

<!-- ATLAS -->
<section class="page" id="page-atlas">
 <h2>MRI eye atlas → 3D</h2><p class="lead">Real MRI volume of the eye (MR-Eye atlas). Opens the VTK viewer with cut-away planes and MRI slices.</p>
 <div class="card">
  <div class="row"><label>Labels (.nii.gz)<input type="text" class="path" id="at_lab"></label><button class="btn alt" onclick="pick('at_lab','file')">📂 Browse</button></div>
  <div class="row"><label>MRI image for slices (optional)<input type="text" class="path" id="at_mri"></label><button class="btn alt" onclick="pick('at_mri','file')">📂 Browse</button></div>
  <div class="row"><label>Smoothing ×<select id="at_up"><option>1</option><option>2</option><option selected>3</option><option>4</option></select></label>
   <label>Part names (id=name,…)<input type="text" id="at_names" value="1=lens,2=globe,3=optic_nerve" style="width:300px"></label>
   <label>Hide parts (ids)<input type="text" id="at_hide" placeholder="e.g. 5,6" style="width:130px"></label></div>
  <div class="row"><button class="btn" id="go-at">Open interactive 3D viewer</button><button class="btn alt" id="go-at2">Export meshes only</button><button class="btn alt" onclick="openOut('atlas_mesh')">📂 Open output</button></div>
 </div>
 <div class="card help"><b>Viewer keys</b><br>
  <kbd>1</kbd>–<kbd>9</kbd> show/hide part · <kbd>a</kbd> show all · <kbd>o</kbd> transparency · <kbd>w</kbd> wireframe<br>
  <kbd>z</kbd> cut-away axis X/Y/Z · <kbd>b</kbd> flip side · <kbd>↑</kbd><kbd>↓</kbd> move cut<br>
  <kbd>v</kbd> MRI slice X/Y/Z · <kbd>←</kbd><kbd>→</kbd> move slice · <kbd>r</kbd> reset view · <kbd>k</kbd> screenshot · <kbd>x</kbd> export · <kbd>q</kbd> quit<br>
  Mouse: left = rotate, wheel = zoom, Shift+left = pan.</div>
</section>

<!-- INSPECT -->
<section class="page" id="page-inspect">
 <h2>Dataset inspector</h2><p class="lead">Find out what a data folder really contains before building anything from it.</p>
 <div class="card"><div class="row"><label>Folder<input type="text" class="path" id="in_folder"></label><button class="btn alt" onclick="pick('in_folder','dir')">📂 Browse</button><button class="btn" id="go-in">Inspect</button></div></div>
 <div id="in-out"></div>
</section>

<!-- SETTINGS -->
<section class="page" id="page-settings">
 <h2>Settings & tools</h2><p class="lead">Paths are saved in ~/.eye_scan_studio.json</p>
 <div class="card" id="cfgbox"></div>
 <div class="card"><div class="row"><button class="btn" id="go-save">Save settings</button><button class="btn alt" id="go-env">Check environment</button><button class="btn alt" id="go-build">🛠 Build / rebuild C++ tools</button><button class="btn alt" onclick="openOut()">📂 Open results folder</button></div><div id="env-out"></div></div>
</section>
</main>

<div id="console" class="small"><div id="conbar"><span>Console <span class="chip" id="jobchip">idle</span></span><span id="contoggle">▲ show</span></div><div id="conlog"></div></div>
<div id="busy"><div id="busytxt">Working…</div></div>
<div id="modal"><div id="mbox"><div id="mtop"><button class="btn alt" id="mup">⬆ Up</button><input type="text" id="mpath"><button class="btn alt" id="mgo">Go</button></div><div id="mlist"></div><div id="mbot"><span id="mhint" class="note" style="margin-right:auto"></span><button class="btn alt" id="mcancel">Cancel</button><button class="btn" id="mselect">Select this folder</button></div></div></div>

<script>
const $=id=>document.getElementById(id);
const val=id=>$(id).value, chk=id=>$(id).checked;
let CFG={};

/* ---------- plumbing ---------- */
async function api(path,body){
  const r=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  let j; try{j=await r.json();}catch(e){throw new Error('Bad server reply ('+r.status+')');}
  if(j.error) throw new Error(j.error); return j;
}
function log(msg,cls){const d=document.createElement('div'); if(cls)d.className=cls; d.textContent=msg; $('conlog').appendChild(d); $('conlog').scrollTop=1e9;}
function showConsole(on){ $('console').classList.toggle('small',!on); $('contoggle').textContent=on?'▼ hide':'▲ show'; }
$('conbar').onclick=()=>showConsole($('console').classList.contains('small'));
let busyN=0;
async function busy(msg,fn){ busyN++; $('busytxt').textContent=msg; $('busy').style.display='flex';
  try{ return await fn(); } catch(e){ log('ERROR: '+e.message,'e'); showConsole(true); alert(e.message); } finally{ busyN--; if(!busyN) $('busy').style.display='none'; } }
async function runJob(startPromise,after){
  const s=await startPromise; log('$ '+s.cmd,'c'); showConsole(true); let since=0; $('jobchip').textContent='running…';
  while(true){
    const st=await api('/api/job?id='+s.job+'&since='+since); since=st.next; st.lines.forEach(l=>log(l));
    if(st.done){ $('jobchip').textContent='done (exit '+st.rc+')'; log('— finished: '+st.title+' (exit '+st.rc+', '+st.elapsed+' s)', st.rc? 'e':'c'); if(after) after(st,s); return st; }
    await new Promise(r=>setTimeout(r,500));
  }
}
function wrapJob(msg,fnStart,after){ return async()=>{ try{ await runJob(fnStart(),after);}catch(e){log('ERROR: '+e.message,'e'); showConsole(true); alert(e.message);} }; }

/* ---------- navigation ---------- */
function goPage(p){ document.querySelectorAll('.page').forEach(e=>e.classList.toggle('on',e.id==='page-'+p));
  document.querySelectorAll('nav button').forEach(b=>b.classList.toggle('on',b.dataset.page===p)); window.scrollTo(0,0); if(p==='settings') loadSettings(); }
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>goPage(b.dataset.page));
document.querySelectorAll('.tile').forEach(t=>t.onclick=()=>{ goPage(t.dataset.go); if(t.dataset.src) octSrc(t.dataset.src); });

/* ---------- file picker ---------- */
let pickTarget=null,pickMode='file';
function pick(id,mode){ pickTarget=id; pickMode=mode; $('modal').style.display='flex';
  $('mselect').style.display=mode==='dir'?'inline-block':'none'; $('mhint').textContent=mode==='dir'?'Open a folder, then press “Select this folder”.':'Click a file to select it.';
  pickLoad(val(id)||CFG.data_dir||''); }
async function pickLoad(path){ try{ const d=await api('/api/ls?path='+encodeURIComponent(path)); $('mpath').value=d.path; $('mup').onclick=()=>pickLoad(d.parent);
   const L=$('mlist'); L.innerHTML=''; d.entries.forEach(e=>{ const div=document.createElement('div'); div.className='ent'; div.innerHTML='<span>'+(e.dir?'📁':'📄')+'</span><span></span><span class="sz">'+(e.dir?'':fmt(e.size))+'</span>'; div.children[1].textContent=e.name;
     div.onclick=()=>{ const full=d.path.replace(/\/$/,'')+'/'+e.name; if(e.dir) pickLoad(full); else if(pickMode==='file'){ $(pickTarget).value=full; $('modal').style.display='none'; pickDone(pickTarget);} };
     L.appendChild(div); }); }catch(e){ alert(e.message); } }
function fmt(n){ return n>1e6?(n/1e6).toFixed(1)+' MB':n>1e3?(n/1e3).toFixed(0)+' kB':n+' B'; }
$('mgo').onclick=()=>pickLoad(val('mpath')); $('mcancel').onclick=()=>$('modal').style.display='none';
$('mselect').onclick=()=>{ $(pickTarget).value=val('mpath'); $('modal').style.display='none'; pickDone(pickTarget); };
function pickDone(id){ if(id==='fu_img'||id==='fu_depth') fundusPreview(); }

/* ---------- A-scan canvas ---------- */
function drawProfile(cv,d){
  const dpr=window.devicePixelRatio||1, W=cv.clientWidth, H=cv.clientHeight; cv.width=W*dpr; cv.height=H*dpr; const c=cv.getContext('2d'); c.scale(dpr,dpr);
  c.clearRect(0,0,W,H); const L=62,R=18,T=18,B=44, x0=L,x1=W-R,y0=T,y1=H-B, ymin=-80,ymax=5;
  const X=z=>x0+(z/d.depth_um)*(x1-x0), Y=v=>y1-(v-ymin)/(ymax-ymin)*(y1-y0);
  c.font='12px system-ui'; c.fillStyle='#475569'; c.strokeStyle='#e2e8f0'; c.lineWidth=1;
  for(let v=-80;v<=0;v+=10){ const y=Y(v); c.beginPath(); c.moveTo(x0,y); c.lineTo(x1,y); c.stroke(); c.textAlign='right'; c.fillText(v,x0-8,y+4); }
  const step=d.depth_um>=1500?250:100; for(let z=0;z<=d.depth_um;z+=step){ const x=X(z); c.beginPath(); c.moveTo(x,y0); c.lineTo(x,y1); c.stroke(); c.textAlign='center'; c.fillText(z,x,y1+16); }
  c.textAlign='center'; c.font='13px system-ui'; c.fillStyle='#0f172a'; c.fillText('Depth (µm)',(x0+x1)/2,H-8);
  c.save(); c.translate(14,(y0+y1)/2); c.rotate(-Math.PI/2); c.fillText('Intensity (dB)',0,0); c.restore();
  // filled area + raw + smooth
  c.beginPath(); d.smooth.forEach((v,i)=>{ const x=X(d.z[i]),y=Y(v); i?c.lineTo(x,y):c.moveTo(x,y); }); c.lineTo(X(d.z[d.z.length-1]),y1); c.lineTo(X(d.z[0]),y1); c.closePath(); c.fillStyle='rgba(11,61,145,.09)'; c.fill();
  c.beginPath(); d.raw.forEach((v,i)=>{ const x=X(d.z[i]),y=Y(v); i?c.lineTo(x,y):c.moveTo(x,y); }); c.strokeStyle='rgba(120,150,200,.75)'; c.lineWidth=1; c.stroke();
  c.beginPath(); d.smooth.forEach((v,i)=>{ const x=X(d.z[i]),y=Y(v); i?c.lineTo(x,y):c.moveTo(x,y); }); c.strokeStyle='#0b3d91'; c.lineWidth=2.8; c.lineJoin='round'; c.stroke();
  d.peaks.forEach((p,j)=>{ const x=X(p.z),y=Y(p.db); c.beginPath(); c.arc(x,y,5.5,0,7); c.fillStyle='#e8590c'; c.fill(); c.strokeStyle='#fff'; c.lineWidth=2; c.stroke();
    const t=Math.round(p.z)+' µm · '+p.db.toFixed(1)+' dB'; c.font='bold 11.5px system-ui'; const tw=c.measureText(t).width+12, yy=Math.max(y-24-(j%2)*16,y0+2), xx=Math.min(Math.max(x-tw/2,x0),x1-tw);
    c.fillStyle='rgba(255,255,255,.95)'; c.strokeStyle='#cbd5e1'; c.lineWidth=1; c.fillRect(xx,yy,tw,18); c.strokeRect(xx,yy,tw,18); c.fillStyle='#111827'; c.textAlign='left'; c.fillText(t,xx+6,yy+13); });
  c.fillStyle='#0b3d91'; c.fillRect(x1-250,y0+6,16,3); c.font='11.5px system-ui'; c.fillStyle='#334155'; c.textAlign='left'; c.fillText('mean of '+(d.hi-d.lo)+' neighbouring A-scans',x1-229,y0+11);
  c.fillStyle='rgba(120,150,200,.9)'; c.fillRect(x1-250,y0+22,16,2); c.fillStyle='#334155'; c.fillText('A-scan '+d.idx+' (raw)',x1-229,y0+27);
}

/* ---------- OCT page ---------- */
let octMeta=null, octT=null;
function octSrc(s){ document.querySelectorAll('#octtabs button').forEach(b=>b.classList.toggle('on',b.dataset.src===s)); document.querySelectorAll('#page-oct .src').forEach(e=>e.classList.toggle('on',e.id==='src-'+s)); }
document.querySelectorAll('#octtabs button').forEach(b=>b.onclick=()=>octSrc(b.dataset.src));
async function octLoad(body){ await busy('Building scan…',async()=>{ const r=await api('/api/oct/load',body); octMeta=r.meta; $('oct-out').style.display='block';
  $('bscan-img').src=r.fig+'?t='+Date.now(); const m=r.meta;
  $('oct-meta').innerHTML='<b>Source</b><span>'+m.source+' — '+m.name+'</span><b>A-scans × depth points</b><span>'+m.n_a+' × '+m.n_z+'</span><b>Physical size (assumed)</b><span>'+m.width_mm+' mm × '+(m.depth_um/1000)+' mm</span>';
  const nb=$('oct-note'); nb.style.display=m.note?'block':'none'; nb.textContent=m.note;
  $('oct_slider').max=m.n_a-1; $('oct_idx').max=m.n_a-1; setIdx(Math.floor(m.n_a/2)); }); }
$('go-sim').onclick=()=>octLoad({source:'sim',seed:val('sim_seed')});
$('go-image').onclick=()=>octLoad({source:'image',path:val('im_path'),width_mm:val('im_w'),depth_mm:val('im_d'),floor_db:val('im_floor'),mask:val('im_mask'),crop:val('im_crop'),auto_crop:chk('im_auto'),invert:chk('im_inv'),show_info:chk('im_info'),inset:chk('im_inset'),force:chk('im_force')});
$('go-data').onclick=()=>octLoad({source:'data',path:val('da_path'),width_mm:val('da_w'),depth_mm:val('da_d'),linear:chk('da_lin'),amplitude:chk('da_amp')});
function setIdx(i){ if(!octMeta) return; i=Math.max(0,Math.min(octMeta.n_a-1,i)); $('oct_idx').value=i; $('oct_slider').value=i; markOct(); clearTimeout(octT); octT=setTimeout(octAscan,60); }
function markOct(){ const a=octMeta.ax, f=(+val('oct_idx')+.5)/octMeta.n_a, mk=$('bscan-marker'); mk.style.left=((a.l+f*(a.r-a.l))*100)+'%'; mk.style.top=(a.t*100)+'%'; mk.style.height=((a.b-a.t)*100)+'%';
  $('oct-pos').textContent='A-scan '+val('oct_idx')+' · x = '+(((+val('oct_idx')+.5)/octMeta.n_a)*octMeta.width_mm).toFixed(2)+' mm'; }
async function octAscan(){ if(!octMeta) return; try{ const d=await api('/api/oct/ascan?idx='+val('oct_idx')+'&avg='+val('oct_avg')); drawProfile($('oct-canvas'),d);}catch(e){log('ERROR: '+e.message,'e');} }
$('oct_slider').oninput=e=>setIdx(+e.target.value); $('oct_idx').onchange=e=>setIdx(+e.target.value); $('oct_avg').onchange=octAscan;
$('bscan-img').onclick=e=>{ if(!octMeta) return; const r=e.target.getBoundingClientRect(), fx=(e.clientX-r.left)/r.width, a=octMeta.ax, fr=(fx-a.l)/(a.r-a.l); if(fr<0||fr>1) return; setIdx(Math.floor(fr*octMeta.n_a)); };
window.addEventListener('resize',()=>{ if(octMeta) octAscan(); });
async function octExport(kind){ await busy('Exporting…',async()=>{ const r=await api('/api/oct/export',{kind,idx:+val('oct_idx'),avg:val('oct_avg'),scale:val('ex_scale'),interp:val('ex_interp'),marker:chk('ex_marker')}); log('Saved: '+r.path,'c'); showConsole(true); }); }
async function openOut(sub){ try{ const base=CFG.out_dir; await api('/api/open',{path: sub? base+'/'+sub : base}); }catch(e){ try{ await api('/api/open',{path:CFG.out_dir}); }catch(e2){ alert(e2.message);} } }

/* ---------- Volume page ---------- */
let vm=null, vSlice=0, vX=0, vT=null;
$('go-vol').onclick=()=>busy('Loading stack…',async()=>{ const r=await api('/api/volume/load',{folder:val('vo_folder'),width_mm:val('vo_w'),depth_mm:val('vo_d'),spacing_mm:val('vo_sp'),max_w:val('vo_maxw'),mask:val('vo_mask'),force:chk('vo_force')});
  vm=r; $('vol-out').style.display='block'; $('vol-meta').innerHTML='<b>Slices</b><span>'+r.n+' ('+r.first+' … '+r.last+')</span><b>Slice size</b><span>'+r.w+' × '+r.h+' px</span><b>Note</b><span>'+r.note+'</span>';
  $('vo_slider').max=r.n-1; $('vo_z0').value=Math.round(r.h*0.35); $('vo_z1').value=Math.round(r.h*0.7); vSlice=Math.floor(r.n/2); vX=Math.floor(r.w/2); volRefresh(true); });
function volRefresh(full){ if(!vm) return; const c=val('vo_cmap'),t=Date.now();
  $('vo_slider').value=vSlice; $('vo-bscan').src='/api/volume/slice?i='+vSlice+'&cmap='+c+'&t='+t;
  if(full||true) $('vo-cscan').src='/api/volume/cscan?z0='+val('vo_z0')+'&z1='+val('vo_z1')+'&mode='+val('vo_mode')+'&cmap='+c+'&t='+t;
  $('vo-pos').textContent='slice '+vSlice+' / '+(vm.n-1); volMarks(); clearTimeout(vT); vT=setTimeout(volAscan,80); }
function volMarks(){ const b=$('vo-bmark'); b.style.left=((vX+.5)/vm.w*100)+'%'; b.style.top='0'; b.style.height='100%';
  const c=$('vo-cmark'); c.style.left=((vX+.5)/vm.w*100)+'%'; c.style.top='0'; c.style.height='100%'; }
async function volAscan(){ try{ const d=await api('/api/volume/ascan?i='+vSlice+'&x='+vX+'&avg='+val('vo_avg')); $('vo-ascan-pos').textContent='slice '+vSlice+' · column '+vX+' · x = '+d.x_mm.toFixed(2)+' mm'; drawProfile($('vo-canvas'),d);}catch(e){} }
$('vo_slider').oninput=e=>{ vSlice=+e.target.value; volRefresh(); };
['vo_cmap','vo_z0','vo_z1','vo_mode','vo_avg'].forEach(i=>$(i).onchange=()=>volRefresh());
$('vo-bscan').onclick=e=>{ if(!vm) return; const r=e.target.getBoundingClientRect(); vX=Math.max(0,Math.min(vm.w-1,Math.floor((e.clientX-r.left)/r.width*vm.w))); volRefresh(); };
$('vo-cscan').onclick=e=>{ if(!vm) return; const r=e.target.getBoundingClientRect(); vX=Math.max(0,Math.min(vm.w-1,Math.floor((e.clientX-r.left)/r.width*vm.w))); vSlice=Math.max(0,Math.min(vm.n-1,Math.floor((e.clientY-r.top)/r.height*vm.n))); volRefresh(); };
$('go-nii').onclick=wrapJob('',()=>api('/api/volume/nifti',{open:true,iso:val('vo_iso')}),null);
$('go-nii2').onclick=()=>busy('Writing NIfTI…',async()=>{ const r=await api('/api/volume/nifti',{open:false}); log('Saved: '+r.path,'c'); showConsole(true); });

/* ---------- Fundus page ---------- */
function fundusPreview(){ const P=$('fu-prev'); P.innerHTML=''; ['fu_img','fu_depth'].forEach(id=>{ if(val(id)){ const im=document.createElement('img'); im.src='/api/preview?path='+encodeURIComponent(val(id))+'&t='+Date.now(); im.title=id; im.onerror=()=>im.remove(); P.appendChild(im);} }); }
$('go-depth').onclick=wrapJob('',()=>api('/api/fundus/depth',{image:val('fu_img')}),(st,s)=>{ if(st.rc===0){ $('fu_depth').value=s.depth; fundusPreview(); }});
function fuBody(inter){ return {image:val('fu_img'),depth:val('fu_depth'),mode:val('fu_mode'),relief:val('fu_relief'),step:val('fu_step'),sigma:val('fu_sigma'),channel:val('fu_ch'),invert:chk('fu_inv'),convex:chk('fu_conv'),interactive:inter}; }
$('go-fu').onclick=wrapJob('',()=>api('/api/fundus/mesh',fuBody(true)),null);
$('go-fu2').onclick=wrapJob('',()=>api('/api/fundus/mesh',fuBody(false)),null);

/* ---------- Atlas page ---------- */
function atBody(inter){ return {labels:val('at_lab'),mri:val('at_mri'),up:val('at_up'),names:val('at_names'),hide:val('at_hide'),interactive:inter}; }
$('go-at').onclick=wrapJob('',()=>api('/api/atlas/run',atBody(true)),null);
$('go-at2').onclick=wrapJob('',()=>api('/api/atlas/run',atBody(false)),null);

/* ---------- Inspector ---------- */
function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
$('go-in').onclick=()=>busy('Scanning folder…',async()=>{ const r=await api('/api/inspect',{folder:val('in_folder')}); let h='<div class="card"><b>'+r.total+' files</b><table style="margin-top:8px"><tr><th>Extension</th><th>Count</th></tr>'+r.ext.map(e=>'<tr><td>'+esc(e[0])+'</td><td>'+e[1]+'</td></tr>').join('')+'</table>';
  h+= r.volumetric_ext.length? '<div class="warn" style="background:#ecfdf5;border-color:#a7f3d0;color:#065f46">Volumetric formats present: '+esc(r.volumetric_ext.join(', '))+'</div>' : '<div class="note">No volumetric formats (.dcm .nii .mhd .nrrd …) found.</div>'; h+='</div>';
  h+='<div class="card"><b>Folders with most files</b><table style="margin-top:8px"><tr><th>Folder</th><th>Files</th></tr>'+r.dirs.map(d=>'<tr><td>'+esc(d[0])+'</td><td>'+d[1]+'</td></tr>').join('')+'</table></div>';
  h+='<div class="card"><b>Sample images</b><table style="margin-top:8px"><tr><th>File</th><th>Size</th><th>Mode</th><th>Looks like</th></tr>'+r.samples.map(s=>'<tr><td>'+esc(s.file)+'</td><td>'+s.size+'</td><td>'+s.mode+'</td><td>'+esc(s.kind)+'</td></tr>').join('')+'</table></div>';
  h+='<div class="card"><b>Is any folder a real slice stack?</b>'+(r.stacks.length? '<table style="margin-top:8px"><tr><th>Folder</th><th>Images</th><th>Neighbour corr.</th><th>Random corr.</th><th>Verdict</th></tr>'+r.stacks.map(s=>'<tr><td>'+esc(s.folder)+'</td><td>'+s.images+'</td><td>'+s.consecutive_corr+'</td><td>'+s.random_corr+'</td><td>'+esc(s.verdict)+'</td></tr>').join('')+'</table><div class="note">Heuristic on 32×32 thumbnails — confirm by eye.</div>' : '<div class="note">No folder with 8+ images.</div>')+'</div>';
  $('in-out').innerHTML=h; });

/* ---------- Settings ---------- */
const CFGLABELS={project_root:'Project folder',build_dir:'C++ build folder (eyeview2, img2mesh)',cmake_src:'C++ source folder (CMakeLists.txt)',depth_python:'Depth tool: python',depth_script:'Depth tool: depth.py',data_dir:'Data folder',out_dir:'Results folder',atlas_labels:'Atlas labels file',atlas_mri:'Atlas MRI file'};
async function loadSettings(){ CFG=await api('/api/config'); const b=$('cfgbox'); b.innerHTML=''; Object.keys(CFGLABELS).forEach(k=>{ const d=document.createElement('div'); d.className='row'; d.innerHTML='<label>'+CFGLABELS[k]+'<input type="text" class="path" id="cfg_'+k+'"></label>'; b.appendChild(d); $('cfg_'+k).value=CFG[k]; }); }
$('go-save').onclick=()=>busy('Saving…',async()=>{ const o={}; Object.keys(CFGLABELS).forEach(k=>o[k]=val('cfg_'+k)); CFG=await api('/api/config',o); applyDefaults(); log('Settings saved.','c'); });
$('go-env').onclick=()=>busy('Checking…',async()=>{ const r=await api('/api/env'); $('env-out').innerHTML='<table style="margin-top:10px"><tr><th></th><th>Item</th><th>Detail</th></tr>'+r.rows.map(x=>'<tr><td class="'+(x.ok?'ok':'no')+'">'+(x.ok?'✔':'✘')+'</td><td>'+esc(x.item)+'</td><td>'+esc(x.detail)+'</td></tr>').join('')+'</table>'; });
$('go-build').onclick=wrapJob('',()=>api('/api/build',{}),null);
function applyDefaults(){ if(!$('at_lab').value) $('at_lab').value=CFG.atlas_labels||''; if(!$('at_mri').value) $('at_mri').value=CFG.atlas_mri||''; if(!$('in_folder').value) $('in_folder').value=CFG.data_dir||''; }
(async()=>{ try{ CFG=await api('/api/config'); applyDefaults(); }catch(e){} })();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description="Eye Scan Studio")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    url = "http://%s:%d" % (a.host, a.port)
    print("Eye Scan Studio running at %s   (Ctrl+C to stop)" % url)
    print("Results are saved in: %s" % X(CFG["out_dir"]))
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
