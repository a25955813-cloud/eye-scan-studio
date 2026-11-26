#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_to_bscan.py : من صورة OCT (B-scan) أو مجلد صور  ->  A-scans  ->  B-scan عالي الدقة + رسم A-scan واضح

مهم: يعمل فقط مع صور OCT المقطعية (مثل مجموعة OCTID: NORMAL / DR / MH / CSR).
صور قاع العين (fundus) ملوّنة وسطحية ولا تحتوي عمقاً، فيرفضها البرنامج (إلا مع --force).

أمثلة:
  python image_to_bscan.py صورة.jpeg --mask 18,378,126,492 --no-info
  python image_to_bscan.py صورة.jpeg --scale 3 --ascan 400 --mark-ascan
  python image_to_bscan.py مجلد_الصور --out-dir نتائج --max 10

الدقة:
  --scale S      مضاعف دقة صورة الخرج (1 = 1280x853، 2 = 2560x1706، 3 = 3840x2559)  [الافتراضي 2]
  --interp M     طريقة العرض: lanczos (ناعمة، افتراضي) | bicubic | bilinear | nearest (بكسلات مربعة)
  --n-ascans N / --n-depth N   عدد الأعمدة ونقاط العمق. الافتراضي = أبعاد الصورة الأصلية (بلا أي تكبير مصطنع)
  --ascan-avg K  رسم A-scan بمتوسط K أعمدة متجاورة لخفض الضجيج الحبيبي (1 = بلا متوسط)  [الافتراضي 5]

المخرجات لكل صورة:
  <اسم>_bscan.png    B-scan بالألوان
  <اسم>_ascans.npy   مصفوفة A-scans بالـdB بشكل (عدد_A-scans, نقاط_العمق)
  <اسم>_ascan.png    رسم A-scan واحد بخط واضح مع تعليم القمم (طبقات النسيج)
"""
import argparse
import glob
import os
import re
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import oct_bscan as ob

plt.rcParams["font.family"] = "DejaVu Sans"
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def load_gray(path, crop=None, auto_crop=False, masks=None):
    """يقرأ الصورة كتدرج رمادي، ويعيد (المصفوفة، هل هي ملوّنة بشدة)"""
    im = Image.open(path)
    is_color = False
    if im.mode not in ('L', 'I', 'I;16', 'F'):
        rgb = np.asarray(im.convert('RGB')).astype(float)
        sat = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
        is_color = sat > 12.0           # صور OCT المنقولة رمادية؛ قاع العين حمراء/برتقالية
    g = np.asarray(im.convert('L')).astype(float)
    for (l, t, r, b) in (masks or []):
        # إخفاء الشعارات/المربعات (مثل مربع اتجاه المسح): نسخ رقعة مجاورة بنفس الحجم فوقها
        w = r - l
        src_l = r if r + w <= g.shape[1] else l - w
        g[t:b, l:r] = g[t:b, src_l:src_l + w]
    if crop:
        l, t, r, b = crop
        g = g[t:b, l:r]
    if auto_crop:
        rows = np.where(g.std(axis=1) > 3.0)[0]
        cols = np.where(g.std(axis=0) > 3.0)[0]
        if len(rows) > 10 and len(cols) > 10:
            g = g[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return g, is_color


def image_to_ascans(g, n_a, n_z, dyn_db, gamma=1.0, invert=False, auto_floor=True, floor_db=-74.0):
    """
    الصورة (ارتفاع x عرض) -> مصفوفة A-scans بالـdB بشكل (n_a, n_z).
    كل عمود في الصورة هو A-scan: القيم من الأعلى (السطح) إلى الأسفل (العمق).
    إن كان (n_a, n_z) يساوي أبعاد الصورة فلا تُجرى أي إعادة معاينة (لا تفاصيل مصطنعة).
    التحويل إلى dB افتراضي (الصور المنشورة سبق تحويلها للعرض): خلفية الصورة -> floor_db، والأبيض -> 0 dB.
    """
    if invert:
        g = 255.0 - g
    h, w = g.shape
    if (n_a, n_z) == (w, h):
        arr = g / 255.0
    else:
        im = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))
        im = im.resize((n_a, n_z), Image.LANCZOS)
        arr = np.asarray(im).astype(float) / 255.0
    arr = np.clip(arr, 0, 1) ** gamma
    if auto_floor:
        # خلفية الصورة (وسيط الأسطر العليا الخالية من النسيج) تُربط بأرضية الضجيج floor_db،
        # وما فوقها يمتد خطياً إلى 0 dB، وما تحتها ينزل إلى -80 dB
        bg = float(np.median(arr[: max(4, n_z // 8), :]))
        bg = min(max(bg, 0.02), 0.6)
        db = np.where(arr < bg, ob.DB_MIN + (floor_db - ob.DB_MIN) * arr / bg,
                      floor_db + (0.0 - floor_db) * (arr - bg) / (1.0 - bg))
    else:
        db = -dyn_db + dyn_db * arr
    return np.clip(db.T, ob.DB_MIN, ob.DB_MAX).copy()      # (n_a, n_z)


def ascan_profile(db, idx, depth_um, avg=5, n_peaks=6):
    """يحسب مسار A-scan الخام والمتوسط المنعَّم وقمم الطبقات (يُستخدم في الرسم وفي الواجهة)"""
    n_a, n_z = db.shape
    idx = int(np.clip(idx, 0, n_a - 1))
    z = (np.arange(n_z) + 0.5) * depth_um / n_z
    raw = db[idx]
    k = max(1, int(avg))
    lo, hi = max(0, idx - k // 2), min(n_a, idx - k // 2 + k)
    lin = (10.0 ** (db[lo:hi] / 10.0)).mean(axis=0)               # متوسط في المجال الخطي ثم dB
    mean_db = 10.0 * np.log10(lin + 1e-12)
    win = 9 if n_z >= 200 else 5
    smooth = savgol_filter(mean_db, win, 3) if n_z > win else mean_db
    cut = 2 if n_z > 50 else 0                    # أسطر الحافة الأخيرة في الصورة تكون عادة سوداء (حدّ الصورة)
    if cut:
        z, raw, smooth = z[:-cut], raw[:-cut], smooth[:-cut]
    dz = depth_um / n_z
    peaks, props = find_peaks(smooth, prominence=8.0, distance=max(3, int(25.0 / dz)))
    plist = []
    if len(peaks):
        top = peaks[np.argsort(props['prominences'])[::-1][:n_peaks]]
        plist = [(int(p), float(z[p]), float(smooth[p])) for p in sorted(top)]
    return dict(idx=idx, z=z, raw=raw, smooth=smooth, lo=lo, hi=hi, dz=dz, peaks=plist, n_z=n_z)


def plot_ascan(db, idx, width_mm, depth_um, out_png, avg=5, scale=2.0):
    """رسم A-scan عالي الوضوح: المسار الخام رفيع، والمتوسط الجانبي سميك، مع تعليم قمم الطبقات"""
    n_a, n_z = db.shape
    P = ascan_profile(db, idx, depth_um, avg)
    idx, z, raw, smooth, lo, hi, dz = P['idx'], P['z'], P['raw'], P['smooth'], P['lo'], P['hi'], P['dz']
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=100)
    ax.plot(z, raw, lw=0.9, color='#8aa4c8', alpha=0.75, label='A-scan %d (raw)' % idx, zorder=2)
    ax.plot(z, smooth, lw=2.6, color='#0b3d91', solid_capstyle='round',
            label='mean of %d neighbouring A-scans (smoothed)' % (hi - lo), zorder=4)
    ax.fill_between(z, ob.DB_MIN, smooth, color='#0b3d91', alpha=0.08, zorder=1)
    for j, (p, zp, dbp) in enumerate(P['peaks']):
        ax.plot(zp, dbp, 'o', ms=7, mfc='#e8590c', mec='white', mew=1.2, zorder=6)
        ax.annotate("%.0f \u00b5m\n%.1f dB" % (zp, dbp), xy=(zp, dbp),
                    xytext=(0, 14 + 10 * (j % 2)), textcoords='offset points', ha='center', fontsize=10,
                    fontweight='bold', color='#222222', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#cccccc', alpha=0.9))
    ax.set_xlim(0, depth_um)
    ax.set_ylim(ob.DB_MIN, 5)
    ax.set_xlabel("Depth (\u00b5m)", fontsize=15)
    ax.set_ylabel("Intensity (dB)", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(which='major', alpha=0.35, lw=0.8)
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.15, lw=0.5)
    x_mm = (idx + 0.5) * width_mm / n_a
    ax.set_title("A-Scan #%d   (x = %.2f mm,  %d depth points,  %.1f \u00b5m / point)"
                 % (idx, x_mm, n_z, dz), fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100 * scale)
    plt.close(fig)


def process(path, args, out_prefix):
    g, is_color = load_gray(path, args.crop, args.auto_crop, args.mask)
    if is_color and not args.force:
        print("تخطّي %s: الصورة ملوّنة وتبدو صورة قاع عين (fundus). لا عمق فيها فلا يمكن استخراج A-scans منها."
              " استخدم صور OCT المقطعية (مثل OCTID)، أو أضف --force لتجاوز هذا الفحص." % os.path.basename(path))
        return False
    h, w = g.shape
    n_a = args.n_ascans or w
    n_z = args.n_depth or h
    db = image_to_ascans(g, n_a, n_z, args.dyn, args.gamma, args.invert, not args.no_auto_floor, args.floor_db)
    np.save(out_prefix + "_ascans.npy", db)

    # ضبط محاور الرسم على الأبعاد الفيزيائية المفترضة
    ob.LAT_MM = float(args.width_mm)
    ob.DEP_UM = float(args.depth_mm) * 1000.0
    ob.N_A, ob.N_Z = n_a, n_z
    name = os.path.basename(path)
    info = ["Source: %s" % (name if len(name) <= 24 else name[:21] + "..."),
            "Image: %d x %d px" % (w, h),
            "A-Scans: %d x %d pts" % (n_a, n_z),
            "Assumed: %g x %g mm, floor %g dB" % (args.width_mm, args.depth_mm, args.floor_db)]
    if args.no_info:
        info = []
    title = "B-Scan from OCT image\n%d A-Scans \u00d7 %d Depth Points" % (n_a, n_z)
    ascan_idx = args.ascan if args.ascan is not None else n_a // 2
    marker = (ascan_idx + 0.5) * ob.LAT_MM / n_a if args.mark_ascan else None
    ob.render(db, out_prefix + "_bscan.png", labels=False, info_lines=info, inset=args.inset, title=title,
              scale=args.scale, interp=args.interp, marker_x=marker)
    plot_ascan(db, ascan_idx, args.width_mm, ob.DEP_UM, out_prefix + "_ascan.png", avg=args.ascan_avg, scale=args.scale)
    b = os.path.basename(out_prefix)
    print("%s  ->  %s_bscan.png | %s_ascans.npy | %s_ascan.png   (%d x %d, B-scan %dx%d px)"
          % (name, b, b, b, n_a, n_z, int(1280 * args.scale), int(853 * args.scale)))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='صورة OCT أو مجلد صور')
    ap.add_argument('--out', help='لصورة واحدة: مسار صورة الخرج (الافتراضي بجانب الصورة)')
    ap.add_argument('--out-dir', help='لمجلد: مجلد النتائج (الافتراضي results_bscan)')
    ap.add_argument('--max', type=int, default=0, help='للمجلد: أقصى عدد صور (0 = الكل)')
    ap.add_argument('--n-ascans', type=int, default=0, help='عدد A-scans (الافتراضي: عرض الصورة بالبكسل)')
    ap.add_argument('--n-depth', type=int, default=0, help='نقاط العمق (الافتراضي: ارتفاع الصورة بالبكسل)')
    ap.add_argument('--width-mm', type=float, default=6.0, help='العرض الجانبي الفيزيائي (افتراض)')
    ap.add_argument('--depth-mm', type=float, default=2.0, help='العمق الفيزيائي (افتراض)')
    ap.add_argument('--dyn', type=float, default=80.0, help='المدى الديناميكي مع --no-auto-floor')
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--no-auto-floor', action='store_true', help='تحويل خطي بسيط 0..255 -> (-dyn..0) dB')
    ap.add_argument('--floor-db', type=float, default=-74.0, help='مستوى أرضية الضجيج المفترض بالـdB')
    ap.add_argument('--scale', type=float, default=2.0, help='مضاعف دقة صورة الخرج (1 إلى 4)')
    ap.add_argument('--interp', default='lanczos', choices=['lanczos', 'bicubic', 'bilinear', 'nearest'])
    ap.add_argument('--ascan-avg', type=int, default=5, help='متوسط أعمدة متجاورة في رسم A-scan (1 = بلا)')
    ap.add_argument('--mark-ascan', action='store_true', help='خط متقطع يبيّن موضع الـA-scan على الـB-scan')
    ap.add_argument('--inset', action='store_true', help='رسم سهمي Depth/Lateral أسفل اليسار')
    ap.add_argument('--no-info', action='store_true', help='إخفاء صندوق المعلومات')
    ap.add_argument('--invert', action='store_true', help='اعكس السطوع (للصور بخلفية بيضاء)')
    ap.add_argument('--crop', type=lambda s: tuple(int(v) for v in s.split(',')), help='يسار,أعلى,يمين,أسفل')
    ap.add_argument('--mask', action='append', type=lambda s: tuple(int(v) for v in s.split(',')),
                    help='يسار,أعلى,يمين,أسفل : إخفاء منطقة (مثل مربع اتجاه المسح)؛ يمكن تكرارها')
    ap.add_argument('--auto-crop', action='store_true', help='قص الحواف الثابتة تلقائياً')
    ap.add_argument('--ascan', type=int, help='رقم الـA-scan المراد رسمه')
    ap.add_argument('--force', action='store_true', help='تجاهل فحص الصور الملوّنة')
    args = ap.parse_args()
    args.scale = float(np.clip(args.scale, 1.0, 4.0))

    if os.path.isdir(args.input):
        files = sorted([f for f in glob.glob(os.path.join(args.input, '*')) if f.lower().endswith(EXTS)],
                       key=natural_key)
        if args.max:
            files = files[:args.max]
        if not files:
            print("لا توجد صور في المجلد"); return 1
        out_dir = args.out_dir or 'results_bscan'
        os.makedirs(out_dir, exist_ok=True)
        ok = 0
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            ok += bool(process(f, args, os.path.join(out_dir, stem)))
        print("تمت معالجة %d من %d صورة. النتائج في: %s" % (ok, len(files), out_dir))
        return 0 if ok else 2
    if not os.path.isfile(args.input):
        print("المسار غير موجود: %s" % args.input); return 1
    if args.out:
        prefix = args.out[:-4] if args.out.lower().endswith('.png') else args.out
        if prefix.endswith('_bscan'):
            prefix = prefix[:-6]
    else:
        prefix = os.path.splitext(args.input)[0]
    return 0 if process(args.input, args, prefix) else 2


if __name__ == '__main__':
    sys.exit(main())
