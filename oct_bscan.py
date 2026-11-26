#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oct_bscan.py : يولّد/يرسم صورة B-Scan لشبكية العين (OCT) بنفس تخطيط الصورة المرجعية.

الأوضاع:
  python oct_bscan.py                         محاكاة + رسم  -> bscan_out.png (1280x853)
  python oct_bscan.py --seed 7                نفس الهندسة بضجيج (speckle) مختلف
  python oct_bscan.py --input data.npy        رسم بيانات حقيقية (مثلاً من OptiSystem)
  python oct_bscan.py --save-data bscan.npy   حفظ مصفوفة dB (512 A-scan x 1024 عمق)
  python oct_bscan.py --compare ref.jpg       مقارنة رقمية مع الصورة المرجعية

الأبعاد: 512 A-scan عبر 5 مم، و1024 نقطة عمق عبر 2 مم (بلا تمديد محوري).
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = True
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------------
# ثوابت الهندسة والرسم (مقاسة من الصورة المرجعية بالبكسل)
# ----------------------------------------------------------------------------
N_A, N_Z = 512, 1024            # عدد A-scans ونقاط العمق
LAT_MM, DEP_UM = 5.0, 2000.0    # المدى الجانبي (مم) والعمق (ميكرومتر)
DB_MIN, DB_MAX = -80.0, 0.0
FIG_W, FIG_H, DPI = 1280, 853, 100
AX_PX = (106, 102, 1127, 713)   # يسار، أعلى، يمين، أسفل (بكسل) لمنطقة الصورة
CB_PX = (1162, 102, 1192, 705)  # الشريط اللوني

# خريطة الألوان المستخرجة من الشريط اللوني في الصورة المرجعية (كل 1 dB من 0 إلى -80)
LUT = [
    (252, 252, 249),
    (250, 228, 223),
    (247, 200, 189),
    (248, 171, 155),
    (241, 142, 120),
    (237, 109, 79),
    (231, 81, 47),
    (228, 52, 11),
    (231, 52, 3),
    (232, 64, 1),
    (208, 64, 3),
    (240, 91, 2),
    (242, 104, 1),
    (246, 116, 1),
    (251, 130, 2),
    (253, 143, 2),
    (254, 154, 1),
    (254, 167, 1),
    (254, 180, 2),
    (253, 192, 3),
    (227, 180, 10),
    (254, 217, 2),
    (253, 231, 1),
    (254, 241, 1),
    (252, 253, 4),
    (229, 246, 12),
    (208, 241, 28),
    (187, 233, 37),
    (162, 226, 52),
    (137, 218, 66),
    (90, 175, 58),
    (94, 202, 88),
    (71, 195, 100),
    (49, 189, 113),
    (34, 184, 124),
    (30, 186, 136),
    (27, 188, 148),
    (23, 190, 162),
    (19, 192, 172),
    (15, 193, 186),
    (13, 172, 176),
    (8, 196, 210),
    (3, 198, 217),
    (2, 199, 231),
    (2, 185, 235),
    (1, 170, 237),
    (1, 156, 239),
    (1, 141, 240),
    (1, 127, 243),
    (2, 114, 245),
    (7, 91, 232),
    (1, 84, 251),
    (1, 69, 252),
    (2, 57, 250),
    (2, 52, 238),
    (0, 45, 231),
    (2, 39, 220),
    (1, 32, 211),
    (1, 25, 199),
    (1, 19, 189),
    (2, 14, 169),
    (2, 7, 171),
    (1, 1, 160),
    (1, 1, 147),
    (1, 1, 133),
    (0, 0, 120),
    (1, 0, 107),
    (0, 1, 93),
    (1, 0, 81),
    (1, 0, 68),
    (0, 0, 49),
    (0, 0, 39),
    (0, 1, 32),
    (0, 0, 28),
    (0, 1, 22),
    (1, 0, 19),
    (2, 0, 14),
    (0, 0, 11),
    (1, 0, 6),
    (0, 0, 3),
    (0, 0, 1)
]

# ----------------------------------------------------------------------------
# نموذج الطبقات (معايَر على الصورة المرجعية)
# ----------------------------------------------------------------------------
KX = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])   # مم؛ الشكل متناظر حول 2.5 مم
# حدود الطبقات (ميكرومتر) عند KX:
# ILM، NFL|GCL، GCL|IPL، IPL|OPL، OPL|ONL، ONL|ELM، IS/OS|RPE، RPE|Choroid، أسفل المشيمية
KNOT_B = np.array([
    [298, 322, 343, 380, 425, 451],
    [402, 425, 441, 459, 473, 479],
    [503, 520, 537, 534, 520, 506],
    [605, 618, 628, 604, 574, 538],
    [689, 700, 708, 690, 639, 601],
    [847, 862, 877, 885, 889, 887],
    [988, 1005, 1021, 1028, 1033, 1040],
    [1026, 1042, 1057, 1067, 1074, 1080],
    [1210, 1231, 1251, 1269, 1283, 1290],
], dtype=float)
MEAN_DB = dict(nfl=-49.0, gcl=-63.3, ipl=-53.5, opl=-27.8, onl=-56.9, isos=-29.6,
               rpe=-1.6, ch_top=-45.6, ch_bot=-59.7)
MEAN_DB_CENTER = dict(nfl=-31.4, gcl=-56.5, ipl=-36.1)    # القيم عند الحفرة المركزية
ILM_LEVEL, ILM_T = -37.2, 24.0
ELM_LEVEL, ELM_T = -41.9, 8.9
EDGE_W = np.array([8.1, 16.7, 10.2, 6.0, 6.6, 3.6, 7.9, 10.6, 13.7])
NOISE_ABOVE, NOISE_BELOW = -73.9, -74.6

# ضجيج البقع (speckle) والقطع السفلي
SPECKLE_SIGMA_Z = 2.4       # ارتباط محوري (بعدد العينات)
SPECKLE_SIGMA_X = 0.8       # ارتباط جانبي (بعدد A-scans)
SPARK_P, SPARK_A0, SPARK_A1 = 0.09, 9.0, 10.0   # احتمال ومقدار النقاط اللامعة (dB)
SPARK_BIAS = 0.55                 # نسبة تعويض متوسط النقاط اللامعة (توازن المتوسط والوسيط)
MID_BUMP = 0.45                  # معامل تباين إضافي للمستويات المتوسطة
TISSUE_STD_DB = 5.57            # الانحراف المعياري النظري للـspeckle بمقياس dB
FLOOR_STD_DB = 4.4              # انحراف أرضية الضجيج المقيس
ROLLOFF_Z0, ROLLOFF_SPAN, ROLLOFF_DEPTH = 1390.0, 50.0, 10.0   # الهبوط إلى -80 dB (أسود)


def _sig(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def x_axis_mm():
    return (np.arange(N_A) + 0.5) * LAT_MM / N_A


def z_axis_um():
    return (np.arange(N_Z) + 0.5) * DEP_UM / N_Z


def layer_boundaries(x_mm):
    u = 2.5 - np.abs(x_mm - 2.5)
    return np.stack([PchipInterpolator(KX, KNOT_B[k])(u) for k in range(KNOT_B.shape[0])])


def mean_map(x_mm, z_um):
    """متوسط شدة الإشارة بالـdB لكل نقطة (شكل: عمق x A-scan)"""
    B = layer_boundaries(x_mm)
    B0 = B[0]
    f = (B0 - B0[0]) / (B0.max() - B0[0])            # 0 عند الأطراف، 1 عند الحفرة
    z = z_um[:, None]
    m, mc = MEAN_DB, MEAN_DB_CENTER

    def lvl(name):
        return m[name] + (mc[name] - m[name]) * f if name in mc else m[name] + 0 * f

    levels = [lvl('nfl'), lvl('gcl'), lvl('ipl'), lvl('opl'), lvl('onl'), lvl('isos'), lvl('rpe')]
    M = np.full((len(z_um), len(x_mm)), NOISE_ABOVE)
    prev = NOISE_ABOVE + 0 * f
    for k in range(7):
        M = M + (levels[k] - prev) * _sig((z - B[k]) / EDGE_W[k])
        prev = levels[k]
    span = np.clip((z - B[7]) / (B[8] - B[7] + 1e-9), 0, 1)
    ch = m['ch_top'] + (m['ch_bot'] - m['ch_top']) * span
    M = M + (ch - prev) * _sig((z - B[7]) / EDGE_W[7])
    M = M + (NOISE_BELOW - ch) * _sig((z - B[8]) / EDGE_W[8])
    a = _sig((z - B[0]) / 2.5) - _sig((z - B[0] - ILM_T) / 2.5)      # خط ILM الساطع
    M = M * (1 - a) + ILM_LEVEL * a
    a = _sig((z - B[5]) / 3.0) - _sig((z - B[5] - ELM_T) / 3.0)      # خط ELM
    M = M * (1 - a) + ELM_LEVEL * a
    return M


def log_speckle(rng):
    """بقع Rayleigh مترابطة، تُعاد بالـdB بمتوسط صفر وانحراف 5.57 dB"""
    g = rng.standard_normal((2, N_Z, N_A))
    g = np.stack([gaussian_filter(g[i], (SPECKLE_SIGMA_Z, SPECKLE_SIGMA_X), mode='wrap') for i in range(2)])
    inten = g[0] ** 2 + g[1] ** 2
    inten /= inten.mean()
    return 10.0 * np.log10(inten + 1e-12) + 2.507      # +2.507: متوسط log لتوزيع أسّي


def simulate(seed=1):
    """يعيد مصفوفة dB بشكل (N_A, N_Z)"""
    rng = np.random.default_rng(seed)
    x, z = x_axis_mm(), z_axis_um()
    M = mean_map(x, z)
    L = log_speckle(rng)
    tissue = _sig((M + 68.0) / 2.0)
    scale = FLOOR_STD_DB / TISSUE_STD_DB + (1 - FLOOR_STD_DB / TISSUE_STD_DB) * tissue
    scale = scale * (1.0 + MID_BUMP * np.exp(-((M + 55.5) / 4.5) ** 2))   # تباين أكبر في الطبقات المتوسطة (ONL/المشيمية)
    # نقاط لامعة منفردة (sparkles) تظهر في الصورة المرجعية داخل الأنسجة، ويُطرح متوسطها كي لا ينحاز المتوسط
    p_loc = SPARK_P * _sig((M + 66.0) / 2.0)
    spark = (rng.random((N_Z, N_A)) < p_loc) * (SPARK_A0 + rng.exponential(SPARK_A1, (N_Z, N_A)))
    db = M - SPARK_BIAS * p_loc * (SPARK_A0 + SPARK_A1) + scale * L + spark
    # السطر الساطع قرب السطح (0-14 ميكرومتر) مع نقاط لامعة
    top = z[:, None] < 14.0
    spikes = (rng.random((N_Z, N_A)) < 0.10) * rng.uniform(20, 34, (N_Z, N_A))
    top_db = -36.0 + 4.0 * rng.standard_normal((N_Z, N_A)) + spikes
    # ذيل الشدة تحت السطح مع خطوط عمودية (streaks) مترابطة بين الـA-scans
    g = gaussian_filter(rng.standard_normal(N_A), 1.4, mode='wrap')
    streak = np.exp(0.5 * g / g.std() - 0.125)
    tail = 19.0 * np.exp(-(np.clip(z - 14.0, 0, None)) / 9.0) + 3.0 * np.exp(-z / 70.0)
    tail = tail[:, None] * streak[None, :]
    near = (z[:, None] >= 14.0) & (z[:, None] < 200.0)
    db = np.where(near, db + tail, db)
    db = np.where(top, top_db, db)
    # هبوط الحساسية في العمق ثم القص عند -80 dB (المنطقة السوداء)
    roll = -ROLLOFF_DEPTH * np.clip((z - ROLLOFF_Z0) / ROLLOFF_SPAN, 0, 1)
    db = db + roll[:, None]
    db[z >= ROLLOFF_Z0 + ROLLOFF_SPAN + 10.0, :] = DB_MIN          # أسود تماماً بعد نهاية نافذة الإشارة
    db = np.clip(db, DB_MIN, DB_MAX)
    return db.T.copy()


# ----------------------------------------------------------------------------
# تحميل بيانات حقيقية (مثلاً تصدير OptiSystem)
# ----------------------------------------------------------------------------
def load_data(path, linear=False, amplitude=False):
    if path.lower().endswith('.npy'):
        a = np.load(path)
    else:
        a = np.loadtxt(path, delimiter=',' if path.lower().endswith('.csv') else None)
    a = np.asarray(a, dtype=float)
    if a.shape == (N_Z, N_A):
        a = a.T
    if a.shape != (N_A, N_Z):
        raise ValueError("الشكل المتوقع (512, 1024) أو (1024, 512) لكن الشكل الحالي %s" % (a.shape,))
    if linear:
        a = np.maximum(a, 1e-30)
        a = (20.0 if amplitude else 10.0) * np.log10(a / a.max())
    return np.clip(a, DB_MIN, DB_MAX)


# ----------------------------------------------------------------------------
# الرسم
# ----------------------------------------------------------------------------
def _fx(px):
    return px / FIG_W


def _fy(py):
    return 1.0 - py / FIG_H


LABELS = [  # الاسم، العمق (ميكرومتر) الذي يشير إليه السهم
    ("ILM", 311), ("NFL", 363), ("GCL", 461), ("IPL", 566), ("OPL", 651),
    ("ONL", 775), ("ELM", 858), ("IS/OS", 926), ("RPE", 1008), ("Choroid", 1130),
]


def render(db, out_png, labels=True, info_lines=None, inset=True, title=None, scale=1.0, interp='nearest', marker_x=None):
    cmap = LinearSegmentedColormap.from_list("oct", [(r / 255, g / 255, b / 255) for r, g, b in LUT[::-1]], N=512)
    fig = plt.figure(figsize=(FIG_W / DPI, FIG_H / DPI), dpi=DPI)
    l, t, r, b = AX_PX
    ax = fig.add_axes([_fx(l), _fy(b), _fx(r) - _fx(l), _fy(t) - _fy(b)])
    kw = {} if interp == 'nearest' else dict(interpolation_stage='data')
    im = ax.imshow(db.T, extent=[0, LAT_MM, DEP_UM, 0], cmap=cmap, vmin=DB_MIN, vmax=DB_MAX,
                   aspect='auto', interpolation=interp, **kw)
    if marker_x is not None:
        ax.axvline(marker_x, color='white', lw=1.0, ls=(0, (5, 4)), alpha=0.9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(0, LAT_MM + 1e-6, 1.0))
    ax.set_yticks(np.arange(0, DEP_UM + 1e-6, 250))
    ax.tick_params(length=0, labelsize=TICK_FS, pad=3)
    ax.set_xlabel("A-Scan Position (Lateral Distance, mm)", fontsize=LABEL_FS, labelpad=XLABEL_PAD)
    ax.set_ylabel("Depth (\u00b5m)", fontsize=LABEL_FS, labelpad=YLABEL_PAD)
    ax.set_title(title or "B-Scan (Retina OCT)\n512 OptiSystem A-Scans \u00d7 1024 Depth Points",
                 fontsize=TITLE_FS, fontweight='bold', pad=TITLE_PAD)

    # الشريط اللوني
    cl, ct, cr, cb_ = CB_PX
    cax = fig.add_axes([_fx(cl), _fy(cb_), _fx(cr) - _fx(cl), _fy(ct) - _fy(cb_)])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks(list(range(0, -81, -10)))
    cb.ax.tick_params(labelsize=TICK_FS, direction='in', length=6)
    cb.set_label("Intensity (dB)", fontsize=CB_LABEL_FS, labelpad=CB_LABEL_PAD)
    cb.outline.set_linewidth(0.8)

    # أسماء الطبقات مع الأسهم
    for name, depth in (LABELS if labels else []):
        ax.annotate(name, xy=(4.472, depth), xytext=(4.69, depth), xycoords='data', textcoords='data',
                    color='white', fontsize=LAYER_FS, fontweight='bold', va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.28', fc=(0, 0, 0, 0.78), ec=(0.55, 0.55, 0.55, 0.9), lw=0.8),
                    arrowprops=dict(arrowstyle='-|>', color='white', lw=2.0, mutation_scale=13,
                                    shrinkA=0, shrinkB=0, connectionstyle='arc3'),
                    annotation_clip=False, zorder=5)

    # محاور الإرشاد أسفل اليسار (مواضعها كنسب من المدى كي تصلح لأي أبعاد)
    X, Z = LAT_MM, DEP_UM
    if inset:
        ap = dict(arrowstyle='-|>', color='white', lw=1.2, mutation_scale=8, shrinkA=0, shrinkB=0)
        ax.annotate('', xy=(0.04 * X, 0.823 * Z), xytext=(0.04 * X, 0.941 * Z), arrowprops=ap)
        ax.annotate('', xy=(0.156 * X, 0.941 * Z), xytext=(0.04 * X, 0.941 * Z), arrowprops=ap)
        ax.text(0.061 * X, 0.878 * Z, "Depth (z)", color='white', fontsize=INSET_FS, va='center', ha='left')
        ax.text(0.0628 * X, 0.964 * Z, "Lateral (x)", color='white', fontsize=INSET_FS, va='center', ha='left')

    # صندوق المعلومات أسفل اليمين
    if info_lines is None:
        info_lines = ["Depth Range: 0\u20132.0 mm", "A-Scans: 512", "Fixed global display", "No axial stretching"]
    if info_lines:
        ax.text(0.9814 * X, 0.905 * Z, "\n".join(info_lines),
                color='white', fontsize=INFO_FS, ha='right', va='center', multialignment='right', linespacing=1.13,
                bbox=dict(boxstyle='square,pad=0.65', fc='black', ec='white', lw=1.0))
    fig.savefig(out_png, dpi=DPI * scale)
    plt.close(fig)


# أحجام الخطوط والإزاحات (معايَرة بالمقارنة مع الصورة المرجعية)
TICK_FS, LABEL_FS, TITLE_FS, LAYER_FS, INSET_FS, INFO_FS, CB_LABEL_FS = 14, 16.85, 23.4, 12.5, 10.6, 11.1, 15.5
TITLE_PAD, XLABEL_PAD, YLABEL_PAD, CB_LABEL_PAD = 21.1, 9.4, 10.0, 8


# ----------------------------------------------------------------------------
# مقارنة رقمية مع الصورة المرجعية
# ----------------------------------------------------------------------------
def _lut_fine():
    grid = np.arange(0, -80.5, -1.0)
    lut = np.array(LUT, dtype=np.float32)
    fine = np.arange(0, -80.01, -0.25)
    return fine, np.stack([np.interp(-fine, -grid, lut[:, c]) for c in range(3)], axis=1).astype(np.float32)


def digitize(img_path):
    from PIL import Image
    from scipy.ndimage import uniform_filter
    a = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
    l, t, r, b = AX_PX
    P = a[t:b, l:r, :]
    fine, lutf = _lut_fine()
    H, W, _ = P.shape
    D = np.zeros((H, W), np.float32)
    for i in range(0, H, 16):
        blk = P[i:i + 16].reshape(-1, 1, 3)
        d = ((blk - lutf[None, :, :]) ** 2).sum(axis=2)
        D[i:i + 16] = fine[d.argmin(axis=1)].reshape(-1, W)
    return D


def compare(ref_path, gen_path):
    from PIL import Image
    from scipy.ndimage import uniform_filter
    # الصورة المرجعية بصيغة JPEG، فنمرّر الصورة المولَّدة بنفس الضغط لمقارنة عادلة
    tmp = gen_path.rsplit('.', 1)[0] + '_jpegtest.jpg'
    Image.open(gen_path).convert('RGB').save(tmp, quality=92)
    A = digitize(ref_path)
    B = digitize(tmp)
    H, W = A.shape
    upp, mpp = DEP_UM / H, LAT_MM / W
    z = (np.arange(H) + 0.5) * upp
    mask = np.ones((H, W), bool)
    mask[int(250 / upp):int(1300 / upp), int(4.3 / mpp):] = False      # أسماء الطبقات
    mask[z < 45, :] = False
    mask[z > 1380, :] = False
    SA, SB = uniform_filter(A, size=(9, 21)), uniform_filter(B, size=(9, 21))
    rmse = float(np.sqrt((((SA - SB)[mask]) ** 2).mean()))
    print("1) هيكل الطبقات (مصفوفة مُنعَّمة 9x21 بكسل، dB):")
    print("   RMSE = %.2f dB   |   الانحياز = %+.2f dB" % (rmse, float((SB - SA)[mask].mean())))

    # إزاحة الحدود: ارتباط متقاطع بين مقطعين رأسيين (مرجع/مولَّد) لكل مجموعة أعمدة
    shifts = []
    zlo, zhi = int(250 / upp), int(1350 / upp)
    maxs = int(40 / upp)
    for x0 in np.arange(0.1, 4.3, 0.2):
        c0, c1 = int(x0 / mpp), int((x0 + 0.2) / mpp)
        pa = A[:, c0:c1].mean(axis=1)
        pb = B[:, c0:c1].mean(axis=1)
        pa = uniform_filter(pa, 5)
        pb = uniform_filter(pb, 5)
        best, bs = 1e18, 0
        for sh in range(-maxs, maxs + 1):
            d = pa[zlo:zhi] - np.roll(pb, sh)[zlo:zhi]
            e = (d ** 2).mean()
            if e < best:
                best, bs = e, sh
        shifts.append(abs(bs) * upp)
    shifts = np.array(shifts)
    print("2) إزاحة مواضع الطبقات بين المرجع والمولَّد (%d مقطعاً): متوسط = %.1f µm ، أقصى = %.1f µm ، (بكسل الصورة = %.1f µm)"
          % (len(shifts), shifts.mean(), shifts.max(), upp))

    regs = {'ضجيج فوق الشبكية': (0.3, 1.2, 120, 250), 'NFL (طرف)': (0.1, 0.7, 330, 380),
            'GCL (طرف)': (0.1, 0.7, 430, 500), 'ONL (طرف)': (0.1, 0.9, 740, 830),
            'IS/OS (طرف)': (0.1, 0.9, 890, 970), 'مشيمية': (0.1, 0.9, 1110, 1190),
            'أسفل المشيمية': (0.3, 1.2, 1330, 1390)}
    print("3) المتوسط والانحراف الصلب (مرجع | مولَّد):")
    for k, (x0, x1, z0, z1) in regs.items():
        ra = A[int(z0 / upp):int(z1 / upp), int(x0 / mpp):int(x1 / mpp)].ravel()
        rb = B[int(z0 / upp):int(z1 / upp), int(x0 / mpp):int(x1 / mpp)].ravel()
        sa = 1.4826 * np.median(np.abs(ra - np.median(ra)))
        sb = 1.4826 * np.median(np.abs(rb - np.median(rb)))
        print("   %-18s %6.1f ± %4.1f  |  %6.1f ± %4.1f" % (k, np.median(ra), sa, np.median(rb), sb))
    ra = np.array(Image.open(ref_path).convert('L')).astype(float)
    rb = np.array(Image.open(tmp).convert('L')).astype(float)
    m = np.ones(ra.shape, bool)
    l, t, r, b = AX_PX
    m[t:b + 1, l:r + 1] = False
    m[CB_PX[1]:CB_PX[3] + 1, CB_PX[0]:CB_PX[2] + 1] = False
    print("4) العناصر الثابتة خارج الصورة (عنوان، محاور، ألوان): متوسط الفرق المطلق = %.2f من 255" % np.abs(ra - rb)[m].mean())
    import os
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='bscan_out.png')
    ap.add_argument('--input', help='ملف بيانات حقيقي (.npy أو .csv أو .txt) بشكل 512x1024')
    ap.add_argument('--linear', action='store_true', help='البيانات خطية وليست dB')
    ap.add_argument('--amplitude', action='store_true', help='مع --linear: سعة (20log) بدل قدرة (10log)')
    ap.add_argument('--save-data', help='حفظ مصفوفة dB الناتجة (.npy)')
    ap.add_argument('--compare', help='صورة مرجعية للمقارنة الرقمية')
    a = ap.parse_args()
    db = load_data(a.input, a.linear, a.amplitude) if a.input else simulate(a.seed)
    if a.save_data:
        np.save(a.save_data, db)
    render(db, a.out)
    print("saved", a.out, "| data shape", db.shape)
    if a.compare:
        compare(a.compare, a.out)


if __name__ == '__main__':
    main()
