#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize summary.json produced by eval_change_caption.py.

Output
  1) change_caption_radar.png      5-axis radar chart (Accuracy vs Coverage)
  2) change_caption_dashboard.png  Dashboard: radar + grouped bars +
                                   sample-count/accuracy dual-axis + confusion matrix

Usage
  python plot_summary.py --summary summary.json --outdir ./
  python plot_summary.py --summary summary.json --font /path/to/times.ttf

Font
  All text is English (Times New Roman has no CJK glyphs).
  Font resolution order:
    1) --font / -F argument
    2) common Times New Roman locations on Linux / macOS / Windows
    3) Times-metric-compatible serif fallbacks (Tinos, Liberation Serif, DejaVu Serif)
    4) matplotlib default
  Whichever is found is reported on stdout, so you can tell whether the
  real Times New Roman was used or a fallback.

Requires matplotlib + numpy.
"""

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patheffects

RULES = ["Type", "Amount", "Position", "Background", "Density"]

# --------------------------------------------------------------------------- #
#  Font: prefer Times New Roman, fall back to metric-compatible serif
# --------------------------------------------------------------------------- #

DEFAULT_FONT_CANDIDATES = [
    # Linux (user-specified layout)
    "/usr/share/fonts/truetype/timesnewroman/times.ttf",
    "/usr/share/fonts/truetype/timesnewroman/Times-New-Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
    # Linux: fontconfig query (catches any install location)
    "fc-match:Times New Roman",
    "fc-match:TimesNewRoman",
    # macOS
    "/Library/Fonts/Times New Roman.ttf",
    "/System/Library/Fonts/Times.ttc",
    # Windows
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesbd.ttf",
    # Metric-compatible serif fallbacks (Times substitute)
    "/usr/share/fonts/truetype/tinos/Tinos-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "fc-match:Tinos",
    "fc-match:Liberation Serif",
    "fc-match:DejaVu Serif",
]

SERIF_FALLBACK_NAMES = ["Times New Roman", "Tinos", "Liberation Serif",
                        "DejaVu Serif", "Nimbus Roman", "STIX Two Text", "serif"]


def _resolve_fc_match(spec):
    """spec like 'fc-match:Times New Roman' -> resolved .ttf path or None"""
    if not spec.startswith("fc-match:"):
        return None
    query = spec.split(":", 1)[1]
    exe = shutil.which("fc-match")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "-f", "%{file}", query],
                             capture_output=True, text=True, timeout=10)
        path = out.stdout.strip().splitlines()[0].strip() if out.stdout else ""
        if path.lower().endswith((".ttf", ".otf", ".ttc")) and os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def setup_font(user_font=None):
    """Register the best available Times-like font. Returns a description."""
    candidates = ([user_font] if user_font else []) + DEFAULT_FONT_CANDIDATES
    chosen_path, chosen_name = None, None

    for spec in candidates:
        if not spec:
            continue
        path = _resolve_fc_match(spec) if spec.startswith("fc-match:") else spec
        if path and os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                name = font_manager.FontProperties(fname=path).get_name()
                if name:
                    chosen_path, chosen_name = path, name
                    break
            except Exception:
                continue

    if chosen_name:
        matplotlib.rcParams["font.family"] = chosen_name
        matplotlib.rcParams["font.serif"] = [chosen_name] + SERIF_FALLBACK_NAMES
        matplotlib.rcParams["font.sans-serif"] = [chosen_name] + SERIF_FALLBACK_NAMES
        # math text should visually match the serif body font
        matplotlib.rcParams["mathtext.fontset"] = "custom"
        matplotlib.rcParams["mathtext.rm"] = chosen_name
        matplotlib.rcParams["mathtext.it"] = chosen_name + ":italic"
        note = "%s  <-  %s" % (chosen_name, chosen_path)
        if "times" not in chosen_name.lower() and "tinos" not in chosen_name.lower():
            note += "   [fallback, not Times New Roman]"
    else:
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = SERIF_FALLBACK_NAMES
        note = "matplotlib default serif  [no Times New Roman found]"

    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 11
    return note


# --------------------------------------------------------------------------- #
#  Palette
# --------------------------------------------------------------------------- #

C_ACC = "#2E6FBA"      # accuracy - blue
C_COV = "#E8833A"      # coverage - orange
C_STRICT = "#7FA9D9"   # strict accuracy - light blue
C_BAR = "#8FA8C8"      # sample-count bars
C_LINE = "#C0392B"     # accuracy line
C_GRID = "#D5DCE5"
C_TEXT_DIM = "#7A7A7A"


def pct(x):
    return 0.0 if x is None else x * 100


# --------------------------------------------------------------------------- #
#  Panels
# --------------------------------------------------------------------------- #

TH_OFFSET = 0.5 * np.pi     # theta = 0 指向正上方
TH_DIR = -1.0               # 辐条顺时针排布
HALO = [patheffects.withStroke(linewidth=2.6, foreground="white")]


def _spoke_dir(theta):
    """辐条在屏幕坐标下的单位方向向量（已计入 theta offset / direction）"""
    phi = TH_OFFSET + TH_DIR * theta
    return np.cos(phi), np.sin(phi)


def _align_for(ux, uy, tol=0.35):
    """由辐条方向决定对齐方式，让文字朝圆外生长，不压到圆内"""
    ha = "left" if ux > tol else ("right" if ux < -tol else "center")
    va = "bottom" if uy > tol else ("top" if uy < -tol else "center")
    return ha, va


def _panel_scale(ax):
    """按面板实际物理尺寸（英寸）给出缩放系数，基准是 6 英寸见方。
    仪表盘里的雷达图比独立图小得多，字号/间距需同步缩小才装得下。"""
    fig = ax.figure
    if fig is None or not ax.get_visible():
        return 1.0
    fig.canvas.draw()
    bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    w = bbox.width / float(fig.dpi)
    h = bbox.height / float(fig.dpi)
    side = min(w, h)
    if side <= 0:
        return 1.0
    return float(np.clip(side / 6.0, 0.62, 1.0))


def radar(ax, values_a, values_b, labels, title,
          label_a="Accuracy", label_b="Coverage", vmax=100.0,
          headroom=1.30, show_legend=True):
    """五维雷达图。

    标签遮挡的根治办法：所有文字都用极坐标数据坐标 (theta, r) 定位，
    而不是 'offset points' 屏幕像素偏移。屏幕偏移对接近水平的辐条（左上、
    右上两根）根本不是"沿半径方向"，会把数值标签斜推到轴名上。

    版式分三层，由内到外互不侵占：
      数值标签  r <= vmax*1.07（ clamp 在圆内一点）
      百分比刻度  画在没有辐条的正下方缺口里
      维度名    r = vmax*1.15，按辐条方向决定 ha/va，朝外生长
    headroom 把 ylim 抬到 vmax*1.30，但 yticks 只到 100，
    所以圆环不往外扩，多出来的全是留给标签的空白。
    """
    a_vals = list(values_a)
    b_vals = list(values_b)
    n = len(a_vals)
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)

    rtop = vmax * headroom
    ax.set_theta_offset(TH_OFFSET)
    ax.set_theta_direction(TH_DIR)
    ax.set_ylim(0.0, rtop)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([])                 # 手动绘制，避免默认位置压住辐条
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    ax.grid(color=C_GRID, linewidth=0.9)
    ax.spines["polar"].set_visible(False)  # 外圈骨架对应 rtop，会误导，隐藏

    # 百分比刻度放在正下方缺口：辐条屏幕角为 18/90/162/234/306 度，
    # -90 度正好处在 234 与 306 之间的空档，左右各留 36 度
    th_ring = np.pi
    ring_texts = []
    for rv in (20, 40, 60, 80, 100):
        t = ax.text(th_ring, rv, "%d%%" % rv, ha="center", va="center",
                    fontsize=8.5, color=C_TEXT_DIM, zorder=6, path_effects=HALO)
        ring_texts.append(t)

    loop = np.append(ang, ang[0])
    pa = np.append(a_vals, a_vals[0])
    pb = np.append(b_vals, b_vals[0])

    ax.plot(loop, pa, "-o", color=C_ACC, linewidth=2.2, markersize=6,
            markerfacecolor="white", markeredgewidth=2, label=label_a, zorder=3)
    ax.fill(loop, pa, color=C_ACC, alpha=0.16, zorder=2)
    ax.plot(loop, pb, "--s", color=C_COV, linewidth=1.8, markersize=5,
            markerfacecolor="white", markeredgewidth=1.6, label=label_b, zorder=4)
    ax.fill(loop, pb, color=C_COV, alpha=0.10, zorder=1)

    r_name = vmax * 1.15
    r_cap = vmax * 1.07

    # 面板越小，字号与最小间距同步缩小。仪表盘里的雷达图只有独立图的 1/5 面积，
    # 用同样的字号和间距放不下，必须按实际尺寸自适应。
    scale = _panel_scale(ax)
    fs_name = 11.0 * scale
    fs_num = 9.5 * scale
    fs_min = 6.0
    fs_name, fs_num = max(fs_name, fs_min + 1.0), max(fs_num, fs_min)
    GAP_MIN = 12.0 * scale

    # 维度名：最外圈之外，朝外生长（固定不动，作为避让时的障碍物）
    name_texts = []
    for i, th in enumerate(ang):
        ha, va = _align_for(*_spoke_dir(th))
        t = ax.text(th, r_name, labels[i], ha=ha, va=va, fontsize=fs_name,
                    fontweight="bold", color="#1F2933", zorder=7)
        name_texts.append(t)

    # 数值标签：先按半径方向摆放
    #   - 两值差得开就贴着各自数据点放
    #   - 差得近就围绕中点对称撑开，保证至少 GAP_MIN 的间隔
    #   - 再叠一个很小的切向错位（弧长固定，半径越小角度越大）
    movable = []
    for i, th in enumerate(ang):
        a, c = a_vals[i], b_vals[i]
        lo, hi = (a, c) if a <= c else (c, a)
        if hi - lo >= GAP_MIN:
            rlo, rhi = lo - 3.0, hi + 4.0
        else:
            mid = 0.5 * (lo + hi)
            rlo, rhi = mid - GAP_MIN / 2.0 - 1.0, mid + GAP_MIN / 2.0 + 1.0
        ra, rc = (rlo, rhi) if a <= c else (rhi, rlo)
        ra = min(max(ra, 6.0), r_cap)
        rc = min(max(rc, 6.0), r_cap)

        dth = min(np.deg2rad(14.0), 4.0 / max(0.5 * (ra + rc), 10.0))
        ta = ax.text(th + dth, ra, "%.1f%%" % a, ha="center", va="center",
                     fontsize=fs_num, color=C_ACC, fontweight="bold", zorder=8,
                     path_effects=HALO)
        tc = ax.text(th - dth, rc, "%.0f%%" % c, ha="center", va="center",
                     fontsize=max(fs_num - 1.0, fs_min), color=C_COV, zorder=8,
                     path_effects=HALO)
        movable += [(ta, (th, a), C_ACC), (tc, (th, c), C_COV)]

    if title:
        ax.set_title(title, fontsize=13.5, pad=16, fontweight="bold")
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
                  frameon=False, ncol=2, fontsize=10.5)

    _deoverlap(ax, movable, name_texts + ring_texts)
    _draw_leaders(ax, movable)
    return movable


def _shift(ax, text, dxy, rmax):
    """把文字按屏幕像素平移，再换算回极坐标 (theta, r)"""
    th, r = text.get_position()
    px, py = ax.transData.transform((th, r))
    nth, nr = ax.transData.inverted().transform((px + dxy[0], py + dxy[1]))
    nr = min(max(nr, 2.0), rmax * 0.98)
    text.set_position((nth, nr))


def _deoverlap(ax, movable, obstacles, pad=2.5, max_iter=90):
    """像素级自动避让。

    初始摆放只能处理常见情况：一旦多个维度同时挤到圆心附近（比如所有
    维度准确率都很低），5 根轴的标签在几何上必然重合，靠调参数无解。
    这里改成按实际包围盒做迭代排斥：每轮取重叠的一对，沿"最小平移量"
    方向（重叠较小的那个轴）各推开一半，直到不重叠或迭代用尽。
    维度名不参与移动，只作为障碍物把数值标签顶开。
    """
    if not movable:
        return
    fig = ax.figure
    fig.canvas.draw()                       # 先把 transform 定下来
    renderer = fig.canvas.get_renderer()
    rmax = ax.get_ylim()[1]
    texts = [m[0] for m in movable]
    fixed = [(t, t.get_window_extent(renderer=renderer)) for t in obstacles]

    for _ in range(max_iter):
        moved = False
        boxes = [t.get_window_extent(renderer=renderer) for t in texts]

        # 数值标签之间互推
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                bi, bj = boxes[i], boxes[j]
                ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0) + pad
                oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0) + pad
                if ox <= 0 or oy <= 0:
                    continue
                if ox < oy:                 # 水平方向推开更省力
                    d, s = ox / 2.0, 1.0 if (bi.x0 + bi.x1) <= (bj.x0 + bj.x1) else -1.0
                    _shift(ax, texts[i], (-s * d, 0.0), rmax)
                    _shift(ax, texts[j], (s * d, 0.0), rmax)
                else:
                    d, s = oy / 2.0, 1.0 if (bi.y0 + bi.y1) <= (bj.y0 + bj.y1) else -1.0
                    _shift(ax, texts[i], (0.0, -s * d), rmax)
                    _shift(ax, texts[j], (0.0, s * d), rmax)
                boxes[i] = texts[i].get_window_extent(renderer=renderer)
                boxes[j] = texts[j].get_window_extent(renderer=renderer)
                moved = True

        # 数值标签被维度名挡住时，只挪数值标签
        for i, t in enumerate(texts):
            b = boxes[i]
            for _, fb in fixed:
                ox = min(b.x1, fb.x1) - max(b.x0, fb.x0) + pad
                oy = min(b.y1, fb.y1) - max(b.y0, fb.y0) + pad
                if ox <= 0 or oy <= 0:
                    continue
                if ox < oy:
                    s = 1.0 if (b.x0 + b.x1) <= (fb.x0 + fb.x1) else -1.0
                    _shift(ax, t, (-s * ox, 0.0), rmax)
                else:
                    s = 1.0 if (b.y0 + b.y1) <= (fb.y0 + fb.y1) else -1.0
                    _shift(ax, t, (0.0, -s * oy), rmax)
                boxes[i] = t.get_window_extent(renderer=renderer)
                moved = True

        if not moved:
            break


def _draw_leaders(ax, movable, thresh=13.0):
    """标签被推开较远时补一条引线，避免看不出它属于哪个数据点。
    正常数据下 initial 摆放就够好，引线不会出现。"""
    for t, (th, val), color in movable:
        th_l, r_l = t.get_position()
        px, py = ax.transData.transform((th, val))
        qx, qy = ax.transData.transform((th_l, r_l))
        if (px - qx) ** 2 + (py - qy) ** 2 <= thresh ** 2:
            continue
        ax.plot([th, th_l], [val, r_l], color=color, lw=0.7, alpha=0.45,
                solid_capstyle="round", zorder=7)


def grouped_bars(ax, rules, acc, strict, cov):
    x = np.arange(len(rules))
    w = 0.26
    b1 = ax.bar(x - w, acc, w, label="Accuracy (partial credit)", color=C_ACC,
                edgecolor="white")
    b2 = ax.bar(x, strict, w, label="Strict accuracy", color=C_STRICT, edgecolor="white")
    b3 = ax.bar(x + w, cov, w, label="Coverage", color=C_COV, edgecolor="white")
    for bars in (b1, b2, b3):
        for r in bars:
            ax.annotate("%.0f" % r.get_height(),
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8.5)
    ax.axhline(50, ls=":", color="#9B9B9B", lw=1.2)
    ax.annotate("random baseline (binary) 50%", (len(rules) - 0.5, 51.5),
                ha="right", fontsize=8.5, color=C_TEXT_DIM)
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=10.5)
    ax.set_ylabel("%", fontsize=10)
    # 顶部留 headroom 给单行图例：柱最高到 100，数值标签顶到 ~101，
    # 图例落在 112 以上，两者不会碰。用 ncol=3 压成一行是关键——
    # 3 项排成两行时会压住高柱的数值标签。
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Per-rule metric comparison", fontsize=13.5, pad=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)


def dist_acc(ax, rules, n_gt, acc):
    x = np.arange(len(rules))
    bars = ax.bar(x, n_gt, 0.5, color=C_BAR, edgecolor="white")
    for r in bars:
        ax.annotate("%d" % int(r.get_height()),
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax.set_ylabel("Samples where rule is present in GT", fontsize=10, color="#5A6B7D")
    ax.set_xticks(x)
    ax.set_xticklabels(rules, fontsize=10.5)
    ax.set_ylim(0, max(n_gt) * 1.28)
    ax.spines[["top"]].set_visible(False)
    ax.grid(axis="y", color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    # twinx 共享 x 轴，会把同一组刻度标签画两遍（位置完全重合，文字发虚）
    ax2.tick_params(axis="x", which="both", length=0, labelbottom=False)
    ax2.plot(x, acc, "-o", color=C_LINE, lw=2, ms=6, markerfacecolor="white",
             markeredgewidth=2, label="Accuracy")
    for xi, v in zip(x, acc):
        ax2.annotate("%.1f%%" % v, (xi, v), textcoords="offset points",
                     xytext=(0, -16), ha="center", fontsize=9, color=C_LINE,
                     fontweight="bold")
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Accuracy (%)", fontsize=10, color=C_LINE)
    ax2.tick_params(axis="y", colors=C_LINE)
    ax2.spines[["top"]].set_visible(False)
    ax.set_title("Sample size vs accuracy (bars = N, line = accuracy)",
                 fontsize=13.5, pad=12, fontweight="bold")


def confusion(ax, nc, N):
    gt_nc, pr_nc, hit = nc["n_gt"], nc["n_pred"], nc["n_correct"]
    mat = np.array([[hit, gt_nc - hit],
                    [pr_nc - hit, N - gt_nc - pr_nc + hit]])
    ax.imshow(mat, cmap="Blues")
    cell_note = [["true negative", "false positive"],
                 ["false negative (miss)", "true positive"]]
    for i in range(2):
        for j in range(2):
            color = "white" if mat[i, j] > mat.max() * 0.55 else "#1F2933"
            ax.text(j, i, "%d" % mat[i, j], ha="center", va="center",
                    fontsize=17, fontweight="bold", color=color)
            ax.text(j, i + 0.30, cell_note[i][j], ha="center", va="center",
                    fontsize=8.5, color=color, alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: no change", "Pred: change"], fontsize=10.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GT: no change", "GT: change"], fontsize=10.5)
    ax.set_title("No-change detection confusion matrix", fontsize=13.5, pad=12,
                 fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    txt = ("recall %.1f%%   precision %.1f%%\n"
           "model over-predicts 'no change' (%d vs %d)"
           % (pct(nc["recall"]), pct(nc["precision"]), pr_nc, gt_nc))
    ax.set_xlabel(txt, fontsize=9.5, color="#5A6B7D", labelpad=10)


def kpi_band(fig, s):
    N = s["n_samples"]
    perfect = s["n_all_dimensions_correct"]
    band = fig.add_axes([0, 0.945, 1, 0.055])
    band.axis("off")
    items = [
        ("Total samples", "%d" % N, "#1F2933"),
        ("All rules correct", "%d  (%.1f%%)" % (perfect, perfect / N * 100), "#1F2933"),
        ("Macro avg (5 rules)", "%.1f%%" % pct(s["macro_accuracy"]), C_ACC),
        ("No-change share", "%d / %d  (%.1f%%)" % (s["no_change"]["n_gt"], N,
                                                   s["no_change"]["n_gt"] / N * 100),
         C_TEXT_DIM),
        ("Type acc on changed", "%.1f%%" % pct(s["type_on_changed_only"]["accuracy"]),
         C_LINE),
    ]
    n = len(items)
    for i, (k, v, c) in enumerate(items):
        x = (i + 0.5) / n
        band.text(x, 0.62, v, ha="center", va="center", fontsize=15,
                  fontweight="bold", color=c, transform=band.transAxes)
        band.text(x, 0.14, k, ha="center", va="center", fontsize=9.5,
                  color=C_TEXT_DIM, transform=band.transAxes)
        if i:
            band.plot([i / n, i / n], [0.15, 0.85], color="#DDE3EA", lw=1.2,
                      transform=band.transAxes)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Visualize change-caption summary.json")
    ap.add_argument("--summary", default="summary.json")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--font", "-F", default=None,
                    help="path to a .ttf font (e.g. .../timesnewroman/times.ttf)")
    args = ap.parse_args()

    font_note = setup_font(args.font)

    with open(args.summary, encoding="utf-8") as f:
        s = json.load(f)
    os.makedirs(args.outdir, exist_ok=True)

    R, N = s["rules"], s["n_samples"]
    acc = [pct(R[r]["accuracy"]) for r in RULES]
    strict = [pct(R[r]["strict_accuracy"]) for r in RULES]
    cov = [pct(R[r]["coverage"]) for r in RULES]
    n_gt = [R[r]["n_gt"] for r in RULES]

    print("font: %s" % font_note)

    # ---------- Figure 1: radar only ----------
    fig = plt.figure(figsize=(8.6, 9.0))
    ax = fig.add_axes([0.05, 0.11, 0.90, 0.76], polar=True)
    radar(ax, acc, cov, RULES,
          "Change captioning: five-rule performance\nAccuracy vs Coverage (N=%d)" % N)
    fig.text(0.5, 0.020,
             "Coverage = share of samples where the model generates that rule at all, "
             "given it is present in GT.\n"
             "Gap between the two curves = 'generated but filled in wrong'.",
             ha="center", fontsize=9, color=C_TEXT_DIM)
    p1 = os.path.join(args.outdir, "change_caption_radar.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------- Figure 2: dashboard ----------
    fig = plt.figure(figsize=(16.5, 11.5))
    fig.patch.set_facecolor("white")
    kpi_band(fig, s)

    ax1 = fig.add_axes([0.030, 0.505, 0.455, 0.395], polar=True)
    radar(ax1, acc, cov, RULES, "Radar: accuracy vs coverage", show_legend=False)

    ax2 = fig.add_axes([0.555, 0.545, 0.415, 0.345])
    grouped_bars(ax2, RULES, acc, strict, cov)

    ax3 = fig.add_axes([0.555, 0.075, 0.415, 0.345])
    dist_acc(ax3, RULES, n_gt, acc)

    ax4 = fig.add_axes([0.060, 0.080, 0.375, 0.355])
    confusion(ax4, s["no_change"], N)

    fig.text(0.5, 0.028,
             "Blue = accuracy (partial credit for Background)    "
             "Orange = coverage\n"
             "Denominator = samples where the rule is present in GT; "
             "no-change samples contribute to Type only.",
             ha="center", fontsize=9, color="#9AA5B1")

    p2 = os.path.join(args.outdir, "change_caption_dashboard.png")
    fig.savefig(p2, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("saved:\n  %s\n  %s" % (os.path.abspath(p1), os.path.abspath(p2)))


if __name__ == "__main__":
    main()