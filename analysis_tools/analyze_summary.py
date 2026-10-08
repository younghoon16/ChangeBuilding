#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 eval_change_caption.py 产出的 summary.json 做深度分析。

summary.json 里只有汇总计数，很多关键结论要反推才能得到。本脚本把这一步
固化下来，输出可直接用于论文/报告的数字：

  1. 无变化判定混淆矩阵（由 n_gt / n_pred / n_correct 反推四格）
     -> 变化漏检率、过报率、多数类基线
  2. 有变化样本上的表现（剔除 no-change 白送的分数）
     -> 有变化样本全维度准确率、每样本平均参与判定维度数
  3. 各维度准确率 + 95% 置信区间（Wilson）
     -> 判断某维度是否真的显著差于随机基线
  4. 误差三分解：漏生成 / 生成了但填错 / 部分命中
     -> 区分「没说」和「说错」，决定该调生成策略还是调识别能力
  5. 与随机基线的显著性检验（二项检验，无需 scipy）

用法
  python analyze_summary.py --summary summary.json
  python analyze_summary.py --summary summary.json --detail detail.csv --out-csv analysis.csv

加 --detail（评测脚本产出的逐条明细 CSV）后会额外给出：
  - 各维度真实类别数 K 与多数类基线（比 1/K 更贴合数据分布）
  - 高频错误模式 Top（GT 值 -> 预测值）
  - 有变化样本全对率的精确值

只依赖标准库。
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, OrderedDict

RULES = ["Type", "Amount", "Position", "Background", "Density"]
RULE_CN = {"Type": "变化类型", "Amount": "变化数量", "Position": "变化位置",
           "Background": "背景地物", "Density": "分布密度"}

# 视为二分类的维度，随机基线 50%
BINARY_RULES = {"Type", "Amount", "Density"}


# --------------------------------------------------------------------------- #
#  统计工具
# --------------------------------------------------------------------------- #

def wilson(k, n, z=1.96):
    """二项比例 Wilson score 95% 置信区间（k 必须为整数计数）"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wald(p, n, z=1.96):
    """含 0.5 计分时的近似区间（score 非整数计数，Wilson 不适用）"""
    if n == 0:
        return (0.0, 0.0)
    h = z * math.sqrt(max(p * (1 - p), 0.0) / n)
    return (max(0.0, p - h), min(1.0, p + h))


def _log_choose(n, k):
    """log(C(n, k))，对数域计算，避免 math.comb 对大 n 返回超出 float 范围的整数"""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_test_two_sided(k, n, p0):
    """精确二项检验双侧 p 值（纯标准库，无需 scipy）。

    注意：不能用 math.comb(n, i) * p0**i * (1-p0)**(n-i) 直接算。
      - math.comb 返回 Python 大整数，n 约 1030 以上时超过 float 上限 1.8e308，
        int * float 会抛 OverflowError；
      - 同时 p0**i 在大 n 时下溢为 0.0。
    因此全程走对数域：log pmf = log C(n,i) + i*log(p0) + (n-i)*log1p(-p0)。

    双侧定义采用「小概率之和」（method of small p-values）：
    把所有 pmf(i) <= pmf(k) 的点累加，这是 scipy.stats.binomtest 的默认做法。
    """
    if n == 0:
        return 1.0
    if not (0.0 < p0 < 1.0):          # 退化基线：全猜某一类
        if p0 <= 0.0:
            return 1.0 if k == 0 else 0.0
        return 1.0 if k == n else 0.0
    if not (0 <= k <= n):
        return 1.0

    log_p = math.log(p0)
    log_q = math.log1p(-p0)           # log(1-p0)，p0 接近 1 时仍精确
    log_norm = _log_choose(n, k)      # 逐点复用，省一次 lgamma 三元组的重复开销
    def logpmf(i):
        if i == k:
            return log_norm + i * log_p + (n - i) * log_q
        return _log_choose(n, i) + i * log_p + (n - i) * log_q

    logs = [logpmf(i) for i in range(n + 1)]
    obs = logs[k]
    # 1e-9 的对数容差，吸收 p0=0.5 时 k 与 n-k 本应相等却相差的浮点误差
    sel = [x for x in logs if x <= obs + 1e-9]
    if not sel:
        return 0.0
    m = max(sel)
    # sum(exp(x)) = exp(m) * sum(exp(x-m))，先平移再求和，避免逐项下溢
    tot = sum(math.exp(x - m) for x in sel)
    try:
        pval = math.exp(m) * tot
    except OverflowError:
        return 1.0
    return min(1.0, max(0.0, pval))


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def pfmt(p):
    """大样本下 p 值可能小到 1e-300，用 %.3f 会一律显示 0.000，改用科学计数法"""
    if p is None:
        return "-"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-4:
        return "%.1e" % p
    return "%.4f" % p


def disp_width(s):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def print_table(rows, headers, aligns=None):
    all_rows = [headers] + rows
    w = [max(disp_width(r[i]) for r in all_rows) for i in range(len(headers))]
    def render(r):
        out = []
        for i, x in enumerate(r):
            s = str(x)
            pad = w[i] - disp_width(s)
            out.append(s + " " * pad if (aligns and i < len(aligns) and aligns[i] == "l")
                       else " " * pad + s)
        return "  ".join(out)
    print(render(headers))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(render(r))


def pctf(x, nd=1):
    return "-" if x is None else "%.*f%%" % (nd, x * 100)


# --------------------------------------------------------------------------- #
#  1. 无变化混淆矩阵反推
# --------------------------------------------------------------------------- #

def confusion_matrix(s):
    """由 (n_gt, n_pred, n_correct) 反推四格。
       四格关系：hit + (n_gt - hit) = n_gt ；hit + (n_pred - hit) = n_pred
       miss_rate / false_alarm 等均可由三者唯一确定。"""
    nc = s["no_change"]
    N = s["n_samples"]
    gt_nc, pr_nc, hit = nc["n_gt"], nc["n_pred"], nc["n_correct"]
    mat = OrderedDict([
        ("true_neg",  hit),                      # GT 无变化 & 预测无变化
        ("false_pos", gt_nc - hit),              # GT 无变化 & 预测有变化（误报）
        ("false_neg", pr_nc - hit),              # GT 有变化 & 预测无变化（漏检）
        ("true_pos",  N - gt_nc - pr_nc + hit),  # GT 有变化 & 预测有变化
    ])
    n_changed = N - gt_nc
    n_unchanged = gt_nc
    derived = {
        "n_samples": N,
        "n_changed": n_changed,
        "n_unchanged": n_unchanged,
        "changed_ratio": n_changed / N if N else 0,
        "miss_rate": (pr_nc - hit) / n_changed if n_changed else None,   # 漏检率
        "false_alarm_rate": (gt_nc - hit) / n_unchanged if n_unchanged else None,
        "majority_baseline": max(n_changed, n_unchanged) / N if N else 0,
        "balanced_acc": 0.5 * ((hit / n_unchanged if n_unchanged else 0.0)
                               + ((N - gt_nc - pr_nc + hit) / n_changed if n_changed else 0.0)),
    }
    return mat, derived


# --------------------------------------------------------------------------- #
#  2. 有变化样本上的表现反推
# --------------------------------------------------------------------------- #

def changed_subset(s):
    """no-change 样本在评测脚本里其余四维被置空、不参与判定，
       所以『无变化且预测也无变化』= 自动全维度全对。
       据此把 n_all_dimensions_correct 拆成两部分。"""
    nc = s["no_change"]
    N = s["n_samples"]
    total_perfect = s["n_all_dimensions_correct"]
    gt_nc, hit = nc["n_gt"], nc["n_correct"]
    n_changed = N - gt_nc

    free_perfect = hit                       # 白送的全对（GT 无变化 + 预测无变化）
    changed_perfect = total_perfect - free_perfect
    # 有变化样本里，非 Type 维度的出现次数全部分布在这 187 条上
    dim_slots = sum(s["rules"][r]["n_gt"] for r in RULES if r != "Type")
    per_sample_dims = (dim_slots / n_changed) if n_changed else 0

    return {
        "n_changed": n_changed,
        "n_perfect_total": total_perfect,
        "n_perfect_free": free_perfect,
        "n_perfect_changed": changed_perfect,
        "changed_perfect_rate": changed_perfect / n_changed if n_changed else None,
        "overall_perfect_rate": total_perfect / N if N else None,
        "gap_pp": (total_perfect / N - changed_perfect / n_changed) * 100 if n_changed and N else None,
        "dim_slots_on_changed": dim_slots,
        "avg_dims_per_changed_sample": per_sample_dims,
        "avg_dims_per_changed_sample_incl_type": per_sample_dims + 1,
    }


# --------------------------------------------------------------------------- #
#  3. 各维度：置信区间 + 随机基线 + 显著性
# --------------------------------------------------------------------------- #

def rule_analysis(s, baseline_map):
    out = []
    for r in RULES:
        d = s["rules"][r]
        n, hit = d["n_gt"], d["n_hit"]
        partial = d["n_partial"]
        acc = d["accuracy"]
        strict = d["strict_accuracy"]
        cov = d["coverage"]
        base = baseline_map.get(r)
        # 整数计数用 Wilson，含 0.5 的 score 用 Wald 近似
        ci = wilson(hit, n)
        ci_acc = ci if partial == 0 else wald(acc, n)
        pval = binom_test_two_sided(hit, n, base) if (n and base is not None) else None
        out.append({
            "rule": r, "cn": RULE_CN[r], "n": n,
            "acc": acc, "acc_lo": ci_acc[0], "acc_hi": ci_acc[1],
            "strict": strict, "strict_lo": ci[0], "strict_hi": ci[1],
            "cov": cov, "n_pred": d["n_pred"], "partial": partial,
            "baseline": base, "p_value": pval,
            "ci_width_pp": (ci_acc[1] - ci_acc[0]) * 100,
            "beats_baseline": (acc > base) if base is not None else None,
            "contains_baseline": (ci_acc[0] <= base <= ci_acc[1]) if base is not None else None,
        })
    return out


# --------------------------------------------------------------------------- #
#  4. 误差三分解
# --------------------------------------------------------------------------- #

def error_breakdown(s, detail_stats=None):
    """把 n_gt 拆成：完全对 / 部分对 / 生成了但填错 / 压根没生成。
       有 detail.csv 时用逐条精确计数；否则用汇总量近似。"""
    rows = []
    for r in RULES:
        d = s["rules"][r]
        n, hit, partial, n_pred = d["n_gt"], d["n_hit"], d["n_partial"], d["n_pred"]
        if detail_stats and r in detail_stats:
            miss, wrong = detail_stats[r]["miss"], detail_stats[r]["wrong"]
            exact = True
        else:
            # 近似：n_pred <= n_gt 时，未生成数 ≈ n_gt - n_pred
            miss = max(0, n - n_pred)
            wrong = max(0, n_pred - hit - partial)
            exact = False
        rows.append({
            "rule": r, "cn": RULE_CN[r], "n": n,
            "ok": hit, "partial": partial, "wrong": wrong, "miss": miss,
            "ok_pct": hit / n if n else 0,
            "partial_pct": partial / n if n else 0,
            "wrong_pct": wrong / n if n else 0,
            "miss_pct": miss / n if n else 0,
            "exact": exact,
        })
    return rows


# --------------------------------------------------------------------------- #
#  5. 可选：读 detail.csv 做细粒度分析
# --------------------------------------------------------------------------- #

def load_detail(path):
    """返回 (per_rule_stats, error_patterns, gt_class_dist, changed_perfect_exact)"""
    if not path or not os.path.exists(path):
        return None, None, None, None
    stats, patterns, gt_dist = {}, {r: Counter() for r in RULES}, {r: Counter() for r in RULES}
    n_changed = n_changed_perfect = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames is None:
            return None, None, None, None
        has_nc = "gt_nochange" in rd.fieldnames
        for row in rd:
            is_nc = has_nc and row.get("gt_nochange") in ("1", "True", "true")
            if not is_nc and has_nc:
                n_changed += 1
                if row.get("all_ok") in ("1", "True", "true"):
                    n_changed_perfect += 1
            for r in RULES:
                gk, pk, fk = "gt_" + r, "pred_" + r, "flag_" + r
                if gk not in row:
                    continue
                gv, pv, fl = row.get(gk, ""), row.get(pk, ""), row.get(fk, "")
                if gv == "":
                    continue
                st = stats.setdefault(r, {"miss": 0, "wrong": 0, "ok": 0, "partial": 0})
                gt_dist[r][gv] += 1
                if fl == "OK":
                    st["ok"] += 1
                elif fl == "PART":
                    st["partial"] += 1
                elif fl == "WRONG":
                    st["wrong"] += 1
                    if pv == "":
                        st["miss"] += 1
                    else:
                        patterns[r][(gv, pv)] += 1
    cpe = (n_changed_perfect / n_changed) if n_changed else None
    return stats, patterns, gt_dist, (n_changed, n_changed_perfect, cpe)


def baseline_from_dist(gt_dist):
    """多数类基线：全猜 GT 里最常见的那个值能拿多少分。
       比 1/K 更贴合真实数据分布。"""
    out = {}
    for r in RULES:
        if r in gt_dist and gt_dist[r]:
            tot = sum(gt_dist[r].values())
            out[r] = max(gt_dist[r].values()) / tot
    return out


def default_baseline(r):
    if r in BINARY_RULES:
        return 0.5
    return None      # 多分类，类别数未知，需 detail.csv


# --------------------------------------------------------------------------- #
#  主流程
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="变化描述评测结果深度分析")
    ap.add_argument("--summary", default="summary.json")
    ap.add_argument("--detail", default="", help="评测产出的逐条明细 CSV（可选）")
    ap.add_argument("--out-csv", default="", help="把分析表导出为 CSV")
    ap.add_argument("--baseline", default="",
                    help="自定义随机基线 JSON 文件，格式 "
                         '{"Position": 0.11, "Background": 0.20}')
    args = ap.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        s = json.load(f)
    N = s["n_samples"]

    detail_stats, patterns, gt_dist, cpe = load_detail(args.detail)

    # 随机基线：有多数类分布就用，否则二分类 0.5 / 多分类标注未知
    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            base_map = json.load(f)
        base_map = {r: base_map.get(r) for r in RULES}
        base_src = "自定义基线（--baseline %s）" % os.path.basename(args.baseline)
    elif gt_dist:
        base_map = baseline_from_dist(gt_dist)
        base_src = "多数类基线（由 detail 的 GT 分布计算）"
    else:
        base_map = {r: default_baseline(r) for r in RULES}
        base_src = "均匀随机基线（二分类取 50%；多分类需 --detail 或 --baseline 指定）"

    print("\n" + "=" * 86)
    print("变化描述评测结果 · 深度分析")
    print("=" * 86)
    print("样本总数 %d    来源 %s" % (N, os.path.abspath(args.summary)))
    if args.detail and detail_stats:
        print("明细表    %s  ✓ 启用精确误差分解" % os.path.abspath(args.detail))
    else:
        print("明细表    未提供  ·  误差分解为汇总近似，加 --detail detail.csv 可精确到条")

    # ---------- 1. 混淆矩阵 ----------
    mat, der = confusion_matrix(s)
    print("\n" + "-" * 86)
    print("【1】无变化判定混淆矩阵")
    print("-" * 86)
    print_table([
        ["GT 无变化", mat["true_neg"], mat["false_pos"], s["no_change"]["n_gt"]],
        ["GT 有变化", mat["false_neg"], mat["true_pos"], der["n_changed"]],
        ["合计", s["no_change"]["n_pred"], N - s["no_change"]["n_pred"], N],
    ], ["", "预测：无变化", "预测：有变化", "合计"], aligns=["l", "r", "r", "r"])

    mr = der["miss_rate"]
    ci_mr = wilson(mat["false_neg"], der["n_changed"])
    print("\n  真实有变化样本占比：%d / %d = %.1f%%"
          % (der["n_changed"], N, der["changed_ratio"] * 100))
    print("  多数类基线（全猜「无变化」即可拿到）：%.1f%%" % (der["majority_baseline"] * 100))
    print("  无变化召回 %s   无变化精确率 %s"
          % (pctf(s["no_change"]["recall"]), pctf(s["no_change"]["precision"])))
    print("  >> 变化漏检率 %.1f%%（%d/%d，95%%CI %.1f%%~%.1f%%）—— 真实变化被判成无变化"
          % (mr * 100, mat["false_neg"], der["n_changed"], ci_mr[0] * 100, ci_mr[1] * 100))
    print("  >> 变化过报率 %.1f%%（%d/%d）—— 无变化被判成有变化"
          % (der["false_alarm_rate"] * 100, mat["false_pos"], der["n_unchanged"]))
    _d = s["no_change"]["n_pred"] - s["no_change"]["n_gt"]
    if _d > 0:
        print("  >> 预测无变化 %d 条 > GT 无变化 %d 条（多 %d 条），模型整体偏保守，"
              "倾向漏报变化" % (s["no_change"]["n_pred"], s["no_change"]["n_gt"], _d))
    elif _d < 0:
        print("  >> 预测无变化 %d 条 < GT 无变化 %d 条（少 %d 条），模型整体偏激进，"
              "倾向误报变化" % (s["no_change"]["n_pred"], s["no_change"]["n_gt"], -_d))
    else:
        print("  >> 预测无变化与 GT 无变化均为 %d 条，整体无系统性偏向，"
              "但漏检与误报可能同时存在并相互抵消" % s["no_change"]["n_pred"])

    # ---------- 2. 有变化子集 ----------
    cs = changed_subset(s)
    print("\n" + "-" * 86)
    print("【2】剔除 no-change 白送分后的真实表现")
    print("-" * 86)
    print("  no-change 样本在评测中其余四维置空，只要 Type 判对就自动『全维度全对』，")
    print("  因此整体全对率受多数类主导。拆分如下：\n")
    print_table([
        ["整体（%d 条）" % N, cs["n_perfect_total"], pctf(cs["overall_perfect_rate"])],
        ["  其中：无变化白送（GT 无变化且预测无变化）", cs["n_perfect_free"], "—"],
        ["  其中：有变化样本（%d 条）" % cs["n_changed"],
         cs["n_perfect_changed"], pctf(cs["changed_perfect_rate"])],
    ], ["全维度全对拆分", "条数", "占比"], aligns=["l", "r", "r"])
    print("\n  >> 有变化样本全对率 %.1f%%，与整体 %.1f%% 相差 %.1f 个百分点——"
          % (cs["changed_perfect_rate"] * 100, cs["overall_perfect_rate"] * 100, cs["gap_pp"]))
    print("     单独报整体全对率会严重高估模型能力，建议改用有变化子集口径。")
    print("  >> 有变化样本平均每条参与判定 %.2f 个维度（含 Type 共 %.2f 个）"
          % (cs["avg_dims_per_changed_sample"], cs["avg_dims_per_changed_sample_incl_type"]))
    if cpe and cpe[2] is not None:
        print("  >> 明细精确值：有变化样本全对 %d/%d = %.1f%%（与反推值 %s 一致）"
              % (cpe[1], cpe[0], cpe[2] * 100,
                 "相同" if abs(cpe[2] - cs["changed_perfect_rate"]) < 1e-9 else "有出入"))

    # ---------- 3. 各维度 + CI + 基线 ----------
    rows = rule_analysis(s, base_map)
    print("\n" + "-" * 86)
    print("【3】各维度准确率 · 95% 置信区间 · 随机基线检验")
    print("-" * 86)
    print("  随机基线：%s\n" % base_src)
    print_table([
        ["%s %s" % (x["cn"], x["rule"]), x["n"],
         pctf(x["acc"]), "%.1f ~ %.1f" % (x["acc_lo"] * 100, x["acc_hi"] * 100),
         pctf(x["baseline"]) if x["baseline"] is not None else "未知",
         pfmt(x["p_value"]),
         stars(x["p_value"]) if x["p_value"] is not None else "-",
         "是" if x["contains_baseline"] else ("否" if x["contains_baseline"] is False else "-"),
         "%.1f" % x["ci_width_pp"]]
        for x in rows
    ], ["Rule", "样本数", "准确率", "95% CI (%)", "随机基线", "p 值", "显著",
        "CI含基线", "CI宽度(pp)"],
        aligns=["l", "r", "r", "r", "r", "r", "l", "r", "r"])
    # Type 行的分母含 no-change 样本，属于三分类，与 50% 二分类基线不可直接比
    t_ch = s.get("type_on_changed_only", {})
    if t_ch.get("n"):
        ci_t = wilson(t_ch["hit"], t_ch["n"])
        print("\n  [口径] Type 行 %.1f%% 的分母含 no-change 样本（三分类），与 50%% 二分类基线不可直接比。"
              % (s["rules"]["Type"]["accuracy"] * 100))
        print("         只看有变化样本的二分类 Type：%.1f%%（%d/%d，95%%CI %.1f%%~%.1f%%）"
              % (t_ch["accuracy"] * 100, t_ch["hit"], t_ch["n"],
                 ci_t[0] * 100, ci_t[1] * 100))
    print("\n  注：p 值用精确二项检验（H0: 准确率 = 随机基线）；"
          "*** p<0.001  ** p<0.01  * p<0.05  n.s. 不显著")
    for x in rows:
        if x["contains_baseline"]:
            print("  [警告] %s：95%% CI [%.1f%%, %.1f%%] 覆盖基线 %.1f%%，"
                  "样本仅 %d 条，无法区分于随机猜测，切勿下确定性结论。"
                  % (x["rule"], x["acc_lo"] * 100, x["acc_hi"] * 100,
                     x["baseline"] * 100, x["n"]))
        if x["n"] < 100:
            print("  [提示] %s：样本 %d 条，CI 宽达 %.1f 个百分点，建议扩充该维度测试样本。"
                  % (x["rule"], x["n"], x["ci_width_pp"]))

    # ---------- 4. 误差分解 ----------
    eb = error_breakdown(s, detail_stats)
    exact = all(x["exact"] for x in eb)
    print("\n" + "-" * 86)
    print("【4】误差三分解：把每个维度的错误拆成『没生成』和『生成错』")
    print("-" * 86)
    print("  %s\n" % ("（逐条明细精确统计）" if exact else "（汇总近似，加 --detail 可精确）"))
    print_table([
        ["%s %s" % (x["cn"], x["rule"]), x["n"],
         "%d (%.0f%%)" % (x["ok"], x["ok_pct"] * 100),
         "%d (%.0f%%)" % (x["partial"], x["partial_pct"] * 100),
         "%d (%.0f%%)" % (x["wrong"], x["wrong_pct"] * 100),
         "%d (%.0f%%)" % (x["miss"], x["miss_pct"] * 100),
         ("∞" if x["miss_pct"] < 1e-9
          else "%.1f" % (x["wrong_pct"] / x["miss_pct"]))]
        for x in eb
    ], ["Rule", "样本数", "完全正确", "部分正确", "生成了但填错", "压根没生成", "错/漏比"],
        aligns=["l", "r", "r", "r", "r", "r", "r"])
    print("\n  解读：错/漏比 >> 1 说明模型『说了但说错』，短板在识别能力，")
    print("        增加生成约束无效，应提升视觉定位/计数能力或补该维度训练数据；")
    print("        错/漏比接近 0 说明模型『压根没说』，短板在生成覆盖，")
    print("        可用提示词/解码策略强制覆盖该维度。")
    for x in eb:
        if x["wrong_pct"] > 0.25 and x["miss_pct"] < 0.15:
            print("  >> %s：错 %.0f%% / 漏 %.0f%% —— 典型『说了但说错』"
                  % (x["rule"], x["wrong_pct"] * 100, x["miss_pct"] * 100))

    # ---------- 5. 高频错误模式 ----------
    if patterns:
        print("\n" + "-" * 86)
        print("【5】高频错误模式 Top5（GT 值 -> 预测值）")
        print("-" * 86)
        for r in RULES:
            if not patterns[r]:
                continue
            print("\n  [%s %s]" % (RULE_CN[r], r))
            for (g, p), c in patterns[r].most_common(5):
                print("    %-28s -> %-28s  %3d 次" % (g if g else "(空)", p if p else "(空)", c))

    # ---------- 6. 建议报数 ----------
    print("\n" + "-" * 86)
    print("【6】论文/报告建议口径")
    print("-" * 86)
    print("  1) Macro 五维平均准确率：%.1f%%" % (s["macro_accuracy"] * 100))
    print("  2) 有变化样本 Type 准确率：%.1f%%（%d/%d）"
          % (s["type_on_changed_only"]["accuracy"] * 100,
             s["type_on_changed_only"]["hit"], s["type_on_changed_only"]["n"]))
    print("  3) 变化漏检率：%.1f%%（%d/%d）" % (mr * 100, mat["false_neg"], der["n_changed"]))
    print("  4) 有变化样本全维度准确率：%.1f%%（%d/%d）"
          % (cs["changed_perfect_rate"] * 100, cs["n_perfect_changed"], cs["n_changed"]))
    print("\n  不推荐单独报：整体全维度全对率 %.1f%%（no-change 白送 %d 条，占 %.0f%%）"
          % (cs["overall_perfect_rate"] * 100, cs["n_perfect_free"],
             cs["n_perfect_free"] / max(cs["n_perfect_total"], 1) * 100))

    # ---------- 导出 ----------
    if args.out_csv:
        with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["section", "item", "value_str", "value_num"])
            w.writerow(["混淆矩阵", "true_neg", "", mat["true_neg"]])
            w.writerow(["混淆矩阵", "false_pos", "", mat["false_pos"]])
            w.writerow(["混淆矩阵", "false_neg", "", mat["false_neg"]])
            w.writerow(["混淆矩阵", "true_pos", "", mat["true_pos"]])
            w.writerow(["关键指标", "miss_rate", pctf(mr, 2), mr])
            w.writerow(["关键指标", "false_alarm_rate", pctf(der["false_alarm_rate"], 2),
                        der["false_alarm_rate"]])
            w.writerow(["关键指标", "changed_perfect_rate", pctf(cs["changed_perfect_rate"], 2),
                        cs["changed_perfect_rate"]])
            w.writerow(["关键指标", "overall_perfect_rate", pctf(cs["overall_perfect_rate"], 2),
                        cs["overall_perfect_rate"]])
            for x in rows:
                w.writerow(["维度准确率", x["rule"], pctf(x["acc"], 2), x["acc"]])
                w.writerow(["维度CI下界", x["rule"], pctf(x["acc_lo"], 2), x["acc_lo"]])
                w.writerow(["维度CI上界", x["rule"], pctf(x["acc_hi"], 2), x["acc_hi"]])
                w.writerow(["维度基线", x["rule"], pctf(x["baseline"], 2), x["baseline"]])
                w.writerow(["维度p值", x["rule"], "", x["p_value"]])
                w.writerow(["维度Coverage", x["rule"], pctf(x["cov"], 2), x["cov"]])
            for x in eb:
                w.writerow(["误差分解-填错", x["rule"], pctf(x["wrong_pct"], 2), x["wrong"]])
                w.writerow(["误差分解-漏生成", x["rule"], pctf(x["miss_pct"], 2), x["miss"]])
                w.writerow(["误差分解-部分对", x["rule"], pctf(x["partial_pct"], 2), x["partial"]])
        print("\n分析表已导出：%s" % os.path.abspath(args.out_csv))
    print()


if __name__ == "__main__":
    main()
