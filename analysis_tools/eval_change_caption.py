#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变化描述（Change Caption）五规则维度评测脚本  v2
======================================================================

输入：一个文件夹下所有 .txt，每个文件内部形如

    pred_caption: no building has appeared in the area and no building has disappeared in the area
    ref_caption:  no building has appeared in the area and no building has disappeared in the area .

（也支持 pred/ 与 gt/ 两个目录按文件名配对，见 --pred-dir / --gt-dir）

按 5 条规则把句子解析回槽位再逐槽比对：

  Rule                取值
  ------------------  ---------------------------------------------------
  Change Type         appear / disappear / none（无变化）
  Change Amount       one / several
  Change Position     center / top / left / right / bottom / top-left ...
  Change Background   road / tree / forest / river / farmland / building ...
  Change Density      dense / sparse

无变化（no-change）样本的特殊处理
  "no building has appeared ... and no building has disappeared ..." 里同时含
  appeared 和 disappeared，直接抽词必然判成"有变化"。因此先做否定检测：
  把 "no building ... appeared/disappeared" 这类否定片段整段剔除，若剩余部分
  再无变化信号，则整句判为 Type=none，其余四个维度一律置空（不计入分母）。
  这样 no-change 样本只影响 Type 一列，不会污染其他维度。

统计口径
  分母 = GT 中该维度有值的样本数；分子 = 归一化后与 GT 一致（Background 部分
  命中记 0.5）。同时给出 Coverage（GT 有时预测也生成的比例），用于区分
  "没生成这一维" 和 "生成错了"。

用法
  python eval_change_caption.py --data-dir ./results --out-csv detail.csv \
      --summary-json summary.json --dump-unmatched unmatched.txt

只依赖标准库，Python 3 直接运行。
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter, OrderedDict

# --------------------------------------------------------------------------- #
#  1. 词表：把各种写法归一到同一个 canonical 值
#     出现表外词时先跑 --dump-unmatched，把新词补进这里或用 --lexicon 传入
# --------------------------------------------------------------------------- #

LEXICON = {
    # ---- Change Type -------------------------------------------------------
    "type": {
        "appear": "appear", "appears": "appear", "appeared": "appear",
        "appearing": "appear", "emerge": "appear", "emerged": "appear",
        "emerging": "appear", "new": "appear", "newly": "appear",
        "construct": "appear", "constructed": "appear", "built": "appear",
        "add": "appear", "added": "appear", "increase": "appear",
        "increased": "appear", "gain": "appear",
        "disappear": "disappear", "disappears": "disappear", "disappeared": "disappear",
        "disappearing": "disappear", "vanish": "disappear", "vanished": "disappear",
        "remove": "disappear", "removed": "disappear", "demolish": "disappear",
        "demolished": "disappear", "destroy": "disappear", "destroyed": "disappear",
        "gone": "disappear", "reduce": "disappear", "reduced": "disappear",
        "decrease": "disappear", "decreased": "disappear", "loss": "disappear",
    },
    # ---- Change Amount -----------------------------------------------------
    "amount": {
        "a": "one", "an": "one", "one": "one", "single": "one", "1": "one",
        "several": "several", "multiple": "several", "few": "several",
        "some": "several", "many": "several", "group": "several", "groups": "several",
        "cluster": "several", "clusters": "several", "row": "several", "rows": "several",
        "set": "several", "couple": "several", "dozens": "several",
        "two": "several", "three": "several", "four": "several", "five": "several",
        "six": "several", "seven": "several", "eight": "several", "nine": "several",
        "ten": "several", "2": "several", "3": "several", "4": "several",
        "5": "several", "6": "several", "7": "several", "8": "several",
        "9": "several", "10": "several", "20": "several", "30": "several",
    },
    # ---- Change Position ---------------------------------------------------
    "position": {
        "center": "center", "centre": "center", "middle": "center", "central": "center",
        "top": "top", "upper": "top", "north": "top", "northern": "top", "uppermost": "top",
        "bottom": "bottom", "lower": "bottom", "south": "bottom", "southern": "bottom",
        "left": "left", "west": "left", "western": "left",
        "right": "right", "east": "right", "eastern": "right",
        "corner": "corner", "cornered": "corner",
        "edge": "edge", "border": "edge", "margin": "edge", "fringe": "edge",
        "top-left": "top-left", "upper-left": "top-left", "northwest": "top-left",
        "top-right": "top-right", "upper-right": "top-right", "northeast": "top-right",
        "bottom-left": "bottom-left", "lower-left": "bottom-left", "southwest": "bottom-left",
        "bottom-right": "bottom-right", "lower-right": "bottom-right", "southeast": "bottom-right",
    },
    # ---- Change Background（取介词短语中心名词归一；介词差异不追究）--------
    "background": {
        "road": "road", "roads": "road", "street": "road", "streets": "road",
        "path": "road", "paths": "road", "highway": "road", "highways": "road",
        "tree": "tree", "trees": "tree",
        "wood": "forest", "woods": "forest", "forest": "forest", "forests": "forest",
        "grove": "forest", "woodland": "forest",
        "river": "river", "rivers": "river", "stream": "river", "creek": "river",
        "lake": "lake", "lakes": "lake", "pond": "lake", "ponds": "lake",
        "reservoir": "lake",
        "sea": "sea", "ocean": "sea", "water": "water", "waters": "water",
        "field": "farmland", "fields": "farmland", "farmland": "farmland",
        "farmlands": "farmland", "cropland": "farmland", "croplands": "farmland",
        "paddy": "farmland", "farms": "farmland", "farm": "farmland",
        "grass": "grass", "grassland": "grass", "grasslands": "grass",
        "meadow": "grass", "meadows": "grass", "lawn": "grass",
        "mountain": "mountain", "mountains": "mountain", "hill": "mountain", "hills": "mountain",
        "building": "building", "buildings": "building", "house": "building",
        "houses": "building", "hut": "building", "huts": "building",
        "parking": "parking", "lot": "parking", "lots": "parking", "garage": "parking",
        "village": "village", "villages": "village", "town": "village",
        "residential": "village", "neighborhood": "village",
        "area": "area", "areas": "area", "region": "area", "regions": "area",
        "district": "area", "zone": "area", "block": "area", "place": "area",
        "factory": "factory", "factories": "factory", "plant": "factory",
        "industrial": "factory",
        "bridge": "bridge", "bridges": "bridge",
        "coast": "coast", "coastline": "coast", "beach": "coast",
        "desert": "desert", "sand": "desert",
        "bare": "bare-land", "barren": "bare-land", "bare-land": "bare-land",
        "bareland": "bare-land", "soil": "bare-land", "vacant": "bare-land",
        "construction": "construction-site", "site": "construction-site",
    },
    # ---- Change Density ----------------------------------------------------
    "density": {
        "dense": "dense", "densely": "dense", "denser": "dense", "densest": "dense",
        "crowded": "dense", "packed": "dense", "compact": "dense", "tight": "dense",
        "tightly": "dense", "concentrated": "dense",
        "sparse": "sparse", "sparsely": "sparse", "sparser": "sparse",
        "sparsest": "sparse", "scattered": "sparse", "scatter": "sparse",
        "dispersed": "sparse", "isolated": "sparse", "loose": "sparse",
        "loosely": "sparse",
    },
}

RULES = ["Type", "Amount", "Position", "Background", "Density"]

# 复合方位（含空格/连字符）优先于单词匹配
POS_COMPOUND = sorted([k for k in LEXICON["position"] if (" " in k or "-" in k)],
                      key=len, reverse=True)
# 方位词全集，background 抽取时要绕开
POS_WORDS = set(LEXICON["position"].keys()) | set(LEXICON["position"].values())
# 背景里要忽略的词（指代图像本身，不是地物）
BG_STOPWORDS = {"image", "images", "picture", "pictures", "scene", "scenes",
                "patch", "patches", "photo", "photograph", "figure", "view",
                "views", "same", "there", "here", "it", "its"}
# 数量槽前的可跳过修饰词
AMOUNT_SKIP = {"new", "newly", "tall", "small", "large", "big", "little", "old",
               "modern", "residential", "industrial", "white", "red", "green",
               "gray", "grey", "blue", "same", "such", "other", "additional",
               "extra", "another", "nearby", "adjacent", "of", "in", "the",
               "and", "with", "very", "commercial", "abandoned", "dilapidated"}

BG_PREP = (r"along|among|amongst|near|next to|beside|around|within|inside|outside|"
           r"between|adjacent to|surrounded by|surrounding|across|throughout|"
           r"on|in|at|by|of|with|behind|besides|beyond|alongside")

# --------------------------------------------------------------------------- #
#  2. 无变化（no-change）检测
# --------------------------------------------------------------------------- #

# 整句级：含这些就直接判 no-change
NO_CHANGE_PATTERNS = [
    r"\bno\s+change", r"\bno\s+changes", r"\bnot\s+changed", r"\bunchanged\b",
    r"\bnothing\s+(?:has\s+)?(?:changed|appeared|disappeared)",
    r"\bremain(?:s|ed|ing)?\s+(?:the\s+)?(?:same|unchanged)",
    r"\bno\s+(?:visible\s+)?difference", r"\bidentical\b",
    r"\bwithout\s+any\s+(?:change|new\s+building)",
]
# 片段级：把 "no building has appeared ..." 这类整段剔除
NEG_SPAN_PATTERNS = [
    r"\bno\s+(?:new\s+|other\s+|more\s+)?(?:buildings?|houses?|structures?|changes?)"
    r"(?:\s+(?:has|have|is|are|was|were|had))?"
    r"(?:\s+(?:been|not))?"
    r"(?:\s+(?:appeared|disappeared|appearing|disappearing|emerged|vanished|"
    r"added|removed|built|constructed|demolished|destroyed|changed|"
    r"increased|decreased|found|detected|observed))?",
    r"\bno\s+(?:visible\s+)?(?:appearance|disappearance|construction|demolition)s?\b",
    r"\bnothing\s+(?:has\s+)?(?:appeared|disappeared|changed)\b",
    r"\bnot\s+(?:any\s+)?(?:appeared|disappeared|changed)\b",
]

VERB_RE = re.compile(r"\b(" + "|".join(sorted(set(LEXICON["type"].keys()), key=len, reverse=True)) + r")")


def strip_negative(text):
    """剔除否定片段，返回 (剩余文本, 是否剔除过)"""
    residual = text
    for pat in NEG_SPAN_PATTERNS:
        residual = re.sub(pat, " ", residual)
    residual = re.sub(r"[,\s]*\band\b[,\s]*$", " ", residual).strip()
    residual = re.sub(r"\s+", " ", residual).strip(" ,.;:")
    return residual


def is_no_change(norm_text):
    """返回 (是否无变化, 剔除否定片段后的剩余句子)"""
    for pat in NO_CHANGE_PATTERNS:
        if re.search(pat, norm_text):
            return True, ""
    residual = strip_negative(norm_text)
    residual = re.sub(r"\s+", " ", residual).strip()
    if not VERB_RE.search(residual):
        # 剩余没有任何变化信号 -> 整句无变化
        if re.search(r"\bno\b|\bnot\b|\bnothing\b|\bnone\b", norm_text) or not norm_text:
            return True, ""
    return False, residual


# --------------------------------------------------------------------------- #
#  3. 归一化 & 槽位抽取
# --------------------------------------------------------------------------- #

_PUNCT = re.compile(r"[^\w\s\-/]")


def normalize(text):
    t = text.lower().strip()
    t = _PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def tokenize(text):
    return [w for w in re.split(r"[\s/]+", text) if w]


def extract_type(toks):
    for t in toks:
        if t in LEXICON["type"]:
            return LEXICON["type"][t]
    for t in toks:
        for k, v in LEXICON["type"].items():
            if len(k) >= 4 and t.startswith(k):
                return v
    return None


def extract_amount(toks):
    target = {"building", "buildings", "house", "houses", "structure", "structures",
              "hut", "huts", "construction", "constructions"}
    for i, t in enumerate(toks):
        if t in target:
            j, guard = i - 1, 0
            while j >= 0 and guard < 4:
                w = toks[j]
                if w in LEXICON["amount"]:
                    return LEXICON["amount"][w]
                if w in LEXICON["density"] or w in LEXICON["position"] or w in AMOUNT_SKIP:
                    j -= 1
                    guard += 1
                    continue
                break
            # "buildings appeared" 前面啥也没有 -> 复数按 several 处理
            return None
    return None


def extract_position(norm_text):
    found, txt = set(), norm_text
    for comp in POS_COMPOUND:
        # "upper-left" / "top left" 都应匹配 "upper left" / "top-left"
        pat = (r"(?<![a-z])" + comp.replace("-", r"[\s\-]+").replace(" ", r"[\s\-]+")
               + r"(?![a-z])")
        if re.search(pat, txt):
            found.add(LEXICON["position"][comp])
            txt = re.sub(pat, " ", txt)
    for t in re.split(r"[\s\-]+", txt):
        if t in LEXICON["position"]:
            found.add(LEXICON["position"][t])
    for a, b in (("top", "left"), ("top", "right"),
                 ("bottom", "left"), ("bottom", "right")):
        comp = "%s-%s" % (a, b)
        if comp in found:
            found.discard(a)
            found.discard(b)
    # corner / edge 只在没有更具体方位时才保留
    specific = found - {"corner", "edge"}
    if specific:
        found = specific
    return sorted(found) if found else None


def extract_background(norm_text):
    pat = re.compile(
        r"\b(?:" + BG_PREP + r")\s+"
        r"(?:(?:the|a|an|some|several|many|its|nearby|adjacent|surrounding|this|that)\s+)*"
        r"((?:[a-z\-]+\s+){0,3}?[a-z\-]+)"
        r"(?=[.,;:]|$|\s+(?:of|in|near|and|with|that|which|where|the|is|are|was|were|to)\b)"
    )
    found = set()
    for m in pat.finditer(norm_text):
        words = m.group(1).split()
        picked = None
        for w in reversed(words):
            if w in POS_WORDS or w in BG_STOPWORDS or w in LEXICON["density"]:
                continue
            picked = LEXICON["background"].get(w, w)
            break
        if picked is None:
            continue
        if picked in ("area",) and len(words) == 1 and re.search(
                r"\b(?:of|in)\s+the\s+area\b", norm_text) and not re.search(
                r"\b(?:along|among|near|next to|beside|around|within)\s+.*area", norm_text):
            # "in the area" 多为泛指，不算有效背景；带具体介词的才保留
            continue
        found.add(picked)
    return sorted(found) if found else None


def extract_density(toks):
    for t in toks:
        if t in LEXICON["density"]:
            return LEXICON["density"][t]
    for t in toks:
        for k, v in LEXICON["density"].items():
            if len(k) >= 5 and t.startswith(k):
                return v
    return None


def extract_all(sentence):
    norm = normalize(sentence)
    no_change, residual = is_no_change(norm)
    if no_change:
        return OrderedDict([("Type", "none"), ("Amount", None), ("Position", None),
                            ("Background", None), ("Density", None)])
    toks = tokenize(residual)
    return OrderedDict([
        ("Type", extract_type(toks)),
        ("Amount", extract_amount(toks)),
        ("Position", extract_position(residual)),
        ("Background", extract_background(residual)),
        ("Density", extract_density(toks)),
    ])


def load_lexicon(path):
    with open(path, "r", encoding="utf-8") as f:
        ext = json.load(f)
    for k, v in ext.items():
        LEXICON[k] = v if k not in LEXICON else dict(LEXICON[k], **v)
    global POS_COMPOUND, POS_WORDS
    POS_COMPOUND = sorted([k for k in LEXICON["position"] if (" " in k or "-" in k)],
                          key=len, reverse=True)
    POS_WORDS = set(LEXICON["position"].keys()) | set(LEXICON["position"].values())


# --------------------------------------------------------------------------- #
#  4. 读文件：扫描目录内所有 txt，抽 pred_caption / ref_caption
# --------------------------------------------------------------------------- #

KEY_PRED = {"pred_caption", "pred", "prediction", "predicted_caption", "hyp",
            "hypothesis", "candidate", "gen_caption", "output"}
KEY_REF = {"ref_caption", "gt_caption", "ref", "gt", "ground_truth", "target",
           "label", "reference", "gold"}
KEY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_ ]{0,30})\s*[:：]\s?(.*)$")


def parse_caption_file(path):
    """返回 [(pred, ref), ...]，一个文件里可含多对"""
    preds, refs, cur_key, cur_val = [], [], None, []

    def flush():
        if cur_key is None:
            return
        val = " ".join(cur_val).strip()
        if cur_key in KEY_PRED:
            preds.append(val)
        elif cur_key in KEY_REF:
            refs.append(val)

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            m = KEY_RE.match(line)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                if key in KEY_PRED or key in KEY_REF:      # 新的一行键
                    flush()
                    cur_key, cur_val = key, [m.group(2).strip()]
                    continue
            if cur_key is not None:                        # 续行
                cur_val.append(line.strip())
    flush()
    return preds, refs


def scan_dir(data_dir):
    """返回 [(sample_id, pred, ref), ...]"""
    files = sorted(
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.lower().endswith(".txt")
    )
    if not files:
        raise SystemExit("[错误] %s 下没有找到任何 .txt 文件" % os.path.abspath(data_dir))
    out, skipped = [], []
    for fp in files:
        preds, refs = parse_caption_file(fp)
        if not preds and not refs:
            skipped.append((os.path.basename(fp), "未识别到 pred_caption / ref_caption"))
            continue
        if len(preds) != len(refs):
            skipped.append((os.path.basename(fp),
                            "pred %d 条 / ref %d 条，数量不匹配" % (len(preds), len(refs))))
            continue
        stem = os.path.splitext(os.path.basename(fp))[0]
        for i, (p, r) in enumerate(zip(preds, refs)):
            sid = stem if len(preds) == 1 else "%s#%d" % (stem, i)
            out.append((sid, p, r))
    return out, skipped, len(files)


def scan_two_dirs(pred_dir, gt_dir):
    def index(d):
        m = {}
        for f in os.listdir(d):
            if f.lower().endswith(".txt"):
                m[os.path.splitext(f)[0]] = os.path.join(d, f)
        return m
    pm, gm = index(pred_dir), index(gt_dir)
    common = sorted(set(pm) & set(gm))
    if not common:
        raise SystemExit("[错误] pred-dir 与 gt-dir 之间没有同名的 .txt 文件")
    out, skipped = [], []
    for k in common:
        p, _ = parse_caption_file(pm[k])
        _, r = parse_caption_file(gm[k])
        flat_p = [x for x in p]
        flat_r = [x for x in r]
        if not p or not r:
            skipped.append((k, "缺少 pred_caption 或 ref_caption"))
            continue
        n = min(len(p), len(r))
        for i in range(n):
            out.append((k if n == 1 else "%s#%d" % (k, i), p[i], r[i]))
    return out, skipped, len(common)


# --------------------------------------------------------------------------- #
#  5. 比对 & 输出
# --------------------------------------------------------------------------- #

def slot_equal(rule, gt, pred, partial_position=False):
    if gt is None:
        return None
    if pred is None:
        return False
    if rule in ("Type", "Amount", "Density"):
        return gt == pred
    if rule == "Position":
        g, p = set(gt), set(pred)
        if g == p:
            return True
        return len(g & p) > 0 if partial_position else False
    if rule == "Background":
        g, p = set(gt), set(pred)
        if g == p:
            return True
        return 0.5 if (g & p) else False
    return False


def disp_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def print_table(rows, headers, aligns=None):
    all_rows = [headers] + rows
    widths = [max(disp_width(r[i]) for r in all_rows) for i in range(len(headers))]
    def render(r):
        cells = []
        for i, x in enumerate(r):
            s = str(x)
            pad = widths[i] - disp_width(s)
            cells.append(s + " " * pad if (aligns and i < len(aligns) and aligns[i] == "l")
                         else " " * pad + s)
        return "  ".join(cells)
    print(render(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(render(r))


def fmt_cell(v):
    if v is None:
        return "-"
    if isinstance(v, list):
        return "|".join(v)
    return str(v)


def main():
    ap = argparse.ArgumentParser(
        description="变化描述五规则维度评测（扫描目录下所有 txt 的 pred_caption / ref_caption）")
    ap.add_argument("--data-dir", default=".", help="含 pred_caption / ref_caption 的 txt 目录")
    ap.add_argument("--pred-dir", default="", help="预测单独目录（与 --gt-dir 配对使用）")
    ap.add_argument("--gt-dir", default="", help="GT 单独目录（与 --pred-dir 配对使用）")
    ap.add_argument("--out-csv", default="", help="逐条明细 CSV")
    ap.add_argument("--summary-json", default="", help="汇总 JSON")
    ap.add_argument("--dump-unmatched", default="", help="导出一个槽位都没解析出来的 GT 句")
    ap.add_argument("--lexicon", default="", help="外部词表 JSON，合并覆盖默认词表")
    ap.add_argument("--bg-stopwords", default="", help="追加背景停用词，逗号分隔")
    ap.add_argument("--partial-position", action="store_true",
                    help="Position 命中任一 GT 方位词即算对（默认要求集合一致）")
    ap.add_argument("--show-errors", type=int, default=3, help="每维度打印错例条数，0 关闭")
    args = ap.parse_args()

    if args.lexicon:
        load_lexicon(args.lexicon)
    if args.bg_stopwords:
        BG_STOPWORDS.update(w.strip().lower() for w in args.bg_stopwords.split(",") if w.strip())

    if args.pred_dir and args.gt_dir:
        samples, skipped, n_file = scan_two_dirs(args.pred_dir, args.gt_dir)
        src = "pred-dir=%s  gt-dir=%s" % (args.pred_dir, args.gt_dir)
    else:
        samples, skipped, n_file = scan_dir(args.data_dir)
        src = "data-dir=%s" % args.data_dir

    if not samples:
        raise SystemExit("[错误] 没有解析到任何 (pred_caption, ref_caption) 样本对")

    stats = {r: {"n_gt": 0, "n_pred": 0, "hit": 0, "partial": 0, "score": 0.0} for r in RULES}
    records, unmatched = [], []
    n_all = len(samples)
    n_perfect = 0
    gt_nochange = pred_nochange = nochange_hit = 0
    type_change_n = type_change_hit = 0

    for sid, p, g in samples:
        gs, ps = extract_all(g), extract_all(p)
        rec = {"id": sid, "gt": g, "pred": p, "flags": {}}
        all_ok = True
        for r in RULES:
            gv, pv = gs[r], ps[r]
            if gv is not None:
                stats[r]["n_gt"] += 1
            if pv is not None:
                stats[r]["n_pred"] += 1
            eq = slot_equal(r, gv, pv, args.partial_position)
            if eq is None:
                rec["flags"][r] = "N/A"
                continue
            if eq is True:
                stats[r]["hit"] += 1
                stats[r]["score"] += 1.0
                rec["flags"][r] = "OK"
            elif eq == 0.5:
                stats[r]["partial"] += 1
                stats[r]["score"] += 0.5
                rec["flags"][r] = "PART"
                all_ok = False
            else:
                rec["flags"][r] = "WRONG"
                all_ok = False
            rec["gt_" + r] = fmt_cell(gv)
            rec["pred_" + r] = fmt_cell(pv)
        # no-change 单独统计
        g_nc = (gs["Type"] == "none")
        p_nc = (ps["Type"] == "none")
        gt_nochange += int(g_nc)
        pred_nochange += int(p_nc)
        nochange_hit += int(g_nc and p_nc)
        if not g_nc:
            type_change_n += 1
            type_change_hit += int(gs["Type"] == ps["Type"])
        rec["all_ok"] = all_ok
        rec["gt_nochange"] = g_nc
        rec["pred_nochange"] = p_nc
        n_perfect += int(all_ok)
        records.append(rec)

        if not g_nc and all(v is None for v in gs.values()) and g.strip():
            unmatched.append("\t".join([sid, g]))

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print("变化描述五规则维度评测结果")
    print("=" * 78)
    print("来源：%s    扫描文件 %d 个    有效样本 %d 条" % (src, n_file, n_all))
    if skipped:
        print("跳过 %d 个文件：" % len(skipped))
        for name, why in skipped[:10]:
            print("    - %s：%s" % (name, why))
        if len(skipped) > 10:
            print("    ... 另有 %d 个" % (len(skipped) - 10))
    print("口径：分母 = GT 中出现该维度的样本数；Background 部分命中记 0.5\n")

    rows = []
    for r in RULES:
        st = stats[r]
        n = st["n_gt"]
        rows.append([
            r, n, st["n_pred"], st["hit"], st["partial"],
            "%.2f%%" % (st["score"] / n * 100) if n else "-",
            "%.2f%%" % (st["hit"] / n * 100) if n else "-",
            "%.2f%%" % (st["n_pred"] / n * 100) if n else "-",
        ])
    rows.append(["— 全维度全对 —", n_all, "", n_perfect, "",
                 "%.2f%%" % (n_perfect / n_all * 100), "", ""])
    print_table(rows,
                ["Rule", "GT出现数", "Pred出现数", "完全命中", "部分命中",
                 "准确率", "严格准确率", "Coverage"],
                aligns=["l", "r", "r", "r", "r", "r", "r", "r"])

    macro = [stats[r]["score"] / stats[r]["n_gt"] for r in RULES if stats[r]["n_gt"]]
    print("\nMacro 平均准确率（五维平均）：%.2f%%"
          % (sum(macro) / len(macro) * 100 if macro else 0))

    print("\n" + "-" * 78)
    print("无变化（no-change）样本")
    print("-" * 78)
    print("GT 无变化样本：%d / %d（%.2f%%）    预测无变化：%d    判对：%d"
          % (gt_nochange, n_all, gt_nochange / n_all * 100, pred_nochange, nochange_hit))
    if gt_nochange:
        rec_ = nochange_hit / gt_nochange * 100
        print("无变化召回（GT 无变化时预测也说无变化）：%.2f%%" % rec_)
    if pred_nochange:
        print("无变化精确率（预测说无变化时确实无变化）：%.2f%%"
              % (nochange_hit / pred_nochange * 100))
    if type_change_n:
        print("只看有变化的样本，Type 准确率：%.2f%%（%d/%d）"
              % (type_change_hit / type_change_n * 100, type_change_hit, type_change_n))
    print("\n注：准确率含 Background 部分命中(0.5)；严格准确率只算完全一致；")
    print("    Coverage 衡量'该维度有没有生成出来'，低于 100% 说明模型漏生成。")

    # ---------------- 错例 ----------------
    if args.show_errors:
        print("\n" + "-" * 78)
        print("典型错例（每维度最多 %d 条）" % args.show_errors)
        print("-" * 78)
        for r in RULES:
            bad = [x for x in records if x["flags"].get(r) in ("WRONG", "PART")]
            if not bad:
                continue
            print("\n[%s]  错 %d 条，示例：" % (r, len(bad)))
            for x in bad[:args.show_errors]:
                print("  id=%s" % x["id"])
                print("    GT  : %s" % x["gt"])
                print("    PRED: %s" % x["pred"])
                print("    -> %s: GT=%s  PRED=%s" % (r, x.get("gt_" + r, "-"),
                                                     x.get("pred_" + r, "-")))

    # ---------------- 落盘 ----------------
    if args.out_csv:
        cols = (["id", "all_ok", "gt_nochange", "pred_nochange"]
                + sum([["gt_" + r, "pred_" + r, "flag_" + r] for r in RULES], [])
                + ["gt_text", "pred_text"])
        with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for x in records:
                row = [x["id"], int(x["all_ok"]), int(x["gt_nochange"]), int(x["pred_nochange"])]
                for r in RULES:
                    row += [x.get("gt_" + r, ""), x.get("pred_" + r, ""), x["flags"][r]]
                row += [x["gt"], x["pred"]]
                w.writerow(row)
        print("\n明细已写出：%s" % os.path.abspath(args.out_csv))

    if args.summary_json:
        summary = {
            "n_samples": n_all,
            "n_all_dimensions_correct": n_perfect,
            "macro_accuracy": (sum(macro) / len(macro) if macro else 0),
            "no_change": {
                "n_gt": gt_nochange, "n_pred": pred_nochange,
                "n_correct": nochange_hit,
                "recall": (nochange_hit / gt_nochange) if gt_nochange else None,
                "precision": (nochange_hit / pred_nochange) if pred_nochange else None,
            },
            "type_on_changed_only": {
                "n": type_change_n, "hit": type_change_hit,
                "accuracy": (type_change_hit / type_change_n) if type_change_n else None,
            },
            "rules": {r: {
                "n_gt": stats[r]["n_gt"], "n_pred": stats[r]["n_pred"],
                "n_hit": stats[r]["hit"], "n_partial": stats[r]["partial"],
                "accuracy": (stats[r]["score"] / stats[r]["n_gt"]) if stats[r]["n_gt"] else None,
                "strict_accuracy": (stats[r]["hit"] / stats[r]["n_gt"]) if stats[r]["n_gt"] else None,
                "coverage": (1 if stats[r]["n_pred"] > stats[r]["n_gt"] else (stats[r]["n_pred"] / stats[r]["n_gt"])) if stats[r]["n_gt"] else None,
            } for r in RULES},
        }
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print("汇总已写出：%s" % os.path.abspath(args.summary_json))

    if args.dump_unmatched:
        with open(args.dump_unmatched, "w", encoding="utf-8") as f:
            f.write("\n".join(unmatched))
        print("无法解析的 GT 句（%d 条）已写出：%s  <- 据此补词表"
              % (len(unmatched), os.path.abspath(args.dump_unmatched)))
    elif unmatched:
        print("\n提示：有 %d 条 GT 句一个槽位都没解析出来，加 --dump-unmatched 导出补词表。"
              % len(unmatched))
    print()


if __name__ == "__main__":
    main()
