"""
VAD pipeline - Bước 1 & 2
  #1 Scan folder, ghép cặp file model <-> ground truth
  #2 Resample hop: gộp các window chồng lấn (0.96s / hop 0.08s) thành các bin 0.08s
     rồi lưu vào output_after_overlap/
  #3 Threshold -> segment speech -> pad trước 100ms / sau 120ms -> merge gap < 500ms
     -> bỏ segment < 100ms, lưu vào output_clean_data/
  #4 Rebin: chia timeline thành bin 0.5s, bin có >= 20% thời lượng là speech -> speech,
     lưu vào output_rebin/
  #5 Đánh giá: so sánh kết quả rebin với GT, xuất Excel (summary, lỗi, histogram, ROC)

Cấu trúc thư mục:
  Groundtruth/<category>/groundtruth/<filename>.txt   (vd: Groundtruth/Asm/groundtruth/Gt_asm_1.txt)
  Model/<category>_<index>.txt                         (vd: Model/Asm_1.txt)
"""

import argparse
from pathlib import Path

import numpy as np

WINDOW = 0.96
HOP = 0.08

THRESHOLD = 0.5      # score >= threshold -> speech
PAD_BEFORE = 0.10    # 100 ms
PAD_AFTER = 0.12     # 120 ms
MERGE_GAP = 0.50     # gộp nếu khoảng trống < 500 ms
MIN_DUR = 0.10       # bỏ segment < 100 ms
REBIN_SIZE = 0.50    # bin 0.5 s (khớp GT)
MIN_SPEECH_RATIO = 0.20  # bin có >= 20% speech -> speech
EPS = 1e-9           # chống lỗi so sánh float


# ---------------------------------------------------------------------------
# #1  SCAN & GHÉP CẶP
# ---------------------------------------------------------------------------
def _find_dir_ci(parent: Path, name: str):
    """Tìm thư mục con theo tên, không phân biệt hoa/thường."""
    if not parent.is_dir():
        return None
    for d in parent.iterdir():
        if d.is_dir() and d.name.lower() == name.lower():
            return d
    return None


def find_pairs(model_root: Path, gt_root: Path):
    """
    Trả về list dict: {category, index, model, gt}
    Model 'Asm_1.txt' -> category 'Asm', index '1'
    GT tương ứng: Groundtruth/Asm/groundtruth/*asm_1.txt (không phân biệt hoa/thường)
    """
    pairs, missing = [], []

    for mf in sorted(model_root.glob("*.txt")):
        if "_" not in mf.stem:
            missing.append((mf, "tên file không đúng dạng category_index"))
            continue
        category, index = mf.stem.rsplit("_", 1)

        cat_dir = _find_dir_ci(gt_root, category)
        gt_dir = _find_dir_ci(cat_dir, "groundtruth") if cat_dir else None
        if gt_dir is None:
            missing.append((mf, f"không có thư mục {gt_root}/{category}/groundtruth"))
            continue

        key = f"{category}_{index}".lower()
        candidates = [
            g for g in gt_dir.glob("*.txt")
            if g.stem.lower() == key
            or g.stem.lower() == f"gt_{key}"
            or g.stem.lower().endswith("_" + key)
        ]
        if not candidates:
            missing.append((mf, f"không tìm thấy GT cho '{key}' trong {gt_dir}"))
            continue
        if len(candidates) > 1:
            # ưu tiên khớp chính xác gt_<key>
            exact = [g for g in candidates if g.stem.lower() == f"gt_{key}"]
            candidates = exact or sorted(candidates)

        pairs.append({"category": category, "index": index,
                      "model": mf, "gt": candidates[0]})

    return pairs, missing


# ---------------------------------------------------------------------------
# #2  XỬ LÝ OVERLAP -> BIN HOP
# ---------------------------------------------------------------------------
def load_model_file(path: Path):
    """Đọc file model: start  end  score (tab/space). Bỏ qua dòng lỗi/rỗng."""
    data = np.loadtxt(path, ndmin=2, usecols=(0, 1, 2))
    return data[:, 0], data[:, 1], data[:, 2]


def resolve_overlap(starts, ends, scores, hop=HOP, method="mean"):
    """
    Chia trục thời gian thành các bin dài `hop`. Mỗi bin nhận score gộp từ
    tất cả window phủ lên nó.

    method="mean": dùng difference array + cumsum -> O(N + B), không cần vòng lặp
                   lồng nhau. Mỗi window [s, e) cộng score vào mọi bin s..e-1.
    method="max" : lấy score lớn nhất trong các window phủ bin (O(N * k), k = window/hop = 12).

    Trả về (bin_start, bin_end, bin_score, bin_count)
    """
    # quy thời gian về chỉ số bin (round để tránh lỗi float 0.08*3 = 0.24000000000000002)
    s_idx = np.rint(starts / hop).astype(np.int64)
    e_idx = np.rint(ends / hop).astype(np.int64)
    n_bins = int(e_idx.max())

    if method == "mean":
        diff_sum = np.zeros(n_bins + 1)
        diff_cnt = np.zeros(n_bins + 1)
        np.add.at(diff_sum, s_idx, scores)
        np.add.at(diff_sum, e_idx, -scores)
        np.add.at(diff_cnt, s_idx, 1)
        np.add.at(diff_cnt, e_idx, -1)
        total = np.cumsum(diff_sum)[:n_bins]
        count = np.cumsum(diff_cnt)[:n_bins]
        count = np.rint(count).astype(np.int64)
        with np.errstate(invalid="ignore", divide="ignore"):
            bin_score = np.where(count > 0, total / np.maximum(count, 1), np.nan)

    elif method == "max":
        bin_score = np.full(n_bins, -np.inf)
        count = np.zeros(n_bins, dtype=np.int64)
        for s, e, sc in zip(s_idx, e_idx, scores):
            bin_score[s:e] = np.maximum(bin_score[s:e], sc)
            count[s:e] += 1
        bin_score[count == 0] = np.nan
    else:
        raise ValueError(f"method không hợp lệ: {method}")

    bin_idx = np.arange(n_bins)
    return bin_idx * hop, (bin_idx + 1) * hop, bin_score, count


def save_bins(path: Path, b_start, b_end, b_score, keep_empty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, e, sc in zip(b_start, b_end, b_score):
            if np.isnan(sc):
                if not keep_empty:
                    continue
                f.write(f"{s:.2f}\t{e:.2f}\tnan\n")
            else:
                f.write(f"{s:.2f}\t{e:.2f}\t{sc:.6f}\n")


# ---------------------------------------------------------------------------
# #3  PADDING -> MERGE -> BỎ SEGMENT NGẮN
# ---------------------------------------------------------------------------
def bins_to_segments(b_start, b_end, b_score, threshold=THRESHOLD):
    """Gom các bin liên tiếp có score >= threshold thành segment speech [start, end)."""
    is_speech = np.nan_to_num(b_score, nan=-1.0) >= threshold
    # tìm điểm bắt đầu/kết thúc của các chuỗi True liên tiếp (vectorized)
    edges = np.diff(np.concatenate(([0], is_speech.astype(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(float(b_start[i]), float(b_end[j])) for i, j in zip(starts, ends)]


def pad_segments(segs, before=PAD_BEFORE, after=PAD_AFTER, t_min=0.0, t_max=None):
    out = []
    for s, e in segs:
        s = max(t_min, s - before)
        e = e + after if t_max is None else min(t_max, e + after)
        out.append((s, e))
    return out


def merge_segments(segs, max_gap=MERGE_GAP):
    """Gộp segment chồng lấn hoặc cách nhau < max_gap. O(N log N) do sort."""
    if not segs:
        return []
    segs = sorted(segs)
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] < max_gap - EPS:   # gap < 500ms (gap âm = chồng lấn)
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(m) for m in merged]


def remove_short(segs, min_dur=MIN_DUR):
    return [(s, e) for s, e in segs if (e - s) >= min_dur - EPS]


def clean_segments(b_start, b_end, b_score, threshold=THRESHOLD,
                   pad_before=PAD_BEFORE, pad_after=PAD_AFTER,
                   merge_gap=MERGE_GAP, min_dur=MIN_DUR,
                   t_max=None, drop_before_pad=False):
    """Tất cả thời gian tính bằng giây."""
    segs = bins_to_segments(b_start, b_end, b_score, threshold)
    if drop_before_pad:
        segs = remove_short(segs, min_dur)
    segs = pad_segments(segs, before=pad_before, after=pad_after, t_max=t_max)
    segs = merge_segments(segs, max_gap=merge_gap)
    segs = remove_short(segs, min_dur)
    return segs


def gt_duration(gt_path: Path):
    """Lấy thời điểm kết thúc lớn nhất trong file GT (để kẹp padding)."""
    try:
        return float(np.loadtxt(gt_path, ndmin=2, usecols=(1,)).max())
    except Exception:
        return None


def save_segments(path: Path, segs, t_max, full_timeline=True):
    """
    full_timeline=True: ghi cả speech và non-speech phủ kín 0 -> t_max (giống định dạng GT)
    full_timeline=False: chỉ ghi các segment speech
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, cur = [], 0.0
    for s, e in segs:
        if full_timeline and s - cur > EPS:
            rows.append((cur, s, "non-speech"))
        rows.append((s, e, "speech"))
        cur = e
    if full_timeline and t_max is not None and t_max - cur > EPS:
        rows.append((cur, t_max, "non-speech"))
    with open(path, "w", encoding="utf-8") as f:
        for s, e, lab in rows:
            f.write(f"{s:.2f}\t{e:.2f}\t{lab}\n")


# ---------------------------------------------------------------------------
# #4  REBIN 0.5s
# ---------------------------------------------------------------------------
def rebin_segments(segs, t_max, bin_size=REBIN_SIZE, min_ratio=MIN_SPEECH_RATIO):
    """
    Chia [0, t_max) thành các bin dài bin_size (bin cuối có thể ngắn hơn).
    Với mỗi segment speech, chỉ duyệt các bin mà nó chạm tới và cộng phần giao
    -> O(S + B), không so từng segment với từng bin.
    Trả về (bin_start, bin_end, speech_ratio, is_speech)
    """
    n_bins = int(np.ceil(t_max / bin_size - EPS))
    b_start = np.arange(n_bins) * bin_size
    b_end = np.minimum(b_start + bin_size, t_max)
    speech_dur = np.zeros(n_bins)

    for s, e in segs:
        s, e = max(0.0, s), min(t_max, e)
        if e - s <= EPS:
            continue
        i0 = int(np.floor(s / bin_size + EPS))
        i1 = min(n_bins - 1, int(np.ceil(e / bin_size - EPS)) - 1)
        for i in range(i0, i1 + 1):
            speech_dur[i] += max(0.0, min(e, b_end[i]) - max(s, b_start[i]))

    ratio = speech_dur / (b_end - b_start)
    is_speech = ratio >= min_ratio - EPS
    return b_start, b_end, ratio, is_speech


def save_rebin(path: Path, b_start, b_end, ratio, is_speech, with_ratio=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    # định dạng giống GT (0.0, 0.5 ...); nếu bin lẻ (vd 250ms) thì tăng số chữ số
    edges = np.concatenate((b_start, b_end))
    nd = next(d for d in (1, 2, 3, 6) if np.allclose(edges, np.round(edges, d), atol=1e-6))
    with open(path, "w", encoding="utf-8") as f:
        for s, e, r, sp in zip(b_start, b_end, ratio, is_speech):
            lab = "speech" if sp else "non-speech"
            extra = f"\t{r:.3f}" if with_ratio else ""
            f.write(f"{s:.{nd}f}\t{e:.{nd}f}\t{lab}{extra}\n")


# ---------------------------------------------------------------------------
# #5  ĐÁNH GIÁ -> EXCEL
# ---------------------------------------------------------------------------
ERROR_TYPES = ["boundary_onset", "boundary_offset", "extension", "spurious",
               "truncation", "gap", "missed"]
ERROR_DESC = {
    "boundary_onset":  "Lỗi (FP/FN) nằm trong vùng dung sai quanh điểm BẮT ĐẦU speech của GT",
    "boundary_offset": "Lỗi (FP/FN) nằm trong vùng dung sai quanh điểm KẾT THÚC speech của GT",
    "extension":       "FP nối liền speech GT nhưng vượt quá vùng dung sai (dự đoán kéo dài quá xa)",
    "spurious":        "FP đứng riêng, không nối với speech GT nào (báo nhầm hoàn toàn)",
    "truncation":      "FN ở đầu/cuối segment GT, vượt quá vùng dung sai (cắt cụt sâu)",
    "gap":             "FN nằm giữa segment GT, 2 bên vẫn bắt được speech (segment bị đứt)",
    "missed":          "Toàn bộ segment speech GT bị bỏ sót",
}
DEPTH_GROUPS = ["FP nối speech (boundary+extension)", "FN ở biên (boundary+truncation)",
                "Spurious", "Gap / Missed"]


def load_gt(gt_path: Path):
    """GT: start  end  label -> (start, end, is_speech)."""
    st, en, lab = [], [], []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                st.append(float(parts[0])); en.append(float(parts[1]))
            except ValueError:
                continue
            lab.append(parts[2].strip().lower() == "speech")
    return np.array(st), np.array(en), np.array(lab, dtype=bool)


def _runs(mask):
    """Trả về list (a, b) các đoạn True liên tiếp (chỉ số bao gồm cả 2 đầu)."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0] - 1))


def score_on_grid(g_start, g_end, bins):
    """Score trung bình (có trọng số theo phần giao) của các bin hop trong mỗi bin GT."""
    bs, be, bsc = bins
    out = np.full(len(g_start), np.nan)
    for k, (s, e) in enumerate(zip(g_start, g_end)):
        i0 = np.searchsorted(be, s + EPS, side="left")
        i1 = np.searchsorted(bs, e - EPS, side="left")
        if i1 <= i0:
            continue
        w = np.minimum(be[i0:i1], e) - np.maximum(bs[i0:i1], s)
        v = bsc[i0:i1]
        ok = ~np.isnan(v) & (w > 0)
        if ok.any():
            out[k] = np.sum(v[ok] * w[ok]) / np.sum(w[ok])
    return out


def classify_errors(gt, pred, bin_size, tol=1):
    """
    Phân loại lỗi từng bin + gom lỗi thành run để đo độ sâu.
    tol = số bin quanh điểm chuyển GT được coi là vùng boundary.
    """
    n = len(gt)
    etype = np.array([""] * n, dtype=object)
    depth = np.zeros(n)
    runs = []   # (outcome, type, group, side, a, b)

    fp = pred & ~gt
    for a, b in _runs(fp):
        left = a > 0 and gt[a - 1]            # speech GT ngay trước -> kéo dài offset
        right = b < n - 1 and gt[b + 1]       # speech GT ngay sau  -> bắt đầu sớm (onset)
        if not left and not right:
            etype[a:b + 1] = "spurious"
            runs.append(("FP", "spurious", DEPTH_GROUPS[2], "-", a, b))
        else:
            for i in range(a, b + 1):
                dl = i - a + 1 if left else np.inf
                dr = b - i + 1 if right else np.inf
                if min(dl, dr) <= tol:
                    etype[i] = "boundary_onset" if dr < dl or (dr == dl) else "boundary_offset"
                else:
                    etype[i] = "extension"
            side = "both" if left and right else ("offset" if left else "onset")
            runs.append(("FP", "extension", DEPTH_GROUPS[0], side, a, b))
        depth[a:b + 1] = (b - a + 1) * bin_size

    for s, e in _runs(gt):                    # từng segment speech GT
        seg_fn = ~pred[s:e + 1]
        if seg_fn.all():
            etype[s:e + 1] = "missed"
            depth[s:e + 1] = (e - s + 1) * bin_size
            runs.append(("FN", "missed", DEPTH_GROUPS[3], "-", s, e))
            continue
        for ra, rb in _runs(seg_fn):
            a, b = s + ra, s + rb
            at_start, at_end = a == s, b == e
            for i in range(a, b + 1):
                if at_start and (i - s + 1) <= tol:
                    etype[i] = "boundary_onset"
                elif at_end and (e - i + 1) <= tol:
                    etype[i] = "boundary_offset"
                elif at_start or at_end:
                    etype[i] = "truncation"
                else:
                    etype[i] = "gap"
            if at_start or at_end:
                runs.append(("FN", "truncation", DEPTH_GROUPS[1],
                             "onset" if at_start else "offset", a, b))
            else:
                runs.append(("FN", "gap", DEPTH_GROUPS[3], "-", a, b))
            depth[a:b + 1] = (b - a + 1) * bin_size
    return etype, depth, runs


def segment_errors(g_start, g_end, gt, pred_segs):
    """
    Với mỗi segment speech GT, tìm các segment dự đoán (#3) chồng lấn:
      onset_err  = pred_onset  - gt_onset   (<0: bắt đầu sớm, >0: trễ)
      offset_err = pred_offset - gt_offset  (<0: kết thúc sớm, >0: kéo dài)
    """
    rows = []
    for a, b in _runs(gt):
        gs, ge = float(g_start[a]), float(g_end[b])
        ov = [(s, e) for s, e in pred_segs if s < ge - EPS and e > gs + EPS]
        if ov:
            ps, pe = min(s for s, _ in ov), max(e for _, e in ov)
            rows.append((gs, ge, "yes", ps, pe, round(ps - gs, 3), round(pe - ge, 3)))
        else:
            rows.append((gs, ge, "no", None, None, None, None))
    return rows


def roc_curve_np(y, score):
    """ROC không cần sklearn. Trả về fpr, tpr, thresholds (giảm dần), auc."""
    order = np.argsort(-score, kind="mergesort")
    y, score = y[order], score[order]
    distinct = np.where(np.diff(score))[0]
    idx = np.r_[distinct, len(y) - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    P, N = y.sum(), (~y).sum()
    tpr = np.r_[0, tps / P] if P else np.r_[0, np.zeros_like(tps, dtype=float)]
    fpr = np.r_[0, fps / N] if N else np.r_[0, np.zeros_like(fps, dtype=float)]
    thr = np.r_[np.inf, score[idx]]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)) if P and N else float("nan")
    return fpr, tpr, thr, auc


def _hist_edges(values, step, lo=None, hi=None, max_bins=40):
    v = np.array([x for x in values if x is not None], dtype=float)
    if lo is None:
        lo = np.floor((v.min() if len(v) else -step) / step) * step
    if hi is None:
        hi = (np.floor((v.max() if len(v) else step) / step) + 1) * step
    n = int(round((hi - lo) / step))
    if n > max_bins:
        step = (hi - lo) / max_bins
        n = max_bins
    return [round(lo + i * step, 4) for i in range(n + 1)]


def evaluate(pairs, bin_size, tol):
    """Gom toàn bộ dữ liệu đánh giá (chưa ghi Excel)."""
    bin_rows, seg_rows, run_rows, file_rows = [], [], [], []
    for p in pairs:
        if "rebin" not in p:
            continue
        g_s, g_e, gt = load_gt(p["gt"])
        rb_s, rb_e, rb_ratio, rb_sp = p["rebin"]
        mid = (g_s + g_e) / 2
        j = np.clip(np.searchsorted(rb_s, mid, side="right") - 1, 0, len(rb_s) - 1)
        pred, ratio = rb_sp[j], rb_ratio[j]
        score = score_on_grid(g_s, g_e, p["bins"])
        etype, depth, runs = classify_errors(gt, pred, bin_size, tol)
        name, cat = p["model"].name, p["category"]

        for k in range(len(gt)):
            out = ("TP" if pred[k] else "FN") if gt[k] else ("FP" if pred[k] else "TN")
            bin_rows.append([name, cat, float(g_s[k]), float(g_e[k]),
                             "speech" if gt[k] else "non-speech",
                             "speech" if pred[k] else "non-speech",
                             None if np.isnan(score[k]) else round(float(score[k]), 6),
                             round(float(ratio[k]), 4), out, etype[k] or "-",
                             round(float(depth[k]), 3) if etype[k] else None])
        for r in segment_errors(g_s, g_e, gt, p["segments"]):
            seg_rows.append([name, cat, *r])
        for o, t, grp, side, a, b in runs:
            run_rows.append([name, cat, o, t, grp, side, float(g_s[a]), float(g_e[b]),
                             round(float(g_e[b] - g_s[a]), 3)])
        file_rows.append([name, cat, str(p["gt"])])
    return bin_rows, seg_rows, run_rows, file_rows


def write_excel(xlsx_path: Path, pairs, cfg, bin_size, tol):
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, ScatterChart, Reference, Series
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter as L

    bin_rows, seg_rows, run_rows, file_rows = evaluate(pairs, bin_size, tol)
    if not bin_rows:
        print("     [BỎ QUA] không có dữ liệu để đánh giá")
        return
    cats = sorted({r[1] for r in bin_rows})

    FONT = "Arial"
    f_norm = Font(name=FONT, size=10)
    f_bold = Font(name=FONT, size=10, bold=True)
    f_head = Font(name=FONT, size=10, bold=True, color="FFFFFF")
    f_title = Font(name=FONT, size=14, bold=True)
    f_sec = Font(name=FONT, size=12, bold=True, color="1F4E78")
    f_note = Font(name=FONT, size=9, italic=True, color="595959")
    f_input = Font(name=FONT, size=10, color="0000FF")
    fill_head = PatternFill("solid", fgColor="1F4E78")
    fill_total = PatternFill("solid", fgColor="DDEBF7")
    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    PCT, NUM2, NUM1 = "0.0%", "0.00", "#,##0.0"

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"

    def header(sh, row, col, titles):
        for i, t in enumerate(titles):
            c = sh.cell(row=row, column=col + i, value=t)
            c.font, c.fill, c.border = f_head, fill_head, border
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    def put(sh, row, col, val, fmt=None, font=None, fill=None):
        c = sh.cell(row=row, column=col, value=val)
        c.font = font or f_norm
        c.border = border
        if fmt:
            c.number_format = fmt
        if fill:
            c.fill = fill
        return c

    def data_sheet(title, cols, rows, fmts=None):
        sh = wb.create_sheet(title)
        header(sh, 1, 1, cols)
        for r, row in enumerate(rows, start=2):
            for c, v in enumerate(row, start=1):
                cell = sh.cell(row=r, column=c, value=v)
                cell.font = f_norm
                if fmts and c in fmts:
                    cell.number_format = fmts[c]
        sh.freeze_panes = "A2"
        sh.auto_filter.ref = f"A1:{L(len(cols))}{max(2, len(rows) + 1)}"
        for c in range(1, len(cols) + 1):
            sh.column_dimensions[L(c)].width = 16
        sh.column_dimensions["A"].width = 22
        return sh

    # ---------------- sheet dữ liệu ----------------
    n_bin = len(bin_rows) + 1
    bd = data_sheet("Bin_detail",
                    ["File", "Category", "Start (s)", "End (s)", "Duration (s)", "GT", "Pred",
                     "Score", "Speech ratio", "Outcome", "Error type", "Error depth (s)"],
                    [r[:4] + [None] + r[4:] for r in bin_rows],
                    {3: NUM2, 4: NUM2, 5: NUM2, 8: "0.0000", 9: PCT, 12: NUM2})
    for r in range(2, n_bin + 1):
        bd.cell(row=r, column=5, value=f"=D{r}-C{r}").number_format = NUM2
    data_sheet("Segments",
               ["File", "Category", "GT start (s)", "GT end (s)", "Detected",
                "Pred start (s)", "Pred end (s)", "Onset error (s)", "Offset error (s)"],
               seg_rows, {3: NUM2, 4: NUM2, 6: NUM2, 7: NUM2, 8: NUM2, 9: NUM2})
    data_sheet("Error_runs",
               ["File", "Category", "Outcome", "Run type", "Depth group", "Side",
                "Start (s)", "End (s)", "Depth (s)"],
               run_rows, {7: NUM2, 8: NUM2, 9: NUM2})

    pf = data_sheet("Per_file",
                    ["File", "Category", "GT file", "Duration (s)", "TP", "FP", "TN", "FN",
                     "FPR", "FNR", "F1"], [fr + [None] * 8 for fr in file_rows])
    pf.column_dimensions["C"].width = 45
    B = "Bin_detail!"
    for r in range(2, len(file_rows) + 2):
        pf.cell(row=r, column=4, value=f"=SUMIFS({B}$E:$E,{B}$A:$A,$A{r})").number_format = NUM1
        for k, o in enumerate(["TP", "FP", "TN", "FN"]):
            pf.cell(row=r, column=5 + k, value=f'=COUNTIFS({B}$A:$A,$A{r},{B}$J:$J,"{o}")')
        pf.cell(row=r, column=9, value=f"=IF(F{r}+G{r}=0,0,F{r}/(F{r}+G{r}))").number_format = PCT
        pf.cell(row=r, column=10, value=f"=IF(H{r}+E{r}=0,0,H{r}/(H{r}+E{r}))").number_format = PCT
        pf.cell(row=r, column=11, value=f"=IF(2*E{r}+F{r}+H{r}=0,0,2*E{r}/(2*E{r}+F{r}+H{r}))").number_format = PCT

    # ---------------- ROC (tính bằng Python) ----------------
    y = np.array([r[4] == "speech" for r in bin_rows])
    sc = np.array([np.nan if r[6] is None else r[6] for r in bin_rows], dtype=float)
    ok = ~np.isnan(sc)
    fpr, tpr, thr, auc = roc_curve_np(y[ok], sc[ok])
    yj = int(np.argmax(tpr - fpr))
    keep = np.unique(np.r_[np.linspace(0, len(fpr) - 1, min(len(fpr), 200)).astype(int), yj])
    roc = wb.create_sheet("ROC_data")
    header(roc, 1, 1, ["Threshold", "FPR", "TPR", "Diag X", "Diag Y"])
    for i, k in enumerate(keep, start=2):
        roc.cell(row=i, column=1, value=None if np.isinf(thr[k]) else float(thr[k])).number_format = "0.0000"
        roc.cell(row=i, column=2, value=float(fpr[k])).number_format = "0.0000"
        roc.cell(row=i, column=3, value=float(tpr[k])).number_format = "0.0000"
    roc["D2"], roc["E2"], roc["D3"], roc["E3"] = 0, 0, 1, 1
    n_roc = len(keep) + 1
    roc.cell(row=1, column=7, value=f"ROC tính bằng Python trên {int(ok.sum())} bin có score "
                                    f"(score = trung bình có trọng số các bin hop {HOP}s trong bin GT); "
                                    f"lấy mẫu {len(keep)}/{len(fpr)} điểm để vẽ.").font = f_note
    for c in "ABCDE":
        roc.column_dimensions[c].width = 12

    # ---------------- Chart_data (histogram bằng COUNTIFS) ----------------
    cd = wb.create_sheet("Chart_data")
    cd_row = 1

    def hist_block(title, edges, series, center_label=False):
        """series: list (tên, công thức COUNTIFS với {lo},{hi} được thay)."""
        nonlocal cd_row
        cd.cell(row=cd_row, column=1, value=title).font = f_bold
        cd_row += 1
        header(cd, cd_row, 1, ["From", "To", "Bin"] + [s[0] for s in series])
        top = cd_row + 1
        for i in range(len(edges) - 1):
            r = top + i
            cd.cell(row=r, column=1, value=edges[i])
            cd.cell(row=r, column=2, value=edges[i + 1])
            cd.cell(row=r, column=3, value=(f'=TEXT((A{r}+B{r})/2,"0.0")&"s"' if center_label
                                            else f'=TEXT(A{r},"0.00")&" – "&TEXT(B{r},"0.00")'))
            for k, (_, tmpl) in enumerate(series):
                cd.cell(row=r, column=4 + k,
                        value=tmpl.format(lo=f"$A{r}", hi=f"$B{r}"))
        bottom = top + len(edges) - 2
        cd_row = bottom + 3
        return top, bottom, len(series)

    onset_vals = [r[7] for r in seg_rows]
    offset_vals = [r[8] for r in seg_rows]
    step_err = 0.1
    e_on = _hist_edges(onset_vals + offset_vals, step_err)
    S = "Segments!"
    h_on = hist_block("Histogram onset error (s)", e_on,
                      [("Onset", f'=COUNTIFS({S}$H:$H,">="&{{lo}},{S}$H:$H,"<"&{{hi}})')])
    h_off = hist_block("Histogram offset error (s)", e_on,
                       [("Offset", f'=COUNTIFS({S}$I:$I,">="&{{lo}},{S}$I:$I,"<"&{{hi}})')])
    e_dep = _hist_edges([r[8] for r in run_rows] or [bin_size], bin_size, lo=bin_size / 2)
    R = "Error_runs!"
    h_dep = hist_block("Histogram độ sâu lỗi (s)", e_dep,
                       [(g, f'=COUNTIFS({R}$E:$E,"{g}",{R}$I:$I,">="&{{lo}},{R}$I:$I,"<"&{{hi}})')
                        for g in DEPTH_GROUPS], center_label=True)
    e_sc = [round(i * 0.05, 2) for i in range(21)]
    e_sc[-1] = 1.0001
    h_sc = hist_block("Histogram score theo nhãn GT", e_sc,
                      [(lab, f'=COUNTIFS({B}$F:$F,"{lab}",{B}$H:$H,">="&{{lo}},{B}$H:$H,"<"&{{hi}})')
                       for lab in ("speech", "non-speech")])
    for c, w in zip("ABCDEFG", (9, 9, 14, 14, 14, 14, 14)):
        cd.column_dimensions[c].width = w

    # ---------------- Summary ----------------
    ws.column_dimensions["A"].width = 26
    for c in range(2, 14):
        ws.column_dimensions[L(c)].width = 13
    ws["A1"] = "VAD Evaluation Summary"
    ws["A1"].font = f_title
    ws["A2"] = (f"Đơn vị so sánh: bin {bin_size:g}s theo lưới GT. Số lỗi tính theo số bin "
                f"(1 bin = {bin_size:g}s). Speech = positive.")
    ws["A2"].font = f_note

    row = 4
    ws.cell(row=row, column=1, value="Tham số pipeline").font = f_sec
    row += 1
    for k, v in cfg.items():
        put(ws, row, 1, k, font=f_bold)
        put(ws, row, 2, v, font=f_input)
        row += 1
    ws.cell(row=row, column=1, value="Giá trị xanh dương = tham số lúc chạy (từ dòng lệnh), "
                                     "không phải công thức.").font = f_note
    row += 2

    # A. Tổng quan dữ liệu
    ws.cell(row=row, column=1, value="A. Tổng quan bộ dữ liệu (theo GT)").font = f_sec
    row += 1
    header(ws, row, 1, ["Category", "Số file", "Tổng thời gian (s)", "Speech GT (s)",
                        "% Speech", "% Non-speech", "Speech dự đoán (s)", "% Speech dự đoán"])
    first = row + 1
    for i, cat in enumerate(cats + ["TOTAL"]):
        r = first + i
        tot = cat == "TOTAL"
        fill = fill_total if tot else None
        fnt = f_bold if tot else f_norm
        put(ws, r, 1, cat, font=f_bold, fill=fill)
        if tot:
            last = r - 1
            for c in (2, 3, 4, 7):
                put(ws, r, c, f"=SUM({L(c)}{first}:{L(c)}{last})", NUM1 if c > 2 else None, fnt, fill)
        else:
            put(ws, r, 2, f"=COUNTIFS(Per_file!$B:$B,$A{r})", None, fnt, fill)
            put(ws, r, 3, f"=SUMIFS({B}$E:$E,{B}$B:$B,$A{r})", NUM1, fnt, fill)
            put(ws, r, 4, f'=SUMIFS({B}$E:$E,{B}$B:$B,$A{r},{B}$F:$F,"speech")', NUM1, fnt, fill)
            put(ws, r, 7, f'=SUMIFS({B}$E:$E,{B}$B:$B,$A{r},{B}$G:$G,"speech")', NUM1, fnt, fill)
        put(ws, r, 5, f"=IF(C{r}=0,0,D{r}/C{r})", PCT, fnt, fill)
        put(ws, r, 6, f"=IF(C{r}=0,0,1-E{r})", PCT, fnt, fill)
        put(ws, r, 8, f"=IF(C{r}=0,0,G{r}/C{r})", PCT, fnt, fill)
    row = first + len(cats) + 2

    # B. Metrics
    ws.cell(row=row, column=1, value="B. FPR / FNR và các chỉ số").font = f_sec
    row += 1
    header(ws, row, 1, ["Category", "TP", "FP", "TN", "FN", "FPR", "FNR",
                        "Precision", "Recall", "F1", "Accuracy"])
    first = row + 1
    for i, cat in enumerate(cats + ["TOTAL"]):
        r = first + i
        tot = cat == "TOTAL"
        fill = fill_total if tot else None
        fnt = f_bold if tot else f_norm
        put(ws, r, 1, cat, font=f_bold, fill=fill)
        for k, o in enumerate(["TP", "FP", "TN", "FN"]):
            c = 2 + k
            val = (f"=SUM({L(c)}{first}:{L(c)}{r - 1})" if tot
                   else f'=COUNTIFS({B}$B:$B,$A{r},{B}$J:$J,"{o}")')
            put(ws, r, c, val, None, fnt, fill)
        put(ws, r, 6, f"=IF(C{r}+D{r}=0,0,C{r}/(C{r}+D{r}))", PCT, fnt, fill)
        put(ws, r, 7, f"=IF(E{r}+B{r}=0,0,E{r}/(E{r}+B{r}))", PCT, fnt, fill)
        put(ws, r, 8, f"=IF(B{r}+C{r}=0,0,B{r}/(B{r}+C{r}))", PCT, fnt, fill)
        put(ws, r, 9, f"=IF(B{r}+E{r}=0,0,B{r}/(B{r}+E{r}))", PCT, fnt, fill)
        put(ws, r, 10, f"=IF(2*B{r}+C{r}+E{r}=0,0,2*B{r}/(2*B{r}+C{r}+E{r}))", PCT, fnt, fill)
        put(ws, r, 11, f"=IF(SUM(B{r}:E{r})=0,0,(B{r}+D{r})/SUM(B{r}:E{r}))", PCT, fnt, fill)
    total_metric_row = first + len(cats)
    row = total_metric_row + 1
    ws.cell(row=row, column=1, value="FPR = FP/(FP+TN) · FNR = FN/(FN+TP) · tính sau toàn bộ "
                                     "post-processing (#3, #4).").font = f_note
    row += 2
    put(ws, row, 1, "AUC (score thô, bin GT)", font=f_bold)
    put(ws, row, 2, round(auc, 4) if not np.isnan(auc) else "n/a", "0.0000")
    put(ws, row + 1, 1, "Ngưỡng tối ưu (Youden)", font=f_bold)
    put(ws, row + 1, 2, float(thr[yj]) if np.isfinite(thr[yj]) else "n/a", "0.0000")
    put(ws, row + 2, 1, "TPR / FPR tại ngưỡng đó", font=f_bold)
    put(ws, row + 2, 2, float(tpr[yj]), PCT)
    put(ws, row + 2, 3, float(fpr[yj]), PCT)
    ws.cell(row=row + 3, column=1, value="AUC/Youden tính bằng Python từ score thô (trước "
                                         "threshold và post-processing), xem sheet ROC_data.").font = f_note
    row += 5

    # C. Error types
    ws.cell(row=row, column=1, value=f"C. Phân loại lỗi (số bin, dung sai boundary = {tol} bin "
                                     f"= {tol * bin_size:g}s)").font = f_sec
    row += 1
    header(ws, row, 1, ["Category"] + ERROR_TYPES + ["Tổng lỗi"])
    first = row + 1
    for i, cat in enumerate(cats + ["TOTAL"]):
        r = first + i
        tot = cat == "TOTAL"
        fill = fill_total if tot else None
        fnt = f_bold if tot else f_norm
        put(ws, r, 1, cat, font=f_bold, fill=fill)
        for k, t in enumerate(ERROR_TYPES):
            c = 2 + k
            val = (f"=SUM({L(c)}{first}:{L(c)}{r - 1})" if tot
                   else f'=COUNTIFS({B}$B:$B,$A{r},{B}$K:$K,"{t}")')
            put(ws, r, c, val, None, fnt, fill)
        c_tot = 2 + len(ERROR_TYPES)
        put(ws, r, c_tot, f"=SUM(B{r}:{L(c_tot - 1)}{r})", None, fnt, fill)
    r_tot = first + len(cats)
    r_pct = r_tot + 1
    put(ws, r_pct, 1, "% trên tổng lỗi", font=f_bold)
    for k in range(len(ERROR_TYPES) + 1):
        c = 2 + k
        put(ws, r_pct, c, f"=IF(${L(c_tot)}${r_tot}=0,0,{L(c)}{r_tot}/${L(c_tot)}${r_tot})", PCT)
    row = r_pct + 2
    header(ws, row, 1, ["Nhóm", "Loại lỗi", "Định nghĩa"])
    ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=11)
    for t in ERROR_TYPES:
        row += 1
        grp = ("Boundary" if t.startswith("boundary") else
               "FP" if t in ("extension", "spurious") else "FN")
        put(ws, row, 1, grp)
        put(ws, row, 2, t, font=f_bold)
        ws.cell(row=row, column=3, value=ERROR_DESC[t]).font = f_norm
        ws.merge_cells(start_row=row, start_column=3, end_row=row, end_column=11)
    row += 2

    # D. Onset/offset thống kê nhanh
    ws.cell(row=row, column=1, value="D. Sai lệch onset / offset theo segment (dự đoán #3 so với GT)").font = f_sec
    row += 1
    header(ws, row, 1, ["", "Số segment GT", "Bắt được", "Trung bình (s)", "Min (s)", "Max (s)"])
    for k, (lab, col) in enumerate((("Onset", "H"), ("Offset", "I"))):
        r = row + 1 + k
        put(ws, r, 1, lab, font=f_bold)
        put(ws, r, 2, f"=COUNTA({S}$C:$C)-1")
        put(ws, r, 3, f'=COUNTIFS({S}$E:$E,"yes")')
        put(ws, r, 4, f"=IF(C{r}=0,0,AVERAGE({S}${col}:${col}))", "0.000")
        put(ws, r, 5, f"=IF(C{r}=0,0,MIN({S}${col}:${col}))", "0.000")
        put(ws, r, 6, f"=IF(C{r}=0,0,MAX({S}${col}:${col}))", "0.000")
    ws.cell(row=row + 3, column=1, value="Âm = sớm hơn GT, dương = trễ hơn GT. GT có độ phân giải "
                                         f"{bin_size:g}s nên sai lệch dưới mức này chủ yếu do lưới GT.").font = f_note
    row += 5

    # E. Charts
    ws.cell(row=row, column=1, value="E. Biểu đồ").font = f_sec
    row += 1

    def bar_chart(title, block, x_title, y_title="Số lượng", stacked=False):
        top, bottom, n_ser = block
        ch = BarChart()
        ch.type = "col"
        ch.title, ch.x_axis.title, ch.y_axis.title = title, x_title, y_title
        ch.add_data(Reference(cd, min_col=4, max_col=3 + n_ser, min_row=top - 1, max_row=bottom),
                    titles_from_data=True)
        ch.set_categories(Reference(cd, min_col=3, min_row=top, max_row=bottom))
        ch.gapWidth = 20
        if stacked:
            ch.grouping, ch.overlap = "stacked", 100
        ch.width, ch.height = 17, 8
        ch.x_axis.delete = False
        ch.y_axis.delete = False
        if n_ser == 1:
            ch.legend = None
        return ch

    charts = [
        bar_chart("Onset error (pred - GT)", h_on, "Sai lệch (s)"),
        bar_chart("Offset error (pred - GT)", h_off, "Sai lệch (s)"),
        bar_chart("Độ sâu lỗi (độ dài run lỗi)", h_dep, "Độ sâu (s)", stacked=True),
        bar_chart("Phân bố score: speech vs non-speech", h_sc, "Score", "Số bin"),
    ]
    rc = ScatterChart()
    rc.title = f"ROC curve (AUC = {auc:.4f})"
    rc.x_axis.title, rc.y_axis.title = "FPR", "TPR"
    rc.x_axis.scaling.min = rc.y_axis.scaling.min = 0
    rc.x_axis.scaling.max = rc.y_axis.scaling.max = 1
    rc.x_axis.delete = rc.y_axis.delete = False
    s1 = Series(Reference(roc, min_col=3, min_row=2, max_row=n_roc),
                Reference(roc, min_col=2, min_row=2, max_row=n_roc), title="ROC")
    s1.marker.symbol = "none"
    s1.smooth = False
    s1.graphicalProperties.line.solidFill = "1F4E78"
    s1.graphicalProperties.line.width = 28575
    s2 = Series(Reference(roc, min_col=5, min_row=2, max_row=3),
                Reference(roc, min_col=4, min_row=2, max_row=3), title="Random")
    s2.marker.symbol = "none"
    s2.graphicalProperties.line.dashStyle = "dash"
    s2.graphicalProperties.line.solidFill = "A6A6A6"
    # điểm vận hành thực tế của pipeline (sau post-processing)
    op = wb["ROC_data"]
    op["G3"], op["H3"], op["I3"] = "Pipeline", "FPR", "TPR"
    op["H4"] = f"=Summary!F{total_metric_row}"
    op["I4"] = f"=1-Summary!G{total_metric_row}"
    s3 = Series(Reference(op, min_col=9, min_row=4, max_row=4),
                Reference(op, min_col=8, min_row=4, max_row=4), title="Pipeline (sau post-process)")
    s3.marker.symbol, s3.marker.size = "diamond", 10
    s3.marker.graphicalProperties.solidFill = "C00000"
    s3.marker.graphicalProperties.line.solidFill = "C00000"
    s3.graphicalProperties.line.noFill = True
    rc.series += [s1, s2, s3]
    rc.width, rc.height = 17, 10
    charts.append(rc)

    for i, ch in enumerate(charts):
        anchor_row = row + (i // 2) * 22
        anchor_col = "A" if i % 2 == 0 else "H"
        ws.add_chart(ch, f"{anchor_col}{anchor_row}")

    ws.freeze_panes = "A4"
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(xlsx_path)
    print(f"     {len(file_rows)} file, {len(bin_rows)} bin, {len(seg_rows)} segment GT, "
          f"{len(run_rows)} run lỗi, AUC={auc:.4f} -> {xlsx_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="VAD: scan + resample overlap")
    ap.add_argument("--model_root", default="Model")
    ap.add_argument("--gt_root", default="Groundtruth")
    ap.add_argument("--out_dir", default="output_after_overlap")
    ap.add_argument("--hop", type=float, default=HOP)
    ap.add_argument("--method", choices=["mean", "max"], default="mean")
    ap.add_argument("--clean_dir", default="output_clean_data")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help=f"score >= threshold -> speech (mặc định {THRESHOLD})")
    ap.add_argument("--pad_before", type=float, default=PAD_BEFORE * 1000,
                    help=f"padding trước, ms (mặc định {PAD_BEFORE*1000:.0f})")
    ap.add_argument("--pad_after", type=float, default=PAD_AFTER * 1000,
                    help=f"padding sau, ms (mặc định {PAD_AFTER*1000:.0f})")
    ap.add_argument("--gap", type=float, default=MERGE_GAP * 1000,
                    help=f"merge nếu gap < giá trị này, ms (mặc định {MERGE_GAP*1000:.0f})")
    ap.add_argument("--drop", type=float, default=MIN_DUR * 1000,
                    help=f"bỏ segment ngắn hơn giá trị này, ms (mặc định {MIN_DUR*1000:.0f}, 0 = không bỏ)")
    ap.add_argument("--drop_before_pad", action="store_true",
                    help="bỏ segment < 100ms cả TRƯỚC khi padding (lọc bin speech lẻ loi)")
    ap.add_argument("--rebin_dir", default="output_rebin")
    ap.add_argument("--rebin_size", type=float, default=REBIN_SIZE * 1000,
                    help=f"độ dài bin khi rebin, ms (mặc định {REBIN_SIZE*1000:.0f})")
    ap.add_argument("--speech_ratio", type=float, default=MIN_SPEECH_RATIO,
                    help=f"tỉ lệ speech tối thiểu trong bin để gán speech, 0-1 (mặc định {MIN_SPEECH_RATIO})")
    ap.add_argument("--with_ratio", action="store_true",
                    help="ghi thêm cột thứ 4 là tỉ lệ speech của mỗi bin")
    ap.add_argument("--report", default="evaluation_report.xlsx",
                    help="đường dẫn file Excel đánh giá")
    ap.add_argument("--boundary_tol", type=int, default=1,
                    help="số bin quanh điểm chuyển GT coi là lỗi boundary (mặc định 1)")
    ap.add_argument("--speech_only", action="store_true",
                    help="chỉ ghi segment speech, không ghi non-speech")
    args = ap.parse_args()

    model_root, gt_root, out_dir = Path(args.model_root), Path(args.gt_root), Path(args.out_dir)

    # #1
    pairs, missing = find_pairs(model_root, gt_root)
    print(f"[#1] Tìm thấy {len(pairs)} cặp model/GT")
    for p in pairs:
        print(f"     {p['model'].name:<25} <-> {p['gt']}")
    for mf, reason in missing:
        print(f"     [BỎ QUA] {mf.name}: {reason}")

    # #2
    print(f"\n[#2] Resample overlap (hop={args.hop}s, method={args.method}) -> {out_dir}/")
    for p in pairs:
        try:
            st, en, sc = load_model_file(p["model"])
            bs, be, bsc, cnt = resolve_overlap(st, en, sc, hop=args.hop, method=args.method)
            out_path = out_dir / p["model"].name
            save_bins(out_path, bs, be, bsc)
            p["overlap_file"] = out_path
            print(f"     {p['model'].name}: {len(sc)} windows -> {len(bsc)} bins "
                  f"(0.00 - {be[-1]:.2f}s)")
            p["bins"] = (bs, be, bsc)
        except Exception as ex:
            print(f"     [LỖI] {p['model'].name}: {ex}")

    # #3
    clean_dir = Path(args.clean_dir)
    for name in ("pad_before", "pad_after", "gap", "drop"):
        if getattr(args, name) < 0:
            ap.error(f"--{name} phải >= 0")
    print(f"\n[#3] Clean (thr={args.threshold}, pad -{args.pad_before:g}/+{args.pad_after:g}ms, "
          f"merge gap<{args.gap:g}ms, drop<{args.drop:g}ms"
          f"{', drop trước pad' if args.drop_before_pad else ''}) -> {clean_dir}/")
    for p in pairs:
        if "bins" not in p:
            continue
        bs, be, bsc = p["bins"]
        t_max = gt_duration(p["gt"]) or float(be[-1])
        segs = clean_segments(bs, be, bsc, threshold=args.threshold,
                              pad_before=args.pad_before / 1000,
                              pad_after=args.pad_after / 1000,
                              merge_gap=args.gap / 1000,
                              min_dur=args.drop / 1000,
                              t_max=t_max,
                              drop_before_pad=args.drop_before_pad)
        out_path = clean_dir / p["model"].name
        save_segments(out_path, segs, t_max, full_timeline=not args.speech_only)
        p["clean_file"], p["segments"] = out_path, segs
        print(f"     {p['model'].name}: {len(segs)} speech segment(s) "
              f"{[(round(s,2), round(e,2)) for s, e in segs]}")
        p["t_max"] = t_max

    # #4
    if args.rebin_size <= 0:
        ap.error("--rebin_size phải > 0")
    if not 0 <= args.speech_ratio <= 1:
        ap.error("--speech_ratio phải trong [0, 1]")
    rebin_dir = Path(args.rebin_dir)
    print(f"\n[#4] Rebin (bin={args.rebin_size:g}ms, speech nếu >= {args.speech_ratio:.0%}) -> {rebin_dir}/")
    for p in pairs:
        if "segments" not in p:
            continue
        rb = rebin_segments(p["segments"], p["t_max"],
                            bin_size=args.rebin_size / 1000, min_ratio=args.speech_ratio)
        out_path = rebin_dir / p["model"].name
        save_rebin(out_path, *rb, with_ratio=args.with_ratio)
        p["rebin_file"], p["rebin"] = out_path, rb
        print(f"     {p['model'].name}: {len(rb[0])} bins, {int(rb[3].sum())} speech")

    # #5
    if args.boundary_tol < 0:
        ap.error("--boundary_tol phải >= 0")
    print(f"\n[#5] Đánh giá -> {args.report}")
    cfg = {
        "Threshold": args.threshold,
        "Overlap method": args.method,
        "Pad trước (ms)": args.pad_before,
        "Pad sau (ms)": args.pad_after,
        "Merge gap < (ms)": args.gap,
        "Drop segment < (ms)": args.drop,
        "Drop trước pad": "yes" if args.drop_before_pad else "no",
        "Rebin size (ms)": args.rebin_size,
        "Speech ratio tối thiểu": args.speech_ratio,
        "Boundary tolerance (bin)": args.boundary_tol,
    }
    write_excel(Path(args.report), pairs, cfg, args.rebin_size / 1000, args.boundary_tol)

    return pairs


if __name__ == "__main__":
    main()
