from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import sys
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config as C
from utils import render_history, write_jsonl

USER_WIDTH = 32
ITEM_WIDTH = 16

ItemMeta = Dict[str, Tuple[str, str, str]]

_ARTICLES = {
    "The", "A", "An", "La", "Le", "Les", "L'", "Il", "Lo", "I",
    "Der", "Das", "Die", "Den", "Ein", "El", "Los", "Las", "Un", "Une", "O",
}
_JUNK_TITLE_MARKERS = ("getTime()", "var aPage", "<span", "<script")


def log(msg: str) -> None:
    print(msg, flush=True)


class InteractionBuffer:

    def __init__(self, user_width: int = USER_WIDTH, item_width: int = ITEM_WIDTH):
        self.uw, self.iw = user_width, item_width
        self.users = bytearray()
        self.items = bytearray()
        self.ts = array("q")

    def add(self, user: str, item: str, ts: int) -> None:
        ub = user.encode("utf-8")[:self.uw]
        ib = item.encode("utf-8")[:self.iw]
        self.users += ub + b"\x00" * (self.uw - len(ub))
        self.items += ib + b"\x00" * (self.iw - len(ib))
        self.ts.append(ts)

    def __len__(self) -> int:
        return len(self.ts)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self) == 0:
            raise SystemExit("no interactions parsed from raw files")
        u_bytes = np.frombuffer(memoryview(self.users), dtype=f"S{self.uw}")
        i_bytes = np.frombuffer(memoryview(self.items), dtype=f"S{self.iw}")
        ts = np.frombuffer(memoryview(self.ts), dtype=np.int64).copy()
        _, users = np.unique(u_bytes, return_inverse=True)
        item_keys, items = np.unique(i_bytes, return_inverse=True)
        return (users.astype(np.int32, copy=False).ravel(),
                items.astype(np.int32, copy=False).ravel(), ts, item_keys)


def clean_title(raw: object, max_chars: int) -> str:
    if not raw:
        return ""
    text = html.unescape(str(raw))
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    if len(text) > max_chars:
        head = text[:max_chars].rsplit(" ", 1)[0]
        text = head if len(head) >= max_chars // 2 else text[:max_chars]
    return text.strip(" ,;:-")


def normalize_movielens_title(raw: str) -> str:
    text = re.sub(r"\s*\((\d{4})\)\s*$", "", raw).strip()
    stripped = re.sub(r"\s*\([^()]*\)", "", text).strip()
    if stripped:
        text = stripped
    m = re.match(r"^(.*),\s+([A-Za-z']{1,4})$", text)
    if m and m.group(2) in _ARTICLES:
        text = f"{m.group(2)} {m.group(1)}"
    return text


def _hierarchical_category(cats: object) -> str:
    valid = [c.strip() for c in (cats or []) if isinstance(c, str) and c.strip()]
    if len(valid) >= 2:
        return valid[1]
    return valid[0] if valid else ""


def load_movielens(ds: C.DatasetConfig, pc: C.PreprocessConfig
                   ) -> Tuple[InteractionBuffer, ItemMeta]:
    meta: ItemMeta = {}
    with open(ds.raw_path("item"), encoding=ds.encoding) as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 3:
                continue
            title = clean_title(normalize_movielens_title(parts[1]), pc.max_title_chars)
            if not title:
                continue
            genres = [g for g in parts[2].split("|") if g]
            primary = genres[0] if genres else "Unknown"
            meta[parts[0]] = (title, primary, primary)

    buf = InteractionBuffer()
    with open(ds.raw_path("inter"), encoding=ds.encoding) as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) < 4 or parts[1] not in meta:
                continue
            if pc.min_rating > 0 and float(parts[2]) < pc.min_rating:
                continue
            buf.add(parts[0], parts[1], int(parts[3]))
    return buf, meta


def load_amazon2023(ds: C.DatasetConfig, pc: C.PreprocessConfig
                    ) -> Tuple[InteractionBuffer, ItemMeta]:
    meta: ItemMeta = {}
    with open(ds.raw_path("meta"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            item_raw = obj.get("parent_asin")
            if not item_raw or item_raw in meta:
                continue
            title = clean_title(obj.get("title"), pc.max_title_chars)
            if not title:
                continue
            primary = _hierarchical_category(obj.get("categories"))
            if not primary:
                primary = str(obj.get("main_category") or "").strip() or "Unknown"
            fallback = str(obj.get("store") or "").strip() or "Unknown"
            meta[item_raw] = (title, primary, fallback)
    log(f"  meta items with usable title: {len(meta):,}")

    buf = InteractionBuffer()
    with open(ds.raw_path("inter"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            item_raw = obj.get("parent_asin")
            user_raw = obj.get("user_id")
            ts = obj.get("timestamp")
            if not user_raw or ts is None or item_raw not in meta:
                continue
            if pc.min_rating > 0 and float(obj.get("rating") or 0) < pc.min_rating:
                continue
            buf.add(user_raw, item_raw, int(ts))
    return buf, meta


def load_amazon2018(ds: C.DatasetConfig, pc: C.PreprocessConfig
                    ) -> Tuple[InteractionBuffer, ItemMeta]:
    meta: ItemMeta = {}
    with open(ds.raw_path("meta"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            item_raw = obj.get("asin")
            if not item_raw or item_raw in meta:
                continue
            raw_title = str(obj.get("title") or "")
            if any(marker in raw_title for marker in _JUNK_TITLE_MARKERS):
                continue
            title = clean_title(raw_title, pc.max_title_chars)
            if not title:
                continue
            primary = _hierarchical_category(obj.get("category"))
            if not primary:
                primary = str(obj.get("main_cat") or "").strip() or "Unknown"
            fallback = str(obj.get("brand") or "").strip() or "Unknown"
            meta[item_raw] = (title, primary, fallback)
    log(f"  meta items with usable title: {len(meta):,}")

    buf = InteractionBuffer()
    with open(ds.raw_path("inter"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            item_raw = obj.get("asin")
            user_raw = obj.get("reviewerID")
            ts = obj.get("unixReviewTime")
            if not user_raw or ts is None or item_raw not in meta:
                continue
            if pc.min_rating > 0 and float(obj.get("overall") or 0) < pc.min_rating:
                continue
            buf.add(user_raw, item_raw, int(ts))
    return buf, meta


LOADERS = {
    "movielens": load_movielens,
    "amazon2023": load_amazon2023,
    "amazon2018": load_amazon2018,
}


def parse_year_month(text: str) -> Tuple[int, int]:
    parts = text.strip().split("-")
    if len(parts) != 2:
        raise SystemExit(f"time window must be YYYY-MM, got {text!r}")
    year, month = int(parts[0]), int(parts[1])
    if not (1 <= month <= 12):
        raise SystemExit(f"invalid month {text!r}")
    return year, month


def month_start_unix(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def next_month(year: int, month: int) -> Tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def resolve_time_bounds(ds: C.DatasetConfig, ts_start: Optional[str] = None,
                        ts_end: Optional[str] = None
                        ) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]],
                                   Optional[int], Optional[int]]:
    if (ts_start is None) != (ts_end is None):
        raise SystemExit("--ts_start and --ts_end must be set together")
    if ts_start:
        start, end = parse_year_month(ts_start), parse_year_month(ts_end)
    elif ds.time_start and ds.time_end:
        start, end = tuple(ds.time_start), tuple(ds.time_end)
    else:
        return None, None, None, None
    if start > end:
        raise SystemExit(f"time-window start {start[0]:04d}-{start[1]:02d} is after end {end[0]:04d}-{end[1]:02d}")
    lo = month_start_unix(*start)
    hi = month_start_unix(*next_month(*end))
    if ds.ts_unit == "ms":
        lo *= 1000
        hi *= 1000
    elif ds.ts_unit != "s":
        raise SystemExit(f"unknown ts_unit {ds.ts_unit!r}")
    return start, end, lo, hi


def apply_time_window(users: np.ndarray, items: np.ndarray, ts: np.ndarray,
                      ts_lo: int, ts_hi: int
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = (ts >= ts_lo) & (ts < ts_hi)
    n_keep = int(mask.sum())
    if n_keep == 0:
        raise SystemExit(f"time window [{ts_lo}, {ts_hi}) contains no interactions")
    return users[mask], items[mask], ts[mask]


def unique_counts(users: np.ndarray, items: np.ndarray) -> Tuple[int, int, int]:
    return int(np.unique(users).size), int(np.unique(items).size), int(len(users))


def counts_dict(users: np.ndarray, items: np.ndarray) -> dict:
    n_u, n_i, n_x = unique_counts(users, items)
    return {"n_users": n_u, "n_items": n_i, "n_interactions": n_x}


def format_counts(d: dict) -> str:
    return (f"users={d['n_users']:,}  items={d['n_items']:,}  "
            f"interactions={d['n_interactions']:,}")


def keep_pct(after: dict, before: dict) -> str:
    def pct(a: int, b: int) -> float:
        return 0.0 if b == 0 else 100.0 * a / b
    return (f"kept {pct(after['n_users'], before['n_users']):.1f}% / "
            f"{pct(after['n_items'], before['n_items']):.1f}% / "
            f"{pct(after['n_interactions'], before['n_interactions']):.1f}%")


def format_ym(ym: Optional[Tuple[int, int]]) -> Optional[str]:
    return None if ym is None else f"{ym[0]:04d}-{ym[1]:02d}"


def dedup_interactions(users: np.ndarray, items: np.ndarray, ts: np.ndarray,
                       n_items: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = users.astype(np.int64) * n_items + items
    order = np.lexsort((ts, key))
    sorted_key = key[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    np.not_equal(sorted_key[1:], sorted_key[:-1], out=first[1:])
    keep = np.sort(order[first])
    return users[keep], items[keep], ts[keep]


def iterative_kcore(users: np.ndarray, items: np.ndarray, ts: np.ndarray, k: int,
                    n_users: int, n_items: int
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    while len(users):
        user_cnt = np.bincount(users, minlength=n_users)
        item_cnt = np.bincount(items, minlength=n_items)
        mask = (user_cnt[users] >= k) & (item_cnt[items] >= k)
        if mask.all():
            break
        users, items, ts = users[mask], items[mask], ts[mask]
    return users, items, ts


def load_kcore_arrays(ds: C.DatasetConfig, pc: C.PreprocessConfig, core: int,
                      ts_start: Optional[str] = None, ts_end: Optional[str] = None,
                      ts_lo: Optional[int] = None, ts_hi: Optional[int] = None,
                      keep_meta: bool = False, log_fn=log):
    log_fn(f"loading {ds.key} via {ds.loader} loader")
    buf, meta = LOADERS[ds.loader](ds, pc)
    log_fn(f"  interactions parsed: {len(buf):,}")
    users, items, ts, item_keys = buf.finalize()
    del buf
    if not keep_meta:
        del meta
        meta = None
    n_users_tmp, n_items_tmp = int(users.max()) + 1, len(item_keys)
    n_before_time = {"n_users": n_users_tmp, "n_items": n_items_tmp,
                     "n_interactions": int(len(users))}
    log_fn(f"  before time filter: {format_counts(n_before_time)}")

    if ts_lo is None or ts_hi is None:
        start_ym, end_ym, ts_lo, ts_hi = resolve_time_bounds(ds, ts_start, ts_end)
    else:
        start_ym, end_ym, _, _ = resolve_time_bounds(ds, ts_start, ts_end)

    n_after_time = None
    if ts_lo is not None and ts_hi is not None:
        label = (f"{format_ym(start_ym)} .. {format_ym(end_ym)}"
                 if start_ym and end_ym else "custom")
        log_fn(f"  time window {label} (inclusive months, UTC)  ts in [{ts_lo}, {ts_hi})")
        users, items, ts = apply_time_window(users, items, ts, ts_lo, ts_hi)
        n_after_time = counts_dict(users, items)
        log_fn(f"  after  time filter: {format_counts(n_after_time)}  "
               f"({keep_pct(n_after_time, n_before_time)})")

    users, items, ts = dedup_interactions(users, items, ts, n_items_tmp)
    n_before_kcore = counts_dict(users, items)
    log_fn(f"  after dedup / before {core}-core: {format_counts(n_before_kcore)}")

    users, items, ts = iterative_kcore(users, items, ts, core, n_users_tmp, n_items_tmp)
    if len(users) == 0:
        raise SystemExit(f"{core}-core filtering removed all interactions")
    uniq_u, users = np.unique(users, return_inverse=True)
    uniq_i, items = np.unique(items, return_inverse=True)
    users = users.astype(np.int32, copy=False).ravel()
    items = items.astype(np.int32, copy=False).ravel()
    n_users, n_items = len(uniq_u), len(uniq_i)
    n_after_kcore = {"n_users": n_users, "n_items": n_items, "n_interactions": int(len(users))}
    log_fn(f"  after {core}-core: {format_counts(n_after_kcore)}")
    return {
        "users": users, "items": items, "ts": ts,
        "n_users": n_users, "n_items": n_items,
        "item_keys": item_keys, "uniq_i": uniq_i, "meta": meta,
        "start_ym": start_ym, "end_ym": end_ym, "ts_lo": ts_lo, "ts_hi": ts_hi,
        "n_after_time": n_after_time, "n_before_kcore": n_before_kcore,
        "n_after_kcore": n_after_kcore, "n_before_time": n_before_time,
    }


def build_windows(users: np.ndarray, items: np.ndarray, ts: np.ndarray, n_users: int,
                  min_hist: int) -> Dict[str, np.ndarray]:
    order = np.lexsort((items, ts, users))
    users_s, items_s, ts_s = users[order], items[order], ts[order]

    uids = np.arange(n_users)
    starts = np.searchsorted(users_s, uids, side="left")
    ends = np.searchsorted(users_s, uids, side="right")
    counts = np.maximum(0, (ends - starts) - min_hist)
    total = int(counts.sum())
    if total == 0:
        raise SystemExit(f"no user has more interactions than min_hist_len={min_hist}, cannot build windows")

    w_user = np.repeat(uids, counts).astype(np.int32)
    group_start = np.repeat(np.cumsum(counts) - counts, counts)
    pos = (np.arange(total) - group_start + min_hist).astype(np.int32)
    gidx = starts[w_user] + pos
    return {
        "users_s": users_s, "items_s": items_s, "ts_s": ts_s,
        "starts": starts, "ends": ends,
        "w_user": w_user, "w_pos": pos, "w_gidx": gidx, "w_ts": ts_s[gidx],
    }


def temporal_split(w_ts: np.ndarray, w_user: np.ndarray, w_pos: np.ndarray,
                   ratios: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((w_pos, w_user, w_ts))
    total = len(order)
    n_train = int(total * ratios[0])
    n_valid = int(total * ratios[1])
    return (order[:n_train], order[n_train:n_train + n_valid],
            order[n_train + n_valid:])


def compute_popularity(all_items: np.ndarray, n_items: int, n_group: int,
                       n_level: int, hot_ratio: float) -> Dict[str, np.ndarray]:
    freq = np.bincount(all_items, minlength=n_items).astype(np.int64)
    f_min, f_max = int(freq.min()), int(freq.max())
    span = (f_max - f_min) or 1

    order = np.lexsort((np.arange(n_items), freq))
    ranks = np.empty(n_items, dtype=np.int64)
    ranks[order] = np.arange(n_items)
    n_rank = ranks / max(n_items - 1, 1)
    pop_group = np.minimum(n_group - 1, ranks * n_group // n_items)

    n_minmax = (freq - f_min) / span
    log_freq = np.log1p(freq)
    log_span = (log_freq.max() - log_freq.min()) or 1.0
    n_log = (log_freq - log_freq.min()) / log_span
    pop_level = np.minimum(n_level - 1, (n_log * n_level).astype(np.int64))

    if abs(hot_ratio - 1.0 / n_group) < 1e-9:
        is_hot = pop_group == n_group - 1
    else:
        is_hot = ranks >= n_items - max(1, round(n_items * hot_ratio))

    return {
        "freq": freq, "n_minmax": n_minmax, "n_log_minmax": n_log, "n_rank": n_rank,
        "pop_level": pop_level, "pop_group": pop_group, "is_hot": is_hot,
        "f_min": f_min, "f_max": f_max, "n_hot": int(is_hot.sum()),
        "n_zero_freq": int((freq == 0).sum()),
        "hot_matches_top_group": bool((is_hot == (pop_group == n_group - 1)).all()),
    }


def render_records(idx: np.ndarray, win: Dict[str, np.ndarray], titles: List[str],
                   ds: C.DatasetConfig, max_hist: int) -> List[dict]:
    instruction = C.INSTRUCTION_TEMPLATE.format(item_noun=ds.item_noun)
    items_s, ts_s = win["items_s"], win["ts_s"]
    records = []
    for i in idx:
        gidx = int(win["w_gidx"][i])
        hist_len = min(int(win["w_pos"][i]), max_hist)
        history = items_s[gidx - hist_len:gidx].tolist()
        target = int(items_s[gidx])
        records.append({
            "user": int(win["w_user"][i]),
            "history": history,
            "target": target,
            "target_ts": int(ts_s[gidx]),
            "instruction": instruction,
            "input": C.INPUT_TEMPLATE.format(
                history_verb=ds.history_verb,
                history=render_history([titles[j] for j in history])),
            "output": titles[target],
        })
    return records


def group_distribution(idx: np.ndarray, win: Dict[str, np.ndarray],
                       pop_group: np.ndarray, n_group: int) -> List[float]:
    targets = win["items_s"][win["w_gidx"][idx]]
    counts = np.bincount(pop_group[targets], minlength=n_group)
    return [round(float(c) / max(counts.sum(), 1), 4) for c in counts]


def main() -> None:
    p = argparse.ArgumentParser(description="CKDLRec data preprocessing")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--core", type=int, default=None, help="override k-core threshold")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_train", type=int, default=None, help="optional train-window cap")
    p.add_argument("--n_eval", type=int, default=None, help="optional valid/test window cap")
    p.add_argument("--ts_start", default=None, help="time-window start YYYY-MM, overrides dataset default")
    p.add_argument("--ts_end", default=None, help="time-window end YYYY-MM (inclusive), overrides dataset default")
    p.add_argument("--stats_only", action="store_true",
                   help="print sizes only (after time filter / k-core / split); do not write processed files")
    args = p.parse_args()

    ds = C.get_dataset(args.dataset)
    pc = C.PREPROCESS
    core = args.core if args.core is not None else ds.core
    seed = args.seed if args.seed is not None else pc.seed
    n_train = args.n_train
    n_eval = args.n_eval
    rng = random.Random(seed)

    log(f"[1/6] loading + time filter + {core}-core")
    packed = load_kcore_arrays(ds, pc, core, ts_start=args.ts_start, ts_end=args.ts_end,
                               keep_meta=not args.stats_only, log_fn=log)
    users, items, ts = packed["users"], packed["items"], packed["ts"]
    n_users, n_items = packed["n_users"], packed["n_items"]

    def emit_summary(n_tr: int, n_va: int, n_te: int, kept: dict) -> None:
        before_time = packed["n_before_time"]
        after_time = packed["n_after_time"]
        before = packed["n_before_kcore"]
        after = packed["n_after_kcore"]
        log("======== SIZE SUMMARY ========")
        log(f"  time window: {format_ym(packed['start_ym'])} .. {format_ym(packed['end_ym'])}")
        log(f"  before time filter: {format_counts(before_time)}")
        if after_time:
            log(f"  after  time filter: {format_counts(after_time)}  "
                f"({keep_pct(after_time, before_time)})")
        log(f"  before {core}-core (after dedup): {format_counts(before)}")
        log(f"  after  {core}-core: {format_counts(after)}  "
            f"({keep_pct(after, before)})")
        log(f"  windows full:  train={n_tr:,}  valid={n_va:,}  test={n_te:,}")
        log(f"  windows kept:  train={kept['train']:,}  valid={kept['valid']:,}  "
            f"test={kept['test']:,}")
        log("==============================")

    if args.stats_only:
        log("[2/6] building sequences and sliding windows")
        win = build_windows(users, items, ts, n_users, pc.min_hist_len)
        log(f"  windows: {len(win['w_gidx']):,} from {len(np.unique(win['w_user'])):,} users")
        log("[3/6] global temporal split")
        train_idx, valid_idx, test_idx = temporal_split(
            win["w_ts"], win["w_user"], win["w_pos"], pc.split_ratios)
        log(f"  train/valid/test windows: {len(train_idx):,} / {len(valid_idx):,} / "
            f"{len(test_idx):,}")
        kept = {
            "train": len(train_idx) if n_train is None else min(len(train_idx), n_train),
            "valid": len(valid_idx) if n_eval is None else min(len(valid_idx), n_eval),
            "test": len(test_idx) if n_eval is None else min(len(test_idx), n_eval),
        }
        emit_summary(len(train_idx), len(valid_idx), len(test_idx), kept)
        log("stats_only: did not write processed files")
        return

    item_keys, uniq_i, meta = packed["item_keys"], packed["uniq_i"], packed["meta"]
    raw_ids = [item_keys[j].decode("utf-8").rstrip("\x00") for j in uniq_i]
    titles = [meta[r][0] for r in raw_ids]
    primary = [meta[r][1] for r in raw_ids]
    fallback = [meta[r][2] for r in raw_ids]
    del meta
    if ds.category_source in ("store", "brand"):
        categories, source = fallback, ds.category_source
    elif len(set(primary)) < pc.min_categories:
        log(f"  WARNING: primary category has only {len(set(primary))} classes "
            f"(< {pc.min_categories}), falling back to store/brand labels")
        categories, source = fallback, "fallback"
    else:
        categories, source = primary, "primary"
    cat_vocab = {c: n for n, c in enumerate(sorted(set(categories)))}
    log(f"  category source={source} | vocabulary: {len(cat_vocab)} classes")

    log("[2/6] popularity on all post-k-core interactions")
    pop = compute_popularity(items, n_items, pc.n_pop_group, pc.n_pop_level, pc.hot_ratio)
    inter_counts = np.bincount(pop["pop_group"][items], minlength=pc.n_pop_group)
    inter_dist = [round(float(c) / max(int(inter_counts.sum()), 1), 4) for c in inter_counts]
    log(f"  interactions: {len(items):,} | freq range [{pop['f_min']}, {pop['f_max']}] | "
        f"zero-freq items: {pop['n_zero_freq']:,}")
    log(f"  hot set H: {pop['n_hot']:,} items (top {pc.hot_ratio:.0%}), "
        f"cold set N: {n_items - pop['n_hot']:,} items | "
        f"consistent with top pop_group: {pop['hot_matches_top_group']}")
    log(f"  interaction pop_group dist: {inter_dist}")
    if pop["n_zero_freq"]:
        log(f"  WARNING: {pop['n_zero_freq']:,} items have freq 0 after k-core")

    log("[3/6] building sequences and sliding windows")
    win = build_windows(users, items, ts, n_users, pc.min_hist_len)
    n_windows = len(win["w_gidx"])
    n_active = len(np.unique(win["w_user"]))
    log(f"  windows: {n_windows:,} from {n_active:,} users "
        f"(dropped {n_users - n_active:,} users with < {pc.min_hist_len + 1} interactions)")

    log("[4/6] global temporal split")
    train_idx, valid_idx, test_idx = temporal_split(
        win["w_ts"], win["w_user"], win["w_pos"], pc.split_ratios)
    t_train_end = int(win["w_ts"][train_idx[-1]])
    log(f"  train/valid/test windows: {len(train_idx):,} / {len(valid_idx):,} / "
        f"{len(test_idx):,}")
    log(f"  train period ends at ts {t_train_end}")

    log("[5/6] split windows")
    dist_before = group_distribution(train_idx, win, pop["pop_group"], pc.n_pop_group)
    splits = {}
    for name, idx, quota in (("train", train_idx, n_train), ("valid", valid_idx, n_eval),
                             ("test", test_idx, n_eval)):
        if quota is not None and len(idx) > quota:
            picked = np.array(sorted(rng.sample(range(len(idx)), quota)))
            idx = idx[picked]
        splits[name] = idx
        log(f"  {name}: {len(idx):,}")
    dist_after = group_distribution(splits["train"], win, pop["pop_group"], pc.n_pop_group)
    log(f"  target pop_group distribution: {dist_before}")
    if n_train is not None or n_eval is not None:
        log(f"  target pop_group distribution after cap: {dist_after}")

    log("[6/6] rendering text and writing files")
    out_dir = ds.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in splits.items():
        rows = render_records(np.sort(idx), win, titles, ds, pc.max_hist_len)
        log(f"  wrote {name}.jsonl: {write_jsonl(out_dir / f'{name}.jsonl', rows):,} rows")

    n = write_jsonl(out_dir / "popularity.jsonl", ({
        "item_id": i,
        "raw_id": raw_ids[i],
        "title": titles[i],
        "category": categories[i],
        "category_id": cat_vocab[categories[i]],
        "freq": int(pop["freq"][i]),
        "n_minmax": round(float(pop["n_minmax"][i]), 6),
        "n_log_minmax": round(float(pop["n_log_minmax"][i]), 6),
        "n_rank": round(float(pop["n_rank"][i]), 6),
        "pop_level": int(pop["pop_level"][i]),
        "pop_group": int(pop["pop_group"][i]),
        "is_hot": bool(pop["is_hot"][i]),
    } for i in range(n_items)))
    log(f"  wrote popularity.jsonl: {n:,} rows")

    stats = {
        "dataset": ds.key,
        "core": core,
        "seed": seed,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": int(len(users)),
        "time_window": {
            "start": format_ym(packed["start_ym"]),
            "end": format_ym(packed["end_ym"]),
            "ts_lo": packed["ts_lo"],
            "ts_hi": packed["ts_hi"],
            "ts_unit": ds.ts_unit,
        },
        "n_before_time_filter": packed["n_before_time"],
        "n_after_time_filter": packed["n_after_time"],
        "n_before_kcore": packed["n_before_kcore"],
        "category_source": source,
        "n_categories": len(cat_vocab),
        "categories": sorted(cat_vocab, key=cat_vocab.get),
        "max_hist_len": pc.max_hist_len,
        "min_hist_len": pc.min_hist_len,
        "split_ratios": list(pc.split_ratios),
        "n_windows": {"train": int(len(train_idx)), "valid": int(len(valid_idx)),
                      "test": int(len(test_idx))},
        "n_kept": {k: int(len(v)) for k, v in splits.items()},
        "train_period_end_ts": t_train_end,
        "ts_unit": ds.ts_unit,
        "popularity": {
            "freq_min": pop["f_min"],
            "freq_max": pop["f_max"],
            "n_zero_freq": pop["n_zero_freq"],
            "hot_ratio": pc.hot_ratio,
            "n_hot": pop["n_hot"],
            "n_pop_group": pc.n_pop_group,
            "n_pop_level": pc.n_pop_level,
            "pop_group_def": "equal-size buckets by ascending frequency rank; "
                             f"group {pc.n_pop_group - 1} is the most popular = hot set H",
            "pop_level_def": f"equal-width bins of n_log_minmax into {pc.n_pop_level} levels",
        },
        "target_pop_group_dist": {"train_full": dist_before, "train_kept": dist_after},
        "interaction_pop_group_dist": inter_dist,
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log("  wrote stats.json")
    emit_summary(len(train_idx), len(valid_idx), len(test_idx),
                 {k: int(len(v)) for k, v in splits.items()})
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
