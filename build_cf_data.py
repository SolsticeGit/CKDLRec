from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import config as C
from dataset import ItemTable
from preprocess import build_windows, load_kcore_arrays, render_records, temporal_split
from utils import Logger, quantile_normalize, save_json, set_seed, write_jsonl


def parse_args() -> argparse.Namespace:
    cf = C.COUNTERFACTUAL
    pc = C.PREPROCESS
    p = argparse.ArgumentParser(description="full-sequence counterfactual replacement, then windowing")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--emb", default=None)
    p.add_argument("--tau", type=float, default=cf.tau)
    p.add_argument("--core", type=int, default=None)
    p.add_argument("--n_train", type=int, default=None)
    p.add_argument("--n_eval", type=int, default=None)
    p.add_argument("--retrieval_topk", type=int, default=cf.retrieval_topk)
    p.add_argument("--cold_scope", default=cf.retrieval_cold_scope, help="all | bottomN")
    p.add_argument("--retrieval_pop_groups", type=int, nargs="+", default=None)
    p.add_argument("--pop_norm", default=cf.pop_norm,
                   choices=["n_rank", "rank_hot", "log_minmax", "minmax"])
    p.add_argument("--profile_norm", default=cf.profile_norm,
                   choices=["unit_mean", "raw_mean"])
    p.add_argument("--no_quantile_mismatch", action="store_true")
    p.add_argument("--chunk_size", type=int, default=2048)
    p.add_argument("--n_preview", type=int, default=12)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=pc.seed)
    p.add_argument("--stats_only", action="store_true",
                   help="score only and print replacement rates by tau; no retrieval or cf jsonl")
    p.add_argument("--score_pop_only", action="store_true",
                   help="ablation: S=popularity, drop mismatch; still replace hot items with S>tau")
    p.add_argument("--random_retrieve", action="store_true",
                   help="ablation: keep full S for positions; sample substitutes uniformly from the cold pool")
    return p.parse_args()


def build_pop_factor(items: ItemTable, mode: str) -> np.ndarray:
    if mode == "n_rank":
        return np.asarray(items.n_rank, dtype=np.float32)
    if mode == "minmax":
        return np.asarray(items.n_minmax, dtype=np.float32)
    if mode == "log_minmax":
        return np.asarray(items.n_log_minmax, dtype=np.float32)

    freq = np.asarray(items.freq, dtype=np.float64)
    hot = np.flatnonzero(np.asarray(items.is_hot, dtype=bool))
    out = np.zeros(len(freq), dtype=np.float32)
    if len(hot) == 1:
        out[hot] = 1.0
    elif len(hot) > 1:
        order = np.lexsort((hot, freq[hot]))
        ranks = np.empty(len(hot), dtype=np.float64)
        ranks[order] = np.arange(len(hot))
        out[hot] = (ranks / (len(hot) - 1)).astype(np.float32)
    return out


def unit_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norm, eps)


def load_embeddings(path: Path, n_items: int, log: Logger) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"not found: {path}. Run python build_item_emb.py")
    emb = np.load(path).astype(np.float32)
    if emb.shape[0] != n_items:
        raise SystemExit(f"embedding rows {emb.shape[0]} ≠ n_items {n_items}")
    log(f"embeddings: {emb.shape} | norm mean {np.linalg.norm(emb, axis=1).mean():.4f}")
    return emb


def _time_bounds_from_stats(processed: Path) -> Tuple[Optional[int], Optional[int]]:
    path = processed / "stats.json"
    if not path.exists():
        return None, None
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    tw = stats.get("time_window") or {}
    ts_lo, ts_hi = tw.get("ts_lo"), tw.get("ts_hi")
    if ts_lo is None or ts_hi is None:
        return None, None
    return int(ts_lo), int(ts_hi)


def load_kcore_stream(ds: C.DatasetConfig, pc: C.PreprocessConfig, core: int,
                      log: Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    ts_lo, ts_hi = _time_bounds_from_stats(ds.processed_dir)
    packed = load_kcore_arrays(ds, pc, core, ts_lo=ts_lo, ts_hi=ts_hi,
                               keep_meta=False, log_fn=log)
    return (packed["users"], packed["items"], packed["ts"],
            packed["n_users"], packed["n_items"])


def score_user_sequences(items_s: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                         emb: np.ndarray, profile_norm: str,
                         log: Logger) -> Tuple[Dict[str, np.ndarray], int]:
    base = emb if profile_norm == "raw_mean" else unit_rows(emb)
    gidx_out: List[np.ndarray] = []
    cos_out: List[np.ndarray] = []
    n_positions = 0
    n_users = len(starts)
    for u in range(n_users):
        s, e = int(starts[u]), int(ends[u])
        length = e - s
        n_positions += max(length - 1, 0)
        if length < 2:
            continue
        seq = items_s[s:e]
        vecs = base[seq]
        weights = np.arange(1, length + 1, dtype=np.float32)
        weighted = np.cumsum(vecs * weights[:, None], axis=0)
        denom = np.cumsum(weights)[:, None]
        profiles = unit_rows(weighted[:-1] / denom[:-1])
        targets = unit_rows(vecs[1:])
        cos = (profiles * targets).sum(-1)
        loc = np.arange(length - 1)
        gidx_out.append(s + loc + 1)
        cos_out.append(cos.astype(np.float32))
        if (u + 1) % 100000 == 0:
            log(f"  scored users {u + 1:,}/{n_users:,}")
    cand = {
        "gidx": np.concatenate(gidx_out) if gidx_out else np.zeros(0, np.int64),
        "cos": np.concatenate(cos_out) if cos_out else np.zeros(0, np.float32),
    }
    return cand, n_positions


def suspicion_scores(item_ids: np.ndarray, cos: np.ndarray, pop: np.ndarray,
                     quantile_mismatch: bool) -> Dict[str, np.ndarray]:
    mismatch = (1.0 - cos) / 2.0
    if quantile_mismatch and len(mismatch) > 1:
        mismatch_n = quantile_normalize(torch.from_numpy(mismatch)).numpy()
    else:
        mismatch_n = mismatch
    return {
        "item": item_ids,
        "mismatch": mismatch,
        "mismatch_n": mismatch_n.astype(np.float32),
        "pop": pop[item_ids],
        "score": (pop[item_ids] * mismatch_n).astype(np.float32),
        "cos": cos,
    }


def recency_prefix_profiles(items_s: np.ndarray, starts: np.ndarray, gidx: np.ndarray,
                            emb: np.ndarray, profile_norm: str) -> np.ndarray:
    base = emb if profile_norm == "raw_mean" else unit_rows(emb)
    out = np.zeros((len(gidx), emb.shape[1]), dtype=np.float32)
    for i, g in enumerate(gidx):
        g = int(g)
        start = int(starts[np.searchsorted(starts, g, side="right") - 1])
        prefix = items_s[start:g]
        t = len(prefix)
        weights = np.arange(1, t + 1, dtype=np.float32)
        out[i] = (base[prefix] * weights[:, None]).sum(0) / weights.sum()
    return out


def retrieve_cold(queries: np.ndarray, cold_ids: np.ndarray, bank: torch.Tensor,
                  exclude: List[set], topk: int, rng: np.random.Generator) -> np.ndarray:
    margin = (max(len(s) for s in exclude) if exclude else 0) + topk
    k = int(min(topk + margin, len(cold_ids)))
    sims = torch.from_numpy(unit_rows(queries)).to(bank.device) @ bank.T
    idx = sims.topk(k, dim=-1).indices.cpu().numpy()

    out = np.empty(len(queries), dtype=np.int64)
    for row in range(idx.shape[0]):
        cand = cold_ids[idx[row]]
        banned = exclude[row]
        pool = [int(c) for c in cand if int(c) not in banned][:topk]
        if not pool:
            pool = [int(cand[0])]
        out[row] = pool[int(rng.integers(len(pool)))]
        banned.add(int(out[row]))
    return out


def retrieve_random(n: int, cold_ids: np.ndarray, exclude: List[set],
                    rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n, dtype=np.int64)
    n_cold = len(cold_ids)
    for i in range(n):
        banned = exclude[i]
        pick = None
        for _ in range(64):
            cand = int(cold_ids[int(rng.integers(n_cold))])
            if cand not in banned:
                pick = cand
                break
        if pick is None:
            pool = [int(c) for c in cold_ids if int(c) not in banned]
            pick = int(pool[int(rng.integers(len(pool)))]) if pool else int(cold_ids[0])
        out[i] = pick
        banned.add(pick)
    return out


def user_index_of(starts: np.ndarray, gidx: np.ndarray) -> np.ndarray:
    return np.searchsorted(starts, gidx, side="right") - 1


def annotate_split(orig_rows: List[dict], cf_rows: List[dict], win: Dict[str, np.ndarray],
                   idx: np.ndarray, edits: Dict[int, dict], max_hist: int,
                   s_at: np.ndarray) -> List[dict]:
    rows = []
    for k, wi in enumerate(idx):
        gidx = int(win["w_gidx"][int(wi)])
        hist_len = min(int(win["w_pos"][int(wi)]), max_hist)
        left = gidx - hist_len
        changes = []
        for g in range(left, gidx + 1):
            if g not in edits:
                continue
            e = dict(edits[g])
            e["pos"] = g - left
            e["is_target"] = g == gidx
            changes.append(e)
        row = dict(cf_rows[k])
        row["history_cf"] = list(row["history"])
        row["input_cf"] = row["input"]
        row["history_orig"] = orig_rows[k]["history"]
        row["target_orig"] = orig_rows[k]["target"]
        row["output_orig"] = orig_rows[k]["output"]
        row["v_y"] = float(s_at[gidx])
        row["replaced"] = changes
        row["n_replaced"] = len(changes)
        row["target_replaced"] = any(c["is_target"] for c in changes)
        row["has_cf"] = len(changes) > 0
        rows.append(row)
    return rows


SCORE_TAU_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def score_percentiles(arr: np.ndarray, pcts: Sequence[int]) -> dict:
    if len(arr) == 0:
        return {}
    return {f"p{p}": round(float(np.percentile(arr, p)), 6) for p in pcts}


def log_score_diagnostics(score: np.ndarray, hot_mask: np.ndarray, log: Logger) -> dict:
    pcts = [1, 5, 25, 50, 75, 90, 95, 99]
    hot_scores = score[hot_mask] if len(score) else score
    n_hot = int(hot_mask.sum()) if len(hot_mask) else 0
    stats = {
        "n_scored": int(len(score)),
        "n_hot_scored": n_hot,
        "score_percentiles_all": score_percentiles(score, pcts),
        "score_percentiles_hot": score_percentiles(hot_scores, pcts),
        "hot_replace_rate_by_tau": {},
    }

    def _fmt(d: dict) -> str:
        return " ".join(f"{k}={d[k]:.4f}" for k in ("p50", "p90", "p95", "p99") if k in d)

    log(f"  S all n={len(score):,}  {_fmt(stats['score_percentiles_all'])}")
    log(f"  S hot n={n_hot:,}  {_fmt(stats['score_percentiles_hot'])}")
    for tau in SCORE_TAU_GRID:
        n_rep = int((hot_mask & (score > tau)).sum()) if len(score) else 0
        rate = n_rep / max(n_hot, 1)
        stats["hot_replace_rate_by_tau"][f"{tau:g}"] = {
            "n": n_rep, "rate": round(float(rate), 6)}
        log(f"  τ={tau:g}: hot replace {n_rep:,}/{n_hot:,} = {rate:.2%}")
    return stats


def summarize(rows: Sequence[dict], detail: Dict[str, np.ndarray], new_id: np.ndarray,
              items: ItemTable, tau: float, n_positions: int, n_candidates: int,
              all_scores: np.ndarray) -> dict:
    n_rep = np.array([r["n_replaced"] for r in rows])
    n_tgt = np.array([int(r["target_replaced"]) for r in rows])
    score = detail["score"]
    pcts = [1, 5, 25, 50, 75, 95, 99]
    uniq, counts = np.unique(new_id, return_counts=True)
    top = np.argsort(-counts)[:10]
    new_freq = np.array([items.freq[int(i)] for i in new_id]) if len(new_id) else np.zeros(1)
    old_freq = (np.array([items.freq[int(i)] for i in detail["item"]])
                if len(new_id) else np.zeros(1))
    new_group = np.array([items.pop_group[int(i)] for i in new_id])
    tgt_cf = np.array([r["target"] for r in rows], dtype=np.int64)
    tgt_orig = np.array([r["target_orig"] for r in rows], dtype=np.int64)
    n_group = C.PREPROCESS.n_pop_group

    def _dist(ids: np.ndarray) -> Dict[str, float]:
        c = np.bincount(np.array([items.pop_group[int(i)] for i in ids]), minlength=n_group)
        return {str(g): round(float(c[g] / max(c.sum(), 1)), 4) for g in range(n_group)}

    return {
        "n_records": len(rows),
        "n_positions_evaluated": int(n_positions),
        "n_candidates_hot": int(n_candidates),
        "n_replaced_total": int(n_rep.sum()),
        "coverage_seq_with_replacement": round(float((n_rep > 0).mean()), 4),
        "target_replaced_rate": round(float(n_tgt.mean()), 4),
        "mean_replaced_per_seq": round(float(n_rep.mean()), 4),
        "replaced_per_seq_hist": {str(k): int(v) for k, v in
                                  zip(*np.unique(n_rep, return_counts=True))},
        "tau": round(float(tau), 6),
        "score_percentiles_all_candidates": {
            f"p{p}": round(float(np.percentile(all_scores, p)), 6)
            for p in pcts} if len(all_scores) else {},
        "score_percentiles_replaced": {f"p{p}": round(float(np.percentile(score, p)), 6)
                                       for p in pcts} if len(score) else {},
        "cos_percentiles_replaced": {f"p{p}": round(float(np.percentile(detail["cos"], p)), 6)
                                     for p in pcts} if len(score) else {},
        "n_above_tau": int((all_scores > tau).sum()) if len(all_scores) else 0,
        "n_distinct_replacements": int(len(uniq)),
        "replacement_reuse_max": int(counts.max()) if len(counts) else 0,
        "top_replacements": [{"title": items.titles[int(uniq[i])],
                              "freq": int(items.freq[int(uniq[i])]),
                              "used": int(counts[i])} for i in top],
        "freq_replaced_median": round(float(np.median(old_freq)), 1),
        "freq_substitute_median": round(float(np.median(new_freq)), 1),
        "freq_drop_ratio_median": round(float(np.median(old_freq) /
                                              max(np.median(new_freq), 1e-9)), 2),
        "substitute_pop_group_dist": {str(g): round(float((new_group == g).mean()), 4)
                                      for g in range(n_group)} if len(new_id) else {},
        "target_orig_pop_group_dist": _dist(tgt_orig) if len(rows) else {},
        "target_cf_pop_group_dist": _dist(tgt_cf) if len(rows) else {},
        "category_match_rate": round(float(np.mean(
            [items.category_ids[int(new_id[i])] == items.category_ids[int(detail["item"][i])]
             for i in range(len(new_id))])), 4) if len(new_id) else 0.0,
    }


def preview(rows: Sequence[dict], items: ItemTable, n: int, log: Logger) -> None:
    shown = 0
    for row in rows:
        if not row["replaced"]:
            continue
        log(f"  --- user {row['user']} | y_orig={row['output_orig']!r} | y_cf={row['output']!r}")
        for e in row["replaced"]:
            where = "target" if e["is_target"] else f"pos {e['pos']}"
            log(f"      {where}: \"{e['orig_title']}\" (freq "
                f"{items.freq[e['orig_id']]}) -> \"{e['new_title']}\" (freq "
                f"{items.freq[e['new_id']]}) | S={e['score']:.3f} "
                f"pop={e['pop']:.3f} mismatch={e['mismatch']:.3f}")
        shown += 1
        if shown >= n:
            break


def sample_split(idx: np.ndarray, quota: int | None, rng: random.Random) -> np.ndarray:
    if quota is None or len(idx) <= quota:
        return idx
    picked = np.array(sorted(rng.sample(range(len(idx)), quota)))
    return idx[picked]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)

    ds = C.get_dataset(args.dataset)
    pc = C.PREPROCESS
    core = args.core if args.core is not None else ds.core
    n_train = args.n_train
    n_eval = args.n_eval
    groups = (tuple(args.retrieval_pop_groups) if args.retrieval_pop_groups
              else C.cold_groups(args.cold_scope))
    cf = dc_replace(C.COUNTERFACTUAL, tau=args.tau,
                    retrieval_topk=args.retrieval_topk,
                    retrieval_cold_scope=args.cold_scope,
                    retrieval_pop_groups=groups,
                    pop_norm=args.pop_norm, profile_norm=args.profile_norm,
                    quantile_norm_mismatch=not args.no_quantile_mismatch)

    processed = ds.processed_dir
    pop_path = processed / "popularity.jsonl"
    if not pop_path.exists():
        raise SystemExit(f"not found: {pop_path}. Run python preprocess.py --dataset {args.dataset}")

    out_dir = Path(args.out_dir) if args.out_dir else C.cf_data_dir(args.dataset, args.tau)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "cf_data_log.txt")
    log(f"=== build counterfactual data | dataset={args.dataset} ===")
    log("score all t>=2; replace only hot & S>tau; v_y = full-sequence S at window target")
    log(f"S = pop * (1-cos)/2 | tau={cf.tau} | pop_norm={cf.pop_norm} | w_j=j | scope={cf.retrieval_cold_scope}")

    items = ItemTable(pop_path)
    log(f"items {len(items):,} | hot {len(items.hot_ids):,} | cold {len(items.cold_ids):,}")
    emb = load_embeddings(Path(args.emb) if args.emb else processed / "item_emb.npy",
                          len(items), log)
    pop = build_pop_factor(items, cf.pop_norm)
    hot_pop = pop[np.asarray(items.hot_ids)]
    log(f"pop factor ({cf.pop_norm}): all mean {pop.mean():.4f} median {np.median(pop):.4f} | "
        f"hot mean {hot_pop.mean():.4f} median {np.median(hot_pop):.4f}")

    users, item_ids, ts, n_users, n_items = load_kcore_stream(ds, pc, core, log)
    if n_items != len(items):
        raise SystemExit(
            f"k-core n_items {n_items} ≠ popularity.jsonl {len(items)}, rerun preprocess with the same --core")

    log("building original windows (sort key = original item id, timestamps unchanged)")
    win = build_windows(users, item_ids, ts, n_users, pc.min_hist_len)
    items_s = win["items_s"]
    starts, ends = win["starts"], win["ends"]
    log(f"  users with windows: {len(np.unique(win['w_user'])):,} | "
        f"windows {len(win['w_gidx']):,}")

    is_hot = np.asarray(items.is_hot, dtype=bool)
    log("scoring all t>=2 positions on original full-user sequences")
    cand, n_positions = score_user_sequences(
        items_s, starts, ends, emb, cf.profile_norm, log)
    item_at = items_s[cand["gidx"]] if len(cand["gidx"]) else np.zeros(0, np.int64)
    detail = suspicion_scores(item_at, cand["cos"], pop, cf.quantile_norm_mismatch)
    if args.score_pop_only:
        detail["score"] = np.asarray(detail["pop"], dtype=np.float32)
        log("score_pop_only: S = pop_factor (mismatch dropped)")
    hot_at = is_hot[item_at] if len(item_at) else np.zeros(0, dtype=bool)
    log(f"  prefix positions {n_positions:,} | scored {len(cand['gidx']):,} "
        f"| hot among scored {int(hot_at.sum()):,}")
    score_diag = log_score_diagnostics(detail["score"], hot_at, log)

    s_at = np.zeros(len(items_s), dtype=np.float32)
    if len(cand["gidx"]):
        s_at[cand["gidx"]] = detail["score"]

    chosen = (hot_at & (detail["score"] > cf.tau)
              if len(detail["score"]) else np.zeros(0, dtype=bool))
    sel = np.flatnonzero(chosen)
    log(f"  tau={cf.tau:.4f} | replace {len(sel):,} / {int(hot_at.sum()):,} hot "
        f"({len(sel) / max(int(hot_at.sum()), 1):.1%}); cold never replaced")

    if args.stats_only:
        save_json(out_dir / "cf_stats.json", {
            "dataset": args.dataset,
            "seed": args.seed,
            "core": core,
            "stats_only": True,
            "score_pop_only": bool(args.score_pop_only),
            "random_retrieve": bool(args.random_retrieve),
            "score": ("S = pop_factor" if args.score_pop_only
                      else "S = pop_factor * (1 - cos(recency_weighted_profile, item_emb)) / 2"),
            "replace_rule": "hot and S>tau",
            "config": {
                "tau": cf.tau, "pop_norm": cf.pop_norm,
                "profile_norm": cf.profile_norm,
                "quantile_norm_mismatch": cf.quantile_norm_mismatch,
            },
            "n_positions_evaluated": int(n_positions),
            "score_diagnostics": score_diag,
        })
        log(f"stats_only -> {out_dir / 'cf_stats.json'}")
        log.close()
        return

    gidx_s = cand["gidx"][sel]
    detail_s = {k: v[sel] for k, v in detail.items()}

    has_emb = np.linalg.norm(emb, axis=1) > 0
    group_set = set(cf.retrieval_pop_groups)
    cold = np.array([i for i in items.cold_ids
                     if has_emb[i] and items.pop_group[i] in group_set
                     and not (cf.exclude_zero_freq_from_retrieval and items.freq[i] == 0)],
                    dtype=np.int64)
    if len(cold) == 0:
        raise SystemExit(f"retrieval_pop_groups={cf.retrieval_pop_groups} : candidate pool is empty")
    pool_freq = np.array([items.freq[i] for i in cold])
    log(f"cold candidate pool: {len(cold):,} | pop_group {sorted(group_set)} | "
        f"freq median {np.median(pool_freq):.0f} max {pool_freq.max()}")

    device = torch.device(args.device if args.device else
                          (C.RUNTIME.device if torch.cuda.is_available() else "cpu"))
    log(f"device: {device}")

    items_cf = items_s.copy()
    new_id = np.zeros(0, dtype=np.int64)
    edits: Dict[int, dict] = {}
    if len(sel):
        uid = user_index_of(starts, gidx_s)
        exclude: List[set] = []
        used: Dict[int, set] = {}
        for u in uid:
            u = int(u)
            base = used.setdefault(u, set(items_s[int(starts[u]):int(ends[u])].tolist()))
            exclude.append(base)
        new_id = np.empty(len(sel), dtype=np.int64)
        if args.random_retrieve:
            log("random_retrieve: sample substitutes uniformly from the cold pool")
            new_id = retrieve_random(len(sel), cold, exclude, rng_np)
        else:
            queries = recency_prefix_profiles(items_s, starts, gidx_s, emb, cf.profile_norm)
            bank = torch.from_numpy(unit_rows(emb[cold])).to(device)
            for start in range(0, len(sel), args.chunk_size):
                stop = min(start + args.chunk_size, len(sel))
                new_id[start:stop] = retrieve_cold(
                    queries[start:stop], cold, bank, exclude[start:stop],
                    cf.retrieval_topk, rng_np)
                if stop % (args.chunk_size * 8) == 0 or stop == len(sel):
                    log(f"  retrieved {stop:,}/{len(sel):,}")
        items_cf[gidx_s] = new_id
        for i, g in enumerate(gidx_s):
            g = int(g)
            edits[g] = {
                "orig_id": int(detail_s["item"][i]),
                "orig_title": items.titles[int(detail_s["item"][i])],
                "new_id": int(new_id[i]),
                "new_title": items.titles[int(new_id[i])],
                "score": round(float(detail_s["score"][i]), 6),
                "pop": round(float(detail_s["pop"][i]), 6),
                "cos": round(float(detail_s["cos"][i]), 6),
                "mismatch": round(float(detail_s["mismatch_n"][i]), 6),
            }
    n_seq = int((ends - starts > 0).sum())
    if edits:
        n_seq_rep = len(np.unique(user_index_of(
            starts, np.array(list(edits.keys()), dtype=np.int64))))
    else:
        n_seq_rep = 0
    log(f"replaced {len(edits):,} positions on {n_seq_rep:,}/{n_seq:,} user sequences")

    log("temporal split")
    train_idx, valid_idx, test_idx = temporal_split(
        win["w_ts"], win["w_user"], win["w_pos"], pc.split_ratios)
    log(f"  windows train/valid/test: {len(train_idx):,} / {len(valid_idx):,} / {len(test_idx):,}")
    splits = {
        "train": sample_split(train_idx, n_train, rng_py),
        "valid": sample_split(valid_idx, n_eval, rng_py),
        "test": sample_split(test_idx, n_eval, rng_py),
    }
    for name, idx in splits.items():
        log(f"  {name}: {len(idx):,}")

    win_cf = dict(win)
    win_cf["items_s"] = items_cf
    titles = items.titles
    report: Dict[str, dict] = {}
    for name, idx in splits.items():
        idx_sorted = np.sort(idx)
        orig_rows = render_records(idx_sorted, win, titles, ds, pc.max_hist_len)
        cf_rows = render_records(idx_sorted, win_cf, titles, ds, pc.max_hist_len)
        rows = annotate_split(orig_rows, cf_rows, win, idx_sorted, edits, pc.max_hist_len, s_at)
        covered = set()
        for wi in idx_sorted:
            gidx = int(win["w_gidx"][int(wi)])
            hist_len = min(int(win["w_pos"][int(wi)]), pc.max_hist_len)
            covered.update(range(gidx - hist_len, gidx + 1))
        if covered and len(gidx_s):
            mask = np.array([int(g) in covered for g in gidx_s])
            det = {k: v[mask] for k, v in detail_s.items()}
            nid = new_id[mask] if len(new_id) else new_id
        else:
            det = {k: v[:0] for k, v in detail_s.items()} if detail_s else {
                "item": np.zeros(0, np.int64), "score": np.zeros(0, np.float32),
                "cos": np.zeros(0, np.float32), "pop": np.zeros(0, np.float32),
                "mismatch_n": np.zeros(0, np.float32)}
            nid = np.zeros(0, np.int64)
        stats = summarize(rows, det, nid, items, cf.tau, n_positions,
                          int(hot_at.sum()), detail["score"])
        vy = np.array([r["v_y"] for r in rows], dtype=np.float32)
        stats["v_y_mean"] = round(float(vy.mean()), 6) if len(vy) else 0.0
        stats["v_y_percentiles"] = score_percentiles(vy, [1, 5, 25, 50, 75, 90, 95, 99])
        n = write_jsonl(out_dir / f"cf_{name}.jsonl", rows)
        log(f"[{name}] wrote {n:,} | coverage {stats['coverage_seq_with_replacement']:.1%} | "
            f"target replaced {stats['target_replaced_rate']:.1%} | "
            f"y pop_group orig {stats['target_orig_pop_group_dist']} "
            f"cf {stats['target_cf_pop_group_dist']}")
        if args.n_preview and name == "train":
            preview(rows, items, args.n_preview, log)
        report[name] = stats

    save_json(out_dir / "cf_stats.json", {
        "dataset": args.dataset,
        "seed": args.seed,
        "core": core,
        "score": ("S = pop_factor" if args.score_pop_only
                  else "S = pop_factor * (1 - cos(recency_weighted_profile, item_emb)) / 2"),
        "replace_scope": "full_user_sequence_then_window",
        "replace_rule": "hot and S>tau",
        "score_pop_only": bool(args.score_pop_only),
        "random_retrieve": bool(args.random_retrieve),
        "profile_recency": "linear_w_j=j",
        "v_y": "S at the window target's full-sequence position; 0 if position 0",
        "config": {
            "tau": cf.tau,
            "retrieval_topk": cf.retrieval_topk,
            "retrieval_cold_scope": cf.retrieval_cold_scope,
            "retrieval_pop_groups": list(cf.retrieval_pop_groups),
            "pop_norm": cf.pop_norm, "profile_norm": cf.profile_norm,
            "profile_recency": cf.profile_recency,
            "quantile_norm_mismatch": cf.quantile_norm_mismatch,
            "exclude_zero_freq_from_retrieval": cf.exclude_zero_freq_from_retrieval,
            "max_hist_len": pc.max_hist_len,
            "min_hist_len": pc.min_hist_len,
            "split_ratios": list(pc.split_ratios),
            "n_train": int(len(splits["train"])),
            "n_valid": int(len(splits["valid"])),
            "n_test": int(len(splits["test"])),
            "n_train_cap": n_train,
            "n_eval_cap": n_eval,
        },
        "n_user_positions_replaced": len(edits),
        "score_diagnostics": score_diag,
        "splits": report,
    })
    log(f"saved -> {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
