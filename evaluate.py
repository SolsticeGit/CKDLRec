from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import config as C
from dataset import ItemTable
from utils import Logger, load_jsonl, normalize_title, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CKDLRec metrics on constrained-decoding preds")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--preds", default=None,
                   help="default outputs/results/<ds>/<run_tag>/preds.jsonl")
    p.add_argument("--output", default=None, help="default: metrics.json next to preds")
    p.add_argument("--tau", type=float, default=C.COUNTERFACTUAL.tau)
    p.add_argument("--alpha", type=float, default=C.CKDLREC.alpha)
    p.add_argument("--beta", type=float, default=C.CKDLREC.beta)
    p.add_argument("--topk", type=int, nargs="+", default=list(C.EVAL.topk))
    p.add_argument("--oratio_n", type=int, default=3,
                   help="ORRatio: count slots occupied by the n most frequent recommended titles (default 3)")
    return p.parse_args()


class TitleCatalog:

    def __init__(self, items: ItemTable):
        self.titles: list[str] = []
        self.title2idx: dict[str, int] = {}
        self.canonical_id: list[int] = []
        self.pop_group: list[int] = []
        for item_id, raw in enumerate(items.titles):
            key = normalize_title(raw)
            if not key or key in self.title2idx:
                continue
            self.title2idx[key] = len(self.titles)
            self.titles.append(key)
            self.canonical_id.append(item_id)
            self.pop_group.append(items.pop_group[item_id])
        self.n_titles = len(self.titles)

    def group(self, text: str) -> int | None:
        i = self.title2idx.get(normalize_title(text), -1)
        return None if i < 0 else self.pop_group[i]


def norm_list(titles: list[str]) -> list[str]:
    return [normalize_title(t) for t in titles]


def hr_at_k(target_titles: list[str], rec_titles: list[list[str]], k: int) -> float:
    if not target_titles:
        return 0.0
    hits = sum(1 for g, rec in zip(target_titles, rec_titles) if g and g in rec[:k])
    return hits / len(target_titles)


def ndcg_at_k(target_titles: list[str], rec_titles: list[list[str]], k: int) -> float:
    if not target_titles:
        return 0.0
    total = 0.0
    for g, rec in zip(target_titles, rec_titles):
        if not g:
            continue
        try:
            rank = rec[:k].index(g) + 1
        except ValueError:
            continue
        total += 1.0 / math.log2(rank + 1)
    return total / len(target_titles)


def div_ratio_at_k(rec_titles: list[list[str]], k: int, n_titles: int) -> float:
    if n_titles <= 0:
        return 0.0
    uniq: set[str] = set()
    for rec in rec_titles:
        uniq.update(t for t in rec[:k] if t)
    return len(uniq) / n_titles


def or_ratio_at_k(rec_titles: list[list[str]], k: int, catalog: TitleCatalog,
                  n_top: int = 3) -> tuple[float, list[dict]]:
    counts: Counter[str] = Counter()
    total = 0
    for rec in rec_titles:
        for t in rec[:k]:
            if t:
                counts[t] += 1
                total += 1
    if total == 0 or n_top <= 0:
        return 0.0, []
    top = counts.most_common(n_top)
    occupied = sum(c for _, c in top)
    details = []
    for title, count in top:
        i = catalog.title2idx.get(title, -1)
        details.append({
            "title": title,
            "item_id": catalog.canonical_id[i] if i >= 0 else -1,
            "count": int(count),
            "pop_group": catalog.pop_group[i] if i >= 0 else None,
        })
    return occupied / total, details


def group_share(titles: list[str], catalog: TitleCatalog, n_group: int
                ) -> tuple[list[int], list[float]]:
    counts = [0] * n_group
    for t in titles:
        g = catalog.group(t)
        if g is None:
            continue
        counts[g] += 1
    total = sum(counts)
    props = [c / total if total else 0.0 for c in counts]
    return counts, props


def fairness_at_k(rec_titles: list[list[str]], k: int, hist_titles: list[str],
                  catalog: TitleCatalog, n_group: int) -> dict:
    rec_flat = [t for rec in rec_titles for t in rec[:k] if t]
    n_rec, gp = group_share(rec_flat, catalog, n_group)
    n_hist, gh = group_share(hist_titles, catalog, n_group)
    gu = [gp[g] - gh[g] for g in range(n_group)]
    mgu = sum(abs(x) for x in gu) / n_group
    dgu = max(gu) - min(gu) if gu else 0.0
    groups = {
        str(g): {
            "n_rec": n_rec[g],
            "gp": gp[g],
            "n_hist": n_hist[g],
            "gh": gh[g],
            "gu": gu[g],
        }
        for g in range(n_group)
    }
    return {
        "MGU": mgu,
        "DGU": dgu,
        "n_rec_total": sum(n_rec),
        "n_hist_total": sum(n_hist),
        "groups": groups,
    }


def align_split(preds: list[dict], records: list[dict]) -> None:
    if len(preds) > len(records):
        raise SystemExit(f"preds ({len(preds)}) is longer than test.jsonl ({len(records)})")
    for i, (pred, rec) in enumerate(zip(preds, records)):
        if int(pred["target_id"]) != int(rec["target"]):
            raise SystemExit(
                f"row {i}: target mismatch: pred={pred['target_id']} test={rec['target']}")


def format_metrics(metrics: dict) -> str:
    n_group = metrics["n_pop_group"]
    lines = [
        f"dataset={metrics['dataset']}  n={metrics['n_samples']}  "
        f"|T|={metrics['n_titles']}  |I|={metrics['n_items']}",
        f"GH from test histories  n_hist={metrics['n_hist_total']}",
    ]
    gh = metrics["gh"]
    lines.append("  " + "  ".join(f"G{g}={gh[g]:.4f}" for g in range(n_group)))
    header = (f"{'K':>4}  {'HR':>8}  {'NDCG':>8}  {'DivRatio':>8}  "
              f"{'ORRatio':>8}  {'MGU':>8}  {'DGU':>8}")
    lines.append(header)
    for k in metrics["topk"]:
        row = metrics["at_k"][str(k)]
        lines.append(
            f"{k:>4}  {row['HR']:.6f}  {row['NDCG']:.6f}  "
            f"{row['DivRatio']:.6f}  {row['ORRatio']:.6f}  "
            f"{row['MGU']:.6f}  {row['DGU']:.6f}"
        )
        for g in range(n_group):
            grp = row["groups"][str(g)]
            lines.append(
                f"       G{g}: n_rec={grp['n_rec']:<7} gp={grp['gp']:.4f}  "
                f"n_hist={grp['n_hist']:<7} gh={grp['gh']:.4f}  gu={grp['gu']:+.4f}"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ds_cfg = C.get_dataset(args.dataset)
    pred_path = Path(args.preds) if args.preds else C.results_dir(
        args.dataset, args.tau, args.alpha, args.beta
    ) / "preds.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"not found: {pred_path}. Run python inference.py --dataset {args.dataset}")
    out_path = Path(args.output) if args.output else pred_path.with_name("metrics.json")

    processed = ds_cfg.processed_dir
    test_path = processed / "test.jsonl"
    pop_path = processed / "popularity.jsonl"
    if not test_path.exists():
        raise SystemExit(f"not found: {test_path}")

    log = Logger(out_path.parent / "eval_log.txt")
    log(f"=== evaluate | dataset={args.dataset} preds={pred_path} | match=title ===")

    preds = load_jsonl(pred_path)
    records = load_jsonl(test_path)[:len(preds)]
    align_split(preds, records)
    items = ItemTable(pop_path)
    catalog = TitleCatalog(items)
    n_group = C.PREPROCESS.n_pop_group
    log(f"titles: unique={catalog.n_titles:,}/{items.n_items:,} items")

    target_titles = []
    rec_titles = []
    for p in preds:
        gold = p.get("target_title") or items.titles[int(p["target_id"])]
        target_titles.append(normalize_title(gold))
        rec_ids = list(map(int, p["rec_ids"]))
        titles = list(p.get("rec_titles") or [])
        if len(titles) < len(rec_ids):
            titles = titles + [items.titles[i] for i in rec_ids[len(titles):]]
        rec_titles.append(norm_list(titles))

    hist_titles = [normalize_title(items.titles[int(i)])
                   for rec in records for i in rec["history"]]
    n_hist, gh = group_share(hist_titles, catalog, n_group)

    at_k = {}
    for k in args.topk:
        fair = fairness_at_k(rec_titles, k, hist_titles, catalog, n_group)
        orr, or_top = or_ratio_at_k(rec_titles, k, catalog, args.oratio_n)
        at_k[str(k)] = {
            "HR": hr_at_k(target_titles, rec_titles, k),
            "NDCG": ndcg_at_k(target_titles, rec_titles, k),
            "DivRatio": div_ratio_at_k(rec_titles, k, catalog.n_titles),
            "ORRatio": orr,
            "OR_top_items": or_top,
            **fair,
        }

    metrics = {
        "dataset": args.dataset,
        "preds": str(pred_path),
        "match": "title",
        "n_samples": len(preds),
        "n_items": items.n_items,
        "n_titles": catalog.n_titles,
        "n_pop_group": n_group,
        "topk": args.topk,
        "n_hist_total": sum(n_hist),
        "n_hist": n_hist,
        "gh": gh,
        "oratio_n": args.oratio_n,
        "formulas": {
            "HR@K": "normalized target title in rec_titles[:K]",
            "NDCG@K": "title match at rank j contributes 1/log2(j+1); IDCG=1",
            "DivRatio@K": "|union rec titles[:K]| / |unique titles|",
            "ORRatio@K": "slots occupied by the n most frequent recommended titles "
                         f"/ all top-K slots (n={args.oratio_n})",
            "GP(G)": "rec title slots in G / all rec title slots",
            "GH(G)": "history titles in G / all history titles",
            "MGU": "mean_G |GP(G)-GH(G)|",
            "DGU": "max_G GU(G) - min_G GU(G)",
        },
        "at_k": at_k,
    }
    save_json(out_path, metrics)
    log("\n" + format_metrics(metrics))
    log(f"wrote {out_path}")
    log.close()


if __name__ == "__main__":
    main()
