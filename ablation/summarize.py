from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ("HR", "NDCG", "DivRatio", "ORRatio", "MGU", "DGU")
SKIP_METHODS = {"pop_only_teacher", "rand_ret_teacher"}
ORDER = (
    "full",
    "teacher",
    "no_kd",
    "no_sft",
    "no_adv",
    "pop_only",
    "rand_ret",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize CKDLRec ablation metrics")
    p.add_argument("--results_root", default="ablation/output/results")
    p.add_argument("--full_root", default=None,
                   help="main-run outputs/results, used as the full row")
    p.add_argument("--run_tag", default="",
                   help="main-run directory name, e.g. tau_0.4_sft_0.4_adv_0.2")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--k", type=int, nargs="+", default=None,
                   help="only these K; default: all topk in metrics")
    p.add_argument("--output", default="ablation/output/ablation_summary.json")
    p.add_argument("--txt", default="ablation/output/ablation_summary.txt")
    return p.parse_args()


def load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def row_at_k(metrics: dict, k: int) -> dict | None:
    at_k = metrics.get("at_k") or {}
    row = at_k.get(str(k))
    if not row:
        return None
    out = {m: row.get(m) for m in METRICS}
    out["n_samples"] = metrics.get("n_samples")
    return out


def method_sort_key(name: str) -> tuple:
    try:
        return (0, ORDER.index(name), name)
    except ValueError:
        return (1, 0, name)


def discover_datasets(results_root: Path, extra: list[str] | None) -> list[str]:
    if extra:
        return list(extra)
    if not results_root.is_dir():
        return []
    return sorted(p.name for p in results_root.iterdir() if p.is_dir())


def collect(args: argparse.Namespace) -> dict:
    results_root = Path(args.results_root)
    datasets = discover_datasets(results_root, args.datasets)
    summary: dict = {"datasets": {}, "metrics": list(METRICS)}

    for ds in datasets:
        methods: dict[str, dict] = {}
        ds_dir = results_root / ds
        names = []
        if ds_dir.is_dir():
            names = [p.name for p in ds_dir.iterdir()
                     if p.is_dir() and p.name not in SKIP_METHODS]
        if args.full_root and args.run_tag:
            names = ["full", *names]
        names = sorted(set(names), key=method_sort_key)

        ks: list[int] = []
        for name in names:
            if name == "full":
                path = Path(args.full_root) / ds / args.run_tag / "metrics.json"
            else:
                path = ds_dir / name / "metrics.json"
            blob = load_metrics(path)
            if blob is None:
                continue
            if args.k:
                topk = list(args.k)
            else:
                topk = [int(x) for x in blob.get("topk") or []]
            for k in topk:
                if k not in ks:
                    ks.append(k)
                row = row_at_k(blob, k)
                if row is None:
                    continue
                methods.setdefault(name, {})[str(k)] = row
        if methods:
            summary["datasets"][ds] = {"topk": ks, "methods": methods}
    return summary


def fmt(v) -> str:
    if v is None:
        return f"{'-':>10}"
    if isinstance(v, float):
        return f"{v:10.6f}"
    return f"{v:>10}"


def format_table(summary: dict) -> str:
    lines: list[str] = []
    header = f"{'method':<20}" + "".join(f"{m:>10}" for m in METRICS)
    for ds, block in summary["datasets"].items():
        for k in block["topk"]:
            lines.append(f"########## {ds}  K={k} ##########")
            lines.append(header)
            for name in sorted(block["methods"], key=method_sort_key):
                row = block["methods"][name].get(str(k))
                if not row:
                    continue
                cells = "".join(fmt(row.get(m)) for m in METRICS)
                lines.append(f"{name:<20}{cells}")
            lines.append("")
    return "\n".join(lines).rstrip() + ("\n" if summary["datasets"] else "no metrics found\n")


def main() -> None:
    args = parse_args()
    summary = collect(args)
    text = format_table(summary)
    out = Path(args.output)
    txt = Path(args.txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    txt.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    print(f"wrote {txt}")


if __name__ == "__main__":
    main()
