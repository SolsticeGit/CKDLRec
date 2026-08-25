from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_D3_DIR = Path(__file__).resolve().parent
if str(_D3_DIR) not in sys.path:
    sys.path.insert(0, str(_D3_DIR))

from _common import (
    ROOT, ckpt_d3_dir, flower_sasrec_path, require_sft_ckpt,
)
from sasrec import save_sasrec, train_sasrec

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from dataset import ItemTable
from utils import Logger, load_jsonl, save_json, set_seed

SASREC_EPOCHS = 200
SASREC_HIDDEN = 64
SASREC_DROPOUT = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DecodingMatters: reuse Flower SASRec (train only if missing)")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--seed", type=int, default=C.RUNTIME.seed)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sft_ckpt", default=None,
                   help="path only; default baseline/sft/output/checkpoints/<ds>/best")
    p.add_argument("--sasrec_epochs", type=int, default=SASREC_EPOCHS)
    p.add_argument("--sasrec_hidden", type=int, default=SASREC_HIDDEN)
    p.add_argument("--sasrec_dropout", type=float, default=SASREC_DROPOUT)
    p.add_argument("--state_size", type=int, default=C.PREPROCESS.max_hist_len)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ds_cfg = C.get_dataset(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_d3_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train_log.txt")
    log(f"=== DecodingMatters | dataset={args.dataset} ===")

    sft_ckpt = require_sft_ckpt(args.dataset, args.sft_ckpt)
    log(f"reuse SFT adapter {sft_ckpt}")

    flower = flower_sasrec_path(args.dataset)
    if flower.exists():
        save_json(out_dir / "train_config.json", {
            "method": "DecodingMatters",
            "dataset": args.dataset,
            "sft_ckpt": str(sft_ckpt),
            "sasrec": str(flower),
            "sasrec_source": "Flower",
            "seed": args.seed,
        })
        log(f"reuse Flower SASRec {flower} (not retrained)")
        log.close()
        return

    log(f"Flower SASRec not found at {flower}; training a local copy")
    device = torch.device(C.RUNTIME.device if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    processed = ds_cfg.processed_dir
    train_path = processed / "train.jsonl"
    valid_path = processed / "valid.jsonl"
    pop_path = processed / "popularity.jsonl"
    if not train_path.exists() or not pop_path.exists():
        raise SystemExit(f"not found: {train_path} or {pop_path}. Run "
                         f"python preprocess.py --dataset {args.dataset}")

    item_table = ItemTable(pop_path)
    train_records = load_jsonl(train_path)
    valid_records = load_jsonl(valid_path)
    log(f"items={item_table.n_items} train={len(train_records)} valid={len(valid_records)}")

    sasrec = train_sasrec(
        train_records, valid_records, item_table.n_items, args.state_size,
        device, log, epochs=args.sasrec_epochs, hidden=args.sasrec_hidden,
        dropout=args.sasrec_dropout, seed=args.seed)
    sasrec_path = out_dir / "sasrec.pt"
    save_sasrec(sasrec, sasrec_path)
    log(f"wrote {sasrec_path}")

    save_json(out_dir / "train_config.json", {
        "method": "DecodingMatters",
        "dataset": args.dataset,
        "sft_ckpt": str(sft_ckpt),
        "sasrec": str(sasrec_path),
        "sasrec_source": "local",
        "seed": args.seed,
        "state_size": args.state_size,
        "sasrec_epochs": args.sasrec_epochs,
        "sasrec_hidden": args.sasrec_hidden,
        "data_dir": str(processed),
        "base_model": str(C.BASE_MODEL_PATH),
    })
    log(f"done | local SASRec {sasrec_path} | LLM={sft_ckpt}")
    log.close()


if __name__ == "__main__":
    main()
