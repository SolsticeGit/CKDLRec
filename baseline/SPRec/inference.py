from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SPREC_DIR = Path(__file__).resolve().parent
if str(_SPREC_DIR) not in sys.path:
    sys.path.insert(0, str(_SPREC_DIR))

from _common import (
    ROOT, build_title_trie, ckpt_sprec_dir, decode_beam_items,
    disable_sampling_flags, results_sprec_dir, unwrap_causal,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config as C
from dataset import GenerateCollator, ItemTable, SFTDataset
from utils import Logger, load_base_model, load_peft_model, save_json, set_seed, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPRec constrained decoding on the test set")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--ckpt", default=None, help="default baseline/SPRec/output/checkpoints/<ds>/best")
    p.add_argument("--output", default=None, help="default baseline/SPRec/output/results/<ds>/preds.jsonl")
    p.add_argument("--batch_size", type=int, default=C.EVAL.batch_size)
    p.add_argument("--max_seq_len", type=int, default=C.RUNTIME.max_seq_len)
    p.add_argument("--max_new_tokens", type=int, default=C.EVAL.max_new_tokens)
    p.add_argument("--num_beams", type=int, default=C.EVAL.num_beams)
    p.add_argument("--topk", type=int, nargs="+", default=list(C.EVAL.topk))
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=C.RUNTIME.num_workers)
    p.add_argument("--seed", type=int, default=C.RUNTIME.seed)
    return p.parse_args()


def resolve_ckpt(dataset: str, ckpt: str | None) -> Path:
    path = Path(ckpt) if ckpt else ckpt_sprec_dir(dataset) / "best"
    if not path.exists():
        raise SystemExit(
            f"not found: {path}. Run python baseline/SPRec/train.py --dataset {dataset}")
    if not (path / "adapter_config.json").exists() and not (path / "config.json").exists():
        raise SystemExit(f"{path} has neither adapter_config.json nor config.json")
    return path


def load_sprec_model(ckpt: Path, device: torch.device):
    base = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype)
    if (ckpt / "adapter_config.json").exists():
        model = load_peft_model(base, ckpt)
    else:
        del base
        model = load_base_model(ckpt, C.RUNTIME.dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    return model


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    ds_cfg = C.get_dataset(args.dataset)
    ckpt = resolve_ckpt(args.dataset, args.ckpt)
    k_max = max(args.topk)
    num_beams = max(args.num_beams, k_max)
    out_path = Path(args.output) if args.output else results_sprec_dir(args.dataset) / "preds.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = Logger(out_path.parent / "infer_log.txt")
    log(f"=== SPRec inference | dataset={args.dataset} ckpt={ckpt} ===")

    device = torch.device(C.RUNTIME.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(C.BASE_MODEL_PATH))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    processed = ds_cfg.processed_dir
    test_path = processed / "test.jsonl"
    if not test_path.exists():
        raise SystemExit(f"not found: {test_path}. Run python preprocess.py --dataset {args.dataset}")

    item_table = ItemTable(processed / "popularity.jsonl")
    log(f"items={item_table.n_items:,} beams={num_beams} topk={args.topk}")
    log("building title prefix trie")
    trie = build_title_trie(item_table, tokenizer, eos_id, pad_id, log)

    model = load_sprec_model(ckpt, device)
    disable_sampling_flags(model)
    unwrap = unwrap_causal(model)
    for module in (model, unwrap):
        module.config.pad_token_id = pad_id
        module.config.eos_token_id = eos_id
        module.config.use_cache = True

    dataset = SFTDataset(test_path, tokenizer, ds_cfg, item_table,
                         args.max_seq_len, mode="generate", history_field="history")
    if args.max_samples:
        dataset.records = dataset.records[:args.max_samples]
    log(f"test samples={len(dataset)}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=GenerateCollator(pad_id),
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )

    preds = []
    n_done = 0
    n_full = 0
    n_batch = 0
    for batch in loader:
        n_batch += 1
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        prompt_width = input_ids.size(1)

        def prefix_allowed_tokens_fn(_batch_id: int, row: torch.Tensor) -> list[int]:
            return trie.allowed_tokens(row[prompt_width:].tolist())

        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            early_stopping=True,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        recs = decode_beam_items(
            sequences, prompt_width, num_beams, trie, item_table, tokenizer, pad_id)
        target_ids = batch["target_id"].tolist()
        for i, rec in enumerate(recs):
            n_full += int(len(rec) >= k_max)
            preds.append({
                "target_id": int(target_ids[i]),
                "target_title": batch["target_title"][i],
                "rec_ids": [r["item_id"] for r in rec],
                "rec_titles": [r["title"] for r in rec],
                "rec_pop_group": [r["pop_group"] for r in rec],
            })
        n_done += len(recs)
        if n_batch % 10 == 0 or n_done == len(dataset):
            log(f"  generated {n_done:,}/{len(dataset):,}  "
                f"full_top{k_max}={n_full / n_done:.3f}")

    write_jsonl(out_path, preds)
    save_json(out_path.with_name("infer_meta.json"), {
        "method": "sprec",
        "dataset": args.dataset,
        "ckpt": str(ckpt),
        "n_samples": len(preds),
        "n_items": item_table.n_items,
        "topk": args.topk,
        "num_beams": num_beams,
        "max_new_tokens": args.max_new_tokens,
        "constrained_decoding": True,
        "exclude_history": False,
        "full_topk_rate": n_full / len(preds) if preds else 0.0,
    })
    log(f"wrote {len(preds)} preds -> {out_path}")
    log.close()


if __name__ == "__main__":
    main()
