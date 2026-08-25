from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import config as C
from dataset import ItemTable
from utils import Logger, load_base_model, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="build static item embeddings e_i")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--out", default=None)
    p.add_argument("--fp32", action="store_true")
    return p.parse_args()


def _tokenize_titles(titles: list[str], tokenizer, device):
    enc = tokenizer(titles, add_special_tokens=False, padding=True, truncation=True,
                    max_length=64, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def embed_input_mean(titles: list[str], tokenizer, model, log: Logger,
                     batch_size: int) -> np.ndarray:
    emb_weight = model.get_input_embeddings().weight
    out = torch.zeros(len(titles), emb_weight.shape[1], dtype=torch.float32)
    for start in range(0, len(titles), batch_size):
        chunk = titles[start:start + batch_size]
        enc = _tokenize_titles(chunk, tokenizer, emb_weight.device)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        vecs = emb_weight[enc["input_ids"]].float()
        pooled = (vecs * mask).sum(1) / mask.sum(1).clamp_min(1)
        out[start:start + len(chunk)] = pooled.cpu()
        if (start // batch_size) % 500 == 0:
            log(f"  input_mean {min(start + batch_size, len(titles)):,}/{len(titles):,}")
    return out.numpy()


def main() -> None:
    args = parse_args()
    ds = C.get_dataset(args.dataset)
    processed = ds.processed_dir
    if not (processed / "popularity.jsonl").exists():
        raise SystemExit(f"not found: {processed / 'popularity.jsonl'}. Run "
                         f"python preprocess.py --dataset {args.dataset}")

    out_path = Path(args.out) if args.out else processed / "item_emb.npy"
    log = Logger(out_path.parent / "item_emb_log.txt")
    log(f"=== build item embeddings | dataset={args.dataset} ===")

    items = ItemTable(processed / "popularity.jsonl")
    log(f"items: {len(items):,}")

    device = torch.device(C.RUNTIME.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(C.BASE_MODEL_PATH))
    model = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype).eval()
    model.get_input_embeddings().to(device)
    emb = embed_input_mean(items.titles, tokenizer, model, log, args.batch_size)

    norms = np.linalg.norm(emb, axis=1)
    log(f"embedding shape: {emb.shape} | norm mean {norms.mean():.4f} "
        f"min {norms.min():.4f} max {norms.max():.4f}")
    if (norms == 0).any():
        log(f"WARNING: {int((norms == 0).sum())} items have zero embedding vectors")

    emb = emb.astype(np.float32 if args.fp32 else np.float16)
    np.save(out_path, emb)
    save_json(out_path.with_name(out_path.stem + "_meta.json"), {
        "dataset": args.dataset,
        "mode": "input_mean",
        "n_items": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "dtype": str(emb.dtype),
        "l2_normalized": False,
        "base_model": str(C.BASE_MODEL_PATH),
    })
    log(f"saved -> {out_path} ({emb.nbytes / 1e6:.1f} MB)")
    log.close()


if __name__ == "__main__":
    main()
