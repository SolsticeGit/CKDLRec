from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config as C
from dataset import GenerateCollator, ItemTable, SFTDataset
from model import CKDLRecModel
from utils import Logger, load_base_model, load_peft_model, set_seed, write_jsonl


class TitlePrefixTrie:

    def __init__(self, eos_id: int, pad_id: int):
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.root = _TrieNode()
        self.n_inserted = 0
        self.n_empty = 0

    def insert(self, token_ids: Sequence[int], item_id: int) -> None:
        if not token_ids:
            self.n_empty += 1
            return
        node = self.root
        for tok in token_ids:
            child = node.children.get(tok)
            if child is None:
                child = _TrieNode()
                node.children[tok] = child
            node = child
        node.item_ids.append(item_id)
        self.n_inserted += 1

    def freeze(self) -> None:
        stack = [self.root]
        while stack:
            node = stack.pop()
            allow = list(node.children)
            if node.item_ids:
                allow.append(self.eos_id)
            node.allowed = allow if allow else [self.eos_id]
            stack.extend(node.children.values())

    def allowed_tokens(self, suffix: Sequence[int]) -> list[int]:
        if any(int(t) == self.eos_id for t in suffix):
            return [self.eos_id, self.pad_id]
        node = self.root
        for tok in suffix:
            node = node.children.get(int(tok))
            if node is None:
                return [self.eos_id]
        return node.allowed

    def lookup(self, suffix: Sequence[int]) -> int:
        node = self.root
        for tok in suffix:
            t = int(tok)
            if t == self.eos_id or t == self.pad_id:
                break
            node = node.children.get(t)
            if node is None:
                return -1
        return node.item_ids[0] if node.item_ids else -1


class _TrieNode:
    __slots__ = ("children", "item_ids", "allowed")

    def __init__(self):
        self.children: dict[int, _TrieNode] = {}
        self.item_ids: list[int] = []
        self.allowed: list[int] = []


def build_title_trie(item_table: ItemTable, tokenizer, eos_id: int, pad_id: int,
                     log: Logger, batch_size: int = 4096) -> TitlePrefixTrie:
    trie = TitlePrefixTrie(eos_id, pad_id)
    titles = item_table.titles
    for start in range(0, len(titles), batch_size):
        chunk = titles[start:start + batch_size]
        encoded = tokenizer(chunk, add_special_tokens=False, padding=False)
        for offset, ids in enumerate(encoded["input_ids"]):
            trie.insert(ids, start + offset)
        done = min(start + batch_size, len(titles))
        if done == len(titles) or done % (batch_size * 8) == 0:
            log(f"  title trie {done:,}/{len(titles):,}")
    trie.freeze()
    log(f"  title trie ready: inserted={trie.n_inserted:,} empty={trie.n_empty} "
        f"root_branch={len(trie.root.allowed)}")
    return trie


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CKDLRec constrained decoding on the test set")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--ckpt", default=None,
                   help="default outputs/checkpoints/ckdlrec/<ds>/<run_tag>/best")
    p.add_argument("--output", default=None,
                   help="default outputs/results/<ds>/<run_tag>/preds.jsonl")
    p.add_argument("--tau", type=float, default=C.COUNTERFACTUAL.tau)
    p.add_argument("--alpha", type=float, default=C.CKDLREC.alpha, help="L_SFT weight, used to locate run_tag")
    p.add_argument("--beta", type=float, default=C.CKDLREC.beta, help="L_adv weight, used to locate run_tag")
    p.add_argument("--grl_lambda_max", type=float, default=C.CKDLREC.grl_lambda_max)
    p.add_argument("--batch_size", type=int, default=C.EVAL.batch_size)
    p.add_argument("--max_seq_len", type=int, default=C.RUNTIME.max_seq_len)
    p.add_argument("--max_new_tokens", type=int, default=C.EVAL.max_new_tokens)
    p.add_argument("--num_beams", type=int, default=C.EVAL.num_beams)
    p.add_argument("--topk", type=int, nargs="+", default=list(C.EVAL.topk))
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=C.RUNTIME.num_workers)
    p.add_argument("--seed", type=int, default=C.RUNTIME.seed)
    return p.parse_args()


def resolve_ckpt(dataset: str, ckpt: str | None, tau: float, alpha: float, beta: float
                 ) -> Path:
    path = Path(ckpt) if ckpt else C.ckpt_ckdlrec_dir(dataset, tau, alpha, beta) / "best"
    if not path.exists():
        raise SystemExit(f"not found: {path}. Run python train_ckdlrec.py --dataset {dataset}")
    if not (path / "adapter_config.json").exists():
        raise SystemExit(f"{path} missing adapter_config.json")
    return path


def load_student(ckpt: Path, device: torch.device) -> CKDLRecModel:
    base = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype)
    llm = load_peft_model(base, ckpt)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False
    student = CKDLRecModel(llm)
    student.to(device)
    student.eval()
    return student


def decode_beam_items(sequences: torch.Tensor, prompt_width: int, n_return: int,
                      trie: TitlePrefixTrie, item_table: ItemTable, tokenizer,
                      pad_id: int) -> list[list[dict]]:
    if sequences.size(0) % n_return != 0:
        raise RuntimeError(
            f"generate output size {sequences.size(0)} is not divisible by num_return={n_return}")
    bsz = sequences.size(0) // n_return
    seqs = sequences.view(bsz, n_return, -1)
    batch_recs = []
    for i in range(bsz):
        recs = []
        seen = set()
        for j in range(n_return):
            suffix = seqs[i, j, prompt_width:].tolist()
            trimmed = [t for t in suffix if t != pad_id]
            item_id = trie.lookup(trimmed)
            if item_id < 0:
                text = tokenizer.decode(trimmed, skip_special_tokens=True).strip()
                item_id = item_table.lookup_title(text)
            if item_id < 0 or item_id in seen:
                continue
            seen.add(item_id)
            recs.append({
                "item_id": item_id,
                "title": item_table.titles[item_id],
                "pop_group": item_table.pop_group[item_id],
            })
        batch_recs.append(recs)
    return batch_recs


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    ds_cfg = C.get_dataset(args.dataset)
    ckpt = resolve_ckpt(args.dataset, args.ckpt, args.tau, args.alpha, args.beta)
    k_max = max(args.topk)
    num_beams = max(args.num_beams, k_max)
    tag = ckpt.parent.name
    out_path = Path(args.output) if args.output else C.RESULTS_DIR / args.dataset / tag / "preds.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = Logger(out_path.parent / "infer_log.txt")
    log(f"=== inference | dataset={args.dataset} ckpt={ckpt} ===")

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
    log(f"items={item_table.n_items:,} α={args.alpha} β={args.beta} "
        f"λ_max={args.grl_lambda_max} beams={num_beams} topk={args.topk}")

    log("building title prefix trie")
    trie = build_title_trie(item_table, tokenizer, eos_id, pad_id, log)

    student = load_student(ckpt, device)
    unwrap = student.llm.get_base_model() if hasattr(student.llm, "get_base_model") else student.llm
    for module in (student.llm, unwrap):
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

        sequences = student.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            early_stopping=True,
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
    meta = {
        "dataset": args.dataset,
        "ckpt": str(ckpt),
        "alpha": args.alpha,
        "beta": args.beta,
        "grl_lambda_max": args.grl_lambda_max,
        "n_samples": len(preds),
        "n_items": item_table.n_items,
        "topk": args.topk,
        "num_beams": num_beams,
        "max_new_tokens": args.max_new_tokens,
        "constrained_decoding": True,
        "exclude_history": False,
        "full_topk_rate": n_full / len(preds) if preds else 0.0,
    }
    meta_path = out_path.with_name("infer_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {len(preds)} preds -> {out_path}")
    log.close()


if __name__ == "__main__":
    main()
