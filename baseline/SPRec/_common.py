from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "output"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C


def ckpt_sprec_dir(dataset: str) -> Path:
    return OUTPUT / "checkpoints" / dataset


def results_sprec_dir(dataset: str) -> Path:
    return OUTPUT / "results" / dataset


def unwrap_causal(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def disable_sampling_flags(model) -> None:
    seen = []
    for module in (model, unwrap_causal(model)):
        cfg = getattr(module, "generation_config", None)
        if cfg is None or any(cfg is x for x in seen):
            continue
        seen.append(cfg)
        cfg.do_sample = False
        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k = None


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


def build_title_trie(item_table, tokenizer, eos_id: int, pad_id: int, log,
                     batch_size: int = 4096) -> TitlePrefixTrie:
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


def decode_beam_items(sequences, prompt_width: int, n_return: int, trie: TitlePrefixTrie,
                      item_table, tokenizer, pad_id: int) -> list[list[dict]]:
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
