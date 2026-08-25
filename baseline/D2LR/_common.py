from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from transformers import LogitsProcessor

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "output"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ckpt_d2_dir(dataset: str) -> Path:
    return OUTPUT / "checkpoints" / dataset


def results_d2_dir(dataset: str) -> Path:
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
        def dfs(node: _TrieNode) -> list[int]:
            ids = list(node.item_ids)
            for child in node.children.values():
                ids.extend(dfs(child))
            node.subtree_ids = ids
            allow = list(node.children)
            if node.item_ids:
                allow.append(self.eos_id)
            node.allowed = allow if allow else [self.eos_id]
            return ids

        dfs(self.root)

    def node_at(self, suffix: Sequence[int]) -> _TrieNode | None:
        node = self.root
        for tok in suffix:
            t = int(tok)
            if t == self.eos_id or t == self.pad_id:
                break
            node = node.children.get(t)
            if node is None:
                return None
        return node

    def allowed_tokens(self, suffix: Sequence[int]) -> list[int]:
        if any(int(t) == self.eos_id for t in suffix):
            return [self.eos_id, self.pad_id]
        node = self.node_at(suffix)
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
    __slots__ = ("children", "item_ids", "subtree_ids", "allowed")

    def __init__(self):
        self.children: dict[int, _TrieNode] = {}
        self.item_ids: list[int] = []
        self.subtree_ids: list[int] = []
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


def item_pop_scores(item_freq: Sequence[int]) -> list[float]:
    peak = max((int(f) for f in item_freq), default=1)
    peak = max(peak, 1)
    return [max(int(f), 1) / peak for f in item_freq]


def token_ips_weights(n_title_tokens: int, item_pop: float, alpha: float) -> list[float]:
    if n_title_tokens <= 0:
        return []
    p_y = max(float(item_pop), 1e-6)
    weights = []
    length = n_title_tokens
    for t_idx in range(length):
        t = t_idx + 1
        rho = p_y * (1.0 - alpha * t / length)
        rho = max(rho, 1e-6)
        weights.append(min(1.0 / rho, 50.0))
    return weights


class CRPrefixLogitsProcessor(LogitsProcessor):

    def __init__(self, prompt_width: int, trie: TitlePrefixTrie,
                 cr_probs: torch.Tensor, beta: float, n_beams: int):
        self.prompt_width = prompt_width
        self.trie = trie
        self.cr_probs = cr_probs
        self.beta = float(beta)
        self.n_beams = int(n_beams)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.beta == 0.0:
            return scores
        scores = scores.clone()
        bsz = input_ids.size(0)
        for row in range(bsz):
            sample = row // self.n_beams
            p = self.cr_probs[sample]
            suffix = input_ids[row, self.prompt_width:].tolist()
            if any(int(t) == self.trie.eos_id for t in suffix):
                continue
            node = self.trie.node_at(suffix)
            if node is None:
                continue
            for tok, child in node.children.items():
                child_mass = _mass(p, child.subtree_ids)
                scores[row, tok] = scores[row, tok] - self.beta * torch.log(
                    child_mass + 1e-12)
            if node.item_ids:
                eos_mass = _mass(p, node.item_ids)
                scores[row, self.trie.eos_id] = scores[row, self.trie.eos_id] - self.beta * (
                    torch.log(eos_mass + 1e-12))
        return scores


def _mass(probs: torch.Tensor, item_ids: Sequence[int]) -> torch.Tensor:
    if not item_ids:
        return probs.new_zeros(())
    idx = torch.tensor(item_ids, device=probs.device, dtype=torch.long)
    return probs.index_select(0, idx).sum()
