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

import config as C

DEFAULT_ALPHA = 0.5


def ckpt_d3_dir(dataset: str) -> Path:
    return OUTPUT / "checkpoints" / dataset


def results_d3_dir(dataset: str) -> Path:
    return OUTPUT / "results" / dataset


def sft_ckpt_dir(dataset: str) -> Path:
    return ROOT / "baseline" / "sft" / "output" / "checkpoints" / dataset / "best"


def require_sft_ckpt(dataset: str, explicit: str | None = None) -> Path:
    path = Path(explicit) if explicit else sft_ckpt_dir(dataset)
    if (path / "adapter_config.json").exists() or (path / "config.json").exists():
        return path
    raise SystemExit(
        f"SFT adapter not found: {path}. DecodingMatters does not train an LLM; run "
        f"bash baseline/sft/train.sh (DATASET={dataset})")


def flower_sasrec_path(dataset: str) -> Path:
    return ROOT / "baseline" / "Flower" / "output" / "checkpoints" / dataset / "sasrec.pt"


def resolve_sasrec(dataset: str, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise SystemExit(f"SASRec not found {path}")
    flower = flower_sasrec_path(dataset)
    if flower.exists():
        return flower
    local = ckpt_d3_dir(dataset) / "sasrec.pt"
    if local.exists():
        return local
    raise SystemExit(
        f"SASRec not found. DecodingMatters reuses Flower SASRec; run "
        f"DATASET={dataset} bash baseline/Flower/train.sh"
        f" (or bash baseline/DecodingMatters/train.sh to train one here)")


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


def decode_beam_items(sequences: torch.Tensor, prompt_width: int, n_return: int,
                      trie: TitlePrefixTrie, item_table, tokenizer,
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


class D3LogitsProcessor(LogitsProcessor):

    def __init__(self, prompt_width: int, trie: TitlePrefixTrie,
                 tf_probs: torch.Tensor, alpha: float, n_beams: int):
        self.prompt_width = prompt_width
        self.trie = trie
        self.tf_probs = tf_probs
        self.alpha = float(alpha)
        self.n_beams = int(n_beams)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.alpha == 1.0:
            return self._mask_illegal(input_ids, scores)

        mixed = torch.full_like(scores, float("-inf"))
        bsz = input_ids.size(0)
        for row in range(bsz):
            sample = row // self.n_beams
            p = self.tf_probs[sample]
            suffix = input_ids[row, self.prompt_width:].tolist()
            if any(int(t) == self.trie.eos_id for t in suffix):
                mixed[row, self.trie.eos_id] = scores[row, self.trie.eos_id]
                mixed[row, self.trie.pad_id] = scores[row, self.trie.pad_id]
                continue
            node = self.trie.node_at(suffix)
            if node is None:
                mixed[row, self.trie.eos_id] = scores[row, self.trie.eos_id]
                continue
            parent_mass = _mass(p, node.subtree_ids)
            if float(parent_mass) <= 0:
                for tok in node.allowed:
                    mixed[row, tok] = scores[row, tok]
                continue
            log_parent = torch.log(parent_mass + 1e-12)
            for tok, child in node.children.items():
                child_mass = _mass(p, child.subtree_ids)
                l_tf = torch.log(child_mass + 1e-12) - log_parent
                mixed[row, tok] = self.alpha * scores[row, tok] + (1.0 - self.alpha) * l_tf
            if node.item_ids:
                eos_mass = _mass(p, node.item_ids)
                l_tf = torch.log(eos_mass + 1e-12) - log_parent
                mixed[row, self.trie.eos_id] = (
                    self.alpha * scores[row, self.trie.eos_id] + (1.0 - self.alpha) * l_tf)
        return mixed

    def _mask_illegal(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        masked = torch.full_like(scores, float("-inf"))
        for row in range(input_ids.size(0)):
            suffix = input_ids[row, self.prompt_width:].tolist()
            for tok in self.trie.allowed_tokens(suffix):
                masked[row, tok] = scores[row, tok]
        return masked


def _mass(probs: torch.Tensor, item_ids: Sequence[int]) -> torch.Tensor:
    if not item_ids:
        return probs.new_zeros(())
    idx = torch.tensor(item_ids, device=probs.device, dtype=torch.long)
    return probs.index_select(0, idx).sum()
