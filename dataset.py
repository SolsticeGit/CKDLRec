from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset

import config as C
from utils import load_jsonl, normalize_title, render_history

IGNORE_INDEX = -100


class ItemTable:

    def __init__(self, path: str | Path):
        rows = load_jsonl(path)
        rows.sort(key=lambda r: r["item_id"])
        n = len(rows)
        if any(r["item_id"] != i for i, r in enumerate(rows)):
            raise ValueError(f"{path} : item_id is not contiguous; preprocess output may be corrupted")

        self.n_items = n
        self.titles: List[str] = [r["title"] for r in rows]
        self.categories: List[str] = [r["category"] for r in rows]
        self.category_ids: List[int] = [r["category_id"] for r in rows]
        self.freq: List[int] = [r["freq"] for r in rows]
        self.pop_group: List[int] = [r["pop_group"] for r in rows]
        self.is_hot: List[bool] = [r["is_hot"] for r in rows]
        self.n_minmax: List[float] = [r["n_minmax"] for r in rows]
        self.n_log_minmax: List[float] = [r["n_log_minmax"] for r in rows]
        self.n_rank: List[float] = [r["n_rank"] for r in rows]

        self.n_categories = max(self.category_ids) + 1 if n else 0
        self.hot_ids = [i for i in range(n) if self.is_hot[i]]
        self.cold_ids = [i for i in range(n) if not self.is_hot[i]]

        self.title2id: Dict[str, int] = {}
        for i, title in enumerate(self.titles):
            self.title2id.setdefault(normalize_title(title), i)

    def history_pop(self, history: Sequence[int]) -> float:
        if not history:
            return 0.0
        return float(sum(self.n_rank[int(i)] for i in history) / len(history))

    def __len__(self) -> int:
        return self.n_items

    def lookup_title(self, text: str) -> int:
        return self.title2id.get(normalize_title(text), -1)


class PromptBuilder:

    def __init__(self, tokenizer, ds_cfg: C.DatasetConfig, item_table: ItemTable,
                 max_seq_len: int):
        self.tok = tokenizer
        self.ds = ds_cfg
        self.items = item_table
        self.max_seq_len = max_seq_len
        self.instruction = C.INSTRUCTION_TEMPLATE.format(item_noun=ds_cfg.item_noun)
        self.system = (C.SYSTEM_PROMPT_TEMPLATE.format(item_noun=ds_cfg.item_noun)
                       if C.USE_SYSTEM_PROMPT else "")

    def _prompt_ids(self, history: Sequence[int]) -> List[int]:
        user = self.instruction + "\n" + C.INPUT_TEMPLATE.format(
            history_verb=self.ds.history_verb,
            history=render_history([self.items.titles[i] for i in history]),
        )
        messages = ([{"role": "system", "content": self.system}] if self.system else []) \
            + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True)

    def build(self, record: dict, history: Sequence[int] | None = None,
              target_id: int | None = None) -> tuple[List[int], List[int]]:
        tid = int(record["target"] if target_id is None else target_id)
        answer_ids = self.tok(self.items.titles[tid],
                              add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [self.tok.eos_token_id]

        history = list(history if history is not None else record["history"])
        prompt_ids = self._prompt_ids(history)
        while len(prompt_ids) + len(answer_ids) > self.max_seq_len and len(history) > 1:
            history = history[1:]
            prompt_ids = self._prompt_ids(history)
        if len(prompt_ids) + len(answer_ids) > self.max_seq_len:
            keep = self.max_seq_len - len(answer_ids)
            prompt_ids = prompt_ids[:keep] if keep > 0 else prompt_ids[:1]
        return prompt_ids, answer_ids


class SFTDataset(Dataset):

    def __init__(self, path: str | Path, tokenizer, ds_cfg: C.DatasetConfig,
                 item_table: ItemTable, max_seq_len: int, mode: str = "train",
                 history_field: str = "history", target_field: str = "target"):
        if mode not in ("train", "generate"):
            raise ValueError(f"unknown mode {mode!r}")
        self.records = load_jsonl(path)
        self.mode = mode
        self.items = item_table
        self.history_field = history_field
        self.target_field = target_field
        self.builder = PromptBuilder(tokenizer, ds_cfg, item_table, max_seq_len)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        if self.history_field not in record:
            raise KeyError(
                f"record missing {self.history_field!r} (keys={sorted(record)}); "
                f"CF-SFT requires history_cf in cf_data; will not fall back to original history"
            )
        if self.target_field not in record:
            raise KeyError(
                f"record missing {self.target_field!r} (keys={sorted(record)})"
            )
        history = record[self.history_field]
        target = int(record[self.target_field])
        prompt_ids, answer_ids = self.builder.build(
            record, history=history, target_id=target)
        sample = {"target_id": target}
        if self.mode == "train":
            sample["input_ids"] = prompt_ids + answer_ids
            sample["labels"] = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
            if "v_y" in record:
                sample["v_y"] = float(record["v_y"])
        else:
            sample["input_ids"] = prompt_ids
            sample["target_title"] = self.items.titles[target]
        return sample


def _pad(seqs: List[List[int]], pad_value: int, left: bool) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    out = []
    for s in seqs:
        pad = [pad_value] * (width - len(s))
        out.append(pad + s if left else s + pad)
    return torch.tensor(out, dtype=torch.long)


class TrainCollator:

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        input_ids = _pad([b["input_ids"] for b in batch], self.pad_token_id, left=False)
        labels = _pad([b["labels"] for b in batch], IGNORE_INDEX, left=False)
        out = {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
            "labels": labels,
            "target_id": torch.tensor([b["target_id"] for b in batch]),
        }
        if "v_y" in batch[0]:
            out["v_y"] = torch.tensor([b["v_y"] for b in batch], dtype=torch.float32)
        return out


class GenerateCollator:

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[dict]) -> Dict[str, object]:
        input_ids = _pad([b["input_ids"] for b in batch], self.pad_token_id, left=True)
        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.pad_token_id).long(),
            "target_id": torch.tensor([b["target_id"] for b in batch]),
            "target_title": [b["target_title"] for b in batch],
        }
