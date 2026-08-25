from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Logger:

    def __init__(self, path: Path | None = None):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fh = open(path, "a", encoding="utf-8")
        else:
            self.fh = None
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        line = f"[{time.time() - self.t0:8.1f}s] {msg}"
        print(line, flush=True)
        if self.fh is not None:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self) -> None:
        if self.fh is not None:
            self.fh.close()


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def human(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def load_base_model(path: str | Path, dtype: str = "bfloat16", attn: str = "sdpa"):
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM

    key = ("dtype" if version.parse(transformers.__version__) >= version.parse("4.56.0")
           else "torch_dtype")
    return AutoModelForCausalLM.from_pretrained(
        str(path), attn_implementation=attn, **{key: getattr(torch, dtype)})


def load_peft_model(base, adapter_path: str | Path, is_trainable: bool = False):
    import dataclasses
    import json
    import shutil
    import tempfile

    from peft import LoraConfig, PeftModel

    adapter_path = Path(adapter_path)

    def _load(path: Path):
        try:
            return PeftModel.from_pretrained(base, str(path), is_trainable=is_trainable)
        except TypeError:
            return PeftModel.from_pretrained(base, str(path))

    try:
        return _load(adapter_path)
    except TypeError:
        pass

    raw = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    valid = {f.name for f in dataclasses.fields(LoraConfig)}
    cleaned = {k: v for k, v in raw.items() if k in valid}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for src in adapter_path.iterdir():
            if src.name == "adapter_config.json":
                (tmp / src.name).write_text(json.dumps(cleaned, indent=2), encoding="utf-8")
            elif src.is_file():
                shutil.copy2(src, tmp / src.name)
        return _load(tmp)


def load_jsonl(path: str | Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def render_history(titles: Sequence[str]) -> str:
    return ", ".join(f'"{t}"' for t in titles)


def normalize_title(text: str) -> str:
    return " ".join(text.replace('"', " ").split()).strip().lower()


def quantile_normalize(values: torch.Tensor) -> torch.Tensor:
    if values.numel() <= 1:
        return torch.zeros_like(values)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=values.dtype)
    return ranks / (values.numel() - 1)
