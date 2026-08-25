from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "output"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C

GROUPS = ("pop", "category")


def ckpt_ifair_dir(dataset: str, group: str) -> Path:
    return OUTPUT / "checkpoints" / dataset / group


def results_ifair_dir(dataset: str, group: str) -> Path:
    return OUTPUT / "results" / dataset / group


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
