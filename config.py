from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LLM_DIR = ROOT / "LLMs"
OUTPUT_DIR = ROOT / "outputs"

CKPT_CF_SFT_DIR = OUTPUT_DIR / "checkpoints" / "cf_sft"
CKPT_CKDLREC_DIR = OUTPUT_DIR / "checkpoints" / "ckdlrec"
CF_DATA_DIR = OUTPUT_DIR / "cf_data"
RESULTS_DIR = OUTPUT_DIR / "results"

BASE_MODEL_PATH = LLM_DIR / "Qwen2.5-1.5B-Instruct"


def _hp(value: float) -> str:
    return f"{float(value):g}"


def cf_run_tag(tau: float | None = None) -> str:
    return f"tau_{_hp(COUNTERFACTUAL.tau if tau is None else tau)}"


def ckdlrec_run_tag(tau: float | None = None, alpha: float | None = None,
                   beta: float | None = None) -> str:
    return (
        f"{cf_run_tag(tau)}"
        f"_sft_{_hp(CKDLREC.alpha if alpha is None else alpha)}"
        f"_adv_{_hp(CKDLREC.beta if beta is None else beta)}"
    )


def cf_data_dir(dataset: str, tau: float | None = None) -> Path:
    return CF_DATA_DIR / dataset / cf_run_tag(tau)


def ckpt_cf_sft_dir(dataset: str, tau: float | None = None) -> Path:
    return CKPT_CF_SFT_DIR / dataset / cf_run_tag(tau)


def ckpt_ckdlrec_dir(dataset: str, tau: float | None = None, alpha: float | None = None,
                    beta: float | None = None) -> Path:
    return CKPT_CKDLREC_DIR / dataset / ckdlrec_run_tag(tau, alpha, beta)


def results_dir(dataset: str, tau: float | None = None, alpha: float | None = None,
                beta: float | None = None) -> Path:
    return RESULTS_DIR / dataset / ckdlrec_run_tag(tau, alpha, beta)


@dataclass
class DatasetConfig:

    key: str
    dir_name: str
    loader: str
    files: Dict[str, str]
    item_noun: str
    history_verb: str
    core: int = 5
    encoding: str = "utf-8"
    ts_unit: str = "s"
    category_source: str = "auto"
    time_start: tuple | None = None
    time_end: tuple | None = None

    @property
    def raw_dir(self) -> Path:
        return DATA_DIR / self.dir_name

    @property
    def processed_dir(self) -> Path:
        return DATA_DIR / self.dir_name / "processed"

    def raw_path(self, key: str) -> Path:
        return self.raw_dir / self.files[key]


DATASETS: Dict[str, DatasetConfig] = {
    "ml1m": DatasetConfig(
        key="ml1m",
        dir_name="MovieLens1M",
        loader="movielens",
        files={"inter": "ratings.dat", "item": "movies.dat"},
        item_noun="movie",
        history_verb="watched",
        core=5,
        encoding="latin-1",
        ts_unit="s",
    ),
    "cds": DatasetConfig(
        key="cds",
        dir_name="CDs_and_Vinyl",
        loader="amazon2023",
        files={"inter": "CDs_and_Vinyl.jsonl", "meta": "meta_CDs_and_Vinyl.jsonl"},
        item_noun="album",
        history_verb="listened to",
        core=5,
        ts_unit="ms",
        time_start=(2020, 9),
        time_end=(2023, 9),
    ),
    "toys": DatasetConfig(
        key="toys",
        dir_name="Toys_and_Games",
        loader="amazon2023",
        files={"inter": "Toys_and_Games.jsonl", "meta": "meta_Toys_and_Games.jsonl"},
        item_noun="toy",
        history_verb="purchased",
        core=5,
        ts_unit="ms",
        time_start=(2022, 9),
        time_end=(2023, 9),
    ),
    "movies": DatasetConfig(
        key="movies",
        dir_name="Movies_and_TV",
        loader="amazon2023",
        files={"inter": "Movies_and_TV.jsonl", "meta": "meta_Movies_and_TV.jsonl"},
        item_noun="movie",
        history_verb="watched",
        core=5,
        ts_unit="ms",
        time_start=(2022, 9),
        time_end=(2023, 9),
    ),
}

DEFAULT_DATASET = "ml1m"


@dataclass
class PreprocessConfig:
    max_hist_len: int = 10
    min_hist_len: int = 3
    split_ratios: tuple = (0.8, 0.1, 0.1)
    n_pop_group: int = 5
    n_pop_level: int = 8
    hot_ratio: float = 0.2
    max_title_chars: int = 100
    min_rating: float = 0.0
    min_categories: int = 5
    seed: int = 42


PREPROCESS = PreprocessConfig()

INSTRUCTION_TEMPLATE = (
    "Based on the user's historical {item_noun} preferences, "
    "recommend the next {item_noun}."
)
INPUT_TEMPLATE = "User {history_verb}: {history}."

SYSTEM_PROMPT_TEMPLATE = (
    "You are a {item_noun} recommender. "
    "Reply with the title of exactly one {item_noun} and nothing else."
)
USE_SYSTEM_PROMPT = True


@dataclass
class RuntimeConfig:

    device: str = "cuda"
    dtype: str = "bfloat16"
    max_seq_len: int = 320
    num_workers: int = 8
    gradient_checkpointing: bool = False
    seed: int = 42


RUNTIME = RuntimeConfig()


@dataclass
class SFTConfig:

    lr: float = 1e-4
    epochs: int = 2
    batch_size: int = 16
    grad_accum: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 20
    eval_steps: int = 200
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])


SFT = SFTConfig()


@dataclass
class CounterfactualConfig:

    tau: float = 0.4
    pop_norm: str = "n_rank"
    quantile_norm_mismatch: bool = True
    profile_norm: str = "unit_mean"
    profile_recency: str = "linear"
    retrieval_topk: int = 10
    retrieval_cold_scope: str = "all"
    retrieval_pop_groups: tuple = ()
    exclude_zero_freq_from_retrieval: bool = True


COUNTERFACTUAL = CounterfactualConfig()


@dataclass
class CKDLRecConfig:

    alpha: float = 0.4
    beta: float = 0.2
    kd_weight: float = 1.0
    tau_distill: float = 2.0
    grl_lambda_max: float = 0.6
    mlp_adv_hidden_ratio: float = 0.5
    lr: float = 1e-4
    epochs: int = 2
    batch_size: int = 8
    grad_accum: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 20
    eval_steps: int = 200


CKDLREC = CKDLRecConfig()


@dataclass
class EvalConfig:
    max_new_tokens: int = 48
    num_beams: int = 10
    topk: tuple = (1, 3, 5, 10)
    batch_size: int = 4


EVAL = EvalConfig()


def cold_groups(scope: str, n_group: int = PREPROCESS.n_pop_group) -> tuple:
    if scope == "all":
        frac = 1.0 - PREPROCESS.hot_ratio
    elif scope.startswith("bottom"):
        frac = int(scope[len("bottom"):]) / 100.0
    else:
        raise KeyError(f"unknown retrieval_cold_scope {scope!r}, expected 'all' or 'bottomN'")
    n_keep = max(1, min(n_group - 1, round(frac * n_group)))
    return tuple(range(n_keep))


def get_dataset(key: str) -> DatasetConfig:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}, choices: {sorted(DATASETS)}")
    return DATASETS[key]
