from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

_D2_DIR = Path(__file__).resolve().parent
if str(_D2_DIR) not in sys.path:
    sys.path.insert(0, str(_D2_DIR))

from _common import (
    ROOT, ckpt_d2_dir, item_pop_scores, token_ips_weights,
)
from sasrec import save_sasrec, train_biased_sasrec

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from dataset import IGNORE_INDEX, ItemTable, SFTDataset, TrainCollator
from utils import (AverageMeter, Logger, count_parameters, human, load_base_model,
                   save_json, set_seed)

SFT_EPOCHS = 3
IPS_ALPHA = 0.5
CRS_GAMMA = 0.5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D2LR: token-IPS SFT + biased SASRec")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--epochs", type=int, default=SFT_EPOCHS)
    p.add_argument("--lr", type=float, default=C.SFT.lr)
    p.add_argument("--batch_size", type=int, default=C.SFT.batch_size)
    p.add_argument("--grad_accum", type=int, default=C.SFT.grad_accum)
    p.add_argument("--max_seq_len", type=int, default=C.RUNTIME.max_seq_len)
    p.add_argument("--num_workers", type=int, default=C.RUNTIME.num_workers)
    p.add_argument("--seed", type=int, default=C.RUNTIME.seed)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--ips_alpha", type=float, default=IPS_ALPHA,
                   help="paper alpha, position decay, search {0.1,...,0.9}")
    p.add_argument("--crs_gamma", type=float, default=CRS_GAMMA)
    p.add_argument("--crs_hidden", type=int, default=50)
    p.add_argument("--crs_state_size", type=int, default=50)
    p.add_argument("--crs_lr", type=float, default=0.01)
    p.add_argument("--crs_batch_size", type=int, default=1024)
    p.add_argument("--crs_dropout", type=float, default=0.2)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   default=C.RUNTIME.gradient_checkpointing)
    p.add_argument("--eval_steps", type=int, default=C.SFT.eval_steps)
    p.add_argument("--logging_steps", type=int, default=C.SFT.logging_steps)
    p.add_argument("--max_train_samples", type=int, default=None)
    return p.parse_args()


def apply_lora(model):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(
        r=C.SFT.lora_r,
        lora_alpha=C.SFT.lora_alpha,
        lora_dropout=C.SFT.lora_dropout,
        target_modules=C.SFT.lora_targets,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return model


def maybe_checkpointing(model, enabled: bool):
    if not enabled:
        return
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()


class IPSDataset(SFTDataset):

    def __init__(self, *args, item_pops: list[float] | None = None,
                 ips_alpha: float = IPS_ALPHA, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_pops = item_pops or []
        self.ips_alpha = ips_alpha

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        if self.mode != "train":
            return sample
        record = self.records[idx]
        prompt_len = next((i for i, x in enumerate(sample["labels"]) if x != IGNORE_INDEX),
                          len(sample["labels"]))
        answer_ids = sample["input_ids"][prompt_len:]
        title_ids = answer_ids[:-1] if answer_ids else []
        y = int(record["target"])
        p_y = float(self.item_pops[y]) if 0 <= y < len(self.item_pops) else 1.0
        w_title = token_ips_weights(len(title_ids), p_y, self.ips_alpha)
        if w_title:
            w_answer = w_title + [w_title[-1]]
        else:
            w_answer = [1.0] * len(answer_ids)
        sample["token_weights"] = [0.0] * prompt_len + w_answer
        return sample


class IPSCollator(TrainCollator):
    def __call__(self, batch: list[dict]) -> dict:
        out = super().__call__(batch)
        weights = [b["token_weights"] for b in batch]
        width = out["input_ids"].size(1)
        rows = [w + [0.0] * (width - len(w)) for w in weights]
        out["token_weights"] = torch.tensor(rows, dtype=torch.float32)
        return out


def ips_loss(model, batch, device, amp_dtype):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    weights = batch["token_weights"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                        enabled=device.type == "cuda"):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    target = labels[:, 1:]
    w = weights[:, 1:]
    mask = target.ne(IGNORE_INDEX)
    gathered = torch.gather(logp, 2, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    n = mask.sum().clamp(min=1)
    return -((gathered * w * mask).sum() / n)


@torch.no_grad()
def evaluate_sft_loss(model, loader, device, amp_dtype) -> float:
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        n_tokens = int((batch["labels"] != IGNORE_INDEX).sum())
        if n_tokens == 0:
            continue
        inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")}
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            loss = model(**inputs).loss
        meter.update(loss.item(), n_tokens)
    model.train()
    return meter.avg


def run_sft(model, tokenizer, train_set, valid_set, args, device, amp_dtype,
            out_dir: Path, log: Logger) -> None:
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=IPSCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=TrainCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=C.SFT.weight_decay, betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * C.SFT.warmup_ratio), total_steps)
    log(f"SFT-IPS steps={total_steps} ({steps_per_epoch}/epoch) alpha={args.ips_alpha}")

    model.train()
    meter = AverageMeter()
    best_loss = float("inf")
    global_step = 0
    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader):
            loss = ips_loss(model, batch, device, amp_dtype)
            (loss / args.grad_accum).backward()
            meter.update(loss.item())
            is_last = micro_step == len(train_loader) - 1
            if (micro_step + 1) % args.grad_accum != 0 and not is_last:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], C.SFT.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % args.logging_steps == 0:
                log(f"  sft epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {meter.avg:.4f}")
                meter.reset()
            if global_step % args.eval_steps == 0 or global_step == total_steps:
                val_loss = evaluate_sft_loss(model, valid_loader, device, amp_dtype)
                log(f"  [sft eval] step {global_step} valid_loss {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    model.save_pretrained(str(out_dir / "best"))
                    tokenizer.save_pretrained(str(out_dir / "best"))
                    log(f"  [sft eval] new best -> {out_dir / 'best'}")
    model.save_pretrained(str(out_dir / "last"))
    tokenizer.save_pretrained(str(out_dir / "last"))
    if not (out_dir / "best" / "adapter_config.json").exists():
        model.save_pretrained(str(out_dir / "best"))
        tokenizer.save_pretrained(str(out_dir / "best"))
    log(f"SFT-IPS done | best valid_loss {best_loss:.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ds_cfg = C.get_dataset(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_d2_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train_log.txt")
    log(f"=== D2LR | dataset={args.dataset} epochs={args.epochs} "
        f"alpha={args.ips_alpha} gamma={args.crs_gamma} ===")

    device = torch.device(C.RUNTIME.device if torch.cuda.is_available() else "cpu")
    amp_dtype = getattr(torch, C.RUNTIME.dtype)
    log(f"device={device} dtype={C.RUNTIME.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(str(C.BASE_MODEL_PATH))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed = ds_cfg.processed_dir
    train_path = processed / "train.jsonl"
    valid_path = processed / "valid.jsonl"
    pop_path = processed / "popularity.jsonl"
    if not train_path.exists() or not pop_path.exists():
        raise SystemExit(f"not found: {train_path} or {pop_path}. Run "
                         f"python preprocess.py --dataset {args.dataset}")

    item_table = ItemTable(pop_path)
    item_pops = item_pop_scores(item_table.freq)
    train_set = IPSDataset(train_path, tokenizer, ds_cfg, item_table, args.max_seq_len,
                           mode="train", history_field="history",
                           item_pops=item_pops, ips_alpha=args.ips_alpha)
    valid_set = SFTDataset(valid_path, tokenizer, ds_cfg, item_table,
                           args.max_seq_len, mode="train", history_field="history")
    if args.max_train_samples:
        train_set.records = train_set.records[:args.max_train_samples]
    log(f"items={len(item_table)} train={len(train_set)} valid={len(valid_set)}")

    log("training popularity-amplified SASRec (inference collaborator)")
    sasrec = train_biased_sasrec(
        train_set.records, valid_set.records, item_table.n_items, item_table.freq,
        device, log, state_size=args.crs_state_size, hidden=args.crs_hidden,
        dropout=args.crs_dropout, lr=args.crs_lr, batch_size=args.crs_batch_size,
        gamma=args.crs_gamma, seed=args.seed)
    save_sasrec(sasrec, out_dir / "sasrec.pt")
    del sasrec
    torch.cuda.empty_cache()

    save_json(out_dir / "train_config.json", {
        "method": "d2lr",
        "dataset": args.dataset,
        "epochs": args.epochs,
        "ips_alpha": args.ips_alpha,
        "crs_gamma": args.crs_gamma,
        "history_field": "history",
        "base_model": str(C.BASE_MODEL_PATH),
        "item_pop": "freq / max(freq)",
        "lora": {"r": C.SFT.lora_r, "alpha": C.SFT.lora_alpha,
                 "dropout": C.SFT.lora_dropout, "targets": C.SFT.lora_targets},
    })

    model = apply_lora(load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype))
    maybe_checkpointing(model, args.gradient_checkpointing)
    model.to(device)
    trainable, total = count_parameters(model)
    log(f"parameters: {human(trainable)} trainable / {human(total)} total")
    run_sft(model, tokenizer, train_set, valid_set, args, device, amp_dtype, out_dir, log)
    del model
    torch.cuda.empty_cache()
    log(f"done | adapter -> {out_dir / 'best'} | sasrec -> {out_dir / 'sasrec.pt'}")
    log.close()


if __name__ == "__main__":
    main()
