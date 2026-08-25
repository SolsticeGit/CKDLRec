from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_SFT_DIR = Path(__file__).resolve().parent
if str(_SFT_DIR) not in sys.path:
    sys.path.insert(0, str(_SFT_DIR))

from _common import ROOT, ckpt_sft_dir, disable_sampling_flags

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import config as C
from dataset import GenerateCollator, ItemTable, SFTDataset, TrainCollator
from utils import (AverageMeter, Logger, count_parameters, human, load_base_model,
                   normalize_title, save_json, set_seed)

EPOCHS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT baseline: LoRA on original processed histories")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=C.SFT.lr)
    p.add_argument("--batch_size", type=int, default=C.SFT.batch_size)
    p.add_argument("--grad_accum", type=int, default=C.SFT.grad_accum)
    p.add_argument("--max_seq_len", type=int, default=C.RUNTIME.max_seq_len)
    p.add_argument("--num_workers", type=int, default=C.RUNTIME.num_workers)
    p.add_argument("--seed", type=int, default=C.RUNTIME.seed)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--full_finetune", action="store_true",
                   help="full finetune instead of LoRA (~28GB VRAM)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   default=C.RUNTIME.gradient_checkpointing)
    p.add_argument("--eval_steps", type=int, default=C.SFT.eval_steps)
    p.add_argument("--logging_steps", type=int, default=C.SFT.logging_steps)
    p.add_argument("--preview_samples", type=int, default=4,
                   help="print a few generations after each eval for format checks")
    p.add_argument("--max_train_samples", type=int, default=None, help="debug: truncate the training set")
    return p.parse_args()


def build_model(args: argparse.Namespace, log: Logger):
    tokenizer = AutoTokenizer.from_pretrained(str(C.BASE_MODEL_PATH))
    model = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype)

    if args.full_finetune:
        log("full fine-tuning: all backbone parameters are trainable")
    else:
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

    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainable, total = count_parameters(model)
    log(f"parameters: {human(trainable)} trainable / {human(total)} total "
        f"({trainable / total:.2%})")
    return model, tokenizer


@torch.no_grad()
def evaluate_loss(model, loader, device, amp_dtype) -> float:
    model.eval()
    meter = AverageMeter()
    for batch in loader:
        n_tokens = int((batch["labels"] != -100).sum())
        if n_tokens == 0:
            continue
        inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask", "labels")}
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            loss = model(**inputs).loss
        meter.update(loss.item(), n_tokens)
    model.train()
    return meter.avg


@torch.no_grad()
def preview_generations(model, tokenizer, dataset, device, n: int, log: Logger) -> None:
    if n <= 0:
        return
    model.eval()
    disable_sampling_flags(model)
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    collate = GenerateCollator(tokenizer.pad_token_id)
    batch = collate([dataset[i] for i in range(min(n, len(dataset)))])
    out = model.generate(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        max_new_tokens=C.EVAL.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    model.config.use_cache = prev_cache
    generated = out[:, batch["input_ids"].size(1):]
    hits = 0
    for i, seq in enumerate(generated):
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        gold = batch["target_title"][i]
        ok = normalize_title(text) == normalize_title(gold)
        hits += ok
        log(f"    [{'hit ' if ok else 'miss'}] pred={text!r} | gold={gold!r}")
    log(f"    preview exact-match: {hits}/{len(generated)}")
    model.train()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ds_cfg = C.get_dataset(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_sft_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train_log.txt")
    log(f"=== SFT baseline | dataset={args.dataset} | original history ===")

    device = torch.device(C.RUNTIME.device if torch.cuda.is_available() else "cpu")
    amp_dtype = getattr(torch, C.RUNTIME.dtype)
    log(f"device={device} dtype={C.RUNTIME.dtype}")

    model, tokenizer = build_model(args, log)
    model.to(device)

    processed = ds_cfg.processed_dir
    train_path = processed / "train.jsonl"
    valid_path = processed / "valid.jsonl"
    pop_path = processed / "popularity.jsonl"
    if not train_path.exists() or not pop_path.exists():
        raise SystemExit(f"not found: {train_path} or {pop_path}. Run "
                         f"python preprocess.py --dataset {args.dataset}")

    item_table = ItemTable(pop_path)
    train_set = SFTDataset(train_path, tokenizer, ds_cfg, item_table,
                           args.max_seq_len, mode="train", history_field="history")
    valid_set = SFTDataset(valid_path, tokenizer, ds_cfg, item_table,
                           args.max_seq_len, mode="train", history_field="history")
    preview_set = SFTDataset(valid_path, tokenizer, ds_cfg, item_table,
                             args.max_seq_len, mode="generate", history_field="history")
    if args.max_train_samples:
        train_set.records = train_set.records[:args.max_train_samples]
    log(f"items={len(item_table)} categories={item_table.n_categories} "
        f"train={len(train_set)} valid={len(valid_set)} | data={processed}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=TrainCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=TrainCollator(tokenizer.pad_token_id), num_workers=args.num_workers,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=C.SFT.weight_decay, betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * C.SFT.warmup_ratio), total_steps)
    log(f"optimizer steps: {total_steps} ({steps_per_epoch}/epoch) | "
        f"effective batch = {args.batch_size * args.grad_accum}")

    save_json(out_dir / "train_config.json", {
        "method": "sft",
        "dataset": args.dataset, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_seq_len": args.max_seq_len, "seed": args.seed,
        "full_finetune": args.full_finetune, "total_steps": total_steps,
        "history_field": "history",
        "data_dir": str(processed),
        "base_model": str(C.BASE_MODEL_PATH),
        "lora": None if args.full_finetune else {
            "r": C.SFT.lora_r, "alpha": C.SFT.lora_alpha,
            "dropout": C.SFT.lora_dropout, "targets": C.SFT.lora_targets,
        },
    })

    model.train()
    meter = AverageMeter()
    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        for micro_step, batch in enumerate(train_loader):
            inputs = {k: batch[k].to(device, non_blocking=True)
                      for k in ("input_ids", "attention_mask", "labels")}
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                loss = model(**inputs).loss
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
                log(f"epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {meter.avg:.4f} lr {scheduler.get_last_lr()[0]:.2e}")
                meter.reset()

            if global_step % args.eval_steps == 0 or global_step == total_steps:
                val_loss = evaluate_loss(model, valid_loader, device, amp_dtype)
                log(f"  [eval] step {global_step} valid_loss {val_loss:.4f} "
                    f"ppl {math.exp(min(val_loss, 20)):.2f}")
                preview_generations(model, tokenizer, preview_set, device,
                                    args.preview_samples, log)
                if val_loss < best_loss:
                    best_loss = val_loss
                    model.save_pretrained(str(out_dir / "best"))
                    tokenizer.save_pretrained(str(out_dir / "best"))
                    save_json(out_dir / "best" / "sft_meta.json",
                              {"step": global_step, "valid_loss": val_loss,
                               "dataset": args.dataset})
                    log(f"  [eval] new best -> {out_dir / 'best'}")

    model.save_pretrained(str(out_dir / "last"))
    tokenizer.save_pretrained(str(out_dir / "last"))
    log(f"done | best valid_loss {best_loss:.4f} | checkpoints under {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
