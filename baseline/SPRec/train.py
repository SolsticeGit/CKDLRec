from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

_SPREC_DIR = Path(__file__).resolve().parent
if str(_SPREC_DIR) not in sys.path:
    sys.path.insert(0, str(_SPREC_DIR))

from _common import (
    ROOT, build_title_trie, ckpt_sprec_dir, decode_beam_items,
    disable_sampling_flags, unwrap_causal,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from dataset import (IGNORE_INDEX, GenerateCollator, ItemTable, PromptBuilder,
                     SFTDataset, TrainCollator)
from utils import (AverageMeter, Logger, count_parameters, human, load_base_model,
                   load_peft_model, save_json, set_seed)

SFT_EPOCHS = 3
ITERS = 3
DPO_BETA = 0.1
DPO_LR = 2e-5
PLAY_SAMPLES = 4096
GEN_BEAMS = 4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPRec: SFT then iterative self-play DPO")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--sft_epochs", type=int, default=SFT_EPOCHS)
    p.add_argument("--iters", type=int, default=ITERS, help="self-play DPO iterations")
    p.add_argument("--beta", type=float, default=DPO_BETA)
    p.add_argument("--dpo_epochs", type=int, default=1)
    p.add_argument("--dpo_lr", type=float, default=DPO_LR)
    p.add_argument("--lr", type=float, default=C.SFT.lr, help="SFT warmup learning rate")
    p.add_argument("--batch_size", type=int, default=C.SFT.batch_size)
    p.add_argument("--dpo_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=C.SFT.grad_accum)
    p.add_argument("--max_seq_len", type=int, default=C.RUNTIME.max_seq_len)
    p.add_argument("--num_workers", type=int, default=C.RUNTIME.num_workers)
    p.add_argument("--seed", type=int, default=C.RUNTIME.seed)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sft_ckpt", default=None,
                   help="existing SFT adapter; default baseline/sft/output/checkpoints/<ds>/best")
    p.add_argument("--init_sft", choices=("auto", "train", "sft_ckpt"), default="auto",
                   help="auto=skip warmup if SFT ckpt exists, else train SFT")
    p.add_argument("--play_samples", type=int, default=PLAY_SAMPLES,
                   help="self-play subsample per iteration; 0=use all")
    p.add_argument("--gen_beams", type=int, default=GEN_BEAMS)
    p.add_argument("--gen_batch_size", type=int, default=C.EVAL.batch_size)
    p.add_argument("--max_new_tokens", type=int, default=C.EVAL.max_new_tokens)
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


def load_trainable_adapter(ckpt: Path, log: Logger):
    base = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype)
    try:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(ckpt), is_trainable=True)
    except TypeError:
        model = load_peft_model(base, ckpt)
        for name, param in model.named_parameters():
            param.requires_grad = "lora_" in name
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    log(f"loaded trainable adapter from {ckpt}")
    return model


def load_frozen_adapter(ckpt: Path):
    base = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype)
    model = load_peft_model(base, ckpt)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def maybe_checkpointing(model, enabled: bool):
    if not enabled:
        return
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()


def resolve_sft_ckpt(dataset: str, explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    auto = ROOT / "baseline" / "sft" / "output" / "checkpoints" / dataset / "best"
    if (auto / "adapter_config.json").exists() or (auto / "config.json").exists():
        return auto
    return None


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = labels[:, 1:]
    mask = target.ne(IGNORE_INDEX)
    gathered = torch.gather(logp, 2, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


class DPOPairDataset(Dataset):

    def __init__(self, pairs: list[dict], builder: PromptBuilder):
        self.pairs = pairs
        self.builder = builder

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        record = pair["record"]
        history = record["history"]
        chosen = dict(record)
        chosen["target"] = pair["chosen_id"]
        rejected = dict(record)
        rejected["target"] = pair["rejected_id"]
        p_w, a_w = self.builder.build(chosen, history=history)
        p_l, a_l = self.builder.build(rejected, history=history)
        return {
            "chosen_ids": p_w + a_w,
            "chosen_labels": [IGNORE_INDEX] * len(p_w) + a_w,
            "rejected_ids": p_l + a_l,
            "rejected_labels": [IGNORE_INDEX] * len(p_l) + a_l,
        }


class DPOCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        def pad(key: str, fill: int) -> torch.Tensor:
            seqs = [b[key] for b in batch]
            width = max(len(s) for s in seqs)
            rows = [s + [fill] * (width - len(s)) for s in seqs]
            return torch.tensor(rows, dtype=torch.long)

        chosen_ids = pad("chosen_ids", self.pad_token_id)
        rejected_ids = pad("rejected_ids", self.pad_token_id)
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": (chosen_ids != self.pad_token_id).long(),
            "chosen_labels": pad("chosen_labels", IGNORE_INDEX),
            "rejected_ids": rejected_ids,
            "rejected_mask": (rejected_ids != self.pad_token_id).long(),
            "rejected_labels": pad("rejected_labels", IGNORE_INDEX),
        }


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
        collate_fn=TrainCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=TrainCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.sft_epochs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=C.SFT.weight_decay, betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * C.SFT.warmup_ratio), total_steps)
    log(f"SFT warmup steps={total_steps} ({steps_per_epoch}/epoch)")

    model.train()
    meter = AverageMeter()
    best_loss = float("inf")
    global_step = 0
    for epoch in range(args.sft_epochs):
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
    log(f"SFT warmup done | best valid_loss {best_loss:.4f}")


@torch.no_grad()
def generate_rejected(model, tokenizer, records: list[dict], item_table, ds_cfg,
                      trie, args, device, log: Logger) -> list[dict]:
    model.eval()
    disable_sampling_flags(model)
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    gen_set = _RecordGenerateSet(records, tokenizer, ds_cfg, item_table, args.max_seq_len)
    loader = DataLoader(
        gen_set, batch_size=args.gen_batch_size, shuffle=False,
        collate_fn=GenerateCollator(pad_id),
        num_workers=0, pin_memory=device.type == "cuda",
    )
    n_beams = max(args.gen_beams, 2)
    pairs = []
    n_same = 0
    n_done = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        prompt_width = input_ids.size(1)

        def prefix_allowed_tokens_fn(_batch_id: int, row: torch.Tensor) -> list[int]:
            return trie.allowed_tokens(row[prompt_width:].tolist())

        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=n_beams,
            num_return_sequences=n_beams,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            early_stopping=True,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        recs = decode_beam_items(
            sequences, prompt_width, n_beams, trie, item_table, tokenizer, pad_id)
        for i, rec in enumerate(recs):
            gold = int(batch["target_id"][i])
            rejected = next((r["item_id"] for r in rec if r["item_id"] != gold), None)
            idx = n_done + i
            if rejected is None:
                n_same += 1
                continue
            pairs.append({
                "record": records[idx],
                "chosen_id": gold,
                "rejected_id": int(rejected),
            })
        n_done += len(recs)
        if n_done % max(args.gen_batch_size * 10, 1) == 0 or n_done == len(records):
            log(f"  self-play generated {n_done:,}/{len(records):,}  "
                f"pairs={len(pairs):,} skipped_same={n_same:,}")
    model.config.use_cache = prev_cache
    model.train()
    return pairs


class _RecordGenerateSet(Dataset):
    def __init__(self, records, tokenizer, ds_cfg, item_table, max_seq_len):
        self.records = records
        self.items = item_table
        self.builder = PromptBuilder(tokenizer, ds_cfg, item_table, max_seq_len)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        history = record["history"]
        prompt_ids, _ = self.builder.build(record, history=history)
        target = record["target"]
        return {
            "input_ids": prompt_ids,
            "target_id": target,
            "target_title": self.items.titles[target],
        }


def dpo_forward(model, ids, mask, labels, device, amp_dtype):
    ids = ids.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype,
                        enabled=device.type == "cuda"):
        logits = model(input_ids=ids, attention_mask=mask).logits
    return sequence_logprob(logits.float(), labels)


def run_dpo(policy, ref, tokenizer, pairs, builder, args, device, amp_dtype,
            log: Logger) -> float:
    dataset = DPOPairDataset(pairs, builder)
    loader = DataLoader(
        dataset, batch_size=args.dpo_batch_size, shuffle=True,
        collate_fn=DPOCollator(tokenizer.pad_token_id),
        num_workers=0, pin_memory=device.type == "cuda",
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.dpo_epochs)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.dpo_lr, weight_decay=C.SFT.weight_decay, betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * C.SFT.warmup_ratio), total_steps)
    log(f"  DPO pairs={len(dataset)} steps={total_steps} beta={args.beta} lr={args.dpo_lr}")

    policy.train()
    ref.eval()
    meter = AverageMeter()
    last_loss = 0.0
    global_step = 0
    for epoch in range(args.dpo_epochs):
        for micro_step, batch in enumerate(loader):
            pi_w = dpo_forward(policy, batch["chosen_ids"], batch["chosen_mask"],
                               batch["chosen_labels"], device, amp_dtype)
            pi_l = dpo_forward(policy, batch["rejected_ids"], batch["rejected_mask"],
                               batch["rejected_labels"], device, amp_dtype)
            with torch.no_grad():
                ref_w = dpo_forward(ref, batch["chosen_ids"], batch["chosen_mask"],
                                    batch["chosen_labels"], device, amp_dtype)
                ref_l = dpo_forward(ref, batch["rejected_ids"], batch["rejected_mask"],
                                    batch["rejected_labels"], device, amp_dtype)
            logits = args.beta * ((pi_w - ref_w) - (pi_l - ref_l))
            loss = -F.logsigmoid(logits).mean()
            (loss / args.grad_accum).backward()
            meter.update(loss.item(), len(batch["chosen_ids"]))
            is_last = micro_step == len(loader) - 1
            if (micro_step + 1) % args.grad_accum != 0 and not is_last:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], C.SFT.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            last_loss = meter.avg
            if global_step % args.logging_steps == 0:
                log(f"    dpo epoch {epoch} step {global_step}/{total_steps} "
                    f"loss {meter.avg:.4f}")
                meter.reset()
    return last_loss


def copy_adapter(src: Path, dst: Path, tokenizer) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    import shutil
    for name in ("adapter_config.json", "adapter_model.safetensors",
                 "adapter_model.bin", "sft_meta.json"):
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    tokenizer.save_pretrained(str(dst))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    ds_cfg = C.get_dataset(args.dataset)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_sprec_dir(args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train_log.txt")
    log(f"=== SPRec | dataset={args.dataset} iters={args.iters} beta={args.beta} ===")

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
    train_set = SFTDataset(train_path, tokenizer, ds_cfg, item_table,
                           args.max_seq_len, mode="train", history_field="history")
    valid_set = SFTDataset(valid_path, tokenizer, ds_cfg, item_table,
                           args.max_seq_len, mode="train", history_field="history")
    if args.max_train_samples:
        train_set.records = train_set.records[:args.max_train_samples]
    log(f"items={len(item_table)} train={len(train_set)} valid={len(valid_set)}")

    found_sft = resolve_sft_ckpt(args.dataset, args.sft_ckpt)
    do_warmup = args.init_sft == "train" or (args.init_sft == "auto" and found_sft is None)
    if args.init_sft == "sft_ckpt" and found_sft is None:
        raise SystemExit("init_sft=sft_ckpt but adapter not found; pass --sft_ckpt")

    sft_dir = out_dir / "sft"
    if do_warmup:
        log("SFT warmup from scratch (no existing SFT adapter)")
        model = apply_lora(load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype))
        maybe_checkpointing(model, args.gradient_checkpointing)
        model.to(device)
        trainable, total = count_parameters(model)
        log(f"parameters: {human(trainable)} trainable / {human(total)} total")
        sft_dir.mkdir(parents=True, exist_ok=True)
        run_sft(model, tokenizer, train_set, valid_set, args, device, amp_dtype,
                sft_dir, log)
        del model
        torch.cuda.empty_cache()
        policy_ckpt = sft_dir / "best"
    else:
        policy_ckpt = found_sft
        log(f"skip SFT warmup, init from {policy_ckpt}")
        copy_adapter(policy_ckpt, sft_dir / "best", tokenizer)
        policy_ckpt = sft_dir / "best"

    log("building title prefix trie for self-play")
    trie = build_title_trie(item_table, tokenizer, tokenizer.eos_token_id,
                            tokenizer.pad_token_id, log)
    builder = PromptBuilder(tokenizer, ds_cfg, item_table, args.max_seq_len)

    save_json(out_dir / "train_config.json", {
        "method": "sprec",
        "dataset": args.dataset,
        "iters": args.iters,
        "beta": args.beta,
        "dpo_epochs": args.dpo_epochs,
        "dpo_lr": args.dpo_lr,
        "play_samples": args.play_samples,
        "gen_beams": args.gen_beams,
        "init_sft": args.init_sft,
        "sft_ckpt": str(found_sft) if found_sft else None,
        "did_sft_warmup": do_warmup,
        "history_field": "history",
        "base_model": str(C.BASE_MODEL_PATH),
        "lora": {"r": C.SFT.lora_r, "alpha": C.SFT.lora_alpha,
                 "dropout": C.SFT.lora_dropout, "targets": C.SFT.lora_targets},
    })

    current = policy_ckpt
    for it in range(args.iters):
        log(f"----- SPRec iteration {it} -----")
        play_records = list(train_set.records)
        if args.play_samples and args.play_samples < len(play_records):
            play_records = rng.sample(play_records, args.play_samples)
        log(f"  self-play on {len(play_records):,} train windows")

        policy = load_trainable_adapter(current, log)
        maybe_checkpointing(policy, args.gradient_checkpointing)
        policy.to(device)
        unwrap = unwrap_causal(policy)
        for module in (policy, unwrap):
            module.config.pad_token_id = tokenizer.pad_token_id
            module.config.eos_token_id = tokenizer.eos_token_id

        pairs = generate_rejected(policy, tokenizer, play_records, item_table,
                                  ds_cfg, trie, args, device, log)
        if len(pairs) < args.dpo_batch_size:
            raise SystemExit(f"iteration {it}: only {len(pairs)} DPO pairs; cannot train")

        it_dir = out_dir / f"it{it}"
        save_json(it_dir / "dpo_pairs_meta.json", {
            "n_play": len(play_records),
            "n_pairs": len(pairs),
            "n_rejected_hot": sum(int(item_table.is_hot[p["rejected_id"]]) for p in pairs),
        })
        log(f"  DPO pairs {len(pairs):,} | "
            f"rejected_hot={sum(int(item_table.is_hot[p['rejected_id']]) for p in pairs):,}")

        ref = load_frozen_adapter(current)
        ref.to(device)
        dpo_loss = run_dpo(policy, ref, tokenizer, pairs, builder, args, device,
                           amp_dtype, log)
        del ref
        torch.cuda.empty_cache()

        policy.save_pretrained(str(it_dir / "best"))
        tokenizer.save_pretrained(str(it_dir / "best"))
        save_json(it_dir / "best" / "sprec_meta.json",
                  {"iteration": it, "dpo_loss": dpo_loss, "n_pairs": len(pairs)})
        log(f"  saved {it_dir / 'best'} | dpo_loss {dpo_loss:.4f}")
        current = it_dir / "best"
        del policy
        torch.cuda.empty_cache()

    copy_adapter(current, out_dir / "best", tokenizer)
    copy_adapter(current, out_dir / "last", tokenizer)
    log(f"done | final adapter -> {out_dir / 'best'}")
    log.close()


if __name__ == "__main__":
    main()
