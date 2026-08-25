from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import config as C
from dataset import GenerateCollator, ItemTable, SFTDataset, TrainCollator
from losses import student_loss
from model import CKDLRecModel, PopAdvHead, dann_lambda
from utils import (AverageMeter, Logger, count_parameters, human, load_base_model,
                   load_peft_model, normalize_title, save_json, set_seed)

LOG_KEYS = ("loss", "distill", "sft", "adv", "adv_mae", "z_norm")


def parse_args() -> argparse.Namespace:
    cfg = C.CKDLREC
    p = argparse.ArgumentParser(description="CKDLRec: distill from frozen CF-SFT teacher")
    p.add_argument("--dataset", default=C.DEFAULT_DATASET, choices=sorted(C.DATASETS))
    p.add_argument("--cf_sft_path", default=None,
                   help="teacher LoRA, default outputs/checkpoints/cf_sft/<ds>/tau_*/best")
    p.add_argument("--cf_dir", default=None,
                   help="cf_{train,valid}.jsonl dir, default outputs/cf_data/<ds>/tau_*")
    p.add_argument("--tau", type=float, default=C.COUNTERFACTUAL.tau,
                   help="must match teacher / cf_data tau_* dir")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--grad_accum", type=int, default=cfg.grad_accum)
    p.add_argument("--max_seq_len", type=int, default=C.RUNTIME.max_seq_len)
    p.add_argument("--num_workers", type=int, default=C.RUNTIME.num_workers)
    p.add_argument("--seed", type=int, default=C.RUNTIME.seed)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--gradient_checkpointing", action="store_true",
                   default=C.RUNTIME.gradient_checkpointing)
    p.add_argument("--eval_steps", type=int, default=cfg.eval_steps)
    p.add_argument("--logging_steps", type=int, default=cfg.logging_steps)
    p.add_argument("--preview_samples", type=int, default=4)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    p.add_argument("--alpha", type=float, default=cfg.alpha, help="L_SFT weight")
    p.add_argument("--beta", type=float, default=cfg.beta, help="L_adv weight")
    p.add_argument("--kd_weight", type=float, default=cfg.kd_weight, help="L_KD weight")
    p.add_argument("--tau_distill", type=float, default=cfg.tau_distill)
    p.add_argument("--grl_lambda_max", type=float, default=cfg.grl_lambda_max)
    p.add_argument("--resume", action="store_true",
                   help="resume from output_dir/last (else best)")
    p.add_argument("--resume_from", default=None,
                   help="checkpoint dir to resume from; overrides --resume auto-detect")
    return p.parse_args()


def cfg_from_args(args: argparse.Namespace):
    from dataclasses import replace
    return replace(
        C.CKDLREC, alpha=args.alpha, beta=args.beta, kd_weight=args.kd_weight,
        tau_distill=args.tau_distill,
        grl_lambda_max=args.grl_lambda_max, lr=args.lr, epochs=args.epochs,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        eval_steps=args.eval_steps, logging_steps=args.logging_steps,
    )


def load_teacher(cf_sft_path: Path):
    if not cf_sft_path.exists():
        raise SystemExit(f"not found: {cf_sft_path}. Run python train_cf_sft.py")
    base = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype)
    teacher = load_peft_model(base, cf_sft_path)
    if hasattr(teacher, "merge_and_unload"):
        teacher = teacher.merge_and_unload()
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _enable_lora_grads(llm) -> None:
    if any(p.requires_grad for n, p in llm.named_parameters() if "lora_" in n):
        return
    for name, param in llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def build_student(cfg, gradient_checkpointing: bool, ckpt_path: Path | None = None):
    from peft import LoraConfig, get_peft_model
    llm = load_base_model(C.BASE_MODEL_PATH, C.RUNTIME.dtype)
    if ckpt_path is not None:
        llm = load_peft_model(llm, ckpt_path, is_trainable=True)
        _enable_lora_grads(llm)
    else:
        llm = get_peft_model(llm, LoraConfig(
            r=C.SFT.lora_r,
            lora_alpha=C.SFT.lora_alpha,
            lora_dropout=C.SFT.lora_dropout,
            target_modules=C.SFT.lora_targets,
            bias="none",
            task_type="CAUSAL_LM",
        ))
    for param in llm.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    adv_head = PopAdvHead(llm.config.hidden_size, cfg.mlp_adv_hidden_ratio)
    if ckpt_path is not None:
        adv_pt = ckpt_path / "adv_head.pt"
        if adv_pt.exists():
            adv_head.load_state_dict(torch.load(adv_pt, map_location="cpu"))
    student = CKDLRecModel(llm, adv_head)
    if gradient_checkpointing:
        unwrap = llm.get_base_model() if hasattr(llm, "get_base_model") else llm
        unwrap.config.use_cache = False
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()
        llm.gradient_checkpointing_enable()
    return student


def resolve_resume_dir(out_dir: Path, resume: bool, resume_from: str | None) -> Path | None:
    if resume_from:
        path = Path(resume_from)
        if not (path / "adapter_config.json").exists():
            raise SystemExit(f"not found: {path}/adapter_config.json")
        return path
    if not resume:
        return None
    for name in ("last", "best"):
        path = out_dir / name
        if (path / "adapter_config.json").exists():
            return path
    raise SystemExit(f"{out_dir}  has no last/best; cannot resume")


def load_json(path: Path) -> dict:
    import json
    return json.loads(path.read_text(encoding="utf-8"))


@torch.no_grad()
def teacher_logits(teacher, batch: dict) -> torch.Tensor:
    return teacher(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits


def _step(student: CKDLRecModel, teacher, batch: dict, cfg) -> dict:
    logits_t = None
    if teacher is not None and cfg.kd_weight != 0:
        with torch.no_grad():
            logits_t = teacher_logits(teacher, batch)
    logits_s, z, v_hat = student(
        batch["input_ids"], batch["attention_mask"], batch["labels"])
    if v_hat is None:
        raise RuntimeError("student has no adversarial head; cannot compute L_adv")
    return student_loss(
        logits_s, logits_t, batch["labels"], z, v_hat, batch["v_y"], cfg,
    )


@torch.no_grad()
def evaluate(student: CKDLRecModel, teacher, loader, device, amp_dtype, cfg) -> dict:
    student.eval()
    meters = {k: AverageMeter() for k in LOG_KEYS}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            parts = _step(student, teacher, batch, cfg)
        n = batch["labels"].size(0)
        for k, m in meters.items():
            m.update(parts[k].item(), n)
    student.train()
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def preview(student: CKDLRecModel, tokenizer, dataset, device, n: int, log: Logger) -> None:
    if n <= 0:
        return
    student.eval()
    collate = GenerateCollator(tokenizer.pad_token_id)
    batch = collate([dataset[i] for i in range(min(n, len(dataset)))])
    out = student.generate(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        max_new_tokens=C.EVAL.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1,
    )
    generated = out[:, batch["input_ids"].size(1):]
    hits = 0
    for i, seq in enumerate(generated):
        text = tokenizer.decode(seq, skip_special_tokens=True).strip()
        gold = batch["target_title"][i]
        hits += normalize_title(text) == normalize_title(gold)
    log(f"    preview exact-match: {hits}/{len(generated)}")
    student.train()


def save_ckpt(student: CKDLRecModel, tokenizer, path: Path, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    student.llm.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    if student.adv_head is not None:
        torch.save(student.adv_head.state_dict(), path / "adv_head.pt")
    save_json(path / "ckdlrec_meta.json", meta)


def save_trainer_state(path: Path, optimizer, scheduler, global_step: int,
                       best: float) -> None:
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "best": best,
    }, path / "trainer_state.pt")


def main() -> None:
    args = parse_args()
    cfg = cfg_from_args(args)
    set_seed(args.seed)

    ds_cfg = C.get_dataset(args.dataset)
    tag = C.ckdlrec_run_tag(args.tau, args.alpha, args.beta)
    out_dir = Path(args.output_dir) if args.output_dir else C.ckpt_ckdlrec_dir(
        args.dataset, args.tau, args.alpha, args.beta)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(out_dir / "train_log.txt")
    log(f"=== CKDLRec student | dataset={args.dataset} | {tag} ===")

    device = torch.device(C.RUNTIME.device if torch.cuda.is_available() else "cpu")
    amp_dtype = getattr(torch, C.RUNTIME.dtype)

    tokenizer = AutoTokenizer.from_pretrained(str(C.BASE_MODEL_PATH))
    cf_dir = Path(args.cf_dir) if args.cf_dir else C.cf_data_dir(args.dataset, args.tau)
    train_path = cf_dir / "cf_train.jsonl"
    valid_path = cf_dir / "cf_valid.jsonl"
    if not train_path.exists():
        raise SystemExit(
            f"not found: {train_path}. Run python build_cf_data.py --dataset {args.dataset}")

    processed = ds_cfg.processed_dir
    item_table = ItemTable(processed / "popularity.jsonl")
    train_set = SFTDataset(
        train_path, tokenizer, ds_cfg, item_table, args.max_seq_len, mode="train",
        history_field="history_orig", target_field="target_orig")
    valid_set = SFTDataset(
        valid_path, tokenizer, ds_cfg, item_table, args.max_seq_len, mode="train",
        history_field="history_orig", target_field="target_orig")
    preview_set = SFTDataset(
        valid_path, tokenizer, ds_cfg, item_table, args.max_seq_len, mode="generate",
        history_field="history_orig", target_field="target_orig")
    if not train_set.records:
        raise SystemExit(f"{train_path}  is empty")
    if "v_y" not in train_set.records[0]:
        raise SystemExit(
            f"{train_path}  missing v_y; rebuild counterfactual data with current build_cf_data.py")
    if args.max_train_samples:
        train_set.records = train_set.records[:args.max_train_samples]
    if args.max_eval_samples:
        valid_set.records = valid_set.records[:args.max_eval_samples]
        preview_set.records = preview_set.records[:args.max_eval_samples]

    cf_sft_path = (Path(args.cf_sft_path) if args.cf_sft_path
                   else C.ckpt_cf_sft_dir(args.dataset, args.tau) / "best")
    teacher = None
    if cfg.kd_weight != 0:
        teacher = load_teacher(cf_sft_path).to(device)
    resume_dir = resolve_resume_dir(out_dir, args.resume, args.resume_from)
    student = build_student(cfg, args.gradient_checkpointing, resume_dir).to(device)
    trainable, _ = count_parameters(student)
    teacher_desc = str(cf_sft_path) if teacher is not None else "off (kd_weight=0)"
    log(f"train={len(train_set)} valid={len(valid_set)} | "
        f"trainable={human(trainable)} (new LoRA + adv MLP) | teacher={teacher_desc}")
    log(f"data={cf_dir} | history_orig / target_orig / v_y")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=TrainCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=TrainCollator(tokenizer.pad_token_id),
        num_workers=args.num_workers,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    params = [p for p in student.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("student has no trainable parameters")
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
    log(f"steps={total_steps} ({steps_per_epoch}/epoch) "
        f"batch={args.batch_size * args.grad_accum} "
        f"γ={cfg.kd_weight} α={cfg.alpha} β={cfg.beta} λ_max={cfg.grl_lambda_max}")

    save_json(out_dir / "train_config.json", {
        "dataset": args.dataset, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "alpha": cfg.alpha, "beta": cfg.beta, "kd_weight": cfg.kd_weight,
        "tau_distill": cfg.tau_distill,
        "grl_lambda_max": cfg.grl_lambda_max,
        "seed": args.seed, "tau": args.tau,
        "cf_sft_path": str(cf_sft_path) if teacher is not None else None,
        "run_tag": tag,
        "history_field": "history_orig",
        "target_field": "target_orig",
        "cf_dir": str(cf_dir),
        "lora": {
            "r": C.SFT.lora_r, "alpha": C.SFT.lora_alpha,
            "dropout": C.SFT.lora_dropout, "targets": C.SFT.lora_targets,
        },
    })

    student.train()
    if teacher is not None:
        teacher.eval()
    meters = {k: AverageMeter() for k in LOG_KEYS}
    best = float("inf")
    global_step = 0
    start_epoch = 0
    skip_opt = 0

    if resume_dir is not None:
        meta_path = resume_dir / "ckdlrec_meta.json"
        meta = load_json(meta_path) if meta_path.exists() else {}
        state_path = resume_dir / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            global_step = int(state["global_step"])
            best = float(state["best"])
        else:
            global_step = int(meta.get("step", 0))
            best = float(meta.get("valid_distill", meta.get("valid_loss", best)))
            for _ in range(global_step):
                scheduler.step()
        if global_step >= total_steps:
            log(f"resume {resume_dir} already at {global_step}/{total_steps}, skip train")
            save_json(out_dir / "train_done.json",
                      {"step": global_step, "total_steps": total_steps, "best": best})
            log.close()
            return
        start_epoch = global_step // steps_per_epoch
        skip_opt = global_step % steps_per_epoch
        log(f"resume {resume_dir} | step {global_step}/{total_steps} "
            f"epoch {start_epoch} skip_opt={skip_opt} best={best:.4f}")

    for epoch in range(start_epoch, args.epochs):
        skip = skip_opt if epoch == start_epoch else 0
        for micro_step, batch in enumerate(train_loader):
            is_last = micro_step == len(train_loader) - 1
            will_step = (micro_step + 1) % args.grad_accum == 0 or is_last
            if skip > 0:
                if will_step:
                    skip -= 1
                continue

            student.grl_lambda = dann_lambda(
                global_step, total_steps, cfg.grl_lambda_max)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=device.type == "cuda"):
                parts = _step(student, teacher, batch, cfg)
            (parts["loss"] / args.grad_accum).backward()
            for k, m in meters.items():
                m.update(parts[k].item())

            if not will_step:
                continue

            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.logging_steps == 0:
                log(f"step {global_step}/{total_steps} "
                    f"distill {meters['distill'].avg:.4f} sft {meters['sft'].avg:.4f} "
                    f"adv {meters['adv'].avg:.4f} adv_mae {meters['adv_mae'].avg:.4f} "
                    f"z {meters['z_norm'].avg:.1f} λ={student.grl_lambda:.4f}")
                for m in meters.values():
                    m.reset()

            if global_step % args.eval_steps == 0 or global_step == total_steps:
                ckpt_meta = {"step": global_step, "dataset": args.dataset}
                save_ckpt(student, tokenizer, out_dir / "last", ckpt_meta)
                save_trainer_state(out_dir / "last", optimizer, scheduler,
                                   global_step, best)
                val = evaluate(student, teacher, valid_loader, device, amp_dtype, cfg)
                log(f"  [eval] distill {val['distill']:.4f} sft {val['sft']:.4f} "
                    f"adv {val['adv']:.4f} adv_mae {val['adv_mae']:.4f} "
                    f"z {val['z_norm']:.1f} λ={student.grl_lambda:.4f}")
                preview(student, tokenizer, preview_set, device, args.preview_samples, log)
                score = val["loss"] if cfg.kd_weight == 0 else val["distill"]
                ckpt_meta.update({
                    "valid_distill": val["distill"],
                    "valid_loss": val["loss"],
                    "valid_sft": val["sft"],
                    "valid_adv": val["adv"],
                })
                save_ckpt(student, tokenizer, out_dir / "last", ckpt_meta)
                save_trainer_state(out_dir / "last", optimizer, scheduler,
                                   global_step, best)
                if score < best:
                    best = score
                    save_ckpt(student, tokenizer, out_dir / "best", ckpt_meta)
                    save_trainer_state(out_dir / "best", optimizer, scheduler,
                                       global_step, best)
                    log(f"  [eval] new best -> {out_dir / 'best'}")
            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    save_ckpt(student, tokenizer, out_dir / "last",
              {"step": global_step, "valid_distill": best, "dataset": args.dataset})
    save_trainer_state(out_dir / "last", optimizer, scheduler, global_step, best)
    save_json(out_dir / "train_done.json",
              {"step": global_step, "total_steps": total_steps, "best": best})
    log(f"done | best valid distill {best:.4f} | checkpoints under {out_dir}")
    log.close()


if __name__ == "__main__":
    main()
