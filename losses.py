from __future__ import annotations

import torch
import torch.nn.functional as F

from dataset import IGNORE_INDEX


def sequence_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    target = labels[:, 1:]
    mask = target != IGNORE_INDEX
    gather_idx = target.clamp(min=0).unsqueeze(-1)
    token = logp.gather(-1, gather_idx).squeeze(-1)
    token = token.masked_fill(~mask, 0.0)
    denom = mask.sum(-1).clamp(min=1)
    return token.sum(-1) / denom


def pack_answer_logits(logits: torch.Tensor, labels: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    pred = logits[:, :-1]
    mask = labels[:, 1:] != IGNORE_INDEX
    lengths = mask.sum(dim=-1)
    tmax = int(lengths.max().clamp(min=1).item())
    start = mask.long().argmax(dim=-1)
    idx = start.unsqueeze(1) + torch.arange(tmax, device=logits.device).unsqueeze(0)
    idx = idx.clamp(max=pred.size(1) - 1)
    packed_mask = torch.arange(tmax, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    idx = idx.masked_fill(~packed_mask, 0)
    packed = pred.gather(1, idx.unsqueeze(-1).expand(-1, -1, pred.size(-1)))
    return packed, packed_mask


def distill_loss(logits_student: torch.Tensor, logits_teacher: torch.Tensor,
                 labels: torch.Tensor, tau: float) -> torch.Tensor:
    stu, mask = pack_answer_logits(logits_student, labels)
    tea, tea_mask = pack_answer_logits(logits_teacher, labels)
    t = min(stu.size(1), tea.size(1))
    token_mask = mask[:, :t] & tea_mask[:, :t]
    student_logp = F.log_softmax(stu[:, :t].float() / tau, dim=-1)
    teacher_prob = F.softmax(tea[:, :t].float() / tau, dim=-1).detach()
    kl = F.kl_div(student_logp, teacher_prob, reduction="none").sum(-1) * (tau ** 2)
    kl = kl * token_mask.float()
    token_den = token_mask.float().sum(-1).clamp(min=1)
    return (kl.sum(-1) / token_den).mean()


def student_loss(logits_student: torch.Tensor, logits_teacher: torch.Tensor | None,
                 labels: torch.Tensor, z: torch.Tensor, v_hat: torch.Tensor,
                 v_y: torch.Tensor, cfg) -> dict[str, torch.Tensor]:
    l_sft = -sequence_logp(logits_student, labels).mean()
    if logits_teacher is None or cfg.kd_weight == 0:
        l_kd = logits_student.new_zeros(())
    else:
        l_kd = distill_loss(logits_student, logits_teacher, labels, cfg.tau_distill)
    l_adv = F.mse_loss(v_hat.float(), v_y.float())
    total = logits_student.new_zeros(())
    if cfg.kd_weight != 0:
        total = total + cfg.kd_weight * l_kd
    if cfg.alpha != 0:
        total = total + cfg.alpha * l_sft
    if cfg.beta != 0:
        total = total + cfg.beta * l_adv
    with torch.no_grad():
        adv_mae = (v_hat.float() - v_y.float()).abs().mean()
        z_norm = z.float().norm(dim=-1).mean()
    return {
        "loss": total,
        "distill": l_kd.detach(),
        "sft": l_sft.detach(),
        "adv": l_adv.detach(),
        "adv_mae": adv_mae.detach(),
        "z_norm": z_norm.detach(),
    }
