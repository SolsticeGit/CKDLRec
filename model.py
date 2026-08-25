from __future__ import annotations

import math

import torch
import torch.nn as nn

from dataset import IGNORE_INDEX


def unwrap_causal(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def disable_sampling_flags(model: nn.Module) -> None:
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


def prompt_lengths(labels: torch.Tensor) -> torch.Tensor:
    return (labels != IGNORE_INDEX).long().argmax(dim=-1)


def dann_lambda(step: int, total_steps: int, lambd_max: float, gamma: float = 10.0) -> float:
    p = min(1.0, float(step) / max(int(total_steps), 1))
    return float(lambd_max) * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, float(lambd))


class PopAdvHead(nn.Module):

    def __init__(self, hidden_size: int, hidden_ratio: float = 0.5):
        super().__init__()
        mid = max(1, int(hidden_size * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.float()).squeeze(-1)


class CKDLRecModel(nn.Module):

    def __init__(self, llm: nn.Module, adv_head: nn.Module | None = None):
        super().__init__()
        self.llm = llm
        self.adv_head = adv_head
        self.grl_lambda = 0.0

    def _decoder(self) -> nn.Module:
        return unwrap_causal(self.llm).model

    def _lm_head(self) -> nn.Module:
        return unwrap_causal(self.llm).lm_head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        hidden = self._decoder()(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state
        logits = self._lm_head()(hidden)
        n_index = prompt_lengths(labels) - 1
        b = torch.arange(hidden.size(0), device=hidden.device)
        z = hidden[b, n_index]
        v_hat = None
        if self.adv_head is not None:
            v_hat = self.adv_head(grad_reverse(z, self.grl_lambda))
        return logits, z, v_hat

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 max_new_tokens: int, pad_token_id: int, eos_token_id: int,
                 num_beams: int = 1, num_return_sequences: int = 1,
                 prefix_allowed_tokens_fn=None, **generate_kwargs):
        self.eval()
        disable_sampling_flags(self.llm)
        gen_kw = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            temperature=None,
            top_p=None,
            top_k=None,
        )
        if num_return_sequences != 1:
            gen_kw["num_return_sequences"] = num_return_sequences
        if prefix_allowed_tokens_fn is not None:
            gen_kw["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        gen_kw.update(generate_kwargs)
        return self.llm.generate(**gen_kw)
