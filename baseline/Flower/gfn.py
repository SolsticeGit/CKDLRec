from __future__ import annotations

import math
from typing import Sequence

import torch

INVALID_LOGR = -15.0


class _FlowNode:
    __slots__ = ("children", "weight", "item_ids", "log_next", "allowed")

    def __init__(self):
        self.children: dict[int, _FlowNode] = {}
        self.weight = 0.0
        self.item_ids: list[int] = []
        self.log_next: dict[int, float] = {}
        self.allowed: list[int] = []


class TokenFlow:

    def __init__(self, eos_id: int):
        self.eos_id = eos_id
        self.root = _FlowNode()
        self.max_len = 1
        self.n_titles = 0

    def insert(self, token_ids: Sequence[int], weight: float, item_id: int) -> None:
        if not token_ids:
            return
        w = float(max(weight, 1.0))
        node = self.root
        node.weight += w
        for tok in token_ids:
            tok = int(tok)
            child = node.children.get(tok)
            if child is None:
                child = _FlowNode()
                node.children[tok] = child
            node = child
            node.weight += w
        node.item_ids.append(item_id)
        eos = node.children.get(self.eos_id)
        if eos is None:
            eos = _FlowNode()
            node.children[self.eos_id] = eos
        eos.weight += w
        eos.item_ids.append(item_id)
        self.max_len = max(self.max_len, len(token_ids) + 1)
        self.n_titles += 1

    def freeze(self) -> None:
        stack = [self.root]
        while stack:
            node = stack.pop()
            total = sum(c.weight for c in node.children.values())
            if total <= 0:
                node.log_next = {}
                node.allowed = [self.eos_id]
            else:
                node.log_next = {
                    t: math.log(c.weight / total) for t, c in node.children.items()
                }
                node.allowed = list(node.children.keys())
            stack.extend(node.children.values())

    def walk(self, tokens: Sequence[int]) -> _FlowNode | None:
        node = self.root
        for tok in tokens:
            node = node.children.get(int(tok))
            if node is None:
                return None
        return node

    def lookup_item(self, tokens: Sequence[int]) -> int:
        cleaned = []
        for tok in tokens:
            t = int(tok)
            if t == self.eos_id:
                break
            cleaned.append(t)
        node = self.walk(cleaned)
        if node is None or not node.item_ids:
            return -1
        return node.item_ids[0]


def build_token_flow(titles: Sequence[str], weights: Sequence[float],
                     tokenizer, eos_id: int, batch_size: int = 4096) -> TokenFlow:
    flow = TokenFlow(eos_id)
    n = len(titles)
    for start in range(0, n, batch_size):
        chunk = list(titles[start:start + batch_size])
        encoded = tokenizer(chunk, add_special_tokens=False, padding=False)
        for offset, ids in enumerate(encoded["input_ids"]):
            idx = start + offset
            flow.insert(ids, float(weights[idx]), idx)
    flow.freeze()
    return flow


def normalize_sasrec_score(row: torch.Tensor, item_id: int) -> float:
    max_val = row.max()
    min_val = row.min()
    denom = (max_val - min_val).item()
    if denom == 0:
        return 1.0
    mean_val = row.float().mean().item()
    return (row[item_id].item() - mean_val) / denom + 1.0


def _mask_allowed(logits: torch.Tensor, allowed: list[int], term_id: int) -> torch.Tensor:
    if not allowed:
        allowed = [term_id]
    out = torch.full_like(logits, float("-inf"))
    idx = torch.tensor(allowed, device=logits.device, dtype=torch.long)
    out[idx] = logits[idx]
    return out


def generate_and_return_termination_logprob(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    flow: TokenFlow,
    termination_token_id: int,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = prompt_ids.device
    bsz = prompt_ids.size(0)
    state = prompt_ids
    attn = prompt_mask
    active = torch.ones(bsz, dtype=torch.bool, device=device)
    log_pf = []
    log_pterm = []
    past = None
    token_ids = state
    prompt_width = prompt_ids.size(1)

    for step in range(max_len):
        if past is None:
            out = model(input_ids=token_ids, attention_mask=attn, use_cache=True)
        else:
            out = model(input_ids=token_ids, attention_mask=attn,
                        past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        logprob = logits.log_softmax(dim=-1)

        with torch.no_grad():
            sampled = torch.full((bsz, 1), termination_token_id,
                                 dtype=torch.long, device=device)
            for i in range(bsz):
                if not active[i]:
                    continue
                prefix = state[i, prompt_width:].tolist()
                node = flow.walk(prefix)
                allowed = node.allowed if node is not None else [termination_token_id]
                masked = _mask_allowed(logits[i], allowed, termination_token_id)
                probs = masked.softmax(dim=-1)
                if not torch.isfinite(probs).any() or probs.sum() <= 0:
                    sampled[i, 0] = termination_token_id
                else:
                    sampled[i, 0] = torch.multinomial(probs, 1)

        sampled = torch.where(active.unsqueeze(-1), sampled,
                              torch.full_like(sampled, termination_token_id))
        log_pterm.append(torch.where(active, logprob[:, termination_token_id],
                                     torch.zeros_like(logprob[:, termination_token_id])))
        active = active & (sampled.squeeze(-1) != termination_token_id)
        log_pf.append(torch.where(
            active, logprob.gather(-1, sampled).squeeze(-1),
            torch.zeros(bsz, device=device, dtype=logprob.dtype),
        ))
        state = torch.cat([state, sampled], dim=1)
        attn = torch.cat([attn, torch.ones(bsz, 1, device=device, dtype=attn.dtype)], dim=1)
        token_ids = sampled
        if not active.any():
            break

    if not log_pf:
        z = torch.zeros(bsz, 1, device=device)
        return state, z, z
    return state, torch.stack(log_pf, dim=1), torch.stack(log_pterm, dim=1)


def token_log_rewards(
    generated: torch.Tensor,
    prompt_width: int,
    flow: TokenFlow,
    termination_token_id: int,
    sasrec_p: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    gen = generated[:, prompt_width:]
    bsz, tlen = gen.shape
    log_r = torch.full((bsz, tlen), INVALID_LOGR, device=generated.device, dtype=dtype)
    if tlen == 0:
        return log_r

    for i in range(bsz):
        tokens = gen[i].tolist()
        stop_at = tlen
        for j, tok in enumerate(tokens):
            if int(tok) == termination_token_id:
                stop_at = j
                break
        title = [int(t) for t in tokens[:stop_at]]
        item_id = flow.lookup_item(title)
        scale = 1.0
        if sasrec_p is not None and item_id >= 0:
            scale = float(sasrec_p[i].item()) + 1e-20

        node = flow.root
        data_logp = []
        valid = True
        for tok in title:
            if node is None or tok not in node.log_next:
                valid = False
                break
            data_logp.append(node.log_next[tok] / scale)
            node = node.children.get(tok)
        if not valid or node is None or termination_token_id not in node.log_next:
            continue
        if item_id < 0:
            continue

        term_prob = node.log_next[termination_token_id] / scale
        stop_logp = [INVALID_LOGR] * tlen
        complete = len(title)
        if complete < tlen:
            stop_logp[complete] = term_prob
        elif complete == tlen:
            stop_logp[tlen - 1] = term_prob

        log_r[i, 0] = stop_logp[0]
        running = 0.0
        for k in range(1, tlen):
            if k - 1 < len(data_logp):
                running += data_logp[k - 1]
            log_r[i, k] = running + stop_logp[k]
        seen_eos = False
        for k in range(tlen):
            if seen_eos:
                log_r[i, k] = 0.0
            if int(tokens[k]) == termination_token_id:
                seen_eos = True
    return log_r


def modified_subtb_loss(
    log_pf: torch.Tensor,
    log_r: torch.Tensor,
    log_pterm: torch.Tensor,
    generated_text: torch.Tensor,
    termination_token_id: int,
    prompt_len: int,
    subtb_lambda: float = 1.0,
) -> torch.Tensor:
    if log_pf.shape[1] <= 1:
        return log_pf.new_zeros(())
    assert log_pf.shape[1] == log_r.shape[1] == log_pterm.shape[1]
    assert log_pf.shape[1] == generated_text.shape[1] - prompt_len

    delta = (log_r[:, :-1] + log_pf[:, :-1] + log_pterm[:, 1:]
             - log_r[:, 1:] - log_pterm[:, :-1])
    delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], 1).cumsum(1)
    mask = (generated_text[:, prompt_len:-1] == termination_token_id).cumsum(-1) >= 1

    batch_loss = log_pf.new_zeros(())
    total_lambda = log_pf.new_zeros(())
    generated_len = generated_text.shape[1] - prompt_len
    for subtraj_len in range(1, generated_len):
        subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
        subtb_term = subtb_term.masked_fill(mask[:, subtraj_len - 1 :], 0)
        batch_loss = batch_loss + (subtb_lambda ** (subtraj_len - 1)) * subtb_term.sum()
        total_lambda = total_lambda + (
            (subtb_lambda ** (subtraj_len - 1)) * (~mask[:, subtraj_len - 1 :]).sum()
        )
    return batch_loss / total_lambda.clamp_min(1.0)
