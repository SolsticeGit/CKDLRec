from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys):
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(keys)
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5

        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)
        paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output = torch.where(torch.eq(key_mask_reshaped, 0), paddings, matmul_output)

        tril = torch.tril(torch.ones_like(matmul_output[0, :, :]))
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)
        matmul_output = torch.where(torch.eq(causality_mask, 0), paddings, matmul_output)
        matmul_output = self.softmax(matmul_output)

        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])
        matmul_output = self.dropout(matmul_output * query_mask)
        output = torch.bmm(matmul_output, V_)
        output = torch.cat(torch.split(output, output.shape[0] // self.num_heads, dim=0), dim=2)
        return output + queries


class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.device = device
        self.item_embeddings = nn.Embedding(item_num + 1, hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num)

    def forward(self, states, len_states):
        pos = torch.arange(self.state_size, device=states.device)
        seq = self.item_embeddings(states) + self.positional_embeddings(pos)
        seq = self.emb_dropout(seq)
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        seq = seq * mask
        ff_out = self.feed_forward(self.ln_2(self.mh_attn(self.ln_1(seq), seq)))
        ff_out = self.ln_3(ff_out * mask)
        indices = (len_states - 1).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
        hidden = torch.gather(ff_out, 1, indices)
        return self.s_fc(hidden).squeeze(1)


class _SeqDataset(Dataset):
    def __init__(self, seqs: np.ndarray, lens: np.ndarray, targets: np.ndarray):
        self.seqs = seqs
        self.lens = lens
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return (
            torch.tensor(self.seqs[i], dtype=torch.long),
            torch.tensor(self.lens[i], dtype=torch.long),
            torch.tensor(self.targets[i], dtype=torch.long),
        )


def pack_histories(records: Sequence[dict], n_items: int, state_size: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs, lens, targets = [], [], []
    pad = n_items
    for rec in records:
        hist = [int(x) for x in rec["history"]][-state_size:]
        n = len(hist)
        if n == 0:
            continue
        seqs.append(hist + [pad] * (state_size - n))
        lens.append(n)
        targets.append(int(rec["target"]))
    return (np.asarray(seqs, dtype=np.int64),
            np.asarray(lens, dtype=np.int64),
            np.asarray(targets, dtype=np.int64))


@torch.no_grad()
def ndcg_at(model: SASRec, loader: DataLoader, device: torch.device, k: int = 20) -> float:
    model.eval()
    total = 0.0
    n = 0
    for seq, len_seq, target in loader:
        seq = seq.to(device)
        len_seq = len_seq.to(device)
        target = target.to(device)
        logits = model(seq, len_seq)
        rank = (logits.shape[1] - 1 - torch.argsort(torch.argsort(logits)))
        target_rank = torch.gather(rank, 1, target.view(-1, 1)).view(-1)
        mask = (target_rank < k).float()
        total += ((1.0 / torch.log2(target_rank + 2)) * mask).sum().item()
        n += target.numel()
    model.train()
    return total / max(n, 1)


def train_sasrec(
    train_records: list[dict],
    valid_records: list[dict],
    n_items: int,
    state_size: int,
    device: torch.device,
    log,
    epochs: int = 200,
    batch_size: int = 1024,
    hidden: int = 64,
    dropout: float = 0.1,
    lr: float = 1e-3,
    l2: float = 1e-5,
    early_stop: int = 20,
    seed: int = 42,
) -> SASRec:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    model = SASRec(hidden, n_items, state_size, dropout, device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=l2)
    bce = nn.BCEWithLogitsLoss()

    tr_seq, tr_len, tr_y = pack_histories(train_records, n_items, state_size)
    va_seq, va_len, va_y = pack_histories(valid_records, n_items, state_size)
    train_loader = DataLoader(
        _SeqDataset(tr_seq, tr_len, tr_y), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(
        _SeqDataset(va_seq, va_len, va_y), batch_size=batch_size, shuffle=False)

    best_ndcg = -1.0
    best_state = None
    stall = 0
    model.train()
    for epoch in range(epochs):
        for seq, len_seq, target in train_loader:
            bsz = target.size(0)
            neg = torch.randint(0, n_items, (bsz,), generator=g)
            same = neg == target
            while same.any():
                neg = torch.where(same, torch.randint(0, n_items, (bsz,), generator=g), neg)
                same = neg == target
            seq = seq.to(device)
            len_seq = len_seq.to(device)
            target = target.to(device)
            neg = neg.to(device)
            logits = model(seq, len_seq)
            pos = torch.gather(logits, 1, target.view(-1, 1))
            neg_s = torch.gather(logits, 1, neg.view(-1, 1))
            scores = torch.cat((pos, neg_s), 0)
            labels = torch.cat((torch.ones_like(pos), torch.zeros_like(neg_s)), 0)
            loss = bce(scores, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        ndcg = ndcg_at(model, valid_loader, device, k=20)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
            log(f"  SASRec epoch {epoch} ndcg@20={ndcg:.4f} *best*")
        else:
            stall += 1
            if epoch % 10 == 0 or stall >= early_stop:
                log(f"  SASRec epoch {epoch} ndcg@20={ndcg:.4f} stall={stall}")
            if stall >= early_stop:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    log(f"  SASRec done | best valid ndcg@20={best_ndcg:.4f}")
    return model


def pack_batch(histories: Sequence[Sequence[int]], n_items: int, state_size: int,
               device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pad = n_items
    seqs, lens = [], []
    for hist in histories:
        h = [int(x) for x in hist][-state_size:]
        n = max(len(h), 1)
        if not h:
            h = [0]
        seqs.append(h + [pad] * (state_size - len(h)))
        lens.append(min(n, state_size))
    return (torch.tensor(seqs, dtype=torch.long, device=device),
            torch.tensor(lens, dtype=torch.long, device=device))


@torch.no_grad()
def sasrec_scores(model: SASRec, histories: Sequence[Sequence[int]],
                  device: torch.device) -> torch.Tensor:
    seq, lens = pack_batch(histories, model.item_num, model.state_size, device)
    model.eval()
    return model(seq, lens)
