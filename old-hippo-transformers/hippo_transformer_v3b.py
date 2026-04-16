"""
Hippocampal Transformer v3b: Variable Binding
===============================================

Task: multi-hop variable resolution requiring genuine depth.

  V0=7, V1=V0+3, V2=V1-2, V3=V2+4, V4=V3-1 ; ?V2=8, ?V4=11, ?V0=7, ...

To answer ?V2, the model must:
  1. Find V2's definition: V2 = V1 - 2
  2. Find V1's definition: V1 = V0 + 3
  3. Find V0's value: 7
  4. Compute: 7 -> 10 -> 8

Each hop plausibly needs a layer of attention + MLP. A 2-layer model
should handle 0-1 hops; a 4-layer model should handle 3-4.

The hippocampal model (2+hippo+1) should learn that common
computational trajectory patterns at layer 1 map to specific
final-layer outputs, shortcutting the intermediate resolution steps.

All arithmetic is mod 16, so answers are single tokens.

Comparison:
  1. 4-layer (upper bound)
  2. 2+hippo+1 (the test)
  3. 3-layer wider (param-matched control)
  4. 2-layer (lower bound)

Key metric: per-hop accuracy on answer tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
from dataclasses import dataclass


# =============================================================================
# Token Vocabulary
# =============================================================================

NUM_OFFSET = 0      # Tokens 0-15: numbers
VAR_OFFSET = 16     # Tokens 16-25: variables V0-V9
OP_ADD = 26
OP_SUB = 27
EQUALS = 28
SEP = 29            # Comma between items
QUERY = 30          # ? marker
SECTION = 31        # ; between definition and query sections
VOCAB_SIZE = 32


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Transformer
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    vocab_size: int = VOCAB_SIZE
    context_len: int = 128
    dropout: float = 0.0

    # Architecture
    n_layers: int = 4
    use_hippo: bool = False
    hippo_front_layers: int = 2
    hippo_back_layers: int = 1

    # Wider baseline
    wider_baseline: bool = False
    wider_d_ff: int = 512

    # Hippocampal component
    N_ca3: int = 256
    k_ca3: int = 15
    hebbian_lr: float = 0.5
    hebbian_decay: float = 0.998

    # Data
    chain_len: int = 5          # Variables V0..V{chain_len-1}
    operand_min: int = 1
    operand_max: int = 5

    # Training
    batch_size: int = 32
    n_steps: int = 8000
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 400
    eval_batches: int = 16
    seed: int = 42


# =============================================================================
# Variable Binding Data Generator
# =============================================================================

class VariableBindingDataset:
    def __init__(self, config, split='train'):
        self.config = config
        self.rng = np.random.RandomState(
            config.seed if split == 'train' else config.seed + 7777)

    def _make_sequence(self):
        cfg = self.config
        ctx_limit = cfg.context_len + 1

        # Generate chain values
        values = []
        ops = []
        operands = []

        base_val = self.rng.randint(0, 16)
        values.append(base_val)

        for i in range(1, cfg.chain_len):
            op = self.rng.choice([0, 1])  # 0=add, 1=sub
            operand = self.rng.randint(cfg.operand_min,
                                        cfg.operand_max + 1)
            if op == 0:
                val = (values[-1] + operand) % 16
            else:
                val = (values[-1] - operand) % 16
            values.append(val)
            ops.append(op)
            operands.append(operand)

        # Tokenize definitions
        tokens = []

        # V0 = base_val ,
        tokens.extend([VAR_OFFSET + 0, EQUALS, base_val, SEP])

        # Vi = V{i-1} op operand ,
        for i in range(1, cfg.chain_len):
            tokens.extend([
                VAR_OFFSET + i,
                EQUALS,
                VAR_OFFSET + (i - 1),
                OP_ADD if ops[i-1] == 0 else OP_SUB,
                operands[i-1],
                SEP,
            ])

        tokens.append(SECTION)

        # Generate query rounds to fill context
        # Each answer position is tracked with its hop depth
        answer_info = []  # (position_in_y, hop_depth)

        while len(tokens) < ctx_limit - cfg.chain_len * 5:
            order = self.rng.permutation(cfg.chain_len)
            for idx in order:
                if len(tokens) >= ctx_limit - 4:
                    break
                tokens.append(QUERY)
                tokens.append(VAR_OFFSET + idx)
                tokens.append(EQUALS)
                # Record: the answer token is at len(tokens) in tokens[]
                # In y = tokens[1:], this maps to index len(tokens) - 1
                ans_pos_in_y = len(tokens) - 1
                tokens.append(values[idx])
                tokens.append(SEP)

                if ans_pos_in_y < cfg.context_len:
                    answer_info.append((ans_pos_in_y, idx))

        # Pad if needed
        while len(tokens) < ctx_limit:
            tokens.append(SEP)

        return tokens[:ctx_limit], answer_info

    def get_batch(self, batch_size=None, device='cpu'):
        bs = batch_size or self.config.batch_size
        ctx = self.config.context_len

        all_tokens = np.zeros((bs, ctx + 1), dtype=np.int64)
        # For each position in y, store hop depth (-1 = not an answer)
        all_hops = np.full((bs, ctx), -1, dtype=np.int64)
        all_is_answer = np.zeros((bs, ctx), dtype=bool)

        for b in range(bs):
            tokens, answer_info = self._make_sequence()
            length = min(len(tokens), ctx + 1)
            all_tokens[b, :length] = tokens[:length]

            for pos_in_y, hop in answer_info:
                if 0 <= pos_in_y < ctx:
                    all_is_answer[b, pos_in_y] = True
                    all_hops[b, pos_in_y] = hop

        x = torch.tensor(all_tokens[:, :-1], device=device)
        y = torch.tensor(all_tokens[:, 1:], device=device)
        is_answer = torch.tensor(all_is_answer, device=device)
        hops = torch.tensor(all_hops, device=device)

        return x, y, is_answer, hops


# =============================================================================
# k-WTA with Straight-Through Estimator
# =============================================================================

class KWTAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        topk_vals, topk_idx = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk_idx, topk_vals)
        mask = (out != 0).float()
        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask, None


def kwta_ste(x, k):
    return KWTAFunction.apply(x, k)


# =============================================================================
# Hippocampal Computational Shortcut
# =============================================================================

class HippocampalShortcut(nn.Module):
    """Persistent cross-layer Hebbian map (same as v3)."""
    def __init__(self, d_model, N_ca3, k_ca3, hebbian_lr=0.5,
                 hebbian_decay=0.998):
        super().__init__()
        self.d_model = d_model
        self.N_ca3 = N_ca3
        self.k_ca3 = k_ca3
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay

        self.down_proj = nn.Linear(d_model, N_ca3, bias=False)
        self.up_proj = nn.Linear(N_ca3, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        self.register_buffer('W', torch.zeros(N_ca3, N_ca3))
        self.register_buffer('n_stored', torch.tensor(0, dtype=torch.long))

        nn.init.xavier_normal_(self.down_proj.weight, gain=0.5)
        nn.init.xavier_normal_(self.up_proj.weight, gain=0.1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self._diag = {}

    def _sparsify(self, x):
        h = F.relu(self.down_proj(x))
        h = kwta_ste(h, self.k_ca3)
        norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp(min=1e-10)
        return h / norm

    def forward(self, x):
        sparse = self._sparsify(x)
        h = F.relu(sparse @ self.W.T)
        h = kwta_ste(h, self.k_ca3)
        norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp(min=1e-10)
        h = h / norm

        prediction = self.up_proj(h)
        g = torch.sigmoid(self.gate_proj(x))
        contribution = g * prediction

        with torch.no_grad():
            self._diag = {
                'gate_mean': g.mean().item(),
                'contrib_norm': contribution.norm(dim=-1).mean().item(),
                'w_norm': self.W.norm().item(),
            }
        return x + contribution

    @torch.no_grad()
    def hebbian_update(self, input_residual, target_residual):
        self.W *= self.hebbian_decay

        h_in = F.relu(self.down_proj(input_residual))
        topk_v, topk_i = torch.topk(h_in, self.k_ca3, dim=-1)
        sparse_in = torch.zeros_like(h_in)
        sparse_in.scatter_(-1, topk_i, topk_v)
        sparse_in = sparse_in / sparse_in.norm(
            dim=-1, keepdim=True).clamp(min=1e-10)

        h_tgt = F.relu(self.down_proj(target_residual))
        topk_v, topk_i = torch.topk(h_tgt, self.k_ca3, dim=-1)
        sparse_tgt = torch.zeros_like(h_tgt)
        sparse_tgt.scatter_(-1, topk_i, topk_v)
        sparse_tgt = sparse_tgt / sparse_tgt.norm(
            dim=-1, keepdim=True).clamp(min=1e-10)

        B, T, N = sparse_in.shape
        flat_in = sparse_in.reshape(-1, N)
        flat_tgt = sparse_tgt.reshape(-1, N)
        n = flat_in.shape[0]

        delta = self.hebbian_lr * (flat_tgt.T @ flat_in) / n
        self.W += delta
        self.W.fill_diagonal_(0)
        self.n_stored += n

    def get_diagnostics(self):
        return self._diag


# =============================================================================
# Transformer Components
# =============================================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.d_head)
        att = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            att = att.masked_fill(mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


# =============================================================================
# GPT Model
# =============================================================================

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_len, config.d_model)

        d_ff = config.wider_d_ff if config.wider_baseline else config.d_ff

        if config.use_hippo:
            self.front_blocks = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads,
                                 d_ff, config.dropout)
                for _ in range(config.hippo_front_layers)
            ])
            self.hippo = HippocampalShortcut(
                d_model=config.d_model, N_ca3=config.N_ca3,
                k_ca3=config.k_ca3, hebbian_lr=config.hebbian_lr,
                hebbian_decay=config.hebbian_decay,
            )
            self.back_blocks = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads,
                                 d_ff, config.dropout)
                for _ in range(config.hippo_back_layers)
            ])
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads,
                                 d_ff, config.dropout)
                for _ in range(config.n_layers)
            ])
            self.front_blocks = None
            self.hippo = None
            self.back_blocks = None

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        mask = torch.tril(torch.ones(config.context_len, config.context_len))
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

        self._hippo_input = None
        self._hippo_target = None
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(positions)

        if self.hippo is not None:
            for block in self.front_blocks:
                h = block(h, self.causal_mask)
            self._hippo_input = h.detach().clone()
            h = self.hippo(h)
            for block in self.back_blocks:
                h = block(h, self.causal_mask)
            self._hippo_target = h.detach().clone()
        else:
            for block in self.blocks:
                h = block(h, self.causal_mask)

        h = self.ln_f(h)
        return self.head(h)

    def hippo_hebbian_update(self):
        if self.hippo is not None and self._hippo_input is not None:
            self.hippo.hebbian_update(self._hippo_input, self._hippo_target)
            self._hippo_input = None
            self._hippo_target = None

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        hippo_proj = 0
        if self.hippo is not None:
            hippo_proj = sum(p.numel() for p in self.hippo.parameters())
        return {'total': total, 'trainable': trainable,
                'hippo_projections': hippo_proj,
                'transformer_params': total - hippo_proj}

    def get_hippo_diagnostics(self):
        return self.hippo.get_diagnostics() if self.hippo else {}


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, dataset, config, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # Per-hop tracking
    hop_correct = {h: 0 for h in range(config.chain_len)}
    hop_total = {h: 0 for h in range(config.chain_len)}
    hop_loss = {h: 0.0 for h in range(config.chain_len)}

    # Overall answer accuracy
    ans_correct = 0
    ans_total = 0

    with torch.no_grad():
        for _ in range(config.eval_batches):
            x, y, is_answer, hops = dataset.get_batch(device=device)
            logits = model(x)

            # Full-sequence loss
            loss_all = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1), reduction='none'
            ).reshape(y.shape)
            total_loss += loss_all.sum().item()
            total_tokens += y.numel()

            # Predictions
            preds = logits.argmax(dim=-1)

            # Per-hop metrics
            for h in range(config.chain_len):
                mask = (hops == h)
                if mask.sum() > 0:
                    hop_correct[h] += (preds[mask] == y[mask]).sum().item()
                    hop_total[h] += mask.sum().item()
                    hop_loss[h] += loss_all[mask].sum().item()

            # Overall answer metrics
            if is_answer.sum() > 0:
                ans_correct += (preds[is_answer] == y[is_answer]).sum().item()
                ans_total += is_answer.sum().item()

    results = {
        'total_loss': total_loss / max(total_tokens, 1),
        'answer_acc': ans_correct / max(ans_total, 1),
        'answer_loss': sum(hop_loss.values()) / max(ans_total, 1),
        'hop_acc': {
            h: hop_correct[h] / max(hop_total[h], 1)
            for h in range(config.chain_len)
        },
        'hop_loss': {
            h: hop_loss[h] / max(hop_total[h], 1)
            for h in range(config.chain_len)
        },
    }
    model.train()
    return results


# =============================================================================
# Training
# =============================================================================

def train_model(config, device, label=""):
    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"{'='*70}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = GPT(config).to(device)
    params = model.count_params()
    print(f"  Params: {params['total']:,} total "
          f"({params['transformer_params']:,} transformer"
          f", {params['hippo_projections']:,} hippo)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )

    train_data = VariableBindingDataset(config, split='train')
    eval_data = VariableBindingDataset(config, split='eval')

    history = []
    t0 = time.time()

    for step in range(1, config.n_steps + 1):
        x, y, _, _ = train_data.get_batch(device=device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if config.use_hippo:
            model.hippo_hebbian_update()

        if step % config.eval_every == 0 or step == 1:
            ev = evaluate(model, eval_data, config, device)
            elapsed = time.time() - t0

            entry = {'step': step, 'train_loss': loss.item(),
                     **ev, 'elapsed': elapsed}
            if config.use_hippo:
                entry['hippo'] = model.get_hippo_diagnostics()

            history.append(entry)

            hop_str = " ".join(
                f"h{h}={ev['hop_acc'][h]:.0%}"
                for h in range(config.chain_len)
            )

            hippo_str = ""
            if 'hippo' in entry:
                d = entry['hippo']
                hippo_str = (f" | gate={d.get('gate_mean',0):.3f}"
                             f" W={d.get('w_norm',0):.1f}")

            print(f"  step {step:5d} | loss {ev['total_loss']:.4f} | "
                  f"ans_acc {ev['answer_acc']:.1%} | "
                  f"{hop_str}"
                  f"{hippo_str}"
                  f" | {elapsed:.0f}s")

    return model, history


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(histories, labels, config, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_hops = config.chain_len
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, max(3, n_hops), figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle("Hippocampal Transformer: Variable Binding Task",
                 fontsize=14, fontweight='bold')

    colors = ['steelblue', 'coral', 'forestgreen', 'mediumpurple']

    # Row 1: Overall metrics
    ax = fig.add_subplot(gs[0, 0])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['total_loss'] for h in hist],
                '-', color=colors[i], lw=2, label=lab)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Overall Loss")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['answer_acc'] for h in hist],
                '-', color=colors[i], lw=2, label=lab)
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_title("Answer Accuracy (all hops)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    ax = fig.add_subplot(gs[0, 2])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['answer_loss'] for h in hist],
                '-', color=colors[i], lw=2, label=lab)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Answer Token Loss")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Row 2: Per-hop accuracy curves
    for h in range(min(n_hops, 5)):
        ax = fig.add_subplot(gs[1, h])
        for i, (hist, lab) in enumerate(zip(histories, labels)):
            steps = [entry['step'] for entry in hist]
            accs = [entry['hop_acc'][h] for entry in hist]
            ax.plot(steps, accs, '-', color=colors[i], lw=2, label=lab)
        ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
        ax.set_title(f"Hop {h} Accuracy")
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # Row 3: Final per-hop comparison + hippo diagnostics
    ax = fig.add_subplot(gs[2, 0:2])
    x_pos = np.arange(n_hops)
    width = 0.18
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        final_accs = [hist[-1]['hop_acc'][h] for h in range(n_hops)]
        ax.bar(x_pos + i * width, final_accs, width,
               color=colors[i], label=lab, alpha=0.85)
    ax.set_xlabel("Hop Depth"); ax.set_ylabel("Accuracy")
    ax.set_title("Final Per-Hop Accuracy")
    ax.set_xticks(x_pos + width * 1.5)
    ax.set_xticklabels([f"Hop {h}" for h in range(n_hops)])
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    # Hippo diagnostics
    ax = fig.add_subplot(gs[2, 2])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        if 'hippo' in hist[0]:
            steps = [h['step'] for h in hist]
            ax.plot(steps, [h['hippo']['gate_mean'] for h in hist],
                    '-', color=colors[i], lw=2, label=f"{lab} gate")
            ax2 = ax.twinx()
            ax2.plot(steps, [h['hippo']['w_norm'] for h in hist],
                     '--', color=colors[i], lw=1.5, alpha=0.6,
                     label=f"{lab} W norm")
            ax2.set_ylabel("W norm")
    ax.set_xlabel("Step"); ax.set_ylabel("Gate")
    ax.set_title("Hippocampal Diagnostics")
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {save_path}")


def print_summary(histories, labels, config):
    print(f"\n{'='*70}")
    print("  FINAL RESULTS: Variable Binding")
    print(f"{'='*70}")

    # Per-hop accuracy table
    print(f"\n  Per-hop accuracy:")
    print(f"  {'Model':<22}", end="")
    for h in range(config.chain_len):
        print(f"  {'Hop '+str(h):>8}", end="")
    print(f"  {'Overall':>8}")
    print(f"  {'-'* (22 + 9 * (config.chain_len + 1))}")

    for hist, label in zip(histories, labels):
        f = hist[-1]
        print(f"  {label:<22}", end="")
        for h in range(config.chain_len):
            print(f"  {f['hop_acc'][h]:>7.1%}", end="")
        print(f"  {f['answer_acc']:>7.1%}")

    # Key comparisons
    if len(histories) >= 4:
        print(f"\n  Depth effect (4L vs 2L):")
        for h in range(config.chain_len):
            acc_4L = histories[0][-1]['hop_acc'][h]
            acc_2L = histories[3][-1]['hop_acc'][h]
            gap = acc_4L - acc_2L
            print(f"    Hop {h}: 4L={acc_4L:.1%}, 2L={acc_2L:.1%}, "
                  f"gap={gap:+.1%}")

        print(f"\n  Hippo effect (2+hippo+1 vs 3L-wider):")
        for h in range(config.chain_len):
            acc_hippo = histories[1][-1]['hop_acc'][h]
            acc_3Lw = histories[2][-1]['hop_acc'][h]
            diff = acc_hippo - acc_3Lw
            marker = " ***" if diff > 0.02 else ""
            print(f"    Hop {h}: hippo={acc_hippo:.1%}, "
                  f"3Lw={acc_3Lw:.1%}, diff={diff:+.1%}{marker}")


# =============================================================================
# Main
# =============================================================================

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data sanity check
    cfg = Config()
    ds = VariableBindingDataset(cfg)
    x, y, is_ans, hops = ds.get_batch(batch_size=1)
    print(f"\nSample sequence (first 60 tokens of x):")
    tokens = x[0, :60].tolist()
    tok_names = []
    for t in tokens:
        if t < 16:
            tok_names.append(str(t))
        elif t < 26:
            tok_names.append(f"V{t-16}")
        elif t == OP_ADD:
            tok_names.append("+")
        elif t == OP_SUB:
            tok_names.append("-")
        elif t == EQUALS:
            tok_names.append("=")
        elif t == SEP:
            tok_names.append(",")
        elif t == QUERY:
            tok_names.append("?")
        elif t == SECTION:
            tok_names.append(";")
        else:
            tok_names.append(f"[{t}]")
    print(f"  {' '.join(tok_names)}")

    n_ans = is_ans.sum().item()
    print(f"  Answer tokens: {n_ans}")
    for h in range(cfg.chain_len):
        c = (hops == h).sum().item()
        print(f"  Hop {h}: {c} answers")

    # ---- 1. Four-layer baseline ----
    cfg_4L = Config(n_layers=4, seed=42)
    _, hist_4L = train_model(cfg_4L, device, "4-layer (upper bound)")

    # ---- 2. Hippocampal model ----
    cfg_hippo = Config(
        use_hippo=True, hippo_front_layers=2, hippo_back_layers=1,
        N_ca3=256, k_ca3=15, hebbian_lr=0.5, hebbian_decay=0.998,
        seed=42,
    )
    _, hist_hippo = train_model(cfg_hippo, device, "2+hippo+1")

    # ---- 3. Three-layer wider (param matched) ----
    hippo_p = GPT(cfg_hippo).count_params()['total']
    base_3L_p = GPT(Config(n_layers=3)).count_params()['total']
    extra = hippo_p - base_3L_p
    wider_ff = int(Config().d_ff + extra / (3 * 2 * Config().d_model))

    cfg_3Lw = Config(
        n_layers=3, wider_baseline=True, wider_d_ff=wider_ff, seed=42)
    p3w = GPT(cfg_3Lw).count_params()['total']
    print(f"\n  Params: 4L={GPT(cfg_4L).count_params()['total']:,}, "
          f"hippo={hippo_p:,}, 3Lw={p3w:,}, "
          f"3L={base_3L_p:,}")

    _, hist_3Lw = train_model(cfg_3Lw, device, "3-layer wider (control)")

    # ---- 4. Two-layer baseline ----
    cfg_2L = Config(n_layers=2, seed=42)
    _, hist_2L = train_model(cfg_2L, device, "2-layer (lower bound)")

    # ---- Results ----
    histories = [hist_4L, hist_hippo, hist_3Lw, hist_2L]
    labels = ["4-layer", "2+hippo+1", "3-layer wider", "2-layer"]

    print_summary(histories, labels, cfg_4L)
    plot_comparison(histories, labels, cfg_4L, "hippo_v3b_results.png")

    results = {}
    for label, hist in zip(labels, histories):
        results[label] = hist
    with open("hippo_v3b_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
