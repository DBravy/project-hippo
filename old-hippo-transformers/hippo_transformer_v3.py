"""
Hippocampal Transformer v3: Computational Shortcut
====================================================

Core hypothesis: the hippocampus doesn't make computation more accurate,
it lets you SKIP computation. A shallow model + hippocampal shortcut
should approach a deeper model's performance.

Architecture comparison:
  1. 4-layer baseline (upper bound)
  2. 2 front layers + hippo + 1 back layer (the test)
  3. 3-layer wider baseline (param-matched control)
  4. 2-layer baseline (lower bound)

The hippocampal component sits where layer 2 would be. It learns a
persistent Hebbian map from layer-1 residual patterns to final-output
residual patterns. During each forward pass:
  - Project layer-1 residual to sparse CA3 space
  - Retrieve the predicted final-state pattern from W
  - Project back up and inject into the residual stream
  - The back layer refines this prediction

The Hebbian W accumulates ACROSS training batches (not within-context):
  - After each forward pass, store the association between
    sparse(layer_1_output) and sparse(final_output)
  - W decays exponentially each step to forget stale associations
  - This is the cross-example "computational motif memory"

Task: synthetic sequences from 200 fixed motifs. Enough motifs that
model depth matters for memorization and retrieval.
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
# Configuration
# =============================================================================

@dataclass
class Config:
    # Transformer
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    vocab_size: int = 64
    context_len: int = 128
    dropout: float = 0.0

    # Architecture
    n_layers: int = 4             # Used when use_hippo=False
    use_hippo: bool = False
    hippo_front_layers: int = 2   # Layers before hippo
    hippo_back_layers: int = 1    # Layers after hippo

    # Wider baseline
    wider_baseline: bool = False
    wider_d_ff: int = 512

    # Hippocampal component
    N_ca3: int = 256
    k_ca3: int = 15
    hebbian_lr: float = 0.5
    hebbian_decay: float = 0.998

    # Data
    n_motifs: int = 200
    motif_len_min: int = 4
    motif_len_max: int = 8
    filler_max: int = 1

    # Training
    batch_size: int = 16
    n_steps: int = 5000
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 250
    eval_batches: int = 8
    seed: int = 42


# =============================================================================
# Synthetic Data: Motif Sequences
# =============================================================================

class MotifDataset:
    """Fixed motifs concatenated with optional filler."""
    def __init__(self, config, split='train'):
        self.config = config
        self.rng = np.random.RandomState(
            config.seed if split == 'train' else config.seed + 999)

        motif_rng = np.random.RandomState(config.seed)
        self.motifs = []
        for _ in range(config.n_motifs):
            length = motif_rng.randint(config.motif_len_min,
                                       config.motif_len_max + 1)
            motif = motif_rng.randint(0, config.vocab_size, size=length)
            self.motifs.append(motif)

    def get_batch(self, batch_size=None, device='cpu'):
        bs = batch_size or self.config.batch_size
        ctx = self.config.context_len

        tokens = np.zeros((bs, ctx + 1), dtype=np.int64)
        is_motif = np.zeros((bs, ctx + 1), dtype=bool)
        motif_pos = np.full((bs, ctx + 1), -1, dtype=np.int64)

        for b in range(bs):
            pos = 0
            while pos < ctx + 1:
                # Optional filler
                n_fill = self.rng.randint(0, self.config.filler_max + 1)
                for _ in range(n_fill):
                    if pos >= ctx + 1:
                        break
                    tokens[b, pos] = self.rng.randint(0, self.config.vocab_size)
                    pos += 1

                midx = self.rng.randint(0, len(self.motifs))
                motif = self.motifs[midx]
                for mi, tok in enumerate(motif):
                    if pos >= ctx + 1:
                        break
                    tokens[b, pos] = tok
                    is_motif[b, pos] = True
                    motif_pos[b, pos] = mi
                    pos += 1

        x = torch.tensor(tokens[:, :-1], device=device)
        y = torch.tensor(tokens[:, 1:], device=device)
        motif_mask = torch.tensor(is_motif[:, 1:], device=device)
        m_pos = torch.tensor(motif_pos[:, 1:], device=device)

        return x, y, motif_mask, m_pos


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
    """
    Learns a persistent Hebbian map from early-layer residual patterns
    to final-output residual patterns across training.

    Forward pass (differentiable through projections):
      residual -> down_proj -> ReLU -> k-WTA -> W @ sparse -> k-WTA ->
      up_proj -> gate -> add to residual

    Hebbian update (after each training step, no gradients):
      Store association between sparse(layer_1_output) and
      sparse(final_output) into W.
    """
    def __init__(self, d_model, N_ca3, k_ca3, hebbian_lr=0.5,
                 hebbian_decay=0.998):
        super().__init__()
        self.d_model = d_model
        self.N_ca3 = N_ca3
        self.k_ca3 = k_ca3
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay

        # Learned projections (backprop)
        self.down_proj = nn.Linear(d_model, N_ca3, bias=False)
        self.up_proj = nn.Linear(N_ca3, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # Persistent Hebbian map (NOT backprop)
        self.register_buffer('W', torch.zeros(N_ca3, N_ca3))
        self.register_buffer('n_stored', torch.tensor(0, dtype=torch.long))

        nn.init.xavier_normal_(self.down_proj.weight, gain=0.5)
        nn.init.xavier_normal_(self.up_proj.weight, gain=0.1)
        nn.init.zeros_(self.gate_proj.weight)
        # Gate starts at ~0.5 (neutral)
        nn.init.constant_(self.gate_proj.bias, 0.0)

        self._diag = {}

    def _sparsify(self, x):
        """x: (..., d_model) -> (..., N_ca3), sparse + normalized"""
        h = F.relu(self.down_proj(x))
        h = kwta_ste(h, self.k_ca3)
        norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp(min=1e-10)
        return h / norm

    def forward(self, x):
        """
        x: (batch, seq_len, d_model) -- layer-1 output
        Returns: x + gated hippocampal prediction
        """
        sparse = self._sparsify(x)  # (B, T, N)

        # Retrieve: what does this input pattern typically map to?
        h = F.relu(sparse @ self.W.T)  # W is buffer, no grad
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
                'retrieval_norm': h.norm(dim=-1).mean().item(),
            }

        return x + contribution

    @torch.no_grad()
    def hebbian_update(self, input_residual, target_residual):
        """
        Store cross-layer association in W.

        input_residual:  (B, T, d_model) - layer-1 output (before hippo)
        target_residual: (B, T, d_model) - final output (after back layers)

        W learns: given sparse(input), predict sparse(target).
        This is the "computational motif" memory.
        """
        # Decay
        self.W *= self.hebbian_decay

        # Project both to sparse CA3 space using the same projection
        h_in = F.relu(self.down_proj(input_residual))
        topk_v, topk_i = torch.topk(h_in, self.k_ca3, dim=-1)
        sparse_in = torch.zeros_like(h_in)
        sparse_in.scatter_(-1, topk_i, topk_v)
        norm_in = torch.linalg.norm(
            sparse_in, dim=-1, keepdim=True).clamp(min=1e-10)
        sparse_in = sparse_in / norm_in

        h_tgt = F.relu(self.down_proj(target_residual))
        topk_v, topk_i = torch.topk(h_tgt, self.k_ca3, dim=-1)
        sparse_tgt = torch.zeros_like(h_tgt)
        sparse_tgt.scatter_(-1, topk_i, topk_v)
        norm_tgt = torch.linalg.norm(
            sparse_tgt, dim=-1, keepdim=True).clamp(min=1e-10)
        sparse_tgt = sparse_tgt / norm_tgt

        B, T, N = sparse_in.shape
        flat_in = sparse_in.reshape(-1, N)
        flat_tgt = sparse_tgt.reshape(-1, N)

        # Hetero-associative Hebbian: map input -> target
        n_samples = flat_in.shape[0]
        delta = self.hebbian_lr * (flat_tgt.T @ flat_in) / n_samples
        self.W += delta
        self.W.fill_diagonal_(0)

        self.n_stored += n_samples

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
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
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
            # Split architecture: front layers + hippo + back layers
            self.front_blocks = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads,
                                 d_ff, config.dropout)
                for _ in range(config.hippo_front_layers)
            ])
            self.hippo = HippocampalShortcut(
                d_model=config.d_model,
                N_ca3=config.N_ca3,
                k_ca3=config.k_ca3,
                hebbian_lr=config.hebbian_lr,
                hebbian_decay=config.hebbian_decay,
            )
            self.back_blocks = nn.ModuleList([
                TransformerBlock(config.d_model, config.n_heads,
                                 d_ff, config.dropout)
                for _ in range(config.hippo_back_layers)
            ])
            self.blocks = None
        else:
            # Standard architecture
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

        # For storing activations for Hebbian update
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
            # Front layers
            for block in self.front_blocks:
                h = block(h, self.causal_mask)

            # Save pre-hippo state for Hebbian update
            self._hippo_input = h.detach().clone()

            # Hippocampal shortcut
            h = self.hippo(h)

            # Back layers
            for block in self.back_blocks:
                h = block(h, self.causal_mask)

            # Save post-processing state for Hebbian target
            self._hippo_target = h.detach().clone()
        else:
            for block in self.blocks:
                h = block(h, self.causal_mask)

        h = self.ln_f(h)
        return self.head(h)

    def hippo_hebbian_update(self):
        """Call after backward pass each training step."""
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
        n_transformer = total - hippo_proj
        return {'total': total, 'trainable': trainable,
                'hippo_projections': hippo_proj,
                'transformer_params': n_transformer}

    def get_hippo_diagnostics(self):
        if self.hippo is not None:
            return self.hippo.get_diagnostics()
        return {}


# =============================================================================
# Training and Evaluation
# =============================================================================

def evaluate(model, dataset, config, device):
    model.eval()
    total_loss = 0.0
    motif_loss = 0.0
    filler_loss = 0.0
    motif_count = 0
    filler_count = 0
    pos_losses = {}
    pos_counts = {}

    with torch.no_grad():
        for _ in range(config.eval_batches):
            x, y, motif_mask, m_pos = dataset.get_batch(device=device)
            logits = model(x)
            loss_per_tok = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1), reduction='none'
            ).reshape(y.shape)

            total_loss += loss_per_tok.mean().item()

            m = motif_mask.float()
            f = 1.0 - m
            if m.sum() > 0:
                motif_loss += (loss_per_tok * m).sum().item()
                motif_count += m.sum().item()
            if f.sum() > 0:
                filler_loss += (loss_per_tok * f).sum().item()
                filler_count += f.sum().item()

            for p in range(config.motif_len_max + 1):
                pmask = (m_pos == p).float()
                if pmask.sum() > 0:
                    if p not in pos_losses:
                        pos_losses[p] = 0.0
                        pos_counts[p] = 0
                    pos_losses[p] += (loss_per_tok * pmask).sum().item()
                    pos_counts[p] += pmask.sum().item()

    n = config.eval_batches
    results = {
        'total_loss': total_loss / n,
        'motif_loss': motif_loss / max(motif_count, 1),
        'filler_loss': filler_loss / max(filler_count, 1),
        'per_position': {
            p: pos_losses[p] / pos_counts[p]
            for p in sorted(pos_losses.keys())
        },
    }
    model.train()
    return results


def train_model(config, device, label=""):
    print(f"\n{'='*70}")
    print(f"  Training: {label}")
    print(f"{'='*70}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = GPT(config).to(device)
    params = model.count_params()
    print(f"  Total params: {params['total']:,}  "
          f"(transformer: {params['transformer_params']:,}"
          f", hippo proj: {params['hippo_projections']:,})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )

    train_data = MotifDataset(config, split='train')
    eval_data = MotifDataset(config, split='eval')

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

        # Hebbian update after backward pass
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

            hippo_str = ""
            if 'hippo' in entry:
                d = entry['hippo']
                hippo_str = (f" | gate={d.get('gate_mean',0):.3f}"
                             f" W={d.get('w_norm',0):.1f}"
                             f" contrib={d.get('contrib_norm',0):.3f}")

            print(f"  step {step:5d} | train {loss.item():.4f} | "
                  f"eval {ev['total_loss']:.4f} | "
                  f"motif {ev['motif_loss']:.4f} | "
                  f"filler {ev['filler_loss']:.4f}"
                  f"{hippo_str}"
                  f" | {elapsed:.1f}s")

    return model, history


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(histories, labels, save_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(
        "Hippocampal Transformer v3: Computational Shortcut",
        fontsize=14, fontweight='bold')

    colors = ['steelblue', 'coral', 'forestgreen', 'mediumpurple']

    # 1. Overall loss
    ax = fig.add_subplot(gs[0, 0])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['total_loss'] for h in hist],
                '-', color=colors[i], linewidth=2, label=lab)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Overall Eval Loss")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 2. Motif loss
    ax = fig.add_subplot(gs[0, 1])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['motif_loss'] for h in hist],
                '-', color=colors[i], linewidth=2, label=lab)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Motif Token Loss")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 3. Filler loss
    ax = fig.add_subplot(gs[0, 2])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['filler_loss'] for h in hist],
                '-', color=colors[i], linewidth=2, label=lab)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Filler Token Loss")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 4. Per-position loss at final eval
    ax = fig.add_subplot(gs[1, 0])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        pp = hist[-1]['per_position']
        positions = sorted(pp.keys())
        ax.plot(positions, [pp[p] for p in positions],
                'o-', color=colors[i], linewidth=2, label=lab,
                markersize=4, alpha=0.85)
    ax.set_xlabel("Position in Motif"); ax.set_ylabel("Loss")
    ax.set_title("Loss by Position in Motif (final)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 5. Hippo gate
    ax = fig.add_subplot(gs[1, 1])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        if 'hippo' in hist[0]:
            steps = [h['step'] for h in hist]
            ax.plot(steps, [h['hippo']['gate_mean'] for h in hist],
                    '-', color=colors[i], linewidth=2, label=f"{lab} gate")
    ax.set_xlabel("Step"); ax.set_ylabel("Gate (sigmoid)")
    ax.set_title("Hippocampal Gate")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # 6. Hippo W norm
    ax = fig.add_subplot(gs[1, 2])
    for i, (hist, lab) in enumerate(zip(histories, labels)):
        if 'hippo' in hist[0]:
            steps = [h['step'] for h in hist]
            ax.plot(steps, [h['hippo']['w_norm'] for h in hist],
                    '-', color=colors[i], linewidth=2, label=f"{lab} ||W||")
    ax.set_xlabel("Step"); ax.set_ylabel("||W||")
    ax.set_title("CA3 Weight Norm")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved to {save_path}")


def print_summary(histories, labels):
    print(f"\n{'='*70}")
    print("  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<28} {'Eval':>8} {'Motif':>8} {'Filler':>8} "
          f"{'Params':>10}")
    print(f"  {'-'*66}")
    for hist, label in zip(histories, labels):
        f = hist[-1]
        print(f"  {label:<28} {f['total_loss']:>8.4f} "
              f"{f['motif_loss']:>8.4f} {f['filler_loss']:>8.4f}")

    print(f"\n  Per-position motif loss:")
    print(f"  {'Pos':>4}", end="")
    for lab in labels:
        print(f"  {lab:>14}", end="")
    print()
    all_pos = sorted(histories[0][-1]['per_position'].keys())
    for p in all_pos:
        print(f"  {p:>4}", end="")
        for hist in histories:
            val = hist[-1]['per_position'].get(p, float('nan'))
            print(f"  {val:>14.4f}", end="")
        print()

    # Key comparison
    if len(histories) >= 3:
        upper = histories[0][-1]['motif_loss']    # 4-layer
        hippo_v = histories[1][-1]['motif_loss']   # hippo
        control = histories[2][-1]['motif_loss']   # 3-layer wider

        gap_closed = 0.0
        if control != upper:
            gap_closed = (control - hippo_v) / (control - upper) * 100

        print(f"\n  4-layer motif loss:          {upper:.4f}")
        print(f"  2+hippo+1 motif loss:        {hippo_v:.4f}")
        print(f"  3-layer (control) motif loss: {control:.4f}")
        print(f"  Gap closed toward 4-layer:   {gap_closed:.1f}%")


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

    # ---- 1. Four-layer baseline (upper bound) ----
    config_4L = Config(n_layers=4, use_hippo=False, seed=42)
    _, hist_4L = train_model(config_4L, device, "4-layer (upper bound)")

    # ---- 2. Hippocampal model: 2 front + hippo + 1 back ----
    config_hippo = Config(
        use_hippo=True,
        hippo_front_layers=2,
        hippo_back_layers=1,
        N_ca3=256,
        k_ca3=15,
        hebbian_lr=0.5,
        hebbian_decay=0.998,
        seed=42,
    )
    _, hist_hippo = train_model(config_hippo, device, "2+hippo+1")

    # ---- 3. Three-layer wider baseline (param-matched) ----
    hippo_params = GPT(config_hippo).count_params()['total']
    base_3L_config = Config(n_layers=3, use_hippo=False, seed=42)
    base_3L_params = GPT(base_3L_config).count_params()['total']
    extra = hippo_params - base_3L_params
    wider_ff = int(Config().d_ff + extra / (3 * 2 * Config().d_model))

    config_3Lw = Config(
        n_layers=3, use_hippo=False,
        wider_baseline=True, wider_d_ff=wider_ff,
        seed=42,
    )
    wider_p = GPT(config_3Lw).count_params()['total']
    print(f"\n  Param check: 4L={GPT(config_4L).count_params()['total']:,}, "
          f"hippo={hippo_params:,}, "
          f"3Lw={wider_p:,}, "
          f"3L={base_3L_params:,}")

    _, hist_3Lw = train_model(config_3Lw, device,
                               "3-layer wider (param match)")

    # ---- 4. Two-layer baseline (lower bound) ----
    config_2L = Config(n_layers=2, use_hippo=False, seed=42)
    _, hist_2L = train_model(config_2L, device, "2-layer (lower bound)")

    # ---- Compare ----
    histories = [hist_4L, hist_hippo, hist_3Lw, hist_2L]
    labels = ["4-layer", "2+hippo+1", "3-layer wider", "2-layer"]

    print_summary(histories, labels)
    plot_comparison(histories, labels, "hippo_v3_results.png")

    # Save
    results = {}
    for label, hist in zip(labels, histories):
        results[label] = hist
    with open("hippo_v3_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
