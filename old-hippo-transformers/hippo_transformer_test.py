"""
Hippocampal Transformer Integration Test
==========================================

Tests whether a Hebbian hippocampal sequence memory component
improves a small decoder-only transformer on sequence prediction.

Architecture:
  - Small GPT (4 layers, d_model=128)
  - Hippocampal component inserted after a chosen layer
  - Down projection (backprop) -> sparse CA3 space -> successor retrieval
    (Hebbian W, no backprop) -> up projection (backprop) -> gated residual

Task:
  Synthetic sequences composed of recurring "motifs" (fixed token subsequences)
  with random filler tokens between them. The hippocampus should help by
  learning transition structure in the projected representation space,
  allowing it to predict continuations of recognized motifs.

Comparison:
  1. Baseline transformer (no hippocampal component)
  2. Hippocampal transformer (same architecture + hippocampal component)
  3. Wider baseline (extra parameters to match hippocampal model's param count)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
from dataclasses import dataclass, asdict


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Transformer
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    vocab_size: int = 64
    context_len: int = 128
    dropout: float = 0.0

    # Hippocampal component
    use_hippo: bool = False
    hippo_layers: tuple = (1,)       # Insert after these layers (0-indexed)
    N_ca3: int = 512
    k_ca3: int = 25
    hebbian_lr: float = 0.005
    hebbian_decay: float = 0.998     # Per-step decay on W_ca3
    ca3_retrieval_steps: int = 1     # Steps of successor traversal

    # Wider baseline (to match param count)
    wider_baseline: bool = False
    wider_d_ff: int = 512            # Will be adjusted to match params

    # Data
    n_motifs: int = 50
    motif_len_min: int = 4
    motif_len_max: int = 8
    filler_max: int = 2              # Max random tokens between motifs

    # Training
    batch_size: int = 32
    n_steps: int = 4000
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 200
    eval_batches: int = 10
    seed: int = 42


# =============================================================================
# Synthetic Data: Motif Sequences
# =============================================================================

class MotifDataset:
    """
    Generates sequences by concatenating random motifs with optional filler.

    Each motif is a fixed sequence of tokens. Training data is produced by
    randomly selecting motifs and concatenating them, with 0-filler_max
    random tokens inserted between motifs.

    Tracks which positions are "in-motif" (predictable from motif structure)
    vs "filler" (random, less predictable).
    """
    def __init__(self, config, split='train'):
        self.config = config
        self.rng = np.random.RandomState(
            config.seed if split == 'train' else config.seed + 999
        )

        # Generate motifs (shared across splits, using fixed seed)
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
        motif_position = np.full((bs, ctx + 1), -1, dtype=np.int64)

        for b in range(bs):
            pos = 0
            while pos < ctx + 1:
                # Possibly insert filler
                if self.config.filler_max > 0:
                    n_filler = self.rng.randint(0, self.config.filler_max + 1)
                    for _ in range(n_filler):
                        if pos >= ctx + 1:
                            break
                        tokens[b, pos] = self.rng.randint(0, self.config.vocab_size)
                        pos += 1

                # Insert a motif
                midx = self.rng.randint(0, len(self.motifs))
                motif = self.motifs[midx]
                for mi, tok in enumerate(motif):
                    if pos >= ctx + 1:
                        break
                    tokens[b, pos] = tok
                    is_motif[b, pos] = True
                    motif_position[b, pos] = mi
                    pos += 1

        x = torch.tensor(tokens[:, :-1], device=device)
        y = torch.tensor(tokens[:, 1:], device=device)
        motif_mask = torch.tensor(is_motif[:, 1:], device=device)
        motif_pos = torch.tensor(motif_position[:, 1:], device=device)

        return x, y, motif_mask, motif_pos


# =============================================================================
# k-WTA with Straight-Through Estimator
# =============================================================================

class KWTAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        topk_vals, topk_idx = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk_idx, topk_vals)
        # Save mask for backward
        mask = (out != 0).float()
        ctx.save_for_backward(mask)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        # Pass gradient only through the k active units
        return grad_output * mask, None


def kwta_ste(x, k):
    return KWTAFunction.apply(x, k)


# =============================================================================
# Hippocampal Component
# =============================================================================

class HippocampalComponent(nn.Module):
    """
    Hippocampal sequence memory for transformer integration.

    Forward pass (differentiable through projections and gate):
      residual -> down_proj -> ReLU -> k-WTA -> W_ca3 matmul -> k-WTA ->
      up_proj -> gate -> add to residual

    Hebbian update (called separately, no gradients):
      Stores transitions between consecutive positions' sparse representations
      into W_ca3 using asymmetric Hebbian rule with mean centering.
    """
    def __init__(self, d_model, N_ca3, k_ca3, hebbian_lr=0.005,
                 hebbian_decay=0.998, retrieval_steps=1):
        super().__init__()
        self.d_model = d_model
        self.N_ca3 = N_ca3
        self.k_ca3 = k_ca3
        self.hebbian_lr = hebbian_lr
        self.hebbian_decay = hebbian_decay
        self.retrieval_steps = retrieval_steps

        # Learned projections
        self.down_proj = nn.Linear(d_model, N_ca3, bias=False)
        self.up_proj = nn.Linear(N_ca3, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # CA3 successor map (Hebbian only)
        self.register_buffer('W_ca3', torch.zeros(N_ca3, N_ca3))
        self.register_buffer('mean_activity', torch.zeros(N_ca3))
        self.register_buffer('n_stored', torch.tensor(0, dtype=torch.long))

        # Diagnostics
        self.register_buffer('_gate_mean', torch.tensor(0.0))
        self.register_buffer('_contrib_norm', torch.tensor(0.0))
        self.register_buffer('_w_norm', torch.tensor(0.0))

        nn.init.xavier_normal_(self.down_proj.weight, gain=0.5)
        nn.init.xavier_normal_(self.up_proj.weight, gain=0.1)
        nn.init.zeros_(self.gate_proj.weight)
        # Initialize gate bias negative so gate starts near-closed
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def _sparsify(self, x):
        """Project and sparsify. x: (..., d_model) -> (..., N_ca3)"""
        h = self.down_proj(x)
        h = F.relu(h)
        h = kwta_ste(h, self.k_ca3)
        norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp(min=1e-10)
        return h / norm

    def _retrieve_successor(self, sparse):
        """Retrieve successor(s) from CA3. sparse: (..., N_ca3)"""
        x = sparse
        W = self.W_ca3  # buffer, no grad
        for _ in range(self.retrieval_steps):
            h = F.relu(x @ W.T)
            if h.sum() < 1e-10:
                break
            h = kwta_ste(h, self.k_ca3)
            norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp(min=1e-10)
            x = h / norm
        return x

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Returns: x + gated hippocampal prediction
        """
        sparse = self._sparsify(x)
        successor = self._retrieve_successor(sparse)
        prediction = self.up_proj(successor)

        g = torch.sigmoid(self.gate_proj(x))
        contribution = g * prediction

        # Track diagnostics
        with torch.no_grad():
            self._gate_mean = g.mean()
            self._contrib_norm = contribution.norm(dim=-1).mean()
            self._w_norm = self.W_ca3.norm()

        return x + contribution

    @torch.no_grad()
    def hebbian_update(self, x):
        """
        Store consecutive-position transitions in W_ca3.
        x: (batch, seq_len, d_model) - the residual stream at this layer.
        Called after backward pass each step.
        """
        # Decay existing associations
        self.W_ca3 *= self.hebbian_decay

        # Project to sparse space (recompute, no grad needed)
        h = F.relu(self.down_proj(x))
        topk_vals, topk_idx = torch.topk(h, self.k_ca3, dim=-1)
        sparse = torch.zeros_like(h)
        sparse.scatter_(-1, topk_idx, topk_vals)
        norm = torch.linalg.norm(sparse, dim=-1, keepdim=True).clamp(min=1e-10)
        sparse = sparse / norm

        B, T, N = sparse.shape

        # Update running mean
        flat = sparse.reshape(-1, N)
        n_new = flat.shape[0]
        batch_mean = flat.mean(dim=0)
        total = self.n_stored.item() + n_new
        if total > 0:
            self.mean_activity = (
                self.mean_activity * self.n_stored + batch_mean * n_new
            ) / total
        self.n_stored.clamp_(max=100000)  # Prevent overflow
        self.n_stored += n_new

        # Mean-center
        centered = sparse - self.mean_activity.unsqueeze(0).unsqueeze(0)

        # Asymmetric Hebbian: store transition t -> t+1
        prev_c = centered[:, :-1, :].reshape(-1, N)  # (B*(T-1), N)
        curr_c = centered[:, 1:, :].reshape(-1, N)   # (B*(T-1), N)

        # Average outer product over batch
        n_pairs = prev_c.shape[0]
        delta_W = (curr_c.T @ prev_c) / n_pairs
        self.W_ca3 += self.hebbian_lr * delta_W
        self.W_ca3.fill_diagonal_(0)

    def get_diagnostics(self):
        return {
            'gate_mean': self._gate_mean.item(),
            'contrib_norm': self._contrib_norm.item(),
            'w_norm': self._w_norm.item(),
        }


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
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
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


class GPT(nn.Module):
    """Small decoder-only transformer with optional hippocampal components."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_len, config.d_model)

        d_ff = config.wider_d_ff if config.wider_baseline else config.d_ff
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Hippocampal components
        self.hippo_components = nn.ModuleDict()
        self._hippo_activations = {}
        if config.use_hippo:
            for layer_idx in config.hippo_layers:
                self.hippo_components[str(layer_idx)] = HippocampalComponent(
                    d_model=config.d_model,
                    N_ca3=config.N_ca3,
                    k_ca3=config.k_ca3,
                    hebbian_lr=config.hebbian_lr,
                    hebbian_decay=config.hebbian_decay,
                    retrieval_steps=config.ca3_retrieval_steps,
                )

        # Causal mask
        mask = torch.tril(torch.ones(config.context_len, config.context_len))
        self.register_buffer('causal_mask', mask.unsqueeze(0).unsqueeze(0))

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

        self._hippo_activations = {}

        for i, block in enumerate(self.blocks):
            h = block(h, self.causal_mask)

            # Apply hippocampal component after this layer if configured
            if str(i) in self.hippo_components:
                # Save pre-hippo activation for Hebbian update
                self._hippo_activations[str(i)] = h.detach().clone()
                h = self.hippo_components[str(i)](h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits

    def hippo_hebbian_update(self):
        """Call after backward pass to update Hebbian weights."""
        for layer_key, activation in self._hippo_activations.items():
            self.hippo_components[layer_key].hebbian_update(activation)

    def get_hippo_diagnostics(self):
        diags = {}
        for key, comp in self.hippo_components.items():
            diags[f'layer_{key}'] = comp.get_diagnostics()
        return diags

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        hippo_proj = 0
        for comp in self.hippo_components.values():
            hippo_proj += sum(p.numel() for p in comp.parameters())
        hippo_hebbian = 0
        for comp in self.hippo_components.values():
            hippo_hebbian += comp.W_ca3.numel()
        return {
            'total': total,
            'trainable': trainable,
            'hippo_projections': hippo_proj,
            'hippo_hebbian': hippo_hebbian,
        }


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
    # Per-position-in-motif loss (position 0 = first token of motif, etc.)
    pos_losses = {}
    pos_counts = {}

    with torch.no_grad():
        for _ in range(config.eval_batches):
            x, y, motif_mask, motif_pos = dataset.get_batch(device=device)
            logits = model(x)
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
                reduction='none'
            ).reshape(y.shape)

            total_loss += loss_per_token.mean().item()

            # Motif vs filler breakdown
            m_mask = motif_mask.float()
            f_mask = 1.0 - m_mask
            if m_mask.sum() > 0:
                motif_loss += (loss_per_token * m_mask).sum().item()
                motif_count += m_mask.sum().item()
            if f_mask.sum() > 0:
                filler_loss += (loss_per_token * f_mask).sum().item()
                filler_count += f_mask.sum().item()

            # Per-position-in-motif
            for pos in range(config.motif_len_max + 1):
                pmask = (motif_pos == pos).float()
                if pmask.sum() > 0:
                    if pos not in pos_losses:
                        pos_losses[pos] = 0.0
                        pos_counts[pos] = 0
                    pos_losses[pos] += (loss_per_token * pmask).sum().item()
                    pos_counts[pos] += pmask.sum().item()

    n = config.eval_batches
    results = {
        'total_loss': total_loss / n,
        'motif_loss': motif_loss / max(motif_count, 1),
        'filler_loss': filler_loss / max(filler_count, 1),
        'per_position': {
            pos: pos_losses[pos] / pos_counts[pos]
            for pos in sorted(pos_losses.keys())
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
    print(f"  Parameters: {params['total']:,} total, "
          f"{params['trainable']:,} trainable")
    if config.use_hippo:
        print(f"  Hippo projections: {params['hippo_projections']:,}, "
              f"Hebbian W: {params['hippo_hebbian']:,}")

    # Only optimize parameters that require grad
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    train_data = MotifDataset(config, split='train')
    eval_data = MotifDataset(config, split='eval')

    history = []
    t0 = time.time()

    for step in range(1, config.n_steps + 1):
        x, y, _, _ = train_data.get_batch(device=device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Hebbian update (after backward, outside computation graph)
        if config.use_hippo:
            model.hippo_hebbian_update()

        if step % config.eval_every == 0 or step == 1:
            eval_results = evaluate(model, eval_data, config, device)
            elapsed = time.time() - t0

            entry = {
                'step': step,
                'train_loss': loss.item(),
                **eval_results,
                'elapsed': elapsed,
            }
            if config.use_hippo:
                entry['hippo'] = model.get_hippo_diagnostics()

            history.append(entry)

            hippo_str = ""
            if config.use_hippo:
                diag = entry['hippo'].get('layer_1', {})
                hippo_str = (f" | gate={diag.get('gate_mean', 0):.3f}"
                             f" contrib={diag.get('contrib_norm', 0):.3f}"
                             f" W={diag.get('w_norm', 0):.1f}")

            print(f"  step {step:5d} | train {loss.item():.4f} | "
                  f"eval {eval_results['total_loss']:.4f} | "
                  f"motif {eval_results['motif_loss']:.4f} | "
                  f"filler {eval_results['filler_loss']:.4f}"
                  f"{hippo_str}"
                  f" | {elapsed:.1f}s")

    return model, history


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(histories, labels, save_path):
    """Plot training curves for all models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle("Hippocampal Transformer Integration Test",
                 fontsize=14, fontweight='bold')

    colors = ['steelblue', 'coral', 'forestgreen', 'purple']

    # 1. Overall eval loss
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        losses = [h['total_loss'] for h in hist]
        ax1.plot(steps, losses, '-', color=colors[i], linewidth=2,
                 label=label, alpha=0.85)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Eval Loss")
    ax1.set_title("Overall Eval Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Motif loss
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        losses = [h['motif_loss'] for h in hist]
        ax2.plot(steps, losses, '-', color=colors[i], linewidth=2,
                 label=label, alpha=0.85)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title("Motif Token Loss (in-pattern)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Filler loss
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        losses = [h['filler_loss'] for h in hist]
        ax3.plot(steps, losses, '-', color=colors[i], linewidth=2,
                 label=label, alpha=0.85)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Loss")
    ax3.set_title("Filler Token Loss (random)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # 4. Per-position-in-motif loss (final eval)
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        pp = hist[-1]['per_position']
        positions = sorted(pp.keys())
        losses = [pp[p] for p in positions]
        ax4.plot(positions, losses, 'o-', color=colors[i], linewidth=2,
                 label=label, markersize=5, alpha=0.85)
    ax4.set_xlabel("Position in Motif")
    ax4.set_ylabel("Loss")
    ax4.set_title("Loss by Position in Motif (final eval)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # 5. Hippo diagnostics: gate mean
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        if 'hippo' in hist[0]:
            steps = [h['step'] for h in hist]
            gate_vals = [h['hippo'].get('layer_1', {}).get('gate_mean', 0)
                         for h in hist]
            ax5.plot(steps, gate_vals, '-', color=colors[i], linewidth=2,
                     label=f"{label} gate", alpha=0.85)
    ax5.set_xlabel("Step")
    ax5.set_ylabel("Gate Value (sigmoid output)")
    ax5.set_title("Hippocampal Gate Activation")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(-0.05, 1.05)

    # 6. Hippo W norm
    ax6 = fig.add_subplot(gs[1, 2])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        if 'hippo' in hist[0]:
            steps = [h['step'] for h in hist]
            w_norms = [h['hippo'].get('layer_1', {}).get('w_norm', 0)
                       for h in hist]
            ax6.plot(steps, w_norms, '-', color=colors[i], linewidth=2,
                     label=f"{label} ||W||", alpha=0.85)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("||W_ca3||")
    ax6.set_title("CA3 Weight Matrix Norm")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to {save_path}")


def print_summary(histories, labels):
    print(f"\n{'='*70}")
    print("  FINAL RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Eval Loss':>10} {'Motif Loss':>12} "
          f"{'Filler Loss':>12}")
    print(f"  {'-'*60}")
    for hist, label in zip(histories, labels):
        final = hist[-1]
        print(f"  {label:<25} {final['total_loss']:>10.4f} "
              f"{final['motif_loss']:>12.4f} {final['filler_loss']:>12.4f}")

    # Compute improvement
    if len(histories) >= 2:
        base = histories[0][-1]
        hippo = histories[1][-1]
        total_pct = (base['total_loss'] - hippo['total_loss']) / base['total_loss'] * 100
        motif_pct = (base['motif_loss'] - hippo['motif_loss']) / base['motif_loss'] * 100
        print(f"\n  Hippo vs Baseline improvement:")
        print(f"    Overall: {total_pct:+.2f}%")
        print(f"    Motif:   {motif_pct:+.2f}%")

    # Per-position comparison
    print(f"\n  Per-position-in-motif loss (final eval):")
    print(f"  {'Pos':>4}", end="")
    for label in labels:
        print(f"  {label:>15}", end="")
    print()
    all_positions = sorted(histories[0][-1]['per_position'].keys())
    for pos in all_positions:
        print(f"  {pos:>4}", end="")
        for hist in histories:
            val = hist[-1]['per_position'].get(pos, float('nan'))
            print(f"  {val:>15.4f}", end="")
        print()


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

    # ---- Model 1: Baseline ----
    base_config = Config(
        use_hippo=False,
        seed=42,
    )

    _, base_history = train_model(base_config, device, label="Baseline")

    # ---- Model 2: Hippocampal ----
    hippo_config = Config(
        use_hippo=True,
        hippo_layers=(1,),
        N_ca3=512,
        k_ca3=25,
        hebbian_lr=0.005,
        hebbian_decay=0.998,
        ca3_retrieval_steps=1,
        seed=42,
    )

    hippo_model, hippo_history = train_model(
        hippo_config, device, label="Hippocampal"
    )

    # ---- Model 3: Wider baseline (match param count) ----
    # Calculate how much wider MLP needs to be
    hippo_params = GPT(hippo_config).count_params()['total']
    base_params = GPT(base_config).count_params()['total']
    param_diff = hippo_params - base_params

    # Each layer MLP has 2 * d_model * d_ff params
    # 4 layers -> need param_diff / (4 * 2 * d_model) extra d_ff
    extra_ff_per_layer = param_diff / (base_config.n_layers * 2 * base_config.d_model)
    wider_ff = int(base_config.d_ff + extra_ff_per_layer)

    wider_config = Config(
        use_hippo=False,
        wider_baseline=True,
        wider_d_ff=wider_ff,
        seed=42,
    )
    wider_params = GPT(wider_config).count_params()['total']
    print(f"\n  Param matching: base={base_params:,}, hippo={hippo_params:,}, "
          f"wider={wider_params:,}")

    _, wider_history = train_model(wider_config, device, label="Wider Baseline")

    # ---- Compare ----
    histories = [base_history, hippo_history, wider_history]
    labels = ["Baseline", "Hippocampal", "Wider Baseline"]

    print_summary(histories, labels)
    plot_comparison(histories, labels,
                    save_path="hippo_transformer_results.png")

    # Save raw results
    results_data = {}
    for label, hist in zip(labels, histories):
        serializable = []
        for entry in hist:
            s = {}
            for k, v in entry.items():
                if isinstance(v, dict):
                    s[k] = {str(kk): vv for kk, vv in v.items()}
                else:
                    s[k] = v
            serializable.append(s)
        results_data[label] = serializable

    with open("hippo_transformer_results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("  ALL DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
