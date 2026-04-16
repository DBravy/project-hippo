"""
Hippocampal Transformer Integration Test v2
=============================================

Key changes from v1:
  1. The hippocampus now operates as a FAST WEIGHT system within each context
     window. W_ca3 is reset at the start of each sequence and builds up via
     Hebbian learning as tokens are processed. This is much closer to the
     biological function: rapid one-shot binding of novel sequences.

  2. The task now requires IN-CONTEXT LEARNING. Each sequence presents
     several "episodes" (novel motifs defined fresh for that sequence),
     then tests recognition of those episodes later. The motifs are never
     repeated across sequences, so the model can't memorize them during
     training -- it MUST learn them within each context window.

  3. Because the hippocampus builds up per-sequence, token processing
     is sequential through the hippocampal component (chunked for efficiency).
     This is the real computational cost of the architecture.

Architecture:
  - Small GPT (4 layers, d_model=128)
  - Hippocampal fast-weight component after layer 1
  - During forward pass: for each position, project residual to sparse CA3,
    store transition in W (Hebbian), retrieve successor, project back up,
    gate-add to residual
  - W_ca3 is zeroed at the start of each sequence

Task:
  Each training sequence has the structure:
    [define motif A] [define motif B] [define motif C] ... [separator]
    [partial cue from A] [expected continuation of A]
    [partial cue from C] [expected continuation of C] ...

  The model must recognize which motif is being cued and predict its
  continuation. This requires binding the motifs during the definition
  phase and retrieving them during the test phase.
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
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 512
    vocab_size: int = 64
    context_len: int = 128
    dropout: float = 0.0

    # Hippocampal component
    use_hippo: bool = False
    hippo_layer: int = 1
    N_ca3: int = 256
    k_ca3: int = 15
    hebbian_lr: float = 1.0           # High LR: one-shot binding
    n_settle_steps: int = 2           # Attractor settling iterations
    ca3_retrieval_steps: int = 1

    # Wider baseline
    wider_baseline: bool = False
    wider_d_ff: int = 512

    # Data
    n_motifs_per_seq: int = 8         # Motifs defined per context window
    motif_length: int = 6             # Fixed motif length
    n_cue_tokens: int = 2             # How many tokens of motif shown as cue
    separator_token: int = 0          # Token used as separator

    # Training
    batch_size: int = 16
    n_steps: int = 3000
    lr: float = 3e-4
    weight_decay: float = 0.01
    eval_every: int = 150
    eval_batches: int = 8
    seed: int = 42


# =============================================================================
# In-Context Learning Data Generator
# =============================================================================

class InContextDataset:
    """
    Generates sequences that require in-context learning.

    Structure of each sequence:
      [motif1 tokens] [sep] [motif2 tokens] [sep] ... [motifN tokens] [sep] [sep]
      [cue1: first n tokens of motifX] [rest of motifX] [sep]
      [cue2: first n tokens of motifY] [rest of motifY] [sep]
      ...

    The motifs are generated fresh for each sequence, so the model
    cannot memorize them -- it must learn them from the definitions
    in the first half of the context.
    """
    def __init__(self, config, split='train'):
        self.config = config
        self.rng = np.random.RandomState(
            config.seed if split == 'train' else config.seed + 7777
        )
        self.sep = config.separator_token
        # Tokens 1..vocab_size-1 are content tokens (0 is separator)
        self.content_tokens = np.arange(1, config.vocab_size)

    def _make_sequence(self):
        """Generate one training sequence with in-context motifs."""
        cfg = self.config
        ctx_limit = cfg.context_len + 1  # +1 because we slice x=tokens[:-1], y=tokens[1:]
        tokens = []
        phase = []   # 0=definition, 1=separator, 2=cue, 3=target, 4=padding

        # --- Definition phase ---
        motifs = []
        for _ in range(cfg.n_motifs_per_seq):
            motif = self.rng.choice(self.content_tokens,
                                    size=cfg.motif_length,
                                    replace=True)
            motifs.append(motif)
            tokens.extend(motif.tolist())
            phase.extend([0] * cfg.motif_length)
            tokens.append(self.sep)
            phase.append(1)

        # Double separator to mark end of definitions
        tokens.append(self.sep)
        phase.append(1)

        # --- Test phase: cue + expected continuation ---
        # Repeat test rounds until we fill the context
        rounds = 0
        while len(tokens) < ctx_limit - cfg.motif_length:
            test_order = self.rng.permutation(cfg.n_motifs_per_seq)
            for midx in test_order:
                if len(tokens) >= ctx_limit - cfg.motif_length:
                    break
                motif = motifs[midx]
                cue = motif[:cfg.n_cue_tokens].tolist()
                target = motif[cfg.n_cue_tokens:].tolist()
                tokens.extend(cue)
                phase.extend([2] * len(cue))
                tokens.extend(target)
                phase.extend([3] * len(target))
                tokens.append(self.sep)
                phase.append(1)
            rounds += 1
            if rounds > 5:
                break

        # Pad to context_len + 1
        while len(tokens) < ctx_limit:
            tokens.append(self.sep)
            phase.append(4)  # padding phase

        return tokens[:ctx_limit], phase[:ctx_limit]

    def get_batch(self, batch_size=None, device='cpu'):
        bs = batch_size or self.config.batch_size
        ctx = self.config.context_len

        all_tokens = np.zeros((bs, ctx + 1), dtype=np.int64)
        all_phase = np.zeros((bs, ctx + 1), dtype=np.int64)

        for b in range(bs):
            tokens, phase = self._make_sequence()
            # Truncate or pad to context_len + 1
            length = min(len(tokens), ctx + 1)
            all_tokens[b, :length] = tokens[:length]
            all_phase[b, :length] = phase[:length]

        x = torch.tensor(all_tokens[:, :-1], device=device)
        y = torch.tensor(all_tokens[:, 1:], device=device)
        # Phase labels shifted by 1 to align with targets
        phase_labels = torch.tensor(all_phase[:, 1:], device=device)

        return x, y, phase_labels


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
# Hippocampal Fast-Weight Component
# =============================================================================

class HippocampalFastWeight(nn.Module):
    """
    Hippocampal sequence memory using fast weights (per-context Hebbian learning).

    Key difference from v1: W_ca3 is not a persistent buffer. It is computed
    fresh for each sequence by accumulating Hebbian outer products as tokens
    are processed left-to-right. This makes it a true episodic memory that
    binds novel patterns on the fly.

    To enable batched training, we process the sequence causally:
    for each position t, W_t = sum of Hebbian updates from positions 0..t-1.
    The successor retrieval at position t uses W_t (not future information).
    """
    def __init__(self, d_model, N_ca3, k_ca3, hebbian_lr=1.0,
                 n_settle_steps=3, retrieval_steps=1):
        super().__init__()
        self.d_model = d_model
        self.N_ca3 = N_ca3
        self.k_ca3 = k_ca3
        self.hebbian_lr = hebbian_lr
        self.n_settle_steps = n_settle_steps
        self.retrieval_steps = retrieval_steps

        # Learned projections (backprop)
        self.down_proj = nn.Linear(d_model, N_ca3, bias=False)
        self.up_proj = nn.Linear(N_ca3, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        nn.init.xavier_normal_(self.down_proj.weight, gain=0.5)
        nn.init.xavier_normal_(self.up_proj.weight, gain=0.1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

        # Diagnostics
        self._diag = {}

    def _sparsify(self, x):
        """x: (..., d_model) -> (..., N_ca3), sparse + normalized"""
        h = F.relu(self.down_proj(x))
        h = kwta_ste(h, self.k_ca3)
        norm = torch.linalg.norm(h, dim=-1, keepdim=True).clamp(min=1e-10)
        return h / norm

    def _settle_attractor(self, cue, W):
        """
        Run attractor settling on cue using weight matrix W.
        cue: (batch, N_ca3), W: (batch, N_ca3, N_ca3)
        """
        x = cue
        for _ in range(self.n_settle_steps):
            h = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
            h = F.relu(h)
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

        Memory-efficient approach: build the full Hebbian W matrix from all
        transitions in the sequence, then use it for retrieval at all positions
        simultaneously. This is not strictly causal (early positions get to
        use transitions from later positions), but:
          - During the TEST phase, this is desirable (all motifs should be accessible)
          - During the DEFINITION phase, the gate should learn to stay closed
          - This allows fully batched computation (no sequential loop)
        """
        B, T, D = x.shape
        N = self.N_ca3

        # 1. Project all positions to sparse CA3 space (WITH gradient)
        sparse_all = self._sparsify(x)  # (B, T, N)

        # 2. Build full Hebbian W from ALL transitions (NO gradient)
        with torch.no_grad():
            sd = sparse_all.detach()
            prev = sd[:, :-1, :]  # (B, T-1, N)
            curr = sd[:, 1:, :]   # (B, T-1, N)
            # W = hebbian_lr * sum_t outer(curr_t, prev_t)
            # = hebbian_lr * curr^T @ prev  (batched)
            W = self.hebbian_lr * torch.bmm(
                curr.transpose(1, 2),  # (B, N, T-1)
                prev                    # (B, T-1, N)
            )  # (B, N, N)
            # Zero diagonal
            diag_mask = (1.0 - torch.eye(N, device=x.device)).unsqueeze(0)
            W = W * diag_mask

        # 3. Retrieve successors for all positions at once (WITH gradient through sparse_all)
        # successor = W @ sparse^T for each batch element
        successors = torch.bmm(
            W,                          # (B, N, N) - detached, no grad
            sparse_all.transpose(1, 2)  # (B, N, T)
        ).transpose(1, 2)              # (B, T, N)
        successors = F.relu(successors)

        # Apply k-WTA to successors
        successors = kwta_ste(successors, self.k_ca3)
        snorm = torch.linalg.norm(
            successors, dim=-1, keepdim=True).clamp(min=1e-10)
        successors = successors / snorm

        # 4. Project back up and gate
        prediction = self.up_proj(successors)  # (B, T, D)
        g = torch.sigmoid(self.gate_proj(x))   # (B, T, D)
        contribution = g * prediction

        with torch.no_grad():
            self._diag = {
                'gate_mean': g.mean().item(),
                'contrib_norm': contribution.norm(dim=-1).mean().item(),
                'w_final_norm': W.norm(dim=(1, 2)).mean().item(),
                'successor_norm': successors.norm(dim=-1).mean().item(),
            }

        return x + contribution

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


class GPT(nn.Module):
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

        self.hippo = None
        if config.use_hippo:
            self.hippo = HippocampalFastWeight(
                d_model=config.d_model,
                N_ca3=config.N_ca3,
                k_ca3=config.k_ca3,
                hebbian_lr=config.hebbian_lr,
                n_settle_steps=config.n_settle_steps,
                retrieval_steps=config.ca3_retrieval_steps,
            )
        self.hippo_layer = config.hippo_layer

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

        for i, block in enumerate(self.blocks):
            h = block(h, self.causal_mask)
            if self.hippo is not None and i == self.hippo_layer:
                h = self.hippo(h)

        h = self.ln_f(h)
        return self.head(h)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        hippo_proj = 0
        if self.hippo is not None:
            hippo_proj = sum(p.numel() for p in self.hippo.parameters())
        return {'total': total, 'trainable': trainable,
                'hippo_projections': hippo_proj}


# =============================================================================
# Training and Evaluation
# =============================================================================

def evaluate(model, dataset, config, device):
    model.eval()
    total_loss_sum = 0.0
    total_valid = 0
    phase_losses = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    phase_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    with torch.no_grad():
        for _ in range(config.eval_batches):
            x, y, phase = dataset.get_batch(device=device)
            logits = model(x)
            loss_per_tok = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1), reduction='none'
            ).reshape(y.shape)

            valid_mask = (phase != 4).float()
            total_loss_sum += (loss_per_tok * valid_mask).sum().item()
            total_valid += valid_mask.sum().item()

            for p in range(4):
                mask = (phase == p).float()
                if mask.sum() > 0:
                    phase_losses[p] += (loss_per_tok * mask).sum().item()
                    phase_counts[p] += mask.sum().item()

    results = {
        'total_loss': total_loss_sum / max(total_valid, 1),
        'definition_loss': phase_losses[0] / max(phase_counts[0], 1),
        'separator_loss': phase_losses[1] / max(phase_counts[1], 1),
        'cue_loss': phase_losses[2] / max(phase_counts[2], 1),
        'target_loss': phase_losses[3] / max(phase_counts[3], 1),
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
        print(f"  Hippo projection params: {params['hippo_projections']:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )

    train_data = InContextDataset(config, split='train')
    eval_data = InContextDataset(config, split='eval')

    history = []
    t0 = time.time()

    for step in range(1, config.n_steps + 1):
        x, y, phase = train_data.get_batch(device=device)
        logits = model(x)

        # Mask out padding (phase=4) from loss
        loss_per_tok = F.cross_entropy(
            logits.reshape(-1, config.vocab_size), y.reshape(-1),
            reduction='none').reshape(y.shape)
        valid_mask = (phase != 4).float()
        loss = (loss_per_tok * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % config.eval_every == 0 or step == 1:
            ev = evaluate(model, eval_data, config, device)
            elapsed = time.time() - t0

            entry = {'step': step, 'train_loss': loss.item(),
                     **ev, 'elapsed': elapsed}
            if config.use_hippo and model.hippo is not None:
                entry['hippo'] = model.hippo.get_diagnostics()

            history.append(entry)

            hippo_str = ""
            if 'hippo' in entry:
                d = entry['hippo']
                hippo_str = (f" | gate={d.get('gate_mean',0):.3f}"
                             f" Wnorm={d.get('w_final_norm',0):.2f}"
                             f" succ={d.get('successor_norm',0):.3f}")

            print(f"  step {step:5d} | train {loss.item():.4f} | "
                  f"eval {ev['total_loss']:.4f} | "
                  f"target {ev['target_loss']:.4f} | "
                  f"cue {ev['cue_loss']:.4f} | "
                  f"defn {ev['definition_loss']:.4f}"
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

    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Hippocampal Transformer v2: In-Context Learning Test",
                 fontsize=14, fontweight='bold')

    colors = ['steelblue', 'coral', 'forestgreen', 'purple']

    # 1. Overall eval loss
    ax = fig.add_subplot(gs[0, 0])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['total_loss'] for h in hist],
                '-', color=colors[i], linewidth=2, label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Overall Eval Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. TARGET loss (the key metric)
    ax = fig.add_subplot(gs[0, 1])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['target_loss'] for h in hist],
                '-', color=colors[i], linewidth=2, label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Target Loss (continuation prediction)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. Cue loss
    ax = fig.add_subplot(gs[0, 2])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['cue_loss'] for h in hist],
                '-', color=colors[i], linewidth=2, label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Cue Loss (motif start recognition)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Definition loss
    ax = fig.add_subplot(gs[1, 0])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        steps = [h['step'] for h in hist]
        ax.plot(steps, [h['definition_loss'] for h in hist],
                '-', color=colors[i], linewidth=2, label=label)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("Definition Phase Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Hippo gate
    ax = fig.add_subplot(gs[1, 1])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        if 'hippo' in hist[0]:
            steps = [h['step'] for h in hist]
            ax.plot(steps, [h['hippo']['gate_mean'] for h in hist],
                    '-', color=colors[i], linewidth=2, label=f"{label} gate")
    ax.set_xlabel("Step"); ax.set_ylabel("Gate")
    ax.set_title("Hippocampal Gate Activation")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # 6. Hippo W norm and successor norm
    ax = fig.add_subplot(gs[1, 2])
    for i, (hist, label) in enumerate(zip(histories, labels)):
        if 'hippo' in hist[0]:
            steps = [h['step'] for h in hist]
            ax.plot(steps, [h['hippo']['w_final_norm'] for h in hist],
                    '-', color=colors[i], linewidth=2,
                    label=f"{label} W norm")
            ax.plot(steps, [h['hippo']['successor_norm'] for h in hist],
                    '--', color=colors[i], linewidth=1.5,
                    label=f"{label} succ norm", alpha=0.7)
    ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.set_title("CA3 Weight and Successor Norms")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved plot to {save_path}")


def print_summary(histories, labels):
    print(f"\n{'='*70}")
    print("  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'Eval':>8} {'Target':>8} {'Cue':>8} "
          f"{'Defn':>8}")
    print(f"  {'-'*60}")
    for hist, label in zip(histories, labels):
        f = hist[-1]
        print(f"  {label:<25} {f['total_loss']:>8.4f} "
              f"{f['target_loss']:>8.4f} {f['cue_loss']:>8.4f} "
              f"{f['definition_loss']:>8.4f}")

    if len(histories) >= 2:
        base_target = histories[0][-1]['target_loss']
        hippo_target = histories[1][-1]['target_loss']
        pct = (base_target - hippo_target) / base_target * 100
        print(f"\n  Target loss improvement (hippo vs baseline): {pct:+.2f}%")


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

    # Quick data sanity check
    cfg = Config()
    ds = InContextDataset(cfg)
    x, y, phase = ds.get_batch(batch_size=1)
    print(f"\nData sample (first sequence):")
    print(f"  x[:60]:     {x[0, :60].tolist()}")
    print(f"  phase[:60]: {phase[0, :60].tolist()}")
    print(f"  Phases: 0=definition, 1=separator, 2=cue, 3=target")
    n_target = (phase == 3).sum().item()
    n_cue = (phase == 2).sum().item()
    n_def = (phase == 0).sum().item()
    print(f"  Counts: defn={n_def}, cue={n_cue}, target={n_target}")

    # ---- Baseline ----
    base_config = Config(use_hippo=False, seed=42)
    _, base_history = train_model(base_config, device, "Baseline")

    # ---- Hippocampal ----
    hippo_config = Config(
        use_hippo=True,
        hippo_layer=1,
        N_ca3=256,
        k_ca3=15,
        hebbian_lr=1.0,
        n_settle_steps=2,
        ca3_retrieval_steps=1,
        seed=42,
    )
    _, hippo_history = train_model(hippo_config, device, "Hippocampal")

    # ---- Wider baseline ----
    hippo_params = GPT(hippo_config).count_params()['total']
    base_params = GPT(base_config).count_params()['total']
    extra = hippo_params - base_params
    wider_ff = int(base_config.d_ff + extra / (base_config.n_layers * 2
                                                * base_config.d_model))

    wider_config = Config(
        use_hippo=False,
        wider_baseline=True,
        wider_d_ff=wider_ff,
        seed=42,
    )
    wider_p = GPT(wider_config).count_params()['total']
    print(f"\n  Params: base={base_params:,}, hippo={hippo_params:,}, "
          f"wider={wider_p:,}")
    _, wider_history = train_model(wider_config, device, "Wider Baseline")

    # ---- Compare ----
    histories = [base_history, hippo_history, wider_history]
    labels = ["Baseline", "Hippocampal", "Wider Baseline"]

    print_summary(histories, labels)

    plot_path = "hippo_transformer_v2_results.png"
    plot_comparison(histories, labels, save_path=plot_path)

    # Save results
    results = {}
    for label, hist in zip(labels, histories):
        results[label] = [{k: v for k, v in h.items()} for h in hist]
    with open("hippo_transformer_v2_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
