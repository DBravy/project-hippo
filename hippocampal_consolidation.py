"""
Hippocampal Consolidation via LoRA Training
=============================================

The hippocampus as a LoRA training system for cortical weights.

During replay, the hippocampal system provides stored residual trajectories
as training targets for low-rank adapter (LoRA) parameters at each layer of
a frozen transformer. After consolidation, the hippocampus is removed and
the LoRA-modified model is tested on fact-test pairs. If the model now
predicts "vinegar" without any injection, consolidation worked.

Architecture:
  1. Frozen distilgpt2 + LoRA adapters (rank-r matrices at each layer)
  2. Hippocampal system (trained on base corpus + facts, as before)
  3. Consolidation loop:
     - Hippocampus replays fact token sequences
     - Per-layer target residuals reconstructed via direct_decoder + B matrices
     - LoRA params trained with cross-entropy (primary) + MSE on residuals (aux)
  4. Evaluation: remove hippocampus, test LoRA-modified model on fact prompts

Depends on: hippocampal_transformer_backprojection.py, fact_learning_paradigm.py
"""

import sys
import os
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hippocampal_transformer_backprojection import (
    cosine_sim,
    CorticalProjection,
    CorticalBackprojection,
    HippocampalSystemTemporal,
    encode_phase_a,
    encode_phase_b,
    DEFAULT_TEXT,
)
from fact_learning_paradigm import (
    FACT_TEST_PAIRS,
    load_model,
    get_hidden_states_and_logits,
    extract_layer_residuals,
    get_token_probs,
    build_hippocampal_system,
    encode_facts,
)


# =============================================================================
# LoRA ADAPTER
# =============================================================================

class LoRAAdapter(nn.Module):
    """
    Low-rank adapter: output = x + scale * (B @ A @ x)

    A projects down from d_model to rank, B projects back up.
    Initialized so A is small random, B is zero (LoRA convention),
    meaning the adapter starts as identity.
    """
    def __init__(self, d_model, rank=4, scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.scale = scale

        # A: d_model -> rank (small random init)
        self.A = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        # B: rank -> d_model (zero init, so adapter starts as no-op)
        self.B = nn.Parameter(torch.zeros(d_model, rank))

    def forward(self, x):
        # x: (..., d_model)
        # Low-rank residual: B @ A @ x
        delta = F.linear(F.linear(x, self.A), self.B)
        return x + self.scale * delta


class LoRAInjectedModel(nn.Module):
    """
    Wraps a frozen distilgpt2 with LoRA adapters at each transformer block.

    The adapters are applied to the output of each transformer block
    (post-attention + FFN residual), modifying the residual stream.
    This is where cortical consolidation happens: the LoRA weights
    are the "cortical weight modifications" driven by hippocampal replay.
    """
    def __init__(self, base_model, n_layers=6, rank=4, lora_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.n_layers = n_layers

        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Create LoRA adapters (one per transformer block)
        d_model = base_model.config.n_embd  # 768 for distilgpt2
        self.lora_adapters = nn.ModuleList([
            LoRAAdapter(d_model, rank=rank, scale=lora_scale)
            for _ in range(n_layers)
        ])

        # Storage for per-layer hidden states (populated during forward)
        self._layer_outputs = [None] * n_layers
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks on each transformer block."""
        self._remove_hooks()

        for l in range(self.n_layers):
            adapter = self.lora_adapters[l]
            layer_idx = l  # capture in closure

            def make_hook(adapter_l, idx):
                def hook_fn(module, input, output):
                    hidden = output[0]  # (batch, seq_len, d_model)
                    # Apply LoRA adapter to the block output
                    modified = adapter_l(hidden)
                    # Store for MSE loss computation
                    self._layer_outputs[idx] = modified
                    return (modified,) + output[1:]
                return hook_fn

            h = self.base_model.transformer.h[l].register_forward_hook(
                make_hook(adapter, layer_idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, input_ids, return_layer_outputs=False):
        """
        Forward pass with LoRA-modified hidden states.

        Returns:
            logits: (batch, seq_len, vocab_size)
            layer_outputs: optional list of per-layer hidden states
        """
        self._register_hooks()

        outputs = self.base_model(input_ids, output_hidden_states=True)

        self._remove_hooks()

        if return_layer_outputs:
            return outputs.logits, [lo.clone() if lo is not None else None
                                    for lo in self._layer_outputs]
        return outputs.logits

    def get_trainable_params(self):
        """Return only LoRA parameters for optimizer."""
        params = []
        for adapter in self.lora_adapters:
            params.extend(adapter.parameters())
        return params

    def total_trainable_params(self):
        return sum(p.numel() for p in self.get_trainable_params())

    def lora_weight_norms(self):
        """Diagnostic: Frobenius norms of LoRA A and B at each layer."""
        norms = []
        for l, adapter in enumerate(self.lora_adapters):
            a_norm = float(torch.linalg.norm(adapter.A))
            b_norm = float(torch.linalg.norm(adapter.B))
            norms.append((a_norm, b_norm))
        return norms


# =============================================================================
# BIAS-ONLY ADAPTER (even simpler than LoRA)
# =============================================================================

class BiasAdapter(nn.Module):
    """
    Additive bias term at each layer: output = x + bias.
    The simplest possible cortical modification. If even this works,
    it shows the consolidation mechanism is robust.
    """
    def __init__(self, d_model):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return x + self.bias


class BiasInjectedModel(nn.Module):
    """Wraps frozen distilgpt2 with per-layer bias adapters."""
    def __init__(self, base_model, n_layers=6):
        super().__init__()
        self.base_model = base_model
        self.n_layers = n_layers

        for param in self.base_model.parameters():
            param.requires_grad = False

        d_model = base_model.config.n_embd
        self.bias_adapters = nn.ModuleList([
            BiasAdapter(d_model) for _ in range(n_layers)
        ])
        self._layer_outputs = [None] * n_layers
        self._hooks = []

    def _register_hooks(self):
        self._remove_hooks()
        for l in range(self.n_layers):
            adapter = self.bias_adapters[l]
            layer_idx = l

            def make_hook(adapter_l, idx):
                def hook_fn(module, input, output):
                    hidden = output[0]
                    modified = adapter_l(hidden)
                    self._layer_outputs[idx] = modified
                    return (modified,) + output[1:]
                return hook_fn

            h = self.base_model.transformer.h[l].register_forward_hook(
                make_hook(adapter, layer_idx))
            self._hooks.append(h)

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, input_ids, return_layer_outputs=False):
        self._register_hooks()
        outputs = self.base_model(input_ids, output_hidden_states=True)
        self._remove_hooks()
        if return_layer_outputs:
            return outputs.logits, [lo.clone() if lo is not None else None
                                    for lo in self._layer_outputs]
        return outputs.logits

    def get_trainable_params(self):
        return list(self.bias_adapters.parameters())

    def total_trainable_params(self):
        return sum(p.numel() for p in self.get_trainable_params())

    def lora_weight_norms(self):
        return [(0.0, float(torch.linalg.norm(a.bias)))
                for a in self.bias_adapters]


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def cosine_loss(pred, target):
    """
    1 - cosine_similarity. Drives pred toward target's direction.
    More natural than MSE for high-dimensional residual matching.
    """
    pred_n = pred / (torch.linalg.norm(pred, dim=-1, keepdim=True) + 1e-10)
    targ_n = target / (torch.linalg.norm(target, dim=-1, keepdim=True) + 1e-10)
    return 1.0 - (pred_n * targ_n).sum(dim=-1).mean()


def compute_residual_loss(layer_outputs, hippo_targets, token_ids,
                          loss_type='cosine'):
    """
    Compute residual-matching loss between model hidden states and
    hippocampal target residuals.

    loss_type: 'cosine' (1 - cos_sim), 'mse' (normalized MSE), or
               'mse_raw' (unnormalized MSE on raw residuals)
    """
    device = layer_outputs[0].device if layer_outputs[0] is not None else 'cpu'
    total = torch.tensor(0.0, device=device, requires_grad=True)
    n_terms = 0

    for l in range(6):
        if layer_outputs[l] is not None:
            model_residual = layer_outputs[l][0]  # (seq_len, d_model)
            for t in range(len(token_ids)):
                target = hippo_targets[t][l].detach()
                pred = model_residual[t]

                if loss_type == 'cosine':
                    total = total + cosine_loss(
                        pred.unsqueeze(0), target.unsqueeze(0))
                elif loss_type == 'mse':
                    pred_n = pred / (torch.linalg.norm(pred) + 1e-10)
                    targ_n = target / (torch.linalg.norm(target) + 1e-10)
                    total = total + F.mse_loss(pred_n, targ_n)
                elif loss_type == 'mse_raw':
                    total = total + F.mse_loss(pred, target)
                n_terms += 1

    if n_terms > 0:
        total = total / n_terms
    return total, n_terms


# =============================================================================
# HIPPOCAMPAL REPLAY TARGET COMPUTATION
# =============================================================================

def compute_replay_targets(system, model_base, tokenizer, fact_pairs,
                           device='cpu', use_successor=False):
    """
    Pre-compute hippocampal replay targets for each fact.

    Two modes:
      - Direct replay (use_successor=False): For each token in the fact,
        encode through hippocampal circuit and reconstruct per-layer targets.
        This simulates the hippocampus replaying what it experienced.

      - Successor replay (use_successor=True): Cue with the first token
        and let the CA3 successor map generate the full trajectory.
        This simulates autonomous replay during sleep (no external input).

    Returns:
        dict mapping fact_id -> {token_ids, targets, ec_targets}
    """
    hippo = system['hippo']
    cortical_proj = system['cortical_proj']
    backproj = system['backproj']

    replay_targets = {}
    for pair in fact_pairs:
        fid = pair['id']
        fact_text = pair['fact']
        token_ids = tokenizer.encode(fact_text)

        with torch.no_grad():
            input_ids = torch.tensor([token_ids], device=device)
            outputs = model_base(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        per_token_targets = []
        per_token_ec = []

        if use_successor and len(token_ids) > 1:
            # Autonomous replay: cue with first token, unroll successor map
            first_residuals = []
            for l in range(1, 7):
                h = hidden_states[l][0, 0, :].clone().to(torch.float32).to(device)
                first_residuals.append(h)
            first_ec = cortical_proj.project(first_residuals)

            ec_traj, _, _ = hippo.retrieve_sequence_ec_deep(
                first_ec, n_steps=len(token_ids),
                adapt_rate=0.0, adapt_decay=0.0)

            for t in range(len(token_ids)):
                if t < len(ec_traj):
                    decoder_out = ec_traj[t]
                else:
                    decoder_out = ec_traj[-1]

                target_residuals = backproj.retrieve(decoder_out)
                per_token_targets.append(target_residuals)
                per_token_ec.append(decoder_out.clone())
        else:
            # Direct replay: encode each token through hippocampal circuit
            for t in range(len(token_ids)):
                layer_residuals = []
                for l in range(1, 7):
                    h = hidden_states[l][0, t, :].clone().to(torch.float32).to(device)
                    layer_residuals.append(h)

                ec_input = cortical_proj.project(layer_residuals)
                stellate, _ = hippo.ec_sup.forward(ec_input)
                dg_out = hippo.dg.forward(stellate)
                ca3_out = hippo.ca3.retrieve(dg_out, hippo.ca3_retrieval_iterations)
                decoder_out = hippo.direct_decoder.retrieve(ca3_out)

                target_residuals = backproj.retrieve(decoder_out)
                per_token_targets.append(target_residuals)
                per_token_ec.append(decoder_out.clone())

        replay_targets[fid] = {
            'token_ids': token_ids,
            'targets': per_token_targets,
            'ec_targets': per_token_ec,
        }

    return replay_targets


def compute_random_targets(replay_targets, backproj, device='cpu'):
    """
    Generate random replay targets (control condition).
    Same structure as hippocampal targets but with random EC vectors.
    This tests whether hippocampal-specific patterns matter
    or whether any gradient signal would work.
    """
    random_targets = {}
    for fid, data in replay_targets.items():
        per_token_targets = []
        for t in range(len(data['token_ids'])):
            random_ec = torch.randn(backproj.d_ec, device=device)
            random_ec = random_ec / (torch.linalg.norm(random_ec) + 1e-10)
            target_residuals = backproj.retrieve(random_ec)
            per_token_targets.append(target_residuals)

        random_targets[fid] = {
            'token_ids': data['token_ids'],
            'targets': per_token_targets,
        }
    return random_targets


# =============================================================================
# BASE CORPUS REPLAY (for interleaved consolidation)
# =============================================================================

def prepare_base_corpus_samples(model_base, tokenizer, text, n_samples=20,
                                seq_length=32, device='cpu'):
    """
    Prepare base corpus samples for interleaved replay.
    During biological sleep, the hippocampus replays both new memories
    and older ones. Interleaving prevents catastrophic forgetting.
    """
    tokens = tokenizer.encode(text)
    samples = []
    for i in range(min(n_samples, len(tokens) // seq_length)):
        start = i * seq_length
        end = start + seq_length
        chunk_ids = tokens[start:end]
        if len(chunk_ids) >= 2:
            samples.append(chunk_ids)
    return samples


# =============================================================================
# CONSOLIDATION: HIPPOCAMPAL REPLAY -> LoRA TRAINING
# =============================================================================

def consolidation_replay(lora_model, system, model_base, tokenizer,
                         fact_pairs, stored_facts,
                         n_epochs=50, lr=1e-3, mse_weight=0.1,
                         device='cpu', verbose=True):
    """
    Consolidation loop: hippocampal replay drives LoRA weight updates.

    For each replay episode:
      1. Select a stored fact
      2. Retrieve its hippocampal trajectory (per-layer target residuals)
      3. Forward the fact text through the LoRA model
      4. Compute loss:
         - Cross-entropy: model should predict the fact's tokens
         - MSE (auxiliary): model's per-layer residuals should approach
           hippocampal targets
      5. Backprop through LoRA params only

    This is the core claim: the hippocampus teaches cortex via replay,
    and LoRA adapters are the cortical weight modifications.

    Returns:
        history: dict of per-epoch loss values and diagnostics
    """
    hippo = system['hippo']
    cortical_proj = system['cortical_proj']
    backproj = system['backproj']

    optimizer = optim.Adam(lora_model.get_trainable_params(), lr=lr)

    history = {
        'epoch': [],
        'ce_loss': [],
        'mse_loss': [],
        'total_loss': [],
        'lora_norms': [],
    }

    # Pre-compute hippocampal replay targets for each fact
    replay_targets = {}
    for pair in fact_pairs:
        fid = pair['id']
        fact_text = pair['fact']
        token_ids = tokenizer.encode(fact_text)

        # Get the ground-truth hidden states from the frozen model
        with torch.no_grad():
            input_ids = torch.tensor([token_ids], device=device)
            outputs = model_base(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of 7 tensors

        # For each token position, get hippocampal target residuals
        # by encoding through hippocampal system and back-projecting
        per_token_targets = []
        for t in range(len(token_ids)):
            layer_residuals = []
            for l in range(1, 7):
                h = hidden_states[l][0, t, :].clone().to(torch.float32).to(device)
                layer_residuals.append(h)

            # Project to EC space
            ec_input = cortical_proj.project(layer_residuals)

            # Hippocampal retrieval (simulating replay)
            # During sleep replay, hippocampus reactivates stored patterns
            stellate, _ = hippo.ec_sup.forward(ec_input)
            dg_out = hippo.dg.forward(stellate)
            ca3_out = hippo.ca3.retrieve(dg_out, hippo.ca3_retrieval_iterations)
            decoder_out = hippo.direct_decoder.retrieve(ca3_out)

            # Back-project to per-layer cortical targets
            target_residuals = backproj.retrieve(decoder_out)
            per_token_targets.append(target_residuals)

        replay_targets[fid] = {
            'token_ids': token_ids,
            'targets': per_token_targets,  # list of (list of 6 tensors)
        }

    n_facts = len(fact_pairs)
    if verbose:
        print(f"\n  Consolidation: {n_epochs} epochs, {n_facts} facts, "
              f"lr={lr}, mse_weight={mse_weight}")
        print(f"  Trainable params: {lora_model.total_trainable_params():,}")

    for epoch in range(n_epochs):
        epoch_ce = 0.0
        epoch_mse = 0.0
        epoch_total = 0.0

        # Shuffle fact order each epoch (as in biological replay,
        # different experiences are replayed in varying order)
        fact_order = np.random.permutation(n_facts)

        for fi in fact_order:
            pair = fact_pairs[fi]
            fid = pair['id']
            targets = replay_targets[fid]
            token_ids = targets['token_ids']
            hippo_targets = targets['targets']

            input_ids = torch.tensor([token_ids], device=device)

            # Forward through LoRA model
            logits, layer_outputs = lora_model(
                input_ids, return_layer_outputs=True)

            # --- Cross-entropy loss: predict fact tokens ---
            # Shift: logits[t] should predict token[t+1]
            if len(token_ids) > 1:
                shift_logits = logits[0, :-1, :]  # (seq_len-1, vocab)
                shift_labels = input_ids[0, 1:]    # (seq_len-1,)
                ce_loss = F.cross_entropy(shift_logits, shift_labels)
            else:
                ce_loss = torch.tensor(0.0, device=device)

            # --- MSE loss: match hippocampal target residuals ---
            mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_mse_terms = 0

            for l in range(6):
                if layer_outputs[l] is not None:
                    model_residual = layer_outputs[l][0]  # (seq_len, d_model)
                    for t in range(len(token_ids)):
                        target = hippo_targets[t][l].detach()
                        # Normalize both to unit vectors for direction-matching
                        pred = model_residual[t]
                        pred_norm = pred / (torch.linalg.norm(pred) + 1e-10)
                        targ_norm = target / (torch.linalg.norm(target) + 1e-10)
                        mse_loss = mse_loss + F.mse_loss(pred_norm, targ_norm)
                        n_mse_terms += 1

            if n_mse_terms > 0:
                mse_loss = mse_loss / n_mse_terms

            # --- Combined loss ---
            total_loss = ce_loss + mse_weight * mse_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_ce += float(ce_loss)
            epoch_mse += float(mse_loss)
            epoch_total += float(total_loss)

        epoch_ce /= n_facts
        epoch_mse /= n_facts
        epoch_total /= n_facts

        history['epoch'].append(epoch)
        history['ce_loss'].append(epoch_ce)
        history['mse_loss'].append(epoch_mse)
        history['total_loss'].append(epoch_total)
        history['lora_norms'].append(lora_model.lora_weight_norms())

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            norms = lora_model.lora_weight_norms()
            mean_b_norm = np.mean([n[1] for n in norms])
            print(f"    Epoch {epoch:>3d}: CE={epoch_ce:.4f}  "
                  f"MSE={epoch_mse:.4f}  Total={epoch_total:.4f}  "
                  f"mean_B_norm={mean_b_norm:.4f}")

    return history


# =============================================================================
# ABLATION-CAPABLE CONSOLIDATION ENGINE
# =============================================================================

def consolidation_ablation(lora_model, replay_targets, tokenizer,
                           n_epochs=50, lr=1e-3,
                           ce_weight=1.0, mse_weight=0.1,
                           residual_loss_type='cosine',
                           base_corpus_samples=None,
                           base_corpus_ratio=0.3,
                           device='cpu', verbose=True):
    """
    General-purpose consolidation loop supporting all ablation conditions.

    Ablation modes (controlled via ce_weight and mse_weight):
      - ce_weight=1, mse_weight=0:   Pure CE (standard fine-tuning baseline)
      - ce_weight=0, mse_weight=1:   Pure hippocampal (strongest claim)
      - ce_weight=1, mse_weight=0.5: Combined (pragmatic)
      - With random targets:          Control (tests specificity)

    base_corpus_samples: list of token_id lists for interleaved replay.
        When provided, base corpus samples are mixed in at base_corpus_ratio
        to prevent catastrophic forgetting (biologically: sleep replays
        both recent and remote memories).

    residual_loss_type: 'cosine', 'mse', or 'mse_raw'
    """
    optimizer = optim.Adam(lora_model.get_trainable_params(), lr=lr)

    history = {
        'epoch': [], 'ce_loss': [], 'mse_loss': [],
        'total_loss': [], 'base_ce_loss': [], 'lora_norms': [],
    }

    fact_ids = list(replay_targets.keys())
    n_facts = len(fact_ids)

    if verbose:
        mode = []
        if ce_weight > 0:
            mode.append(f"CE(w={ce_weight})")
        if mse_weight > 0:
            mode.append(f"MSE(w={mse_weight}, type={residual_loss_type})")
        if base_corpus_samples:
            mode.append(f"interleaved({base_corpus_ratio})")
        print(f"\n  Consolidation [{' + '.join(mode)}]")
        print(f"  {n_epochs} epochs, {n_facts} facts, lr={lr}")
        print(f"  Trainable params: {lora_model.total_trainable_params():,}")

    for epoch in range(n_epochs):
        epoch_ce = 0.0
        epoch_mse = 0.0
        epoch_total = 0.0
        epoch_base_ce = 0.0
        n_fact_steps = 0
        n_base_steps = 0

        # Build replay schedule: facts + interleaved base corpus
        replay_schedule = []
        fact_order = np.random.permutation(n_facts)
        for fi in fact_order:
            replay_schedule.append(('fact', fact_ids[fi]))

        if base_corpus_samples:
            n_base = max(1, int(len(replay_schedule) * base_corpus_ratio))
            base_indices = np.random.choice(
                len(base_corpus_samples), size=n_base, replace=True)
            for bi in base_indices:
                replay_schedule.append(('base', int(bi)))
            np.random.shuffle(replay_schedule)

        for item_type, item_id in replay_schedule:
            if item_type == 'fact':
                fid = item_id
                targets = replay_targets[fid]
                token_ids = targets['token_ids']
                hippo_targets = targets['targets']

                input_ids = torch.tensor([token_ids], device=device)
                logits, layer_outputs = lora_model(
                    input_ids, return_layer_outputs=True)

                # Cross-entropy component
                ce_loss = torch.tensor(0.0, device=device)
                if ce_weight > 0 and len(token_ids) > 1:
                    shift_logits = logits[0, :-1, :]
                    shift_labels = input_ids[0, 1:]
                    ce_loss = F.cross_entropy(shift_logits, shift_labels)

                # Residual-matching component
                mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if mse_weight > 0:
                    mse_loss, _ = compute_residual_loss(
                        layer_outputs, hippo_targets, token_ids,
                        loss_type=residual_loss_type)

                total_loss = ce_weight * ce_loss + mse_weight * mse_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_ce += float(ce_loss)
                epoch_mse += float(mse_loss)
                epoch_total += float(total_loss)
                n_fact_steps += 1

            elif item_type == 'base' and base_corpus_samples:
                # Interleaved base corpus replay (prevent forgetting)
                base_ids = base_corpus_samples[item_id]
                input_ids = torch.tensor([base_ids], device=device)

                logits = lora_model(input_ids)
                if len(base_ids) > 1:
                    shift_logits = logits[0, :-1, :]
                    shift_labels = input_ids[0, 1:]
                    base_ce = F.cross_entropy(shift_logits, shift_labels)

                    optimizer.zero_grad()
                    base_ce.backward()
                    optimizer.step()

                    epoch_base_ce += float(base_ce)
                    n_base_steps += 1

        if n_fact_steps > 0:
            epoch_ce /= n_fact_steps
            epoch_mse /= n_fact_steps
            epoch_total /= n_fact_steps
        if n_base_steps > 0:
            epoch_base_ce /= n_base_steps

        history['epoch'].append(epoch)
        history['ce_loss'].append(epoch_ce)
        history['mse_loss'].append(epoch_mse)
        history['total_loss'].append(epoch_total)
        history['base_ce_loss'].append(epoch_base_ce)
        history['lora_norms'].append(lora_model.lora_weight_norms())

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            norms = lora_model.lora_weight_norms()
            mean_b_norm = np.mean([n[1] for n in norms])
            parts = f"Epoch {epoch:>3d}: Total={epoch_total:.4f}"
            if ce_weight > 0:
                parts += f"  CE={epoch_ce:.4f}"
            if mse_weight > 0:
                parts += f"  MSE={epoch_mse:.4f}"
            if n_base_steps > 0:
                parts += f"  BaseCE={epoch_base_ce:.4f}"
            parts += f"  B_norm={mean_b_norm:.4f}"
            print(f"    {parts}")

    return history


# =============================================================================
# ABLATION FRAMEWORK
# =============================================================================

ABLATION_CONDITIONS = {
    'ce_only': {
        'description': 'Pure cross-entropy (standard fine-tuning baseline)',
        'ce_weight': 1.0, 'mse_weight': 0.0,
        'use_random_targets': False, 'interleave_base': False,
        'adapter_type': 'lora',
    },
    'mse_only': {
        'description': 'Pure hippocampal (strongest consolidation claim)',
        'ce_weight': 0.0, 'mse_weight': 1.0,
        'use_random_targets': False, 'interleave_base': False,
        'adapter_type': 'lora',
    },
    'combined': {
        'description': 'CE + hippocampal MSE (pragmatic)',
        'ce_weight': 1.0, 'mse_weight': 0.5,
        'use_random_targets': False, 'interleave_base': False,
        'adapter_type': 'lora',
    },
    'combined_interleaved': {
        'description': 'CE + MSE + base corpus interleaving',
        'ce_weight': 1.0, 'mse_weight': 0.5,
        'use_random_targets': False, 'interleave_base': True,
        'adapter_type': 'lora',
    },
    'random_targets': {
        'description': 'Random targets control (tests specificity)',
        'ce_weight': 0.0, 'mse_weight': 1.0,
        'use_random_targets': True, 'interleave_base': False,
        'adapter_type': 'lora',
    },
    'mse_only_bias': {
        'description': 'Pure hippocampal with bias-only adapters',
        'ce_weight': 0.0, 'mse_weight': 1.0,
        'use_random_targets': False, 'interleave_base': False,
        'adapter_type': 'bias',
    },
    'mse_successor': {
        'description': 'Pure hippocampal with autonomous successor replay',
        'ce_weight': 0.0, 'mse_weight': 1.0,
        'use_random_targets': False, 'interleave_base': False,
        'adapter_type': 'lora', 'use_successor': True,
    },
}


def run_ablation(condition_name, condition, model_base, tokenizer, system,
                 fact_pairs, replay_targets_hippo, replay_targets_random,
                 replay_targets_successor, base_corpus_samples,
                 n_epochs=100, lr=5e-4, lora_rank=8, device='cpu'):
    """
    Run a single ablation condition. Returns eval_results, gen_results, history.
    Creates a fresh adapter model for each condition.
    """
    print(f"\n{'='*60}")
    print(f"  ABLATION: {condition_name}")
    print(f"  {condition['description']}")
    print(f"{'='*60}")

    # Create fresh adapter model
    adapter_type = condition.get('adapter_type', 'lora')
    if adapter_type == 'bias':
        adapted_model = BiasInjectedModel(model_base, n_layers=6)
    else:
        adapted_model = LoRAInjectedModel(
            model_base, n_layers=6, rank=lora_rank, lora_scale=1.0)
    adapted_model.to(device)

    # Select targets
    if condition.get('use_random_targets', False):
        targets = replay_targets_random
    elif condition.get('use_successor', False):
        targets = replay_targets_successor
    else:
        targets = replay_targets_hippo

    # Select base corpus
    base_samples = base_corpus_samples if condition.get('interleave_base') else None

    # Run consolidation
    history = consolidation_ablation(
        adapted_model, targets, tokenizer,
        n_epochs=n_epochs, lr=lr,
        ce_weight=condition['ce_weight'],
        mse_weight=condition['mse_weight'],
        residual_loss_type='cosine',
        base_corpus_samples=base_samples,
        base_corpus_ratio=0.3,
        device=device, verbose=True)

    # Evaluate
    eval_results = evaluate_consolidation(
        adapted_model, tokenizer, fact_pairs, device=device)

    gen_texts = [
        "The quick brown fox jumps over the lazy dog and runs through the forest",
        "In a shocking turn of events, the president announced new policies",
        "The scientific community was surprised to learn about the new particle",
    ]
    gen_results = evaluate_generalization(
        adapted_model, model_base, tokenizer, gen_texts, device=device)

    return eval_results, gen_results, history


def plot_ablation_comparison(all_ablation_results, fact_pairs,
                             save_path="ablation_comparison.png"):
    """
    Compare all ablation conditions side by side.
    The key figure for the paper.
    """
    conditions = list(all_ablation_results.keys())
    n_conditions = len(conditions)
    n_pairs = len(fact_pairs)

    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle(
        "Hippocampal Consolidation: Ablation Study\n"
        "Which components of hippocampal replay drive cortical learning?",
        fontsize=14, fontweight='bold', y=0.99)

    # --- (0,0:2): Mean P(expected) across conditions ---
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(n_conditions)
    width = 0.25

    means_baseline = []
    means_consolidated = []
    means_reference = []

    for cname in conditions:
        evals = all_ablation_results[cname]['eval']
        means_baseline.append(
            np.mean([r['baseline_p_expected'] for r in evals]))
        means_consolidated.append(
            np.mean([r['consolidated_p_expected'] for r in evals]))
        means_reference.append(
            np.mean([r['reference_p_expected'] for r in evals]))

    ax.bar(x - width, means_baseline, width, label='Baseline', color='gray',
           alpha=0.7)
    ax.bar(x, means_consolidated, width, label='Consolidated', color='steelblue',
           alpha=0.9)
    ax.bar(x + width, means_reference, width, label='Reference (ceiling)',
           color='forestgreen', alpha=0.7)

    # Add shift annotations
    for i, cname in enumerate(conditions):
        shift = means_consolidated[i] - means_baseline[i]
        color = 'green' if shift > 0 else 'red'
        ax.annotate(f'{shift:+.4f}',
                    xy=(x[i], means_consolidated[i]),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=8, ha='center', color=color, fontweight='bold')

    short_labels = [c.replace('_', '\n') for c in conditions]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_ylabel("Mean P(expected token)")
    ax.set_title("Mean Probability of Correct Token Across Ablation Conditions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (1,0): Per-fact comparison (key conditions only) ---
    ax = fig.add_subplot(gs[1, 0])
    key_conditions = [c for c in conditions
                      if c in ('ce_only', 'mse_only', 'combined', 'random_targets')]
    if not key_conditions:
        key_conditions = conditions[:4]

    pair_ids = [p['id'] for p in fact_pairs]
    x_pairs = np.arange(n_pairs)
    w = 0.8 / (len(key_conditions) + 1)

    for ci, cname in enumerate(key_conditions):
        evals = all_ablation_results[cname]['eval']
        shifts = [r['shift'] for r in evals]
        offset = (ci - len(key_conditions)/2) * w
        ax.bar(x_pairs + offset, shifts, w, label=cname, alpha=0.8)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x_pairs)
    ax.set_xticklabels(pair_ids, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel("P(expected) Shift")
    ax.set_title("Per-Fact Probability Shift by Condition")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (1,1): Generalization (PPL ratios) ---
    ax = fig.add_subplot(gs[1, 1])
    ppl_ratios = []
    for cname in conditions:
        gen = all_ablation_results[cname]['gen']
        if gen:
            ppl_ratios.append(np.mean([r['ppl_ratio'] for r in gen]))
        else:
            ppl_ratios.append(1.0)

    colors_ppl = ['green' if r < 1.2 else 'orange' if r < 1.5 else 'red'
                  for r in ppl_ratios]
    ax.barh(range(n_conditions), ppl_ratios, color=colors_ppl, alpha=0.8)
    ax.axvline(x=1.0, color='black', linewidth=0.5, linestyle='--')
    ax.axvline(x=1.2, color='orange', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_yticks(range(n_conditions))
    ax.set_yticklabels([c.replace('_', ' ') for c in conditions], fontsize=8)
    ax.set_xlabel("Perplexity Ratio (LoRA / Base)")
    ax.set_title("Generalization: Perplexity Degradation")
    ax.grid(True, alpha=0.3, axis='x')

    # --- (2,0): Training curves (loss) ---
    ax = fig.add_subplot(gs[2, 0])
    for cname in conditions:
        hist = all_ablation_results[cname]['history']
        ax.plot(hist['epoch'], hist['total_loss'], label=cname, linewidth=1.5)
    ax.set_xlabel("Replay Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Consolidation Training Curves")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- (2,1): Summary table ---
    ax = fig.add_subplot(gs[2, 1])
    ax.axis('off')

    summary = "ABLATION SUMMARY\n" + "=" * 60 + "\n\n"
    summary += f"{'Condition':<22s} {'Shift':>8s} {'Recovery':>10s} "
    summary += f"{'PPL':>6s} {'N+':>4s}\n"
    summary += "-" * 55 + "\n"

    for ci, cname in enumerate(conditions):
        evals = all_ablation_results[cname]['eval']
        shift = means_consolidated[ci] - means_baseline[ci]
        if means_reference[ci] - means_baseline[ci] > 1e-8:
            recovery = shift / (means_reference[ci] - means_baseline[ci])
        else:
            recovery = 0.0
        n_pos = sum(1 for r in evals if r['shift'] > 0)
        ppl = ppl_ratios[ci]
        summary += (f"{cname:<22s} {shift:>+8.5f} {recovery:>10.3f} "
                    f"{ppl:>6.3f} {n_pos:>2d}/{n_pairs}\n")

    summary += "\n" + "-" * 55 + "\n"
    summary += "Key: Shift = consolidated - baseline P(expected)\n"
    summary += "     Recovery = shift / (reference - baseline)\n"
    summary += "     PPL = perplexity ratio (1.0 = no degradation)\n"
    summary += "     N+ = facts with positive shift\n"

    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")


# =============================================================================
# REPLAY QUALITY DIAGNOSTICS
# =============================================================================

def diagnose_replay_quality(system, model_base, tokenizer, fact_pairs,
                            replay_targets, device='cpu'):
    """
    Measure how faithfully the hippocampal system reconstructs the
    original per-layer residuals. This tells us how much information
    the LoRA training signal actually carries.
    """
    cortical_proj = system['cortical_proj']

    print("\n  Replay Target Quality Diagnostics:")
    print(f"  {'Fact':<20s} {'Mean CosSim':>12s} {'Min CosSim':>12s} "
          f"{'Max CosSim':>12s}")
    print("  " + "-" * 60)

    all_sims = []
    for pair in fact_pairs:
        fid = pair['id']
        fact_text = pair['fact']
        token_ids = tokenizer.encode(fact_text)

        with torch.no_grad():
            input_ids = torch.tensor([token_ids], device=device)
            outputs = model_base(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states

        targets = replay_targets[fid]['targets']
        fact_sims = []

        for t in range(len(token_ids)):
            for l in range(6):
                actual = hidden_states[l + 1][0, t, :].to(torch.float32).to(device)
                predicted = targets[t][l]
                sim = cosine_sim(predicted, actual)
                fact_sims.append(sim)

        mean_sim = np.mean(fact_sims)
        min_sim = np.min(fact_sims)
        max_sim = np.max(fact_sims)
        all_sims.extend(fact_sims)

        print(f"  {fid:<20s} {mean_sim:>12.4f} {min_sim:>12.4f} {max_sim:>12.4f}")

    print("  " + "-" * 60)
    print(f"  {'OVERALL':<20s} {np.mean(all_sims):>12.4f} "
          f"{np.min(all_sims):>12.4f} {np.max(all_sims):>12.4f}")

    return all_sims


# =============================================================================
# EVALUATION: POST-CONSOLIDATION (NO HIPPOCAMPUS)
# =============================================================================

def evaluate_consolidation(lora_model, tokenizer, fact_pairs, device='cpu'):
    """
    Test the LoRA-modified model on fact-test pairs WITHOUT the hippocampus.

    If consolidation worked, the model's own weights (via LoRA adapters)
    should now encode the facts, predicting "vinegar" etc. without any
    hippocampal retrieval or injection.

    Also evaluates the base model (no LoRA) as a baseline,
    and in-context reference as a ceiling.
    """
    results = []

    # We need the base model for baselines. The LoRA model wraps it,
    # so we can get it via lora_model.base_model.
    base_model = lora_model.base_model

    for pair in fact_pairs:
        test_text = pair['test']
        expected = pair['expected']
        default = pair['default']
        all_tokens = expected + default

        # --- Baseline: frozen model, no LoRA, no hippocampus ---
        token_ids = tokenizer.encode(test_text)
        input_ids = torch.tensor([token_ids], device=device)
        last_pos = len(token_ids) - 1

        with torch.no_grad():
            base_logits = base_model(input_ids).logits
        base_probs = F.softmax(base_logits[0, last_pos, :], dim=-1)

        baseline_p_expected = sum(
            float(base_probs[tokenizer.encode(t)[0]])
            for t in expected if len(tokenizer.encode(t)) >= 1
        )
        baseline_p_default = sum(
            float(base_probs[tokenizer.encode(t)[0]])
            for t in default if len(tokenizer.encode(t)) >= 1
        )

        top5_v, top5_i = torch.topk(base_probs, 5)
        baseline_top5 = [(tokenizer.decode([tid.item()]), float(tv))
                         for tid, tv in zip(top5_i, top5_v)]

        # --- Consolidated: LoRA model, no hippocampus ---
        with torch.no_grad():
            lora_logits = lora_model(input_ids)
        lora_probs = F.softmax(lora_logits[0, last_pos, :], dim=-1)

        consolidated_p_expected = sum(
            float(lora_probs[tokenizer.encode(t)[0]])
            for t in expected if len(tokenizer.encode(t)) >= 1
        )
        consolidated_p_default = sum(
            float(lora_probs[tokenizer.encode(t)[0]])
            for t in default if len(tokenizer.encode(t)) >= 1
        )

        top5_v_c, top5_i_c = torch.topk(lora_probs, 5)
        consolidated_top5 = [(tokenizer.decode([tid.item()]), float(tv))
                             for tid, tv in zip(top5_i_c, top5_v_c)]

        # --- Reference: fact + test in same context (ceiling) ---
        ref_text = pair['fact'] + " " + test_text
        ref_ids = tokenizer.encode(ref_text)
        ref_input = torch.tensor([ref_ids], device=device)
        ref_last = len(ref_ids) - 1

        with torch.no_grad():
            ref_logits = base_model(ref_input).logits
        ref_probs = F.softmax(ref_logits[0, ref_last, :], dim=-1)

        reference_p_expected = sum(
            float(ref_probs[tokenizer.encode(t)[0]])
            for t in expected if len(tokenizer.encode(t)) >= 1
        )

        top5_v_r, top5_i_r = torch.topk(ref_probs, 5)
        reference_top5 = [(tokenizer.decode([tid.item()]), float(tv))
                          for tid, tv in zip(top5_i_r, top5_v_r)]

        # --- Compute shift metrics ---
        shift = consolidated_p_expected - baseline_p_expected
        if reference_p_expected - baseline_p_expected > 1e-8:
            recovery = (consolidated_p_expected - baseline_p_expected) / \
                       (reference_p_expected - baseline_p_expected)
        else:
            recovery = 0.0

        results.append({
            'id': pair['id'],
            'fact': pair['fact'],
            'test': test_text,
            'baseline_p_expected': baseline_p_expected,
            'baseline_p_default': baseline_p_default,
            'baseline_top5': baseline_top5,
            'consolidated_p_expected': consolidated_p_expected,
            'consolidated_p_default': consolidated_p_default,
            'consolidated_top5': consolidated_top5,
            'reference_p_expected': reference_p_expected,
            'reference_top5': reference_top5,
            'shift': shift,
            'recovery_ratio': recovery,
        })

    return results


def evaluate_generalization(lora_model, base_model, tokenizer,
                            test_texts, device='cpu'):
    """
    Check that LoRA adapters don't catastrophically degrade general performance.
    Compares perplexity on held-out text between base and LoRA models.
    """
    results = []
    for text in test_texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            continue
        input_ids = torch.tensor([token_ids], device=device)

        with torch.no_grad():
            base_logits = base_model(input_ids).logits
            lora_logits = lora_model(input_ids)

        # Per-token cross-entropy (shifted)
        shift_labels = input_ids[0, 1:]

        base_ce = F.cross_entropy(base_logits[0, :-1, :], shift_labels)
        lora_ce = F.cross_entropy(lora_logits[0, :-1, :], shift_labels)

        results.append({
            'text': text[:60] + '...',
            'base_ppl': float(torch.exp(base_ce)),
            'lora_ppl': float(torch.exp(lora_ce)),
            'ppl_ratio': float(torch.exp(lora_ce)) / (float(torch.exp(base_ce)) + 1e-10),
        })

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_consolidation_results(history, eval_results, gen_results,
                               save_path="consolidation_results.png"):
    """Comprehensive results figure for the consolidation experiment."""
    n_pairs = len(eval_results)

    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        "Hippocampal Consolidation via LoRA:\n"
        "Hippocampus as a Training System for Cortical Weights",
        fontsize=14, fontweight='bold', y=0.99)

    # --- (0,0): Training loss curves ---
    ax = fig.add_subplot(gs[0, 0])
    epochs = history['epoch']
    ax.plot(epochs, history['ce_loss'], 'b-', label='Cross-Entropy', linewidth=2)
    ax.plot(epochs, history['mse_loss'], 'r--', label='MSE (aux)', linewidth=1.5)
    ax.plot(epochs, history['total_loss'], 'k-', label='Total', linewidth=1,
            alpha=0.5)
    ax.set_xlabel("Replay Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Consolidation Training Loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (0,1): LoRA weight norms over training ---
    ax = fig.add_subplot(gs[0, 1])
    n_layers = len(history['lora_norms'][0])
    for l in range(n_layers):
        b_norms = [norms[l][1] for norms in history['lora_norms']]
        ax.plot(epochs, b_norms, label=f'Layer {l+1}', linewidth=1.5)
    ax.set_xlabel("Replay Epoch")
    ax.set_ylabel("LoRA B Matrix Norm")
    ax.set_title("LoRA Adapter Growth During Consolidation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- (0,2): LoRA A norms ---
    ax = fig.add_subplot(gs[0, 2])
    for l in range(n_layers):
        a_norms = [norms[l][0] for norms in history['lora_norms']]
        ax.plot(epochs, a_norms, label=f'Layer {l+1}', linewidth=1.5)
    ax.set_xlabel("Replay Epoch")
    ax.set_ylabel("LoRA A Matrix Norm")
    ax.set_title("LoRA A Matrix Norms")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- (1, 0:3): P(expected) across conditions ---
    ax = fig.add_subplot(gs[1, :])
    pair_ids = [r['id'] for r in eval_results]
    x = np.arange(n_pairs)
    width = 0.25

    baseline_p = [r['baseline_p_expected'] for r in eval_results]
    consolidated_p = [r['consolidated_p_expected'] for r in eval_results]
    reference_p = [r['reference_p_expected'] for r in eval_results]

    ax.bar(x - width, baseline_p, width, label='Baseline (no fact)',
           color='gray', alpha=0.8)
    ax.bar(x, consolidated_p, width,
           label='Consolidated (LoRA, no hippocampus)',
           color='steelblue', alpha=0.9)
    ax.bar(x + width, reference_p, width, label='Reference (in-context)',
           color='forestgreen', alpha=0.8)

    ax.set_ylabel("P(expected token)")
    ax.set_title("Post-Consolidation: Does the Model Know the Facts?")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_ids, fontsize=8, rotation=30, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add shift annotations
    for i in range(n_pairs):
        shift = eval_results[i]['shift']
        color = 'green' if shift > 0 else 'red'
        ax.annotate(f'{shift:+.4f}',
                    xy=(x[i], consolidated_p[i]),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=6, ha='center', color=color)

    # --- (2,0): Recovery ratios ---
    ax = fig.add_subplot(gs[2, 0])
    recoveries = [r['recovery_ratio'] for r in eval_results]
    colors = ['steelblue' if r > 0 else 'salmon' for r in recoveries]
    ax.barh(range(n_pairs), recoveries, color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.axvline(x=1, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=8)
    ax.set_xlabel("Recovery Ratio")
    ax.set_title("Recovery: (Consolidated - Baseline) / (Reference - Baseline)")
    ax.grid(True, alpha=0.3, axis='x')

    # --- (2,1): Generalization check ---
    ax = fig.add_subplot(gs[2, 1])
    if gen_results:
        gen_labels = [f"Text {i}" for i in range(len(gen_results))]
        base_ppls = [r['base_ppl'] for r in gen_results]
        lora_ppls = [r['lora_ppl'] for r in gen_results]
        x_gen = np.arange(len(gen_results))
        ax.bar(x_gen - 0.15, base_ppls, 0.3, label='Base', color='gray')
        ax.bar(x_gen + 0.15, lora_ppls, 0.3, label='LoRA', color='steelblue')
        ax.set_ylabel("Perplexity")
        ax.set_title("Generalization: Perplexity on Held-Out Text")
        ax.set_xticks(x_gen)
        ax.set_xticklabels(gen_labels, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "No generalization data", transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title("Generalization Check")

    # --- (2,2): Top predictions comparison ---
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    table_text = "Top-3 Predictions\n" + "=" * 55 + "\n\n"
    for r in eval_results[:6]:
        table_text += f"{r['id']}:\n"
        table_text += f"  Test: \"{r['test']}\"\n"
        table_text += f"  Base:  "
        table_text += ", ".join(
            f"{tok}({p:.3f})" for tok, p in r['baseline_top5'][:3])
        table_text += "\n"
        table_text += f"  LoRA:  "
        table_text += ", ".join(
            f"{tok}({p:.3f})" for tok, p in r['consolidated_top5'][:3])
        table_text += "\n"
        table_text += f"  Ref:   "
        table_text += ", ".join(
            f"{tok}({p:.3f})" for tok, p in r['reference_top5'][:3])
        table_text += "\n\n"

    ax.text(0.02, 0.98, table_text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- (3, 0:3): Summary statistics ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    mean_baseline = np.mean(baseline_p)
    mean_consolidated = np.mean(consolidated_p)
    mean_reference = np.mean(reference_p)
    mean_shift = np.mean([r['shift'] for r in eval_results])
    mean_recovery = np.mean(recoveries)
    n_improved = sum(1 for r in eval_results if r['shift'] > 0)
    n_strong = sum(1 for r in eval_results if r['recovery_ratio'] > 0.1)

    summary = "CONSOLIDATION SUMMARY\n" + "=" * 65 + "\n\n"
    summary += "The hippocampus as a LoRA training system for cortical weights.\n"
    summary += "After consolidation, the hippocampus is removed entirely.\n\n"
    summary += f"Mean P(expected) across {n_pairs} fact-test pairs:\n"
    summary += f"  Baseline (frozen model):      {mean_baseline:.6f}\n"
    summary += f"  Consolidated (LoRA, no hippo): {mean_consolidated:.6f}\n"
    summary += f"  Reference (in-context):        {mean_reference:.6f}\n\n"
    summary += f"Mean probability shift:          {mean_shift:+.6f}\n"
    summary += f"Mean recovery ratio:             {mean_recovery:.3f}\n"
    summary += f"Facts with positive shift:       {n_improved}/{n_pairs}\n"
    summary += f"Facts with recovery > 10%:       {n_strong}/{n_pairs}\n"

    if gen_results:
        mean_ppl_ratio = np.mean([r['ppl_ratio'] for r in gen_results])
        summary += f"\nGeneralization (PPL ratio):       {mean_ppl_ratio:.3f}\n"
        summary += f"  (1.0 = no degradation, <1.2 = acceptable)\n"

    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("=" * 70)
    print("HIPPOCAMPAL CONSOLIDATION VIA LoRA")
    print("The Hippocampus as a Training System for Cortical Weights")
    print("=" * 70)

    # ---- Phase 0: Build hippocampal system (base corpus training) ----
    print("\n--- Phase 0: Building hippocampal system ---")
    system = build_hippocampal_system(device)

    # ---- Load model ----
    model, tokenizer = load_model(device)

    # ---- Phase 1: Encode facts into hippocampus ----
    print("\n--- Phase 1: Encoding facts into hippocampal system ---")
    stored_facts = encode_facts(
        model, tokenizer, system, FACT_TEST_PAIRS,
        device=device, n_repetitions=3)

    print(f"\n  Stored {len(stored_facts)} facts")

    # ---- Phase 2: Pre-consolidation baseline ----
    print("\n--- Phase 2: Pre-consolidation baseline ---")
    for pair in FACT_TEST_PAIRS[:3]:
        test_text = pair['test']
        token_ids = tokenizer.encode(test_text)
        input_ids = torch.tensor([token_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids).logits
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5_v, top5_i = torch.topk(probs, 5)
        top5 = [(tokenizer.decode([tid.item()]), float(tv))
                for tid, tv in zip(top5_i, top5_v)]
        print(f"  {pair['id']}: {top5[:3]}")

    # ---- Phase 3: Prepare replay targets ----
    print("\n--- Phase 3: Computing replay targets ---")

    # Hippocampal direct replay targets
    print("  Computing hippocampal direct replay targets...")
    replay_targets_hippo = compute_replay_targets(
        system, model, tokenizer, FACT_TEST_PAIRS,
        device=device, use_successor=False)

    # Hippocampal successor replay targets (autonomous)
    print("  Computing successor replay targets...")
    replay_targets_successor = compute_replay_targets(
        system, model, tokenizer, FACT_TEST_PAIRS,
        device=device, use_successor=True)

    # Random targets (control)
    print("  Computing random control targets...")
    replay_targets_random = compute_random_targets(
        replay_targets_hippo, system['backproj'], device=device)

    # Base corpus samples for interleaved replay
    print("  Preparing base corpus samples...")
    base_corpus_samples = prepare_base_corpus_samples(
        model, tokenizer, DEFAULT_TEXT, n_samples=20, seq_length=32,
        device=device)
    print(f"  Prepared {len(base_corpus_samples)} base corpus samples")

    # ---- Phase 3.5: Replay quality diagnostics ----
    print("\n--- Phase 3.5: Replay target quality ---")
    diagnose_replay_quality(
        system, model, tokenizer, FACT_TEST_PAIRS,
        replay_targets_hippo, device=device)

    # ---- Phase 4: Hyperparameters ----
    lora_rank = 8
    consolidation_epochs = 100
    consolidation_lr = 5e-4

    print(f"\n--- Hyperparameters ---")
    print(f"  LoRA rank: {lora_rank}")
    print(f"  Epochs: {consolidation_epochs}")
    print(f"  Learning rate: {consolidation_lr}")

    # ---- Phase 5: Run ablations ----
    # Select which conditions to run. For a quick test, use a subset.
    # For the paper, run all of them.
    conditions_to_run = [
        'ce_only',          # Baseline: is this just fine-tuning?
        'mse_only',         # Core claim: pure hippocampal consolidation
        'combined',         # Pragmatic: CE + hippocampal signal
        'random_targets',   # Control: do hippocampal targets matter?
    ]

    # Uncomment for the full paper version:
    # conditions_to_run = list(ABLATION_CONDITIONS.keys())

    all_ablation_results = {}

    for cname in conditions_to_run:
        condition = ABLATION_CONDITIONS[cname]
        eval_results, gen_results, history = run_ablation(
            cname, condition, model, tokenizer, system,
            FACT_TEST_PAIRS, replay_targets_hippo, replay_targets_random,
            replay_targets_successor, base_corpus_samples,
            n_epochs=consolidation_epochs, lr=consolidation_lr,
            lora_rank=lora_rank, device=device)

        all_ablation_results[cname] = {
            'eval': eval_results,
            'gen': gen_results,
            'history': history,
            'condition': condition,
        }

        # Print this condition's results
        print(f"\n  Results for {cname}:")
        print(f"  {'Pair':<20s} {'Baseline':>10s} {'Consolidated':>12s} "
              f"{'Reference':>10s} {'Shift':>10s}")
        print("  " + "-" * 65)
        for r in eval_results:
            print(f"  {r['id']:<20s} {r['baseline_p_expected']:>10.6f} "
                  f"{r['consolidated_p_expected']:>12.6f} "
                  f"{r['reference_p_expected']:>10.6f} "
                  f"{r['shift']:>+10.6f}")

        mean_shift = np.mean([r['shift'] for r in eval_results])
        n_improved = sum(1 for r in eval_results if r['shift'] > 0)
        print(f"  Mean shift: {mean_shift:+.6f}, "
              f"Improved: {n_improved}/{len(eval_results)}")

    # ---- Phase 6: Also run the original combined consolidation
    # for the single-condition plot ----
    print("\n--- Phase 6: Primary consolidation (for detailed plot) ---")
    primary_model = LoRAInjectedModel(
        model, n_layers=6, rank=lora_rank, lora_scale=1.0)
    primary_model.to(device)

    # Verify identity
    test_ids = tokenizer.encode("Alice always drinks")
    test_input = torch.tensor([test_ids], device=device)
    with torch.no_grad():
        base_logits = model(test_input).logits
        lora_logits = primary_model(test_input)
    diff = float(torch.abs(base_logits - lora_logits).max())
    print(f"  LoRA identity check: max_diff={diff:.6f}")

    primary_history = consolidation_replay(
        primary_model, system, model, tokenizer,
        FACT_TEST_PAIRS, stored_facts,
        n_epochs=consolidation_epochs,
        lr=consolidation_lr,
        mse_weight=0.5,
        device=device,
        verbose=True)

    primary_eval = evaluate_consolidation(
        primary_model, tokenizer, FACT_TEST_PAIRS, device=device)

    gen_texts = [
        "The quick brown fox jumps over the lazy dog and runs through the forest",
        "In a shocking turn of events, the president announced new policies regarding",
        "The scientific community was surprised to learn that the new particle",
        "Once upon a time in a land far away, there lived a young princess who",
        "The stock market experienced significant volatility as investors reacted to",
    ]
    primary_gen = evaluate_generalization(
        primary_model, model, tokenizer, gen_texts, device=device)

    # ---- Phase 7: Plot everything ----
    print("\n--- Phase 7: Plotting ---")

    # Detailed single-condition plot
    plot_consolidation_results(
        primary_history, primary_eval, primary_gen,
        save_path="consolidation_results.png")

    # Ablation comparison plot
    if len(all_ablation_results) > 1:
        plot_ablation_comparison(
            all_ablation_results, FACT_TEST_PAIRS,
            save_path="ablation_comparison.png")

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(f"\n{'Condition':<24s} {'Mean Shift':>12s} {'Recovery':>10s} "
          f"{'N Improved':>12s}")
    print("-" * 62)

    for cname in conditions_to_run:
        evals = all_ablation_results[cname]['eval']
        mean_b = np.mean([r['baseline_p_expected'] for r in evals])
        mean_c = np.mean([r['consolidated_p_expected'] for r in evals])
        mean_r = np.mean([r['reference_p_expected'] for r in evals])
        shift = mean_c - mean_b
        recovery = shift / (mean_r - mean_b) if (mean_r - mean_b) > 1e-8 else 0
        n_imp = sum(1 for r in evals if r['shift'] > 0)
        print(f"  {cname:<22s} {shift:>+12.6f} {recovery:>10.3f} "
              f"{n_imp:>5d}/{len(evals)}")

    print(f"\nPrimary (combined):")
    mean_b = np.mean([r['baseline_p_expected'] for r in primary_eval])
    mean_c = np.mean([r['consolidated_p_expected'] for r in primary_eval])
    mean_r = np.mean([r['reference_p_expected'] for r in primary_eval])
    mean_shift = mean_c - mean_b
    n_improved = sum(1 for r in primary_eval if r['shift'] > 0)
    print(f"  P(expected): {mean_b:.6f} -> {mean_c:.6f} (shift: {mean_shift:+.6f})")
    print(f"  Facts improved: {n_improved}/{len(primary_eval)}")
    print(f"  LoRA params: {primary_model.total_trainable_params():,}")
    print(f"  Hippocampus: REMOVED for evaluation")
    print("=" * 70)


if __name__ == "__main__":
    main()
