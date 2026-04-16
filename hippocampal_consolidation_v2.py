"""
Hippocampal Consolidation v2: Coordinated Training
====================================================

Two new consolidation losses that fix the broken per-layer approach.

The original per-layer MSE gave each LoRA adapter an independent target:
    L = (1/L) sum_l d(h'_l, h~_l)
This fails because the targets assume the frozen residual stream, which
the adapters are actively modifying. Six uncoordinated regression problems.

Variant 1 - Final-residual matching:
    L = d(h'_L, h~_L)
Single loss on the last layer. Backprop flows through all frozen blocks
and all LoRA adapters. The gradient for adapter l includes the Jacobian
chain through blocks l+1...L, so adapters are implicitly coordinated.

Variant 2 - Logit distillation:
    L = KL( softmax(lm_head(ln_f(h~_L))/T) || softmax(lm_head(ln_f(h'_L))/T) )
The hippocampal target for the last layer is decoded through the frozen
model head into a probability distribution, and the LoRA model learns
to match it. "Close in residual space" becomes "predicts similar tokens."

Both variants require NO ground-truth token labels. The hippocampus
provides the only training signal.

Depends on: hippocampal_transformer_backprojection.py,
            fact_learning_paradigm.py, ca1_mechanism_test.py
"""

import sys
import os
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
    build_hippocampal_system,
    encode_facts,
)


# =============================================================================
# LoRA ADAPTER (same as v1)
# =============================================================================

class LoRAAdapter(nn.Module):
    def __init__(self, d_model, rank=4, scale=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, d_model) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        self.scale = scale

    def forward(self, x):
        return x + self.scale * F.linear(F.linear(x, self.A), self.B)


class LoRAInjectedModel(nn.Module):
    def __init__(self, base_model, n_layers=6, rank=4, lora_scale=1.0):
        super().__init__()
        self.base_model = base_model
        self.n_layers = n_layers

        for param in self.base_model.parameters():
            param.requires_grad = False

        d_model = base_model.config.n_embd
        self.lora_adapters = nn.ModuleList([
            LoRAAdapter(d_model, rank=rank, scale=lora_scale)
            for _ in range(n_layers)
        ])
        self._layer_outputs = [None] * n_layers
        self._hooks = []

    def _register_hooks(self):
        self._remove_hooks()
        for l in range(self.n_layers):
            adapter = self.lora_adapters[l]
            idx = l
            def make_hook(a, i):
                def hook_fn(module, input, output):
                    hidden = output[0]
                    modified = a(hidden)
                    self._layer_outputs[i] = modified
                    return (modified,) + output[1:]
                return hook_fn
            h = self.base_model.transformer.h[l].register_forward_hook(
                make_hook(adapter, idx))
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
        params = []
        for adapter in self.lora_adapters:
            params.extend(adapter.parameters())
        return params

    def total_trainable_params(self):
        return sum(p.numel() for p in self.get_trainable_params())

    def lora_weight_norms(self):
        return [(float(torch.linalg.norm(a.A)),
                 float(torch.linalg.norm(a.B)))
                for a in self.lora_adapters]


# =============================================================================
# REPLAY TARGET COMPUTATION
# =============================================================================

def compute_replay_targets(system, model_base, tokenizer, fact_pairs,
                           device='cpu'):
    """
    Compute hippocampal replay targets for each fact.

    For each token position, returns:
      - Per-layer hippocampal target residuals (for the old per-layer loss)
      - The last-layer target (for final-residual loss)
      - Hippocampal logits: h~_L pushed through ln_f + lm_head (for distillation)
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

        per_token_data = []
        for t in range(len(token_ids)):
            # Extract original residuals
            layer_residuals = []
            for l in range(1, 7):
                h = hidden_states[l][0, t, :].clone().to(torch.float32).to(device)
                layer_residuals.append(h)

            # Hippocampal encode -> retrieve -> backproject
            ec_input = cortical_proj.project(layer_residuals)
            stellate, _ = hippo.ec_sup.forward(ec_input)
            dg_out = hippo.dg.forward(stellate)
            ca3_out = hippo.ca3.retrieve(dg_out, hippo.ca3_retrieval_iterations)
            decoder_out = hippo.direct_decoder.retrieve(ca3_out)
            target_residuals = backproj.retrieve(decoder_out)

            # Last-layer target
            h_tilde_L = target_residuals[-1]  # (768,)

            # Hippocampal logits: push h~_L through frozen model head
            with torch.no_grad():
                h_for_head = h_tilde_L.unsqueeze(0).unsqueeze(0)  # (1,1,768)
                normed = model_base.transformer.ln_f(h_for_head)
                hippo_logits = model_base.lm_head(normed)[0, 0, :]  # (vocab,)

            per_token_data.append({
                'all_layer_targets': target_residuals,
                'last_layer_target': h_tilde_L,
                'hippo_logits': hippo_logits,
                'original_last': layer_residuals[-1],
            })

        replay_targets[fid] = {
            'token_ids': token_ids,
            'per_token': per_token_data,
        }

    return replay_targets


def compute_random_targets(replay_targets, model_base, backproj, device='cpu'):
    """Random-target control: same structure, random EC vectors."""
    random_targets = {}
    for fid, data in replay_targets.items():
        per_token_data = []
        for t in range(len(data['token_ids'])):
            random_ec = torch.randn(backproj.d_ec, device=device)
            random_ec = random_ec / (torch.linalg.norm(random_ec) + 1e-10)
            target_residuals = backproj.retrieve(random_ec)
            h_tilde_L = target_residuals[-1]

            with torch.no_grad():
                h_for_head = h_tilde_L.unsqueeze(0).unsqueeze(0)
                normed = model_base.transformer.ln_f(h_for_head)
                hippo_logits = model_base.lm_head(normed)[0, 0, :]

            per_token_data.append({
                'all_layer_targets': target_residuals,
                'last_layer_target': h_tilde_L,
                'hippo_logits': hippo_logits,
                'original_last': torch.zeros(768, device=device),
            })
        random_targets[fid] = {
            'token_ids': data['token_ids'],
            'per_token': per_token_data,
        }
    return random_targets


def add_noise_to_targets(replay_targets, noise_level, model_base, device='cpu'):
    """
    Degrade hippocampal targets by adding Gaussian noise at a given level.
    noise_level=0.0 means perfect reconstruction; 1.0 means noise has
    the same norm as the signal.
    """
    noisy_targets = {}
    for fid, data in replay_targets.items():
        per_token_data = []
        for orig in data['per_token']:
            h_L = orig['last_layer_target']
            noise = torch.randn_like(h_L)
            noise = noise * (torch.linalg.norm(h_L) / (torch.linalg.norm(noise) + 1e-10))
            h_noisy = h_L + noise_level * noise

            # Recompute logits from noisy target
            with torch.no_grad():
                h_for_head = h_noisy.unsqueeze(0).unsqueeze(0)
                normed = model_base.transformer.ln_f(h_for_head)
                hippo_logits = model_base.lm_head(normed)[0, 0, :]

            # Recompute per-layer (just copy originals with noise on last)
            noisy_all = [t.clone() for t in orig['all_layer_targets']]
            noisy_all[-1] = h_noisy

            per_token_data.append({
                'all_layer_targets': noisy_all,
                'last_layer_target': h_noisy,
                'hippo_logits': hippo_logits,
                'original_last': orig['original_last'],
            })
        noisy_targets[fid] = {
            'token_ids': data['token_ids'],
            'per_token': per_token_data,
        }
    return noisy_targets


# =============================================================================
# CONSOLIDATION ENGINE
# =============================================================================

def consolidation(lora_model, model_base, replay_targets, tokenizer,
                  loss_mode='logit_distill',
                  n_epochs=100, lr=5e-4, temperature=2.0,
                  ce_weight=0.0,
                  device='cpu', verbose=True):
    """
    Unified consolidation loop supporting all loss modes.

    loss_mode:
      'per_layer_cosine'  - old broken approach (independent per-layer matching)
      'final_residual'    - single loss on last layer, backprop coordinates all
      'logit_distill'     - KL divergence on hippocampal-decoded logits
      'ce_only'           - standard CE fine-tuning (no hippocampus)

    ce_weight: if >0, adds CE loss on ground-truth tokens alongside the
               hippocampal loss. Set to 0 for pure hippocampal experiments.
    """
    optimizer = optim.Adam(lora_model.get_trainable_params(), lr=lr)

    history = {
        'epoch': [], 'primary_loss': [], 'ce_loss': [],
        'total_loss': [], 'lora_norms': [],
    }

    fact_ids = list(replay_targets.keys())
    n_facts = len(fact_ids)

    if verbose:
        print(f"\n  Consolidation [{loss_mode}]"
              f"{f' + CE(w={ce_weight})' if ce_weight > 0 else ''}")
        print(f"  {n_epochs} epochs, {n_facts} facts, lr={lr}"
              f"{f', T={temperature}' if loss_mode == 'logit_distill' else ''}")
        print(f"  Trainable params: {lora_model.total_trainable_params():,}")

    for epoch in range(n_epochs):
        epoch_primary = 0.0
        epoch_ce = 0.0
        epoch_total = 0.0

        fact_order = np.random.permutation(n_facts)

        for fi in fact_order:
            fid = fact_ids[fi]
            data = replay_targets[fid]
            token_ids = data['token_ids']
            per_token = data['per_token']

            input_ids = torch.tensor([token_ids], device=device)

            # Forward through LoRA model
            logits, layer_outputs = lora_model(
                input_ids, return_layer_outputs=True)

            # ---- Primary loss (hippocampal signal) ----
            primary_loss = torch.tensor(0.0, device=device)

            if loss_mode == 'per_layer_cosine':
                # Old approach: independent per-layer cosine matching
                n_terms = 0
                for t in range(len(token_ids)):
                    for l in range(6):
                        if layer_outputs[l] is not None:
                            pred = layer_outputs[l][0, t, :]
                            targ = per_token[t]['all_layer_targets'][l].detach()
                            pred_n = pred / (torch.linalg.norm(pred) + 1e-10)
                            targ_n = targ / (torch.linalg.norm(targ) + 1e-10)
                            primary_loss = primary_loss + (
                                1.0 - (pred_n * targ_n).sum())
                            n_terms += 1
                if n_terms > 0:
                    primary_loss = primary_loss / n_terms

            elif loss_mode == 'final_residual':
                # Variant 1: single loss on last-layer output.
                # Backprop flows through all frozen blocks and all adapters.
                # layer_outputs[5] is the output of the last transformer block
                # AFTER LoRA modification, shape (1, seq_len, 768).
                # This single tensor carries gradients through the full chain.
                n_terms = 0
                for t in range(len(token_ids)):
                    if layer_outputs[5] is not None:
                        h_prime_L = layer_outputs[5][0, t, :]
                        h_tilde_L = per_token[t]['last_layer_target'].detach()
                        # Cosine distance
                        pred_n = h_prime_L / (torch.linalg.norm(h_prime_L) + 1e-10)
                        targ_n = h_tilde_L / (torch.linalg.norm(h_tilde_L) + 1e-10)
                        primary_loss = primary_loss + (
                            1.0 - (pred_n * targ_n).sum())
                        n_terms += 1
                if n_terms > 0:
                    primary_loss = primary_loss / n_terms

            elif loss_mode == 'logit_distill':
                # Variant 2: KL divergence between hippocampal logits and
                # LoRA model logits. Hippocampal target decoded into vocab
                # space via frozen ln_f + lm_head.
                n_terms = 0
                for t in range(len(token_ids)):
                    hippo_logits_t = per_token[t]['hippo_logits'].detach()
                    model_logits_t = logits[0, t, :]

                    # Temperature-scaled softmax
                    p_hippo = F.softmax(hippo_logits_t / temperature, dim=-1)
                    log_q_model = F.log_softmax(model_logits_t / temperature,
                                                dim=-1)

                    # KL(hippo || model), scaled by T^2 (standard distillation)
                    kl = F.kl_div(log_q_model, p_hippo, reduction='sum')
                    primary_loss = primary_loss + kl * (temperature ** 2)
                    n_terms += 1
                if n_terms > 0:
                    primary_loss = primary_loss / n_terms

            elif loss_mode == 'ce_only':
                # No hippocampal loss; ce_weight handles everything
                pass

            # ---- Optional CE loss ----
            ce_loss = torch.tensor(0.0, device=device)
            if ce_weight > 0 and len(token_ids) > 1:
                shift_logits = logits[0, :-1, :]
                shift_labels = input_ids[0, 1:]
                ce_loss = F.cross_entropy(shift_logits, shift_labels)

            # ---- Combined ----
            total_loss = primary_loss + ce_weight * ce_loss

            if loss_mode == 'ce_only':
                total_loss = ce_loss

            optimizer.zero_grad()
            if total_loss.requires_grad or (ce_weight > 0 and ce_loss.requires_grad):
                total_loss.backward()
                optimizer.step()

            epoch_primary += float(primary_loss)
            epoch_ce += float(ce_loss)
            epoch_total += float(total_loss)

        epoch_primary /= n_facts
        epoch_ce /= n_facts
        epoch_total /= n_facts

        history['epoch'].append(epoch)
        history['primary_loss'].append(epoch_primary)
        history['ce_loss'].append(epoch_ce)
        history['total_loss'].append(epoch_total)
        history['lora_norms'].append(lora_model.lora_weight_norms())

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            norms = lora_model.lora_weight_norms()
            mean_b = np.mean([n[1] for n in norms])
            parts = f"Epoch {epoch:>3d}: Total={epoch_total:.4f}"
            if loss_mode != 'ce_only':
                parts += f"  Primary={epoch_primary:.4f}"
            if ce_weight > 0:
                parts += f"  CE={epoch_ce:.4f}"
            parts += f"  B_norm={mean_b:.4f}"
            print(f"    {parts}")

    return history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(lora_model, tokenizer, fact_pairs, device='cpu'):
    """Test LoRA model on fact-test pairs. No hippocampus."""
    results = []
    base_model = lora_model.base_model

    for pair in fact_pairs:
        test_text = pair['test']
        expected = pair['expected']
        default = pair['default']
        all_tokens = expected + default

        token_ids = tokenizer.encode(test_text)
        input_ids = torch.tensor([token_ids], device=device)
        last_pos = len(token_ids) - 1

        with torch.no_grad():
            base_logits = base_model(input_ids).logits
            lora_logits = lora_model(input_ids)

        base_probs = F.softmax(base_logits[0, last_pos, :], dim=-1)
        lora_probs = F.softmax(lora_logits[0, last_pos, :], dim=-1)

        def sum_probs(probs, tokens):
            total = 0.0
            for t in tokens:
                ids = tokenizer.encode(t)
                if ids:
                    total += float(probs[ids[0]])
            return total

        baseline_p = sum_probs(base_probs, expected)
        consolidated_p = sum_probs(lora_probs, expected)

        # Reference: in-context
        ref_text = pair['fact'] + " " + test_text
        ref_ids = tokenizer.encode(ref_text)
        ref_input = torch.tensor([ref_ids], device=device)
        with torch.no_grad():
            ref_logits = base_model(ref_input).logits
        ref_probs = F.softmax(ref_logits[0, len(ref_ids)-1, :], dim=-1)
        reference_p = sum_probs(ref_probs, expected)

        # Top-5
        def top5(probs):
            v, i = torch.topk(probs, 5)
            return [(tokenizer.decode([ti.item()]), float(tv))
                    for ti, tv in zip(i, v)]

        shift = consolidated_p - baseline_p
        gap = reference_p - baseline_p
        recovery = shift / gap if abs(gap) > 1e-8 else 0.0

        results.append({
            'id': pair['id'],
            'baseline': baseline_p,
            'consolidated': consolidated_p,
            'reference': reference_p,
            'shift': shift,
            'recovery': recovery,
            'baseline_top5': top5(base_probs),
            'consolidated_top5': top5(lora_probs),
            'reference_top5': top5(ref_probs),
        })

    return results


def evaluate_generalization(lora_model, base_model, tokenizer, device='cpu'):
    """Perplexity on held-out text."""
    texts = [
        "The quick brown fox jumps over the lazy dog and runs through the forest",
        "In a shocking turn of events the president announced new policies regarding",
        "The scientific community was surprised to learn that the new particle had",
        "Once upon a time in a land far away there lived a young princess who loved",
        "The stock market experienced significant volatility as investors reacted to",
    ]
    results = []
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids], device=device)
        with torch.no_grad():
            base_logits = base_model(input_ids).logits
            lora_logits = lora_model(input_ids)
        labels = input_ids[0, 1:]
        base_ppl = float(torch.exp(F.cross_entropy(base_logits[0, :-1], labels)))
        lora_ppl = float(torch.exp(F.cross_entropy(lora_logits[0, :-1], labels)))
        results.append({
            'base_ppl': base_ppl, 'lora_ppl': lora_ppl,
            'ratio': lora_ppl / (base_ppl + 1e-10),
        })
    return results


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnose_targets(replay_targets, model_base, tokenizer, fact_pairs,
                     device='cpu'):
    """How much information do the hippocampal targets carry?"""
    print("\n  Target Quality Diagnostics:")
    print(f"  {'Fact':<20s} {'Resid CosSim':>12s} {'Logit KL':>10s} "
          f"{'Top1 Match':>10s} {'HippoTop1':>14s}")
    print("  " + "-" * 70)

    for pair in fact_pairs:
        fid = pair['id']
        data = replay_targets[fid]
        token_ids = data['token_ids']

        with torch.no_grad():
            input_ids = torch.tensor([token_ids], device=device)
            outputs = model_base(input_ids, output_hidden_states=True)
            true_logits = outputs.logits

        resid_sims = []
        kl_divs = []
        top1_matches = 0

        for t in range(len(token_ids)):
            pt = data['per_token'][t]

            # Residual similarity (last layer)
            true_h_L = outputs.hidden_states[-1][0, t, :].to(torch.float32)
            sim = cosine_sim(pt['last_layer_target'], true_h_L)
            resid_sims.append(sim)

            # Logit-space KL divergence
            true_p = F.softmax(true_logits[0, t, :], dim=-1)
            hippo_p = F.softmax(pt['hippo_logits'], dim=-1)
            kl = float(F.kl_div(hippo_p.log(), true_p, reduction='sum'))
            kl_divs.append(kl)

            # Top-1 token match
            true_top1 = true_logits[0, t, :].argmax().item()
            hippo_top1 = pt['hippo_logits'].argmax().item()
            if true_top1 == hippo_top1:
                top1_matches += 1

        mean_sim = np.mean(resid_sims)
        mean_kl = np.mean(kl_divs)
        match_rate = top1_matches / len(token_ids)

        # What does hippocampus predict for last token?
        last_hippo_top1_id = data['per_token'][-1]['hippo_logits'].argmax().item()
        last_hippo_top1 = tokenizer.decode([last_hippo_top1_id])

        print(f"  {fid:<20s} {mean_sim:>12.4f} {mean_kl:>10.2f} "
              f"{match_rate:>10.1%} {last_hippo_top1:>14s}")


def diagnose_gradient_flow(lora_model, replay_targets, tokenizer, device='cpu'):
    """
    Verify that the final-residual loss sends gradients to ALL adapters,
    not just the last one.
    """
    fid = list(replay_targets.keys())[0]
    data = replay_targets[fid]
    token_ids = data['token_ids']
    input_ids = torch.tensor([token_ids], device=device)

    logits, layer_outputs = lora_model(input_ids, return_layer_outputs=True)

    # Final-residual loss on a single token
    h_prime_L = layer_outputs[5][0, 0, :]
    h_tilde_L = data['per_token'][0]['last_layer_target'].detach()
    pred_n = h_prime_L / (torch.linalg.norm(h_prime_L) + 1e-10)
    targ_n = h_tilde_L / (torch.linalg.norm(h_tilde_L) + 1e-10)
    loss = 1.0 - (pred_n * targ_n).sum()

    lora_model.zero_grad()
    loss.backward()

    print("\n  Gradient flow check (final-residual loss):")
    print(f"  {'Layer':<8s} {'A grad norm':>12s} {'B grad norm':>12s} {'Receiving?':>12s}")
    print("  " + "-" * 48)
    for l, adapter in enumerate(lora_model.lora_adapters):
        a_grad = float(adapter.A.grad.norm()) if adapter.A.grad is not None else 0.0
        b_grad = float(adapter.B.grad.norm()) if adapter.B.grad is not None else 0.0
        has_grad = "YES" if (a_grad > 1e-10 or b_grad > 1e-10) else "NO"
        print(f"  {l+1:<8d} {a_grad:>12.6f} {b_grad:>12.6f} {has_grad:>12s}")


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(all_results, degradation_results=None,
                 save_path="consolidation_v2_results.png"):
    """Comprehensive figure for the paper."""
    conditions = list(all_results.keys())
    n_cond = len(conditions)
    n_pairs = len(FACT_TEST_PAIRS)

    n_rows = 4 if degradation_results else 3
    fig = plt.figure(figsize=(22, 5 * n_rows))
    gs = GridSpec(n_rows, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(
        "Hippocampal Consolidation v2: Coordinated LoRA Training\n"
        "Comparing per-layer, final-residual, and logit-distillation losses",
        fontsize=13, fontweight='bold', y=0.99)

    # --- (0,0:3): Mean P(expected) across conditions ---
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(n_cond)
    w = 0.25

    baselines = []
    consolidateds = []
    references = []
    for cname in conditions:
        evals = all_results[cname]['eval']
        baselines.append(np.mean([r['baseline'] for r in evals]))
        consolidateds.append(np.mean([r['consolidated'] for r in evals]))
        references.append(np.mean([r['reference'] for r in evals]))

    ax.bar(x - w, baselines, w, label='Baseline', color='gray', alpha=0.7)
    ax.bar(x, consolidateds, w, label='Consolidated', color='steelblue', alpha=0.9)
    ax.bar(x + w, references, w, label='Reference', color='forestgreen', alpha=0.7)

    for i in range(n_cond):
        shift = consolidateds[i] - baselines[i]
        color = 'green' if shift > 0 else 'red'
        ax.annotate(f'{shift:+.4f}',
                    xy=(x[i], consolidateds[i]),
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=7, ha='center', color=color, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in conditions], fontsize=8)
    ax.set_ylabel("Mean P(expected)")
    ax.set_title("Consolidation Quality Across Loss Functions")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (1,0): Per-fact shifts for key conditions ---
    ax = fig.add_subplot(gs[1, 0])
    key = [c for c in conditions if c in
           ('per_layer_cosine', 'final_residual', 'logit_distill', 'ce_only')]
    if not key:
        key = conditions[:4]
    pair_ids = [p['id'] for p in FACT_TEST_PAIRS]
    xp = np.arange(n_pairs)
    bw = 0.8 / max(len(key), 1)
    for ci, cname in enumerate(key):
        evals = all_results[cname]['eval']
        shifts = [r['shift'] for r in evals]
        ax.bar(xp + (ci - len(key)/2) * bw, shifts, bw, label=cname, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(xp)
    ax.set_xticklabels(pair_ids, fontsize=5, rotation=45, ha='right')
    ax.set_ylabel("P(expected) Shift")
    ax.set_title("Per-Fact Shifts")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (1,1): Training curves ---
    ax = fig.add_subplot(gs[1, 1])
    for cname in conditions:
        hist = all_results[cname]['history']
        ax.plot(hist['epoch'], hist['total_loss'], label=cname, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Training Curves")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- (1,2): Generalization ---
    ax = fig.add_subplot(gs[1, 2])
    ppl_ratios = []
    for cname in conditions:
        gen = all_results[cname]['gen']
        ppl_ratios.append(np.mean([r['ratio'] for r in gen]) if gen else 1.0)
    colors = ['green' if r < 1.5 else 'orange' if r < 3.0 else 'red'
              for r in ppl_ratios]
    ax.barh(range(n_cond), ppl_ratios, color=colors, alpha=0.8)
    ax.axvline(1.0, color='black', linewidth=0.5, linestyle='--')
    ax.set_yticks(range(n_cond))
    ax.set_yticklabels([c.replace('_', ' ') for c in conditions], fontsize=8)
    ax.set_xlabel("PPL Ratio (LoRA / Base)")
    ax.set_title("Generalization")
    ax.grid(True, alpha=0.3, axis='x')

    # --- (2,0): Top predictions table ---
    ax = fig.add_subplot(gs[2, 0:2])
    ax.axis('off')
    table = "Top-3 Predictions (selected pairs, selected conditions)\n"
    table += "=" * 80 + "\n\n"
    show_conds = [c for c in ('logit_distill', 'final_residual', 'ce_only')
                  if c in all_results]
    for pair in FACT_TEST_PAIRS[:5]:
        pid = pair['id']
        table += f"{pid}:  test=\"{pair['test']}\"\n"
        # baseline (same for all)
        r0 = all_results[conditions[0]]['eval']
        r0_match = [r for r in r0 if r['id'] == pid][0]
        table += f"  Base:  {', '.join(f'{t}({p:.3f})' for t,p in r0_match['baseline_top5'][:3])}\n"
        for cname in show_conds:
            evals = all_results[cname]['eval']
            r = [r for r in evals if r['id'] == pid][0]
            table += f"  {cname:20s}: {', '.join(f'{t}({p:.3f})' for t,p in r['consolidated_top5'][:3])}\n"
        table += f"  Ref:   {', '.join(f'{t}({p:.3f})' for t,p in r0_match['reference_top5'][:3])}\n\n"

    ax.text(0.02, 0.98, table, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- (2,2): Summary table ---
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    summary = "SUMMARY\n" + "=" * 45 + "\n\n"
    summary += f"{'Condition':<22s} {'Shift':>7s} {'Recv':>6s} "
    summary += f"{'PPL':>5s} {'N+':>5s}\n"
    summary += "-" * 48 + "\n"
    for ci, cname in enumerate(conditions):
        evals = all_results[cname]['eval']
        shift = consolidateds[ci] - baselines[ci]
        gap = references[ci] - baselines[ci]
        recv = shift / gap if abs(gap) > 1e-8 else 0.0
        n_pos = sum(1 for r in evals if r['shift'] > 0)
        summary += f"{cname:<22s} {shift:>+7.4f} {recv:>6.3f} "
        summary += f"{ppl_ratios[ci]:>5.2f} {n_pos:>2d}/{n_pairs}\n"

    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    # --- (3,0:3): Degradation study ---
    if degradation_results:
        ax = fig.add_subplot(gs[3, :])
        noise_levels = sorted(degradation_results.keys())

        for loss_mode in ('logit_distill', 'final_residual'):
            shifts = []
            for nl in noise_levels:
                if loss_mode in degradation_results[nl]:
                    evals = degradation_results[nl][loss_mode]['eval']
                    mean_b = np.mean([r['baseline'] for r in evals])
                    mean_c = np.mean([r['consolidated'] for r in evals])
                    shifts.append(mean_c - mean_b)
                else:
                    shifts.append(0.0)
            ax.plot(noise_levels, shifts, 'o-', label=loss_mode,
                    linewidth=2, markersize=6)

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlabel("Noise Level (0 = perfect, 1 = SNR=1)")
        ax.set_ylabel("Mean P(expected) Shift")
        ax.set_title("Consolidation Quality vs Hippocampal Reconstruction Fidelity")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

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
    print("HIPPOCAMPAL CONSOLIDATION v2")
    print("Coordinated LoRA Training via Final-Residual and Logit Distillation")
    print("=" * 70)

    # ---- Phase 0: Build hippocampal system ----
    print("\n--- Phase 0: Building hippocampal system ---")
    system = build_hippocampal_system(device)

    # ---- Load model ----
    model, tokenizer = load_model(device)

    # ---- Phase 1: Encode facts ----
    print("\n--- Phase 1: Encoding facts ---")
    stored_facts = encode_facts(
        model, tokenizer, system, FACT_TEST_PAIRS,
        device=device, n_repetitions=3)
    print(f"  Stored {len(stored_facts)} facts")

    # ---- Phase 2: Compute replay targets ----
    print("\n--- Phase 2: Computing replay targets ---")
    targets_hippo = compute_replay_targets(
        system, model, tokenizer, FACT_TEST_PAIRS, device=device)
    targets_random = compute_random_targets(
        targets_hippo, model, system['backproj'], device=device)

    # ---- Phase 2.5: Diagnostics ----
    diagnose_targets(targets_hippo, model, tokenizer, FACT_TEST_PAIRS,
                     device=device)

    # Gradient flow check
    print("\n--- Gradient flow verification ---")
    test_model = LoRAInjectedModel(model, n_layers=6, rank=8)
    test_model.to(device)
    diagnose_gradient_flow(test_model, targets_hippo, tokenizer, device=device)
    del test_model

    # ---- Phase 3: Hyperparameters ----
    lora_rank = 8
    n_epochs = 100
    lr = 5e-4

    # ---- Phase 4: Run all conditions ----
    CONDITIONS = {
        # -- Pure hippocampal (no token labels) --
        'per_layer_cosine': {
            'loss_mode': 'per_layer_cosine', 'ce_weight': 0.0,
            'temperature': 1.0,
            'desc': 'Old approach: independent per-layer cosine (broken)',
        },
        'final_residual': {
            'loss_mode': 'final_residual', 'ce_weight': 0.0,
            'temperature': 1.0,
            'desc': 'Variant 1: single final-layer loss, coordinated backprop',
        },
        'logit_distill': {
            'loss_mode': 'logit_distill', 'ce_weight': 0.0,
            'temperature': 2.0,
            'desc': 'Variant 2: KL divergence on hippocampal logits',
        },
        'logit_distill_T4': {
            'loss_mode': 'logit_distill', 'ce_weight': 0.0,
            'temperature': 4.0,
            'desc': 'Variant 2 with higher temperature (softer targets)',
        },
        # -- Baselines and controls --
        'ce_only': {
            'loss_mode': 'ce_only', 'ce_weight': 1.0,
            'temperature': 1.0,
            'desc': 'Standard CE fine-tuning (no hippocampus)',
        },
        'random_logit_distill': {
            'loss_mode': 'logit_distill', 'ce_weight': 0.0,
            'temperature': 2.0,
            'desc': 'Random targets control (logit distillation)',
            'use_random': True,
        },
        # -- Combined --
        'logit_distill_plus_ce': {
            'loss_mode': 'logit_distill', 'ce_weight': 0.5,
            'temperature': 2.0,
            'desc': 'Logit distillation + CE (combined)',
        },
    }

    all_results = {}

    for cname, cfg in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"  {cname}: {cfg['desc']}")
        print(f"{'='*60}")

        lora_model = LoRAInjectedModel(model, n_layers=6, rank=lora_rank)
        lora_model.to(device)

        targets = targets_random if cfg.get('use_random') else targets_hippo

        history = consolidation(
            lora_model, model, targets, tokenizer,
            loss_mode=cfg['loss_mode'],
            n_epochs=n_epochs, lr=lr,
            temperature=cfg['temperature'],
            ce_weight=cfg['ce_weight'],
            device=device, verbose=True)

        eval_results = evaluate(lora_model, tokenizer, FACT_TEST_PAIRS,
                                device=device)
        gen_results = evaluate_generalization(lora_model, model, tokenizer,
                                             device=device)

        all_results[cname] = {
            'eval': eval_results, 'gen': gen_results, 'history': history,
        }

        # Print
        mean_shift = np.mean([r['shift'] for r in eval_results])
        n_improved = sum(1 for r in eval_results if r['shift'] > 0)
        print(f"\n  Results: mean_shift={mean_shift:+.6f}, "
              f"improved={n_improved}/{len(eval_results)}")

        print(f"  {'Pair':<20s} {'Base':>8s} {'Consol':>8s} "
              f"{'Ref':>8s} {'Shift':>10s}")
        print("  " + "-" * 58)
        for r in eval_results:
            print(f"  {r['id']:<20s} {r['baseline']:>8.4f} "
                  f"{r['consolidated']:>8.4f} {r['reference']:>8.4f} "
                  f"{r['shift']:>+10.6f}")

    # ---- Phase 5: Degradation study ----
    print(f"\n{'='*60}")
    print("  DEGRADATION STUDY")
    print(f"  How does consolidation degrade with hippocampal noise?")
    print(f"{'='*60}")

    noise_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    degradation_results = {}

    for nl in noise_levels:
        print(f"\n  --- Noise level: {nl} ---")
        degradation_results[nl] = {}

        noisy_targets = add_noise_to_targets(
            targets_hippo, nl, model, device=device)

        for loss_mode in ('logit_distill', 'final_residual'):
            lora_model = LoRAInjectedModel(model, n_layers=6, rank=lora_rank)
            lora_model.to(device)

            history = consolidation(
                lora_model, model, noisy_targets, tokenizer,
                loss_mode=loss_mode, n_epochs=n_epochs, lr=lr,
                temperature=2.0, ce_weight=0.0,
                device=device, verbose=False)

            eval_results = evaluate(lora_model, tokenizer, FACT_TEST_PAIRS,
                                    device=device)

            mean_shift = np.mean([r['shift'] for r in eval_results])
            n_improved = sum(1 for r in eval_results if r['shift'] > 0)
            print(f"    {loss_mode:<22s}: shift={mean_shift:+.6f}, "
                  f"improved={n_improved}/{len(eval_results)}")

            degradation_results[nl][loss_mode] = {
                'eval': eval_results, 'history': history,
            }

    # ---- Phase 6: Plot ----
    print("\n--- Plotting ---")
    plot_results(all_results, degradation_results,
                 save_path="consolidation_v2_results.png")

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n{'Condition':<26s} {'Shift':>8s} {'Recovery':>10s} "
          f"{'PPL':>6s} {'N+':>6s}")
    print("-" * 60)
    for cname in CONDITIONS:
        evals = all_results[cname]['eval']
        mean_b = np.mean([r['baseline'] for r in evals])
        mean_c = np.mean([r['consolidated'] for r in evals])
        mean_r = np.mean([r['reference'] for r in evals])
        shift = mean_c - mean_b
        gap = mean_r - mean_b
        recv = shift / gap if abs(gap) > 1e-8 else 0.0
        gen = all_results[cname]['gen']
        ppl = np.mean([r['ratio'] for r in gen]) if gen else 1.0
        n_pos = sum(1 for r in evals if r['shift'] > 0)
        print(f"  {cname:<24s} {shift:>+8.5f} {recv:>10.3f} "
              f"{ppl:>6.2f} {n_pos:>3d}/{len(evals)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
