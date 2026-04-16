"""
Mean-Centered Projection with Pseudoinverse Reconstruction
============================================================

Key insight: the shared structural component (declarative sentence,
final token position, similar length) dominates the raw residuals,
making all fact EC vectors look the same. Mean-centering removes this
shared component so A projects only content-specific deviations.

Architecture:
  - Pass 1: compute per-layer running mean across training corpus
  - Pass 2: project (h - mean) through A, store in hippocampus
  - B = A^+ (pseudoinverse), no Hebbian training needed
  - Injection: substitutive (replace deviation-subspace content)

  h_new = h_current + A^+ @ (ec_stored - A @ (h_current - mean))

This swaps the stored deviation into the current hidden state within
A's row space, leaving everything orthogonal to A untouched.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hippocampal_transformer_backprojection import (
    cosine_sim,
    HippocampalSystemTemporal,
    encode_phase_a,
)
from fact_learning_paradigm import (
    FACT_TEST_PAIRS,
    load_model,
    get_hidden_states_and_logits,
    extract_layer_residuals,
    get_token_probs,
    inject_all_layers,
)
from diverse_corpus_training import DIVERSE_CORPUS


# =============================================================================
# MEAN-CENTERED PROJECTION SYSTEM
# =============================================================================

class MeanCenteredProjection:
    """
    Layer-specific A matrices that project (h - mean) into EC subspace.
    B = A^+ (pseudoinverse). No training needed for B.
    """
    def __init__(self, n_layers, d_model, r_per_layer,
                 device='cpu', dtype=torch.float32):
        self.n_layers = n_layers
        self.d_model = d_model
        self.r_per_layer = r_per_layer
        self.d_ec = n_layers * r_per_layer
        self.device = device
        self.dtype = dtype

        # Random A matrices (one per layer)
        self.A = []
        self.A_pinv = []
        for l in range(n_layers):
            A_l = torch.randn(r_per_layer, d_model, device=device, dtype=dtype)
            # Row-normalize for stability
            row_norms = torch.linalg.norm(A_l, dim=1, keepdim=True) + 1e-10
            A_l = A_l / row_norms
            self.A.append(A_l)
            # Pseudoinverse: A^+ = A^T (A A^T)^{-1}
            # For row-normalized A with r << d, A A^T is well-conditioned
            A_pinv = torch.linalg.pinv(A_l)
            self.A_pinv.append(A_pinv)

        # Per-layer running means (initialized to zero, updated in pass 1)
        self.means = [torch.zeros(d_model, device=device, dtype=dtype)
                      for _ in range(n_layers)]
        self.mean_count = 0
        self.means_frozen = False

    def update_means(self, layer_residuals):
        """Pass 1: accumulate running mean per layer."""
        assert not self.means_frozen, "Means are frozen, cannot update"
        self.mean_count += 1
        for l in range(self.n_layers):
            delta = layer_residuals[l] - self.means[l]
            self.means[l] += delta / self.mean_count

    def freeze_means(self):
        """Lock means after pass 1."""
        self.means_frozen = True
        print(f"  Means frozen after {self.mean_count} samples")
        for l in range(self.n_layers):
            print(f"    L{l+1} mean norm: {float(torch.linalg.norm(self.means[l])):.2f}")

    def project(self, layer_residuals):
        """
        Project (h - mean) through A for each layer, concatenate.
        Returns EC input vector of dimension d_ec.
        """
        projections = []
        for l in range(self.n_layers):
            deviation = layer_residuals[l] - self.means[l]
            proj_l = self.A[l] @ deviation
            projections.append(proj_l)
        return torch.cat(projections, dim=0)

    def reconstruct_deviations(self, ec_input):
        """
        Reconstruct per-layer deviations from EC input using A^+.
        Returns list of n_layers tensors, each (d_model,).
        """
        deviations = []
        for l in range(self.n_layers):
            start = l * self.r_per_layer
            end = (l + 1) * self.r_per_layer
            ec_l = ec_input[start:end]
            dev_l = self.A_pinv[l] @ ec_l
            deviations.append(dev_l)
        return deviations

    def substitutive_injection(self, h_current_layers, ec_stored):
        """
        Replace the deviation-subspace content of h_current with stored content.

        h_new_l = h_current_l + A^+ @ (ec_stored_l - A @ (h_current_l - mean_l))

        Returns list of INJECTION DELTAS (what to add to h_current at each layer).
        """
        deltas = []
        for l in range(self.n_layers):
            start = l * self.r_per_layer
            end = (l + 1) * self.r_per_layer
            ec_stored_l = ec_stored[start:end]

            current_deviation = h_current_layers[l] - self.means[l]
            current_ec_l = self.A[l] @ current_deviation

            ec_diff = ec_stored_l - current_ec_l
            delta_l = self.A_pinv[l] @ ec_diff
            deltas.append(delta_l)
        return deltas

    def additive_injection(self, ec_stored):
        """
        Simply reconstruct the stored deviation and return as injection.
        """
        return self.reconstruct_deviations(ec_stored)


# =============================================================================
# SYSTEM BUILDING
# =============================================================================

def build_system(device, dtype=torch.float32):
    """Build everything with mean-centered projections."""
    n_layers = 6
    d_model = 768
    r_per_layer = 128
    d_ec = n_layers * r_per_layer

    gpt2_device = 'cpu'
    if torch.cuda.is_available():
        gpt2_device = 'cuda'

    from hippocampal_transformer_backprojection import load_gpt2_and_extract

    print("\n--- Extracting diverse corpus ---")
    sequences_tokens, sequences_residuals, tokenizer = load_gpt2_and_extract(
        DIVERSE_CORPUS, seq_length=64, n_sequences=20, device=gpt2_device)

    if device != torch.device(gpt2_device):
        for seq_idx in range(len(sequences_residuals)):
            for t in range(len(sequences_residuals[seq_idx])):
                for l in range(n_layers):
                    sequences_residuals[seq_idx][t][l] = \
                        sequences_residuals[seq_idx][t][l].to(device)

    proj = MeanCenteredProjection(
        n_layers, d_model, r_per_layer, device=device, dtype=dtype)

    # --- Pass 1: compute means ---
    print("\n--- Pass 1: Computing per-layer means ---")
    n_tokens_seen = 0
    for seq_residuals in sequences_residuals:
        for t, layer_residuals in enumerate(seq_residuals):
            proj.update_means(layer_residuals)
            n_tokens_seen += 1
    proj.freeze_means()
    print(f"  Computed means from {n_tokens_seen} tokens")

    # --- Verify mean-centering helps ---
    print("\n--- Verifying mean-centering effect ---")
    # Collect a sample of EC vectors with and without centering
    sample_ecs_centered = []
    sample_ecs_raw = []
    for seq_residuals in sequences_residuals[:3]:
        for t in range(0, len(seq_residuals), 4):
            layer_res = seq_residuals[t]
            # Centered
            ec_c = proj.project(layer_res)
            sample_ecs_centered.append(ec_c)
            # Raw (no centering)
            raw_proj = []
            for l in range(n_layers):
                raw_proj.append(proj.A[l] @ layer_res[l])
            ec_r = torch.cat(raw_proj, dim=0)
            sample_ecs_raw.append(ec_r)

    # Mean pairwise similarity
    n_samples = len(sample_ecs_centered)
    centered_sims = []
    raw_sims = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            centered_sims.append(cosine_sim(
                sample_ecs_centered[i], sample_ecs_centered[j]))
            raw_sims.append(cosine_sim(
                sample_ecs_raw[i], sample_ecs_raw[j]))
    print(f"  Mean pairwise similarity (raw):      {np.mean(raw_sims):.4f}")
    print(f"  Mean pairwise similarity (centered): {np.mean(centered_sims):.4f}")
    print(f"  Spread increase: {np.std(centered_sims)/np.std(raw_sims):.2f}x")

    # No hippocampal training on base corpus.
    # Means are computed; projection system is ready.
    # Hippocampal system will be created fresh for fact encoding only.

    return {
        'proj': proj,
        'd_ec': d_ec,
        'n_layers': n_layers,
    }


def make_fresh_hippo(d_ec, device='cpu', dtype=torch.float32):
    """Create a fresh hippocampal system with no stored patterns."""
    hippo_kwargs = {
        "d_ec": d_ec, "D_dg": d_ec * 4, "N_ca3": d_ec, "N_ca1": d_ec,
        "k_ca3": 50, "N_sub": d_ec,
        "ca3_lr": 1.0, "direct_lr": 0.3, "direct_decay": 0.998,
        "ca3_retrieval_iterations": 5,
        "ec_sup_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "pyr_to_stel_strength": 0.3,
        },
        "dg_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "noise_scale": 0.0,
        },
        "ca1_params": {
            "lr": 0.3, "plateau_threshold": 0.7, "plateau_sharpness": 20.0,
            "weight_decay": 1.0, "div_norm_sigma": 0.1,
            "connectivity_prob": 0.33, "ltd_rate": 0.05,
            "ltd_ca3_threshold": 0.0, "sigma_inh": 25, "gamma_inh": 4.0,
            "n_inh_steps": 5, "E_inh": -0.4,
        },
        "sub_params": {"lr": 0.05, "ltd_rate": 0.05, "connectivity_prob": 0.33},
        "ec_deep_params": {"lr": 1.0, "weight_decay": 0.998},
        "direct_decoder_lr": 0.3,
    }
    return HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)


# =============================================================================
# FACT ENCODING
# =============================================================================

def encode_facts_centered(model, tokenizer, system, pairs,
                          device='cpu', n_repetitions=3):
    """
    Encode facts using mean-centered projections into a FRESH hippocampal system.
    Randomizes encoding order across repetitions to avoid recency bias.

    Stores EC from the pre-answer position (last token of the test prefix)
    rather than the last token of the full fact sentence, so the stored
    representation carries next-token predictive signal for the answer.
    """
    proj = system['proj']
    d_ec = system['d_ec']

    # Create fresh hippocampal system -- no base corpus contamination
    print("  Creating fresh hippocampal system (facts only)")
    hippo = make_fresh_hippo(d_ec, device=device)
    system['hippo'] = hippo  # attach to system for downstream use

    stored_facts = {}
    # Pre-extract all fact hidden states (only need to do this once)
    fact_hidden = {}
    for pair in pairs:
        hidden, _, token_ids = get_hidden_states_and_logits(
            model, tokenizer, pair['fact'], device)
        test_token_ids = tokenizer.encode(pair['test'])
        # The pre-answer position: last token of the test prefix within
        # the full fact's forward pass. This is where the model is primed
        # to predict the answer token.
        store_pos = len(test_token_ids) - 1
        fact_hidden[pair['id']] = {
            'hidden': hidden,
            'token_ids': token_ids,
            'store_pos': store_pos,
        }
        print(f"    {pair['id']:<20s} fact_len={len(token_ids)}  "
              f"store_pos={store_pos}  "
              f"token='{tokenizer.decode([token_ids[store_pos]])}'")

    for rep in range(n_repetitions):
        # Randomize encoding order each repetition
        order = list(range(len(pairs)))
        # np.random.shuffle(order)
        print(f"  Fact encoding rep {rep + 1}/{n_repetitions}  "
              f"order: {[pairs[i]['id'][:8] for i in order]}")

        for idx in order:
            pair = pairs[idx]
            fh = fact_hidden[pair['id']]
            hidden = fh['hidden']
            token_ids = fh['token_ids']
            store_pos = fh['store_pos']

            hippo.begin_sequence()
            target_ec = None
            for t in range(len(token_ids)):
                layer_res = extract_layer_residuals(hidden, t, device)
                ec_input = proj.project(layer_res)
                hippo.encode_single(ec_input)
                if t == store_pos:
                    target_ec = ec_input.clone()
            hippo.end_sequence()

            if rep == 0:
                store_layer_res = extract_layer_residuals(
                    hidden, store_pos, device)
                stored_facts[pair['id']] = {
                    'ec_input': target_ec,
                    'layer_residuals': store_layer_res,
                    'n_tokens': len(token_ids),
                    'store_pos': store_pos,
                }

    # Comprehensive diagnostic: check encode-decode for ALL stored facts
    print("\n  Encode-decode diagnostic (all facts):")
    for pair in pairs:
        stored_ec = stored_facts[pair['id']]['ec_input']
        stel, _ = hippo.ec_sup.forward(stored_ec)
        dg = hippo._mossy_fiber(hippo.dg.forward(stel))
        ca3 = hippo.ca3.retrieve(dg, hippo.ca3_retrieval_iterations)
        dec = hippo.direct_decoder.retrieve(ca3)
        sim = cosine_sim(dec, stored_ec)
        print(f"    {pair['id']:<20s} sim={sim:.4f}")

    return stored_facts


# =============================================================================
# RETRIEVAL SPECIFICITY TEST
# =============================================================================

def test_retrieval_specificity(model, tokenizer, system, stored_facts,
                               device='cpu'):
    """Check if correct fact is retrieved for each test sentence."""
    hippo = system['hippo']
    proj = system['proj']

    fact_ids = list(stored_facts.keys())
    fact_ecs = {fid: stored_facts[fid]['ec_input'] for fid in fact_ids}

    results = []
    for pair in FACT_TEST_PAIRS:
        hidden, _, test_tokens = get_hidden_states_and_logits(
            model, tokenizer, pair['test'], device)
        last_pos = len(test_tokens) - 1
        test_res = extract_layer_residuals(hidden, last_pos, device)
        test_ec = proj.project(test_res)

        # Hippocampal retrieval
        retrieved_ec, _, _ = hippo.retrieve_single_ec_deep(test_ec)

        # Compare to all stored facts
        sims = {fid: float(cosine_sim(retrieved_ec, fact_ecs[fid]))
                for fid in fact_ids}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        correct_rank = next(i for i, (fid, _) in enumerate(ranked)
                            if fid == pair['id']) + 1

        # Raw EC similarity (before hippocampus)
        raw_sims = {fid: float(cosine_sim(test_ec, fact_ecs[fid]))
                    for fid in fact_ids}
        raw_ranked = sorted(raw_sims.items(), key=lambda x: x[1], reverse=True)
        raw_correct_rank = next(i for i, (fid, _) in enumerate(raw_ranked)
                                if fid == pair['id']) + 1

        results.append({
            'id': pair['id'],
            'correct_rank': correct_rank,
            'correct_sim': sims[pair['id']],
            'top_id': ranked[0][0],
            'top_sim': ranked[0][1],
            'margin': ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0,
            'raw_rank': raw_correct_rank,
            'raw_correct_sim': raw_sims[pair['id']],
            'raw_top_id': raw_ranked[0][0],
            'raw_top_sim': raw_ranked[0][1],
            'raw_margin': raw_ranked[0][1] - raw_ranked[1][1] if len(raw_ranked) > 1 else 0,
            'ranked': ranked[:5],
            'raw_ranked': raw_ranked[:5],
        })

    return results


# =============================================================================
# FACT-LEARNING EVALUATION
# =============================================================================

def inject_substitutive(model, tokenizer, text, proj, ec_stored,
                        device='cpu'):
    """
    Run model with substitutive injection at every layer.
    Uses hooks to apply proj.substitutive_injection at each block.
    """
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], device=device)
    last_pos = len(token_ids) - 1

    # We need h_current at each layer to compute the substitution.
    # Strategy: first forward pass to get all hidden states,
    # compute deltas, then second forward pass with hooks.
    with torch.no_grad():
        outputs_first = model(input_ids, output_hidden_states=True)

    # Extract current layer residuals at last position
    h_current_layers = []
    for l in range(1, 7):
        h = outputs_first.hidden_states[l][0, last_pos, :].clone().float().to(
            proj.device)
        h_current_layers.append(h)

    # Compute substitution deltas
    deltas = proj.substitutive_injection(h_current_layers, ec_stored)

    # Second forward pass with hooks
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0]
            hidden[0, last_pos, :] = (
                hidden[0, last_pos, :] + deltas[layer_idx].to(hidden.device))
            return (hidden,) + output[1:]
        return hook_fn

    for l in range(6):
        h = model.transformer.h[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    for h in hooks:
        h.remove()

    return outputs.logits[0, last_pos, :], outputs.hidden_states


def evaluate_fact_learning(model, tokenizer, system, stored_facts,
                           device='cpu'):
    """Full evaluation: retrieval specificity + logit shift + representation."""
    proj = system['proj']
    hippo = system['hippo']
    d_ec = system['d_ec']
    n_layers = 6

    all_results = []

    for pair in FACT_TEST_PAIRS:
        test_text = pair['test']
        fact_text = pair['fact']
        expected = pair['expected']
        default = pair['default']
        all_tokens = expected + default
        reference_text = fact_text + " " + test_text

        # --- Baseline ---
        hidden_base, logits_base, test_tokens = \
            get_hidden_states_and_logits(model, tokenizer, test_text, device)
        last_pos = len(test_tokens) - 1
        baseline_probs = get_token_probs(logits_base, last_pos,
                                         tokenizer, all_tokens)
        p_base = sum(baseline_probs.get(t, 0) for t in expected)

        probs_base = F.softmax(logits_base[0, last_pos, :], dim=-1)
        top5_v, top5_i = torch.topk(probs_base, 5)
        base_top5 = [(tokenizer.decode([tid.item()]), float(tv))
                     for tid, tv in zip(top5_i, top5_v)]

        # --- Reference ---
        hidden_ref, logits_ref, ref_tokens = \
            get_hidden_states_and_logits(model, tokenizer, reference_text, device)
        ref_last = len(ref_tokens) - 1
        ref_probs = get_token_probs(logits_ref, ref_last, tokenizer, all_tokens)
        p_ref = sum(ref_probs.get(t, 0) for t in expected)

        probs_ref = F.softmax(logits_ref[0, ref_last, :], dim=-1)
        top5_v, top5_i = torch.topk(probs_ref, 5)
        ref_top5 = [(tokenizer.decode([tid.item()]), float(tv))
                    for tid, tv in zip(top5_i, top5_v)]

        # --- Oracle substitutive injection ---
        stored_ec = stored_facts[pair['id']]['ec_input']
        oracle_logits, oracle_hidden = inject_substitutive(
            model, tokenizer, test_text, proj, stored_ec, device=device)
        oracle_probs = F.softmax(oracle_logits, dim=-1)
        p_oracle = sum(float(oracle_probs[tokenizer.encode(t)[0]])
                       for t in expected if len(tokenizer.encode(t)) >= 1)

        top5_v, top5_i = torch.topk(oracle_probs, 5)
        oracle_top5 = [(tokenizer.decode([tid.item()]), float(tv))
                       for tid, tv in zip(top5_i, top5_v)]

        # --- Hippocampal retrieval + substitutive injection ---
        test_res = extract_layer_residuals(hidden_base, last_pos, device)
        test_ec = proj.project(test_res)
        retrieved_ec, _, _ = hippo.retrieve_single_ec_deep(test_ec)
        retrieval_sim = float(cosine_sim(retrieved_ec, stored_ec))

        hippo_logits, hippo_hidden = inject_substitutive(
            model, tokenizer, test_text, proj, retrieved_ec, device=device)
        hippo_probs = F.softmax(hippo_logits, dim=-1)
        p_hippo = sum(float(hippo_probs[tokenizer.encode(t)[0]])
                      for t in expected if len(tokenizer.encode(t)) >= 1)

        # --- Representational analysis per layer ---
        per_layer = []
        for l in range(n_layers):
            base_h = hidden_base[l + 1][0, last_pos, :].clone().float()
            ref_h = hidden_ref[l + 1][0, ref_last, :].clone().float()
            oracle_h = oracle_hidden[l + 1][0, last_pos, :].clone().float()

            direction_needed = ref_h - base_h
            oracle_delta = oracle_h - base_h

            alignment = cosine_sim(oracle_delta, direction_needed)
            base_to_ref = cosine_sim(base_h, ref_h)
            oracle_to_ref = cosine_sim(oracle_h, ref_h)

            per_layer.append({
                'alignment': alignment,
                'base_to_ref': base_to_ref,
                'oracle_to_ref': oracle_to_ref,
                'delta_sim': oracle_to_ref - base_to_ref,
            })

        r = {
            'id': pair['id'],
            'p_base': p_base,
            'p_ref': p_ref,
            'p_oracle': p_oracle,
            'p_hippo': p_hippo,
            'retrieval_sim': retrieval_sim,
            'base_top5': base_top5,
            'ref_top5': ref_top5,
            'oracle_top5': oracle_top5,
            'per_layer': per_layer,
        }
        all_results.append(r)

    return all_results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(eval_results, retrieval_results, save_path):
    n_pairs = len(eval_results)
    fig = plt.figure(figsize=(24, 22))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Mean-Centered Projection + Pseudoinverse + Substitution",
                 fontsize=14, fontweight='bold', y=0.99)

    pair_ids = [r['id'] for r in eval_results]
    x = np.arange(n_pairs)

    # --- Row 0: P(expected) ---
    ax = fig.add_subplot(gs[0, :2])
    width = 0.2
    ax.bar(x - 1.5*width, [r['p_base'] for r in eval_results], width,
           label='Baseline', color='gray')
    ax.bar(x - 0.5*width, [r['p_oracle'] for r in eval_results], width,
           label='Oracle subst.', color='steelblue')
    ax.bar(x + 0.5*width, [r['p_hippo'] for r in eval_results], width,
           label='Hippo subst.', color='coral')
    ax.bar(x + 1.5*width, [r['p_ref'] for r in eval_results], width,
           label='Reference', color='forestgreen')
    ax.set_ylabel("P(expected)")
    ax.set_title("Token Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_ids, fontsize=7, rotation=30, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # Retrieval specificity
    ax = fig.add_subplot(gs[0, 2])
    ranks = [r['correct_rank'] for r in retrieval_results]
    colors = ['forestgreen' if r == 1 else 'orange' if r <= 3 else 'tomato'
              for r in ranks]
    ax.barh(range(n_pairs), ranks, color=colors, alpha=0.8)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_xlabel("Rank of Correct Fact")
    ax.set_title("Retrieval Specificity")
    ax.axvline(x=1.5, color='green', linestyle='--', alpha=0.5)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # --- Row 1: Alignment heatmap ---
    ax = fig.add_subplot(gs[1, 0])
    align_matrix = np.array(
        [[lr['alignment'] for lr in r['per_layer']] for r in eval_results])
    im = ax.imshow(align_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"L{l+1}" for l in range(6)])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_title("Injection Alignment")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Delta sim heatmap
    ax = fig.add_subplot(gs[1, 1])
    delta_matrix = np.array(
        [[lr['delta_sim'] for lr in r['per_layer']] for r in eval_results])
    vmax = max(0.01, np.max(np.abs(delta_matrix)))
    im = ax.imshow(delta_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"L{l+1}" for l in range(6)])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_title("Sim to Reference: Delta")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Mean alignment by layer
    ax = fig.add_subplot(gs[1, 2])
    mean_align = np.mean(align_matrix, axis=0)
    std_align = np.std(align_matrix, axis=0)
    ax.bar(range(1, 7), mean_align, yerr=std_align,
           color='steelblue', alpha=0.8, capsize=3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Mean Alignment by Layer")
    ax.set_xticks(range(1, 7))
    ax.grid(True, alpha=0.3, axis='y')

    # --- Row 2: Top predictions ---
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    text = "Top-5 Predictions\n" + "=" * 90 + "\n\n"
    for r in eval_results:
        text += f"{r['id']}:\n"
        text += f"  Baseline:  {', '.join(f'{t}({p:.3f})' for t,p in r['base_top5'][:4])}\n"
        text += f"  Oracle:    {', '.join(f'{t}({p:.3f})' for t,p in r['oracle_top5'][:4])}\n"
        text += f"  Reference: {', '.join(f'{t}({p:.3f})' for t,p in r['ref_top5'][:4])}\n\n"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Row 3: Summary ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    mean_base = np.mean([r['p_base'] for r in eval_results])
    mean_oracle = np.mean([r['p_oracle'] for r in eval_results])
    mean_hippo = np.mean([r['p_hippo'] for r in eval_results])
    mean_ref = np.mean([r['p_ref'] for r in eval_results])
    n_rank1 = sum(1 for r in retrieval_results if r['correct_rank'] == 1)
    n_top3 = sum(1 for r in retrieval_results if r['correct_rank'] <= 3)
    mean_align_all = np.mean(align_matrix)
    n_pos = np.sum(align_matrix > 0)

    if mean_ref - mean_base > 1e-8:
        rec_o = (mean_oracle - mean_base) / (mean_ref - mean_base)
        rec_h = (mean_hippo - mean_base) / (mean_ref - mean_base)
    else:
        rec_o = rec_h = 0

    summary = "SUMMARY\n" + "=" * 55 + "\n\n"
    summary += f"Retrieval: {n_rank1}/10 rank-1, {n_top3}/10 top-3\n\n"
    summary += f"Mean P(expected):\n"
    summary += f"  Baseline:  {mean_base:.6f}\n"
    summary += f"  Oracle:    {mean_oracle:.6f}\n"
    summary += f"  Hippo:     {mean_hippo:.6f}\n"
    summary += f"  Reference: {mean_ref:.6f}\n\n"
    summary += f"Recovery (oracle):  {rec_o:+.4f}\n"
    summary += f"Recovery (hippo):   {rec_h:+.4f}\n\n"
    summary += f"Alignment: mean={mean_align_all:+.4f}, "
    summary += f"positive={n_pos}/{align_matrix.size}\n"
    summary += f"Mean delta sim: {np.mean(delta_matrix):+.6f}\n"

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
    print("MEAN-CENTERED PROJECTION + PSEUDOINVERSE + SUBSTITUTION")
    print("=" * 70)

    system = build_system(device)
    model, tokenizer = load_model(device)

    print("\n--- Encoding facts ---")
    stored_facts = encode_facts_centered(
        model, tokenizer, system, FACT_TEST_PAIRS,
        device=device, n_repetitions=3)

    print("\n--- Retrieval specificity (hippocampal) ---")
    retrieval_results = test_retrieval_specificity(
        model, tokenizer, system, stored_facts, device=device)

    n_rank1 = sum(1 for r in retrieval_results if r['correct_rank'] == 1)
    n_top3 = sum(1 for r in retrieval_results if r['correct_rank'] <= 3)
    print(f"\n  Correct rank-1: {n_rank1}/10")
    print(f"  Correct top-3:  {n_top3}/10")

    for r in retrieval_results:
        marker = " <-- CORRECT" if r['correct_rank'] == 1 else ""
        print(f"  {r['id']:<20s} rank={r['correct_rank']:>2d}  "
              f"sim={r['correct_sim']:.4f}  "
              f"top={r['top_id']} ({r['top_sim']:.4f})  "
              f"margin={r['margin']:.4f}  "
              f"raw_rank={r['raw_rank']}{marker}")

    print("\n--- Retrieval specificity (oracle: raw EC, no hippocampus) ---")
    raw_rank1 = sum(1 for r in retrieval_results if r['raw_rank'] == 1)
    raw_top3 = sum(1 for r in retrieval_results if r['raw_rank'] <= 3)
    print(f"\n  Correct rank-1: {raw_rank1}/10")
    print(f"  Correct top-3:  {raw_top3}/10")

    for r in retrieval_results:
        marker = " <-- CORRECT" if r['raw_rank'] == 1 else ""
        print(f"  {r['id']:<20s} rank={r['raw_rank']:>2d}  "
              f"sim={r['raw_correct_sim']:.4f}  "
              f"top={r['raw_top_id']} ({r['raw_top_sim']:.4f})  "
              f"margin={r['raw_margin']:.4f}{marker}")

    print("\n--- Fact learning evaluation ---")
    eval_results = evaluate_fact_learning(
        model, tokenizer, system, stored_facts, device=device)

    print(f"\n{'Pair':<20s} {'Baseline':>10s} {'Oracle':>10s} "
          f"{'Hippo':>10s} {'Ref':>10s} {'Ret.Sim':>8s}")
    print("-" * 72)
    for r in eval_results:
        print(f"{r['id']:<20s} {r['p_base']:>10.6f} {r['p_oracle']:>10.6f} "
              f"{r['p_hippo']:>10.6f} {r['p_ref']:>10.6f} "
              f"{r['retrieval_sim']:>8.4f}")

    mean_b = np.mean([r['p_base'] for r in eval_results])
    mean_o = np.mean([r['p_oracle'] for r in eval_results])
    mean_h = np.mean([r['p_hippo'] for r in eval_results])
    mean_r = np.mean([r['p_ref'] for r in eval_results])
    print("-" * 72)
    print(f"{'MEAN':<20s} {mean_b:>10.6f} {mean_o:>10.6f} "
          f"{mean_h:>10.6f} {mean_r:>10.6f}")

    if mean_r - mean_b > 1e-8:
        print(f"\nRecovery (oracle): {(mean_o-mean_b)/(mean_r-mean_b):+.4f}")
        print(f"Recovery (hippo):  {(mean_h-mean_b)/(mean_r-mean_b):+.4f}")

    print(f"\nAlignment by layer:")
    for l in range(6):
        aligns = [r['per_layer'][l]['alignment'] for r in eval_results]
        deltas = [r['per_layer'][l]['delta_sim'] for r in eval_results]
        print(f"  L{l+1}: align={np.mean(aligns):+.4f}  "
              f"delta_sim={np.mean(deltas):+.6f}")

    print("\n--- Plotting ---")
    plot_results(eval_results, retrieval_results, "mean_centered_results.png")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()