"""
Representational Analysis of Hippocampal Injection
====================================================

The fact-learning logit test showed zero recovery. But is the injection
vector even pointing in the right direction?

Key diagnostic: for each fact-test pair, at each layer:
  - baseline_h: hidden state from test sentence alone
  - reference_h: hidden state from fact+test in same context
  - injection: B matrix reconstruction from stored EC vector
  - direction_needed: reference_h - baseline_h (what needs to change)

If cos(injection, direction_needed) > 0, the injection is pointing
toward the correct answer and the problem is magnitude.
If cos(injection, direction_needed) ~ 0, the B matrices are producing
content orthogonal to what's needed.
If cos(injection, direction_needed) < 0, injection actively hurts.

Also measures: how does cos(baseline+alpha*injection, reference) change
with alpha? If it increases then decreases, there's a sweet spot.

Depends on: hippocampal_transformer_backprojection.py,
            fact_learning_paradigm.py
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

from hippocampal_transformer_backprojection import cosine_sim
from fact_learning_paradigm import (
    FACT_TEST_PAIRS,
    load_model,
    get_hidden_states_and_logits,
    extract_layer_residuals,
    build_hippocampal_system,
    encode_facts,
)


def analyze_pair(model, tokenizer, system, pair, stored_facts, device='cpu'):
    """
    For one fact-test pair, compute detailed representational diagnostics
    at every layer.
    """
    cortical_proj = system['cortical_proj']
    backproj = system['backproj']
    hippo = system['hippo']
    n_layers = 6

    fact_text = pair['fact']
    test_text = pair['test']
    reference_text = fact_text + " " + test_text

    # Get hidden states for all conditions
    hidden_baseline, logits_baseline, test_tokens = \
        get_hidden_states_and_logits(model, tokenizer, test_text, device)
    hidden_reference, logits_ref, ref_tokens = \
        get_hidden_states_and_logits(model, tokenizer, reference_text, device)

    last_pos_test = len(test_tokens) - 1
    last_pos_ref = len(ref_tokens) - 1

    # Get stored EC and reconstruct per-layer via B matrices
    stored_ec = stored_facts[pair['id']]['ec_input']
    oracle_layers = backproj.retrieve(stored_ec)

    # Also get hippocampal retrieval version
    test_last_residuals = extract_layer_residuals(
        hidden_baseline, last_pos_test, device)
    test_ec = cortical_proj.project(test_last_residuals)
    retrieved_ec, _, _ = hippo.retrieve_single_ec_deep(test_ec)
    hippo_layers = backproj.retrieve(retrieved_ec)

    results = {
        'id': pair['id'],
        'per_layer': [],
    }

    for l in range(n_layers):
        # Hidden states at last token position for this layer
        # hidden_states index: 0=embedding, 1-6=blocks
        baseline_h = hidden_baseline[l + 1][0, last_pos_test, :].clone().float()
        reference_h = hidden_reference[l + 1][0, last_pos_ref, :].clone().float()

        oracle_inj = oracle_layers[l].clone().float()
        hippo_inj = hippo_layers[l].clone().float()

        # Direction needed: reference - baseline
        direction_needed = reference_h - baseline_h
        dir_norm = float(torch.linalg.norm(direction_needed))

        # Norms
        baseline_norm = float(torch.linalg.norm(baseline_h))
        reference_norm = float(torch.linalg.norm(reference_h))
        oracle_inj_norm = float(torch.linalg.norm(oracle_inj))
        hippo_inj_norm = float(torch.linalg.norm(hippo_inj))

        # Core diagnostic: alignment of injection with needed direction
        oracle_alignment = cosine_sim(oracle_inj, direction_needed)
        hippo_alignment = cosine_sim(hippo_inj, direction_needed)

        # Baseline similarity to reference
        baseline_to_ref = cosine_sim(baseline_h, reference_h)

        # Alpha sweep: does injection improve similarity to reference?
        alphas = np.logspace(-3, 1, 30)  # 0.001 to 10
        oracle_sweep = []
        hippo_sweep = []

        for alpha in alphas:
            # Scale injection to match hidden state norm, then multiply by alpha
            if oracle_inj_norm > 1e-10:
                scaled_oracle = oracle_inj * (baseline_norm / oracle_inj_norm) * alpha
            else:
                scaled_oracle = torch.zeros_like(oracle_inj)
            if hippo_inj_norm > 1e-10:
                scaled_hippo = hippo_inj * (baseline_norm / hippo_inj_norm) * alpha
            else:
                scaled_hippo = torch.zeros_like(hippo_inj)

            injected_oracle = baseline_h + scaled_oracle
            injected_hippo = baseline_h + scaled_hippo

            oracle_sweep.append(cosine_sim(injected_oracle, reference_h))
            hippo_sweep.append(cosine_sim(injected_hippo, reference_h))

        # Find optimal alpha
        best_oracle_idx = int(np.argmax(oracle_sweep))
        best_hippo_idx = int(np.argmax(hippo_sweep))

        # Projection analysis: decompose injection into components
        # parallel and orthogonal to direction_needed
        if dir_norm > 1e-10:
            dir_unit = direction_needed / dir_norm
            oracle_parallel = float(torch.dot(oracle_inj, dir_unit))
            oracle_orthogonal = float(torch.linalg.norm(
                oracle_inj - oracle_parallel * dir_unit))
            hippo_parallel = float(torch.dot(hippo_inj, dir_unit))
            hippo_orthogonal = float(torch.linalg.norm(
                hippo_inj - hippo_parallel * dir_unit))
        else:
            oracle_parallel = oracle_orthogonal = 0.0
            hippo_parallel = hippo_orthogonal = 0.0

        # What fraction of the injection magnitude is in the useful direction?
        if oracle_inj_norm > 1e-10:
            oracle_useful_frac = abs(oracle_parallel) / oracle_inj_norm
        else:
            oracle_useful_frac = 0.0
        if hippo_inj_norm > 1e-10:
            hippo_useful_frac = abs(hippo_parallel) / hippo_inj_norm
        else:
            hippo_useful_frac = 0.0

        layer_result = {
            'layer': l,
            'baseline_norm': baseline_norm,
            'reference_norm': reference_norm,
            'oracle_inj_norm': oracle_inj_norm,
            'hippo_inj_norm': hippo_inj_norm,
            'direction_needed_norm': dir_norm,
            'oracle_alignment': oracle_alignment,
            'hippo_alignment': hippo_alignment,
            'baseline_to_ref': baseline_to_ref,
            'alphas': alphas.tolist(),
            'oracle_sweep': oracle_sweep,
            'hippo_sweep': hippo_sweep,
            'best_oracle_alpha': alphas[best_oracle_idx],
            'best_oracle_sim': oracle_sweep[best_oracle_idx],
            'best_hippo_alpha': alphas[best_hippo_idx],
            'best_hippo_sim': hippo_sweep[best_hippo_idx],
            'oracle_parallel_magnitude': oracle_parallel,
            'oracle_orthogonal_magnitude': oracle_orthogonal,
            'hippo_parallel_magnitude': hippo_parallel,
            'hippo_orthogonal_magnitude': hippo_orthogonal,
            'oracle_useful_fraction': oracle_useful_frac,
            'hippo_useful_fraction': hippo_useful_frac,
        }
        results['per_layer'].append(layer_result)

    return results


def plot_all_results(all_results, save_path):
    """Comprehensive figure."""
    n_pairs = len(all_results)
    n_layers = 6

    fig = plt.figure(figsize=(26, 30))
    gs = GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Representational Analysis of Hippocampal Injection",
                 fontsize=15, fontweight='bold', y=0.99)

    # --- Row 0: Alignment of injection with needed direction ---
    ax = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(all_results):
        oracle_align = [lr['oracle_alignment'] for lr in r['per_layer']]
        ax.plot(range(1, 7), oracle_align, 'o-', alpha=0.6, markersize=4,
                label=r['id'])
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine(injection, direction_needed)")
    ax.set_title("Oracle: Injection Alignment")
    ax.set_xticks(range(1, 7))
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    for i, r in enumerate(all_results):
        hippo_align = [lr['hippo_alignment'] for lr in r['per_layer']]
        ax.plot(range(1, 7), hippo_align, 'o-', alpha=0.6, markersize=4,
                label=r['id'])
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine(injection, direction_needed)")
    ax.set_title("Hippocampal: Injection Alignment")
    ax.set_xticks(range(1, 7))
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    # Mean alignment across pairs
    ax = fig.add_subplot(gs[0, 2])
    mean_oracle = np.mean([[lr['oracle_alignment'] for lr in r['per_layer']]
                           for r in all_results], axis=0)
    mean_hippo = np.mean([[lr['hippo_alignment'] for lr in r['per_layer']]
                          for r in all_results], axis=0)
    std_oracle = np.std([[lr['oracle_alignment'] for lr in r['per_layer']]
                         for r in all_results], axis=0)
    std_hippo = np.std([[lr['hippo_alignment'] for lr in r['per_layer']]
                        for r in all_results], axis=0)
    layers = np.arange(1, 7)
    ax.errorbar(layers - 0.1, mean_oracle, yerr=std_oracle, fmt='o-',
                color='steelblue', label='Oracle', capsize=3)
    ax.errorbar(layers + 0.1, mean_hippo, yerr=std_hippo, fmt='s-',
                color='coral', label='Hippocampal', capsize=3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Mean Injection Alignment (+/- std)")
    ax.set_xticks(range(1, 7))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Row 1: Alpha sweep (oracle) for selected pairs ---
    for col, pair_idx in enumerate([0, 4, 9]):
        if pair_idx >= n_pairs:
            continue
        ax = fig.add_subplot(gs[1, col])
        r = all_results[pair_idx]
        for lr in r['per_layer']:
            baseline_sim = lr['baseline_to_ref']
            ax.plot(lr['alphas'], lr['oracle_sweep'], '-', alpha=0.7,
                    linewidth=1.5, label=f"L{lr['layer']+1}")
            ax.axhline(y=baseline_sim, color='gray', linestyle=':',
                       alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Cos(injected, reference)")
        ax.set_title(f"Oracle Alpha Sweep: {r['id']}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # --- Row 2: Useful fraction and component analysis ---
    ax = fig.add_subplot(gs[2, 0])
    mean_useful_oracle = np.mean(
        [[lr['oracle_useful_fraction'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    mean_useful_hippo = np.mean(
        [[lr['hippo_useful_fraction'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    x = np.arange(1, 7)
    width = 0.35
    ax.bar(x - width/2, mean_useful_oracle, width,
           label='Oracle', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, mean_useful_hippo, width,
           label='Hippocampal', color='coral', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of injection in useful direction")
    ax.set_title("Useful Component Fraction")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Norm comparison
    ax = fig.add_subplot(gs[2, 1])
    mean_dir_norm = np.mean(
        [[lr['direction_needed_norm'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    mean_inj_norm = np.mean(
        [[lr['oracle_inj_norm'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    mean_base_norm = np.mean(
        [[lr['baseline_norm'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    ax.plot(x, mean_base_norm, 'o-', label='Baseline hidden', color='gray')
    ax.plot(x, mean_dir_norm, 's-', label='Direction needed', color='forestgreen')
    ax.plot(x, mean_inj_norm, '^-', label='Oracle injection (raw)', color='steelblue')
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Norm Comparison")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Best achievable sim vs baseline sim
    ax = fig.add_subplot(gs[2, 2])
    mean_baseline_sim = np.mean(
        [[lr['baseline_to_ref'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    mean_best_oracle = np.mean(
        [[lr['best_oracle_sim'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    mean_best_hippo = np.mean(
        [[lr['best_hippo_sim'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    ax.plot(x, mean_baseline_sim, 'o-', label='Baseline', color='gray')
    ax.plot(x, mean_best_oracle, 's-', label='Best oracle', color='steelblue')
    ax.plot(x, mean_best_hippo, '^-', label='Best hippo', color='coral')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Sim to Reference")
    ax.set_title("Best Achievable Sim (optimal alpha)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Row 3: Per-pair alignment heatmaps ---
    pair_ids = [r['id'] for r in all_results]

    ax = fig.add_subplot(gs[3, 0])
    oracle_alignment_matrix = np.array(
        [[lr['oracle_alignment'] for lr in r['per_layer']]
         for r in all_results])
    im = ax.imshow(oracle_alignment_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-0.3, vmax=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fact-Test Pair")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"L{l+1}" for l in range(6)])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_title("Oracle Alignment Heatmap")
    plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine')

    ax = fig.add_subplot(gs[3, 1])
    useful_frac_matrix = np.array(
        [[lr['oracle_useful_fraction'] for lr in r['per_layer']]
         for r in all_results])
    im = ax.imshow(useful_frac_matrix, aspect='auto', cmap='viridis',
                   vmin=0, vmax=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fact-Test Pair")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"L{l+1}" for l in range(6)])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_title("Useful Fraction Heatmap")
    plt.colorbar(im, ax=ax, shrink=0.8, label='Fraction')

    # Improvement heatmap: best_sim - baseline_sim
    ax = fig.add_subplot(gs[3, 2])
    improvement_matrix = np.array(
        [[lr['best_oracle_sim'] - lr['baseline_to_ref']
          for lr in r['per_layer']]
         for r in all_results])
    im = ax.imshow(improvement_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-0.02, vmax=0.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fact-Test Pair")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"L{l+1}" for l in range(6)])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_title("Sim Improvement (best - baseline)")
    plt.colorbar(im, ax=ax, shrink=0.8, label='Delta')

    # --- Row 4: Best alpha distribution ---
    ax = fig.add_subplot(gs[4, 0])
    best_alphas_oracle = np.array(
        [[lr['best_oracle_alpha'] for lr in r['per_layer']]
         for r in all_results])
    for l in range(6):
        ax.hist(best_alphas_oracle[:, l], bins=15, alpha=0.4,
                label=f"L{l+1}")
    ax.set_xlabel("Best Alpha")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Optimal Alpha (Oracle)")
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Parallel vs orthogonal magnitude
    ax = fig.add_subplot(gs[4, 1])
    mean_parallel = np.mean(
        [[abs(lr['oracle_parallel_magnitude']) for lr in r['per_layer']]
         for r in all_results], axis=0)
    mean_orthogonal = np.mean(
        [[lr['oracle_orthogonal_magnitude'] for lr in r['per_layer']]
         for r in all_results], axis=0)
    ax.bar(x - width/2, mean_parallel, width,
           label='Parallel (useful)', color='forestgreen', alpha=0.8)
    ax.bar(x + width/2, mean_orthogonal, width,
           label='Orthogonal (wasted)', color='tomato', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Magnitude")
    ax.set_title("Injection Components: Useful vs Wasted")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # --- Row 5: Summary ---
    ax = fig.add_subplot(gs[4, 2])
    ax.axis('off')
    # Compute key stats
    mean_align_oracle = np.mean(oracle_alignment_matrix)
    mean_align_hippo = np.mean(
        [[lr['hippo_alignment'] for lr in r['per_layer']]
         for r in all_results])
    mean_useful = np.mean(useful_frac_matrix)
    mean_improvement = np.mean(improvement_matrix)
    n_positive_align = np.sum(oracle_alignment_matrix > 0)
    n_total = oracle_alignment_matrix.size
    mean_best_alpha = np.exp(np.mean(np.log(best_alphas_oracle + 1e-10)))

    summary = "KEY FINDINGS\n" + "=" * 40 + "\n\n"
    summary += f"Oracle alignment with\n"
    summary += f"  needed direction:\n"
    summary += f"  Mean: {mean_align_oracle:.4f}\n"
    summary += f"  Positive: {n_positive_align}/{n_total}\n"
    summary += f"    ({100*n_positive_align/n_total:.0f}%)\n\n"
    summary += f"Hippo alignment:\n"
    summary += f"  Mean: {mean_align_hippo:.4f}\n\n"
    summary += f"Useful fraction of\n"
    summary += f"  injection: {mean_useful:.4f}\n\n"
    summary += f"Mean sim improvement\n"
    summary += f"  at optimal alpha:\n"
    summary += f"  {mean_improvement:+.6f}\n\n"
    summary += f"Geometric mean of\n"
    summary += f"  best alpha: {mean_best_alpha:.4f}\n"

    if mean_align_oracle > 0.05:
        summary += "\nVERDICT: Injection points\n"
        summary += "in right direction.\n"
        summary += "Problem is magnitude/format."
    elif mean_align_oracle > -0.05:
        summary += "\nVERDICT: Injection is\n"
        summary += "roughly orthogonal to\n"
        summary += "needed change. B matrices\n"
        summary += "produce irrelevant content."
    else:
        summary += "\nVERDICT: Injection points\n"
        summary += "AWAY from reference.\n"
        summary += "B matrices are harmful."

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Row 5: alpha sweep for hippo retrieval
    for col, pair_idx in enumerate([0, 4, 9]):
        if pair_idx >= n_pairs:
            continue
        ax = fig.add_subplot(gs[5, col])
        r = all_results[pair_idx]
        for lr in r['per_layer']:
            ax.plot(lr['alphas'], lr['hippo_sweep'], '-', alpha=0.7,
                    linewidth=1.5, label=f"L{lr['layer']+1}")
            ax.axhline(y=lr['baseline_to_ref'], color='gray',
                       linestyle=':', alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Cos(injected, reference)")
        ax.set_title(f"Hippo Alpha Sweep: {r['id']}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    gpt2_device = 'cpu'
    if torch.cuda.is_available():
        gpt2_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpt2_device = 'mps'

    print("=" * 70)
    print("REPRESENTATIONAL ANALYSIS OF HIPPOCAMPAL INJECTION")
    print("=" * 70)

    # Build system
    system = build_hippocampal_system(device)
    model, tokenizer = load_model(gpt2_device)

    # Encode facts
    print("\n--- Encoding facts ---")
    stored_facts = encode_facts(
        model, tokenizer, system, FACT_TEST_PAIRS,
        device=device, n_repetitions=3)

    # Analyze each pair
    print("\n--- Analyzing representations ---")
    all_results = []
    for pair in FACT_TEST_PAIRS:
        print(f"\n  {pair['id']}:")
        result = analyze_pair(
            model, tokenizer, system, pair, stored_facts, device=device)

        for lr in result['per_layer']:
            print(f"    L{lr['layer']+1}: alignment={lr['oracle_alignment']:+.4f}  "
                  f"useful_frac={lr['oracle_useful_fraction']:.4f}  "
                  f"best_alpha={lr['best_oracle_alpha']:.4f}  "
                  f"best_sim={lr['best_oracle_sim']:.4f} "
                  f"(baseline={lr['baseline_to_ref']:.4f}, "
                  f"delta={lr['best_oracle_sim']-lr['baseline_to_ref']:+.6f})")

        all_results.append(result)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Layer':<8s} {'Oracle Align':>13s} {'Hippo Align':>12s} "
          f"{'Useful Frac':>12s} {'Best Alpha':>11s} "
          f"{'Baseline Sim':>13s} {'Best Sim':>9s} {'Delta':>10s}")
    print("-" * 92)

    for l in range(6):
        oracle_aligns = [r['per_layer'][l]['oracle_alignment']
                         for r in all_results]
        hippo_aligns = [r['per_layer'][l]['hippo_alignment']
                        for r in all_results]
        useful_fracs = [r['per_layer'][l]['oracle_useful_fraction']
                        for r in all_results]
        best_alphas = [r['per_layer'][l]['best_oracle_alpha']
                       for r in all_results]
        baseline_sims = [r['per_layer'][l]['baseline_to_ref']
                         for r in all_results]
        best_sims = [r['per_layer'][l]['best_oracle_sim']
                     for r in all_results]
        deltas = [b - bl for b, bl in zip(best_sims, baseline_sims)]

        print(f"L{l+1:<6d} {np.mean(oracle_aligns):>+13.4f} "
              f"{np.mean(hippo_aligns):>+12.4f} "
              f"{np.mean(useful_fracs):>12.4f} "
              f"{np.exp(np.mean(np.log(np.array(best_alphas)+1e-10))):>11.4f} "
              f"{np.mean(baseline_sims):>13.4f} "
              f"{np.mean(best_sims):>9.4f} "
              f"{np.mean(deltas):>+10.6f}")

    # Overall verdict
    all_oracle_aligns = [[lr['oracle_alignment'] for lr in r['per_layer']]
                         for r in all_results]
    mean_align = np.mean(all_oracle_aligns)
    n_positive = np.sum(np.array(all_oracle_aligns) > 0)
    n_total = np.array(all_oracle_aligns).size

    all_deltas = [[lr['best_oracle_sim'] - lr['baseline_to_ref']
                   for lr in r['per_layer']]
                  for r in all_results]
    mean_delta = np.mean(all_deltas)
    n_improved = np.sum(np.array(all_deltas) > 0)

    print(f"\nOverall oracle alignment: {mean_align:+.4f}")
    print(f"Positive alignment: {n_positive}/{n_total} "
          f"({100*n_positive/n_total:.1f}%)")
    print(f"Mean sim improvement at optimal alpha: {mean_delta:+.6f}")
    print(f"Layer-pairs improved: {n_improved}/{n_total} "
          f"({100*n_improved/n_total:.1f}%)")

    # Plot
    print("\n--- Plotting ---")
    plot_all_results(all_results, "repr_analysis_results.png")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
