"""
Retrieval Specificity Diagnostic
==================================

For each fact-test pair, cue the hippocampal system with the test
sentence and check:
1. Is the retrieved EC most similar to the CORRECT stored fact?
2. How much more similar is it to the correct fact than to other facts?
3. What are the top-k nearest stored patterns to the retrieval?

This tells us whether the hippocampal system is actually recalling
the right memory or just producing a generic pattern.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hippocampal_transformer_backprojection import cosine_sim
from fact_learning_paradigm import (
    FACT_TEST_PAIRS,
    load_model,
    get_hidden_states_and_logits,
    extract_layer_residuals,
    encode_facts,
)
from diverse_corpus_training import build_diverse_system


def diagnose_retrieval(model, tokenizer, system, stored_facts, device='cpu'):
    """
    For each test sentence, retrieve from hippocampus and compare
    against ALL stored fact EC vectors.
    """
    hippo = system['hippo']
    cortical_proj = system['cortical_proj']

    # Collect all stored EC vectors with labels
    fact_ids = list(stored_facts.keys())
    fact_ecs = {fid: stored_facts[fid]['ec_input'] for fid in fact_ids}

    results = []

    for pair in FACT_TEST_PAIRS:
        test_text = pair['test']
        correct_id = pair['id']

        # Get test sentence hidden states
        hidden, _, test_tokens = get_hidden_states_and_logits(
            model, tokenizer, test_text, device)
        last_pos = len(test_tokens) - 1

        # Compute EC input for test sentence
        test_residuals = extract_layer_residuals(hidden, last_pos, device)
        test_ec = cortical_proj.project(test_residuals)

        # Hippocampal retrieval
        retrieved_ec, ca3_cue, ca3_succ = hippo.retrieve_single_ec_deep(test_ec)

        # Compare retrieved EC to ALL stored fact ECs
        similarities = {}
        for fid in fact_ids:
            sim = cosine_sim(retrieved_ec, fact_ecs[fid])
            similarities[fid] = float(sim)

        # Rank by similarity
        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        correct_rank = next(i for i, (fid, _) in enumerate(ranked)
                            if fid == correct_id) + 1
        correct_sim = similarities[correct_id]
        top_sim = ranked[0][1]
        top_id = ranked[0][0]

        # Also check: similarity of test_ec itself to stored ECs
        # (before hippocampal processing)
        raw_similarities = {}
        for fid in fact_ids:
            sim = cosine_sim(test_ec, fact_ecs[fid])
            raw_similarities[fid] = float(sim)
        raw_ranked = sorted(raw_similarities.items(),
                            key=lambda x: x[1], reverse=True)
        raw_correct_rank = next(i for i, (fid, _) in enumerate(raw_ranked)
                                if fid == correct_id) + 1

        # Margin: how much better is the top match than the second?
        if len(ranked) > 1:
            margin = ranked[0][1] - ranked[1][1]
        else:
            margin = 0

        r = {
            'test_id': correct_id,
            'correct_rank_retrieved': correct_rank,
            'correct_sim_retrieved': correct_sim,
            'top_id_retrieved': top_id,
            'top_sim_retrieved': top_sim,
            'margin': margin,
            'correct_rank_raw': raw_correct_rank,
            'all_sims_retrieved': similarities,
            'all_sims_raw': raw_similarities,
            'ranked': ranked,
            'raw_ranked': raw_ranked,
        }
        results.append(r)

    return results


def print_results(results):
    """Print diagnostic results."""
    print(f"\n{'Test Pair':<20s} {'Correct Rank':>13s} {'Correct Sim':>12s} "
          f"{'Top Match':>20s} {'Top Sim':>8s} {'Margin':>8s} "
          f"{'Raw Rank':>9s}")
    print("-" * 95)

    n_correct_top1 = 0
    n_correct_top3 = 0

    for r in results:
        is_top = r['correct_rank_retrieved'] == 1
        if is_top:
            n_correct_top1 += 1
        if r['correct_rank_retrieved'] <= 3:
            n_correct_top3 += 1

        marker = " <-- CORRECT" if is_top else ""
        print(f"{r['test_id']:<20s} {r['correct_rank_retrieved']:>13d} "
              f"{r['correct_sim_retrieved']:>12.4f} "
              f"{r['top_id_retrieved']:>20s} {r['top_sim_retrieved']:>8.4f} "
              f"{r['margin']:>8.4f} {r['correct_rank_raw']:>9d}{marker}")

    print(f"\nCorrect fact ranked #1: {n_correct_top1}/{len(results)}")
    print(f"Correct fact in top 3:  {n_correct_top3}/{len(results)}")

    # Show full ranking for a few interesting cases
    print("\n--- Detailed rankings for selected pairs ---")
    for r in results:
        print(f"\n  {r['test_id']} (correct rank: {r['correct_rank_retrieved']}):")
        print(f"    After hippocampal retrieval:")
        for i, (fid, sim) in enumerate(r['ranked'][:5]):
            marker = " <-- CORRECT" if fid == r['test_id'] else ""
            print(f"      #{i+1}: {fid:<20s} sim={sim:.4f}{marker}")
        print(f"    Raw EC similarity (no hippocampus):")
        for i, (fid, sim) in enumerate(r['raw_ranked'][:5]):
            marker = " <-- CORRECT" if fid == r['test_id'] else ""
            print(f"      #{i+1}: {fid:<20s} sim={sim:.4f}{marker}")


def plot_similarity_matrix(results, save_path):
    """Heatmap: retrieved EC similarity to each stored fact."""
    n = len(results)
    fact_ids = [r['test_id'] for r in results]

    # Build similarity matrix
    sim_matrix = np.zeros((n, n))
    for i, r in enumerate(results):
        for j, fid in enumerate(fact_ids):
            sim_matrix[i, j] = r['all_sims_retrieved'].get(fid, 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Retrieval Specificity Diagnostic", fontsize=14,
                 fontweight='bold')

    # Retrieved EC similarity
    ax = axes[0]
    im = ax.imshow(sim_matrix, cmap='viridis', aspect='auto',
                   vmin=0, vmax=1)
    ax.set_xlabel("Stored Fact")
    ax.set_ylabel("Test Cue")
    ax.set_xticks(range(n))
    ax.set_xticklabels(fact_ids, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(fact_ids, fontsize=6)
    ax.set_title("Retrieved EC vs Stored Facts")
    plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Sim')
    # Mark diagonal
    for i in range(n):
        ax.plot(i, i, 'rx', markersize=8, markeredgewidth=2)

    # Raw EC similarity (before hippocampus)
    raw_matrix = np.zeros((n, n))
    for i, r in enumerate(results):
        for j, fid in enumerate(fact_ids):
            raw_matrix[i, j] = r['all_sims_raw'].get(fid, 0)

    ax = axes[1]
    im = ax.imshow(raw_matrix, cmap='viridis', aspect='auto',
                   vmin=0, vmax=1)
    ax.set_xlabel("Stored Fact")
    ax.set_ylabel("Test Cue")
    ax.set_xticks(range(n))
    ax.set_xticklabels(fact_ids, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(fact_ids, fontsize=6)
    ax.set_title("Raw Test EC vs Stored Facts\n(before hippocampus)")
    plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Sim')
    for i in range(n):
        ax.plot(i, i, 'rx', markersize=8, markeredgewidth=2)

    plt.tight_layout()
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

    gpt2_device = device

    print("=" * 70)
    print("RETRIEVAL SPECIFICITY DIAGNOSTIC")
    print("=" * 70)

    system = build_diverse_system(device)
    model, tokenizer = load_model(gpt2_device)

    print("\n--- Encoding facts ---")
    stored_facts = encode_facts(
        model, tokenizer, system, FACT_TEST_PAIRS,
        device=device, n_repetitions=3)

    print("\n--- Diagnosing retrieval ---")
    results = diagnose_retrieval(
        model, tokenizer, system, stored_facts, device=device)

    print_results(results)

    print("\n--- Plotting ---")
    plot_similarity_matrix(results, "retrieval_specificity.png")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
