"""
Hippocampal Model v2
====================

Fixes from v1:
1. CA3 uses covariance learning rule: W += (p - a) ⊗ (p - a)
   where a is the mean activity level. This removes the DC bias
   that kills capacity with sparse patterns.
2. Finer capacity testing around the transition point
3. Pattern completion tested at multiple load levels including near-capacity

Theoretical sparse Hopfield capacity with sparsity a:
  C ≈ N / (a * ln(1/a))
  For N=1000, a=0.05: C ≈ 6,667 patterns
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 1. MODEL COMPONENTS
# =============================================================================

class DentateGyrus:
    def __init__(self, d_input, D_output, k_active):
        self.d_input = d_input
        self.D_output = D_output
        self.k_active = k_active

        connectivity_prob = 0.33
        mask = (np.random.rand(D_output, d_input) < connectivity_prob).astype(float)
        weights = np.random.randn(D_output, d_input) * mask
        row_norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-10
        self.W = weights / row_norms

    def forward(self, x):
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        h = x @ self.W.T
        h = np.maximum(h, 0)
        out = np.zeros_like(h)
        for i in range(h.shape[0]):
            if self.k_active < h.shape[1]:
                top_k_idx = np.argpartition(h[i], -self.k_active)[-self.k_active:]
                out[i, top_k_idx] = h[i, top_k_idx]
            else:
                out[i] = h[i]
        if single:
            out = out[0]
        return out


class CA3:
    """
    Autoassociative network with COVARIANCE learning rule.

    Storage: W += (p - a) ⊗ (p - a)  where a = mean activity
    This removes the DC bias that kills capacity for sparse patterns.

    Retrieval: iterative dynamics with k-WTA sparsification.
    """
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.stored_patterns = []
        self.mean_activity = np.zeros(N)

    def store(self, pattern):
        """Store via covariance outer product."""
        # Normalize
        norm = np.linalg.norm(pattern) + 1e-10
        p = pattern / norm

        # Update running mean activity
        self.mean_activity = (self.mean_activity * self.n_stored + p) / (self.n_stored + 1)

        # Covariance rule: subtract mean activity
        p_centered = p - self.mean_activity
        self.W += np.outer(p_centered, p_centered)

        # Zero diagonal
        np.fill_diagonal(self.W, 0)
        self.n_stored += 1
        self.stored_patterns.append(pattern.copy())

    def retrieve(self, cue, n_iterations=15):
        """Pattern completion via iterative dynamics."""
        norm = np.linalg.norm(cue) + 1e-10
        x = cue / norm

        for _ in range(n_iterations):
            h = self.W @ x
            h = np.maximum(h, 0)

            x_new = np.zeros_like(h)
            if self.k_active < len(h):
                top_k_idx = np.argpartition(h, -self.k_active)[-self.k_active:]
                x_new[top_k_idx] = h[top_k_idx]
            else:
                x_new = h

            norm = np.linalg.norm(x_new) + 1e-10
            x_new = x_new / norm

            if np.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x


class Hippocampus:
    def __init__(self, d_ec, D_dg, k_dg, k_ca3=None):
        self.d_ec = d_ec
        if k_ca3 is None:
            k_ca3 = k_dg
        self.dg = DentateGyrus(d_ec, D_dg, k_dg)
        self.ca3 = CA3(D_dg, k_ca3)

    def encode_and_store(self, ec_input):
        dg_output = self.dg.forward(ec_input)
        self.ca3.store(dg_output)
        return dg_output

    def retrieve(self, ec_cue, n_iterations=15):
        dg_cue = self.dg.forward(ec_cue)
        return self.ca3.retrieve(dg_cue, n_iterations)


# =============================================================================
# 2. METRICS
# =============================================================================

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def measure_sparsity(pattern):
    return np.count_nonzero(pattern) / len(pattern)


def pattern_match(retrieved, stored_patterns):
    best_sim = -1
    best_idx = -1
    for i, sp in enumerate(stored_patterns):
        sim = cosine_similarity(retrieved, sp)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_sim, best_idx


def degrade_pattern(pattern, frac_removed):
    degraded = pattern.copy()
    active_idx = np.where(pattern != 0)[0]
    if len(active_idx) == 0:
        return degraded
    n_remove = int(len(active_idx) * frac_removed)
    if n_remove > 0:
        remove_idx = np.random.choice(active_idx, size=n_remove, replace=False)
        degraded[remove_idx] = 0
    return degraded


# =============================================================================
# 3. PARAMETERS
# =============================================================================

d_ec = 100
D_dg = 1000
k_dg = 50
k_ca3 = 50

theoretical_capacity = int(D_dg / (k_dg/D_dg * np.log(D_dg/k_dg)))

print(f"Model: EC({d_ec}) -> DG({D_dg}, k={k_dg}) -> CA3({D_dg}, k={k_ca3})")
print(f"Sparsity: {k_dg/D_dg:.1%}")
print(f"Theoretical sparse capacity: ~{theoretical_capacity} patterns")
print(f"Dense Hopfield limit: {int(0.14 * D_dg)} patterns")
print()

# =============================================================================
# 4. EXPERIMENT 1: Pattern Separation (same as v1 for comparison)
# =============================================================================

print("=" * 60)
print("EXPERIMENT 1: Pattern Separation")
print("=" * 60)

hippo_sep = Hippocampus(d_ec, D_dg, k_dg, k_ca3)

n_separation_tests = 20
input_similarities = np.linspace(0, 0.99, n_separation_tests)
dg_output_sims = []

ref_ec = np.random.randn(d_ec)
ref_ec = ref_ec / np.linalg.norm(ref_ec)

for target_sim in input_similarities:
    sims_dg = []
    for trial in range(30):
        noise = np.random.randn(d_ec)
        noise = noise - np.dot(noise, ref_ec) * ref_ec
        noise = noise / (np.linalg.norm(noise) + 1e-10)
        test_ec = target_sim * ref_ec + np.sqrt(max(0, 1 - target_sim**2)) * noise
        test_ec = test_ec / (np.linalg.norm(test_ec) + 1e-10)
        dg_ref = hippo_sep.dg.forward(ref_ec)
        dg_test = hippo_sep.dg.forward(test_ec)
        sims_dg.append(cosine_similarity(dg_ref, dg_test))
    dg_output_sims.append(float(np.mean(sims_dg)))

print("Input sim -> DG output sim (sampled):")
for i in range(0, len(input_similarities), 4):
    print(f"  {input_similarities[i]:.2f} -> {dg_output_sims[i]:.4f}")

# =============================================================================
# 5. EXPERIMENT 2: Fine-Grained Capacity Testing
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 2: Storage Capacity (fine-grained)")
print("=" * 60)

# Test capacity at many points, especially around expected transitions
capacity_tests = [5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 750,
                  1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]

capacity_results = {}
n_test_per = 50  # test this many patterns at each load level

for n_patterns in capacity_tests:
    print(f"  Testing N={n_patterns}...", end=" ", flush=True)

    hippo_cap = Hippocampus(d_ec, D_dg, k_dg, k_ca3)

    ec_patterns = []
    dg_patterns = []
    for i in range(n_patterns):
        ec = np.random.randn(d_ec)
        ec = ec / np.linalg.norm(ec)
        ec_patterns.append(ec)
        dg_out = hippo_cap.encode_and_store(ec)
        dg_patterns.append(dg_out)

    # Test retrieval: full cue
    test_indices = np.random.choice(n_patterns, size=min(n_test_per, n_patterns), replace=False)
    correct_full = 0
    sims_full = []

    for i in test_indices:
        retrieved = hippo_cap.retrieve(ec_patterns[i])
        sim, best_idx = pattern_match(retrieved, dg_patterns)
        sims_full.append(float(sim))
        if best_idx == i:
            correct_full += 1

    n_tested = len(test_indices)

    # Test retrieval: 50% degraded cue (in DG space)
    correct_degraded = 0
    sims_degraded = []

    for i in test_indices:
        degraded = degrade_pattern(dg_patterns[i], 0.5)
        retrieved = hippo_cap.ca3.retrieve(degraded)
        sim, best_idx = pattern_match(retrieved, dg_patterns)
        sims_degraded.append(float(sim))
        if best_idx == i:
            correct_degraded += 1

    capacity_results[n_patterns] = {
        "full_cue_accuracy": correct_full / n_tested,
        "full_cue_mean_sim": float(np.mean(sims_full)),
        "full_cue_median_sim": float(np.median(sims_full)),
        "full_cue_min_sim": float(np.min(sims_full)),
        "degraded_cue_accuracy": correct_degraded / n_tested,
        "degraded_cue_mean_sim": float(np.mean(sims_degraded)),
        "n_tested": n_tested,
    }

    print(f"full_acc={correct_full/n_tested:.2f}, "
          f"full_sim={np.mean(sims_full):.4f}, "
          f"deg_acc={correct_degraded/n_tested:.2f}")

# =============================================================================
# 6. EXPERIMENT 3: Pattern Completion at Multiple Load Levels
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 3: Pattern Completion vs Degradation at Multiple Loads")
print("=" * 60)

completion_load_levels = [50, 200, 500, 1000, 2000]
degradation_levels = np.linspace(0, 0.95, 20)
completion_results = {}

for n_load in completion_load_levels:
    print(f"\n  Load N={n_load}:")
    hippo_comp = Hippocampus(d_ec, D_dg, k_dg, k_ca3)
    ec_pats = []
    dg_pats = []

    for i in range(n_load):
        ec = np.random.randn(d_ec)
        ec = ec / np.linalg.norm(ec)
        ec_pats.append(ec)
        dg_pats.append(hippo_comp.encode_and_store(ec))

    load_results = {}
    n_test_comp = min(30, n_load)
    test_idx = np.random.choice(n_load, size=n_test_comp, replace=False)

    for deg_level in degradation_levels:
        correct = 0
        sims = []

        for i in test_idx:
            degraded = degrade_pattern(dg_pats[i], deg_level)
            retrieved = hippo_comp.ca3.retrieve(degraded)
            sim, best_idx = pattern_match(retrieved, dg_pats)
            sims.append(float(sim))
            if best_idx == i:
                correct += 1

        load_results[f"{deg_level:.3f}"] = {
            "accuracy": correct / n_test_comp,
            "mean_similarity": float(np.mean(sims)),
        }

        if deg_level < 0.01 or abs(deg_level - 0.5) < 0.03 or deg_level > 0.94:
            print(f"    deg={deg_level:.0%}: acc={correct/n_test_comp:.2f}, "
                  f"sim={np.mean(sims):.4f}")

    completion_results[n_load] = load_results

# =============================================================================
# 7. EXPERIMENT 4: Compare Covariance vs Standard Hopfield Rule
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 4: Covariance Rule vs Standard Hopfield Rule")
print("=" * 60)

class CA3_Standard:
    """Standard Hopfield rule (no mean subtraction) for comparison."""
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.stored_patterns = []

    def store(self, pattern):
        norm = np.linalg.norm(pattern) + 1e-10
        p = pattern / norm
        self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)
        self.n_stored += 1
        self.stored_patterns.append(pattern.copy())

    def retrieve(self, cue, n_iterations=15):
        norm = np.linalg.norm(cue) + 1e-10
        x = cue / norm
        for _ in range(n_iterations):
            h = self.W @ x
            h = np.maximum(h, 0)
            x_new = np.zeros_like(h)
            if self.k_active < len(h):
                top_k_idx = np.argpartition(h, -self.k_active)[-self.k_active:]
                x_new[top_k_idx] = h[top_k_idx]
            else:
                x_new = h
            norm = np.linalg.norm(x_new) + 1e-10
            x_new = x_new / norm
            if np.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x

comparison_tests = [50, 100, 200, 500, 1000, 2000, 5000]
comparison_results = {}

for n_patterns in comparison_tests:
    print(f"  N={n_patterns}...", end=" ", flush=True)

    # Shared DG
    dg_shared = DentateGyrus(d_ec, D_dg, k_dg)
    ca3_cov = CA3(D_dg, k_ca3)
    ca3_std = CA3_Standard(D_dg, k_ca3)

    ec_pats = []
    dg_pats = []
    for i in range(n_patterns):
        ec = np.random.randn(d_ec)
        ec = ec / np.linalg.norm(ec)
        ec_pats.append(ec)
        dg_out = dg_shared.forward(ec)
        dg_pats.append(dg_out)
        ca3_cov.store(dg_out)
        ca3_std.store(dg_out)

    test_idx = np.random.choice(n_patterns, size=min(50, n_patterns), replace=False)

    # Covariance rule
    correct_cov = 0
    sims_cov = []
    for i in test_idx:
        ret = ca3_cov.retrieve(dg_pats[i])
        sim, best_idx = pattern_match(ret, dg_pats)
        sims_cov.append(float(sim))
        if best_idx == i:
            correct_cov += 1

    # Standard rule
    correct_std = 0
    sims_std = []
    for i in test_idx:
        ret = ca3_std.retrieve(dg_pats[i])
        sim, best_idx = pattern_match(ret, dg_pats)
        sims_std.append(float(sim))
        if best_idx == i:
            correct_std += 1

    n_t = len(test_idx)
    comparison_results[n_patterns] = {
        "covariance_accuracy": correct_cov / n_t,
        "covariance_mean_sim": float(np.mean(sims_cov)),
        "standard_accuracy": correct_std / n_t,
        "standard_mean_sim": float(np.mean(sims_std)),
    }

    print(f"cov_acc={correct_cov/n_t:.2f} (sim={np.mean(sims_cov):.4f}), "
          f"std_acc={correct_std/n_t:.2f} (sim={np.mean(sims_std):.4f})")

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================

results = {
    "model_params": {
        "d_ec": d_ec,
        "D_dg": D_dg,
        "k_dg": k_dg,
        "k_ca3": k_ca3,
        "expansion_ratio": D_dg / d_ec,
        "target_sparsity": k_dg / D_dg,
        "theoretical_sparse_capacity": theoretical_capacity,
        "dense_hopfield_limit": int(0.14 * D_dg),
    },
    "pattern_separation": {
        "input_similarities": input_similarities.tolist(),
        "dg_output_similarities": dg_output_sims,
    },
    "storage_capacity": {
        str(k): v for k, v in capacity_results.items()
    },
    "pattern_completion_by_load": {
        str(k): v for k, v in completion_results.items()
    },
    "covariance_vs_standard": {
        str(k): v for k, v in comparison_results.items()
    },
}

with open('hippocampal_model_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# 9. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Hippocampal Model v2: Covariance Rule + Fine-Grained Capacity",
             fontsize=14, fontweight='bold')

# Panel 1: Pattern Separation
ax = axes[0, 0]
ax.plot(input_similarities, dg_output_sims, 'bo-', markersize=4, label='DG output')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No separation')
ax.set_xlabel('Input (EC) cosine similarity')
ax.set_ylabel('Output (DG) cosine similarity')
ax.set_title('Pattern Separation')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Capacity (full cue)
ax = axes[0, 1]
n_pats_list = sorted(capacity_results.keys())
full_accs = [capacity_results[n]["full_cue_accuracy"] for n in n_pats_list]
full_sims = [capacity_results[n]["full_cue_mean_sim"] for n in n_pats_list]
deg_accs = [capacity_results[n]["degraded_cue_accuracy"] for n in n_pats_list]
ax.semilogx(n_pats_list, full_accs, 'bo-', markersize=5, label='Full cue accuracy')
ax.semilogx(n_pats_list, deg_accs, 'rs-', markersize=5, label='50% degraded accuracy')
ax.semilogx(n_pats_list, full_sims, 'g^-', markersize=5, label='Full cue similarity')
ax.axvline(x=0.14*D_dg, color='gray', linestyle='--', alpha=0.5,
           label=f'Dense limit ({int(0.14*D_dg)})')
ax.axvline(x=theoretical_capacity, color='orange', linestyle='--', alpha=0.5,
           label=f'Sparse theory (~{theoretical_capacity})')
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Retrieval quality')
ax.set_title('Storage Capacity (Covariance Rule)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 3: Covariance vs Standard comparison
ax = axes[0, 2]
comp_n = sorted(comparison_results.keys())
cov_accs = [comparison_results[n]["covariance_accuracy"] for n in comp_n]
std_accs = [comparison_results[n]["standard_accuracy"] for n in comp_n]
ax.semilogx(comp_n, cov_accs, 'go-', markersize=6, linewidth=2, label='Covariance rule')
ax.semilogx(comp_n, std_accs, 'r^-', markersize=6, linewidth=2, label='Standard Hopfield')
ax.axvline(x=0.14*D_dg, color='gray', linestyle='--', alpha=0.5, label=f'Dense limit')
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Retrieval accuracy')
ax.set_title('Covariance vs Standard Rule')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 4-6: Pattern completion at different loads
for idx, n_load in enumerate([50, 500, 2000]):
    ax = axes[1, idx]
    if n_load in completion_results:
        load_data = completion_results[n_load]
        degs = sorted([float(k) for k in load_data.keys()])
        accs = [load_data[f"{d:.3f}"]["accuracy"] for d in degs]
        sims = [load_data[f"{d:.3f}"]["mean_similarity"] for d in degs]
        ax.plot(degs, accs, 'bo-', markersize=4, label='Accuracy')
        ax.plot(degs, sims, 'rs-', markersize=4, label='Similarity')
    ax.set_xlabel('Fraction degraded')
    ax.set_ylabel('Retrieval quality')
    ax.set_title(f'Pattern Completion (N={n_load})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('hippocampal_model_v2_results.png', dpi=150, bbox_inches='tight')

print(f"\nResults saved to hippocampal_model_v2_results.json")
print(f"Plot saved to hippocampal_model_v2_results.png")
