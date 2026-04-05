"""
Simplified Hippocampal Model
============================

Architecture:
  Entorhinal Cortex (EC) input → Dentate Gyrus (DG) → CA3 → retrieval

DG: Random sparse expansion
  - Projects d-dim EC input to D-dim space (D >> d) via random weights
  - Applies k-winners-take-all sparsification
  - This orthogonalizes similar inputs (pattern separation)

CA3: Autoassociative network
  - Stores sparse DG patterns via Hebbian outer products
  - Retrieves via single-step or iterative pattern completion
  - Recurrent collaterals enable attractor dynamics

Validation metrics (with literature reference values):
  1. Pattern separation: input similarity vs output similarity curve
     - DG should decorrelate similar inputs dramatically
     - CA3 shows threshold behavior (completion or separation)
  2. Pattern completion: retrieval accuracy from partial cues
     - Should work well up to ~50-70% degradation
  3. Storage capacity: retrieval quality vs number of stored patterns
     - Sparse coding capacity >> 0.14N (dense Hopfield limit)
  4. Sparsity: fraction of active neurons per pattern
     - Target: 1-5% for CA3 (literature range)
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
    """
    Pattern separation via expansion and sparsification.

    Takes d-dim input, projects to D-dim space (D >> d),
    then keeps only top-k active units (winner-take-all).
    """
    def __init__(self, d_input, D_output, k_active):
        self.d_input = d_input
        self.D_output = D_output
        self.k_active = k_active

        # Random projection weights (fixed, not learned)
        # Sparse connectivity: each granule cell receives from a random
        # subset of EC cells (biologically, ~1/3 of EC cells)
        connectivity_prob = 0.33
        mask = (np.random.rand(D_output, d_input) < connectivity_prob).astype(float)
        weights = np.random.randn(D_output, d_input) * mask
        # Normalize rows so each granule cell has comparable input magnitude
        row_norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-10
        self.W = weights / row_norms

    def forward(self, x):
        """
        x: (d_input,) or (batch, d_input)
        Returns sparse D_output-dim representation
        """
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        # Project to high-dimensional space
        h = x @ self.W.T  # (batch, D_output)

        # ReLU (granule cells have non-negative firing rates)
        h = np.maximum(h, 0)

        # k-winners-take-all sparsification
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
    Autoassociative network with Hebbian storage and pattern completion.

    Stores sparse patterns via outer products.
    Retrieves via iterative attractor dynamics with the same
    k-winners-take-all sparsity constraint.
    """
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.stored_patterns = []

    def store(self, pattern):
        """Store a pattern via Hebbian outer product."""
        # Normalize before storage
        norm = np.linalg.norm(pattern) + 1e-10
        p = pattern / norm
        self.W += np.outer(p, p)
        # Zero diagonal (no self-connections, standard for Hopfield)
        np.fill_diagonal(self.W, 0)
        self.n_stored += 1
        self.stored_patterns.append(pattern.copy())

    def retrieve(self, cue, n_iterations=10):
        """
        Pattern completion via iterative dynamics.
        Apply W @ x, then k-WTA sparsification, repeat.
        """
        x = cue.copy()
        for _ in range(n_iterations):
            h = self.W @ x
            h = np.maximum(h, 0)  # ReLU

            # k-WTA sparsification
            x_new = np.zeros_like(h)
            if self.k_active < len(h):
                top_k_idx = np.argpartition(h, -self.k_active)[-self.k_active:]
                x_new[top_k_idx] = h[top_k_idx]
            else:
                x_new = h

            # Normalize
            norm = np.linalg.norm(x_new) + 1e-10
            x_new = x_new / norm

            # Check convergence
            if np.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x


class Hippocampus:
    """Full hippocampal model: EC → DG → CA3"""
    def __init__(self, d_ec, D_dg, k_dg, k_ca3=None):
        self.d_ec = d_ec
        if k_ca3 is None:
            k_ca3 = k_dg  # same sparsity in CA3 as DG output
        self.dg = DentateGyrus(d_ec, D_dg, k_dg)
        self.ca3 = CA3(D_dg, k_ca3)

    def encode_and_store(self, ec_input):
        """Encode through DG and store in CA3."""
        dg_output = self.dg.forward(ec_input)
        self.ca3.store(dg_output)
        return dg_output

    def retrieve(self, ec_cue, n_iterations=10):
        """Encode cue through DG and retrieve from CA3."""
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
    """Fraction of neurons that are active (non-zero)."""
    return np.count_nonzero(pattern) / len(pattern)


def pattern_match(retrieved, stored_patterns):
    """Find best-matching stored pattern and return similarity."""
    best_sim = -1
    best_idx = -1
    for i, sp in enumerate(stored_patterns):
        sim = cosine_similarity(retrieved, sp)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_sim, best_idx


def degrade_pattern(pattern, frac_removed):
    """Zero out a fraction of the active elements."""
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
# 3. EXPERIMENTS
# =============================================================================

# Model parameters
d_ec = 100       # entorhinal cortex dimensionality
D_dg = 1000      # dentate gyrus size (10x expansion)
k_dg = 50        # active DG granule cells per pattern (5% sparsity)
k_ca3 = 50       # active CA3 cells (5% sparsity)

hippo = Hippocampus(d_ec, D_dg, k_dg, k_ca3)

print(f"Model: EC({d_ec}) -> DG({D_dg}, k={k_dg}) -> CA3({D_dg}, k={k_ca3})")
print(f"Target sparsity: {k_dg/D_dg:.1%}")
print()

# --- Experiment 1: Pattern Separation ---
print("="*60)
print("EXPERIMENT 1: Pattern Separation")
print("="*60)

n_separation_tests = 20
input_similarities = np.linspace(0, 0.99, n_separation_tests)
dg_output_sims = []
ca3_output_sims = []  # after storage and retrieval

# Create a reference pattern
ref_ec = np.random.randn(d_ec)
ref_ec = ref_ec / np.linalg.norm(ref_ec)

for target_sim in input_similarities:
    sims_dg = []
    sims_ca3 = []

    for trial in range(30):
        # Create a pattern with approximately target_sim similarity to ref
        noise = np.random.randn(d_ec)
        noise = noise - np.dot(noise, ref_ec) * ref_ec  # orthogonal component
        noise = noise / (np.linalg.norm(noise) + 1e-10)

        # Blend: target_sim * ref + sqrt(1-target_sim^2) * noise
        test_ec = target_sim * ref_ec + np.sqrt(max(0, 1 - target_sim**2)) * noise
        test_ec = test_ec / (np.linalg.norm(test_ec) + 1e-10)

        actual_sim = cosine_similarity(ref_ec, test_ec)

        # DG outputs
        dg_ref = hippo.dg.forward(ref_ec)
        dg_test = hippo.dg.forward(test_ec)
        dg_sim = cosine_similarity(dg_ref, dg_test)
        sims_dg.append(dg_sim)

    dg_output_sims.append(np.mean(sims_dg))

print("Input similarity -> DG output similarity:")
for i in range(0, len(input_similarities), 4):
    print(f"  {input_similarities[i]:.2f} -> {dg_output_sims[i]:.4f}")


# --- Experiment 2: Storage and Retrieval ---
print(f"\n{'='*60}")
print("EXPERIMENT 2: Storage Capacity and Retrieval")
print("="*60)

# Store increasing numbers of patterns, measure retrieval quality
capacity_tests = [5, 10, 20, 50, 100, 200, 500]
capacity_results = {}

for n_patterns in capacity_tests:
    hippo_cap = Hippocampus(d_ec, D_dg, k_dg, k_ca3)

    # Generate and store random EC patterns
    ec_patterns = []
    dg_patterns = []
    for i in range(n_patterns):
        ec = np.random.randn(d_ec)
        ec = ec / np.linalg.norm(ec)
        ec_patterns.append(ec)
        dg_out = hippo_cap.encode_and_store(ec)
        dg_patterns.append(dg_out)

    # Test retrieval with full cues
    correct_full = 0
    sims_full = []
    for i in range(min(n_patterns, 50)):  # test up to 50
        retrieved = hippo_cap.retrieve(ec_patterns[i])
        sim, best_idx = pattern_match(retrieved, dg_patterns)
        sims_full.append(sim)
        if best_idx == i:
            correct_full += 1

    n_tested = min(n_patterns, 50)

    # Test retrieval with degraded cues (50% removed)
    correct_degraded = 0
    sims_degraded = []
    for i in range(min(n_patterns, 50)):
        dg_cue = dg_patterns[i]
        degraded_cue = degrade_pattern(dg_cue, 0.5)
        retrieved = hippo_cap.ca3.retrieve(degraded_cue)
        sim, best_idx = pattern_match(retrieved, dg_patterns)
        sims_degraded.append(sim)
        if best_idx == i:
            correct_degraded += 1

    # Sparsity measurement
    sparsities = [measure_sparsity(dp) for dp in dg_patterns]

    capacity_results[n_patterns] = {
        "full_cue_accuracy": correct_full / n_tested,
        "full_cue_mean_sim": float(np.mean(sims_full)),
        "degraded_cue_accuracy": correct_degraded / n_tested,
        "degraded_cue_mean_sim": float(np.mean(sims_degraded)),
        "mean_sparsity": float(np.mean(sparsities)),
        "n_tested": n_tested,
    }

    print(f"  N={n_patterns:4d}: full_acc={correct_full/n_tested:.2f}, "
          f"full_sim={np.mean(sims_full):.4f}, "
          f"degraded_acc={correct_degraded/n_tested:.2f}, "
          f"degraded_sim={np.mean(sims_degraded):.4f}, "
          f"sparsity={np.mean(sparsities):.3f}")


# --- Experiment 3: Pattern Completion Curve ---
print(f"\n{'='*60}")
print("EXPERIMENT 3: Pattern Completion vs Degradation Level")
print("="*60)

# Store a moderate number of patterns
n_store = 50
hippo_comp = Hippocampus(d_ec, D_dg, k_dg, k_ca3)
ec_pats_comp = []
dg_pats_comp = []

for i in range(n_store):
    ec = np.random.randn(d_ec)
    ec = ec / np.linalg.norm(ec)
    ec_pats_comp.append(ec)
    dg_out = hippo_comp.encode_and_store(ec)
    dg_pats_comp.append(dg_out)

degradation_levels = np.linspace(0, 0.95, 20)
completion_results = {}

for deg_level in degradation_levels:
    correct = 0
    sims = []
    n_test_pat = min(n_store, 30)

    for i in range(n_test_pat):
        degraded = degrade_pattern(dg_pats_comp[i], deg_level)
        retrieved = hippo_comp.ca3.retrieve(degraded)
        sim, best_idx = pattern_match(retrieved, dg_pats_comp)
        sims.append(sim)
        if best_idx == i:
            correct += 1

    completion_results[float(deg_level)] = {
        "accuracy": correct / n_test_pat,
        "mean_similarity": float(np.mean(sims)),
    }

    if deg_level < 0.01 or abs(deg_level - 0.5) < 0.03 or deg_level > 0.94:
        print(f"  Degradation {deg_level:.0%}: "
              f"acc={correct/n_test_pat:.2f}, "
              f"sim={np.mean(sims):.4f}")


# --- Experiment 4: Sparsity Analysis ---
print(f"\n{'='*60}")
print("EXPERIMENT 4: Sparsity Analysis")
print("="*60)

# Generate many patterns and measure sparsity statistics
n_sparsity_test = 200
hippo_sp = Hippocampus(d_ec, D_dg, k_dg, k_ca3)
sparsity_data = []

for i in range(n_sparsity_test):
    ec = np.random.randn(d_ec)
    ec = ec / np.linalg.norm(ec)
    dg_out = hippo_sp.dg.forward(ec)
    sp = measure_sparsity(dg_out)
    n_active = np.count_nonzero(dg_out)
    sparsity_data.append({
        "sparsity": float(sp),
        "n_active": int(n_active),
    })

mean_sp = np.mean([s["sparsity"] for s in sparsity_data])
mean_active = np.mean([s["n_active"] for s in sparsity_data])
print(f"DG output sparsity: {mean_sp:.4f} ({mean_active:.1f} active out of {D_dg})")
print(f"Target: ~{k_dg/D_dg:.4f} ({k_dg} active out of {D_dg})")
print(f"Literature reference: 1-5% for CA3 in macaques")


# =============================================================================
# 4. SAVE ALL RESULTS
# =============================================================================

results = {
    "model_params": {
        "d_ec": d_ec,
        "D_dg": D_dg,
        "k_dg": k_dg,
        "k_ca3": k_ca3,
        "expansion_ratio": D_dg / d_ec,
        "target_sparsity": k_dg / D_dg,
    },
    "pattern_separation": {
        "input_similarities": input_similarities.tolist(),
        "dg_output_similarities": dg_output_sims,
    },
    "storage_capacity": {
        str(k): v for k, v in capacity_results.items()
    },
    "pattern_completion": {
        f"{k:.3f}": v for k, v in completion_results.items()
    },
    "sparsity": {
        "mean_sparsity": float(mean_sp),
        "mean_n_active": float(mean_active),
        "target_sparsity": k_dg / D_dg,
        "all_sparsities": [s["sparsity"] for s in sparsity_data],
    },
    "reference_values": {
        "hopfield_dense_capacity_limit": f"0.14 * N = {0.14 * D_dg:.0f} patterns",
        "literature_ca3_sparsity": "1-5% in macaques",
        "literature_expansion_ratio": "~5-10x (EC to DG in rat)",
    }
}

with open('hippocampal_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# =============================================================================
# 5. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Simplified Hippocampal Model: Validation Metrics",
             fontsize=14, fontweight='bold')

# Panel 1: Pattern Separation
ax = axes[0, 0]
ax.plot(input_similarities, dg_output_sims, 'bo-', markersize=4, label='DG output')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Identity (no separation)')
ax.set_xlabel('Input (EC) cosine similarity')
ax.set_ylabel('Output (DG) cosine similarity')
ax.set_title('Pattern Separation: DG decorrelates similar inputs')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)

# Panel 2: Storage Capacity
ax = axes[0, 1]
n_pats = sorted(capacity_results.keys())
full_accs = [capacity_results[n]["full_cue_accuracy"] for n in n_pats]
deg_accs = [capacity_results[n]["degraded_cue_accuracy"] for n in n_pats]
ax.semilogx(n_pats, full_accs, 'bo-', markersize=6, label='Full cue')
ax.semilogx(n_pats, deg_accs, 'rs-', markersize=6, label='50% degraded cue')
ax.axvline(x=0.14*D_dg, color='gray', linestyle='--', alpha=0.5,
           label=f'Hopfield limit (0.14N={0.14*D_dg:.0f})')
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Retrieval accuracy')
ax.set_title('Storage Capacity')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# Panel 3: Pattern Completion Curve
ax = axes[1, 0]
deg_levels = sorted(completion_results.keys())
comp_accs = [completion_results[d]["accuracy"] for d in deg_levels]
comp_sims = [completion_results[d]["mean_similarity"] for d in deg_levels]
ax.plot(deg_levels, comp_accs, 'bo-', markersize=4, label='Correct retrieval')
ax.plot(deg_levels, comp_sims, 'rs-', markersize=4, label='Cosine similarity')
ax.set_xlabel('Fraction of pattern degraded')
ax.set_ylabel('Retrieval quality')
ax.set_title(f'Pattern Completion (N={n_store} stored patterns)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.0)
ax.set_ylim(-0.05, 1.05)

# Panel 4: Sparsity Distribution
ax = axes[1, 1]
all_sp = [s["sparsity"] for s in sparsity_data]
ax.hist(all_sp, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=k_dg/D_dg, color='red', linestyle='--', linewidth=2,
           label=f'Target: {k_dg/D_dg:.1%}')
ax.axvline(x=mean_sp, color='green', linestyle='-', linewidth=2,
           label=f'Measured: {mean_sp:.1%}')
ax.set_xlabel('Fraction of active neurons')
ax.set_ylabel('Count')
ax.set_title('DG Output Sparsity Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hippocampal_model_results.png', dpi=150, bbox_inches='tight')

print(f"\nResults saved to hippocampal_model_results.json")
print(f"Plot saved to hippocampal_model_results.png")
