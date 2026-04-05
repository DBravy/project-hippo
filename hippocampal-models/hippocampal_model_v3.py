"""
Hippocampal Model v3
====================

Changes from v2:
1. DG uses lateral inhibition on a 1D ring topology instead of k-WTA
   - Gaussian inhibitory connectivity (broad but not all-to-all)
   - Iterative inhibition dynamics produce input-dependent variable sparsity
   - Small noise injection for effective dimensionality expansion
2. CA3 uses batch covariance rule (global mean computed in one pass,
   fixes the running-mean drift from v2)
3. New diagnostics:
   - Effective dimensionality via participation ratio
   - Sparsity distribution across stored patterns
   - Retrieval failure correlation with per-pattern sparsity
4. Key controlled comparison: variable sparsity vs renormalized fixed
   sparsity, isolating the dimensionality benefit from the variable
   density effect

Architecture: EC(100) -> DG(1000, lateral inhibition) -> CA3(1000, batch covariance)
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================

def make_feedforward_weights(D_output, d_input, connectivity_prob=0.33):
    """Create normalized sparse random feedforward projection weights."""
    mask = (np.random.rand(D_output, d_input) < connectivity_prob).astype(float)
    weights = np.random.randn(D_output, d_input) * mask
    row_norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-10
    return weights / row_norms


def build_ring_inhibition(D, sigma, connection_prob=None):
    """
    Build lateral inhibition matrix on a 1D ring topology.

    Gaussian distance-dependent weights, truncated at 3*sigma.
    Row-normalized so inhibition represents a weighted local average.

    Parameters
    ----------
    D : int
        Number of neurons (ring circumference).
    sigma : float
        Gaussian width in neuron-index units. Controls the spatial reach
        of each inhibitory neuron. With sigma=50 on a 1000-neuron ring,
        each neuron's inhibition reaches ~300 neighbors (3*sigma in each
        direction), covering ~30% of the ring.
    connection_prob : float or None
        If set, applies an additional random sparse mask so that not all
        neurons within range are connected. This adds structural
        variability to the inhibitory network. None means all within-range
        connections are kept.
    """
    positions = np.arange(D)
    dist = np.abs(positions[:, None] - positions[None, :])
    dist = np.minimum(dist, D - dist)  # wrap-around ring distance

    W = np.exp(-dist**2 / (2 * sigma**2))
    W[dist > 3 * sigma] = 0
    np.fill_diagonal(W, 0)

    # Optional additional sparsification of inhibitory connectivity
    if connection_prob is not None:
        mask = (np.random.rand(D, D) < connection_prob).astype(float)
        np.fill_diagonal(mask, 0)
        W *= mask

    # Normalize rows so inhibition = weighted average of neighbors
    row_sums = W.sum(axis=1, keepdims=True) + 1e-10
    W /= row_sums
    return W


def apply_kwta(pattern, k):
    """Apply k-WTA to a pattern, returning a copy with only top-k active."""
    out = np.zeros_like(pattern)
    if k < len(pattern):
        top_k_idx = np.argpartition(pattern, -k)[-k:]
        out[top_k_idx] = pattern[top_k_idx]
    else:
        out = pattern.copy()
    return out


# =============================================================================
# 2. MODEL COMPONENTS
# =============================================================================

class DentateGyrusLateral:
    """
    DG with lateral inhibition on a 1D ring topology.

    Instead of k-WTA, sparsity emerges from iterative competition between
    excitatory feedforward drive and local lateral inhibition. This produces
    input-dependent variable sparsity: coherent inputs that drive a few
    neurons strongly yield denser outputs than diffuse inputs that drive
    many neurons weakly.

    Small noise injection after the feedforward pass pushes patterns into
    dimensions not spanned by the EC input, addressing the effective
    dimensionality bottleneck from v2.

    Parameters
    ----------
    d_input : int
        EC input dimensionality.
    D_output : int
        Number of DG granule cells.
    sigma_inh : float
        Spatial reach of lateral inhibition (Gaussian width on the ring).
    gamma_inh : float
        Inhibition strength. With row-normalized W_inh, gamma=1.0 means
        inhibition equals the weighted local average of neighbor activity.
        Neurons must exceed their local average to survive.
    n_inh_steps : int
        Number of lateral inhibition iterations.
    noise_scale : float
        Noise magnitude as fraction of mean active excitation.
    W_ff : ndarray or None
        Feedforward weights. If None, generates new random weights.
        Pass shared weights for controlled comparisons.
    inh_connection_prob : float or None
        Sparsification of inhibitory connections within range.
    """
    def __init__(self, d_input, D_output, sigma_inh=50, gamma_inh=1.0,
                 n_inh_steps=5, noise_scale=0.05, W_ff=None,
                 inh_connection_prob=None):
        self.d_input = d_input
        self.D_output = D_output
        self.sigma_inh = sigma_inh
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.noise_scale = noise_scale

        if W_ff is not None:
            self.W_ff = W_ff.copy()
        else:
            self.W_ff = make_feedforward_weights(D_output, d_input)

        self.W_inh = build_ring_inhibition(
            D_output, sigma_inh, connection_prob=inh_connection_prob
        )

    def forward(self, x):
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        results = []
        for i in range(x.shape[0]):
            # Feedforward excitation
            h_raw = x[i] @ self.W_ff.T
            h_raw = np.maximum(h_raw, 0)  # ReLU

            # Noise injection for dimensionality expansion
            # Scaled to mean activation of excited units so noise is
            # proportional to signal strength, not fixed
            if self.noise_scale > 0 and np.any(h_raw > 0):
                mean_active = np.mean(h_raw[h_raw > 0])
                noise = np.random.randn(self.D_output) * self.noise_scale * mean_active
                h_raw = np.maximum(h_raw + noise, 0)

            # Iterative lateral inhibition
            # h_raw is the constant feedforward drive; inhibition is
            # recomputed from current activity each step. This converges
            # to a fixed point where surviving neurons' excitation exceeds
            # the inhibition from their active neighbors.
            h = h_raw.copy()
            for step in range(self.n_inh_steps):
                inh = self.W_inh @ h
                h = np.maximum(h_raw - self.gamma_inh * inh, 0)

            results.append(h)

        out = np.array(results)
        if single:
            out = out[0]
        return out


class DentateGyrusKWTA:
    """Original k-WTA DG from v2, for controlled comparison."""
    def __init__(self, d_input, D_output, k_active, W_ff=None):
        self.d_input = d_input
        self.D_output = D_output
        self.k_active = k_active

        if W_ff is not None:
            self.W_ff = W_ff.copy()
        else:
            self.W_ff = make_feedforward_weights(D_output, d_input)

    def forward(self, x):
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        h = x @ self.W_ff.T
        h = np.maximum(h, 0)
        out = np.zeros_like(h)
        for i in range(h.shape[0]):
            out[i] = apply_kwta(h[i], self.k_active)
        if single:
            out = out[0]
        return out


class CA3:
    """
    Autoassociative network with batch covariance learning rule.

    Batch storage computes the global mean across all patterns before
    building the weight matrix, fixing v2's running-mean drift where
    early and late patterns were centered against different means.

    Retrieval uses iterative dynamics with k-WTA sparsification.
    NOTE: k-WTA in retrieval always produces fixed-density outputs.
    When stored patterns have variable density (from lateral inhibition
    DG), this is a known simplification. The retrieved pattern's active
    set may not perfectly match the stored pattern's, but the nearest-
    neighbor matching in evaluation handles this.
    """
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.stored_patterns = []
        self.mean_activity = np.zeros(N)

    def store_batch(self, patterns):
        """
        Store all patterns using batch covariance rule.

        W = sum_i (p_i - mu)(p_i - mu)^T,  mu = mean of all p_i
        Each pattern is L2-normalized before computing the mean.
        """
        normalized = []
        for p in patterns:
            norm = np.linalg.norm(p) + 1e-10
            normalized.append(p / norm)
        normalized = np.array(normalized)

        self.mean_activity = np.mean(normalized, axis=0)
        centered = normalized - self.mean_activity

        self.W = centered.T @ centered
        np.fill_diagonal(self.W, 0)
        self.n_stored = len(patterns)
        self.stored_patterns = [p.copy() for p in patterns]

    def retrieve(self, cue, n_iterations=15):
        """Pattern completion via iterative dynamics with k-WTA."""
        norm = np.linalg.norm(cue) + 1e-10
        x = cue / norm

        for _ in range(n_iterations):
            h = self.W @ x
            h = np.maximum(h, 0)

            x_new = apply_kwta(h, self.k_active)

            norm = np.linalg.norm(x_new) + 1e-10
            x_new = x_new / norm

            if np.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x


# =============================================================================
# 3. METRICS
# =============================================================================

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


def measure_sparsity(pattern):
    """Fraction of non-zero elements."""
    return np.count_nonzero(pattern) / len(pattern)


def pattern_match(retrieved, stored_patterns):
    """Find best-matching stored pattern by cosine similarity."""
    best_sim = -1
    best_idx = -1
    for i, sp in enumerate(stored_patterns):
        sim = cosine_similarity(retrieved, sp)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_sim, best_idx


def degrade_pattern(pattern, frac_removed):
    """Zero out a fraction of the active elements in a pattern."""
    degraded = pattern.copy()
    active_idx = np.where(pattern != 0)[0]
    if len(active_idx) == 0:
        return degraded
    n_remove = int(len(active_idx) * frac_removed)
    if n_remove > 0:
        remove_idx = np.random.choice(active_idx, size=n_remove, replace=False)
        degraded[remove_idx] = 0
    return degraded


def participation_ratio(patterns):
    """
    Effective dimensionality via participation ratio.

    PR = (sum(lambda_i))^2 / sum(lambda_i^2)

    Ranges from 1 (all variance concentrated in one dimension)
    to min(n_patterns, n_dims) (variance uniformly spread).
    """
    X = np.array(patterns)
    X = X - X.mean(axis=0)
    # Use SVD for numerical stability (avoids forming D x D covariance)
    # Eigenvalues of X^T X / (n-1) are s^2 / (n-1)
    s = np.linalg.svd(X, compute_uv=False)
    eigenvalues = s**2 / max(X.shape[0] - 1, 1)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues) == 0:
        return 0.0
    pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
    return float(pr)


def sparsity_stats(patterns):
    """Compute sparsity statistics across a set of patterns."""
    sparsities = [measure_sparsity(p) for p in patterns]
    active_counts = [np.count_nonzero(p) for p in patterns]
    return {
        "mean_sparsity": float(np.mean(sparsities)),
        "std_sparsity": float(np.std(sparsities)),
        "min_sparsity": float(np.min(sparsities)),
        "max_sparsity": float(np.max(sparsities)),
        "median_sparsity": float(np.median(sparsities)),
        "mean_active_count": float(np.mean(active_counts)),
        "std_active_count": float(np.std(active_counts)),
        "min_active_count": int(np.min(active_counts)),
        "max_active_count": int(np.max(active_counts)),
        "all_sparsities": [float(s) for s in sparsities],
        "all_active_counts": [int(c) for c in active_counts],
    }


def generate_ec_patterns(n, d_ec):
    """Generate n random unit-norm EC input patterns."""
    patterns = []
    for _ in range(n):
        ec = np.random.randn(d_ec)
        ec = ec / np.linalg.norm(ec)
        patterns.append(ec)
    return patterns


# =============================================================================
# 4. PARAMETERS
# =============================================================================

d_ec = 100       # EC input dimensionality
D_dg = 1000      # DG granule cell count
k_ca3 = 50       # k-WTA in CA3 retrieval (kept from v2 for comparability)

# Lateral inhibition parameters
sigma_inh = 25    # Gaussian width on ring (~30% coverage at 3*sigma)
gamma_inh = 5.0   # Inhibition strength (1.0 = must exceed local average)
n_inh_steps = 5   # Inhibition iterations
noise_scale = 0.00  # Noise as fraction of mean active excitation

# v2 parameters for comparison
k_dg_v2 = 50     # k-WTA sparsity from v2 (5% of 1000)

print(f"Model: EC({d_ec}) -> DG({D_dg}, lateral inhibition) -> CA3({D_dg}, k={k_ca3})")
print(f"Lateral inhibition: sigma={sigma_inh}, gamma={gamma_inh}, "
      f"steps={n_inh_steps}, noise={noise_scale}")
print(f"v2 comparison: DG k-WTA with k={k_dg_v2}")
print()

# =============================================================================
# 5. EXPERIMENT 1: DG Output Diagnostics
#    Compare lateral inhibition DG vs k-WTA DG on dimensionality and sparsity
# =============================================================================

print("=" * 60)
print("EXPERIMENT 1: DG Output Diagnostics")
print("=" * 60)

# Shared feedforward weights so differences are due to sparsification only
W_ff_shared = make_feedforward_weights(D_dg, d_ec)

dg_lateral = DentateGyrusLateral(
    d_ec, D_dg, sigma_inh=sigma_inh, gamma_inh=gamma_inh,
    n_inh_steps=n_inh_steps, noise_scale=noise_scale, W_ff=W_ff_shared
)
dg_kwta = DentateGyrusKWTA(d_ec, D_dg, k_dg_v2, W_ff=W_ff_shared)

n_diag = 500
ec_diag = generate_ec_patterns(n_diag, d_ec)

print(f"  Generating {n_diag} DG outputs for each DG type...")
dg_lateral_outputs = [dg_lateral.forward(ec) for ec in ec_diag]
dg_kwta_outputs = [dg_kwta.forward(ec) for ec in ec_diag]

# Sparsity statistics
lateral_sparsity = sparsity_stats(dg_lateral_outputs)
kwta_sparsity = sparsity_stats(dg_kwta_outputs)

print(f"  Lateral inhibition DG:")
print(f"    Mean active units: {lateral_sparsity['mean_active_count']:.1f} "
      f"(+/- {lateral_sparsity['std_active_count']:.1f})")
print(f"    Range: [{lateral_sparsity['min_active_count']}, "
      f"{lateral_sparsity['max_active_count']}]")
print(f"    Mean sparsity: {lateral_sparsity['mean_sparsity']:.4f}")
print(f"  k-WTA DG:")
print(f"    Mean active units: {kwta_sparsity['mean_active_count']:.1f} "
      f"(+/- {kwta_sparsity['std_active_count']:.1f})")
print(f"    Mean sparsity: {kwta_sparsity['mean_sparsity']:.4f}")

# Effective dimensionality
print(f"\n  Computing participation ratios...")
pr_lateral = participation_ratio(dg_lateral_outputs)
pr_kwta = participation_ratio(dg_kwta_outputs)
# Also measure EC input dimensionality for reference
pr_ec = participation_ratio(ec_diag)

print(f"    EC inputs:           PR = {pr_ec:.1f}")
print(f"    k-WTA DG outputs:    PR = {pr_kwta:.1f}")
print(f"    Lateral DG outputs:  PR = {pr_lateral:.1f}")

dg_diagnostics = {
    "n_patterns": n_diag,
    "lateral_sparsity": lateral_sparsity,
    "kwta_sparsity": kwta_sparsity,
    "participation_ratio_ec": pr_ec,
    "participation_ratio_kwta": pr_kwta,
    "participation_ratio_lateral": pr_lateral,
}

# =============================================================================
# 6. EXPERIMENT 2: Pattern Separation
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 2: Pattern Separation")
print("=" * 60)

n_separation_tests = 20
input_similarities = np.linspace(0, 0.99, n_separation_tests)

ref_ec = np.random.randn(d_ec)
ref_ec = ref_ec / np.linalg.norm(ref_ec)

sep_results = {"lateral": [], "kwta": []}

for target_sim in input_similarities:
    sims_lat = []
    sims_kwta = []
    for trial in range(30):
        # Generate EC input at target similarity to reference
        noise = np.random.randn(d_ec)
        noise = noise - np.dot(noise, ref_ec) * ref_ec
        noise = noise / (np.linalg.norm(noise) + 1e-10)
        test_ec = target_sim * ref_ec + np.sqrt(max(0, 1 - target_sim**2)) * noise
        test_ec = test_ec / (np.linalg.norm(test_ec) + 1e-10)

        dg_ref_lat = dg_lateral.forward(ref_ec)
        dg_test_lat = dg_lateral.forward(test_ec)
        sims_lat.append(cosine_similarity(dg_ref_lat, dg_test_lat))

        dg_ref_kwta = dg_kwta.forward(ref_ec)
        dg_test_kwta = dg_kwta.forward(test_ec)
        sims_kwta.append(cosine_similarity(dg_ref_kwta, dg_test_kwta))

    sep_results["lateral"].append(float(np.mean(sims_lat)))
    sep_results["kwta"].append(float(np.mean(sims_kwta)))

print("Input sim -> DG output sim (sampled):")
for i in range(0, len(input_similarities), 4):
    print(f"  {input_similarities[i]:.2f} -> lateral: {sep_results['lateral'][i]:.4f}, "
          f"kwta: {sep_results['kwta'][i]:.4f}")

separation_data = {
    "input_similarities": input_similarities.tolist(),
    "lateral_output_sims": sep_results["lateral"],
    "kwta_output_sims": sep_results["kwta"],
}

# =============================================================================
# 7. EXPERIMENT 3: Storage Capacity (Lateral Inhibition DG + Batch Covariance)
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 3: Storage Capacity")
print("=" * 60)

capacity_tests = [5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 750,
                  1000, 1500, 2000, 3000, 5000]

n_test_per = 50
capacity_results = {}

for n_patterns in capacity_tests:
    print(f"  Testing N={n_patterns}...", end=" ", flush=True)

    # Fresh DG for each load level (independent feedforward weights)
    dg_cap = DentateGyrusLateral(
        d_ec, D_dg, sigma_inh=sigma_inh, gamma_inh=gamma_inh,
        n_inh_steps=n_inh_steps, noise_scale=noise_scale
    )
    ca3_cap = CA3(D_dg, k_ca3)

    ec_patterns = generate_ec_patterns(n_patterns, d_ec)
    dg_patterns = [dg_cap.forward(ec) for ec in ec_patterns]

    # Store with batch covariance
    ca3_cap.store_batch(dg_patterns)

    # Sparsity of stored patterns
    sp_stats = sparsity_stats(dg_patterns)

    # Effective dimensionality (skip for very large n to save time)
    if n_patterns <= 2000:
        pr = participation_ratio(dg_patterns)
    else:
        pr = -1  # sentinel: not computed

    # Test retrieval: full cue (using stored DG patterns directly)
    test_indices = np.random.choice(n_patterns, size=min(n_test_per, n_patterns),
                                    replace=False)
    correct_full = 0
    sims_full = []
    per_pattern_sparsity = []
    per_pattern_correct = []

    for idx in test_indices:
        retrieved = ca3_cap.retrieve(dg_patterns[idx])
        sim, best_idx = pattern_match(retrieved, dg_patterns)
        sims_full.append(float(sim))
        is_correct = int(best_idx == idx)
        correct_full += is_correct
        per_pattern_sparsity.append(measure_sparsity(dg_patterns[idx]))
        per_pattern_correct.append(is_correct)

    n_tested = len(test_indices)

    # Test retrieval: 50% degraded cue
    correct_degraded = 0
    sims_degraded = []

    for idx in test_indices:
        degraded = degrade_pattern(dg_patterns[idx], 0.5)
        retrieved = ca3_cap.retrieve(degraded)
        sim, best_idx = pattern_match(retrieved, dg_patterns)
        sims_degraded.append(float(sim))
        if best_idx == idx:
            correct_degraded += 1

    # Correlation between pattern sparsity and retrieval success
    if len(set(per_pattern_correct)) > 1 and len(set(per_pattern_sparsity)) > 1:
        sparsity_retrieval_corr = float(
            np.corrcoef(per_pattern_sparsity, per_pattern_correct)[0, 1]
        )
    else:
        sparsity_retrieval_corr = 0.0

    capacity_results[n_patterns] = {
        "full_cue_accuracy": correct_full / n_tested,
        "full_cue_mean_sim": float(np.mean(sims_full)),
        "degraded_cue_accuracy": correct_degraded / n_tested,
        "degraded_cue_mean_sim": float(np.mean(sims_degraded)),
        "n_tested": n_tested,
        "participation_ratio": pr,
        "mean_active_count": sp_stats["mean_active_count"],
        "std_active_count": sp_stats["std_active_count"],
        "sparsity_retrieval_correlation": sparsity_retrieval_corr,
    }

    print(f"full_acc={correct_full/n_tested:.2f}, "
          f"sim={np.mean(sims_full):.4f}, "
          f"deg_acc={correct_degraded/n_tested:.2f}, "
          f"PR={pr:.1f}, "
          f"active={sp_stats['mean_active_count']:.0f}+/-{sp_stats['std_active_count']:.0f}")

# =============================================================================
# 8. EXPERIMENT 4: Pattern Completion at Multiple Load Levels
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 4: Pattern Completion vs Degradation at Multiple Loads")
print("=" * 60)

completion_load_levels = [50, 200, 500, 1000]
degradation_levels = np.linspace(0, 0.95, 20)
completion_results = {}

for n_load in completion_load_levels:
    print(f"\n  Load N={n_load}:")

    dg_comp = DentateGyrusLateral(
        d_ec, D_dg, sigma_inh=sigma_inh, gamma_inh=gamma_inh,
        n_inh_steps=n_inh_steps, noise_scale=noise_scale
    )
    ca3_comp = CA3(D_dg, k_ca3)

    ec_pats = generate_ec_patterns(n_load, d_ec)
    dg_pats = [dg_comp.forward(ec) for ec in ec_pats]
    ca3_comp.store_batch(dg_pats)

    load_results = {}
    n_test_comp = min(30, n_load)
    test_idx = np.random.choice(n_load, size=n_test_comp, replace=False)

    for deg_level in degradation_levels:
        correct = 0
        sims = []

        for i in test_idx:
            degraded = degrade_pattern(dg_pats[i], deg_level)
            retrieved = ca3_comp.retrieve(degraded)
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
# 9. EXPERIMENT 5: Lateral Inhibition DG vs k-WTA DG
#    Direct comparison with shared feedforward weights and batch covariance
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 5: Lateral Inhibition DG vs k-WTA DG")
print("=" * 60)

comparison_tests = [50, 100, 200, 500, 1000, 2000]
dg_comparison_results = {}

for n_patterns in comparison_tests:
    print(f"  N={n_patterns}...", end=" ", flush=True)

    # Shared feedforward weights
    W_ff_comp = make_feedforward_weights(D_dg, d_ec)

    dg_lat_comp = DentateGyrusLateral(
        d_ec, D_dg, sigma_inh=sigma_inh, gamma_inh=gamma_inh,
        n_inh_steps=n_inh_steps, noise_scale=noise_scale, W_ff=W_ff_comp
    )
    dg_kwta_comp = DentateGyrusKWTA(d_ec, D_dg, k_dg_v2, W_ff=W_ff_comp)

    ca3_lat = CA3(D_dg, k_ca3)
    ca3_kwta = CA3(D_dg, k_ca3)

    ec_pats = generate_ec_patterns(n_patterns, d_ec)
    dg_lat_pats = [dg_lat_comp.forward(ec) for ec in ec_pats]
    dg_kwta_pats = [dg_kwta_comp.forward(ec) for ec in ec_pats]

    ca3_lat.store_batch(dg_lat_pats)
    ca3_kwta.store_batch(dg_kwta_pats)

    test_idx = np.random.choice(n_patterns, size=min(50, n_patterns), replace=False)

    # Lateral inhibition DG -> CA3
    correct_lat = 0
    sims_lat = []
    for i in test_idx:
        ret = ca3_lat.retrieve(dg_lat_pats[i])
        sim, best_idx = pattern_match(ret, dg_lat_pats)
        sims_lat.append(float(sim))
        if best_idx == i:
            correct_lat += 1

    # k-WTA DG -> CA3
    correct_kwta = 0
    sims_kwta = []
    for i in test_idx:
        ret = ca3_kwta.retrieve(dg_kwta_pats[i])
        sim, best_idx = pattern_match(ret, dg_kwta_pats)
        sims_kwta.append(float(sim))
        if best_idx == i:
            correct_kwta += 1

    n_t = len(test_idx)

    # Dimensionality comparison
    if n_patterns <= 2000:
        pr_lat = participation_ratio(dg_lat_pats)
        pr_kwta_c = participation_ratio(dg_kwta_pats)
    else:
        pr_lat = -1
        pr_kwta_c = -1

    dg_comparison_results[n_patterns] = {
        "lateral_accuracy": correct_lat / n_t,
        "lateral_mean_sim": float(np.mean(sims_lat)),
        "kwta_accuracy": correct_kwta / n_t,
        "kwta_mean_sim": float(np.mean(sims_kwta)),
        "lateral_PR": pr_lat,
        "kwta_PR": pr_kwta_c,
    }

    print(f"lat_acc={correct_lat/n_t:.2f} (PR={pr_lat:.1f}), "
          f"kwta_acc={correct_kwta/n_t:.2f} (PR={pr_kwta_c:.1f})")

# =============================================================================
# 10. EXPERIMENT 6: Variable Sparsity vs Renormalized Fixed Sparsity
#     The key diagnostic: isolate dimensionality benefit from variable density
#
#     Both conditions use lateral inhibition DG (same feedforward weights,
#     same inhibition). The "fixed" condition applies k-WTA on top of the
#     lateral inhibition output to force exactly k active units. This keeps
#     the lateral inhibition's choice of WHICH neurons win (mostly) but
#     normalizes density, isolating the variable-sparsity effect.
# =============================================================================

print(f"\n{'=' * 60}")
print("EXPERIMENT 6: Variable Sparsity vs Renormalized Fixed Sparsity")
print("=" * 60)

sparsity_comparison_tests = [50, 100, 200, 500, 1000, 2000]
sparsity_comparison_results = {}

# Use median active count from Experiment 1 as the renormalization target
k_renorm = int(lateral_sparsity["mean_active_count"])
print(f"  Renormalization target: k={k_renorm} "
      f"(mean active count from lateral DG)")

for n_patterns in sparsity_comparison_tests:
    print(f"  N={n_patterns}...", end=" ", flush=True)

    W_ff_sp = make_feedforward_weights(D_dg, d_ec)

    dg_sp = DentateGyrusLateral(
        d_ec, D_dg, sigma_inh=sigma_inh, gamma_inh=gamma_inh,
        n_inh_steps=n_inh_steps, noise_scale=noise_scale, W_ff=W_ff_sp
    )

    ec_pats = generate_ec_patterns(n_patterns, d_ec)

    # Variable sparsity: raw lateral inhibition output
    dg_variable = [dg_sp.forward(ec) for ec in ec_pats]

    # Fixed sparsity: lateral inhibition output + k-WTA renormalization
    dg_fixed = [apply_kwta(p, k_renorm) for p in dg_variable]

    ca3_var = CA3(D_dg, k_ca3)
    ca3_fix = CA3(D_dg, k_ca3)
    ca3_var.store_batch(dg_variable)
    ca3_fix.store_batch(dg_fixed)

    test_idx = np.random.choice(n_patterns, size=min(50, n_patterns), replace=False)

    # Variable sparsity retrieval
    correct_var = 0
    sims_var = []
    for i in test_idx:
        ret = ca3_var.retrieve(dg_variable[i])
        sim, best_idx = pattern_match(ret, dg_variable)
        sims_var.append(float(sim))
        if best_idx == i:
            correct_var += 1

    # Fixed sparsity retrieval
    correct_fix = 0
    sims_fix = []
    for i in test_idx:
        ret = ca3_fix.retrieve(dg_fixed[i])
        sim, best_idx = pattern_match(ret, dg_fixed)
        sims_fix.append(float(sim))
        if best_idx == i:
            correct_fix += 1

    n_t = len(test_idx)

    # Dimensionality comparison
    if n_patterns <= 2000:
        pr_var = participation_ratio(dg_variable)
        pr_fix = participation_ratio(dg_fixed)
    else:
        pr_var = -1
        pr_fix = -1

    sparsity_comparison_results[n_patterns] = {
        "variable_accuracy": correct_var / n_t,
        "variable_mean_sim": float(np.mean(sims_var)),
        "fixed_accuracy": correct_fix / n_t,
        "fixed_mean_sim": float(np.mean(sims_fix)),
        "variable_PR": pr_var,
        "fixed_PR": pr_fix,
    }

    print(f"variable_acc={correct_var/n_t:.2f} (PR={pr_var:.1f}), "
          f"fixed_acc={correct_fix/n_t:.2f} (PR={pr_fix:.1f})")

# =============================================================================
# 11. SAVE RESULTS
# =============================================================================

results = {
    "model_params": {
        "d_ec": d_ec,
        "D_dg": D_dg,
        "k_ca3": k_ca3,
        "sigma_inh": sigma_inh,
        "gamma_inh": gamma_inh,
        "n_inh_steps": n_inh_steps,
        "noise_scale": noise_scale,
        "k_dg_v2": k_dg_v2,
        "k_renorm": k_renorm,
    },
    "dg_diagnostics": {
        # Exclude the full sparsity/count lists from top level for readability
        "lateral_mean_active": lateral_sparsity["mean_active_count"],
        "lateral_std_active": lateral_sparsity["std_active_count"],
        "lateral_min_active": lateral_sparsity["min_active_count"],
        "lateral_max_active": lateral_sparsity["max_active_count"],
        "kwta_mean_active": kwta_sparsity["mean_active_count"],
        "participation_ratio_ec": pr_ec,
        "participation_ratio_kwta": pr_kwta,
        "participation_ratio_lateral": pr_lateral,
    },
    "pattern_separation": separation_data,
    "storage_capacity": {str(k): v for k, v in capacity_results.items()},
    "pattern_completion_by_load": {str(k): v for k, v in completion_results.items()},
    "lateral_vs_kwta": {str(k): v for k, v in dg_comparison_results.items()},
    "variable_vs_fixed_sparsity": {str(k): v for k, v in sparsity_comparison_results.items()},
}

with open('hippocampal_model_v3_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# 12. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(3, 3, figsize=(22, 18))
fig.suptitle("Hippocampal Model v3: Lateral Inhibition DG + Batch Covariance CA3",
             fontsize=14, fontweight='bold')

# --- Panel 1: DG Sparsity Distribution ---
ax = axes[0, 0]
lateral_counts = lateral_sparsity["all_active_counts"]
kwta_counts = kwta_sparsity["all_active_counts"]
ax.hist(lateral_counts, bins=30, alpha=0.6, color='blue', label='Lateral inhibition',
        density=True)
ax.hist(kwta_counts, bins=5, alpha=0.6, color='red', label='k-WTA',
        density=True)
ax.set_xlabel('Number of active units')
ax.set_ylabel('Density')
ax.set_title('DG Sparsity Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Panel 2: Pattern Separation ---
ax = axes[0, 1]
ax.plot(input_similarities, sep_results["lateral"], 'bo-', markersize=4,
        label='Lateral inhibition')
ax.plot(input_similarities, sep_results["kwta"], 'rs-', markersize=4,
        label='k-WTA')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No separation')
ax.set_xlabel('Input (EC) cosine similarity')
ax.set_ylabel('Output (DG) cosine similarity')
ax.set_title('Pattern Separation')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Panel 3: Effective Dimensionality vs Load ---
ax = axes[0, 2]
cap_ns = sorted(capacity_results.keys())
prs = [capacity_results[n]["participation_ratio"] for n in cap_ns]
prs_valid = [(n, p) for n, p in zip(cap_ns, prs) if p > 0]
if prs_valid:
    ns_v, prs_v = zip(*prs_valid)
    ax.semilogx(ns_v, prs_v, 'go-', markersize=5, label='Lateral DG (PR)')
ax.axhline(y=d_ec, color='gray', linestyle='--', alpha=0.5,
           label=f'EC dimensionality ({d_ec})')
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Participation Ratio')
ax.set_title('Effective Dimensionality of DG Outputs')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Panel 4: Storage Capacity ---
ax = axes[1, 0]
full_accs = [capacity_results[n]["full_cue_accuracy"] for n in cap_ns]
deg_accs = [capacity_results[n]["degraded_cue_accuracy"] for n in cap_ns]
full_sims = [capacity_results[n]["full_cue_mean_sim"] for n in cap_ns]
ax.semilogx(cap_ns, full_accs, 'bo-', markersize=5, label='Full cue accuracy')
ax.semilogx(cap_ns, deg_accs, 'rs-', markersize=5, label='50% degraded accuracy')
ax.semilogx(cap_ns, full_sims, 'g^-', markersize=5, label='Full cue similarity')
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Retrieval quality')
ax.set_title('Storage Capacity (Lateral DG + Batch Cov)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# --- Panel 5: Lateral vs k-WTA DG Comparison ---
ax = axes[1, 1]
comp_ns = sorted(dg_comparison_results.keys())
lat_accs = [dg_comparison_results[n]["lateral_accuracy"] for n in comp_ns]
kwta_accs = [dg_comparison_results[n]["kwta_accuracy"] for n in comp_ns]
ax.semilogx(comp_ns, lat_accs, 'go-', markersize=6, linewidth=2,
            label='Lateral inhibition DG')
ax.semilogx(comp_ns, kwta_accs, 'r^-', markersize=6, linewidth=2,
            label='k-WTA DG')
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Retrieval accuracy')
ax.set_title('Lateral Inhibition vs k-WTA DG')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# --- Panel 6: Variable vs Fixed Sparsity ---
ax = axes[1, 2]
sp_ns = sorted(sparsity_comparison_results.keys())
var_accs = [sparsity_comparison_results[n]["variable_accuracy"] for n in sp_ns]
fix_accs = [sparsity_comparison_results[n]["fixed_accuracy"] for n in sp_ns]
ax.semilogx(sp_ns, var_accs, 'bo-', markersize=6, linewidth=2,
            label='Variable sparsity')
ax.semilogx(sp_ns, fix_accs, 'ms-', markersize=6, linewidth=2,
            label='Renormalized fixed sparsity')
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Retrieval accuracy')
ax.set_title('Variable vs Fixed Sparsity\n(same DG, k-WTA applied post-hoc)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# --- Panel 7: Pattern Completion at N=500 ---
ax = axes[2, 0]
if 500 in completion_results:
    load_data = completion_results[500]
    degs = sorted([float(k) for k in load_data.keys()])
    accs = [load_data[f"{d:.3f}"]["accuracy"] for d in degs]
    sims = [load_data[f"{d:.3f}"]["mean_similarity"] for d in degs]
    ax.plot(degs, accs, 'bo-', markersize=4, label='Accuracy')
    ax.plot(degs, sims, 'rs-', markersize=4, label='Similarity')
ax.set_xlabel('Fraction degraded')
ax.set_ylabel('Retrieval quality')
ax.set_title('Pattern Completion (N=500)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# --- Panel 8: Sparsity-Retrieval Correlation ---
ax = axes[2, 1]
cap_ns_corr = [n for n in cap_ns if capacity_results[n]["full_cue_accuracy"] < 0.98
               and capacity_results[n]["full_cue_accuracy"] > 0.05]
if cap_ns_corr:
    corrs = [capacity_results[n]["sparsity_retrieval_correlation"] for n in cap_ns_corr]
    ax.bar(range(len(cap_ns_corr)), corrs, color='teal', alpha=0.7)
    ax.set_xticks(range(len(cap_ns_corr)))
    ax.set_xticklabels([str(n) for n in cap_ns_corr], rotation=45)
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Correlation(sparsity, correct)')
ax.set_title('Per-Pattern Sparsity vs Retrieval Success')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.grid(True, alpha=0.3)

# --- Panel 9: Pattern Completion at N=1000 ---
ax = axes[2, 2]
if 1000 in completion_results:
    load_data = completion_results[1000]
    degs = sorted([float(k) for k in load_data.keys()])
    accs = [load_data[f"{d:.3f}"]["accuracy"] for d in degs]
    sims = [load_data[f"{d:.3f}"]["mean_similarity"] for d in degs]
    ax.plot(degs, accs, 'bo-', markersize=4, label='Accuracy')
    ax.plot(degs, sims, 'rs-', markersize=4, label='Similarity')
ax.set_xlabel('Fraction degraded')
ax.set_ylabel('Retrieval quality')
ax.set_title('Pattern Completion (N=1000)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('hippocampal_model_v3_results.png', dpi=150, bbox_inches='tight')

print(f"\nResults saved to hippocampal_model_v3_results.json")
print(f"Plot saved to hippocampal_model_v3_results.png")
