"""
Hippocampal-Cortical Consolidation Model
=========================================

Combines:
- The hippocampal model v4 (EC -> DG -> CA3 -> CA1 with BTSP)
- A world with known latent compositional structure (from v1)
- A cortical weight matrix that accumulates CA1 replay outputs
- Analysis of whether cortex recovers the latent structure

The consolidation loop:
1. Sample a stored EC pattern (simulating hippocampal replay initiation)
2. Run full retrieval: EC -> DG -> CA3 (pattern completion) -> CA1 (decoding)
3. Deposit CA1 output into cortical matrix via Hebbian outer product
4. After many cycles, extract cortical PCs and compare to ground truth

Key question: does the cortical matrix converge on the compositional
variables that generated the experiences, even though the hippocampus
stored the specific experiences in superposition?
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import subspace_angles

np.random.seed(42)

# =============================================================================
# 1. REUSED HIPPOCAMPAL COMPONENTS (from v4)
# =============================================================================

def make_feedforward_weights(D_output, d_input, connectivity_prob=0.33):
    mask = (np.random.rand(D_output, d_input) < connectivity_prob).astype(float)
    weights = np.random.randn(D_output, d_input) * mask
    row_norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-10
    return weights / row_norms


def build_ring_inhibition(D, sigma, connection_prob=None):
    positions = np.arange(D)
    dist = np.abs(positions[:, None] - positions[None, :])
    dist = np.minimum(dist, D - dist)
    W = np.exp(-dist**2 / (2 * sigma**2))
    W[dist > 3 * sigma] = 0
    np.fill_diagonal(W, 0)
    if connection_prob is not None:
        mask = (np.random.rand(D, D) < connection_prob).astype(float)
        np.fill_diagonal(mask, 0)
        W *= mask
    row_sums = W.sum(axis=1, keepdims=True) + 1e-10
    W /= row_sums
    return W


def apply_kwta(pattern, k):
    out = np.zeros_like(pattern)
    if k < len(pattern):
        top_k_idx = np.argpartition(pattern, -k)[-k:]
        out[top_k_idx] = pattern[top_k_idx]
    else:
        out = pattern.copy()
    return out


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)


class DentateGyrusLateral:
    def __init__(self, d_input, D_output, sigma_inh=25, gamma_inh=5.0,
                 n_inh_steps=5, noise_scale=0.0, W_ff=None,
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
            D_output, sigma_inh, connection_prob=inh_connection_prob)

    def forward(self, x):
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        results = []
        for i in range(x.shape[0]):
            h_raw = x[i] @ self.W_ff.T
            h_raw = np.maximum(h_raw, 0)
            if self.noise_scale > 0 and np.any(h_raw > 0):
                mean_active = np.mean(h_raw[h_raw > 0])
                noise = np.random.randn(self.D_output) * self.noise_scale * mean_active
                h_raw = np.maximum(h_raw + noise, 0)
            h = h_raw.copy()
            for step in range(self.n_inh_steps):
                inh = self.W_inh @ h
                h = np.maximum(h_raw - self.gamma_inh * inh, 0)
            results.append(h)
        out = np.array(results)
        if single:
            out = out[0]
        return out


class CA3:
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.stored_patterns = []
        self.mean_activity = np.zeros(N)

    def store_online(self, pattern):
        p = pattern.copy()
        norm = np.linalg.norm(p) + 1e-10
        p = p / norm
        self.n_stored += 1
        self.mean_activity = self.mean_activity + (p - self.mean_activity) / self.n_stored
        p_centered = p - self.mean_activity
        self.W += np.outer(p_centered, p_centered)
        np.fill_diagonal(self.W, 0)
        self.stored_patterns.append(pattern.copy())

    def retrieve(self, cue, n_iterations=5):
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


class CA1:
    def __init__(self, N_ca1, N_ca3, d_ec,
                 lr=0.3, plateau_threshold=0.7, plateau_sharpness=20.0,
                 weight_decay=0.9999, div_norm_sigma=0.1,
                 connectivity_prob=0.33):
        self.N_ca1 = N_ca1
        self.N_ca3 = N_ca3
        self.d_ec = d_ec
        self.lr = lr
        self.plateau_threshold = plateau_threshold
        self.plateau_sharpness = plateau_sharpness
        self.weight_decay = weight_decay
        self.div_norm_sigma = div_norm_sigma

        self.W_ta = make_feedforward_weights(N_ca1, d_ec, connectivity_prob)
        mask = (np.random.rand(N_ca1, N_ca3) < connectivity_prob).astype(float)
        self.W_sc = np.random.randn(N_ca1, N_ca3) * 0.01 * mask
        self.n_episodes = 0

    def _sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _divisive_normalize(self, h):
        pool = np.mean(h) + self.div_norm_sigma
        return h / pool

    def compute_activations(self, x_ca3, x_ec):
        h_ta_raw = np.maximum(self.W_ta @ x_ec, 0)
        h_sc_raw = np.maximum(self.W_sc @ x_ca3, 0)
        h_ta = self._divisive_normalize(h_ta_raw)
        h_sc = self._divisive_normalize(h_sc_raw)
        if np.max(h_ta_raw) > 1e-10:
            threshold = self.plateau_threshold * np.max(h_ta_raw)
        else:
            threshold = 0.0
        gate = self._sigmoid(self.plateau_sharpness * (h_ta_raw - threshold))
        return h_ta, h_sc, gate, h_ta_raw

    def encode(self, x_ca3, x_ec):
        h_ta, h_sc, gate, h_ta_raw = self.compute_activations(x_ca3, x_ec)
        error = h_ta - h_sc
        gated_error = gate * error
        h_ta_norm = np.linalg.norm(h_ta) + 1e-10
        mismatch = float(np.linalg.norm(gated_error) / h_ta_norm)
        delta_W = self.lr * np.outer(gated_error, x_ca3)
        self.W_sc += delta_W
        self.W_sc *= self.weight_decay
        self.n_episodes += 1
        return mismatch

    def retrieve(self, x_ca3, x_ec):
        h_ta, h_sc, gate, h_ta_raw = self.compute_activations(x_ca3, x_ec)
        error = h_ta - h_sc
        gated_error = gate * error
        h_ta_norm = np.linalg.norm(h_ta) + 1e-10
        mismatch = float(np.linalg.norm(gated_error) / h_ta_norm)
        return h_sc.copy(), mismatch

    def get_ec_target(self, x_ec):
        h_raw = np.maximum(self.W_ta @ x_ec, 0)
        return self._divisive_normalize(h_raw)


class HippocampalSystem:
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                 dg_params=None, ca1_params=None, ca3_retrieval_iterations=5):
        self.d_ec = d_ec
        self.ca3_retrieval_iterations = ca3_retrieval_iterations
        dg_p = dg_params or {}
        self.dg = DentateGyrusLateral(d_ec, D_dg, **dg_p)
        self.ca3 = CA3(N_ca3, k_ca3)
        ca1_p = ca1_params or {}
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **ca1_p)
        self.ec_store = []
        self.dg_store = []

    def encode_batch(self, ec_patterns, train_ca1=True):
        self.ec_store = [ec.copy() for ec in ec_patterns]
        dg_patterns = []
        ca1_mismatches = []
        for ec in ec_patterns:
            dg_pat = self.dg.forward(ec)
            dg_patterns.append(dg_pat)
            self.ca3.store_online(dg_pat)
            if train_ca1:
                mm = self.ca1.encode(dg_pat, ec)
                ca1_mismatches.append(mm)
        self.dg_store = dg_patterns
        return ca1_mismatches

    def retrieve_from_ec(self, ec_query):
        dg_output = self.dg.forward(ec_query)
        ca3_output = self.ca3.retrieve(dg_output, n_iterations=self.ca3_retrieval_iterations)
        ca1_output, mismatch = self.ca1.retrieve(ca3_output, ec_query)
        return {
            "dg_output": dg_output,
            "ca3_output": ca3_output,
            "ca1_output": ca1_output,
            "mismatch": mismatch,
        }


# =============================================================================
# 2. WORLD WITH LATENT COMPOSITIONAL STRUCTURE
# =============================================================================

k_latent = 5       # compositional dimensions (the "ground truth" PCA)
N_animals = 200     # specific animal types
d_ec = 100          # EC dimensionality
idiosyncratic_noise = 0.3

# Ground truth latent basis (k orthogonal directions in EC space)
latent_basis = np.linalg.qr(np.random.randn(d_ec, k_latent))[0][:, :k_latent]

# Generate animal types as points in latent space + noise
latent_coords = np.random.randn(N_animals, k_latent)
animal_ec_reps = latent_coords @ latent_basis.T
animal_ec_reps += idiosyncratic_noise * np.random.randn(N_animals, d_ec)
# Normalize to unit norm (EC patterns are unit vectors)
animal_ec_reps = animal_ec_reps / np.linalg.norm(animal_ec_reps, axis=1, keepdims=True)

print(f"World: {k_latent} latent dims, {N_animals} animal types, {d_ec}-dim EC space")
print(f"Idiosyncratic noise: {idiosyncratic_noise}")

# =============================================================================
# 3. HIPPOCAMPAL ENCODING
# =============================================================================

D_dg = 1000
N_ca3 = 1000
N_ca1 = 500
k_ca3 = 50

dg_params = {"sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5, "noise_scale": 0.0}
ca1_params = {"lr": 0.3, "plateau_threshold": 0.7, "plateau_sharpness": 20.0,
              "weight_decay": 0.9999, "div_norm_sigma": 0.1}

system = HippocampalSystem(d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                           dg_params=dg_params, ca1_params=ca1_params,
                           ca3_retrieval_iterations=5)

# Encode all animals (each seen once, single-pass)
print(f"\nEncoding {N_animals} patterns through hippocampal circuit...")
ca1_mismatches = system.encode_batch(animal_ec_reps, train_ca1=True)
print(f"  CA1 mismatch: first 10 mean={np.mean(ca1_mismatches[:10]):.4f}, "
      f"last 10 mean={np.mean(ca1_mismatches[-10:]):.4f}")

# =============================================================================
# 4. CONSOLIDATION: REPLAY INTO CORTICAL MATRIX
# =============================================================================

# The cortical matrix accumulates in CA1 output space (N_ca1 dimensions)
# We also track a matrix in EC space for comparison

print(f"\n{'='*60}")
print("CONSOLIDATION VIA REPLAY")
print(f"{'='*60}")

n_total_replays = 5000
checkpoints = [10, 25, 50, 100, 250, 500, 1000, 2000, 3000, 5000]

W_cortex_ca1 = np.zeros((N_ca1, N_ca1))   # accumulates CA1 outputs
W_cortex_ec = np.zeros((d_ec, d_ec))       # accumulates EC-space projections

# For comparison: direct accumulation of EC patterns (no hippocampus)
W_direct_ec = np.zeros((d_ec, d_ec))
for ec in animal_ec_reps:
    W_direct_ec += np.outer(ec, ec)

# Also build the "hippocampal matrix" in EC space for comparison
# (what you'd get from reading CA3 without CA1 decoding)
W_hippo_ec = np.zeros((d_ec, d_ec))

alignment_ca1_over_time = []
alignment_ec_over_time = []
spectral_concentration_over_time = []
replay_quality_over_time = []

for cycle in range(1, n_total_replays + 1):
    # Sample a stored pattern for replay
    idx = np.random.randint(N_animals)
    ec_cue = animal_ec_reps[idx]

    # Full retrieval through hippocampal circuit
    result = system.retrieve_from_ec(ec_cue)
    ca1_output = result["ca1_output"]

    # Deposit CA1 output into cortical matrix (Hebbian)
    ca1_norm = np.linalg.norm(ca1_output) + 1e-10
    ca1_normed = ca1_output / ca1_norm
    W_cortex_ca1 += np.outer(ca1_normed, ca1_normed)

    # Also project CA1 output back to EC space for alignment measurement
    # Use the fixed TA weights as a rough inverse mapping
    # (CA1 -> EC projection via transpose of EC -> CA1 weights)
    ec_reconstruction = system.ca1.W_ta.T @ ca1_output
    ec_recon_norm = np.linalg.norm(ec_reconstruction) + 1e-10
    ec_reconstruction = ec_reconstruction / ec_recon_norm
    W_cortex_ec += np.outer(ec_reconstruction, ec_reconstruction)

    # Hippocampal EC-space matrix: project CA3 output back
    # (for comparison, what you'd get without CA1)
    if cycle <= N_animals:
        W_hippo_ec += np.outer(ec_cue, ec_cue)  # just the raw EC patterns

    # Track replay quality
    target = system.ca1.get_ec_target(ec_cue)
    replay_sim = cosine_similarity(ca1_output, target)

    if cycle in checkpoints:
        # Analyze cortical matrix in CA1 space
        evals_ca1, evecs_ca1 = np.linalg.eigh(W_cortex_ca1)
        evals_ca1, evecs_ca1 = evals_ca1[::-1], evecs_ca1[:, ::-1]

        # Spectral concentration in CA1 space
        top_k_frac_ca1 = evals_ca1[:k_latent].sum() / (evals_ca1.sum() + 1e-10)
        spectral_concentration_over_time.append((cycle, float(top_k_frac_ca1)))

        # Analyze cortical matrix in EC space
        evals_ec, evecs_ec = np.linalg.eigh(W_cortex_ec)
        evals_ec, evecs_ec = evals_ec[::-1], evecs_ec[:, ::-1]

        # Alignment with true latent basis (in EC space)
        angles_ec = subspace_angles(evecs_ec[:, :k_latent], latent_basis)
        alignment_ec = float(np.cos(angles_ec).mean())
        alignment_ec_over_time.append((cycle, alignment_ec))

        # Spectral concentration in EC space
        top_k_frac_ec = evals_ec[:k_latent].sum() / (evals_ec.sum() + 1e-10)

        # Mean replay quality over last 100 cycles
        mean_replay_sim = replay_sim  # just current for checkpoints

        print(f"  Replay {cycle:5d}: EC align={alignment_ec:.4f}, "
              f"EC spectral top-{k_latent}={top_k_frac_ec:.4f}, "
              f"CA1 spectral top-{k_latent}={top_k_frac_ca1:.4f}")

# =============================================================================
# 5. BASELINES FOR COMPARISON
# =============================================================================

print(f"\n{'='*60}")
print("BASELINE COMPARISONS")
print(f"{'='*60}")

# Direct PCA on EC patterns
evals_direct, evecs_direct = np.linalg.eigh(W_direct_ec)
evals_direct, evecs_direct = evals_direct[::-1], evecs_direct[:, ::-1]
angles_direct = subspace_angles(evecs_direct[:, :k_latent], latent_basis)
align_direct = float(np.cos(angles_direct).mean())
spectral_direct = evals_direct[:k_latent].sum() / evals_direct.sum()

# Hippocampal matrix (raw EC patterns accumulated)
evals_hippo, evecs_hippo = np.linalg.eigh(W_hippo_ec)
evals_hippo, evecs_hippo = evals_hippo[::-1], evecs_hippo[:, ::-1]
angles_hippo = subspace_angles(evecs_hippo[:, :k_latent], latent_basis)
align_hippo = float(np.cos(angles_hippo).mean())
spectral_hippo = evals_hippo[:k_latent].sum() / evals_hippo.sum()

# Final cortical matrix in EC space
evals_cortex_final, evecs_cortex_final = np.linalg.eigh(W_cortex_ec)
evals_cortex_final = evals_cortex_final[::-1]
evecs_cortex_final = evecs_cortex_final[:, ::-1]
angles_cortex = subspace_angles(evecs_cortex_final[:, :k_latent], latent_basis)
align_cortex = float(np.cos(angles_cortex).mean())
spectral_cortex = evals_cortex_final[:k_latent].sum() / evals_cortex_final.sum()

print(f"\nSubspace alignment with true latent basis (mean cos of principal angles):")
print(f"  Direct PCA on EC patterns:      {align_direct:.4f}")
print(f"  Hippocampal matrix (raw EC):    {align_hippo:.4f}")
print(f"  Cortical matrix ({n_total_replays} replays): {align_cortex:.4f}")

print(f"\nSpectral concentration (fraction in top-{k_latent}):")
print(f"  Direct PCA:   {spectral_direct:.4f}")
print(f"  Hippocampal:  {spectral_hippo:.4f}")
print(f"  Cortical:     {spectral_cortex:.4f}")

# Per-component alignment
angles_direct_per = np.cos(subspace_angles(evecs_direct[:, :k_latent], latent_basis))
angles_hippo_per = np.cos(subspace_angles(evecs_hippo[:, :k_latent], latent_basis))
angles_cortex_per = np.cos(subspace_angles(evecs_cortex_final[:, :k_latent], latent_basis))

print(f"\nPer-component alignment:")
print(f"  Direct:  {np.array2string(angles_direct_per, precision=4)}")
print(f"  Hippo:   {np.array2string(angles_hippo_per, precision=4)}")
print(f"  Cortex:  {np.array2string(angles_cortex_per, precision=4)}")

# =============================================================================
# 6. EXPERIMENT: Does multi-pass replay improve cortical structure?
# =============================================================================

print(f"\n{'='*60}")
print("MULTI-PASS CONSOLIDATION EXPERIMENT")
print(f"{'='*60}")

# Reset cortex, do consolidation in passes with CA1 re-training between passes
n_consolidation_passes = 5
replays_per_pass = 1000

system_mp = HippocampalSystem(d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                               dg_params=dg_params, ca1_params=ca1_params,
                               ca3_retrieval_iterations=5)

# Initial encoding
system_mp.encode_batch(animal_ec_reps, train_ca1=True)

W_cortex_mp = np.zeros((d_ec, d_ec))
multipass_results = []

for pass_num in range(n_consolidation_passes):
    # Replay phase: retrieve and accumulate
    pass_sims = []
    for cycle in range(replays_per_pass):
        idx = np.random.randint(N_animals)
        ec_cue = animal_ec_reps[idx]
        result = system_mp.retrieve_from_ec(ec_cue)
        ca1_out = result["ca1_output"]

        # Project to EC space and accumulate
        ec_recon = system_mp.ca1.W_ta.T @ ca1_out
        ec_recon = ec_recon / (np.linalg.norm(ec_recon) + 1e-10)
        W_cortex_mp += np.outer(ec_recon, ec_recon)

        target = system_mp.ca1.get_ec_target(ec_cue)
        pass_sims.append(cosine_similarity(ca1_out, target))

    # CA1 re-training phase (simulate replay improving the decoder)
    order = np.random.permutation(N_animals)
    retrain_mm = []
    for idx in order:
        dg_pat = system_mp.dg_store[idx]
        mm = system_mp.ca1.encode(dg_pat, animal_ec_reps[idx])
        retrain_mm.append(mm)

    # Analyze cortical matrix
    evals_mp, evecs_mp = np.linalg.eigh(W_cortex_mp)
    evals_mp, evecs_mp = evals_mp[::-1], evecs_mp[:, ::-1]
    angles_mp = subspace_angles(evecs_mp[:, :k_latent], latent_basis)
    align_mp = float(np.cos(angles_mp).mean())
    spectral_mp = evals_mp[:k_latent].sum() / (evals_mp.sum() + 1e-10)

    total_replays = (pass_num + 1) * replays_per_pass
    multipass_results.append({
        "pass": pass_num,
        "total_replays": total_replays,
        "alignment": align_mp,
        "spectral_concentration": float(spectral_mp),
        "mean_replay_sim": float(np.mean(pass_sims)),
        "mean_retrain_mismatch": float(np.mean(retrain_mm)),
    })

    print(f"  Pass {pass_num+1}: align={align_mp:.4f}, "
          f"spectral={spectral_mp:.4f}, "
          f"replay_sim={np.mean(pass_sims):.4f}, "
          f"retrain_mm={np.mean(retrain_mm):.4f}")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

results = {
    "world": {
        "k_latent": k_latent,
        "N_animals": N_animals,
        "d_ec": d_ec,
        "idiosyncratic_noise": idiosyncratic_noise,
    },
    "hippocampal_params": {
        "D_dg": D_dg, "N_ca3": N_ca3, "N_ca1": N_ca1, "k_ca3": k_ca3,
        "dg_params": dg_params, "ca1_params": ca1_params,
    },
    "consolidation": {
        "n_total_replays": n_total_replays,
        "alignment_ec_over_time": alignment_ec_over_time,
        "spectral_concentration_over_time": spectral_concentration_over_time,
    },
    "baselines": {
        "direct_pca": {
            "alignment": align_direct,
            "spectral_concentration": float(spectral_direct),
            "per_component": angles_direct_per.tolist(),
            "eigenvalues_top15": evals_direct[:15].tolist(),
        },
        "hippocampal": {
            "alignment": align_hippo,
            "spectral_concentration": float(spectral_hippo),
            "per_component": angles_hippo_per.tolist(),
            "eigenvalues_top15": evals_hippo[:15].tolist(),
        },
        "cortical_final": {
            "alignment": align_cortex,
            "spectral_concentration": float(spectral_cortex),
            "per_component": angles_cortex_per.tolist(),
            "eigenvalues_top15": evals_cortex_final[:15].tolist(),
        },
    },
    "multipass_consolidation": multipass_results,
    "ca1_encoding_mismatches": {
        "first_10": ca1_mismatches[:10],
        "last_10": ca1_mismatches[-10:],
    },
}

with open('consolidation_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# 8. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Hippocampal-Cortical Consolidation: Does Replay Recover Latent Structure?",
             fontsize=14, fontweight='bold')

# Panel 1: Alignment convergence over replay cycles
ax = axes[0, 0]
cycles_ec = [c for c, _ in alignment_ec_over_time]
aligns_ec = [a for _, a in alignment_ec_over_time]
ax.semilogx(cycles_ec, aligns_ec, 'gs-', markersize=6, linewidth=2,
            label='Cortical (replay)')
ax.axhline(y=align_direct, color='blue', linestyle='--', alpha=0.7,
           label=f'Direct PCA ({align_direct:.3f})')
ax.axhline(y=align_hippo, color='red', linestyle='--', alpha=0.7,
           label=f'Hippocampal ({align_hippo:.3f})')
ax.set_xlabel('Number of replay cycles')
ax.set_ylabel('Mean subspace alignment')
ax.set_title('Latent Structure Recovery')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

# Panel 2: Spectral concentration over replay
ax = axes[0, 1]
cycles_sp = [c for c, _ in spectral_concentration_over_time]
specs = [s for _, s in spectral_concentration_over_time]
ax.semilogx(cycles_sp, specs, 'gs-', markersize=6, linewidth=2,
            label='Cortical (CA1 space)')
ax.axhline(y=spectral_direct, color='blue', linestyle='--', alpha=0.7,
           label=f'Direct PCA ({spectral_direct:.3f})')
ax.axhline(y=spectral_hippo, color='red', linestyle='--', alpha=0.7,
           label=f'Hippocampal ({spectral_hippo:.3f})')
ax.set_xlabel('Number of replay cycles')
ax.set_ylabel(f'Fraction of variance in top-{k_latent}')
ax.set_title('Spectral Concentration')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 3: Eigenvalue spectra comparison
ax = axes[0, 2]
n_show = 15
ax.plot(evals_direct[:n_show] / evals_direct.sum(), 'b^-', markersize=5,
        label='Direct PCA')
ax.plot(evals_hippo[:n_show] / evals_hippo.sum(), 'ro-', markersize=5,
        label='Hippocampal')
ax.plot(evals_cortex_final[:n_show] / evals_cortex_final.sum(), 'gs-', markersize=5,
        label='Cortical')
ax.axvline(x=k_latent-0.5, color='gray', linestyle='--', alpha=0.5,
           label=f'k={k_latent} true dims')
ax.set_xlabel('Component index')
ax.set_ylabel('Fraction of total variance')
ax.set_title('Eigenvalue Spectra')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 4: Per-component alignment
ax = axes[1, 0]
x_pos = np.arange(k_latent)
width = 0.25
ax.bar(x_pos - width, angles_direct_per, width, label='Direct PCA',
       color='steelblue', alpha=0.7)
ax.bar(x_pos, angles_hippo_per, width, label='Hippocampal',
       color='firebrick', alpha=0.7)
ax.bar(x_pos + width, angles_cortex_per, width, label='Cortical',
       color='forestgreen', alpha=0.7)
ax.set_xlabel('Principal angle index')
ax.set_ylabel('Cosine alignment')
ax.set_title('Per-Component Subspace Alignment')
ax.set_xticks(x_pos)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 5: Multi-pass consolidation
ax = axes[1, 1]
mp_passes = [r["pass"]+1 for r in multipass_results]
mp_align = [r["alignment"] for r in multipass_results]
mp_spec = [r["spectral_concentration"] for r in multipass_results]
ax.plot(mp_passes, mp_align, 'go-', markersize=6, linewidth=2, label='Alignment')
ax.plot(mp_passes, mp_spec, 'bs-', markersize=6, linewidth=2, label='Spectral conc.')
ax.set_xlabel('Consolidation pass')
ax.set_ylabel('Quality metric')
ax.set_title('Multi-Pass Consolidation')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 6: Multi-pass CA1 improvement
ax = axes[1, 2]
mp_replay_sim = [r["mean_replay_sim"] for r in multipass_results]
mp_retrain_mm = [r["mean_retrain_mismatch"] for r in multipass_results]
ax2 = ax.twinx()
l1 = ax.plot(mp_passes, mp_replay_sim, 'g-o', markersize=5, label='Replay similarity')
l2 = ax2.plot(mp_passes, mp_retrain_mm, 'b-s', markersize=5, label='Retrain mismatch')
ax.set_xlabel('Consolidation pass')
ax.set_ylabel('CA1 replay similarity', color='green')
ax2.set_ylabel('CA1 retrain mismatch', color='blue')
ax.set_title('CA1 Improvement Over Passes')
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('consolidation_model_results.png', dpi=150, bbox_inches='tight')

print(f"\nResults saved to consolidation_model_results.json")
print(f"Plot saved to consolidation_model_results.png")
