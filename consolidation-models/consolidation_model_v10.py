"""
Hippocampal Consolidation v10: 1000-dim CA1 + Sub, Element-wise Addition
=========================================================================

Architecture:
  Cortical input (d_ec=1000)
        |
  EC Superficial (II/III) -- 1000-dim, passthrough
        |
        +---> DG(1000)   [perforant path: sparsification within 1000-dim]
        +---> CA3(1000)   [direct EC-CA3]
        +---> CA1(1000)   [temporoammonic L3, via W_ta]
        +---> Sub(1000)   [temporoammonic L3, via W_ec]
        |
  DG -> CA3 -> CA1 -> Sub
                |      |
           CA1 -+------+--> element-wise sum --> EC Vb (1000-dim)
                                                   |
                                              EC output --> cortex

Key changes from v9:
  - CA1 and Sub both 1000-dim (matching EC, DG, CA3).
  - EC Vb is pure element-wise addition: no learned backprojection weights.
    CA1 provides sparse (~20-25% active), episode-specific signal.
    Sub provides dense (~50% active), statistically accumulated signal.
    Their sum yields ~60% active, recovering the density distribution of
    the original EC input.
  - This eliminates the format conversion problem entirely. Both signals
    are trained to approximate EC input through their respective TA
    pathways, and the sum is a noise-reducing combination of two estimates.
  - Sub LTD reduced to near-zero to prevent sparse CA1 inputs from being
    catastrophically depressed before learning can accumulate.
  - CA1 gamma_inh increased to 5.0 for sparser CA1 codes.
  - No cortex experiments. Focus on hippocampal pipeline quality.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)


# =============================================================================
# 1. UTILITY FUNCTIONS
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


# =============================================================================
# 2. DENTATE GYRUS
# =============================================================================

class DentateGyrusLateral:
    def __init__(self, d_input, D_output, sigma_inh=50, gamma_inh=1.0,
                 n_inh_steps=5, noise_scale=0.0, W_ff=None,
                 inh_connection_prob=None):
        self.d_input = d_input
        self.D_output = D_output
        if W_ff is not None:
            self.W_ff = W_ff.copy()
        else:
            self.W_ff = make_feedforward_weights(D_output, d_input)
        self.W_inh = build_ring_inhibition(
            D_output, sigma_inh, connection_prob=inh_connection_prob)
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.noise_scale = noise_scale

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
        return out[0] if single else out


# =============================================================================
# 3. CA3
# =============================================================================

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
        self.mean_activity += (p - self.mean_activity) / self.n_stored
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


# =============================================================================
# 4. CA1
# =============================================================================

class CA1:
    def __init__(self, N_ca1, N_ca3, d_ec,
                 lr=0.3, plateau_threshold=0.7, plateau_sharpness=20.0,
                 weight_decay=1.0, div_norm_sigma=0.1,
                 connectivity_prob=0.33,
                 ltd_rate=0.05, ltd_ca3_threshold=0.0,
                 sigma_inh=25, gamma_inh=3.0, n_inh_steps=5,
                 inh_connection_prob=None, E_inh=-0.1):
        self.N_ca1 = N_ca1
        self.N_ca3 = N_ca3
        self.d_ec = d_ec
        self.lr = lr
        self.plateau_threshold = plateau_threshold
        self.plateau_sharpness = plateau_sharpness
        self.weight_decay = weight_decay
        self.div_norm_sigma = div_norm_sigma
        self.ltd_rate = ltd_rate
        self.ltd_ca3_threshold = ltd_ca3_threshold
        self.E_inh = E_inh
        self.n_inh_steps = n_inh_steps
        self.gamma_inh = gamma_inh

        self.W_ta = make_feedforward_weights(N_ca1, d_ec, connectivity_prob)
        mask = (np.random.rand(N_ca1, N_ca3) < connectivity_prob).astype(float)
        self.W_sc = np.random.randn(N_ca1, N_ca3) * 0.01 * mask
        self.connectivity_mask = mask.copy()
        self.W_inh = build_ring_inhibition(
            N_ca1, sigma_inh, connection_prob=inh_connection_prob)
        self.n_episodes = 0

    def _sigmoid(self, x):
        return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

    def _divisive_normalize(self, h):
        pool = np.mean(h) + self.div_norm_sigma
        return h / pool

    def compute_activations(self, x_ca3, x_ec):
        h_ta_raw = np.maximum(self.W_ta @ x_ec, 0)
        h_sc_raw = np.maximum(self.W_sc @ x_ca3, 0)
        h_ta = self._divisive_normalize(h_ta_raw)
        h_sc = self._divisive_normalize(h_sc_raw)
        threshold = self.plateau_threshold * np.max(h_ta_raw) if np.max(h_ta_raw) > 1e-10 else 0.0
        gate = self._sigmoid(self.plateau_sharpness * (h_ta_raw - threshold))
        return h_ta, h_sc, gate, h_ta_raw, h_sc_raw

    def encode(self, x_ca3, x_ec):
        h_ta, h_sc, gate, h_ta_raw, h_sc_raw = self.compute_activations(x_ca3, x_ec)
        error = h_ta - h_sc
        gated_error = gate * error
        h_ta_norm = np.linalg.norm(h_ta) + 1e-10
        mismatch = float(np.linalg.norm(gated_error) / h_ta_norm)
        self.W_sc += self.lr * np.outer(gated_error, x_ca3)
        ca3_inactive = (x_ca3 <= self.ltd_ca3_threshold).astype(float)
        self.W_sc *= (1.0 - self.ltd_rate * np.outer(gate, ca3_inactive))
        if self.weight_decay < 1.0:
            self.W_sc *= self.weight_decay
        self.W_sc *= self.connectivity_mask
        self.n_episodes += 1
        return mismatch

    def retrieve(self, x_ca3, x_ec):
        h_ta, h_sc, gate, h_ta_raw, h_sc_raw = self.compute_activations(x_ca3, x_ec)
        error = h_ta - h_sc
        gated_error = gate * error
        h_ta_norm = np.linalg.norm(h_ta) + 1e-10
        mismatch = float(np.linalg.norm(gated_error) / h_ta_norm)
        h_out = h_sc_raw.copy()
        for step in range(self.n_inh_steps):
            g_inh = self.gamma_inh * (self.W_inh @ h_out)
            h_out = np.maximum(
                (h_sc_raw + g_inh * self.E_inh) / (1.0 + g_inh), 0.0)
        return h_out, mismatch


# =============================================================================
# 5. SUBICULUM (1000-dim, linear de-sparsifier, minimal LTD)
# =============================================================================

class Subiculum:
    """
    Regular-spiking EC-projecting population. Linear + ReLU.
    1000-dim, same as CA1 and EC. Slow learning (conventional LTP).
    LTD minimized to prevent catastrophic weight erasure from sparse CA1.
    """
    def __init__(self, N_sub, N_ca1, d_ec,
                 lr=0.05, ltd_rate=0.001, connectivity_prob=0.33):
        self.N_sub = N_sub
        self.N_ca1 = N_ca1
        self.d_ec = d_ec
        self.lr = lr
        self.ltd_rate = ltd_rate

        mask_ca1 = (np.random.rand(N_sub, N_ca1) < connectivity_prob).astype(float)
        self.W_ca1 = np.random.randn(N_sub, N_ca1) * 0.01 * mask_ca1
        self.mask_ca1 = mask_ca1.copy()

        self.W_ec = make_feedforward_weights(N_sub, d_ec, connectivity_prob)
        self.n_episodes = 0

    def encode(self, ca1_output, ec_pattern):
        h_ca1 = self.W_ca1 @ ca1_output
        h_ec = self.W_ec @ ec_pattern
        h_sub = np.maximum(h_ca1 + h_ec, 0)

        error = h_ec - h_ca1
        self.W_ca1 += self.lr * np.outer(error, ca1_output)

        # Minimal LTD
        if self.ltd_rate > 0:
            ca1_inactive = (ca1_output <= 0).astype(float)
            sub_active = (h_sub > 0).astype(float)
            self.W_ca1 *= (1.0 - self.ltd_rate * np.outer(sub_active, ca1_inactive))
        self.W_ca1 *= self.mask_ca1

        # Row-norm clipping
        row_norms = np.linalg.norm(self.W_ca1, axis=1, keepdims=True) + 1e-10
        max_norm = np.percentile(row_norms, 95)
        if max_norm > 1e-10:
            self.W_ca1 = np.where(
                row_norms > max_norm,
                self.W_ca1 * (max_norm / row_norms), self.W_ca1)

        self.n_episodes += 1
        return h_sub

    def replay(self, ca1_output):
        return np.maximum(self.W_ca1 @ ca1_output, 0)


# =============================================================================
# 6. ENTORHINAL CORTEX (element-wise addition at Vb, no learned weights)
# =============================================================================

class EntorhinalCortex:
    """
    EC with superficial (II/III) and deep (Vb) compartments.

    Superficial: passthrough (1000-dim).
    Deep Vb: element-wise addition of CA1 + Sub outputs.

    No learned backprojection weights. Both CA1 and Sub are 1000-dim
    and trained to approximate EC input through their TA pathways.
    Their sum combines a sparse episode-specific signal (CA1) with a
    dense statistical signal (Sub), recovering ~60% active fraction
    that approximates the original EC input distribution.

    Mismatch detection: compare the sum with EC II/III input (available
    during encoding, absent during replay).
    """

    def __init__(self, d_ec):
        self.d_ec = d_ec

    def get_superficial_output(self, cortical_input):
        return cortical_input.copy()

    def combine(self, ca1_output, sub_output):
        """Element-wise addition at Vb."""
        return ca1_output + sub_output

    def compute_mismatch(self, ec_superficial, ca1_output, sub_output):
        """
        Mismatch between hippocampal reconstruction and current input.
        Measured as cosine distance (1 - cosine_sim).
        """
        h_sum = ca1_output + sub_output
        return 1.0 - cosine_similarity(h_sum, ec_superficial)


# =============================================================================
# 7. HIPPOCAMPAL SYSTEM
# =============================================================================

class HippocampalSystem:
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                 dg_params=None, ca1_params=None, sub_params=None,
                 N_sub=1000, ca3_retrieval_iterations=5):
        self.d_ec = d_ec
        self.ca3_retrieval_iterations = ca3_retrieval_iterations

        self.ec = EntorhinalCortex(d_ec)
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}))
        self.ca3 = CA3(N_ca3, k_ca3)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}))
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}))

        self.ec_store = []

    def encode_batch(self, ec_patterns):
        self.ec_store = [ec.copy() for ec in ec_patterns]
        ca1_mismatches = []

        for ec in ec_patterns:
            ec_sup = self.ec.get_superficial_output(ec)
            dg_out = self.dg.forward(ec_sup)
            self.ca3.store_online(dg_out)
            self.ca1.encode(dg_out, ec_sup)

            ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
            ca1_out, mm = self.ca1.retrieve(ca3_out, ec_sup)
            ca1_mismatches.append(mm)

            self.sub.encode(ca1_out, ec_sup)

        return ca1_mismatches

    def replay_to_output(self, ec_query):
        ec_sup = self.ec.get_superficial_output(ec_query)
        dg_out = self.dg.forward(ec_sup)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, ca1_mm = self.ca1.retrieve(ca3_out, ec_sup)
        sub_out = self.sub.replay(ca1_out)
        ec_output = self.ec.combine(ca1_out, sub_out)
        return ec_output, {
            "ca1_mismatch": ca1_mm,
            "ca1_active_frac": float(np.mean(ca1_out > 0)),
            "sub_active_frac": float(np.mean(sub_out > 0)),
            "combined_active_frac": float(np.mean(ec_output > 0)),
        }

    def generate_replay_batch(self):
        return [self.replay_to_output(ec)[0] for ec in self.ec_store]

    def get_stage_outputs(self, ec_pattern):
        ec_sup = self.ec.get_superficial_output(ec_pattern)
        dg_out = self.dg.forward(ec_sup)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, _ = self.ca1.retrieve(ca3_out, ec_sup)
        sub_out = self.sub.replay(ca1_out)
        combined = self.ec.combine(ca1_out, sub_out)
        return {
            "ec_sup": ec_sup, "dg": dg_out, "ca3": ca3_out,
            "ca1": ca1_out, "sub": sub_out, "combined": combined,
        }


# =============================================================================
# 8. WORLD MODEL
# =============================================================================

class SparseFeatureWorld:
    def __init__(self, N_features, d_ec, sparsity_base=0.1, sparsity_decay=0.99):
        self.N_features = N_features
        self.d_ec = d_ec
        self.sparsities = np.clip(
            [sparsity_base * (sparsity_decay ** i) for i in range(N_features)],
            0.005, 0.5)
        self.sparsities = np.array(self.sparsities)
        raw = np.random.randn(d_ec, N_features)
        col_norms = np.linalg.norm(raw, axis=0, keepdims=True) + 1e-10
        self.W_ec = raw / col_norms
        self.W_ec_pinv = np.linalg.pinv(self.W_ec)

    def generate_observation(self):
        active = (np.random.rand(self.N_features) < self.sparsities).astype(float)
        values = np.random.uniform(0, 1, self.N_features)
        features = active * values
        ec = self.W_ec @ features
        return features, ec

    def generate_batch(self, n):
        features_list, ec_list = [], []
        for _ in range(n):
            f, ec = self.generate_observation()
            features_list.append(f)
            ec_list.append(ec)
        return np.array(features_list), np.array(ec_list)


# =============================================================================
# 9. ANALYSIS TOOLS
# =============================================================================

def measure_stage(reprs, dim, features_probe=None, world=None):
    active_frac = float(np.mean(reprs > 0))

    cov = np.cov(reprs.T)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
    total_var = np.sum(eigvals) + 1e-10
    pr = total_var ** 2 / (np.sum(eigvals ** 2) + 1e-10)

    n_sample = min(200, len(reprs))
    idx = np.random.choice(len(reprs), n_sample, replace=False)
    pair_sims = []
    for ii in range(n_sample):
        for jj in range(ii + 1, min(ii + 20, n_sample)):
            pair_sims.append(cosine_similarity(reprs[idx[ii]], reprs[idx[jj]]))

    decode_r2_mean = 0.0
    if features_probe is not None and world is not None:
        lam = 0.01
        n_feat = min(100, world.N_features)
        HTH_inv = np.linalg.inv(reprs.T @ reprs + lam * np.eye(reprs.shape[1]))
        r2s = []
        for fi in range(n_feat):
            y = features_probe[:len(reprs), fi]
            w = HTH_inv @ (reprs.T @ y)
            y_hat = reprs @ w
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
            r2s.append(max(0.0, 1.0 - ss_res / ss_tot))
        decode_r2_mean = float(np.mean(r2s))

    return {
        "n_dims": dim,
        "active_fraction": active_frac,
        "participation_ratio": float(pr),
        "pr_normalized": float(pr / dim) if dim > 0 else 0.0,
        "mean_pairwise_sim": float(np.mean(pair_sims)) if pair_sims else 0.0,
        "feature_decode_r2_mean": decode_r2_mean,
    }


# =============================================================================
# 10. PARAMETERS
# =============================================================================

d_ec = 1000
N_features = 200
sparsity_base = 0.1
sparsity_decay = 0.99

D_dg = 1000
N_ca3 = 1000
N_ca1 = 1000         # same as EC
N_sub = 1000          # same as CA1
k_ca3 = 50
ca3_retrieval_iters = 5

dg_params = {
    "sigma_inh": 25, "gamma_inh": 5.0,
    "n_inh_steps": 5, "noise_scale": 0.0,
}

ca1_params = {
    "lr": 0.3, "plateau_threshold": 0.7, "plateau_sharpness": 20.0,
    "weight_decay": 1.0, "div_norm_sigma": 0.0,
    "ltd_rate": 0.05, "ltd_ca3_threshold": 0.0,
    "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5, "E_inh": -0.4,
}

sub_params = {
    "lr": 0.3,
    "ltd_rate": 0.001,        # minimal LTD
    "connectivity_prob": 0.33,
}

N_experience = 500
N_probe = 500        # reduced for speed with larger matrices

print("=" * 70)
print("HIPPOCAMPAL CONSOLIDATION v10: 1000-dim CA1+Sub, ELEMENT-WISE ADD")
print("=" * 70)
print(f"\nWorld: {N_features} features -> EC({d_ec})")
print(f"Hippocampus: EC({d_ec}) -> DG({D_dg}) -> CA3({N_ca3}) -> CA1({N_ca1})")
print(f"  Sub({N_sub}), output = CA1 + Sub (element-wise)")
print(f"Data: {N_experience} experiences, {N_probe} probes")


# =============================================================================
# 11. GENERATE WORLD
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 1: Generating world")
print("=" * 70)

world = SparseFeatureWorld(N_features, d_ec,
                           sparsity_base=sparsity_base,
                           sparsity_decay=sparsity_decay)
features_exp, ec_exp = world.generate_batch(N_experience)
features_probe, ec_probe = world.generate_batch(N_probe)

print(f"  {N_experience} experiences, {N_probe} probes")
print(f"  Mean features active: {np.mean(np.sum(features_exp > 0, axis=1)):.1f}")
print(f"  EC pattern mean norm: {np.mean(np.linalg.norm(ec_exp, axis=1)):.3f}")


# =============================================================================
# 12. HIPPOCAMPAL ENCODING
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 2: Hippocampal encoding")
print("=" * 70)

hippocampus = HippocampalSystem(
    d_ec, D_dg, N_ca3, N_ca1, k_ca3,
    dg_params=dg_params, ca1_params=ca1_params,
    sub_params=sub_params, N_sub=N_sub,
    ca3_retrieval_iterations=ca3_retrieval_iters)

ca1_mm = hippocampus.encode_batch(ec_exp)
print(f"  Encoded {N_experience} patterns")
print(f"  CA1 mismatch: first={ca1_mm[0]:.4f}, last={ca1_mm[-1]:.4f}, "
      f"mean={np.mean(ca1_mm):.4f}")


# =============================================================================
# 13. REPLAY QUALITY
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 3: Replay quality")
print("=" * 70)

replay_signals = hippocampus.generate_replay_batch()

# Cosine similarity of combined replay to original EC input
replay_sims = [cosine_similarity(ec_exp[i], replay_signals[i])
               for i in range(N_experience)]
print(f"  Combined (CA1+Sub) replay cosine sim to EC: "
      f"{np.mean(replay_sims):.4f} +/- {np.std(replay_sims):.4f}")
print(f"  Min: {np.min(replay_sims):.4f}, Max: {np.max(replay_sims):.4f}")

# Also measure CA1-only and Sub-only replay quality
ca1_only_sims = []
sub_only_sims = []
combined_fracs = {"ca1": [], "sub": [], "combined": []}

for i, ec in enumerate(ec_exp[:200]):
    stages = hippocampus.get_stage_outputs(ec)
    ca1_only_sims.append(cosine_similarity(ec, stages["ca1"]))
    sub_only_sims.append(cosine_similarity(ec, stages["sub"]))
    combined_fracs["ca1"].append(float(np.mean(stages["ca1"] > 0)))
    combined_fracs["sub"].append(float(np.mean(stages["sub"] > 0)))
    combined_fracs["combined"].append(float(np.mean(stages["combined"] > 0)))

print(f"\n  Individual component replay (first 200):")
print(f"    CA1 only cosine sim:  {np.mean(ca1_only_sims):.4f}")
print(f"    Sub only cosine sim:  {np.mean(sub_only_sims):.4f}")
print(f"    Combined cosine sim:  {np.mean(replay_sims[:200]):.4f}")

print(f"\n  Active fractions during replay:")
print(f"    CA1:      {np.mean(combined_fracs['ca1']):.3f}")
print(f"    Sub:      {np.mean(combined_fracs['sub']):.3f}")
print(f"    Combined: {np.mean(combined_fracs['combined']):.3f}")


# =============================================================================
# 14. MISMATCH DETECTION
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 4: Mismatch detection")
print("=" * 70)

familiar_mm = []
for ec in ec_exp[:200]:
    mm = hippocampus.ec.compute_mismatch(
        ec, *[hippocampus.get_stage_outputs(ec)[k] for k in ["ca1", "sub"]])
    familiar_mm.append(mm)

novel_mm = []
for ec in ec_probe[:200]:
    mm = hippocampus.ec.compute_mismatch(
        ec, *[hippocampus.get_stage_outputs(ec)[k] for k in ["ca1", "sub"]])
    novel_mm.append(mm)

print(f"  Familiar mismatch (cosine dist): {np.mean(familiar_mm):.4f} "
      f"+/- {np.std(familiar_mm):.4f}")
print(f"  Novel mismatch (cosine dist):    {np.mean(novel_mm):.4f} "
      f"+/- {np.std(novel_mm):.4f}")
print(f"  Separation: {np.mean(novel_mm) - np.mean(familiar_mm):.4f}")

t_stat, p_val = stats.ttest_ind(novel_mm, familiar_mm)
print(f"  t-test: t={t_stat:.3f}, p={p_val:.2e}")


# =============================================================================
# 15. STAGE REPRESENTATION ANALYSIS
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 5: Representation quality by stage")
print("=" * 70)

n_analysis = min(300, N_probe)
stage_reprs = {k: [] for k in ["EC", "DG", "CA3", "CA1", "Sub", "CA1+Sub"]}

for ec in ec_probe[:n_analysis]:
    stages = hippocampus.get_stage_outputs(ec)
    stage_reprs["EC"].append(stages["ec_sup"])
    stage_reprs["DG"].append(stages["dg"])
    stage_reprs["CA3"].append(stages["ca3"])
    stage_reprs["CA1"].append(stages["ca1"])
    stage_reprs["Sub"].append(stages["sub"])
    stage_reprs["CA1+Sub"].append(stages["combined"])

stage_dims = {"EC": d_ec, "DG": D_dg, "CA3": N_ca3,
              "CA1": N_ca1, "Sub": N_sub, "CA1+Sub": d_ec}

stage_results = {}
for name in ["EC", "DG", "CA3", "CA1", "Sub", "CA1+Sub"]:
    reprs = np.array(stage_reprs[name])
    result = measure_stage(reprs, stage_dims[name], features_probe, world)
    stage_results[name] = result
    print(f"  {name:>7s} (d={stage_dims[name]:4d}): "
          f"active={result['active_fraction']:.3f}, "
          f"PR={result['participation_ratio']:.1f} "
          f"({result['pr_normalized']:.3f}), "
          f"pair_sim={result['mean_pairwise_sim']:.4f}, "
          f"decode_R2={result['feature_decode_r2_mean']:.4f}")


# =============================================================================
# 16. SAVE RESULTS
# =============================================================================

print(f"\n{'=' * 70}")
print("Saving results...")
print("=" * 70)

results = {
    "params": {
        "d_ec": d_ec, "N_features": N_features,
        "D_dg": D_dg, "N_ca3": N_ca3, "N_ca1": N_ca1, "N_sub": N_sub,
        "k_ca3": k_ca3, "N_experience": N_experience, "N_probe": N_probe,
        "ca1_params": ca1_params, "sub_params": sub_params,
    },
    "replay_quality": {
        "combined_mean_sim": float(np.mean(replay_sims)),
        "combined_std_sim": float(np.std(replay_sims)),
        "ca1_only_mean_sim": float(np.mean(ca1_only_sims)),
        "sub_only_mean_sim": float(np.mean(sub_only_sims)),
    },
    "active_fractions": {
        "ca1": float(np.mean(combined_fracs["ca1"])),
        "sub": float(np.mean(combined_fracs["sub"])),
        "combined": float(np.mean(combined_fracs["combined"])),
    },
    "mismatch_detection": {
        "familiar_mean": float(np.mean(familiar_mm)),
        "familiar_std": float(np.std(familiar_mm)),
        "novel_mean": float(np.mean(novel_mm)),
        "novel_std": float(np.std(novel_mm)),
        "separation": float(np.mean(novel_mm) - np.mean(familiar_mm)),
        "t_stat": float(t_stat), "p_val": float(p_val),
    },
    "stage_representations": stage_results,
}

with open("consolidation_v10_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved to consolidation_v10_results.json")


# =============================================================================
# 17. VISUALIZATION
# =============================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Hippocampal v10: 1000-dim CA1+Sub, Element-wise Addition",
             fontsize=14, fontweight='bold')

# Panel 1: CA1 mismatch during encoding
ax = axes[0, 0]
ax.plot(ca1_mm, alpha=0.3, color="steelblue", linewidth=0.5)
window = 20
if len(ca1_mm) > window:
    smoothed = np.convolve(ca1_mm, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(ca1_mm)), smoothed, color="darkblue", linewidth=2)
ax.set_xlabel("Pattern #")
ax.set_ylabel("CA1 mismatch")
ax.set_title("CA1 Learning During Encoding")
ax.grid(True, alpha=0.3)

# Panel 2: Replay quality comparison (CA1 vs Sub vs Combined)
ax = axes[0, 1]
methods = ["CA1 only", "Sub only", "CA1+Sub"]
means = [np.mean(ca1_only_sims), np.mean(sub_only_sims), np.mean(replay_sims[:200])]
ax.bar(methods, means, color=["steelblue", "coral", "forestgreen"], alpha=0.8)
ax.set_ylabel("Cosine similarity to original EC")
ax.set_title("Replay Quality by Component")
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: Mismatch detection
ax = axes[0, 2]
ax.hist(familiar_mm, bins=30, alpha=0.6, label=f"Familiar ({np.mean(familiar_mm):.3f})",
        color="forestgreen")
ax.hist(novel_mm, bins=30, alpha=0.6, label=f"Novel ({np.mean(novel_mm):.3f})",
        color="coral")
ax.set_xlabel("Cosine distance (1 - cos_sim)")
ax.set_ylabel("Count")
ax.set_title(f"Mismatch Detection (p={p_val:.2e})")
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 4: Active fraction by stage
ax = axes[1, 0]
stages_ordered = ["EC", "DG", "CA3", "CA1", "Sub", "CA1+Sub"]
active_fracs = [stage_results[s]["active_fraction"] for s in stages_ordered]
x_pos = np.arange(len(stages_ordered))
ax.bar(x_pos, active_fracs, color="steelblue", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(stages_ordered, fontsize=10)
ax.set_ylabel("Active fraction")
ax.set_title("Sparsity by Stage")
ax.grid(True, alpha=0.3, axis="y")

# Panel 5: Pairwise similarity by stage
ax = axes[1, 1]
pair_sims_stage = [stage_results[s]["mean_pairwise_sim"] for s in stages_ordered]
ax.bar(x_pos, pair_sims_stage, color="coral", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(stages_ordered, fontsize=10)
ax.set_ylabel("Mean pairwise cosine sim")
ax.set_title("Pattern Discriminability by Stage")
ax.grid(True, alpha=0.3)

# Panel 6: Feature decode R^2 by stage
ax = axes[1, 2]
decode_r2s = [stage_results[s]["feature_decode_r2_mean"] for s in stages_ordered]
ax.bar(x_pos, decode_r2s, color="forestgreen", alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(stages_ordered, fontsize=10)
ax.set_ylabel("Linear decode R^2")
ax.set_title("Feature Decodability by Stage")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("consolidation_v10_results.png", dpi=150, bbox_inches='tight')
print("  Saved plot")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
