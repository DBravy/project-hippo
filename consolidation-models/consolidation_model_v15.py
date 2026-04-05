"""
Hippocampal Consolidation v15: CA1 Plateau Gating + Depression-Dominated
=========================================================================

CA1 adapted for sparse stellate TA:
  - Plateau gating: only CA1 neurons with strong stellate TA drive
    undergo plasticity. High threshold selects ~5% of neurons per
    episode, preventing diffuse weight accumulation.
  - Depression-dominated: mild weight decay every episode pulls all
    Schaffer weights toward zero. Only recently or repeatedly
    reinforced associations survive. This mirrors stellate neurons'
    bias toward forgetting, keeping only what is consistent or current.
  - Combined effect: selective writing (plateau) + continuous erasing
    (decay) produces sparse, recent, discriminative CA1 output.

Sub learns dense (pyramidal) from CA1 sparse. FiLM at Vb.
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


def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


# =============================================================================
# 2. EC SUPERFICIAL (II/III): Sparse + Dense Split
# =============================================================================

class ECSuperficial:
    """
    Splits cortical input into:
    - Stellate (sparse): lateral inhibition, ~4% active, episode-specific
    - Pyramidal (dense): no inhibition, ~50% active, general context

    Pyramidal excites stellate (dense informs sparse), not vice versa.
    """
    def __init__(self, d_ec, sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
                 pyr_to_stel_strength=0.3, connectivity_prob=0.33):
        self.d_ec = d_ec
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.pyr_to_stel_strength = pyr_to_stel_strength

        self.W_stellate = make_feedforward_weights(d_ec, d_ec, connectivity_prob)
        self.W_inh = build_ring_inhibition(d_ec, sigma_inh)
        self.W_pyramidal = make_feedforward_weights(d_ec, d_ec, connectivity_prob)
        self.W_pyr_to_stel = make_feedforward_weights(d_ec, d_ec, connectivity_prob)

    def forward(self, cortical_input):
        pyramidal = np.maximum(self.W_pyramidal @ cortical_input, 0)

        h_raw = self.W_stellate @ cortical_input
        h_raw += self.pyr_to_stel_strength * (self.W_pyr_to_stel @ pyramidal)
        h_raw = np.maximum(h_raw, 0)

        h = h_raw.copy()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = np.maximum(h_raw - self.gamma_inh * inh, 0)

        return h, pyramidal  # stellate (sparse), pyramidal (dense)


# =============================================================================
# 3. DENTATE GYRUS
# =============================================================================

class DentateGyrusLateral:
    def __init__(self, d_input, D_output, sigma_inh=50, gamma_inh=1.0,
                 n_inh_steps=5, noise_scale=0.0, inh_connection_prob=None):
        self.D_output = D_output
        self.W_ff = make_feedforward_weights(D_output, d_input)
        self.W_inh = build_ring_inhibition(
            D_output, sigma_inh, connection_prob=inh_connection_prob)
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.noise_scale = noise_scale

    def forward(self, x):
        h_raw = x @ self.W_ff.T
        h_raw = np.maximum(h_raw, 0)
        if self.noise_scale > 0 and np.any(h_raw > 0):
            mean_active = np.mean(h_raw[h_raw > 0])
            h_raw = np.maximum(
                h_raw + np.random.randn(self.D_output) * self.noise_scale * mean_active, 0)
        h = h_raw.copy()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = np.maximum(h_raw - self.gamma_inh * inh, 0)
        return h


# =============================================================================
# 4. CA3
# =============================================================================

class CA3:
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.mean_activity = np.zeros(N)

    def store_online(self, pattern):
        p = pattern / (np.linalg.norm(pattern) + 1e-10)
        self.n_stored += 1
        self.mean_activity += (p - self.mean_activity) / self.n_stored
        p_c = p - self.mean_activity
        self.W += np.outer(p_c, p_c)
        np.fill_diagonal(self.W, 0)

    def retrieve(self, cue, n_iterations=5):
        x = cue / (np.linalg.norm(cue) + 1e-10)
        for _ in range(n_iterations):
            h = np.maximum(self.W @ x, 0)
            x_new = apply_kwta(h, self.k_active)
            x_new = x_new / (np.linalg.norm(x_new) + 1e-10)
            if np.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x


# =============================================================================
# 5. CA1 (TA from stellate/sparse signal)
# =============================================================================

class CA1:
    """
    CA1 learns to reproduce stellate from CA3, with two control mechanisms:

    1. Plateau gating: only neurons with strong stellate TA drive learn.
       Since stellate is ~4% active and W_ta has 33% connectivity, most
       CA1 neurons get weak TA input. The plateau threshold selects only
       the ~5% most strongly driven, preventing diffuse weight growth.

    2. Depression-dominated dynamics: mild weight decay every episode.
       All Schaffer weights continuously drift toward zero. Only patterns
       that are recent or repeatedly reinforced survive the decay.
       This mirrors stellate neurons' forgetting bias: the system
       retains what is consistent and sheds what is transient.

    Combined: selective writing + continuous erasing = sparse, recent,
    discriminative output from ReLU(W_sc @ ca3).
    """
    def __init__(self, N_ca1, N_ca3, d_ec,
                 lr=0.3, plateau_threshold=0.7, plateau_sharpness=20.0,
                 weight_decay=0.998, connectivity_prob=0.33,
                 ltd_rate=0.05, ltd_ca3_threshold=0.0):
        self.N_ca1 = N_ca1
        self.lr = lr
        self.plateau_threshold = plateau_threshold
        self.plateau_sharpness = plateau_sharpness
        self.weight_decay = weight_decay
        self.ltd_rate = ltd_rate
        self.ltd_ca3_threshold = ltd_ca3_threshold

        # TA from stellate (fixed)
        self.W_ta = make_feedforward_weights(N_ca1, d_ec, connectivity_prob)

        # Schaffer from CA3 (plastic, zero-initialized = silent synapses)
        mask = (np.random.rand(N_ca1, N_ca3) < connectivity_prob).astype(float)
        self.W_sc = np.zeros((N_ca1, N_ca3))
        self.connectivity_mask = mask.copy()

        self.n_episodes = 0

    def encode(self, x_ca3, x_ec_stel):
        """
        Plateau-gated error-driven learning with weight decay.
        """
        h_ta = np.maximum(self.W_ta @ x_ec_stel, 0)
        h_sc = np.maximum(self.W_sc @ x_ca3, 0)

        # Plateau gating: high threshold selects strongly-driven neurons
        if np.max(h_ta) > 1e-10:
            threshold = self.plateau_threshold * np.max(h_ta)
        else:
            threshold = 0.0
        gate = sigmoid(self.plateau_sharpness * (h_ta - threshold))

        # Gated error: only plateau neurons contribute
        error = h_ta - h_sc
        gated_error = gate * error
        mismatch = float(np.linalg.norm(gated_error) / (np.linalg.norm(h_ta) + 1e-10))

        # Weight update at gated neurons
        self.W_sc += self.lr * np.outer(gated_error, x_ca3)

        # Depression-dominated: continuous weight decay (forgetting)
        self.W_sc *= self.weight_decay

        # Heterosynaptic LTD: depress inactive CA3 synapses at gated neurons
        ca3_inactive = (x_ca3 <= self.ltd_ca3_threshold).astype(float)
        self.W_sc *= (1.0 - self.ltd_rate * np.outer(gate, ca3_inactive))

        # Enforce connectivity
        self.W_sc *= self.connectivity_mask

        self.n_episodes += 1
        return mismatch

    def retrieve(self, x_ca3, x_ec_stel=None):
        """
        Output = ReLU(Schaffer @ CA3).
        Sparsity from plateau-selective learning + depression decay.
        """
        h_sc = np.maximum(self.W_sc @ x_ca3, 0)

        # Mismatch diagnostic if stellate available
        mismatch = 0.0
        if x_ec_stel is not None:
            h_ta = np.maximum(self.W_ta @ x_ec_stel, 0)
            error = h_ta - h_sc
            mismatch = float(np.linalg.norm(error) / (np.linalg.norm(h_ta) + 1e-10))

        return h_sc, mismatch


# =============================================================================
# 6. SUBICULUM (learns dense component from CA1 sparse)
# =============================================================================

class Subiculum:
    """
    Learns to produce the dense (pyramidal) component from CA1's sparse output.

    Teaching signals:
    - EC pyramidal (strong): the dense representation to reconstruct
    - EC stellate (weak): episode-specific context

    Slow learning (conventional Hebbian LTP). Minimal LTD.
    """
    def __init__(self, N_sub, N_ca1, d_ec,
                 lr=0.05, ltd_rate=0.001, connectivity_prob=0.33,
                 stel_teaching_strength=0.2):
        self.N_sub = N_sub
        self.lr = lr
        self.ltd_rate = ltd_rate
        self.stel_teaching_strength = stel_teaching_strength

        mask_ca1 = (np.random.rand(N_sub, N_ca1) < connectivity_prob).astype(float)
        self.W_ca1 = np.random.randn(N_sub, N_ca1) * 0.01 * mask_ca1
        self.mask_ca1 = mask_ca1.copy()

        self.W_ec_pyr = make_feedforward_weights(N_sub, d_ec, connectivity_prob)
        self.W_ec_stel = make_feedforward_weights(N_sub, d_ec, connectivity_prob)
        self.n_episodes = 0

    def encode(self, ca1_output, ec_pyramidal, ec_stellate):
        h_ca1 = self.W_ca1 @ ca1_output
        h_ec_pyr = self.W_ec_pyr @ ec_pyramidal
        h_ec_stel = self.W_ec_stel @ ec_stellate
        h_teach = h_ec_pyr + self.stel_teaching_strength * h_ec_stel

        h_sub = np.maximum(h_ca1 + h_teach, 0)

        error = h_teach - h_ca1
        self.W_ca1 += self.lr * np.outer(error, ca1_output)

        if self.ltd_rate > 0:
            ca1_inactive = (ca1_output <= 0).astype(float)
            sub_active = (h_sub > 0).astype(float)
            self.W_ca1 *= (1.0 - self.ltd_rate * np.outer(sub_active, ca1_inactive))
        self.W_ca1 *= self.mask_ca1

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
# 7. EC DEEP Vb (FiLM gating)
# =============================================================================

class ECDeepVb:
    """FiLM: dense modulates sparse. No learned weights."""
    @staticmethod
    def gate(sparse_signal, dense_signal):
        gamma = sigmoid(dense_signal)
        return gamma * sparse_signal

    @staticmethod
    def hippocampal_output(ca1_output, sub_output):
        return ECDeepVb.gate(ca1_output, sub_output)

    @staticmethod
    def ec_target(stellate, pyramidal):
        return ECDeepVb.gate(stellate, pyramidal)


# =============================================================================
# 8. HIPPOCAMPAL SYSTEM
# =============================================================================

class HippocampalSystem:
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, N_sub=1000, ca3_retrieval_iterations=5):
        self.d_ec = d_ec
        self.ca3_retrieval_iterations = ca3_retrieval_iterations

        self.ec_sup = ECSuperficial(d_ec, **(ec_sup_params or {}))
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}))
        self.ca3 = CA3(N_ca3, k_ca3)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}))
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}))

        self.ec_store = []

    def encode_batch(self, ec_patterns):
        self.ec_store = [ec.copy() for ec in ec_patterns]
        ca1_mismatches = []

        for ec in ec_patterns:
            stellate, pyramidal = self.ec_sup.forward(ec)

            # Storage: stellate -> DG -> CA3
            dg_out = self.dg.forward(stellate)
            self.ca3.store_online(dg_out)

            # CA1 encoding: Schaffer from DG, TA from stellate
            self.ca1.encode(dg_out, stellate)

            # CA1 output for Sub training
            ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
            ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
            ca1_mismatches.append(mm)

            # Sub: learns dense (pyramidal) from CA1 sparse
            self.sub.encode(ca1_out, pyramidal, stellate)

        return ca1_mismatches

    def replay_to_output(self, ec_query):
        stellate, pyramidal = self.ec_sup.forward(ec_query)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, ca1_mm = self.ca1.retrieve(ca3_out, stellate)
        sub_out = self.sub.replay(ca1_out)
        gated = ECDeepVb.hippocampal_output(ca1_out, sub_out)
        return gated, {
            "ca1_mismatch": ca1_mm,
            "ca1_out": ca1_out, "sub_out": sub_out,
            "stellate": stellate, "pyramidal": pyramidal,
        }

    def generate_replay_batch(self):
        return [self.replay_to_output(ec)[0] for ec in self.ec_store]

    def get_stage_outputs(self, ec_pattern):
        stellate, pyramidal = self.ec_sup.forward(ec_pattern)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, _ = self.ca1.retrieve(ca3_out, stellate)
        sub_out = self.sub.replay(ca1_out)
        gated_hippo = ECDeepVb.hippocampal_output(ca1_out, sub_out)
        gated_ec = ECDeepVb.ec_target(stellate, pyramidal)
        return {
            "ec_input": ec_pattern,
            "stellate": stellate, "pyramidal": pyramidal,
            "dg": dg_out, "ca3": ca3_out,
            "ca1": ca1_out, "sub": sub_out,
            "gated_hippo": gated_hippo, "gated_ec": gated_ec,
        }


# =============================================================================
# 9. WORLD MODEL
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
# 10. ANALYSIS TOOLS
# =============================================================================

def measure_stage(reprs, dim, features_probe=None, world=None, n_feat=100):
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

    decode_r2 = 0.0
    if features_probe is not None and world is not None:
        lam = 0.01
        HTH_inv = np.linalg.inv(reprs.T @ reprs + lam * np.eye(reprs.shape[1]))
        r2s = []
        for fi in range(min(n_feat, world.N_features)):
            y = features_probe[:len(reprs), fi]
            w = HTH_inv @ (reprs.T @ y)
            ss_res = np.sum((y - reprs @ w) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
            r2s.append(max(0.0, 1.0 - ss_res / ss_tot))
        decode_r2 = float(np.mean(r2s))

    return {
        "n_dims": dim, "active_fraction": active_frac,
        "participation_ratio": float(pr),
        "pr_normalized": float(pr / dim) if dim > 0 else 0.0,
        "mean_pairwise_sim": float(np.mean(pair_sims)) if pair_sims else 0.0,
        "feature_decode_r2_mean": decode_r2,
    }


# =============================================================================
# 11. PARAMETERS
# =============================================================================

d_ec = 1000
N_features = 200
sparsity_base = 0.1
sparsity_decay = 0.99

D_dg = 1000
N_ca3 = 1000
N_ca1 = 1000
N_sub = 1000
k_ca3 = 50
ca3_retrieval_iters = 5

ec_sup_params = {
    "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
    "pyr_to_stel_strength": 0.3,
}

dg_params = {
    "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5, "noise_scale": 0.0,
}

ca1_params = {
    "lr": 0.3, "plateau_threshold": 0.7, "plateau_sharpness": 20.0,
    "weight_decay": 0.998, "ltd_rate": 0.05, "ltd_ca3_threshold": 0.0,
}

sub_params = {
    "lr": 0.05, "ltd_rate": 0.001,
    "connectivity_prob": 0.33, "stel_teaching_strength": 0.2,
}

N_experience = 500
N_probe = 500

print("=" * 70)
print("HIPPOCAMPAL v15: PLATEAU + DEPRESSION CA1, FiLM GATING")
print("=" * 70)
print(f"\nWorld: {N_features} features -> EC({d_ec})")
print(f"EC: stellate(sparse ~4%) + pyramidal(dense ~50%)")
print(f"Storage: stellate -> DG({D_dg}) -> CA3({N_ca3})")
print(f"CA1(1000): plateau gating + weight decay.")
print(f"Sub({N_sub}): learns dense from CA1 sparse.")
print(f"Vb: FiLM = sigmoid(sub) * ca1")
print(f"Data: {N_experience} experiences, {N_probe} probes")


# =============================================================================
# 12. GENERATE WORLD
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 1: Generating world")
print("=" * 70)

world = SparseFeatureWorld(N_features, d_ec, sparsity_base, sparsity_decay)
features_exp, ec_exp = world.generate_batch(N_experience)
features_probe, ec_probe = world.generate_batch(N_probe)
print(f"  {N_experience} exp, {N_probe} probes")
print(f"  Mean features active: {np.mean(np.sum(features_exp > 0, axis=1)):.1f}")


# =============================================================================
# 13. ENCODING
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 2: Encoding")
print("=" * 70)

hipp = HippocampalSystem(
    d_ec, D_dg, N_ca3, N_ca1, k_ca3,
    dg_params=dg_params, ca1_params=ca1_params, sub_params=sub_params,
    ec_sup_params=ec_sup_params, N_sub=N_sub,
    ca3_retrieval_iterations=ca3_retrieval_iters)

ca1_mm = hipp.encode_batch(ec_exp)
print(f"  Encoded {N_experience} patterns")
print(f"  CA1 mismatch: first={ca1_mm[0]:.4f}, last={ca1_mm[-1]:.4f}, "
      f"mean={np.mean(ca1_mm):.4f}")


# =============================================================================
# 13b. DETAILED CA1 DIAGNOSTICS
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 2b: Detailed CA1 diagnostics")
print("=" * 70)

# --- W_sc weight statistics ---
W = hipp.ca1.W_sc
W_nonzero = W[W != 0]
print(f"\n  W_sc weight statistics:")
print(f"    Shape: {W.shape}")
print(f"    Fraction nonzero: {np.mean(W != 0):.4f}")
print(f"    Fraction positive: {np.mean(W > 0):.4f}")
print(f"    Fraction negative: {np.mean(W < 0):.4f}")
if len(W_nonzero) > 0:
    print(f"    Nonzero mean: {np.mean(W_nonzero):.6f}")
    print(f"    Nonzero std: {np.std(W_nonzero):.6f}")
    print(f"    Nonzero abs mean: {np.mean(np.abs(W_nonzero)):.6f}")
    print(f"    Max: {np.max(W):.6f}, Min: {np.min(W):.6f}")
    print(f"    Percentiles (abs): 50th={np.percentile(np.abs(W_nonzero), 50):.6f}, "
          f"90th={np.percentile(np.abs(W_nonzero), 90):.6f}, "
          f"99th={np.percentile(np.abs(W_nonzero), 99):.6f}")

# Row norms (each row = one CA1 neuron's incoming weights)
row_norms = np.linalg.norm(W, axis=1)
print(f"\n  W_sc row norms (per CA1 neuron):")
print(f"    Mean: {np.mean(row_norms):.6f}")
print(f"    Std: {np.std(row_norms):.6f}")
print(f"    Max: {np.max(row_norms):.6f}")
print(f"    Fraction with norm > 0.01: {np.mean(row_norms > 0.01):.4f}")
print(f"    Fraction with norm > 0.1: {np.mean(row_norms > 0.1):.4f}")
print(f"    Fraction with norm > 1.0: {np.mean(row_norms > 1.0):.4f}")

# --- Per-pattern CA1 internals (sample 10 patterns) ---
print(f"\n  Per-pattern CA1 internals (10 encoded patterns):")
sample_idx = np.linspace(0, N_experience-1, 10, dtype=int)

for idx in sample_idx:
    ec = ec_exp[idx]
    stellate, pyramidal = hipp.ec_sup.forward(ec)
    dg_out = hipp.dg.forward(stellate)
    ca3_out = hipp.ca3.retrieve(dg_out, ca3_retrieval_iters)

    # TA activation
    h_ta = np.maximum(hipp.ca1.W_ta @ stellate, 0)
    h_ta_active = np.mean(h_ta > 0)
    h_ta_max = np.max(h_ta) if np.any(h_ta > 0) else 0

    # Gate
    if h_ta_max > 1e-10:
        threshold = hipp.ca1.plateau_threshold * h_ta_max
    else:
        threshold = 0.0
    gate = sigmoid(hipp.ca1.plateau_sharpness * (h_ta - threshold))
    gate_active = np.mean(gate > 0.5)
    gate_strong = np.mean(gate > 0.9)

    # Schaffer activation
    h_sc = np.maximum(hipp.ca1.W_sc @ ca3_out, 0)
    h_sc_active = np.mean(h_sc > 0)
    h_sc_max = np.max(h_sc) if np.any(h_sc > 0) else 0

    # CA3 input stats
    ca3_active = np.mean(ca3_out > 0)

    print(f"    Pattern {idx:3d}: "
          f"stel_active={np.mean(stellate>0):.3f}, "
          f"ca3_active={ca3_active:.3f}, "
          f"h_ta_active={h_ta_active:.3f}, "
          f"h_ta_max={h_ta_max:.4f}, "
          f"gate>0.5={gate_active:.3f}, "
          f"gate>0.9={gate_strong:.3f}, "
          f"h_sc_active={h_sc_active:.3f}, "
          f"h_sc_max={h_sc_max:.4f}")

# --- Test: what does CA1 output look like for first few patterns? ---
print(f"\n  CA1 retrieve output stats (first 20 encoded patterns):")
for idx in range(20):
    ec = ec_exp[idx]
    stellate, pyramidal = hipp.ec_sup.forward(ec)
    dg_out = hipp.dg.forward(stellate)
    ca3_out = hipp.ca3.retrieve(dg_out, ca3_retrieval_iters)
    ca1_out, mm = hipp.ca1.retrieve(ca3_out, stellate)

    ca1_active = np.mean(ca1_out > 0)
    ca1_norm = np.linalg.norm(ca1_out)
    h_ta = np.maximum(hipp.ca1.W_ta @ stellate, 0)
    cos_to_ta = cosine_similarity(ca1_out, h_ta)

    print(f"    Pattern {idx:3d}: "
          f"ca1_active={ca1_active:.3f}, "
          f"ca1_norm={ca1_norm:.4f}, "
          f"cos(ca1, h_ta)={cos_to_ta:.4f}, "
          f"mismatch={mm:.4f}")

# --- Pairwise similarity of CA1 outputs (first 50) ---
print(f"\n  CA1 output pairwise similarities (first 50 patterns):")
ca1_outputs_sample = []
for idx in range(min(50, N_experience)):
    ec = ec_exp[idx]
    stellate, _ = hipp.ec_sup.forward(ec)
    dg_out = hipp.dg.forward(stellate)
    ca3_out = hipp.ca3.retrieve(dg_out, ca3_retrieval_iters)
    ca1_out, _ = hipp.ca1.retrieve(ca3_out, stellate)
    ca1_outputs_sample.append(ca1_out)

ca1_pair_sims = []
for i in range(len(ca1_outputs_sample)):
    for j in range(i+1, min(i+10, len(ca1_outputs_sample))):
        ca1_pair_sims.append(cosine_similarity(
            ca1_outputs_sample[i], ca1_outputs_sample[j]))
print(f"    Mean pairwise sim: {np.mean(ca1_pair_sims):.4f}")
print(f"    Std: {np.std(ca1_pair_sims):.4f}")
print(f"    Min: {np.min(ca1_pair_sims):.4f}, Max: {np.max(ca1_pair_sims):.4f}")

# --- Check if the issue is CA3 input being too similar ---
print(f"\n  CA3 retrieval pairwise similarities (first 50 patterns):")
ca3_outputs_sample = []
for idx in range(min(50, N_experience)):
    ec = ec_exp[idx]
    stellate, _ = hipp.ec_sup.forward(ec)
    dg_out = hipp.dg.forward(stellate)
    ca3_out = hipp.ca3.retrieve(dg_out, ca3_retrieval_iters)
    ca3_outputs_sample.append(ca3_out)

ca3_pair_sims = []
for i in range(len(ca3_outputs_sample)):
    for j in range(i+1, min(i+10, len(ca3_outputs_sample))):
        ca3_pair_sims.append(cosine_similarity(
            ca3_outputs_sample[i], ca3_outputs_sample[j]))
print(f"    Mean pairwise sim: {np.mean(ca3_pair_sims):.4f}")
print(f"    Std: {np.std(ca3_pair_sims):.4f}")

# --- Check stellate -> h_ta pathway (is TA too dense?) ---
print(f"\n  Stellate -> h_ta density analysis (20 patterns):")
ta_active_fracs = []
for idx in range(min(20, N_experience)):
    stellate, _ = hipp.ec_sup.forward(ec_exp[idx])
    h_ta = np.maximum(hipp.ca1.W_ta @ stellate, 0)
    ta_active_fracs.append(np.mean(h_ta > 0))
print(f"    h_ta active fraction: mean={np.mean(ta_active_fracs):.3f}, "
      f"std={np.std(ta_active_fracs):.3f}")
print(f"    (stellate is {np.mean([np.mean(hipp.ec_sup.forward(ec_exp[i])[0] > 0) for i in range(20)]):.3f} active)")
print(f"    W_ta connectivity: {np.mean(np.abs(hipp.ca1.W_ta) > 1e-10):.3f}")


# =============================================================================
# 14. REPLAY QUALITY
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 3: Replay quality")
print("=" * 70)

ca1_sims, sub_sims, gated_sims, ec_target_sims = [], [], [], []
fracs = {"stel": [], "pyr": [], "ca1": [], "sub": [], "gated": []}

for i, ec in enumerate(ec_exp[:200]):
    s = hipp.get_stage_outputs(ec)

    ca1_sims.append(cosine_similarity(s["stellate"], s["ca1"]))
    sub_sims.append(cosine_similarity(s["pyramidal"], s["sub"]))
    gated_sims.append(cosine_similarity(s["gated_ec"], s["gated_hippo"]))
    ec_target_sims.append(cosine_similarity(ec, s["gated_hippo"]))

    fracs["stel"].append(float(np.mean(s["stellate"] > 0)))
    fracs["pyr"].append(float(np.mean(s["pyramidal"] > 0)))
    fracs["ca1"].append(float(np.mean(s["ca1"] > 0)))
    fracs["sub"].append(float(np.mean(s["sub"] > 0)))
    fracs["gated"].append(float(np.mean(s["gated_hippo"] > 0)))

print(f"  Component reconstruction quality:")
print(f"    CA1 vs Stellate (sparse target):   {np.mean(ca1_sims):.4f}")
print(f"    Sub vs Pyramidal (dense target):    {np.mean(sub_sims):.4f}")
print(f"    Gated hippo vs Gated EC:            {np.mean(gated_sims):.4f}")
print(f"    Gated hippo vs original EC input:   {np.mean(ec_target_sims):.4f}")

print(f"\n  Active fractions:")
for k, v in fracs.items():
    print(f"    {k:>8s}: {np.mean(v):.3f}")


# =============================================================================
# 15. MISMATCH DETECTION
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 4: Mismatch detection")
print("=" * 70)

familiar_mm = []
for ec in ec_exp[:200]:
    s = hipp.get_stage_outputs(ec)
    familiar_mm.append(1.0 - cosine_similarity(s["gated_hippo"], s["gated_ec"]))

novel_mm = []
for ec in ec_probe[:200]:
    s = hipp.get_stage_outputs(ec)
    novel_mm.append(1.0 - cosine_similarity(s["gated_hippo"], s["gated_ec"]))

print(f"  Familiar cosine dist: {np.mean(familiar_mm):.4f} +/- {np.std(familiar_mm):.4f}")
print(f"  Novel cosine dist:    {np.mean(novel_mm):.4f} +/- {np.std(novel_mm):.4f}")
print(f"  Separation: {np.mean(novel_mm) - np.mean(familiar_mm):.4f}")

t_stat, p_val = stats.ttest_ind(novel_mm, familiar_mm)
print(f"  t-test: t={t_stat:.3f}, p={p_val:.2e}")


# =============================================================================
# 16. STAGE REPRESENTATION ANALYSIS
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 5: Representation quality by stage")
print("=" * 70)

n_a = min(300, N_probe)
all_reprs = {k: [] for k in
    ["EC", "Stellate", "Pyramidal", "DG", "CA3", "CA1", "Sub",
     "Gated_hippo", "Gated_EC"]}

for ec in ec_probe[:n_a]:
    s = hipp.get_stage_outputs(ec)
    all_reprs["EC"].append(s["ec_input"])
    all_reprs["Stellate"].append(s["stellate"])
    all_reprs["Pyramidal"].append(s["pyramidal"])
    all_reprs["DG"].append(s["dg"])
    all_reprs["CA3"].append(s["ca3"])
    all_reprs["CA1"].append(s["ca1"])
    all_reprs["Sub"].append(s["sub"])
    all_reprs["Gated_hippo"].append(s["gated_hippo"])
    all_reprs["Gated_EC"].append(s["gated_ec"])

stage_results = {}
for name in all_reprs:
    reprs = np.array(all_reprs[name])
    result = measure_stage(reprs, d_ec, features_probe, world)
    stage_results[name] = result
    print(f"  {name:>12s}: active={result['active_fraction']:.3f}, "
          f"PR={result['participation_ratio']:.1f}, "
          f"pair_sim={result['mean_pairwise_sim']:.4f}, "
          f"decode_R2={result['feature_decode_r2_mean']:.4f}")


# =============================================================================
# 17. SAVE RESULTS
# =============================================================================

print(f"\n{'=' * 70}")
print("Saving results...")
print("=" * 70)

results = {
    "params": {
        "d_ec": d_ec, "N_features": N_features,
        "D_dg": D_dg, "N_ca3": N_ca3, "N_ca1": N_ca1, "N_sub": N_sub,
        "k_ca3": k_ca3, "N_experience": N_experience, "N_probe": N_probe,
    },
    "replay_quality": {
        "ca1_vs_stellate": float(np.mean(ca1_sims)),
        "sub_vs_pyramidal": float(np.mean(sub_sims)),
        "gated_hippo_vs_gated_ec": float(np.mean(gated_sims)),
        "gated_hippo_vs_ec_input": float(np.mean(ec_target_sims)),
    },
    "active_fractions": {k: float(np.mean(v)) for k, v in fracs.items()},
    "mismatch_detection": {
        "familiar_mean": float(np.mean(familiar_mm)),
        "novel_mean": float(np.mean(novel_mm)),
        "separation": float(np.mean(novel_mm) - np.mean(familiar_mm)),
        "p_val": float(p_val),
    },
    "stage_representations": stage_results,
}

with open("consolidation_v15d_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved to consolidation_v15d_results.json")


# =============================================================================
# 18. VISUALIZATION
# =============================================================================

print("\nGenerating plots...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Hippocampal v15: Plateau + Depression CA1, FiLM Gating",
             fontsize=14, fontweight='bold')

# Panel 1: CA1 mismatch
ax = axes[0, 0]
ax.plot(ca1_mm, alpha=0.3, color="steelblue", linewidth=0.5)
w = 20
if len(ca1_mm) > w:
    sm = np.convolve(ca1_mm, np.ones(w)/w, mode='valid')
    ax.plot(range(w-1, len(ca1_mm)), sm, color="darkblue", linewidth=2)
ax.set_xlabel("Pattern #"); ax.set_ylabel("CA1 mismatch")
ax.set_title("CA1 Learning (plateau+decay)"); ax.grid(True, alpha=0.3)

# Panel 2: Component reconstruction
ax = axes[0, 1]
methods = ["CA1 vs\nStellate", "Sub vs\nPyramidal", "Gated hippo\nvs Gated EC"]
vals = [np.mean(ca1_sims), np.mean(sub_sims), np.mean(gated_sims)]
colors = ["steelblue", "coral", "forestgreen"]
ax.bar(methods, vals, color=colors, alpha=0.8)
ax.set_ylabel("Cosine similarity"); ax.set_title("Reconstruction Quality")
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: Mismatch detection
ax = axes[0, 2]
ax.hist(familiar_mm, bins=30, alpha=0.6, label=f"Familiar ({np.mean(familiar_mm):.3f})",
        color="forestgreen")
ax.hist(novel_mm, bins=30, alpha=0.6, label=f"Novel ({np.mean(novel_mm):.3f})",
        color="coral")
ax.set_xlabel("Cosine distance"); ax.set_title(f"Mismatch (p={p_val:.2e})")
ax.legend(); ax.grid(True, alpha=0.3)

# Panel 4: Active fraction
ax = axes[1, 0]
stage_order = ["EC", "Stellate", "Pyramidal", "DG", "CA3", "CA1", "Sub",
               "Gated_hippo", "Gated_EC"]
af = [stage_results[s]["active_fraction"] for s in stage_order]
x = np.arange(len(stage_order))
ax.bar(x, af, color="steelblue", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(stage_order, fontsize=7, rotation=30)
ax.set_ylabel("Active fraction"); ax.set_title("Sparsity"); ax.grid(True, alpha=0.3, axis="y")

# Panel 5: Pairwise similarity
ax = axes[1, 1]
ps = [stage_results[s]["mean_pairwise_sim"] for s in stage_order]
ax.bar(x, ps, color="coral", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(stage_order, fontsize=7, rotation=30)
ax.set_ylabel("Pairwise sim"); ax.set_title("Discriminability"); ax.grid(True, alpha=0.3)

# Panel 6: Feature decode R^2
ax = axes[1, 2]
dr = [stage_results[s]["feature_decode_r2_mean"] for s in stage_order]
ax.bar(x, dr, color="forestgreen", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(stage_order, fontsize=7, rotation=30)
ax.set_ylabel("Decode R^2"); ax.set_title("Feature Decodability"); ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("consolidation_v15d_results.png", dpi=150, bbox_inches='tight')
print("  Saved plot")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)