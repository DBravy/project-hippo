"""
Hippocampal Consolidation v11: EC Sparse/Dense Split, FiLM Gating at Vb
=========================================================================

The fundamental problem the hippocampus solves: store efficiently, replay
fully. Storage requires sparsity (CA3). But the full EC representation is
sparse AND dense, working together: dense modulating sparse. Only the sparse
component goes through CA3. The downstream pathway (CA1 -> Sub) must
reconstruct BOTH components from only the sparse retrieval.

Architecture:

  Cortical input (1000-dim)
        |
  EC Superficial (II/III):
    Stellate (sparse, lateral inh) <--- Pyramidal can excite stellate
    Pyramidal (dense, no inh)
        |
        +--- Stellate ---> DG(1000) ---> CA3(1000)  [storage pathway]
        |
        +--- Pyramidal --> CA1 TA (excitatory teaching signal)
        +--- Stellate ---> CA1 inhibitory interneurons (sparsity pattern)
        +--- Pyramidal --> Sub (strong teaching)
        +--- Stellate ---> Sub (weak teaching)
        |
  CA3 retrieval ---> CA1 (Schaffer, plastic)
  CA1 output ------> Sub (plastic, slow learning)
        |               |
        |          Sub output (dense, general context)
        |          CA1 output (sparse, episode-specific)
        |               |
  EC Deep Vb: FiLM gating
    gamma = sigmoid(sub_output)
    gated_hippo = gamma * ca1_output
        |
    For comparison (during encoding):
    gamma_ec = sigmoid(pyramidal)
    gated_ec = gamma_ec * stellate
        |
    Mismatch = discrepancy between gated pairs

All dimensions 1000. CA1 is sparse (~10-15%), Sub is dense (~50%),
combined FiLM output has sparse structure modulated by dense gain.
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
    Splits cortical input into two representations:
    - Stellate (sparse): lateral inhibition enforces sparsity.
      Context-specific, fast-adapting.
    - Pyramidal (dense): no lateral inhibition.
      General context, slow-adapting.

    Pyramidal excites stellate (dense informs sparse), not vice versa.
    Both receive the same cortical input through independent fixed weights.
    """
    def __init__(self, d_ec, sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
                 pyr_to_stel_strength=0.3, connectivity_prob=0.33):
        self.d_ec = d_ec
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.pyr_to_stel_strength = pyr_to_stel_strength

        # Stellate pathway: fixed weights + lateral inhibition
        self.W_stellate = make_feedforward_weights(d_ec, d_ec, connectivity_prob)
        self.W_inh = build_ring_inhibition(d_ec, sigma_inh)

        # Pyramidal pathway: fixed weights, no inhibition
        self.W_pyramidal = make_feedforward_weights(d_ec, d_ec, connectivity_prob)

        # Pyramidal -> Stellate excitation (dense informs sparse)
        self.W_pyr_to_stel = make_feedforward_weights(d_ec, d_ec, connectivity_prob)

    def forward(self, cortical_input):
        """Returns (stellate, pyramidal) representations."""
        # Dense: straightforward, no inhibition
        pyramidal = np.maximum(self.W_pyramidal @ cortical_input, 0)

        # Sparse: feedforward + pyramidal context + lateral inhibition
        h_raw = self.W_stellate @ cortical_input
        h_raw += self.pyr_to_stel_strength * (self.W_pyr_to_stel @ pyramidal)
        h_raw = np.maximum(h_raw, 0)

        h = h_raw.copy()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = np.maximum(h_raw - self.gamma_inh * inh, 0)

        return h, pyramidal  # stellate (sparse), pyramidal (dense)


# =============================================================================
# 3. DENTATE GYRUS (receives stellate/sparse signal)
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
# 5. CA1 (stellate-informed inhibition)
# =============================================================================

class CA1:
    """
    CA1 receives three streams:
    - CA3 via Schaffer collaterals (plastic, distal)
    - EC pyramidal/dense via TA (fixed, excitatory teaching signal)
    - EC stellate/sparse via inhibitory interneurons (fixed, shapes sparsity)

    BTSP: plateau gating from TA (dense) enables learning at Schaffer synapses.
    Stellate-informed inhibition: during retrieval, stellate signal drives
    feedforward inhibition that enforces a sparsity pattern consistent with
    the current sparse EC representation.
    """
    def __init__(self, N_ca1, N_ca3, d_ec,
                 lr=0.3, plateau_threshold=0.7, plateau_sharpness=20.0,
                 weight_decay=1.0, div_norm_sigma=0.0,
                 connectivity_prob=0.33,
                 ltd_rate=0.05, ltd_ca3_threshold=0.0,
                 sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
                 E_inh=-0.4, gamma_stel=1.0):
        self.N_ca1 = N_ca1
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
        self.gamma_stel = gamma_stel

        # TA pathway from EC pyramidal (dense, excitatory teaching)
        self.W_ta = make_feedforward_weights(N_ca1, d_ec, connectivity_prob)

        # Schaffer collateral from CA3 (plastic)
        mask = (np.random.rand(N_ca1, N_ca3) < connectivity_prob).astype(float)
        self.W_sc = np.random.randn(N_ca1, N_ca3) * 0.01 * mask
        self.connectivity_mask = mask.copy()

        # Lateral inhibition (ring topology)
        self.W_inh = build_ring_inhibition(N_ca1, sigma_inh)

        # Stellate -> CA1 inhibitory interneurons (fixed)
        # Stellate drives feedforward inhibition at CA1
        self.W_stel_inh = make_feedforward_weights(N_ca1, d_ec, connectivity_prob)

        self.n_episodes = 0

    def _sigmoid(self, x):
        return sigmoid(x)

    def _divisive_normalize(self, h):
        return h / (np.mean(h) + self.div_norm_sigma + 1e-10)

    def compute_activations(self, x_ca3, x_ec_pyr):
        """Compute TA and SC activations. TA uses pyramidal/dense EC."""
        h_ta_raw = np.maximum(self.W_ta @ x_ec_pyr, 0)
        h_sc_raw = np.maximum(self.W_sc @ x_ca3, 0)
        h_ta = self._divisive_normalize(h_ta_raw)
        h_sc = self._divisive_normalize(h_sc_raw)
        threshold = self.plateau_threshold * np.max(h_ta_raw) if np.max(h_ta_raw) > 1e-10 else 0.0
        gate = self._sigmoid(self.plateau_sharpness * (h_ta_raw - threshold))
        return h_ta, h_sc, gate, h_ta_raw, h_sc_raw

    def encode(self, x_ca3, x_ec_pyr):
        """BTSP encoding. Uses pyramidal EC for plateau gating."""
        h_ta, h_sc, gate, _, _ = self.compute_activations(x_ca3, x_ec_pyr)
        gated_error = gate * (h_ta - h_sc)
        mismatch = float(np.linalg.norm(gated_error) / (np.linalg.norm(h_ta) + 1e-10))

        self.W_sc += self.lr * np.outer(gated_error, x_ca3)
        ca3_inactive = (x_ca3 <= self.ltd_ca3_threshold).astype(float)
        self.W_sc *= (1.0 - self.ltd_rate * np.outer(gate, ca3_inactive))
        if self.weight_decay < 1.0:
            self.W_sc *= self.weight_decay
        self.W_sc *= self.connectivity_mask

        self.n_episodes += 1
        return mismatch

    def retrieve(self, x_ca3, x_ec_pyr, x_ec_stel):
        """
        Retrieval with stellate-informed inhibition.
        EC pyramidal drives TA (excitatory).
        EC stellate drives feedforward inhibition (shapes sparsity pattern).
        """
        h_ta, h_sc, gate, h_ta_raw, h_sc_raw = self.compute_activations(x_ca3, x_ec_pyr)
        gated_error = gate * (h_ta - h_sc)
        mismatch = float(np.linalg.norm(gated_error) / (np.linalg.norm(h_ta) + 1e-10))

        # Stellate-driven feedforward inhibition
        h_stel_drive = np.maximum(self.W_stel_inh @ x_ec_stel, 0)

        # Conductance-based lateral inhibition + stellate feedforward inhibition
        h_out = h_sc_raw.copy()
        for _ in range(self.n_inh_steps):
            g_lateral = self.gamma_inh * (self.W_inh @ h_out)
            g_stel = self.gamma_stel * h_stel_drive
            g_inh_total = g_lateral + g_stel
            h_out = np.maximum(
                (h_sc_raw + g_inh_total * self.E_inh) / (1.0 + g_inh_total), 0.0)

        return h_out, mismatch


# =============================================================================
# 6. SUBICULUM (receives CA1 + EC pyramidal strong + EC stellate weak)
# =============================================================================

class Subiculum:
    """
    Learns to produce the dense component from CA1's sparse output.
    Slow learning (conventional Hebbian LTP).

    Teaching signals:
    - EC pyramidal (strong): the dense representation to match
    - EC stellate (weak): some episode-specific context

    The combined teaching target = EC_pyr + weak * EC_stel
    """
    def __init__(self, N_sub, N_ca1, d_ec,
                 lr=0.05, ltd_rate=0.001, connectivity_prob=0.33,
                 stel_teaching_strength=0.2):
        self.N_sub = N_sub
        self.lr = lr
        self.ltd_rate = ltd_rate
        self.stel_teaching_strength = stel_teaching_strength

        # CA1 -> Sub: plastic
        mask_ca1 = (np.random.rand(N_sub, N_ca1) < connectivity_prob).astype(float)
        self.W_ca1 = np.random.randn(N_sub, N_ca1) * 0.01 * mask_ca1
        self.mask_ca1 = mask_ca1.copy()

        # EC pyramidal -> Sub: fixed, strong teaching
        self.W_ec_pyr = make_feedforward_weights(N_sub, d_ec, connectivity_prob)

        # EC stellate -> Sub: fixed, weak teaching
        self.W_ec_stel = make_feedforward_weights(N_sub, d_ec, connectivity_prob)

        self.n_episodes = 0

    def encode(self, ca1_output, ec_pyramidal, ec_stellate):
        h_ca1 = self.W_ca1 @ ca1_output
        h_ec_pyr = self.W_ec_pyr @ ec_pyramidal
        h_ec_stel = self.W_ec_stel @ ec_stellate

        # Combined teaching signal (pyramidal dominant + weak stellate)
        h_teach = h_ec_pyr + self.stel_teaching_strength * h_ec_stel

        h_sub = np.maximum(h_ca1 + h_teach, 0)

        # Error-driven learning
        error = h_teach - h_ca1
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
# 7. EC DEEP Vb (FiLM gating, no learned weights)
# =============================================================================

class ECDeepVb:
    """
    FiLM gating: dense modulates sparse.

    Hippocampal output: sigmoid(sub) * ca1
    EC ground truth:    sigmoid(pyramidal) * stellate

    No learned weights. All learning is upstream. Vb is a fixed
    combining operation that preserves the sparse structure while
    applying dense-derived gain.
    """
    @staticmethod
    def gate(sparse_signal, dense_signal):
        """FiLM: dense modulates sparse via sigmoid gating."""
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
        self.stel_store = []
        self.pyr_store = []

    def encode_batch(self, ec_patterns):
        self.ec_store = [ec.copy() for ec in ec_patterns]
        ca1_mismatches = []
        self.stel_store = []
        self.pyr_store = []

        for ec in ec_patterns:
            stellate, pyramidal = self.ec_sup.forward(ec)
            self.stel_store.append(stellate.copy())
            self.pyr_store.append(pyramidal.copy())

            # Storage: stellate (sparse) enters DG -> CA3
            dg_out = self.dg.forward(stellate)
            self.ca3.store_online(dg_out)

            # CA1 encoding: Schaffer from DG, TA from pyramidal
            self.ca1.encode(dg_out, pyramidal)

            # Get CA1 output for Sub training (full retrieve with stellate inh)
            ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
            ca1_out, mm = self.ca1.retrieve(ca3_out, pyramidal, stellate)
            ca1_mismatches.append(mm)

            # Train Sub: CA1 + EC pyramidal (strong) + EC stellate (weak)
            self.sub.encode(ca1_out, pyramidal, stellate)

        return ca1_mismatches

    def replay_to_output(self, ec_query):
        """Full replay pipeline."""
        stellate, pyramidal = self.ec_sup.forward(ec_query)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, ca1_mm = self.ca1.retrieve(ca3_out, pyramidal, stellate)
        sub_out = self.sub.replay(ca1_out)
        gated = ECDeepVb.hippocampal_output(ca1_out, sub_out)
        return gated, {
            "ca1_mismatch": ca1_mm,
            "ca1_out": ca1_out,
            "sub_out": sub_out,
            "stellate": stellate,
            "pyramidal": pyramidal,
        }

    def generate_replay_batch(self):
        return [self.replay_to_output(ec)[0] for ec in self.ec_store]

    def get_stage_outputs(self, ec_pattern):
        stellate, pyramidal = self.ec_sup.forward(ec_pattern)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, _ = self.ca1.retrieve(ca3_out, pyramidal, stellate)
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
    "weight_decay": 1.0, "div_norm_sigma": 0.0,
    "ltd_rate": 0.05, "ltd_ca3_threshold": 0.0,
    "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
    "E_inh": -0.4, "gamma_stel": 1.0,
}

sub_params = {
    "lr": 0.05, "ltd_rate": 0.001,
    "connectivity_prob": 0.33, "stel_teaching_strength": 0.2,
}

N_experience = 5000
N_probe = 5000

print("=" * 70)
print("HIPPOCAMPAL v11: EC SPARSE/DENSE SPLIT, FiLM GATING")
print("=" * 70)
print(f"\nWorld: {N_features} features -> EC({d_ec})")
print(f"EC superficial: stellate(sparse) + pyramidal(dense)")
print(f"Hippocampus: stellate -> DG({D_dg}) -> CA3({N_ca3}) -> CA1({N_ca1})")
print(f"  CA1 + pyramidal(TA) + stellate(inh) -> Sub({N_sub})")
print(f"  Vb: FiLM = sigmoid(sub) * ca1")
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
print(f"  CA1 mismatch: first={ca1_mm[0]:.4f}, last={ca1_mm[-1]:.4f}")


# =============================================================================
# 14. REPLAY QUALITY
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 3: Replay quality")
print("=" * 70)

# Component-level analysis (first 200)
ca1_only_sims, sub_only_sims, gated_sims, ec_target_sims = [], [], [], []
fracs = {"stel": [], "pyr": [], "ca1": [], "sub": [], "gated": []}

for i, ec in enumerate(ec_exp[:200]):
    s = hipp.get_stage_outputs(ec)

    # How well does each component match original EC input?
    ca1_only_sims.append(cosine_similarity(ec, s["ca1"]))
    sub_only_sims.append(cosine_similarity(ec, s["sub"]))
    gated_sims.append(cosine_similarity(ec, s["gated_hippo"]))

    # How well does gated_hippo match gated_ec? (both in Vb space)
    ec_target_sims.append(cosine_similarity(s["gated_hippo"], s["gated_ec"]))

    fracs["stel"].append(float(np.mean(s["stellate"] > 0)))
    fracs["pyr"].append(float(np.mean(s["pyramidal"] > 0)))
    fracs["ca1"].append(float(np.mean(s["ca1"] > 0)))
    fracs["sub"].append(float(np.mean(s["sub"] > 0)))
    fracs["gated"].append(float(np.mean(s["gated_hippo"] > 0)))

print(f"  Cosine sim to original EC input:")
print(f"    CA1 only:      {np.mean(ca1_only_sims):.4f}")
print(f"    Sub only:      {np.mean(sub_only_sims):.4f}")
print(f"    FiLM gated:    {np.mean(gated_sims):.4f}")
print(f"  Gated hippo vs gated EC: {np.mean(ec_target_sims):.4f}")

print(f"\n  Active fractions:")
for k, v in fracs.items():
    print(f"    {k:>8s}: {np.mean(v):.3f}")


# =============================================================================
# 15. MISMATCH DETECTION
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 4: Mismatch detection")
print("=" * 70)

# Familiar: encoded patterns
familiar_mm = []
for ec in ec_exp[:200]:
    s = hipp.get_stage_outputs(ec)
    familiar_mm.append(1.0 - cosine_similarity(s["gated_hippo"], s["gated_ec"]))

# Novel: never-seen patterns
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
        "ca1_only_mean": float(np.mean(ca1_only_sims)),
        "sub_only_mean": float(np.mean(sub_only_sims)),
        "gated_mean": float(np.mean(gated_sims)),
        "gated_hippo_vs_gated_ec": float(np.mean(ec_target_sims)),
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

with open("consolidation_v11_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved to consolidation_v11_results.json")


# =============================================================================
# 18. VISUALIZATION
# =============================================================================

print("\nGenerating plots...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Hippocampal v11: EC Sparse/Dense Split, FiLM Gating",
             fontsize=14, fontweight='bold')

# Panel 1: CA1 mismatch during encoding
ax = axes[0, 0]
ax.plot(ca1_mm, alpha=0.3, color="steelblue", linewidth=0.5)
w = 20
if len(ca1_mm) > w:
    sm = np.convolve(ca1_mm, np.ones(w)/w, mode='valid')
    ax.plot(range(w-1, len(ca1_mm)), sm, color="darkblue", linewidth=2)
ax.set_xlabel("Pattern #"); ax.set_ylabel("CA1 mismatch")
ax.set_title("CA1 Learning"); ax.grid(True, alpha=0.3)

# Panel 2: Replay quality by component
ax = axes[0, 1]
methods = ["CA1\nonly", "Sub\nonly", "FiLM\ngated", "Gated hippo\nvs gated EC"]
vals = [np.mean(ca1_only_sims), np.mean(sub_only_sims),
        np.mean(gated_sims), np.mean(ec_target_sims)]
ax.bar(methods, vals, color=["steelblue","coral","forestgreen","goldenrod"], alpha=0.8)
ax.set_ylabel("Cosine similarity"); ax.set_title("Replay Quality")
ax.grid(True, alpha=0.3, axis="y")

# Panel 3: Mismatch detection
ax = axes[0, 2]
ax.hist(familiar_mm, bins=30, alpha=0.6, label=f"Familiar ({np.mean(familiar_mm):.3f})",
        color="forestgreen")
ax.hist(novel_mm, bins=30, alpha=0.6, label=f"Novel ({np.mean(novel_mm):.3f})",
        color="coral")
ax.set_xlabel("Cosine distance"); ax.set_title(f"Mismatch (p={p_val:.2e})")
ax.legend(); ax.grid(True, alpha=0.3)

# Panel 4: Active fraction by stage
ax = axes[1, 0]
stage_order = ["EC","Stellate","Pyramidal","DG","CA3","CA1","Sub",
               "Gated_hippo","Gated_EC"]
af = [stage_results[s]["active_fraction"] for s in stage_order]
x = np.arange(len(stage_order))
ax.bar(x, af, color="steelblue", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(stage_order, fontsize=7, rotation=30)
ax.set_ylabel("Active fraction"); ax.set_title("Sparsity by Stage")
ax.grid(True, alpha=0.3, axis="y")

# Panel 5: Pairwise similarity
ax = axes[1, 1]
ps = [stage_results[s]["mean_pairwise_sim"] for s in stage_order]
ax.bar(x, ps, color="coral", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(stage_order, fontsize=7, rotation=30)
ax.set_ylabel("Pairwise cosine sim"); ax.set_title("Discriminability by Stage")
ax.grid(True, alpha=0.3, axis="y")

# Panel 6: Feature decode R^2
ax = axes[1, 2]
dr = [stage_results[s]["feature_decode_r2_mean"] for s in stage_order]
ax.bar(x, dr, color="forestgreen", alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(stage_order, fontsize=7, rotation=30)
ax.set_ylabel("Decode R^2"); ax.set_title("Feature Decodability by Stage")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("consolidation_v11_results.png", dpi=150, bbox_inches='tight')
print("  Saved plot")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
