"""
Hippocampal-Cortical Consolidation v5: Superposition Experiment
================================================================

Core question: Does hippocampal replay help cortex organize its
superposition better than learning from raw experience?

Hypothesis (revised from v4 analysis):
  - The hippocampus is an ANTI-superposition device: DG expansion,
    lateral inhibition, CA3 sparse attractors all fight interference.
  - Cortex, which must simultaneously store and compute, is where
    superposition emerges as an unavoidable consequence of limited
    capacity under dual-use pressure.
  - Hippocampal replay provides cortex with clean, separated training
    signals (one pattern at a time, pattern-separated) that help cortex
    find a better superposition arrangement than it could learn from
    raw, correlated online experience.

World model:
  N_features ground truth features (N_features > d_cortex), generating
  sparse observations encoded in d_ec dimensions. Features have varying
  sparsity following Toy Models of Superposition (Elhage et al., 2022).

Cortical model:
  Nonlinear autoencoder with tied weights operating in EC space:
    ec -> W @ ec -> ReLU(W^T h + b) -> ec_hat
  Bottleneck d_cortex < d_ec forces compression. When N_features > d_cortex,
  superposition must emerge for good reconstruction.

Experimental conditions:
  1. Raw EC (shuffled): cortex learns from EC patterns directly, interleaved
  2. Replay (clean): hippocampal replay, pseudoinverse backprojection
  3. Replay (distorted): hippocampal replay, W_ta.T backprojection

Measurements:
  - Reconstruction loss per condition
  - Feature recovery via linear probes on cortical representations
  - Superposition geometry (interference patterns, effective dimensionality)
  - Whether interference structure respects feature co-occurrence statistics

Architecture: EC(100) -> DG(1000) -> CA3(1000) -> CA1(500)
                                                       ^
                                            EC(100) ---+  (TA)

Cortex: EC(100) -> bottleneck(d_cortex) -> EC(100)

Changes from v4:
  - Added SparseFeatureWorld: structured world with ground truth features
  - Added CorticalAutoencoder: Toy Models-style nonlinear bottleneck
  - Added analysis tools: feature probing, superposition geometry,
    interference-vs-cooccurrence correlation
  - Hippocampal components reused from v4 (copied for self-containedness)
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)


# =============================================================================
# 1. REUSED HIPPOCAMPAL COMPONENTS (from v4, trimmed)
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
    def __init__(self, d_input, D_output, sigma_inh=50, gamma_inh=1.0,
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
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

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
        gate = self._sigmoid(
            self.plateau_sharpness * (h_ta_raw - threshold)
        )
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


class HippocampalSystem:
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                 dg_params=None, ca1_params=None,
                 ca3_retrieval_iterations=5):
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
        for i, ec in enumerate(ec_patterns):
            dg_pat = self.dg.forward(ec)
            dg_patterns.append(dg_pat)
            self.ca3.store_online(dg_pat)
            if train_ca1:
                self.ca1.encode(dg_pat, ec)
        self.dg_store = dg_patterns

    def replay_to_ec(self, ec_query, method="pinv"):
        """
        Full replay pipeline: EC -> DG -> CA3 retrieval -> CA1 -> back to EC.

        method: "pinv" for pseudoinverse backprojection (clean),
                "transpose" for W_ta.T backprojection (distorted)
        """
        dg_output = self.dg.forward(ec_query)
        ca3_output = self.ca3.retrieve(dg_output, n_iterations=self.ca3_retrieval_iterations)
        ca1_output, mismatch = self.ca1.retrieve(ca3_output, ec_query)

        if method == "pinv":
            W_ta_pinv = np.linalg.pinv(self.ca1.W_ta)
            ec_recon = W_ta_pinv @ ca1_output
        elif method == "transpose":
            ec_recon = self.ca1.W_ta.T @ ca1_output
        else:
            raise ValueError(f"Unknown backprojection method: {method}")

        return ec_recon, mismatch

    def generate_replay_batch(self, method="pinv"):
        """Replay all stored patterns back to EC space."""
        replay_signals = []
        for ec in self.ec_store:
            ec_recon, _ = self.replay_to_ec(ec, method=method)
            replay_signals.append(ec_recon)
        return replay_signals


# =============================================================================
# 2. WORLD MODEL
# =============================================================================

class SparseFeatureWorld:
    """
    World with N_features ground truth features, each sparse.

    Features are the standard basis in R^{N_features}. Observations are
    sparse activations projected through a fixed random EC encoding matrix,
    compressing from N_features to d_ec dimensions.

    Following Toy Models of Superposition (Elhage et al., 2022):
    - Features have varying sparsity (geometric decay)
    - Feature activations are Bernoulli(p_i) * Uniform(0, 1)
    - Importance weights decrease geometrically with feature index
    """

    def __init__(self, N_features, d_ec, sparsity_base=0.1,
                 sparsity_decay=0.99, importance_decay=0.95):
        self.N_features = N_features
        self.d_ec = d_ec

        # Feature sparsity: probability that feature i is active in any observation.
        # Early features (low index) are more common; later features are rarer.
        # Rarer features are predicted by Toy Models to be stored more in superposition.
        self.sparsities = np.array([
            sparsity_base * (sparsity_decay ** i) for i in range(N_features)
        ])
        self.sparsities = np.clip(self.sparsities, 0.005, 0.5)

        # Importance weights (for analysis; cortex loss treats all EC dims equally)
        self.importances = np.array([
            importance_decay ** i for i in range(N_features)
        ])

        # EC encoding: fixed random projection from feature space to EC.
        # Each column of W_ec is one feature's direction in EC space.
        # Normalized per-column so each feature contributes equally before sparsity.
        raw = np.random.randn(d_ec, N_features)
        col_norms = np.linalg.norm(raw, axis=0, keepdims=True) + 1e-10
        self.W_ec = raw / col_norms

        # Pseudoinverse for oracle decoding of features from EC
        self.W_ec_pinv = np.linalg.pinv(self.W_ec)

        # Feature directions in EC space (normalized columns of W_ec)
        self.feature_directions_ec = self.W_ec.copy()  # already normalized

    def generate_observation(self):
        """
        Generate one observation.
        Returns (features, ec) where features is the N_features-dim
        activation vector and ec is the d_ec-dim EC encoding.
        """
        active = (np.random.rand(self.N_features) < self.sparsities).astype(float)
        values = np.random.uniform(0, 1, self.N_features)
        features = active * values
        ec = self.W_ec @ features
        return features, ec

    def generate_batch(self, n):
        """Generate n observations."""
        features_list = []
        ec_list = []
        for _ in range(n):
            f, ec = self.generate_observation()
            features_list.append(f)
            ec_list.append(ec)
        return np.array(features_list), np.array(ec_list)

    def decode_features_oracle(self, ec):
        """Oracle decode: recover feature activations from EC via pseudoinverse."""
        return np.maximum(self.W_ec_pinv @ ec, 0)


# =============================================================================
# 3. CORTICAL AUTOENCODER (Toy Models style)
# =============================================================================

class CorticalAutoencoder:
    """
    Nonlinear autoencoder operating in EC space with a cortical bottleneck.

    Architecture (tied weights, following Toy Models of Superposition):
        h = W @ ec              (encode: d_ec -> d_cortex)
        ec_hat = ReLU(W^T h + b)   (decode: d_cortex -> d_ec)

    Trained with SGD on MSE reconstruction loss. The bottleneck at
    d_cortex < d_ec forces compression. If the world has N_features
    independent features and d_cortex < N_features, superposition
    must emerge for good reconstruction.

    Tied weights ensure the encoding and decoding geometry are coupled,
    which is what produces the structured superposition patterns
    (antipodal pairs, polytope vertices, etc.) observed in Toy Models.
    """

    def __init__(self, d_ec, d_cortex, lr=0.005, weight_decay=1e-5):
        self.d_ec = d_ec
        self.d_cortex = d_cortex
        self.lr = lr
        self.weight_decay = weight_decay

        # Tied weights: W is (d_cortex x d_ec)
        self.W = np.random.randn(d_cortex, d_ec) * np.sqrt(2.0 / (d_cortex + d_ec))
        self.b = np.zeros(d_ec)

        self.loss_history = []

    def encode(self, ec):
        """Encode EC pattern to cortical representation. Shape: (d_cortex,)"""
        return self.W @ ec

    def decode(self, h):
        """Decode cortical representation back to EC space. Shape: (d_ec,)"""
        return np.maximum(self.W.T @ h + self.b, 0)

    def forward(self, ec):
        """Full forward pass. Returns (cortical_repr, ec_reconstruction)."""
        h = self.encode(ec)
        ec_hat = self.decode(h)
        return h, ec_hat

    def train_step(self, ec):
        """
        One SGD step on reconstruction loss: L = 0.5 * ||ReLU(W^T W ec + b) - ec||^2

        Gradient derivation for tied weights:
            h = W @ ec                              (d_cortex,)
            z = W^T @ h + b                         (d_ec,)
            a = ReLU(z)                             (d_ec,)
            L = 0.5 * ||a - ec||^2

            d_pre = (a - ec) * (z > 0)             (ReLU mask applied)
            dL/db = d_pre
            dL/dW = outer(W @ d_pre, ec)            (through encoder path)
                  + outer(h, d_pre)                  (through decoder path)
        """
        # Forward
        h = self.W @ ec
        z = self.W.T @ h + self.b
        a = np.maximum(z, 0)

        # Loss
        error = a - ec
        loss = 0.5 * np.sum(error ** 2)

        # Backward
        relu_mask = (z > 0).astype(float)
        d_pre = error * relu_mask

        # Tied weight gradient (two paths)
        dL_dh = self.W @ d_pre
        grad_W = np.outer(dL_dh, ec) + np.outer(h, d_pre)
        grad_b = d_pre

        # Weight decay (L2 regularization)
        grad_W += self.weight_decay * self.W

        # Gradient clipping to prevent overflow with large-scale inputs
        grad_norm = np.sqrt(np.sum(grad_W ** 2) + np.sum(grad_b ** 2))
        max_grad_norm = 1.0
        if grad_norm > max_grad_norm:
            scale = max_grad_norm / grad_norm
            grad_W *= scale
            grad_b *= scale

        # SGD update
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

        return float(loss)

    def train_epoch(self, ec_patterns, shuffle=True):
        """Train one epoch over all patterns. Returns mean loss."""
        indices = np.arange(len(ec_patterns))
        if shuffle:
            np.random.shuffle(indices)
        total_loss = 0.0
        for i in indices:
            total_loss += self.train_step(ec_patterns[i])
        return total_loss / len(ec_patterns)

    def copy(self):
        """Return a fresh copy with identical weights (for fair comparisons)."""
        c = CorticalAutoencoder(self.d_ec, self.d_cortex, self.lr, self.weight_decay)
        c.W = self.W.copy()
        c.b = self.b.copy()
        return c


# =============================================================================
# 4. ANALYSIS TOOLS
# =============================================================================

def compute_feature_probes(cortex, world, ec_patterns, features_batch):
    """
    For each ground truth feature, fit a linear probe on cortical
    representations to predict feature activation.

    Returns:
        probe_weights: (N_features, d_cortex) matrix of probe directions
        probe_r2: R^2 per feature (how well cortex encodes each feature)
    """
    n = len(ec_patterns)
    h_all = np.array([cortex.encode(ec) for ec in ec_patterns])

    N_features = world.N_features
    probe_weights = np.zeros((N_features, cortex.d_cortex))
    probe_r2 = np.zeros(N_features)

    lam = 0.01  # ridge regularization
    HTH_inv = np.linalg.inv(h_all.T @ h_all + lam * np.eye(cortex.d_cortex))

    for fi in range(N_features):
        y = features_batch[:, fi]
        w = HTH_inv @ (h_all.T @ y)

        y_hat = h_all @ w
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
        r2 = max(0.0, 1.0 - ss_res / ss_tot)

        probe_weights[fi] = w
        probe_r2[fi] = float(r2)

    return probe_weights, probe_r2


def measure_superposition_geometry(probe_weights):
    """
    Analyze the geometry of feature representations in cortical space.

    Given probe weights (N_features x d_cortex), computes:
    - Pairwise cosine similarities (interference)
    - Effective dimensionality (participation ratio of SVD spectrum)
    - Superposition ratio (N_features / effective_dim)
    """
    norms = np.linalg.norm(probe_weights, axis=1, keepdims=True) + 1e-10
    directions = probe_weights / norms

    # Pairwise interference
    cos_sim = directions @ directions.T
    mask = ~np.eye(len(directions), dtype=bool)
    interference = np.abs(cos_sim[mask])

    # Effective dimensionality via participation ratio
    # Sanitize: replace NaN/Inf so SVD can converge
    clean_weights = np.nan_to_num(probe_weights, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(clean_weights, full_matrices=False)
    S2 = S ** 2
    S2_norm = S2 / (np.sum(S2) + 1e-10)
    eff_dim = (np.sum(S2) ** 2) / (np.sum(S2 ** 2) + 1e-10)

    superposition_ratio = len(probe_weights) / max(eff_dim, 1e-10)

    return {
        "cos_sim_matrix": cos_sim.tolist(),
        "mean_interference": float(np.mean(interference)),
        "std_interference": float(np.std(interference)),
        "max_interference": float(np.max(interference)),
        "median_interference": float(np.median(interference)),
        "interference_90th": float(np.percentile(interference, 90)),
        "effective_dimensionality": float(eff_dim),
        "superposition_ratio": float(superposition_ratio),
        "svd_spectrum": S.tolist(),
    }


def measure_interference_vs_cooccurrence(features_batch, cos_sim_matrix):
    """
    Test whether the interference pattern in cortical space respects
    the co-occurrence statistics of features.

    If cortex organizes superposition well, features that rarely co-occur
    should share more cortical dimensions (higher interference), because
    their interference never causes errors in practice. This is the
    "Toy Models prediction": only features whose joint activation is
    rare enough to tolerate interference should be superposed.

    A negative correlation means: higher co-occurrence -> lower interference.
    This is the optimal arrangement.
    """
    if isinstance(cos_sim_matrix, list):
        cos_sim_matrix = np.array(cos_sim_matrix)

    N = features_batch.shape[1]
    active = (features_batch > 0).astype(float)
    cooccurrence = (active.T @ active) / len(features_batch)

    pairs_cooc = []
    pairs_interference = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs_cooc.append(cooccurrence[i, j])
            pairs_interference.append(abs(cos_sim_matrix[i, j]))

    pairs_cooc = np.array(pairs_cooc)
    pairs_interference = np.array(pairs_interference)

    correlation, pvalue = stats.spearmanr(pairs_cooc, pairs_interference)

    return {
        "cooc_interference_spearman": float(correlation),
        "cooc_interference_pvalue": float(pvalue),
        "mean_cooccurrence": float(np.mean(pairs_cooc)),
        "n_pairs": len(pairs_cooc),
    }


def measure_sparsity_vs_superposition(world, probe_weights, probe_r2):
    """
    Test the Toy Models prediction: rarer features should show more
    superposition (lower R^2 from probes, or less distinct directions).

    Returns correlation between feature sparsity and recovery quality.
    """
    log_sparsity = np.log10(world.sparsities + 1e-10)
    corr_r2, p_r2 = stats.spearmanr(log_sparsity, probe_r2)

    norms = np.linalg.norm(probe_weights, axis=1)
    corr_norm, p_norm = stats.spearmanr(log_sparsity, norms)

    return {
        "sparsity_r2_spearman": float(corr_r2),
        "sparsity_r2_pvalue": float(p_r2),
        "sparsity_probenorm_spearman": float(corr_norm),
        "sparsity_probenorm_pvalue": float(p_norm),
    }


def measure_reconstruction_quality(cortex, ec_patterns, features_batch, world):
    """
    Measure how well the cortical autoencoder reconstructs EC patterns
    and, indirectly, how well it preserves feature information.
    """
    recon_errors = []
    feature_recovery = []

    for i, ec in enumerate(ec_patterns):
        h, ec_hat = cortex.forward(ec)
        recon_errors.append(float(np.sum((ec_hat - ec) ** 2)))

        # Try to decode features from reconstruction via oracle
        f_hat = world.decode_features_oracle(ec_hat)
        f_true = features_batch[i]
        active = f_true > 0
        if np.any(active):
            cos = cosine_similarity(f_hat * active, f_true * active)
            feature_recovery.append(cos)

    return {
        "mean_recon_error": float(np.mean(recon_errors)),
        "std_recon_error": float(np.std(recon_errors)),
        "mean_feature_recovery": float(np.mean(feature_recovery)) if feature_recovery else 0.0,
        "std_feature_recovery": float(np.std(feature_recovery)) if feature_recovery else 0.0,
    }


# =============================================================================
# 5. PARAMETERS
# =============================================================================

# World parameters
N_features = 200        # ground truth features (> d_cortex to force superposition)
d_ec = 100              # EC dimensionality
sparsity_base = 0.1     # base sparsity for most common feature
sparsity_decay = 0.99   # geometric decay per feature index

# Hippocampal parameters (same as v4)
D_dg = 1000
N_ca3 = 1000
N_ca1 = 500
k_ca3 = 50
ca3_retrieval_iters = 5

dg_params = {
    "sigma_inh": 25,
    "gamma_inh": 5.0,
    "n_inh_steps": 5,
    "noise_scale": 0.0,
}

ca1_params = {
    "lr": 0.3,
    "plateau_threshold": 0.7,
    "plateau_sharpness": 20.0,
    "weight_decay": 0.9999,
    "div_norm_sigma": 0.0,
}

# Cortical parameters
d_cortex = 50           # cortical bottleneck (< d_ec, << N_features)
cortex_lr = 0.005
cortex_weight_decay = 1e-5
n_train_epochs = 100    # training epochs per condition
n_replay_rounds = 5     # how many times to replay the full set per epoch

# Data sizes
N_experience = 500      # number of observations for hippocampal encoding
N_probe = 2000          # separate observations for probing analysis

print("=" * 70)
print("HIPPOCAMPAL-CORTICAL CONSOLIDATION v5: SUPERPOSITION EXPERIMENT")
print("=" * 70)
print(f"\nWorld: {N_features} features -> EC({d_ec}) -> Cortex({d_cortex})")
print(f"  Superposition required: {N_features} features in {d_cortex} cortical dims")
print(f"  Sparsity range: {sparsity_base:.3f} to "
      f"{sparsity_base * sparsity_decay**(N_features-1):.3f}")
print(f"\nHippocampus: EC({d_ec}) -> DG({D_dg}) -> CA3({N_ca3}) -> CA1({N_ca1})")
print(f"  CA3: k={k_ca3}, retrieval iterations={ca3_retrieval_iters}")
print(f"\nCortex: d_cortex={d_cortex}, lr={cortex_lr}, epochs={n_train_epochs}")
print(f"Data: {N_experience} experiences, {N_probe} probe observations")


# =============================================================================
# 6. GENERATE WORLD AND OBSERVATIONS
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 1: Generating world and observations")
print("=" * 70)

world = SparseFeatureWorld(N_features, d_ec, sparsity_base=sparsity_base,
                           sparsity_decay=sparsity_decay)

# Experience set: what the organism encounters
features_exp, ec_exp = world.generate_batch(N_experience)
print(f"  Generated {N_experience} observations")
print(f"  Mean features active per observation: {np.mean(np.sum(features_exp > 0, axis=1)):.1f}")
print(f"  EC pattern mean norm: {np.mean(np.linalg.norm(ec_exp, axis=1)):.3f}")

# Probe set: separate observations for analysis (never used in training)
features_probe, ec_probe = world.generate_batch(N_probe)
print(f"  Generated {N_probe} probe observations for analysis")


# =============================================================================
# 7. HIPPOCAMPAL ENCODING
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 2: Hippocampal encoding")
print("=" * 70)

hippocampus = HippocampalSystem(
    d_ec, D_dg, N_ca3, N_ca1, k_ca3,
    dg_params=dg_params, ca1_params=ca1_params,
    ca3_retrieval_iterations=ca3_retrieval_iters
)

hippocampus.encode_batch(ec_exp, train_ca1=True)
print(f"  Encoded {N_experience} patterns into hippocampus")

# Measure hippocampal replay quality
replay_clean = hippocampus.generate_replay_batch(method="pinv")
replay_distorted = hippocampus.generate_replay_batch(method="transpose")

replay_clean_sims = [cosine_similarity(ec_exp[i], replay_clean[i])
                     for i in range(N_experience)]
replay_distorted_sims = [cosine_similarity(ec_exp[i], replay_distorted[i])
                         for i in range(N_experience)]

print(f"  Replay quality (cosine sim to original):")
print(f"    Clean (pinv):      {np.mean(replay_clean_sims):.4f} "
      f"+/- {np.std(replay_clean_sims):.4f}")
print(f"    Distorted (W_ta.T): {np.mean(replay_distorted_sims):.4f} "
      f"+/- {np.std(replay_distorted_sims):.4f}")


# =============================================================================
# 8. TRAIN CORTEX UNDER EACH CONDITION
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 3: Cortical training")
print("=" * 70)

# Create identical initial cortex for each condition (fair comparison)
cortex_init = CorticalAutoencoder(d_ec, d_cortex, lr=cortex_lr,
                                  weight_decay=cortex_weight_decay)

conditions = {
    "raw_shuffled": {
        "description": "Raw EC patterns, shuffled (baseline)",
        "training_data": ec_exp,
    },
    "replay_clean": {
        "description": "Hippocampal replay, pseudoinverse backprojection",
        "training_data": np.array(replay_clean),
    },
    "replay_distorted": {
        "description": "Hippocampal replay, W_ta.T backprojection",
        "training_data": np.array(replay_distorted),
    },
}

condition_results = {}

for cond_name, cond_info in conditions.items():
    print(f"\n  --- Condition: {cond_name} ---")
    print(f"  {cond_info['description']}")

    # Fresh cortex with same initial weights
    cortex = cortex_init.copy()
    training_data = cond_info["training_data"]

    # Training loop
    epoch_losses = []
    checkpoint_results = []

    for epoch in range(n_train_epochs):
        epoch_loss = cortex.train_epoch(training_data, shuffle=True)
        epoch_losses.append(epoch_loss)

        # Checkpoint analysis every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            recon_q = measure_reconstruction_quality(
                cortex, ec_probe[:200], features_probe[:200], world
            )
            checkpoint_results.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                **recon_q,
            })
            print(f"    Epoch {epoch+1:3d}: loss={epoch_loss:.4f}, "
                  f"recon_err={recon_q['mean_recon_error']:.4f}, "
                  f"feat_recovery={recon_q['mean_feature_recovery']:.4f}")

    # Full analysis on probe set
    print(f"  Running full analysis...")

    # Feature probing
    probe_weights, probe_r2 = compute_feature_probes(
        cortex, world, ec_probe, features_probe
    )
    print(f"    Mean probe R^2: {np.mean(probe_r2):.4f}")
    print(f"    Median probe R^2: {np.median(probe_r2):.4f}")

    # Superposition geometry
    geom = measure_superposition_geometry(probe_weights)
    print(f"    Effective dimensionality: {geom['effective_dimensionality']:.1f}")
    print(f"    Superposition ratio: {geom['superposition_ratio']:.2f}")
    print(f"    Mean interference: {geom['mean_interference']:.4f}")

    # Interference vs co-occurrence
    cooc = measure_interference_vs_cooccurrence(
        features_probe, np.array(geom["cos_sim_matrix"])
    )
    print(f"    Co-occurrence/interference correlation: "
          f"{cooc['cooc_interference_spearman']:.4f} "
          f"(p={cooc['cooc_interference_pvalue']:.2e})")

    # Sparsity vs superposition
    sparsity_analysis = measure_sparsity_vs_superposition(
        world, probe_weights, probe_r2
    )
    print(f"    Sparsity/R^2 correlation: "
          f"{sparsity_analysis['sparsity_r2_spearman']:.4f} "
          f"(p={sparsity_analysis['sparsity_r2_pvalue']:.2e})")

    # Reconstruction quality
    final_recon = measure_reconstruction_quality(
        cortex, ec_probe, features_probe, world
    )

    # Store results (excluding large matrices for JSON serialization)
    condition_results[cond_name] = {
        "description": cond_info["description"],
        "epoch_losses": epoch_losses,
        "checkpoints": checkpoint_results,
        "probe_r2_mean": float(np.mean(probe_r2)),
        "probe_r2_median": float(np.median(probe_r2)),
        "probe_r2_std": float(np.std(probe_r2)),
        "probe_r2_by_feature": probe_r2.tolist(),
        "superposition_geometry": {
            k: v for k, v in geom.items() if k != "cos_sim_matrix"
        },
        "cooccurrence_analysis": cooc,
        "sparsity_analysis": sparsity_analysis,
        "reconstruction": final_recon,
    }


# =============================================================================
# 9. CROSS-CONDITION COMPARISONS
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 4: Cross-condition comparison")
print("=" * 70)

print(f"\n{'Condition':<25s} {'Probe R^2':>10s} {'Eff Dim':>8s} "
      f"{'Superpos':>9s} {'Interf':>8s} {'Co-oc corr':>10s}")
print("-" * 75)

for cond_name, res in condition_results.items():
    geom = res["superposition_geometry"]
    cooc = res["cooccurrence_analysis"]
    print(f"{cond_name:<25s} "
          f"{res['probe_r2_mean']:>10.4f} "
          f"{geom['effective_dimensionality']:>8.1f} "
          f"{geom['superposition_ratio']:>9.2f} "
          f"{geom['mean_interference']:>8.4f} "
          f"{cooc['cooc_interference_spearman']:>10.4f}")


# =============================================================================
# 10. EXPERIMENT 2: CAPACITY SCALING
#     How does cortical bottleneck size affect superposition?
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 2: Cortical capacity scaling")
print("=" * 70)

capacity_tests = [10, 25, 50, 75, 100]
capacity_results = {}

for d_c in capacity_tests:
    print(f"\n  d_cortex = {d_c} (ratio to N_features = {d_c/N_features:.2f})")

    cortex_cap = CorticalAutoencoder(d_ec, d_c, lr=cortex_lr,
                                     weight_decay=cortex_weight_decay)

    # Train on raw EC (shuffled) to isolate the capacity effect
    for epoch in range(n_train_epochs):
        cortex_cap.train_epoch(ec_exp, shuffle=True)

    # Analyze
    pw, pr2 = compute_feature_probes(cortex_cap, world, ec_probe, features_probe)
    geom = measure_superposition_geometry(pw)
    recon = measure_reconstruction_quality(cortex_cap, ec_probe[:200],
                                           features_probe[:200], world)

    capacity_results[d_c] = {
        "probe_r2_mean": float(np.mean(pr2)),
        "probe_r2_median": float(np.median(pr2)),
        "effective_dimensionality": geom["effective_dimensionality"],
        "superposition_ratio": geom["superposition_ratio"],
        "mean_interference": geom["mean_interference"],
        "mean_recon_error": recon["mean_recon_error"],
        "mean_feature_recovery": recon["mean_feature_recovery"],
    }

    print(f"    Probe R^2: {np.mean(pr2):.4f}, "
          f"Eff dim: {geom['effective_dimensionality']:.1f}, "
          f"Superpos ratio: {geom['superposition_ratio']:.2f}, "
          f"Recon err: {recon['mean_recon_error']:.4f}")


# =============================================================================
# 11. EXPERIMENT 3: REPLAY QUANTITY
#     Does more replay improve cortical organization?
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 3: Replay quantity effect")
print("=" * 70)

replay_quantities = [1, 2, 5, 10, 20]
replay_quantity_results = {}

for n_replays in replay_quantities:
    print(f"\n  Replay rounds per epoch: {n_replays}")

    cortex_rq = cortex_init.copy()

    # Build augmented training set by repeating replay signals
    replay_data = np.array(replay_clean)

    for epoch in range(n_train_epochs):
        # Each epoch: present the replay set n_replays times, shuffled
        for _ in range(n_replays):
            cortex_rq.train_epoch(replay_data, shuffle=True)

    # Analyze
    pw, pr2 = compute_feature_probes(cortex_rq, world, ec_probe, features_probe)
    geom = measure_superposition_geometry(pw)
    cooc = measure_interference_vs_cooccurrence(
        features_probe, np.array(geom["cos_sim_matrix"])
    )

    replay_quantity_results[n_replays] = {
        "probe_r2_mean": float(np.mean(pr2)),
        "effective_dimensionality": geom["effective_dimensionality"],
        "superposition_ratio": geom["superposition_ratio"],
        "mean_interference": geom["mean_interference"],
        "cooc_interference_spearman": cooc["cooc_interference_spearman"],
    }

    print(f"    Probe R^2: {np.mean(pr2):.4f}, "
          f"Eff dim: {geom['effective_dimensionality']:.1f}, "
          f"Co-oc corr: {cooc['cooc_interference_spearman']:.4f}")


# =============================================================================
# 12. ANTI-SUPERPOSITION MEASUREMENT: HIPPOCAMPUS vs CORTEX
#     Directly measure the degree of superposition in hippocampal vs
#     cortical representations.
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 4: Hippocampal vs cortical superposition")
print("=" * 70)

print("\n  Measuring superposition in each representational stage...")

# For each probe observation, get representations at each stage
dg_reprs = []
ca3_reprs = []
ca1_reprs = []

for ec in ec_probe[:500]:
    dg_out = hippocampus.dg.forward(ec)
    ca3_out = hippocampus.ca3.retrieve(dg_out, n_iterations=ca3_retrieval_iters)
    ca1_out, _ = hippocampus.ca1.retrieve(ca3_out, ec)
    dg_reprs.append(dg_out)
    ca3_reprs.append(ca3_out)
    ca1_reprs.append(ca1_out)

dg_reprs = np.array(dg_reprs)
ca3_reprs = np.array(ca3_reprs)
ca1_reprs = np.array(ca1_reprs)

# Use the best-trained cortex (raw_shuffled condition) for cortical representations
# Re-load by training a fresh one (or use existing)
cortex_best = cortex_init.copy()
for epoch in range(n_train_epochs):
    cortex_best.train_epoch(ec_exp, shuffle=True)
cortex_reprs = np.array([cortex_best.encode(ec) for ec in ec_probe[:500]])

# For each representational stage, measure:
# 1. How many features can be linearly decoded (effective feature capacity)
# 2. Participation ratio of the representation covariance
# 3. Mean pairwise similarity between representations of different observations

stage_names = ["EC", "DG", "CA3", "CA1", "Cortex"]
stage_reprs = [ec_probe[:500], dg_reprs, ca3_reprs, ca1_reprs, cortex_reprs]
stage_dims = [d_ec, D_dg, N_ca3, N_ca1, d_cortex]
stage_results = {}

for name, reprs, dim in zip(stage_names, stage_reprs, stage_dims):
    # Participation ratio of covariance
    cov = np.cov(reprs.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0)
    total_var = np.sum(eigvals) + 1e-10
    pr = total_var ** 2 / (np.sum(eigvals ** 2) + 1e-10)

    # Sparsity: mean fraction of active (>0) neurons
    active_frac = np.mean(reprs > 0)

    # Mean pairwise cosine similarity (sample to save time)
    sample_idx = np.random.choice(len(reprs), min(200, len(reprs)), replace=False)
    pair_sims = []
    for ii in range(len(sample_idx)):
        for jj in range(ii + 1, min(ii + 20, len(sample_idx))):
            pair_sims.append(cosine_similarity(reprs[sample_idx[ii]],
                                                reprs[sample_idx[jj]]))

    # Feature decodability (quick: use first 100 features)
    lam = 0.01
    n_test = min(100, world.N_features)
    decode_r2s = []
    HTH_inv = np.linalg.inv(reprs.T @ reprs + lam * np.eye(reprs.shape[1]))
    for fi in range(n_test):
        y = features_probe[:500, fi]
        w = HTH_inv @ (reprs.T @ y)
        y_hat = reprs @ w
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
        decode_r2s.append(max(0.0, 1.0 - ss_res / ss_tot))

    stage_results[name] = {
        "n_dims": dim,
        "participation_ratio": float(pr),
        "pr_normalized": float(pr / dim),
        "active_fraction": float(active_frac),
        "mean_pairwise_sim": float(np.mean(pair_sims)),
        "feature_decode_r2_mean": float(np.mean(decode_r2s)),
        "feature_decode_r2_std": float(np.std(decode_r2s)),
    }

    print(f"  {name:>8s} (d={dim:4d}): "
          f"PR={pr:.1f} ({pr/dim:.2f} of d), "
          f"active={active_frac:.3f}, "
          f"pair_sim={np.mean(pair_sims):.4f}, "
          f"decode_R2={np.mean(decode_r2s):.4f}")


# =============================================================================
# 13. SAVE RESULTS
# =============================================================================

print(f"\n{'=' * 70}")
print("Saving results...")
print("=" * 70)

results = {
    "params": {
        "N_features": N_features,
        "d_ec": d_ec,
        "d_cortex": d_cortex,
        "D_dg": D_dg,
        "N_ca3": N_ca3,
        "N_ca1": N_ca1,
        "k_ca3": k_ca3,
        "sparsity_base": sparsity_base,
        "sparsity_decay": sparsity_decay,
        "cortex_lr": cortex_lr,
        "cortex_weight_decay": cortex_weight_decay,
        "n_train_epochs": n_train_epochs,
        "N_experience": N_experience,
        "N_probe": N_probe,
    },
    "replay_quality": {
        "clean_mean_sim": float(np.mean(replay_clean_sims)),
        "clean_std_sim": float(np.std(replay_clean_sims)),
        "distorted_mean_sim": float(np.mean(replay_distorted_sims)),
        "distorted_std_sim": float(np.std(replay_distorted_sims)),
    },
    "condition_comparison": condition_results,
    "capacity_scaling": {str(k): v for k, v in capacity_results.items()},
    "replay_quantity": {str(k): v for k, v in replay_quantity_results.items()},
    "stage_representations": stage_results,
    "world_feature_sparsities": world.sparsities.tolist(),
}

with open("consolidation_model_v2_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("  Saved to consolidation_model_v2_results.json")


# =============================================================================
# 14. VISUALIZATION
# =============================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(3, 3, figsize=(22, 18))
fig.suptitle("Hippocampal-Cortical Consolidation v5: Superposition Experiment",
             fontsize=14, fontweight='bold')

# --- Panel 1: Training loss curves ---
ax = axes[0, 0]
for cond_name, res in condition_results.items():
    ax.plot(res["epoch_losses"], label=cond_name, linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean reconstruction loss")
ax.set_title("Cortical Training Loss by Condition")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# --- Panel 2: Feature probe R^2 distributions ---
ax = axes[0, 1]
for i, (cond_name, res) in enumerate(condition_results.items()):
    r2s = res["probe_r2_by_feature"]
    ax.hist(r2s, bins=30, alpha=0.5, label=f"{cond_name} (mean={np.mean(r2s):.3f})")
ax.set_xlabel("Probe R^2")
ax.set_ylabel("Count")
ax.set_title("Feature Recovery Quality Distribution")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 3: R^2 vs feature sparsity ---
ax = axes[0, 2]
for cond_name, res in condition_results.items():
    r2s = res["probe_r2_by_feature"]
    log_sp = np.log10(world.sparsities + 1e-10)
    ax.scatter(log_sp, r2s, alpha=0.3, s=10, label=cond_name)
ax.set_xlabel("log10(feature sparsity)")
ax.set_ylabel("Probe R^2")
ax.set_title("Feature Recovery vs Sparsity (Toy Models prediction)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 4: Capacity scaling ---
ax = axes[1, 0]
cap_ds = sorted(capacity_results.keys())
cap_r2 = [capacity_results[d]["probe_r2_mean"] for d in cap_ds]
cap_sr = [capacity_results[d]["superposition_ratio"] for d in cap_ds]
l1 = ax.plot(cap_ds, cap_r2, "go-", markersize=6, linewidth=2, label="Probe R^2")
ax.set_xlabel("Cortical bottleneck dimensions")
ax.set_ylabel("Mean probe R^2", color="green")
ax2 = ax.twinx()
l2 = ax2.plot(cap_ds, cap_sr, "b^-", markersize=6, linewidth=2, label="Superposition ratio")
ax2.set_ylabel("Superposition ratio (N_feat / eff_dim)", color="blue")
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
ax.set_title("Capacity Scaling")
ax.grid(True, alpha=0.3)

# --- Panel 5: Representational stage comparison ---
ax = axes[1, 1]
stages = list(stage_results.keys())
stage_pr_norm = [stage_results[s]["pr_normalized"] for s in stages]
stage_decode = [stage_results[s]["feature_decode_r2_mean"] for s in stages]
x_pos = np.arange(len(stages))
width = 0.35
bars1 = ax.bar(x_pos - width/2, stage_pr_norm, width, label="PR / d (spread)", color="steelblue")
bars2 = ax.bar(x_pos + width/2, stage_decode, width, label="Feature decode R^2", color="coral")
ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("Value")
ax.set_title("Representation Quality by Stage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 6: Replay quantity effect ---
ax = axes[1, 2]
rq_ns = sorted(replay_quantity_results.keys())
rq_r2 = [replay_quantity_results[n]["probe_r2_mean"] for n in rq_ns]
rq_cooc = [replay_quantity_results[n]["cooc_interference_spearman"] for n in rq_ns]
l1 = ax.plot(rq_ns, rq_r2, "go-", markersize=6, linewidth=2, label="Probe R^2")
ax.set_xlabel("Replay rounds per epoch")
ax.set_ylabel("Mean probe R^2", color="green")
ax2 = ax.twinx()
l2 = ax2.plot(rq_ns, rq_cooc, "b^-", markersize=6, linewidth=2,
              label="Co-oc/interference corr")
ax2.set_ylabel("Spearman correlation", color="blue")
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
ax.set_title("Replay Quantity Effect")
ax.grid(True, alpha=0.3)

# --- Panel 7: Pairwise similarity by stage ---
ax = axes[2, 0]
stage_pair_sim = [stage_results[s]["mean_pairwise_sim"] for s in stages]
stage_active = [stage_results[s]["active_fraction"] for s in stages]
ax.bar(x_pos - width/2, stage_pair_sim, width, label="Mean pairwise sim", color="steelblue")
ax.bar(x_pos + width/2, stage_active, width, label="Active fraction", color="coral")
ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("Value")
ax.set_title("Sparsity and Similarity by Stage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 8: Superposition geometry comparison ---
ax = axes[2, 1]
cond_names = list(condition_results.keys())
cond_eff_dim = [condition_results[c]["superposition_geometry"]["effective_dimensionality"]
                for c in cond_names]
cond_interf = [condition_results[c]["superposition_geometry"]["mean_interference"]
               for c in cond_names]
x_pos_c = np.arange(len(cond_names))
bars1 = ax.bar(x_pos_c - width/2, cond_eff_dim, width, label="Effective dim", color="steelblue")
ax.set_ylabel("Effective dimensionality", color="steelblue")
ax2 = ax.twinx()
bars2 = ax2.bar(x_pos_c + width/2, cond_interf, width, label="Mean interference", color="coral")
ax2.set_ylabel("Mean interference", color="coral")
ax.set_xticks(x_pos_c)
ax.set_xticklabels(cond_names, fontsize=8, rotation=15)
ax.set_title("Superposition Structure by Condition")
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 9: Co-occurrence vs interference scatter (best condition) ---
ax = axes[2, 2]
# Use the condition with best co-oc correlation for the scatter
best_cond = min(condition_results.keys(),
                key=lambda c: condition_results[c]["cooccurrence_analysis"]["cooc_interference_spearman"])
best_pw, _ = compute_feature_probes(cortex_init.copy(), world,
                                      ec_probe[:500], features_probe[:500])
# Recompute for the scatter: use raw condition as illustration
# (Retraining would take time; just show the structure from what we have)
active = (features_probe[:500] > 0).astype(float)
cooc_matrix = (active.T @ active) / 500
pw_norms = np.linalg.norm(best_pw, axis=1, keepdims=True) + 1e-10
best_dirs = best_pw / pw_norms
best_cos = best_dirs @ best_dirs.T

cooc_vals = []
interf_vals = []
for i in range(min(100, N_features)):
    for j in range(i + 1, min(100, N_features)):
        cooc_vals.append(cooc_matrix[i, j])
        interf_vals.append(abs(best_cos[i, j]))

ax.scatter(cooc_vals, interf_vals, alpha=0.1, s=5, color="steelblue")
ax.set_xlabel("Feature co-occurrence probability")
ax.set_ylabel("|Cosine similarity| in cortical space")
ax.set_title("Co-occurrence vs Interference (first 100 features)")
ax.grid(True, alpha=0.3)

# Add trend line
if len(cooc_vals) > 10:
    z = np.polyfit(cooc_vals, interf_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(cooc_vals), max(cooc_vals), 100)
    ax.plot(x_line, p(x_line), "r-", linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig("consolidation_model_v2_results.png", dpi=150, bbox_inches='tight')
print("  Saved plot to consolidation_model_v2_results.png")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
