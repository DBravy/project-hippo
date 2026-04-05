"""
Hippocampal Model v4: CA1 with BTSP Learning + Online CA3 Storage
=================================================================

Changes from v3:
  - CA1 added with BTSP-based Schaffer collateral learning
  - CA3 storage switched from batch covariance to online Hebbian rule
    with Welford running mean (patterns stored one at a time)
  - CA3 retrieval uses 5 iterations (gamma-cycle timescale)
  - Divisive normalization in CA1 for pathway comparison
  - Normalized mismatch signal for novelty detection

CA1 receives two inputs:
  1. Schaffer collaterals from CA3 (learnable weights, BTSP rule)
  2. Temporoammonic pathway from EC Layer III (fixed random projection)

CA1 learns online: each encoding episode updates Schaffer weights via
a BTSP-like rule gated by dendritic plateau potentials (driven by EC
input). Over time, CA1 learns to decode CA3's internal code into a
representation that matches the EC-driven target.

The mismatch signal (divergence between CA3-predicted and EC-driven
activation) emerges naturally as the error term of the learning rule.

Architecture: EC(100) -> DG(1000) -> CA3(1000, online Hebbian) -> CA1(500)
                                                                    ^
                                                         EC(100) ---+  (TA)

Experiments:
  1. CA1 learning curve: mismatch and decoding accuracy over episodes
  2. Novelty detection: familiar vs novel discrimination
  3. Retrieval improvement: CA1 output vs raw CA3 output
  4. Generalization: retrieval from similar-but-new cues
  5. Multi-pass learning: improvement with repeated exposure
  6. Capacity scaling
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# =============================================================================
# 1. REUSED COMPONENTS (from v3)
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


def generate_ec_patterns(n, d_ec):
    patterns = []
    for _ in range(n):
        ec = np.random.randn(d_ec)
        ec = ec / np.linalg.norm(ec)
        patterns.append(ec)
    return patterns


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
    """
    Autoassociative network with online Hebbian storage.

    Each pattern is stored incrementally as it arrives, using a running
    mean (Welford's algorithm) to center the outer-product updates.
    This replaces the batch covariance rule from v3, which required
    seeing all patterns before computing the weight matrix.

    The online rule:
        For each new pattern p (L2-normalized):
            n += 1
            μ = μ + (p - μ) / n          (Welford running mean)
            p_centered = p - μ
            W += p_centered * p_centered^T
            zero diagonal of W

    This is mathematically equivalent to the batch rule if the mean
    were known in advance. Because early patterns are centered against
    an inaccurate mean estimate, they are stored with slightly more
    error than later patterns (running-mean drift). This is biologically
    plausible: the hippocampus needs some initial experience to
    calibrate its baseline, and earliest memories in a novel context
    are less stable.

    Retrieval uses iterative dynamics with k-WTA sparsification.
    """
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.stored_patterns = []
        self.mean_activity = np.zeros(N)

    def store_online(self, pattern):
        """
        Store a single pattern using online Hebbian rule.

        The pattern is L2-normalized, centered against the running mean,
        and its outer product is added to W.
        """
        # Normalize
        p = pattern.copy()
        norm = np.linalg.norm(p) + 1e-10
        p = p / norm

        # Update running mean (Welford's online algorithm)
        self.n_stored += 1
        self.mean_activity = self.mean_activity + (p - self.mean_activity) / self.n_stored

        # Center against current mean estimate and store
        p_centered = p - self.mean_activity
        self.W += np.outer(p_centered, p_centered)
        np.fill_diagonal(self.W, 0)

        # Keep a copy for evaluation
        self.stored_patterns.append(pattern.copy())

    def store_batch(self, patterns):
        """
        Store all patterns using batch covariance rule (kept for
        comparison). Equivalent to the v3 implementation.
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

    def retrieve(self, cue, n_iterations=5):
        """
        Pattern completion via iterative dynamics with k-WTA.

        Default n_iterations=5, corresponding roughly to the number
        of gamma cycles within one theta half-cycle during which
        CA3 recurrent dynamics settle (~5-7 gamma cycles at ~40Hz
        within ~125ms theta half-period).
        """
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
# 2. CA1 IMPLEMENTATION
# =============================================================================

class CA1:
    """
    CA1 with BTSP-based learning of Schaffer collateral weights.

    Two input pathways:
      - Schaffer collaterals (CA3 -> CA1): learnable weights W_sc
      - Temporoammonic (EC III -> CA1): fixed weights W_ta

    Learning rule (BTSP):
      1. EC input drives target activation via fixed W_ta
      2. CA3 input drives predicted activation via learnable W_sc
      3. Plateau potential (gated by strong EC drive) enables learning
         at a SPARSE subset of CA1 neurons (~3-5%)
      4. Error between target and prediction updates W_sc at gated neurons

    Changes from v4a:
      - Sharper plateau gating: threshold raised so only ~3-5% of CA1
        neurons get plateau potentials per episode. Each pattern "owns"
        a small set of neurons, reducing cross-pattern interference.
      - Higher learning rate to compensate: fewer neurons learning means
        each one can make larger weight changes per episode.
      - Divisive normalization on both pathways before error computation.
        Each neuron's activation is divided by the local population
        activity pool, making the comparison invariant to overall drive
        level (Carandini & Heeger, 2012).
      - Normalized mismatch signal: relative error (||error||/||target||)
        instead of raw absolute error. Separates the novelty signal from
        the baseline activation magnitude.
      - Weight decay reduced: the error-correcting rule is self-limiting,
        so aggressive decay is unnecessary and was erasing learning.

    Parameters
    ----------
    N_ca1 : int
        Number of CA1 pyramidal cells.
    N_ca3 : int
        Dimensionality of CA3 input.
    d_ec : int
        Dimensionality of EC input.
    lr : float
        BTSP learning rate. Large because plateau gating restricts
        updates to a sparse subset of neurons.
    plateau_threshold : float
        Threshold for plateau potential. Fraction of max EC-driven
        activation. Higher = fewer neurons with plateaus = sparser
        learning. 0.7 targets ~3-5% of CA1.
    plateau_sharpness : float
        Inverse temperature for the sigmoid plateau gate. Higher values
        produce a sharper on/off gate.
    weight_decay : float
        Per-episode multiplicative decay on W_sc. Set near 1.0 since
        the error-correcting rule is self-limiting.
    div_norm_sigma : float
        Semi-saturation constant for divisive normalization. Controls
        the operating point: larger values make normalization weaker
        (activations stay closer to raw values).
    connectivity_prob : float
        Sparsity of both W_ta and W_sc initial connectivity.
    """
    def __init__(self, N_ca1, N_ca3, d_ec,
                 lr=0.3,
                 plateau_threshold=0.7,
                 plateau_sharpness=20.0,
                 weight_decay=0.9999,
                 div_norm_sigma=0.1,
                 connectivity_prob=0.33):
        self.N_ca1 = N_ca1
        self.N_ca3 = N_ca3
        self.d_ec = d_ec
        self.lr = lr
        self.plateau_threshold = plateau_threshold
        self.plateau_sharpness = plateau_sharpness
        self.weight_decay = weight_decay
        self.div_norm_sigma = div_norm_sigma

        # Fixed temporoammonic weights (EC III -> CA1 distal dendrites)
        self.W_ta = make_feedforward_weights(N_ca1, d_ec, connectivity_prob)

        # Learnable Schaffer collateral weights (CA3 -> CA1 proximal dendrites)
        # Initialized as small random values (not the large normalized random
        # projection used for W_ta), so initial CA3-driven activation is weak
        mask = (np.random.rand(N_ca1, N_ca3) < connectivity_prob).astype(float)
        self.W_sc = np.random.randn(N_ca1, N_ca3) * 0.01 * mask

        # Tracking
        self.n_episodes = 0
        self.learning_history = []

    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def _divisive_normalize(self, h):
        """
        Divisive normalization (Carandini & Heeger, 2012).

        R_i = h_i / (sigma + mean(h))

        Each neuron's response is normalized by the global pool activity.
        This makes the representation invariant to overall drive level
        while preserving the relative pattern of activation.

        Uses global mean rather than local neighborhood because CA1
        pyramidal cells don't have strong topographic organization.
        """
        pool = np.mean(h) + self.div_norm_sigma
        return h / pool

    def compute_activations(self, x_ca3, x_ec):
        """
        Compute both input-driven activations without learning.

        Both pathways undergo divisive normalization independently
        before any comparison, so the error signal reflects pattern
        mismatch rather than drive magnitude differences.

        Returns
        -------
        h_ta : ndarray
            EC-driven (temporoammonic) activation, divisively normalized.
        h_sc : ndarray
            CA3-driven (Schaffer collateral) activation, divisively normalized.
        gate : ndarray
            Plateau potential gate. Near 1 for neurons with strong
            EC drive, near 0 otherwise.
        h_ta_raw : ndarray
            Raw (pre-normalization) EC-driven activation. Used for
            plateau threshold computation.
        """
        # Raw activations
        h_ta_raw = np.maximum(self.W_ta @ x_ec, 0)
        h_sc_raw = np.maximum(self.W_sc @ x_ca3, 0)

        # Divisive normalization (independent per pathway)
        h_ta = self._divisive_normalize(h_ta_raw)
        h_sc = self._divisive_normalize(h_sc_raw)

        # Plateau gate: based on RAW EC activation (before normalization)
        # because the plateau potential is a dendritic event driven by
        # absolute input strength, not normalized activity
        if np.max(h_ta_raw) > 1e-10:
            threshold = self.plateau_threshold * np.max(h_ta_raw)
        else:
            threshold = 0.0
        gate = self._sigmoid(
            self.plateau_sharpness * (h_ta_raw - threshold)
        )

        return h_ta, h_sc, gate, h_ta_raw

    def encode(self, x_ca3, x_ec):
        """
        Process one encoding episode. Updates Schaffer weights via BTSP.

        Parameters
        ----------
        x_ca3 : ndarray
            CA3 output (the stored/retrieved DG pattern after CA3 processing).
        x_ec : ndarray
            EC Layer III input (current sensory state).

        Returns
        -------
        mismatch : float
            Mismatch magnitude (gated error norm). High for novel inputs,
            low for well-learned associations.
        info : dict
            Diagnostic information about this encoding step.
        """
        h_ta, h_sc, gate, h_ta_raw = self.compute_activations(x_ca3, x_ec)

        # Error: what EC says should be active vs what CA3 predicts
        # Computed on divisively normalized activations so the error
        # reflects pattern mismatch, not drive magnitude
        error = h_ta - h_sc

        # Gated error: only at neurons with plateau potentials
        gated_error = gate * error

        # Normalized mismatch signal: relative error
        # ||gated_error|| / ||h_ta|| makes the signal invariant to
        # overall activation level. A mismatch of 0.5 means the
        # Schaffer prediction accounts for 50% of the EC target.
        h_ta_norm = np.linalg.norm(h_ta) + 1e-10
        mismatch = float(np.linalg.norm(gated_error) / h_ta_norm)

        # BTSP weight update
        # Only neurons with plateau potentials update their Schaffer weights
        # The update direction is the error, scaled by CA3 input
        delta_W = self.lr * np.outer(gated_error, x_ca3)
        self.W_sc += delta_W

        # Weight decay (homeostatic)
        self.W_sc *= self.weight_decay

        self.n_episodes += 1

        # Diagnostics
        n_plateau = int(np.sum(gate > 0.5))
        mean_error = float(np.mean(np.abs(error)))
        mean_gated_error = float(np.mean(np.abs(gated_error)))

        info = {
            "mismatch": mismatch,
            "n_plateau_neurons": n_plateau,
            "plateau_fraction": n_plateau / self.N_ca1,
            "mean_abs_error": mean_error,
            "mean_abs_gated_error": mean_gated_error,
            "h_ta_mean": float(np.mean(h_ta[h_ta > 0])) if np.any(h_ta > 0) else 0.0,
            "h_sc_mean": float(np.mean(h_sc[h_sc > 0])) if np.any(h_sc > 0) else 0.0,
            "w_sc_norm": float(np.linalg.norm(self.W_sc)),
        }
        self.learning_history.append(info)

        return mismatch, info

    def retrieve(self, x_ca3, x_ec):
        """
        Retrieval mode: compute CA1 output from both inputs.

        The output combines CA3-driven and EC-driven activation.
        In retrieval, the Schaffer pathway (CA3) is primary, and the
        EC pathway provides context and enables mismatch detection.

        Returns
        -------
        output : ndarray
            CA1 output activation.
        mismatch : float
            Mismatch magnitude.
        info : dict
            Diagnostic information.
        """
        h_ta, h_sc, gate, h_ta_raw = self.compute_activations(x_ca3, x_ec)

        # Error and mismatch (same computation as encoding, but no learning)
        error = h_ta - h_sc
        gated_error = gate * error
        h_ta_norm = np.linalg.norm(h_ta) + 1e-10
        mismatch = float(np.linalg.norm(gated_error) / h_ta_norm)

        # CA1 output: the CA3-driven activation (what the learned decoder
        # produces). This is what gets sent to subiculum / EC deep layers.
        # We use h_sc as the primary output because in retrieval mode,
        # the Schaffer pathway carries the memory signal.
        output = h_sc.copy()

        info = {
            "mismatch": mismatch,
            "h_ta_active": int(np.count_nonzero(h_ta)),
            "h_sc_active": int(np.count_nonzero(h_sc)),
            "output_active": int(np.count_nonzero(output)),
        }

        return output, mismatch, info

    def get_ec_target(self, x_ec):
        """Get the EC-driven activation pattern (for evaluation).
        Returns divisively normalized activation to match what
        compute_activations produces."""
        h_raw = np.maximum(self.W_ta @ x_ec, 0)
        return self._divisive_normalize(h_raw)


# =============================================================================
# 3. FULL PIPELINE
# =============================================================================

class HippocampalSystem:
    """
    Full pipeline: EC -> DG -> CA3 -> CA1 (with EC -> CA1 direct).

    Provides encode/retrieve methods that run the full circuit.

    CA3 storage is online: patterns are stored one at a time as they
    arrive, using a running mean for centering.

    CA3 retrieval uses iterative recurrent dynamics (default 5 steps,
    corresponding to ~5 gamma cycles within a theta half-period).
    """
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

        # Storage for the EC patterns (for evaluation)
        self.ec_store = []
        self.dg_store = []

    def encode_batch(self, ec_patterns, train_ca1=True):
        """
        Encode a batch of EC patterns, stored in CA3 ONLINE (one by one).

        1. Each pattern goes through DG, then is stored in CA3 incrementally.
        2. If train_ca1, each pattern is also presented to CA1 for
           online BTSP learning.

        During encoding, CA1 sees the clean DG pattern via Schaffer
        collaterals (mossy fiber forcing mode, no CA3 recurrent dynamics).

        Returns list of per-episode CA1 learning info.
        """
        self.ec_store = [ec.copy() for ec in ec_patterns]

        # Process and store each pattern one at a time (online)
        dg_patterns = []
        ca1_infos = []

        for i, ec in enumerate(ec_patterns):
            # DG forward pass
            dg_pat = self.dg.forward(ec)
            dg_patterns.append(dg_pat)

            # CA3 online storage (one pattern at a time)
            self.ca3.store_online(dg_pat)

            # CA1 online learning
            if train_ca1:
                # During encoding, mossy fiber forcing: CA1 sees the clean
                # DG pattern directly (no CA3 retrieval dynamics)
                mismatch, info = self.ca1.encode(dg_pat, ec)
                info["episode_idx"] = i
                ca1_infos.append(info)

        self.dg_store = dg_patterns
        return ca1_infos

    def retrieve_from_ec(self, ec_query, ca3_iterations=None):
        """
        Full retrieval pipeline.

        ec_query goes through DG -> CA3 -> CA1, and also directly
        to CA1 via temporoammonic pathway.

        ca3_iterations: how many CA3 recurrent steps to run.
            None = use system default (self.ca3_retrieval_iterations)
            0 = use DG pattern directly (strong mossy fiber mode)
            >0 = run pattern completion (weak cue mode)
        """
        if ca3_iterations is None:
            ca3_iterations = self.ca3_retrieval_iterations

        # DG processes the query
        dg_output = self.dg.forward(ec_query)

        # CA3: either pass-through or pattern completion
        if ca3_iterations == 0:
            # Strong cue mode: DG pattern used directly
            ca3_output = dg_output / (np.linalg.norm(dg_output) + 1e-10)
        else:
            ca3_output = self.ca3.retrieve(dg_output, n_iterations=ca3_iterations)

        # CA1: integrate CA3 output with direct EC input
        ca1_output, mismatch, info = self.ca1.retrieve(ca3_output, ec_query)

        return {
            "dg_output": dg_output,
            "ca3_output": ca3_output,
            "ca1_output": ca1_output,
            "mismatch": mismatch,
            "ca1_info": info,
        }


# =============================================================================
# 4. PARAMETERS
# =============================================================================

d_ec = 100
D_dg = 1000
N_ca3 = 1000
N_ca1 = 500       # CA1 is smaller than CA3 (dimensionality compression)
k_ca3 = 50

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
    "div_norm_sigma": 0.1,
}

ca3_retrieval_iters = 5  # ~5 gamma cycles within a theta half-period

print(f"Architecture: EC({d_ec}) -> DG({D_dg}) -> CA3({N_ca3}) -> CA1({N_ca1})")
print(f"                                                          ^")
print(f"                                               EC({d_ec}) -+  (TA)")
print(f"CA3: online Hebbian storage, retrieval iterations={ca3_retrieval_iters}")
print(f"CA1 params: lr={ca1_params['lr']}, plateau_θ={ca1_params['plateau_threshold']}, "
      f"decay={ca1_params['weight_decay']}, div_norm_σ={ca1_params['div_norm_sigma']}")
print()


# =============================================================================
# 5. EXPERIMENT 1: CA1 Learning Curve
#    Store N patterns, train CA1 sequentially, track mismatch over time
# =============================================================================

print("=" * 70)
print("EXPERIMENT 1: CA1 Learning Curve")
print("=" * 70)

N_store = 200
system = HippocampalSystem(d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                           dg_params=dg_params, ca1_params=ca1_params, ca3_retrieval_iterations=ca3_retrieval_iters)

ec_patterns = generate_ec_patterns(N_store, d_ec)
ca1_infos = system.encode_batch(ec_patterns, train_ca1=True)

# Extract learning curve
mismatches = [info["mismatch"] for info in ca1_infos]
plateau_fracs = [info["plateau_fraction"] for info in ca1_infos]
w_norms = [info["w_sc_norm"] for info in ca1_infos]

print(f"  N={N_store} patterns encoded")
print(f"  Mismatch:  first 10 mean={np.mean(mismatches[:10]):.4f}, "
      f"last 10 mean={np.mean(mismatches[-10:]):.4f}")
print(f"  Plateau fraction: mean={np.mean(plateau_fracs):.3f}")
print(f"  W_sc norm: start={w_norms[0]:.4f}, end={w_norms[-1]:.4f}")


# =============================================================================
# 6. EXPERIMENT 2: Re-presentation Mismatch
#    After encoding, re-present each stored pattern. Does CA1 recognize
#    them (low mismatch)? Compare to novel patterns.
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 2: Familiar vs Novel Mismatch")
print("=" * 70)

# Familiar: re-present stored EC patterns through full pipeline
familiar_mismatches = []
for ec in ec_patterns:
    result = system.retrieve_from_ec(ec)
    familiar_mismatches.append(result["mismatch"])

# Novel: present new EC patterns
novel_ec = generate_ec_patterns(N_store, d_ec)
novel_mismatches = []
for ec in novel_ec:
    result = system.retrieve_from_ec(ec)
    novel_mismatches.append(result["mismatch"])

print(f"  Familiar mismatch: mean={np.mean(familiar_mismatches):.4f}, "
      f"std={np.std(familiar_mismatches):.4f}")
print(f"  Novel mismatch:    mean={np.mean(novel_mismatches):.4f}, "
      f"std={np.std(novel_mismatches):.4f}")

# ROC-like discrimination
all_mismatches = familiar_mismatches + novel_mismatches
all_labels = [0] * len(familiar_mismatches) + [1] * len(novel_mismatches)

# Sort by mismatch and compute AUC
pairs = sorted(zip(all_mismatches, all_labels))
n_pos = sum(all_labels)
n_neg = len(all_labels) - n_pos
auc = 0.0
tp = 0
for mismatch, label in pairs:
    if label == 1:
        tp += 1
    else:
        auc += tp
auc = auc / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.5
# We want novel=high mismatch, so if auc < 0.5 we flip
if auc < 0.5:
    auc = 1 - auc

print(f"  Novelty detection AUC: {auc:.4f}")

# Separation at various thresholds
fam_arr = np.array(familiar_mismatches)
nov_arr = np.array(novel_mismatches)
d_prime = (np.mean(nov_arr) - np.mean(fam_arr)) / (
    np.sqrt(0.5 * (np.var(fam_arr) + np.var(nov_arr))) + 1e-10
)
print(f"  d-prime: {d_prime:.4f}")


# =============================================================================
# 7. EXPERIMENT 3: CA1 Decoding Quality
#    Does CA1 output for a stored pattern match the EC-driven target?
#    Compare decoding quality at different points in the learning sequence.
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 3: CA1 Decoding Quality Over Learning")
print("=" * 70)

# Re-encode with checkpoints
checkpoint_episodes = [10, 25, 50, 100, 150, 200]
decoding_results = {}

for n_train in checkpoint_episodes:
    sys_ck = HippocampalSystem(d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                                dg_params=dg_params, ca1_params=ca1_params, ca3_retrieval_iterations=ca3_retrieval_iters)

    # Only encode first n_train patterns
    ec_subset = ec_patterns[:n_train]
    sys_ck.encode_batch(ec_subset, train_ca1=True)

    # Test: re-present each encoded pattern, measure how well
    # CA1's Schaffer-driven output matches the EC-driven target
    sims = []
    for ec in ec_subset:
        result = sys_ck.retrieve_from_ec(ec)
        # Target: what EC drives in CA1 through the fixed TA pathway
        target = sys_ck.ca1.get_ec_target(ec)
        # Output: what CA3 drives in CA1 through the learned Schaffer pathway
        output = result["ca1_output"]
        sims.append(cosine_similarity(output, target))

    decoding_results[n_train] = {
        "mean_sim": float(np.mean(sims)),
        "std_sim": float(np.std(sims)),
        "min_sim": float(np.min(sims)),
    }

    print(f"  After {n_train:3d} episodes: "
          f"Schaffer-vs-TA similarity = {np.mean(sims):.4f} "
          f"(+/- {np.std(sims):.4f}), min={np.min(sims):.4f}")


# =============================================================================
# 8. EXPERIMENT 4: Retrieval from Similar (but New) Cues
#    Generate EC inputs similar to stored ones. Does CA1 output
#    match the target for the STORED pattern?
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 4: Retrieval from Similar Cues")
print("=" * 70)

# Use the N=200 system from Experiment 1
similarity_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
n_tests_per_level = 50

retrieval_results = {}

for target_sim in similarity_levels:
    correct_ca1 = 0
    correct_ca3 = 0
    ca1_to_target_sims = []
    ca3_to_target_sims = []

    for _ in range(n_tests_per_level):
        # Pick a random stored pattern
        idx = np.random.randint(N_store)
        stored_ec = ec_patterns[idx]

        # Generate a similar EC input
        noise = np.random.randn(d_ec)
        noise = noise - np.dot(noise, stored_ec) * stored_ec
        noise = noise / (np.linalg.norm(noise) + 1e-10)
        query_ec = target_sim * stored_ec + np.sqrt(max(0, 1 - target_sim**2)) * noise
        query_ec = query_ec / (np.linalg.norm(query_ec) + 1e-10)

        # Full retrieval
        result = system.retrieve_from_ec(query_ec)

        # CA1 output: compare to EC-driven target for the STORED pattern
        stored_target = system.ca1.get_ec_target(stored_ec)
        ca1_sim = cosine_similarity(result["ca1_output"], stored_target)
        ca1_to_target_sims.append(ca1_sim)

        # Find best match among all stored targets
        best_ca1_sim = -1
        best_ca1_idx = -1
        for j in range(N_store):
            s = cosine_similarity(result["ca1_output"],
                                  system.ca1.get_ec_target(ec_patterns[j]))
            if s > best_ca1_sim:
                best_ca1_sim = s
                best_ca1_idx = j
        if best_ca1_idx == idx:
            correct_ca1 += 1

        # Raw CA3 output: compare to stored DG patterns
        best_ca3_sim = -1
        best_ca3_idx = -1
        stored_dg = system.ca3.stored_patterns
        for j in range(min(N_store, len(stored_dg))):
            s = cosine_similarity(result["ca3_output"], stored_dg[j])
            if s > best_ca3_sim:
                best_ca3_sim = s
                best_ca3_idx = j
        if best_ca3_idx == idx:
            correct_ca3 += 1
        ca3_to_target_sims.append(best_ca3_sim)

    retrieval_results[target_sim] = {
        "ca1_accuracy": correct_ca1 / n_tests_per_level,
        "ca3_accuracy": correct_ca3 / n_tests_per_level,
        "ca1_mean_sim": float(np.mean(ca1_to_target_sims)),
        "ca3_mean_sim": float(np.mean(ca3_to_target_sims)),
    }

    print(f"  EC sim={target_sim:.2f}: "
          f"CA1 acc={correct_ca1/n_tests_per_level:.2f}, "
          f"CA3 acc={correct_ca3/n_tests_per_level:.2f}, "
          f"CA1 sim={np.mean(ca1_to_target_sims):.4f}")


# =============================================================================
# 9. EXPERIMENT 5: Learning Curve with Repeated Exposures
#    Present the same set of patterns multiple times. Does CA1 improve?
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 5: Multi-Pass Learning")
print("=" * 70)

N_small = 50
n_passes = 10
sys_mp = HippocampalSystem(d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                            dg_params=dg_params, ca1_params=ca1_params, ca3_retrieval_iterations=ca3_retrieval_iters)

ec_small = generate_ec_patterns(N_small, d_ec)

# Store in DG/CA3 once (online, one at a time)
dg_pats_small = [sys_mp.dg.forward(ec) for ec in ec_small]
for dg_pat in dg_pats_small:
    sys_mp.ca3.store_online(dg_pat)
sys_mp.ec_store = [ec.copy() for ec in ec_small]

multi_pass_results = []

for pass_num in range(n_passes):
    # Present all patterns to CA1 in random order
    order = np.random.permutation(N_small)
    pass_mismatches = []

    for idx in order:
        mismatch, info = sys_mp.ca1.encode(dg_pats_small[idx], ec_small[idx])
        pass_mismatches.append(mismatch)

    # Evaluate decoding quality after this pass
    sims = []
    for ec in ec_small:
        result = sys_mp.retrieve_from_ec(ec)
        target = sys_mp.ca1.get_ec_target(ec)
        sims.append(cosine_similarity(result["ca1_output"], target))

    pass_result = {
        "pass": pass_num,
        "mean_mismatch": float(np.mean(pass_mismatches)),
        "decoding_sim": float(np.mean(sims)),
        "decoding_min": float(np.min(sims)),
    }
    multi_pass_results.append(pass_result)

    print(f"  Pass {pass_num+1:2d}/{n_passes}: "
          f"mismatch={np.mean(pass_mismatches):.4f}, "
          f"decoding sim={np.mean(sims):.4f} (min={np.min(sims):.4f})")


# =============================================================================
# 10. EXPERIMENT 6: Capacity Test
#     How does CA1 performance scale with number of stored patterns?
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 6: CA1 Capacity Scaling")
print("=" * 70)

capacity_tests = [10, 25, 50, 100, 200, 500]
capacity_results = {}

for N in capacity_tests:
    sys_cap = HippocampalSystem(d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                                 dg_params=dg_params, ca1_params=ca1_params, ca3_retrieval_iterations=ca3_retrieval_iters)
    ec_cap = generate_ec_patterns(N, d_ec)
    sys_cap.encode_batch(ec_cap, train_ca1=True)

    # Decoding quality
    sims = []
    for ec in ec_cap:
        result = sys_cap.retrieve_from_ec(ec)
        target = sys_cap.ca1.get_ec_target(ec)
        sims.append(cosine_similarity(result["ca1_output"], target))

    # Novelty detection
    novel_ec_cap = generate_ec_patterns(N, d_ec)
    fam_mm = [sys_cap.retrieve_from_ec(ec)["mismatch"]
              for ec in ec_cap]
    nov_mm = [sys_cap.retrieve_from_ec(ec)["mismatch"]
              for ec in novel_ec_cap]

    d_pr = (np.mean(nov_mm) - np.mean(fam_mm)) / (
        np.sqrt(0.5 * (np.var(fam_mm) + np.var(nov_mm))) + 1e-10)

    capacity_results[N] = {
        "decoding_sim": float(np.mean(sims)),
        "decoding_std": float(np.std(sims)),
        "d_prime": float(d_pr),
        "familiar_mismatch": float(np.mean(fam_mm)),
        "novel_mismatch": float(np.mean(nov_mm)),
    }

    print(f"  N={N:4d}: decoding={np.mean(sims):.4f}, "
          f"d'={d_pr:.3f}, "
          f"fam_mm={np.mean(fam_mm):.2f}, nov_mm={np.mean(nov_mm):.2f}")


# =============================================================================
# 11. SAVE RESULTS
# =============================================================================

results = {
    "params": {
        "d_ec": d_ec, "D_dg": D_dg, "N_ca3": N_ca3, "N_ca1": N_ca1,
        "k_ca3": k_ca3, "ca3_retrieval_iterations": ca3_retrieval_iters,
        "ca3_storage": "online_hebbian",
        "dg_params": dg_params, "ca1_params": ca1_params,
    },
    "learning_curve": {
        "mismatches": mismatches,
        "plateau_fractions": plateau_fracs,
        "w_sc_norms": w_norms,
    },
    "novelty_detection": {
        "familiar_mean": float(np.mean(familiar_mismatches)),
        "familiar_std": float(np.std(familiar_mismatches)),
        "novel_mean": float(np.mean(novel_mismatches)),
        "novel_std": float(np.std(novel_mismatches)),
        "auc": auc,
        "d_prime": float(d_prime),
    },
    "decoding_over_learning": {str(k): v for k, v in decoding_results.items()},
    "similar_cue_retrieval": {str(k): v for k, v in retrieval_results.items()},
    "multi_pass": multi_pass_results,
    "capacity": {str(k): v for k, v in capacity_results.items()},
}

with open('hippocampal_model_v4_results.json', 'w') as f:
    json.dump(results, f, indent=2)


# =============================================================================
# 12. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(3, 3, figsize=(22, 18))
fig.suptitle("CA1 with BTSP Learning: Evaluation",
             fontsize=14, fontweight='bold')

# --- Panel 1: Learning Curve (mismatch over episodes) ---
ax = axes[0, 0]
# Smooth with moving average
window = 10
if len(mismatches) >= window:
    smoothed = np.convolve(mismatches, np.ones(window)/window, mode='valid')
    ax.plot(range(len(smoothed)), smoothed, 'b-', linewidth=1.5, label='Smoothed')
ax.plot(mismatches, 'b-', alpha=0.2, linewidth=0.5, label='Raw')
ax.set_xlabel('Episode')
ax.set_ylabel('Mismatch magnitude')
ax.set_title(f'CA1 Learning Curve (N={N_store})')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Panel 2: Novelty Detection ---
ax = axes[0, 1]
bins = np.linspace(
    min(min(familiar_mismatches), min(novel_mismatches)),
    max(max(familiar_mismatches), max(novel_mismatches)),
    40
)
ax.hist(familiar_mismatches, bins=bins, alpha=0.6, color='blue',
        label='Familiar', density=True)
ax.hist(novel_mismatches, bins=bins, alpha=0.6, color='red',
        label='Novel', density=True)
ax.set_xlabel('Mismatch magnitude')
ax.set_ylabel('Density')
ax.set_title(f"Novelty Detection (AUC={auc:.3f}, d'={d_prime:.2f})")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Panel 3: Decoding Quality Over Learning ---
ax = axes[0, 2]
ck_ns = sorted(decoding_results.keys())
ck_sims = [decoding_results[n]["mean_sim"] for n in ck_ns]
ck_stds = [decoding_results[n]["std_sim"] for n in ck_ns]
ax.errorbar(ck_ns, ck_sims, yerr=ck_stds, fmt='go-', markersize=6,
            capsize=4, linewidth=2)
ax.set_xlabel('Number of training episodes')
ax.set_ylabel('Schaffer-vs-TA cosine similarity')
ax.set_title('CA1 Decoding Quality Over Learning')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.05)

# --- Panel 4: Similar Cue Retrieval ---
ax = axes[1, 0]
sim_levels = sorted(retrieval_results.keys())
ca1_accs = [retrieval_results[s]["ca1_accuracy"] for s in sim_levels]
ca3_accs = [retrieval_results[s]["ca3_accuracy"] for s in sim_levels]
ax.plot(sim_levels, ca1_accs, 'go-', markersize=6, linewidth=2, label='CA1')
ax.plot(sim_levels, ca3_accs, 'r^-', markersize=6, linewidth=2, label='Raw CA3')
ax.set_xlabel('Query-to-stored EC similarity')
ax.set_ylabel('Retrieval accuracy')
ax.set_title('Retrieval from Similar Cues')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)

# --- Panel 5: Multi-Pass Learning ---
ax = axes[1, 1]
pass_nums = [r["pass"]+1 for r in multi_pass_results]
pass_mm = [r["mean_mismatch"] for r in multi_pass_results]
pass_dec = [r["decoding_sim"] for r in multi_pass_results]
ax2 = ax.twinx()
l1 = ax.plot(pass_nums, pass_mm, 'b-o', markersize=5, label='Mismatch')
l2 = ax2.plot(pass_nums, pass_dec, 'g-s', markersize=5, label='Decoding sim')
ax.set_xlabel('Training pass')
ax.set_ylabel('Mean mismatch', color='blue')
ax2.set_ylabel('Decoding similarity', color='green')
ax.set_title(f'Multi-Pass Learning (N={N_small})')
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')
ax.grid(True, alpha=0.3)

# --- Panel 6: Capacity Scaling ---
ax = axes[1, 2]
cap_ns = sorted(capacity_results.keys())
cap_dec = [capacity_results[n]["decoding_sim"] for n in cap_ns]
cap_dp = [capacity_results[n]["d_prime"] for n in cap_ns]
ax.plot(cap_ns, cap_dec, 'go-', markersize=6, linewidth=2, label='Decoding sim')
ax2 = ax.twinx()
ax2.plot(cap_ns, cap_dp, 'b^-', markersize=6, linewidth=2, label="d-prime")
ax.set_xlabel('Number of stored patterns')
ax.set_ylabel('Decoding similarity', color='green')
ax2.set_ylabel("d-prime (novelty detection)", color='blue')
ax.set_title('CA1 Capacity Scaling')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, loc='center right')
ax.grid(True, alpha=0.3)

# --- Panel 7: W_sc weight norm over learning ---
ax = axes[2, 0]
ax.plot(w_norms, 'b-', linewidth=1.5)
ax.set_xlabel('Episode')
ax.set_ylabel('||W_sc|| (Frobenius norm)')
ax.set_title('Schaffer Weight Growth')
ax.grid(True, alpha=0.3)

# --- Panel 8: CA1 output similarity to target for similar cues ---
ax = axes[2, 1]
ca1_sims_plot = [retrieval_results[s]["ca1_mean_sim"] for s in sim_levels]
ca3_sims_plot = [retrieval_results[s]["ca3_mean_sim"] for s in sim_levels]
ax.plot(sim_levels, ca1_sims_plot, 'go-', markersize=6, linewidth=2,
        label='CA1 to stored target')
ax.plot(sim_levels, ca3_sims_plot, 'r^-', markersize=6, linewidth=2,
        label='CA3 to stored DG pattern')
ax.set_xlabel('Query-to-stored EC similarity')
ax.set_ylabel('Cosine similarity to target')
ax.set_title('Output Quality for Similar Cues')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Panel 9: Plateau fraction during learning ---
ax = axes[2, 2]
if len(plateau_fracs) >= window:
    pf_smooth = np.convolve(plateau_fracs, np.ones(window)/window, mode='valid')
    ax.plot(range(len(pf_smooth)), pf_smooth, 'purple', linewidth=1.5)
ax.plot(plateau_fracs, color='purple', alpha=0.2, linewidth=0.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Fraction of CA1 neurons with plateau')
ax.set_title('Plateau Potential Activation')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hippocampal_model_v4_results.png', dpi=150, bbox_inches='tight')

print(f"\nResults saved to ca1_results.json")
print(f"Plot saved to ca1_results.png")