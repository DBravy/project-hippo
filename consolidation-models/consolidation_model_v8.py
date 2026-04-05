"""
Hippocampal-Cortical Consolidation v8: Entorhinal Cortex as Input/Output Interface
====================================================================================

Core question: Does hippocampal replay help cortex organize its
superposition better than learning from raw experience?

Architecture (v8 -- EC as proper input/output stage):

  Cortical input (d_ec)
        |
  EC Superficial (II/III) ---- passthrough for now
        |
        +---> DG(1000)  [perforant path, via L2]
        +---> CA3(1000)  [direct EC-CA3, via L2]
        +---> CA1(500)   [temporoammonic, via L3]
        +---> Sub(250)   [temporoammonic, via L3]
        |
  DG -> CA3 -> CA1 -> Sub  [trisynaptic + subicular de-sparsification]
        |         |      |
        |    CA1 -+------+--> EC Deep (V)  <--- Cortical input (teaching)
        |                         |
        |                    EC output (d_ec) ---> Cortex
        |                         |
        +---- deep-to-superficial feedback (for replay re-entry)

Changes from v7:
  - Subiculum simplified: removed burst nonlinearity. Now models the
    regular-spiking, EC-projecting population as a linear de-sparsifier
    (CA1 -> ReLU). Burst-firing cells project to subcortical targets
    (fornix, mammillary bodies) not modeled here.

  - EntorhinalCortex class: two compartments:
    * Superficial (II/III): input interface. Passthrough for now, but
      architecturally separate so downstream modules (DG, CA1) read
      from this output via their own weight matrices.
    * Deep (V): output interface. Receives BOTH cortical input (teaching
      signal, proximal to reality) AND hippocampal output (CA1 sparse +
      Sub dense, distal/internal). Error-driven Hebbian learning trains
      hippocampal->deep weights so that during replay (no cortical input),
      hippocampal activity alone produces output matching cortical input.

  - This implements the dual-input convergence motif uniformly:
    CA1:  TA(proximal) + SC(distal)  -> SC learns to match TA
    Sub:  EC(proximal) + CA1(distal) -> CA1 learns to match EC
    EC V: cortex(proximal) + hipp(distal) -> hipp learns to match cortex
    Each stage trains itself to simulate its reality-adjacent input from
    its internally-generated input. No backpropagation anywhere.

  - HippocampalSystem is now a self-contained module: takes d_ec vectors
    in, produces d_ec vectors out. Any upstream cortical architecture
    can plug in on top.
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
# 2. DENTATE GYRUS (unchanged)
# =============================================================================

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


# =============================================================================
# 3. CA3 (unchanged)
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


# =============================================================================
# 4. CA1 (unchanged)
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
        self.sigma_inh = sigma_inh
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.E_inh = E_inh

        self.W_ta = make_feedforward_weights(N_ca1, d_ec, connectivity_prob)
        mask = (np.random.rand(N_ca1, N_ca3) < connectivity_prob).astype(float)
        self.W_sc = np.random.randn(N_ca1, N_ca3) * 0.01 * mask
        self.connectivity_mask = mask.copy()
        self.W_inh = build_ring_inhibition(
            N_ca1, sigma_inh, connection_prob=inh_connection_prob)
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
        return h_ta, h_sc, gate, h_ta_raw, h_sc_raw

    def encode(self, x_ca3, x_ec):
        h_ta, h_sc, gate, h_ta_raw, h_sc_raw = self.compute_activations(x_ca3, x_ec)
        error = h_ta - h_sc
        gated_error = gate * error
        h_ta_norm = np.linalg.norm(h_ta) + 1e-10
        mismatch = float(np.linalg.norm(gated_error) / h_ta_norm)
        delta_W = self.lr * np.outer(gated_error, x_ca3)
        self.W_sc += delta_W
        ca3_inactive = (x_ca3 <= self.ltd_ca3_threshold).astype(float)
        ltd_matrix = self.ltd_rate * np.outer(gate, ca3_inactive)
        self.W_sc *= (1.0 - ltd_matrix)
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
                (h_sc_raw + g_inh * self.E_inh) / (1.0 + g_inh),
                0.0
            )
        return h_out, mismatch


# =============================================================================
# 5. SUBICULUM (MODIFIED: linear de-sparsifier, no burst nonlinearity)
# =============================================================================

class Subiculum:
    """
    Subiculum: linear de-sparsifier for the EC-facing output pathway.

    Models the regular-spiking subicular population that projects to
    entorhinal cortex. Burst-firing cells (which project to subcortical
    targets via fornix) are not modeled here.

    Receives two input streams:
      1. CA1 output (sparse, via plastic weights)
      2. EC layer III (dense, via fixed TA pathway -- teaching signal)

    During encoding: both streams active. Error-driven Hebbian learning
    trains CA1->Sub weights so CA1 input alone reproduces what EC would
    have driven. Same dual-input convergence motif as CA1.

    During replay: only CA1 input. Learned weights produce a dense code
    from the sparse CA1 pattern, expanding the active fraction so that
    downstream (EC deep) can learn from a linearly decodable signal.

    No lateral inhibition, no burst nonlinearity. The de-sparsification
    comes from convergence: each Sub neuron pools across many CA1 inputs,
    so even sparse CA1 activation drives most Sub neurons above zero.
    """

    def __init__(self, N_sub, N_ca1, d_ec,
                 lr=0.05, ltd_rate=0.05, connectivity_prob=0.33):
        self.N_sub = N_sub
        self.N_ca1 = N_ca1
        self.d_ec = d_ec
        self.lr = lr
        self.ltd_rate = ltd_rate

        # CA1 -> Sub: plastic weights (learned during encoding)
        mask_ca1 = (np.random.rand(N_sub, N_ca1) < connectivity_prob).astype(float)
        self.W_ca1 = np.random.randn(N_sub, N_ca1) * 0.01 * mask_ca1
        self.mask_ca1 = mask_ca1.copy()

        # EC -> Sub: fixed TA pathway (teaching signal)
        # Independent weights from CA1's W_ta -- same source (EC L3)
        # but different synapses shaped by different local plasticity
        self.W_ec = make_feedforward_weights(N_sub, d_ec, connectivity_prob)

        self.n_episodes = 0

    def encode(self, ca1_output, ec_pattern):
        """
        During encoding: both CA1 and EC inputs present.
        EC input shapes the target; CA1 pathway learns to match it.
        """
        h_ca1 = self.W_ca1 @ ca1_output
        h_ec = self.W_ec @ ec_pattern

        # Combined activation (linear + ReLU, no burst)
        h_sub = np.maximum(h_ca1 + h_ec, 0)

        # --- Error-driven learning at CA1 -> Sub ---
        error = h_ec - h_ca1
        delta_W = self.lr * np.outer(error, ca1_output)
        self.W_ca1 += delta_W

        # --- Heterosynaptic LTD on inactive CA1 synapses ---
        ca1_inactive = (ca1_output <= 0).astype(float)
        sub_active = (h_sub > 0).astype(float)
        ltd_matrix = self.ltd_rate * np.outer(sub_active, ca1_inactive)
        self.W_ca1 *= (1.0 - ltd_matrix)
        self.W_ca1 *= self.mask_ca1

        # Row-norm clipping for stability
        row_norms = np.linalg.norm(self.W_ca1, axis=1, keepdims=True) + 1e-10
        max_norm = np.percentile(row_norms, 95)
        if max_norm > 1e-10:
            self.W_ca1 = np.where(
                row_norms > max_norm,
                self.W_ca1 * (max_norm / row_norms),
                self.W_ca1
            )

        self.n_episodes += 1
        return h_sub

    def replay(self, ca1_output):
        """
        During replay: CA1 input only. Learned weights produce dense code.
        """
        h_ca1 = self.W_ca1 @ ca1_output
        h_sub = np.maximum(h_ca1, 0)
        return h_sub


# =============================================================================
# 6. ENTORHINAL CORTEX (NEW)
# =============================================================================

class EntorhinalCortex:
    """
    Entorhinal Cortex with superficial (II/III) and deep (V) compartments.

    This is the hippocampal module's interface to the outside world.
    Input and output are both d_ec-dimensional vectors, so any upstream
    cortical architecture can plug in on top.

    == Superficial layers (II/III): Input Interface ==
    Receives cortical input and provides it to hippocampal stages:
      - Layer II stellate/fan cells -> DG, CA3 (perforant path)
      - Layer III pyramidal cells -> CA1, Sub (temporoammonic pathway)
    For now, superficial is a passthrough: the downstream modules (DG, CA1,
    Sub) each have their own input weight matrices that read from this.

    == Deep layer V: Output Interface ==
    Receives hippocampal output AND cortical input simultaneously:
      - CA1 sparse output (plastic weights) -- for selection/indexing
      - Subiculum dense output (plastic weights) -- for content/reconstruction
      - Cortical input (direct) -- teaching signal during encoding

    Error-driven Hebbian learning trains the hippocampal->deep weights
    so that the combined CA1+Sub signal approximates cortical input.
    During replay (no cortical input), the hippocampal signal alone
    produces output that matches what cortex would have provided.

    This implements the same dual-input convergence motif seen at CA1
    and subiculum, one level up in the hierarchy:
      proximal (cortex) + distal (hippocampus) -> distal learns to match

    == Layer IV (lamina dissecans) ==
    Not modeled as a computational element. Its anatomical role as a
    cell-sparse divider between superficial and deep EC is reflected
    in the design: the deep-to-superficial connection is an explicit,
    modulated projection rather than shared state.
    """

    def __init__(self, d_ec, N_ca1, N_sub,
                 lr=0.05, ltd_rate=0.03, connectivity_prob=0.33):
        self.d_ec = d_ec
        self.N_ca1 = N_ca1
        self.N_sub = N_sub
        self.lr = lr
        self.ltd_rate = ltd_rate

        # --- Deep layer V: hippocampal input weights (plastic) ---
        # CA1 -> deep: sparse signal for selection/indexing
        mask_ca1 = (np.random.rand(d_ec, N_ca1) < connectivity_prob).astype(float)
        self.W_ca1_to_deep = np.random.randn(d_ec, N_ca1) * 0.01 * mask_ca1
        self.mask_ca1 = mask_ca1.copy()

        # Sub -> deep: dense signal for content reconstruction
        mask_sub = (np.random.rand(d_ec, N_sub) < connectivity_prob).astype(float)
        self.W_sub_to_deep = np.random.randn(d_ec, N_sub) * 0.01 * mask_sub
        self.mask_sub = mask_sub.copy()

        # --- Deep -> Superficial feedback (for replay re-entry) ---
        # Modulated by system state: suppressed during encoding (high ACh),
        # active during replay/consolidation (low ACh).
        # Not used in the current experiments but architecturally present.
        self.W_deep_to_superficial = make_feedforward_weights(
            d_ec, d_ec, connectivity_prob)

        self.n_episodes = 0

    def get_superficial_output(self, cortical_input):
        """
        EC superficial: format cortical input for hippocampal consumption.
        Passthrough for now. DG, CA1, Sub each have their own weight
        matrices that read from this output.
        """
        return cortical_input.copy()

    def encode(self, cortical_input, ca1_output, sub_output):
        """
        During encoding: cortical input (teaching signal) and hippocampal
        output (CA1 + Sub) both arrive at deep layer V.

        The learning goal: make the combined hippocampal signal approximate
        the cortical input, so that during replay the hippocampal signal
        alone can substitute for cortex.

        Returns the deep layer activation for diagnostics.
        """
        # Hippocampal drives to deep layer V
        h_ca1 = self.W_ca1_to_deep @ ca1_output
        h_sub = self.W_sub_to_deep @ sub_output
        h_hippo = h_ca1 + h_sub

        # Cortical input is the teaching signal (proximal to reality)
        # In the biology, cortical afferents drive layer V directly.
        # Since both are d_ec, no projection needed.
        h_cortical = cortical_input

        # --- Error-driven learning ---
        # Goal: h_hippo should approximate h_cortical
        error = h_cortical - h_hippo

        # Update CA1 -> deep weights
        delta_ca1 = self.lr * np.outer(error, ca1_output)
        self.W_ca1_to_deep += delta_ca1

        # Update Sub -> deep weights
        delta_sub = self.lr * np.outer(error, sub_output)
        self.W_sub_to_deep += delta_sub

        # --- Heterosynaptic LTD ---
        # CA1 pathway: depress synapses from inactive CA1 neurons
        ca1_inactive = (ca1_output <= 0).astype(float)
        ltd_ca1 = self.ltd_rate * np.ones((self.d_ec, 1)) @ ca1_inactive.reshape(1, -1)
        self.W_ca1_to_deep *= (1.0 - ltd_ca1)
        self.W_ca1_to_deep *= self.mask_ca1

        # Sub pathway: depress synapses from inactive Sub neurons
        sub_inactive = (sub_output <= 0).astype(float)
        ltd_sub = self.ltd_rate * np.ones((self.d_ec, 1)) @ sub_inactive.reshape(1, -1)
        self.W_sub_to_deep *= (1.0 - ltd_sub)
        self.W_sub_to_deep *= self.mask_sub

        # Row-norm clipping for stability on both weight matrices
        for W, mask in [(self.W_ca1_to_deep, self.mask_ca1),
                        (self.W_sub_to_deep, self.mask_sub)]:
            row_norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
            max_norm = np.percentile(row_norms, 95)
            if max_norm > 1e-10:
                clip = row_norms > max_norm
                W_clipped = np.where(clip, W * (max_norm / row_norms), W)
                W[:] = W_clipped

        self.n_episodes += 1

        # Deep layer activation (both inputs present during encoding)
        h_deep = h_hippo + h_cortical
        return h_deep

    def replay(self, ca1_output, sub_output):
        """
        During replay: no cortical input. Hippocampal signals drive
        deep layer V. After learning, h_hippo approximates the cortical
        input that was present during encoding.

        Returns (ec_output, h_deep_internal):
          ec_output: d_ec vector, the replay signal for cortex
          h_deep_internal: ReLU'd deep activation (for feedback/diagnostics)
        """
        h_ca1 = self.W_ca1_to_deep @ ca1_output
        h_sub = self.W_sub_to_deep @ sub_output
        h_hippo = h_ca1 + h_sub

        # The replay output IS h_hippo (which has learned to approximate
        # cortical input). This is a signed signal matching the format
        # of cortical input vectors.
        ec_output = h_hippo

        # Internal deep layer activation (non-negative, for feedback
        # to superficial layers and for diagnostics)
        h_deep_internal = np.maximum(h_hippo, 0)

        return ec_output, h_deep_internal

    def get_superficial_from_deep(self, h_deep_internal):
        """
        Deep -> superficial feedback for replay re-entry.
        Used when replay signal needs to re-enter the hippocampal loop
        (e.g., for multi-pass retrieval or iterative refinement).
        """
        return self.W_deep_to_superficial @ h_deep_internal

    def fit_regression(self, ca1_outputs, sub_outputs, ec_patterns,
                       ridge_lambda=0.01):
        """
        Diagnostic: fit hippocampal->deep weights by ridge regression.
        Separates architecture limitations from learning rule limitations.

        Solves: min ||ec_patterns - (W_ca1 @ ca1 + W_sub @ sub)||^2 + lambda*||W||^2
        """
        ca1_mat = np.array(ca1_outputs)   # (N, N_ca1)
        sub_mat = np.array(sub_outputs)   # (N, N_sub)
        ec_mat = np.array(ec_patterns)    # (N, d_ec)

        # Combined input matrix
        combined = np.hstack([ca1_mat, sub_mat])  # (N, N_ca1+N_sub)

        # Ridge regression
        C = combined.T @ combined + ridge_lambda * np.eye(combined.shape[1])
        C_inv = np.linalg.inv(C)
        W_combined = ec_mat.T @ combined @ C_inv  # (d_ec, N_ca1+N_sub)

        # Split and apply connectivity masks
        self.W_ca1_to_deep = W_combined[:, :self.N_ca1] * self.mask_ca1
        self.W_sub_to_deep = W_combined[:, self.N_ca1:] * self.mask_sub

        # Diagnostics: measure reconstruction quality
        recon = combined @ W_combined.T  # (N, d_ec) -- before masking
        sims = [cosine_similarity(ec_mat[i], recon[i]) for i in range(len(ec_mat))]
        return {
            "mean_cosine_sim": float(np.mean(sims)),
            "std_cosine_sim": float(np.std(sims)),
            "n_patterns": len(ec_mat),
        }


# =============================================================================
# 7. HIPPOCAMPAL SYSTEM (MODIFIED: integrates EC)
# =============================================================================

class HippocampalSystem:
    """
    Self-contained hippocampal module.

    Interface: takes d_ec vectors in, produces d_ec vectors out.
    Internally: EC(superficial) -> DG -> CA3 -> CA1 -> Sub -> EC(deep)
    """
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_params=None, N_sub=250, ca3_retrieval_iterations=5):
        self.d_ec = d_ec
        self.ca3_retrieval_iterations = ca3_retrieval_iterations

        # Build components
        self.ec = EntorhinalCortex(d_ec, N_ca1, N_sub, **(ec_params or {}))
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}))
        self.ca3 = CA3(N_ca3, k_ca3)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}))
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}))

        self.ec_store = []
        self.dg_store = []

    def encode_batch(self, ec_patterns, train_all=True):
        """
        Encode a batch of cortical input patterns through the full circuit.
        Trains CA1, Sub, and EC deep layer weights.
        """
        self.ec_store = [ec.copy() for ec in ec_patterns]
        dg_patterns = []

        for i, ec in enumerate(ec_patterns):
            # EC superficial: passthrough
            ec_sup = self.ec.get_superficial_output(ec)

            # Trisynaptic loop: EC -> DG -> CA3
            dg_out = self.dg.forward(ec_sup)
            dg_patterns.append(dg_out)
            self.ca3.store_online(dg_out)

            if train_all:
                # Train CA1: DG/CA3 pattern + EC teaching signal
                self.ca1.encode(dg_out, ec_sup)

                # Get CA1 output for downstream training
                ca3_out = self.ca3.retrieve(
                    dg_out, n_iterations=self.ca3_retrieval_iterations)
                ca1_out, _ = self.ca1.retrieve(ca3_out, ec_sup)

                # Train Subiculum: CA1 sparse + EC teaching signal
                sub_out = self.sub.encode(ca1_out, ec_sup)

                # Train EC deep: CA1 + Sub + cortical teaching signal
                self.ec.encode(ec_sup, ca1_out, sub_out)

        self.dg_store = dg_patterns

    def replay_to_output(self, ec_query):
        """
        Full replay pipeline: cued recall through the entire circuit.
        EC query -> DG -> CA3(retrieve) -> CA1(retrieve) -> Sub -> EC deep

        Returns (ec_output, diagnostics_dict)
        """
        ec_sup = self.ec.get_superficial_output(ec_query)
        dg_out = self.dg.forward(ec_sup)
        ca3_out = self.ca3.retrieve(
            dg_out, n_iterations=self.ca3_retrieval_iterations)
        ca1_out, mismatch = self.ca1.retrieve(ca3_out, ec_sup)
        sub_out = self.sub.replay(ca1_out)
        ec_output, h_deep = self.ec.replay(ca1_out, sub_out)

        return ec_output, {
            "ca1_mismatch": mismatch,
            "ca1_active_frac": float(np.mean(ca1_out > 0)),
            "sub_active_frac": float(np.mean(sub_out > 0)),
            "deep_active_frac": float(np.mean(h_deep > 0)),
        }

    def generate_replay_batch(self):
        """Replay all stored patterns through the full pipeline."""
        replay_signals = []
        for ec in self.ec_store:
            ec_output, _ = self.replay_to_output(ec)
            replay_signals.append(ec_output)
        return replay_signals

    def replay_pinv(self, ec_query):
        """Oracle replay using pseudoinverse of CA1 TA weights (ceiling)."""
        ec_sup = self.ec.get_superficial_output(ec_query)
        dg_out = self.dg.forward(ec_sup)
        ca3_out = self.ca3.retrieve(
            dg_out, n_iterations=self.ca3_retrieval_iterations)
        ca1_out, _ = self.ca1.retrieve(ca3_out, ec_sup)
        W_ta_pinv = np.linalg.pinv(self.ca1.W_ta)
        return W_ta_pinv @ ca1_out

    def fit_ec_deep_regression(self):
        """
        Diagnostic: after encoding, collect (CA1, Sub, EC) triples using
        final weights, then fit EC deep weights by ridge regression.
        Creates self.ec_regress with optimal weights.
        """
        import copy

        ca1_outputs = []
        sub_outputs = []
        for ec in self.ec_store:
            dg_out = self.dg.forward(ec)
            ca3_out = self.ca3.retrieve(
                dg_out, n_iterations=self.ca3_retrieval_iterations)
            ca1_out, _ = self.ca1.retrieve(ca3_out, ec)
            sub_out = self.sub.replay(ca1_out)
            ca1_outputs.append(ca1_out)
            sub_outputs.append(sub_out)

        # Create fresh EC for regression
        self.ec_regress = EntorhinalCortex(
            self.d_ec, self.ec.N_ca1, self.ec.N_sub,
        )
        diag = self.ec_regress.fit_regression(
            ca1_outputs, sub_outputs, self.ec_store)
        return diag

    def replay_regression(self, ec_query):
        """Replay using regression-fit EC deep weights (architecture ceiling)."""
        ec_sup = self.ec.get_superficial_output(ec_query)
        dg_out = self.dg.forward(ec_sup)
        ca3_out = self.ca3.retrieve(
            dg_out, n_iterations=self.ca3_retrieval_iterations)
        ca1_out, _ = self.ca1.retrieve(ca3_out, ec_sup)
        sub_out = self.sub.replay(ca1_out)
        ec_output, _ = self.ec_regress.replay(ca1_out, sub_out)
        return ec_output

    def get_stage_outputs(self, ec_pattern):
        """Get representations at each stage for analysis."""
        ec_sup = self.ec.get_superficial_output(ec_pattern)
        dg_out = self.dg.forward(ec_sup)
        ca3_out = self.ca3.retrieve(
            dg_out, n_iterations=self.ca3_retrieval_iterations)
        ca1_out, _ = self.ca1.retrieve(ca3_out, ec_sup)
        sub_out = self.sub.replay(ca1_out)
        ec_replay, h_deep = self.ec.replay(ca1_out, sub_out)
        return {
            "ec_sup": ec_sup,
            "dg": dg_out,
            "ca3": ca3_out,
            "ca1": ca1_out,
            "sub": sub_out,
            "ec_deep": h_deep,
            "ec_output": ec_replay,
        }


# =============================================================================
# 8. WORLD MODEL (unchanged)
# =============================================================================

class SparseFeatureWorld:
    def __init__(self, N_features, d_ec, sparsity_base=0.1,
                 sparsity_decay=0.99, importance_decay=0.95):
        self.N_features = N_features
        self.d_ec = d_ec
        self.sparsities = np.array([
            sparsity_base * (sparsity_decay ** i) for i in range(N_features)
        ])
        self.sparsities = np.clip(self.sparsities, 0.005, 0.5)
        self.importances = np.array([
            importance_decay ** i for i in range(N_features)
        ])
        raw = np.random.randn(d_ec, N_features)
        col_norms = np.linalg.norm(raw, axis=0, keepdims=True) + 1e-10
        self.W_ec = raw / col_norms
        self.W_ec_pinv = np.linalg.pinv(self.W_ec)
        self.feature_directions_ec = self.W_ec.copy()

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

    def decode_features_oracle(self, ec):
        return np.maximum(self.W_ec_pinv @ ec, 0)


# =============================================================================
# 9. CORTICAL AUTOENCODER (unchanged)
# =============================================================================

class CorticalAutoencoder:
    def __init__(self, d_ec, d_cortex, lr=0.005, weight_decay=1e-5):
        self.d_ec = d_ec
        self.d_cortex = d_cortex
        self.lr = lr
        self.weight_decay = weight_decay
        self.W = np.random.randn(d_cortex, d_ec) * np.sqrt(2.0 / (d_cortex + d_ec))
        self.b = np.zeros(d_ec)
        self.loss_history = []

    def encode(self, ec):
        return self.W @ ec

    def decode(self, h):
        return np.maximum(self.W.T @ h + self.b, 0)

    def forward(self, ec):
        h = self.encode(ec)
        ec_hat = self.decode(h)
        return h, ec_hat

    def train_step(self, ec):
        h = self.W @ ec
        z = self.W.T @ h + self.b
        a = np.maximum(z, 0)
        error = a - ec
        loss = 0.5 * np.sum(error ** 2)
        relu_mask = (z > 0).astype(float)
        d_pre = error * relu_mask
        dL_dh = self.W @ d_pre
        grad_W = np.outer(dL_dh, ec) + np.outer(h, d_pre)
        grad_b = d_pre
        grad_W += self.weight_decay * self.W
        grad_norm = np.sqrt(np.sum(grad_W ** 2) + np.sum(grad_b ** 2))
        if grad_norm > 1.0:
            scale = 1.0 / grad_norm
            grad_W *= scale
            grad_b *= scale
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return float(loss)

    def train_epoch(self, ec_patterns, shuffle=True):
        indices = np.arange(len(ec_patterns))
        if shuffle:
            np.random.shuffle(indices)
        total_loss = 0.0
        for i in indices:
            total_loss += self.train_step(ec_patterns[i])
        return total_loss / len(ec_patterns)

    def copy(self):
        c = CorticalAutoencoder(self.d_ec, self.d_cortex, self.lr, self.weight_decay)
        c.W = self.W.copy()
        c.b = self.b.copy()
        return c


# =============================================================================
# 10. ANALYSIS TOOLS
# =============================================================================

def compute_feature_probes(cortex, world, ec_patterns, features_batch):
    n = len(ec_patterns)
    h_all = np.array([cortex.encode(ec) for ec in ec_patterns])
    N_features = world.N_features
    probe_weights = np.zeros((N_features, cortex.d_cortex))
    probe_r2 = np.zeros(N_features)
    lam = 0.01
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
    norms = np.linalg.norm(probe_weights, axis=1, keepdims=True) + 1e-10
    directions = probe_weights / norms
    cos_sim = directions @ directions.T
    mask = ~np.eye(len(directions), dtype=bool)
    interference = np.abs(cos_sim[mask])
    clean_weights = np.nan_to_num(probe_weights, nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vt = np.linalg.svd(clean_weights, full_matrices=False)
    S2 = S ** 2
    eff_dim = (np.sum(S2) ** 2) / (np.sum(S2 ** 2) + 1e-10)
    superposition_ratio = len(probe_weights) / max(eff_dim, 1e-10)
    return {
        "cos_sim_matrix": cos_sim.tolist(),
        "mean_interference": float(np.mean(interference)),
        "std_interference": float(np.std(interference)),
        "max_interference": float(np.max(interference)),
        "effective_dimensionality": float(eff_dim),
        "superposition_ratio": float(superposition_ratio),
        "svd_spectrum": S.tolist(),
    }


def measure_interference_vs_cooccurrence(features_batch, cos_sim_matrix):
    if isinstance(cos_sim_matrix, list):
        cos_sim_matrix = np.array(cos_sim_matrix)
    N = features_batch.shape[1]
    active = (features_batch > 0).astype(float)
    cooccurrence = (active.T @ active) / len(features_batch)
    pairs_cooc, pairs_interference = [], []
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
    }


def measure_reconstruction_quality(cortex, ec_patterns, features_batch, world):
    recon_errors, feature_recovery = [], []
    for i, ec in enumerate(ec_patterns):
        h, ec_hat = cortex.forward(ec)
        recon_errors.append(float(np.sum((ec_hat - ec) ** 2)))
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
    }


def measure_stage_representations(reprs, features_probe, world, dim, name):
    """Measure representation quality at a single hippocampal stage."""
    # Participation ratio of covariance
    cov = np.cov(reprs.T)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
    total_var = np.sum(eigvals) + 1e-10
    pr = total_var ** 2 / (np.sum(eigvals ** 2) + 1e-10)

    # Sparsity
    active_frac = np.mean(reprs > 0)

    # Pairwise similarity (sample)
    sample_idx = np.random.choice(len(reprs), min(200, len(reprs)), replace=False)
    pair_sims = []
    for ii in range(len(sample_idx)):
        for jj in range(ii + 1, min(ii + 20, len(sample_idx))):
            pair_sims.append(cosine_similarity(
                reprs[sample_idx[ii]], reprs[sample_idx[jj]]))

    # Feature decodability (first 100 features)
    lam = 0.01
    n_test = min(100, world.N_features)
    decode_r2s = []
    HTH_inv = np.linalg.inv(reprs.T @ reprs + lam * np.eye(reprs.shape[1]))
    for fi in range(n_test):
        y = features_probe[:len(reprs), fi]
        w = HTH_inv @ (reprs.T @ y)
        y_hat = reprs @ w
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
        decode_r2s.append(max(0.0, 1.0 - ss_res / ss_tot))

    return {
        "n_dims": dim,
        "participation_ratio": float(pr),
        "pr_normalized": float(pr / dim) if dim > 0 else 0.0,
        "active_fraction": float(active_frac),
        "mean_pairwise_sim": float(np.mean(pair_sims)) if pair_sims else 0.0,
        "feature_decode_r2_mean": float(np.mean(decode_r2s)),
        "feature_decode_r2_std": float(np.std(decode_r2s)),
    }


# =============================================================================
# 11. PARAMETERS
# =============================================================================

N_features = 200
d_ec = 100
sparsity_base = 0.1
sparsity_decay = 0.99

D_dg = 1000
N_ca3 = 1000
N_ca1 = 500
N_sub = 250
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
    "lr": 0.05, "ltd_rate": 0.05, "connectivity_prob": 0.33,
}

ec_params = {
    "lr": 0.05, "ltd_rate": 0.03, "connectivity_prob": 0.33,
}

d_cortex = 50
cortex_lr = 0.005
cortex_weight_decay = 1e-5
n_train_epochs = 100
n_replay_rounds = 5

N_experience = 500
N_probe = 2000

print("=" * 70)
print("HIPPOCAMPAL-CORTICAL CONSOLIDATION v8: ENTORHINAL CORTEX EXPERIMENT")
print("=" * 70)
print(f"\nWorld: {N_features} features -> EC({d_ec}) -> Cortex({d_cortex})")
print(f"  Superposition required: {N_features} features in {d_cortex} dims")
print(f"  Sparsity range: {sparsity_base:.3f} to "
      f"{sparsity_base * sparsity_decay**(N_features-1):.3f}")
print(f"\nHippocampus: EC({d_ec}) -> DG({D_dg}) -> CA3({N_ca3}) -> CA1({N_ca1})")
print(f"  Sub({N_sub}) -> EC_deep({d_ec}) [output]")
print(f"  CA3: k={k_ca3}, retrieval iterations={ca3_retrieval_iters}")
print(f"\nCortex: d_cortex={d_cortex}, lr={cortex_lr}, epochs={n_train_epochs}")
print(f"Data: {N_experience} experiences, {N_probe} probe observations")


# =============================================================================
# 12. GENERATE WORLD AND OBSERVATIONS
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 1: Generating world and observations")
print("=" * 70)

world = SparseFeatureWorld(N_features, d_ec, sparsity_base=sparsity_base,
                           sparsity_decay=sparsity_decay)

features_exp, ec_exp = world.generate_batch(N_experience)
print(f"  Generated {N_experience} observations")
print(f"  Mean features active: {np.mean(np.sum(features_exp > 0, axis=1)):.1f}")
print(f"  EC pattern mean norm: {np.mean(np.linalg.norm(ec_exp, axis=1)):.3f}")

features_probe, ec_probe = world.generate_batch(N_probe)
print(f"  Generated {N_probe} probe observations")


# =============================================================================
# 13. HIPPOCAMPAL ENCODING
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 2: Hippocampal encoding (with EC deep learning)")
print("=" * 70)

hippocampus = HippocampalSystem(
    d_ec, D_dg, N_ca3, N_ca1, k_ca3,
    dg_params=dg_params, ca1_params=ca1_params,
    sub_params=sub_params, ec_params=ec_params,
    N_sub=N_sub, ca3_retrieval_iterations=ca3_retrieval_iters
)

hippocampus.encode_batch(ec_exp, train_all=True)
print(f"  Encoded {N_experience} patterns")

# --- Fit EC deep regression (architecture ceiling) ---
print(f"\n  Fitting EC deep regression diagnostic...")
regress_diag = hippocampus.fit_ec_deep_regression()
print(f"    Regression ceiling cosine sim: {regress_diag['mean_cosine_sim']:.4f}")


# =============================================================================
# 14. REPLAY QUALITY MEASUREMENT
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 3: Replay quality")
print("=" * 70)

# EC deep replay (online learned)
replay_ec = hippocampus.generate_replay_batch()

# Oracle ceiling (pseudoinverse)
replay_pinv = [hippocampus.replay_pinv(ec) for ec in hippocampus.ec_store]

# EC deep replay (regression fit)
replay_regress = [hippocampus.replay_regression(ec) for ec in hippocampus.ec_store]

# Measure cosine similarities
ec_sims = [cosine_similarity(ec_exp[i], replay_ec[i])
           for i in range(N_experience)]
pinv_sims = [cosine_similarity(ec_exp[i], replay_pinv[i])
             for i in range(N_experience)]
regress_sims = [cosine_similarity(ec_exp[i], replay_regress[i])
                for i in range(N_experience)]

print(f"  Replay quality (cosine sim to original):")
print(f"    EC deep (online):     {np.mean(ec_sims):.4f} +/- {np.std(ec_sims):.4f}")
print(f"    EC deep (regression): {np.mean(regress_sims):.4f} +/- {np.std(regress_sims):.4f}")
print(f"    Pseudoinverse (ceil): {np.mean(pinv_sims):.4f} +/- {np.std(pinv_sims):.4f}")

# Measure active fractions at each stage during replay
stage_fracs = {"ca1": [], "sub": [], "ec_deep": []}
for ec in ec_exp[:100]:
    _, diag = hippocampus.replay_to_output(ec)
    stage_fracs["ca1"].append(diag["ca1_active_frac"])
    stage_fracs["sub"].append(diag["sub_active_frac"])
    stage_fracs["ec_deep"].append(diag["deep_active_frac"])

print(f"\n  Active fractions during replay:")
print(f"    CA1:     {np.mean(stage_fracs['ca1']):.3f}")
print(f"    Sub:     {np.mean(stage_fracs['sub']):.3f}")
print(f"    EC deep: {np.mean(stage_fracs['ec_deep']):.3f}")


# =============================================================================
# 15. STAGE REPRESENTATION ANALYSIS
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 4: Representation quality by stage")
print("=" * 70)

n_analysis = 500
dg_reprs, ca3_reprs, ca1_reprs, sub_reprs, ec_deep_reprs = [], [], [], [], []

for ec in ec_probe[:n_analysis]:
    stages = hippocampus.get_stage_outputs(ec)
    dg_reprs.append(stages["dg"])
    ca3_reprs.append(stages["ca3"])
    ca1_reprs.append(stages["ca1"])
    sub_reprs.append(stages["sub"])
    ec_deep_reprs.append(stages["ec_deep"])

stage_data = {
    "EC": (np.array(ec_probe[:n_analysis]), d_ec),
    "DG": (np.array(dg_reprs), D_dg),
    "CA3": (np.array(ca3_reprs), N_ca3),
    "CA1": (np.array(ca1_reprs), N_ca1),
    "Sub": (np.array(sub_reprs), N_sub),
    "EC_deep": (np.array(ec_deep_reprs), d_ec),
}

stage_results = {}
for name, (reprs, dim) in stage_data.items():
    result = measure_stage_representations(
        reprs, features_probe, world, dim, name)
    stage_results[name] = result
    print(f"  {name:>8s} (d={dim:4d}): "
          f"PR={result['participation_ratio']:.1f} "
          f"({result['pr_normalized']:.2f} of d), "
          f"active={result['active_fraction']:.3f}, "
          f"pair_sim={result['mean_pairwise_sim']:.4f}, "
          f"decode_R2={result['feature_decode_r2_mean']:.4f}")


# =============================================================================
# 16. CORTICAL TRAINING
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 5: Cortical training")
print("=" * 70)

cortex_init = CorticalAutoencoder(d_ec, d_cortex, lr=cortex_lr,
                                  weight_decay=cortex_weight_decay)

conditions = {
    "raw_shuffled": {
        "description": "Raw EC patterns, shuffled (baseline)",
        "training_data": ec_exp,
    },
    "replay_ec_deep": {
        "description": "Hippocampal replay via EC deep (biologically grounded)",
        "training_data": np.array(replay_ec),
    },
    "replay_ec_regress": {
        "description": "Replay via regression-fit EC deep (architecture ceiling)",
        "training_data": np.array(replay_regress),
    },
    "replay_pinv": {
        "description": "Replay via pseudoinverse (mathematical ceiling)",
        "training_data": np.array(replay_pinv),
    },
}

condition_results = {}

for cond_name, cond_info in conditions.items():
    print(f"\n  --- Condition: {cond_name} ---")
    print(f"  {cond_info['description']}")

    cortex = cortex_init.copy()
    training_data = cond_info["training_data"]
    epoch_losses = []

    for epoch in range(n_train_epochs):
        epoch_loss = cortex.train_epoch(training_data, shuffle=True)
        epoch_losses.append(epoch_loss)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            recon_q = measure_reconstruction_quality(
                cortex, ec_probe[:200], features_probe[:200], world)
            print(f"    Epoch {epoch+1:3d}: loss={epoch_loss:.4f}, "
                  f"recon_err={recon_q['mean_recon_error']:.4f}, "
                  f"feat_recovery={recon_q['mean_feature_recovery']:.4f}")

    # Full analysis on probe set
    print(f"  Running full analysis...")
    probe_weights, probe_r2 = compute_feature_probes(
        cortex, world, ec_probe, features_probe)
    geom = measure_superposition_geometry(probe_weights)
    cooc = measure_interference_vs_cooccurrence(
        features_probe, np.array(geom["cos_sim_matrix"]))
    final_recon = measure_reconstruction_quality(
        cortex, ec_probe, features_probe, world)

    print(f"    Probe R^2: {np.mean(probe_r2):.4f}, "
          f"Eff dim: {geom['effective_dimensionality']:.1f}, "
          f"Interf: {geom['mean_interference']:.4f}, "
          f"Co-oc corr: {cooc['cooc_interference_spearman']:.4f}")

    condition_results[cond_name] = {
        "description": cond_info["description"],
        "epoch_losses": epoch_losses,
        "probe_r2_mean": float(np.mean(probe_r2)),
        "probe_r2_median": float(np.median(probe_r2)),
        "probe_r2_by_feature": probe_r2.tolist(),
        "superposition_geometry": {
            k: v for k, v in geom.items() if k != "cos_sim_matrix"
        },
        "cooccurrence_analysis": cooc,
        "reconstruction": final_recon,
    }


# =============================================================================
# 17. CROSS-CONDITION COMPARISON
# =============================================================================

print(f"\n{'=' * 70}")
print("PHASE 6: Cross-condition comparison")
print("=" * 70)

print(f"\n{'Condition':<22s} {'Probe R^2':>10s} {'Eff Dim':>8s} "
      f"{'Superpos':>9s} {'Interf':>8s} {'Co-oc corr':>10s}")
print("-" * 72)

for cond_name, res in condition_results.items():
    geom = res["superposition_geometry"]
    cooc = res["cooccurrence_analysis"]
    print(f"{cond_name:<22s} "
          f"{res['probe_r2_mean']:>10.4f} "
          f"{geom['effective_dimensionality']:>8.1f} "
          f"{geom['superposition_ratio']:>9.2f} "
          f"{geom['mean_interference']:>8.4f} "
          f"{cooc['cooc_interference_spearman']:>10.4f}")


# =============================================================================
# 18. EXPERIMENT 2: CAPACITY SCALING
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 2: Cortical capacity scaling")
print("=" * 70)

capacity_tests = [10, 25, 50, 75, 100]
capacity_results = {}

for d_c in capacity_tests:
    print(f"\n  d_cortex = {d_c} (ratio = {d_c/N_features:.2f})")
    cortex_cap = CorticalAutoencoder(d_ec, d_c, lr=cortex_lr,
                                     weight_decay=cortex_weight_decay)
    for epoch in range(n_train_epochs):
        cortex_cap.train_epoch(ec_exp, shuffle=True)

    pw, pr2 = compute_feature_probes(cortex_cap, world, ec_probe, features_probe)
    geom = measure_superposition_geometry(pw)
    recon = measure_reconstruction_quality(cortex_cap, ec_probe[:200],
                                           features_probe[:200], world)
    capacity_results[d_c] = {
        "probe_r2_mean": float(np.mean(pr2)),
        "effective_dimensionality": geom["effective_dimensionality"],
        "superposition_ratio": geom["superposition_ratio"],
        "mean_interference": geom["mean_interference"],
        "mean_recon_error": recon["mean_recon_error"],
    }
    print(f"    R^2: {np.mean(pr2):.4f}, Eff dim: {geom['effective_dimensionality']:.1f}")


# =============================================================================
# 19. EXPERIMENT 3: REPLAY QUANTITY
# =============================================================================

print(f"\n{'=' * 70}")
print("EXPERIMENT 3: Replay quantity effect")
print("=" * 70)

replay_quantities = [1, 2, 5, 10, 20]
replay_quantity_results = {}

for n_replays in replay_quantities:
    print(f"\n  Replay rounds per epoch: {n_replays}")
    cortex_rq = cortex_init.copy()
    replay_data = np.array(replay_ec)  # Use EC deep replay

    for epoch in range(n_train_epochs):
        for _ in range(n_replays):
            cortex_rq.train_epoch(replay_data, shuffle=True)

    pw, pr2 = compute_feature_probes(cortex_rq, world, ec_probe, features_probe)
    geom = measure_superposition_geometry(pw)
    cooc = measure_interference_vs_cooccurrence(
        features_probe, np.array(geom["cos_sim_matrix"]))
    replay_quantity_results[n_replays] = {
        "probe_r2_mean": float(np.mean(pr2)),
        "effective_dimensionality": geom["effective_dimensionality"],
        "superposition_ratio": geom["superposition_ratio"],
        "cooc_interference_spearman": cooc["cooc_interference_spearman"],
    }
    print(f"    R^2: {np.mean(pr2):.4f}, "
          f"Co-oc corr: {cooc['cooc_interference_spearman']:.4f}")


# =============================================================================
# 20. SAVE RESULTS
# =============================================================================

print(f"\n{'=' * 70}")
print("Saving results...")
print("=" * 70)

results = {
    "params": {
        "N_features": N_features, "d_ec": d_ec, "d_cortex": d_cortex,
        "D_dg": D_dg, "N_ca3": N_ca3, "N_ca1": N_ca1, "N_sub": N_sub,
        "k_ca3": k_ca3, "n_train_epochs": n_train_epochs,
        "N_experience": N_experience, "N_probe": N_probe,
        "ca1_params": ca1_params, "sub_params": sub_params,
        "ec_params": ec_params,
    },
    "replay_quality": {
        "ec_deep_mean_sim": float(np.mean(ec_sims)),
        "ec_deep_std_sim": float(np.std(ec_sims)),
        "ec_regress_mean_sim": float(np.mean(regress_sims)),
        "ec_regress_std_sim": float(np.std(regress_sims)),
        "pinv_mean_sim": float(np.mean(pinv_sims)),
        "pinv_std_sim": float(np.std(pinv_sims)),
    },
    "stage_active_fractions": {
        k: float(np.mean(v)) for k, v in stage_fracs.items()
    },
    "condition_comparison": condition_results,
    "capacity_scaling": {str(k): v for k, v in capacity_results.items()},
    "replay_quantity": {str(k): v for k, v in replay_quantity_results.items()},
    "stage_representations": stage_results,
}

with open("consolidation_model_v8_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved to consolidation_model_v8_results.json")


# =============================================================================
# 21. VISUALIZATION
# =============================================================================

print("\nGenerating plots...")

fig, axes = plt.subplots(3, 3, figsize=(22, 18))
fig.suptitle("Hippocampal-Cortical Consolidation v8: EC Deep Layer Experiment",
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

# --- Panel 2: Replay quality comparison ---
ax = axes[0, 1]
methods = ["EC deep\n(online)", "EC deep\n(regress)", "Pseudoinverse\n(ceiling)"]
means = [np.mean(ec_sims), np.mean(regress_sims), np.mean(pinv_sims)]
stds = [np.std(ec_sims), np.std(regress_sims), np.std(pinv_sims)]
bars = ax.bar(methods, means, yerr=stds, capsize=5,
              color=["steelblue", "coral", "forestgreen"], alpha=0.8)
ax.set_ylabel("Cosine similarity to original")
ax.set_title("Replay Reconstruction Quality")
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 3: Feature probe R^2 distributions ---
ax = axes[0, 2]
for cond_name, res in condition_results.items():
    r2s = res["probe_r2_by_feature"]
    ax.hist(r2s, bins=30, alpha=0.4, label=f"{cond_name} ({np.mean(r2s):.3f})")
ax.set_xlabel("Probe R^2")
ax.set_ylabel("Count")
ax.set_title("Feature Recovery Distribution")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# --- Panel 4: Stage representation quality ---
ax = axes[1, 0]
stages = list(stage_results.keys())
stage_decode = [stage_results[s]["feature_decode_r2_mean"] for s in stages]
stage_active = [stage_results[s]["active_fraction"] for s in stages]
x_pos = np.arange(len(stages))
width = 0.35
ax.bar(x_pos - width/2, stage_decode, width, label="Feature decode R^2", color="coral")
ax.bar(x_pos + width/2, stage_active, width, label="Active fraction", color="steelblue")
ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("Value")
ax.set_title("Representation Quality by Stage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 5: Sparsity and similarity by stage ---
ax = axes[1, 1]
stage_pair_sim = [stage_results[s]["mean_pairwise_sim"] for s in stages]
stage_pr_norm = [stage_results[s]["pr_normalized"] for s in stages]
ax.bar(x_pos - width/2, stage_pair_sim, width, label="Mean pairwise sim", color="steelblue")
ax.bar(x_pos + width/2, stage_pr_norm, width, label="PR / d (spread)", color="coral")
ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("Value")
ax.set_title("Sparsity and Similarity by Stage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 6: Capacity scaling ---
ax = axes[1, 2]
cap_ds = sorted(capacity_results.keys())
cap_r2 = [capacity_results[d]["probe_r2_mean"] for d in cap_ds]
cap_sr = [capacity_results[d]["superposition_ratio"] for d in cap_ds]
l1 = ax.plot(cap_ds, cap_r2, "go-", markersize=6, linewidth=2, label="Probe R^2")
ax.set_xlabel("Cortical bottleneck dimensions")
ax.set_ylabel("Mean probe R^2", color="green")
ax2 = ax.twinx()
l2 = ax2.plot(cap_ds, cap_sr, "b^-", markersize=6, linewidth=2, label="Superposition ratio")
ax2.set_ylabel("Superposition ratio", color="blue")
lines = l1 + l2
ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
ax.set_title("Capacity Scaling")
ax.grid(True, alpha=0.3)

# --- Panel 7: Replay quantity effect ---
ax = axes[2, 0]
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
ax.set_title("Replay Quantity Effect (EC deep)")
ax.grid(True, alpha=0.3)

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
ax.set_xticklabels(cond_names, fontsize=7, rotation=15)
ax.set_title("Superposition Structure by Condition")
ax.legend(loc="upper left", fontsize=8)
ax2.legend(loc="upper right", fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# --- Panel 9: Active fraction progression through circuit ---
ax = axes[2, 2]
circuit_stages = ["EC", "DG", "CA3", "CA1", "Sub", "EC_deep"]
circuit_active = [stage_results[s]["active_fraction"] for s in circuit_stages]
circuit_decode = [stage_results[s]["feature_decode_r2_mean"] for s in circuit_stages]
x_circuit = np.arange(len(circuit_stages))
ax.plot(x_circuit, circuit_active, "bo-", markersize=8, linewidth=2, label="Active fraction")
ax.plot(x_circuit, circuit_decode, "r^-", markersize=8, linewidth=2, label="Decode R^2")
ax.set_xticks(x_circuit)
ax.set_xticklabels(circuit_stages, fontsize=9)
ax.set_ylabel("Value")
ax.set_title("Information Flow Through Circuit")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("consolidation_model_v8_results.png", dpi=150, bbox_inches='tight')
print("  Saved plot to consolidation_model_v8_results.png")

print(f"\n{'=' * 70}")
print("DONE")
print("=" * 70)
