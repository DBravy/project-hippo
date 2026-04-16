"""
Hippocampal-Transformer Backprojection Test
=============================================

Tests whether the hippocampal system can retrieve a successor pattern
and reconstruct the per-layer cortical (distilgpt2) residual states
of the next token via layer-specific backprojection matrices.

Architecture:
  distilgpt2 (frozen) --> 6 post-block residuals (R^768 each)
       |
  A_l projections (R^128 x R^768, random/fixed, one per layer)
       |
  concat --> ec_input (R^768)
       |
  Hippocampal system (EC_sup -> DG -> CA3 successor map)
       |
  Direct decoder (CA3 -> ec_input, delta rule)
       |
  B_l projections (R^768 x R^768, Hebbian-learned, one per layer)
       |
  predicted residual states per layer

  Note: CA1, Subiculum, and ECDeep are retained in the code and still
  train during Phase A, but are NOT in the active readout path.
  The direct decoder bypasses them entirely.

Evaluation:
  - Single-step reconstruction: cue with token t, retrieve t+1, compare
  - Multi-step retrieval degradation by layer
  - Token-by-token heatmaps
  - Baselines: random, identity (no hippocampus), same-token
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# UTILITY FUNCTIONS (from original model)
# =============================================================================

def make_feedforward_weights(D_output, d_input, connectivity_prob=0.33,
                             device='cpu', dtype=torch.float32):
    mask = (torch.rand(D_output, d_input, device=device) < connectivity_prob).to(dtype)
    weights = torch.randn(D_output, d_input, device=device, dtype=dtype) * mask
    row_norms = torch.linalg.norm(weights, dim=1, keepdim=True) + 1e-10
    return weights / row_norms


def build_ring_inhibition(D, sigma, connection_prob=None,
                          device='cpu', dtype=torch.float32):
    positions = torch.arange(D, device=device, dtype=dtype)
    dist = torch.abs(positions[:, None] - positions[None, :])
    dist = torch.minimum(dist, D - dist)
    W = torch.exp(-dist**2 / (2 * sigma**2))
    W[dist > 3 * sigma] = 0
    W.fill_diagonal_(0)
    if connection_prob is not None:
        mask = (torch.rand(D, D, device=device) < connection_prob).to(dtype)
        mask.fill_diagonal_(0)
        W *= mask
    row_sums = W.sum(dim=1, keepdim=True) + 1e-10
    W /= row_sums
    return W


def apply_kwta(pattern, k):
    out = torch.zeros_like(pattern)
    if k < pattern.shape[0]:
        values, indices = torch.topk(pattern, k)
        out[indices] = values
    else:
        out = pattern.clone()
    return out


def cosine_sim(a, b):
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


# =============================================================================
# CIRCUIT COMPONENTS (from original model)
# =============================================================================

class ECSuperficial:
    def __init__(self, d_ec, sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
                 pyr_to_stel_strength=0.3, connectivity_prob=0.33,
                 device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.pyr_to_stel_strength = pyr_to_stel_strength
        self.W_stellate = make_feedforward_weights(d_ec, d_ec, connectivity_prob, device, dtype)
        self.W_inh = build_ring_inhibition(d_ec, sigma_inh, device=device, dtype=dtype)
        self.W_pyramidal = make_feedforward_weights(d_ec, d_ec, connectivity_prob, device, dtype)
        self.W_pyr_to_stel = make_feedforward_weights(d_ec, d_ec, connectivity_prob, device, dtype)

    def forward(self, cortical_input):
        pyramidal = cortical_input  # dense signal passes through unchanged
        h_raw = self.W_stellate @ cortical_input
        h_raw = h_raw + self.pyr_to_stel_strength * (self.W_pyr_to_stel @ pyramidal)
        h_raw = torch.relu(h_raw)
        h = h_raw.clone()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = torch.relu(h_raw - self.gamma_inh * inh)
        return h, pyramidal


class DentateGyrusLateral:
    def __init__(self, d_input, D_output, sigma_inh=50, gamma_inh=1.0,
                 n_inh_steps=5, noise_scale=0.0, inh_connection_prob=None,
                 device='cpu', dtype=torch.float32):
        self.D_output = D_output
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.noise_scale = noise_scale
        self.device = device
        self.dtype = dtype
        self.W_ff = make_feedforward_weights(D_output, d_input, device=device, dtype=dtype)
        self.W_inh = build_ring_inhibition(
            D_output, sigma_inh, connection_prob=inh_connection_prob,
            device=device, dtype=dtype)

    def forward(self, x):
        h_raw = x @ self.W_ff.T
        h_raw = torch.relu(h_raw)
        if self.noise_scale > 0 and torch.any(h_raw > 0):
            mean_active = h_raw[h_raw > 0].mean()
            h_raw = torch.relu(
                h_raw + torch.randn(self.D_output, device=self.device, dtype=self.dtype)
                * self.noise_scale * mean_active)
        h = h_raw.clone()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = torch.relu(h_raw - self.gamma_inh * inh)
        return h


class CA1:
    """
    CA1 with error-driven Schaffer collateral learning, heterosynaptic LTD,
    divisive normalization, and conductance-based lateral inhibition.
    Adapted from consolidation_model_v4.

    During encoding:
      - EC input (via fixed W_ta) provides the target activation (h_ta)
      - CA3 input (via learned W_sc) provides the current estimate (h_sc)
      - Plateau potential gating: only neurons with strong TA drive learn
      - Error = h_ta - h_sc drives W_sc updates (self-limiting)
      - Heterosynaptic LTD depresses synapses from inactive CA3 neurons

    During retrieval:
      - Schaffer collateral drives CA1 + conductance-based lateral inhibition
      - Sparsity emerges from inhibition (no k-WTA)
    """
    def __init__(self, N_ca1, N_ca3, d_ec,
                 lr=0.3, plateau_threshold=0.7, plateau_sharpness=20.0,
                 weight_decay=1.0, div_norm_sigma=0.1,
                 connectivity_prob=0.33,
                 ltd_rate=0.05, ltd_ca3_threshold=0.0,
                 sigma_inh=25, gamma_inh=4.0, n_inh_steps=5,
                 E_inh=-0.4,
                 device='cpu', dtype=torch.float32):
        self.N_ca1 = N_ca1
        self.lr = lr
        self.plateau_threshold = plateau_threshold
        self.plateau_sharpness = plateau_sharpness
        self.weight_decay = weight_decay
        self.div_norm_sigma = div_norm_sigma
        self.ltd_rate = ltd_rate
        self.ltd_ca3_threshold = ltd_ca3_threshold
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.E_inh = E_inh
        self.device = device
        self.dtype = dtype

        # W_ta: fixed random feedforward (EC -> CA1, temporoammonic pathway)
        self.W_ta = make_feedforward_weights(
            N_ca1, d_ec, connectivity_prob, device, dtype)
        # W_sc: learned Schaffer collateral (CA3/DG -> CA1), initialized small
        mask = (torch.rand(N_ca1, N_ca3, device=device) < connectivity_prob).to(dtype)
        self.W_sc = torch.randn(N_ca1, N_ca3, device=device, dtype=dtype) * 0.01 * mask
        self.connectivity_mask = mask.clone()
        # Lateral inhibition
        self.W_inh = build_ring_inhibition(
            N_ca1, sigma_inh, device=device, dtype=dtype)
        self.n_episodes = 0

    def _divisive_normalize(self, h):
        pool = torch.mean(h) + self.div_norm_sigma
        return h / pool

    def compute_activations(self, x_ca3, x_ec):
        h_ta_raw = torch.relu(self.W_ta @ x_ec)
        h_sc_raw = torch.relu(self.W_sc @ x_ca3)
        h_ta = self._divisive_normalize(h_ta_raw)
        h_sc = self._divisive_normalize(h_sc_raw)
        max_ta = float(h_ta_raw.max())
        threshold = self.plateau_threshold * max_ta if max_ta > 1e-10 else 0.0
        gate = torch.sigmoid(
            self.plateau_sharpness * (h_ta_raw - threshold))
        return h_ta, h_sc, gate, h_ta_raw, h_sc_raw

    def encode(self, x_ca3, x_ec_stel):
        h_ta, h_sc, gate, _, _ = self.compute_activations(x_ca3, x_ec_stel)
        error = h_ta - h_sc
        gated_error = gate * error
        # LTP: error-driven update at gated synapses
        self.W_sc += self.lr * torch.outer(gated_error, x_ca3)
        # Heterosynaptic LTD: depress inactive CA3 synapses on gated neurons
        ca3_inactive = (x_ca3 <= self.ltd_ca3_threshold).to(self.dtype)
        self.W_sc *= (1.0 - self.ltd_rate * torch.outer(gate, ca3_inactive))
        # Optional global weight decay
        if self.weight_decay < 1.0:
            self.W_sc *= self.weight_decay
        # Enforce connectivity mask
        self.W_sc *= self.connectivity_mask
        self.n_episodes += 1

    def retrieve(self, x_ca3, x_ec_stel=None):
        h_sc_raw = torch.relu(self.W_sc @ x_ca3)
        mismatch = 0.0
        if x_ec_stel is not None:
            h_ta, h_sc, gate, _, _ = self.compute_activations(x_ca3, x_ec_stel)
            error = h_ta - h_sc
            gated_error = gate * error
            mismatch = float(torch.linalg.norm(gated_error)
                             / (torch.linalg.norm(h_ta) + 1e-10))
        # Conductance-based lateral inhibition
        h_out = h_sc_raw.clone()
        for _ in range(self.n_inh_steps):
            g_inh = self.gamma_inh * (self.W_inh @ h_out)
            h_out = torch.relu(
                (h_sc_raw + g_inh * self.E_inh) / (1.0 + g_inh))
        return h_out, mismatch


class Subiculum:
    """
    Subiculum: sparse-to-dense decoder in the replay pathway.
    Adapted from consolidation_model_v4/v10.

    During encoding: receives CA1 output and EC (pyramidal) input.
    EC shapes the target via fixed W_ec; error-driven learning at
    CA1->Sub synapses teaches the mapping. Heterosynaptic LTD and
    row-norm clipping prevent unbounded weight growth.

    During replay: only CA1 input arrives. Learned W_ca1 produces
    a dense code from the sparse CA1 pattern.
    """
    def __init__(self, N_sub, N_ca1, d_ec,
                 lr=0.05, ltd_rate=0.05, connectivity_prob=0.33,
                 device='cpu', dtype=torch.float32):
        self.N_sub = N_sub
        self.N_ca1 = N_ca1
        self.d_ec = d_ec
        self.lr = lr
        self.ltd_rate = ltd_rate
        self.device = device
        self.dtype = dtype

        # CA1 -> Sub: learned plastic weights, initialized small
        mask_ca1 = (torch.rand(N_sub, N_ca1, device=device) < connectivity_prob).to(dtype)
        self.W_ca1 = torch.randn(N_sub, N_ca1, device=device, dtype=dtype) * 0.01 * mask_ca1
        self.mask_ca1 = mask_ca1.clone()

        # EC -> Sub: fixed feedforward (temporoammonic pathway, independent from CA1's W_ta)
        self.W_ec = make_feedforward_weights(
            N_sub, d_ec, connectivity_prob, device, dtype)

        self.n_episodes = 0

    def encode(self, ca1_output, ec_pyramidal):
        # Normalize EC input: v4 operated on ~O(1) EC patterns, but GPT-2
        # residuals have norm ~168. Without normalization, h_ec >> h_ca1,
        # the error is huge, and the LMS update overshoots/diverges.
        ec_normed = ec_pyramidal / (torch.linalg.norm(ec_pyramidal) + 1e-10)
        h_ca1 = self.W_ca1 @ ca1_output
        h_ec = self.W_ec @ ec_normed
        h_sub = torch.relu(h_ca1 + h_ec)

        # Error-driven learning: teach W_ca1 to reproduce what EC drives
        error = h_ec - h_ca1
        self.W_ca1 += self.lr * torch.outer(error, ca1_output)

        # Heterosynaptic LTD
        if self.ltd_rate > 0:
            ca1_inactive = (ca1_output <= 0).to(self.dtype)
            sub_active = (h_sub > 0).to(self.dtype)
            self.W_ca1 *= (1.0 - self.ltd_rate
                           * torch.outer(sub_active, ca1_inactive))
        self.W_ca1 *= self.mask_ca1

        # Row-norm clipping (synaptic scaling)
        row_norms = torch.linalg.norm(self.W_ca1, dim=1, keepdim=True) + 1e-10
        max_norm = float(torch.quantile(row_norms.flatten(), 0.95))
        if max_norm > 1e-10:
            self.W_ca1 = torch.where(
                row_norms > max_norm,
                self.W_ca1 * (max_norm / row_norms),
                self.W_ca1)

        self.n_episodes += 1

    def replay(self, ca1_output):
        return torch.relu(self.W_ca1 @ ca1_output)


class ECDeep:
    """
    Deep entorhinal cortex: learns to reconstruct ec_input from
    CA1 + Subiculum signals. Produces a dense, full-dimensional output.
    
    Hebbian learning: W += lr * outer(ec_input, concat(ca1, sub))
    Retrieval: ec_deep_out = W @ concat(ca1, sub)
    """
    def __init__(self, d_ec, N_ca1, N_sub, lr=1.0, weight_decay=0.998,
                 device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.d_input = N_ca1 + N_sub
        self.lr = lr
        self.weight_decay = weight_decay
        self.W = torch.zeros((d_ec, self.d_input), device=device, dtype=dtype)
        self.n_episodes = 0

    def _combine_inputs(self, ca1_output, sub_output):
        return torch.cat([ca1_output, sub_output], dim=0)

    def encode(self, ca1_output, sub_output, ec_input):
        """Learn to map CA1+Sub -> ec_input."""
        combined = self._combine_inputs(ca1_output, sub_output)
        combined_norm = combined / (torch.linalg.norm(combined) + 1e-10)
        ec_norm = ec_input / (torch.linalg.norm(ec_input) + 1e-10)
        self.W += self.lr * torch.outer(ec_norm, combined_norm)
        self.W *= self.weight_decay
        self.n_episodes += 1

    def retrieve(self, ca1_output, sub_output):
        """Reconstruct ec_input from CA1+Sub. Dense output, no sparsity."""
        combined = self._combine_inputs(ca1_output, sub_output)
        combined_norm = combined / (torch.linalg.norm(combined) + 1e-10)
        return self.W @ combined_norm


class DirectDecoder:
    """
    Single-layer delta-rule decoder: CA3 -> ec_input.
    Bypasses CA1/Sub/ECDeep entirely.
    """
    def __init__(self, d_output, N_ca3, lr=0.3, device='cpu', dtype=torch.float32):
        self.W = torch.zeros((d_output, N_ca3), device=device, dtype=dtype)
        self.lr = lr

    def encode(self, ca3_state, ec_input):
        prediction = self.W @ ca3_state
        error = ec_input - prediction
        self.W += self.lr * torch.outer(error, ca3_state)

    def retrieve(self, ca3_state):
        return self.W @ ca3_state


# =============================================================================
# CA3: Pure Temporal Association (from original model)
# =============================================================================

class CA3Temporal:
    def __init__(self, N, k_active, lr=1.0, device='cpu', dtype=torch.float32):
        self.N = N
        self.k_active = k_active
        self.lr = lr
        self.device = device
        self.dtype = dtype
        self.W = torch.zeros((N, N), device=device, dtype=dtype)
        self.n_stored = 0
        self.mean_activity = torch.zeros(N, device=device, dtype=dtype)

    def _normalize_and_center(self, pattern):
        p = pattern / (torch.linalg.norm(pattern) + 1e-10)
        self.n_stored += 1
        self.mean_activity += (p - self.mean_activity) / self.n_stored
        p_c = p - self.mean_activity
        return p, p_c

    def store_online(self, pattern, prev_pattern=None):
        _, curr_c = self._normalize_and_center(pattern)
        if prev_pattern is not None:
            prev_p = prev_pattern / (torch.linalg.norm(prev_pattern) + 1e-10)
            prev_c = prev_p - self.mean_activity
            self.W += self.lr * torch.outer(curr_c, prev_c)
            self.W.fill_diagonal_(0)

    def retrieve(self, cue, n_iterations=5):
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        for _ in range(n_iterations):
            h = torch.relu(self.W @ x)
            x_new = apply_kwta(h, self.k_active)
            norm = torch.linalg.norm(x_new)
            if norm < 1e-10:
                break
            x_new = x_new / norm
            if torch.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x

    def retrieve_sequence(self, cue, n_steps, adapt_rate=0.15,
                          adapt_decay=0.85):
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        adaptation = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        trajectory = [x.clone()]
        adaptation_history = [0.0]

        for step in range(n_steps - 1):
            h = torch.relu(self.W @ x - adaptation)
            x_new = apply_kwta(h, self.k_active)
            norm = torch.linalg.norm(x_new)
            if norm < 1e-10:
                trajectory.append(torch.zeros(self.N, device=self.device,
                                              dtype=self.dtype))
                adaptation_history.append(float(torch.linalg.norm(adaptation)))
                x = x_new
                continue
            x_new = x_new / norm
            trajectory.append(x_new.clone())
            adaptation_history.append(float(torch.linalg.norm(adaptation)))
            adaptation = adapt_decay * adaptation + adapt_rate * x_new
            x = x_new

        return trajectory, adaptation_history


# =============================================================================
# HIPPOCAMPAL SYSTEM (modified to return ECDeep output)
# =============================================================================

class HippocampalSystemTemporal:
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3, ca3_lr=1.0,
                 direct_lr=0.3, direct_decay=0.998,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, ec_deep_params=None,
                 N_sub=1000, ca3_retrieval_iterations=5,
                 direct_decoder_lr=0.3,
                 device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.ca3_retrieval_iterations = ca3_retrieval_iterations
        self.device = device
        self.dtype = dtype

        self.ec_sup = ECSuperficial(d_ec, **(ec_sup_params or {}),
                                    device=device, dtype=dtype)
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}),
                                      device=device, dtype=dtype)
        self.ca3 = CA3Temporal(N_ca3, k_ca3, lr=ca3_lr,
                               device=device, dtype=dtype)

        # Mossy fiber projection: DG (D_dg) -> CA3 (N_ca3)
        self.W_mf = torch.randn((N_ca3, D_dg), device=device, dtype=dtype) / (D_dg ** 0.5)

        # --- CA1/Sub/ECDeep: retained but not in active readout path ---
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}),
                       device=device, dtype=dtype)
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}),
                             device=device, dtype=dtype)
        self.ec_deep = ECDeep(d_ec, N_ca1, N_sub,
                              **(ec_deep_params or {}),
                              device=device, dtype=dtype)

        # --- Direct decoder: active readout path ---
        self.direct_decoder = DirectDecoder(d_ec, N_ca3, lr=direct_decoder_lr,
                                            device=device, dtype=dtype)

        self.W_direct = torch.zeros((N_ca3, d_ec), device=device, dtype=dtype)
        self.direct_lr = direct_lr
        self.direct_decay = direct_decay
        self.k_ca3 = k_ca3

        self._prev_dg_pattern = None
        self._in_sequence = False

    def begin_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = True

    def end_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = False

    def _mossy_fiber(self, dg_out):
        """Project DG output (D_dg) to CA3 space (N_ca3) via mossy fibers."""
        return torch.relu(self.W_mf @ dg_out)

    def encode_single(self, ec_input):
        """Encode a single token. Returns dg_out, decoder_out."""
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        ca3_input = self._mossy_fiber(dg_out)

        prev = self._prev_dg_pattern if self._in_sequence else None
        self.ca3.store_online(ca3_input, prev_pattern=prev)

        if self._in_sequence:
            self._prev_dg_pattern = ca3_input.clone()

        # Direct pathway learning (stellate -> CA3 cue)
        ca3_norm = ca3_input / (torch.linalg.norm(ca3_input) + 1e-10)
        stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
        self.W_direct += self.direct_lr * torch.outer(ca3_norm, stel_norm)
        self.W_direct *= self.direct_decay

        # Get settled CA3 pattern for this token
        ca3_out = self.ca3.retrieve(ca3_input, self.ca3_retrieval_iterations)

        # --- Direct decoder learning (active readout path) ---
        self.direct_decoder.encode(ca3_out, ec_input)
        decoder_out = self.direct_decoder.retrieve(ca3_out)

        # --- CA1/Sub/ECDeep: not needed for reconstruction. ---
        # --- These would matter for integration with ongoing cortical ---
        # --- activity (e.g. replay into an active network), not for ---
        # --- offline readout of stored patterns. Disabled for now.  ---
        # self.ca1.encode(dg_out, stellate)
        # ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
        # self.sub.encode(ca1_out, pyramidal)
        # sub_out = self.sub.replay(ca1_out)
        # self.ec_deep.encode(ca1_out, sub_out, ec_input)

        return dg_out, decoder_out

    def _raw_direct_cue(self, stellate):
        return torch.relu(self.W_direct @ stellate)

    def _raw_dg_cue(self, stellate):
        return self._mossy_fiber(self.dg.forward(stellate))

    def _apply_ca3_competition(self, raw_drive):
        h = apply_kwta(raw_drive, self.k_ca3)
        norm = torch.linalg.norm(h)
        if norm > 1e-10:
            h = h / norm
        return h

    def retrieve_single_ec_deep(self, ec_input):
        """Retrieve one step: cue CA3, get successor, decode via direct decoder."""
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        raw_dg = self._raw_dg_cue(stellate)
        raw_direct = self._raw_direct_cue(stellate)
        combined = raw_dg + raw_direct
        ca3_cue = self._apply_ca3_competition(combined)

        # Retrieve successor from CA3
        ca3_successor = self.ca3.retrieve(ca3_cue, self.ca3_retrieval_iterations)

        # Direct decoder readout
        decoder_out = self.direct_decoder.retrieve(ca3_successor)

        return decoder_out, ca3_cue, ca3_successor

    def retrieve_sequence_ec_deep(self, ec_input, n_steps, adapt_rate=0.0,
                                  adapt_decay=0.0):
        """Retrieve a sequence of decoded outputs via the successor map."""
        stellate, _ = self.ec_sup.forward(ec_input)
        raw_dg = self._raw_dg_cue(stellate)
        raw_direct = self._raw_direct_cue(stellate)
        combined = raw_dg + raw_direct
        ca3_cue = self._apply_ca3_competition(combined)

        # Unroll successor map
        ca3_trajectory, adapt_hist = self.ca3.retrieve_sequence(
            ca3_cue, n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)

        # Convert each CA3 state via direct decoder
        decoder_trajectory = []
        for ca3_state in ca3_trajectory:
            decoder_out = self.direct_decoder.retrieve(ca3_state)
            decoder_trajectory.append(decoder_out)

        return decoder_trajectory, ca3_trajectory, adapt_hist


# =============================================================================
# NEW: CORTICAL PROJECTION SYSTEM
# =============================================================================

class CorticalProjection:
    """Layer-specific A matrices: cortex -> EC input."""

    def __init__(self, n_layers, d_model, r_per_layer,
                 device='cpu', dtype=torch.float32):
        self.n_layers = n_layers
        self.d_model = d_model
        self.r_per_layer = r_per_layer
        self.d_ec = n_layers * r_per_layer
        self.device = device
        self.dtype = dtype

        # Random, fixed A matrices (one per layer)
        self.A = []
        for l in range(n_layers):
            A_l = torch.randn(r_per_layer, d_model, device=device, dtype=dtype)
            # Row-normalize
            row_norms = torch.linalg.norm(A_l, dim=1, keepdim=True) + 1e-10
            A_l = A_l / row_norms
            self.A.append(A_l)

    def project(self, layer_residuals):
        """
        layer_residuals: list of n_layers tensors, each R^d_model
        Returns: concatenated EC input, R^d_ec
        """
        projections = []
        for l in range(self.n_layers):
            proj_l = self.A[l] @ layer_residuals[l]
            projections.append(proj_l)
        return torch.cat(projections, dim=0)


class CorticalBackprojection:
    """Layer-specific B matrices: EC deep output -> per-layer cortical predictions."""

    def __init__(self, n_layers, d_model, d_ec, lr=1.0, weight_decay=0.998,
                 device='cpu', dtype=torch.float32):
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ec = d_ec
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.dtype = dtype

        # Learned B matrices (one per layer), initialized to zeros
        self.B = []
        for l in range(n_layers):
            B_l = torch.zeros(d_model, d_ec, device=device, dtype=dtype)
            self.B.append(B_l)

    def encode(self, ec_deep_out, layer_residuals):
        """
        Hebbian learning: associate ec_deep_out with each layer's residual.
        ec_deep_out: R^d_ec (hippocampal output)
        layer_residuals: list of n_layers tensors, each R^d_model
        """
        ec_norm = ec_deep_out / (torch.linalg.norm(ec_deep_out) + 1e-10)
        for l in range(self.n_layers):
            h_l_norm = layer_residuals[l] / (torch.linalg.norm(layer_residuals[l]) + 1e-10)
            self.B[l] += self.lr * torch.outer(h_l_norm, ec_norm)
            self.B[l] *= self.weight_decay

    def retrieve(self, ec_deep_out):
        """
        Project ec_deep_out through each B matrix to predict per-layer residuals.
        Returns: list of n_layers tensors, each R^d_model
        """
        predictions = []
        for l in range(self.n_layers):
            pred_l = self.B[l] @ ec_deep_out
            predictions.append(pred_l)
        return predictions


# =============================================================================
# GPT-2 EXTRACTION
# =============================================================================

def load_gpt2_and_extract(text, seq_length=32, n_sequences=10,
                          device='cpu'):
    """
    Load distilgpt2, tokenize text, extract per-layer residual states.
    Returns:
        sequences_tokens: list of n_sequences lists of token ids
        sequences_residuals: list of n_sequences lists of per-token residuals
            Each per-token residual is a list of 6 tensors (R^768)
        tokenizer: for decoding
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading distilgpt2...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", output_hidden_states=True)
    model.eval()
    model.to(device)

    # Tokenize
    tokens = tokenizer.encode(text)
    print(f"  Total tokens: {len(tokens)}")

    total_needed = seq_length * n_sequences
    if len(tokens) < total_needed:
        print(f"  WARNING: only {len(tokens)} tokens available, "
              f"need {total_needed}. Reducing n_sequences.")
        n_sequences = len(tokens) // seq_length
        if n_sequences == 0:
            raise ValueError("Text too short for even one sequence")

    sequences_tokens = []
    sequences_residuals = []

    with torch.no_grad():
        for i in range(n_sequences):
            start = i * seq_length
            end = start + seq_length
            seq_token_ids = tokens[start:end]
            sequences_tokens.append(seq_token_ids)

            input_ids = torch.tensor([seq_token_ids], device=device)
            outputs = model(input_ids, output_hidden_states=True)

            # hidden_states: tuple of 7 tensors (embedding + 6 blocks)
            # Each is (1, seq_len, 768)
            hidden_states = outputs.hidden_states

            # Take post-block residuals (indices 1-6), skip embedding (index 0)
            per_token_residuals = []
            for t in range(seq_length):
                layer_residuals = []
                for l in range(1, 7):  # 6 transformer blocks
                    h = hidden_states[l][0, t, :].clone().to(torch.float32)
                    layer_residuals.append(h)
                per_token_residuals.append(layer_residuals)

            sequences_residuals.append(per_token_residuals)

    print(f"  Extracted {n_sequences} sequences of {seq_length} tokens")
    print(f"  Each token has 6 layer residuals of dim {sequences_residuals[0][0][0].shape[0]}")

    return sequences_tokens, sequences_residuals, tokenizer


# =============================================================================
# ENCODING
# =============================================================================

def encode_phase_a(hippo, cortical_proj, sequences_residuals,
                   n_repetitions=5):
    """
    Phase A: Hippocampal learning only.
    Train CA3 successor map, CA1, Subiculum, and direct pathway.
    No B matrix updates. Let the hippocampal circuit stabilize.

    Returns:
        all_dg_patterns: per-sequence, per-token DG patterns (from rep 0)
        all_ec_inputs: per-sequence, per-token EC input vectors (from rep 0)
    """
    n_sequences = len(sequences_residuals)
    all_dg_patterns = [[] for _ in range(n_sequences)]
    all_ec_inputs = [[] for _ in range(n_sequences)]

    global_step = 0
    for rep in range(n_repetitions):
        print(f"  Phase A repetition {rep + 1}/{n_repetitions}")
        for seq_idx, seq_residuals in enumerate(sequences_residuals):
            hippo.begin_sequence()
            for t, layer_residuals in enumerate(seq_residuals):
                ec_input = cortical_proj.project(layer_residuals)
                dg_out, decoder_out = hippo.encode_single(ec_input)

                if rep == 0:
                    all_dg_patterns[seq_idx].append(dg_out.clone())
                    all_ec_inputs[seq_idx].append(ec_input.clone())

                global_step += 1

            hippo.end_sequence()

        # --- Per-rep diagnostics ---
        ca3_w_norm = float(torch.linalg.norm(hippo.ca3.W))
        direct_w_norm = float(torch.linalg.norm(hippo.W_direct))
        decoder_w_norm = float(torch.linalg.norm(hippo.direct_decoder.W))

        # Probe one token to see activation magnitudes
        sample_ec = cortical_proj.project(sequences_residuals[0][0])
        stel, pyr = hippo.ec_sup.forward(sample_ec)
        dg_probe = hippo.dg.forward(stel)
        ca3_probe = hippo.ca3.retrieve(dg_probe, hippo.ca3_retrieval_iterations)
        decoder_probe = hippo.direct_decoder.retrieve(ca3_probe)
        decoder_sim = cosine_sim(decoder_probe, sample_ec)

        print(f"    [rep {rep+1}] steps={global_step}")
        print(f"      CA3 W:      Frob={ca3_w_norm:.4g}")
        print(f"      Direct W:   Frob={direct_w_norm:.4g}")
        print(f"      Decoder W:  Frob={decoder_w_norm:.4g}")
        print(f"      --- Activation norms (token 0,0) ---")
        print(f"      stellate:   {float(torch.linalg.norm(stel)):.4g}")
        print(f"      DG:         {float(torch.linalg.norm(dg_probe)):.4g}")
        print(f"      CA3 retr:   {float(torch.linalg.norm(ca3_probe)):.4g}")
        print(f"      Decoder out:{float(torch.linalg.norm(decoder_probe)):.4g}  sim_to_ec={decoder_sim:.4f}")

    return all_dg_patterns, all_ec_inputs


def evaluate_ec_bottleneck_after_phase_a(hippo, cortical_proj, sequences_residuals):
    """
    Per-stage pairwise similarity diagnostic.
    
    Measures where token discriminability dies in the pipeline by computing
    mean pairwise cosine similarity across tokens at every processing stage.
    
    Low pairwise sim = tokens are distinguishable at that stage.
    High pairwise sim = tokens look the same = information lost.
    
    Stages: ec_input -> stellate -> DG -> CA3 -> Decoder_out
    """
    print("\n  --- Per-stage discriminability diagnostic (post Phase A) ---")

    # Collect per-stage representations for each sequence
    # We'll compute stats across all sequences but show detail for seq 0
    stage_names = [
        "ec_input", "stellate", "DG", "CA3_retr", "Decoder_out"
    ]
    n_stages = len(stage_names)

    # Per-sequence, per-stage pairwise sims
    all_seq_pairwise = []
    # Per-sequence bottleneck (ec_input vs decoder_out)
    all_bottleneck_sims = []
    # Per-stage activation stats (from seq 0)
    stage_norms_seq0 = {name: [] for name in stage_names}
    stage_n_active_seq0 = {name: [] for name in stage_names}

    for seq_idx, seq_residuals in enumerate(sequences_residuals):
        # Collect all stage representations for all tokens in this sequence
        stage_reps = {name: [] for name in stage_names}
        seq_bottle = []

        for t, layer_residuals in enumerate(seq_residuals):
            ec_input = cortical_proj.project(layer_residuals)

            # Forward through hippocampal circuit without learning
            stellate, _ = hippo.ec_sup.forward(ec_input)
            dg_out = hippo.dg.forward(stellate)
            ca3_out = hippo.ca3.retrieve(dg_out, hippo.ca3_retrieval_iterations)
            decoder_out = hippo.direct_decoder.retrieve(ca3_out)

            stage_reps["ec_input"].append(ec_input.clone())
            stage_reps["stellate"].append(stellate.clone())
            stage_reps["DG"].append(dg_out.clone())
            stage_reps["CA3_retr"].append(ca3_out.clone())
            stage_reps["Decoder_out"].append(decoder_out.clone())

            seq_bottle.append(cosine_sim(ec_input, decoder_out))

            # Activation stats for seq 0
            if seq_idx == 0:
                for name, vec in [
                    ("ec_input", ec_input), ("stellate", stellate),
                    ("DG", dg_out), ("CA3_retr", ca3_out),
                    ("Decoder_out", decoder_out),
                ]:
                    stage_norms_seq0[name].append(float(torch.linalg.norm(vec)))
                    stage_n_active_seq0[name].append(
                        int((vec.abs() > 1e-8).sum()))

        all_bottleneck_sims.append(seq_bottle)

        # Compute pairwise similarity for each stage
        n_tokens = len(seq_residuals)
        seq_stage_pairwise = {}
        for name in stage_names:
            reps = stage_reps[name]
            pw_sims = []
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    pw_sims.append(cosine_sim(reps[i], reps[j]))
            seq_stage_pairwise[name] = {
                'mean': np.mean(pw_sims),
                'std': np.std(pw_sims),
                'min': np.min(pw_sims),
                'max': np.max(pw_sims),
            }
        all_seq_pairwise.append(seq_stage_pairwise)

    # === Print results ===

    # 1. Per-stage pairwise similarity (averaged across sequences)
    print("\n    Stage-by-stage pairwise cosine similarity (mean across sequences):")
    print(f"    {'Stage':<12s} {'Mean PW':>8s} {'Std PW':>8s} {'Min PW':>8s} {'Max PW':>8s}  Discriminability")
    print(f"    {'-'*72}")

    prev_mean = None
    for name in stage_names:
        means = [s[name]['mean'] for s in all_seq_pairwise]
        stds = [s[name]['std'] for s in all_seq_pairwise]
        mins = [s[name]['min'] for s in all_seq_pairwise]
        maxs = [s[name]['max'] for s in all_seq_pairwise]
        m = np.mean(means)
        s = np.mean(stds)
        mn = np.mean(mins)
        mx = np.mean(maxs)

        # Flag where discriminability drops
        if prev_mean is not None:
            delta = m - prev_mean
            if delta > 0.05:
                flag = f"  <-- JUMP +{delta:.3f}"
            elif delta > 0.01:
                flag = f"  (delta +{delta:.3f})"
            else:
                flag = ""
        else:
            flag = ""
        prev_mean = m

        print(f"    {name:<12s} {m:>8.4f} {s:>8.4f} {mn:>8.4f} {mx:>8.4f}{flag}")

    # 2. Activation statistics for seq 0
    print(f"\n    Activation statistics (seq 0, mean across tokens):")
    print(f"    {'Stage':<12s} {'Mean norm':>10s} {'Mean active':>12s} {'Dim':>6s}")
    print(f"    {'-'*44}")
    for name in stage_names:
        mean_norm = np.mean(stage_norms_seq0[name])
        mean_active = np.mean(stage_n_active_seq0[name])
        print(f"    {name:<12s} {mean_norm:>10.2f} {mean_active:>12.1f}     {hippo.d_ec}")

    # 3. EC bottleneck summary
    mean_bottle = np.mean(all_bottleneck_sims)
    print(f"\n    EC bottleneck (ec_input vs decoder_out): {mean_bottle:.4f}")

    # 4. Adjacent token similarity (do successive tokens differ more or less
    #    than random pairs? This tells us if sequence structure is preserved)
    print(f"\n    Adjacent vs non-adjacent token similarity (seq 0):")

    # Re-forward seq 0 to get reps
    stage_reps_0 = {name: [] for name in stage_names}
    for t, layer_residuals in enumerate(sequences_residuals[0]):
        ec_input = cortical_proj.project(layer_residuals)
        stellate, _ = hippo.ec_sup.forward(ec_input)
        dg_out = hippo.dg.forward(stellate)
        ca3_out = hippo.ca3.retrieve(dg_out, hippo.ca3_retrieval_iterations)
        decoder_out = hippo.direct_decoder.retrieve(ca3_out)

        stage_reps_0["ec_input"].append(ec_input)
        stage_reps_0["stellate"].append(stellate)
        stage_reps_0["DG"].append(dg_out)
        stage_reps_0["CA3_retr"].append(ca3_out)
        stage_reps_0["Decoder_out"].append(decoder_out)

    key_stages = ["ec_input", "stellate", "DG", "CA3_retr", "Decoder_out"]
    print(f"    {'Stage':<12s} {'Adjacent':>10s} {'Non-adj':>10s} {'Gap':>8s}")
    print(f"    {'-'*44}")
    for name in key_stages:
        reps = stage_reps_0[name]
        n_tok = len(reps)

        adj_sims = [cosine_sim(reps[t], reps[t+1]) for t in range(n_tok - 1)]
        nonadj_sims = []
        for i in range(n_tok):
            for j in range(i + 2, n_tok):  # skip adjacent
                nonadj_sims.append(cosine_sim(reps[i], reps[j]))

        mean_adj = np.mean(adj_sims)
        mean_nonadj = np.mean(nonadj_sims) if nonadj_sims else 0.0
        gap = mean_adj - mean_nonadj

        print(f"    {name:<12s} {mean_adj:>10.4f} {mean_nonadj:>10.4f} {gap:>+8.4f}")

    return mean_bottle


def encode_phase_b(hippo, cortical_proj, backproj, sequences_residuals,
                   n_repetitions=5):
    """
    Phase B: Backprojection learning only.
    Hippocampal weights are frozen (we simply don't call encode_single).
    Run forward passes to produce decoder output, then train B matrices.

    Returns:
        all_decoder_outs: per-sequence, per-token direct decoder outputs
    """
    n_sequences = len(sequences_residuals)
    all_decoder_outs = [[] for _ in range(n_sequences)]
    n_layers = backproj.n_layers

    for rep in range(n_repetitions):
        print(f"  Phase B repetition {rep + 1}/{n_repetitions}")

        # Track stats for first token of first seq this rep
        first_token_logged = False

        for seq_idx, seq_residuals in enumerate(sequences_residuals):
            for t, layer_residuals in enumerate(seq_residuals):
                ec_input = cortical_proj.project(layer_residuals)

                # Forward through hippocampal circuit (no learning)
                stellate, pyramidal = hippo.ec_sup.forward(ec_input)
                dg_out = hippo.dg.forward(stellate)
                ca3_out = hippo.ca3.retrieve(dg_out, hippo.ca3_retrieval_iterations)

                # Direct decoder readout (no learning)
                decoder_out = hippo.direct_decoder.retrieve(ca3_out)

                if not first_token_logged:
                    dec_norm = float(torch.linalg.norm(decoder_out))
                    dec_max = float(decoder_out.abs().max())
                    dec_sim = cosine_sim(decoder_out, ec_input)
                    n_inf = int(torch.isinf(decoder_out).sum())
                    n_nan = int(torch.isnan(decoder_out).sum())
                    print(f"    [rep {rep+1}] decoder sample: norm={dec_norm:.4g}, "
                          f"max={dec_max:.4g}, sim_to_ec={dec_sim:.4f}, "
                          f"inf={n_inf}, nan={n_nan}")
                    first_token_logged = True

                # Only B matrices learn
                backproj.encode(decoder_out, layer_residuals)

                if rep == 0:
                    all_decoder_outs[seq_idx].append(decoder_out.clone())

        # B matrix norms after each rep
        b_norms = [float(torch.linalg.norm(backproj.B[l])) for l in range(n_layers)]
        b_maxes = [float(backproj.B[l].abs().max()) if backproj.B[l].abs().max() > 0
                   else 0.0 for l in range(n_layers)]
        b_nans = [int(torch.isnan(backproj.B[l]).sum()) for l in range(n_layers)]
        print(f"    [rep {rep+1}] B norms: {['%.4g' % n for n in b_norms]}")
        print(f"    [rep {rep+1}] B max:   {['%.4g' % m for m in b_maxes]}")
        print(f"    [rep {rep+1}] B nans:  {b_nans}")

    return all_decoder_outs


# =============================================================================
# RETRIEVAL AND EVALUATION
# =============================================================================

def evaluate_single_step(hippo, cortical_proj, backproj,
                         sequences_residuals, all_dg_patterns,
                         all_ec_inputs, all_readout_outs):
    """
    For each token t, cue with t, retrieve successor (t+1),
    project through B matrices, compare to actual residuals at t+1.

    Baselines:
      - encode_decode: project stored readout_out[t] through B, compare to
        token t's own residuals. Measures how much the hippocampal circuit
        preserves about its own input (output fidelity).
      - random: project a random vector through B.
      - ec_bottleneck: cosine sim between ec_input and readout_out in EC
        space, measuring information loss through the hippocampal circuit
        before B matrices are even involved.
    """
    n_sequences = len(sequences_residuals)
    n_layers = backproj.n_layers
    results = {
        'per_layer_sims': [],            # (n_sequences, seq_len-1, n_layers)
        'baseline_encode_decode': [],    # (n_sequences, seq_len, n_layers)
        'baseline_random': [],           # (n_sequences, n_layers)
        'ec_bottleneck_sims': [],        # (n_sequences, seq_len)
    }

    for seq_idx, seq_residuals in enumerate(sequences_residuals):
        seq_len = len(seq_residuals)

        # --- Main evaluation: successor retrieval ---
        seq_sims = []
        for t in range(seq_len - 1):
            # Cue with token t
            ec_input_t = cortical_proj.project(seq_residuals[t])
            ec_deep_retrieved, _, _ = hippo.retrieve_single_ec_deep(ec_input_t)

            # Predict per-layer residuals for t+1
            predicted = backproj.retrieve(ec_deep_retrieved)
            actual = seq_residuals[t + 1]

            # Per-layer cosine similarity
            layer_sims = []
            for l in range(n_layers):
                sim = cosine_sim(predicted[l], actual[l])
                layer_sims.append(sim)
            seq_sims.append(layer_sims)

        results['per_layer_sims'].append(seq_sims)

        # --- Baseline 1: Encode-decode fidelity (same token) ---
        enc_dec_sims = []
        for t in range(seq_len):
            readout_t = all_readout_outs[seq_idx][t]
            predicted = backproj.retrieve(readout_t)
            actual = seq_residuals[t]
            layer_sims = [cosine_sim(predicted[l], actual[l])
                          for l in range(n_layers)]
            enc_dec_sims.append(layer_sims)
        results['baseline_encode_decode'].append(enc_dec_sims)

        # --- Baseline 2: EC bottleneck measurement ---
        ec_bottle_sims = []
        for t in range(seq_len):
            ec_in = all_ec_inputs[seq_idx][t]
            ec_out = all_readout_outs[seq_idx][t]
            ec_bottle_sims.append(cosine_sim(ec_in, ec_out))
        results['ec_bottleneck_sims'].append(ec_bottle_sims)

        # --- Baseline 3: Random vector through B ---
        random_vec = torch.randn(backproj.d_ec, device=backproj.device,
                                 dtype=backproj.dtype)
        random_pred = backproj.retrieve(random_vec)
        random_sims = []
        for l in range(n_layers):
            sims_l = [cosine_sim(random_pred[l], seq_residuals[t][l])
                      for t in range(seq_len)]
            random_sims.append(np.mean(sims_l))
        results['baseline_random'].append(random_sims)

    return results


def evaluate_multi_step(hippo, cortical_proj, backproj,
                        sequences_residuals, max_steps=8):
    """
    Cue with first token, unroll successor map for max_steps,
    project each step through B, compare to actual residuals.
    """
    n_sequences = len(sequences_residuals)
    n_layers = backproj.n_layers

    # (n_sequences, max_steps, n_layers)
    all_sims = []

    for seq_idx, seq_residuals in enumerate(sequences_residuals):
        seq_len = len(seq_residuals)
        n_steps = min(max_steps, seq_len - 1)

        # Cue with token 0
        ec_input_0 = cortical_proj.project(seq_residuals[0])

        # Retrieve sequence of ECDeep outputs
        ec_deep_traj, ca3_traj, _ = hippo.retrieve_sequence_ec_deep(
            ec_input_0, n_steps=n_steps + 1,  # +1 because first is the cue
            adapt_rate=0.0, adapt_decay=0.0)

        # ec_deep_traj[0] is the cue pattern, [1] is successor, etc.
        step_sims = []
        for step in range(n_steps):
            # ec_deep_traj[step + 1] is the retrieved pattern for token step+1
            predicted = backproj.retrieve(ec_deep_traj[step + 1])
            actual = seq_residuals[step + 1]

            layer_sims = [cosine_sim(predicted[l], actual[l])
                          for l in range(n_layers)]
            step_sims.append(layer_sims)

        # Pad if needed
        while len(step_sims) < max_steps:
            step_sims.append([0.0] * n_layers)

        all_sims.append(step_sims)

    return np.array(all_sims)  # (n_sequences, max_steps, n_layers)


def evaluate_dg_retrieval(hippo, cortical_proj, sequences_residuals,
                          all_dg_patterns):
    """
    Evaluate intra-hippocampal retrieval quality (DG pattern similarity)
    for comparison with cortical reconstruction quality.
    """
    n_sequences = len(sequences_residuals)
    all_sim_matrices = []

    for seq_idx in range(min(n_sequences, 3)):
        seq_residuals = sequences_residuals[seq_idx]
        seq_len = len(seq_residuals)
        ref = all_dg_patterns[seq_idx]

        # Cue with token 0, retrieve full sequence
        ec_input_0 = cortical_proj.project(seq_residuals[0])

        _, ca3_traj, _ = hippo.retrieve_sequence_ec_deep(
            ec_input_0, n_steps=seq_len,
            adapt_rate=0.0, adapt_decay=0.0)

        # Similarity matrix: retrieved CA3 vs stored DG
        sim = np.zeros((len(ca3_traj), len(ref)))
        for t in range(len(ca3_traj)):
            for r in range(len(ref)):
                sim[t, r] = cosine_sim(ca3_traj[t], ref[r])

        all_sim_matrices.append(sim)

    return all_sim_matrices


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(single_step_results, multi_step_sims, dg_sim_matrices,
                 sequences_residuals, cortical_proj, backproj,
                 save_path="hippocampal_transformer_results.png"):

    n_layers = backproj.n_layers
    fig = plt.figure(figsize=(24, 28))
    gs = GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.3)
    fig.suptitle("Hippocampal-Transformer Backprojection",
                 fontsize=16, fontweight='bold', y=0.99)

    layer_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_layers))
    layer_labels = [f"Layer {l+1}" for l in range(n_layers)]

    # ----- Row 1: Per-layer single-step reconstruction vs baselines -----
    per_layer = np.array(single_step_results['per_layer_sims'])  # (n_seq, T-1, n_layers)
    mean_per_layer = np.mean(per_layer, axis=(0, 1))  # (n_layers,)
    std_per_layer = np.std(np.mean(per_layer, axis=1), axis=0)  # across sequences

    enc_dec = np.array(single_step_results['baseline_encode_decode'])  # (n_seq, T, n_layers)
    mean_enc_dec = np.mean(enc_dec, axis=(0, 1))
    std_enc_dec = np.std(np.mean(enc_dec, axis=1), axis=0)

    baseline_random = np.mean(single_step_results['baseline_random'], axis=0)

    # Bar chart: successor retrieval vs encode-decode fidelity vs random
    ax1a = fig.add_subplot(gs[0, 0:2])
    x = np.arange(n_layers)
    width = 0.25
    ax1a.bar(x - width, mean_per_layer, width, yerr=std_per_layer,
             color='steelblue', label='Successor retrieval (t+1)', capsize=3)
    ax1a.bar(x, mean_enc_dec, width, yerr=std_enc_dec,
             color='coral', label='Encode-decode fidelity (same t)', capsize=3)
    ax1a.bar(x + width, baseline_random, width,
             color='gray', label='Random')
    ax1a.set_xlabel("Transformer Layer")
    ax1a.set_ylabel("Cosine Similarity")
    ax1a.set_title("Cortical Reconstruction by Layer")
    ax1a.set_xticks(x)
    ax1a.set_xticklabels(layer_labels, fontsize=8)
    ax1a.legend(fontsize=7)
    ax1a.grid(True, alpha=0.3, axis='y')

    # EC bottleneck: how much does the hippocampal circuit preserve in EC space?
    ec_bottle = np.array(single_step_results['ec_bottleneck_sims'])  # (n_seq, T)
    mean_bottle = np.mean(ec_bottle)
    std_bottle = np.std(np.mean(ec_bottle, axis=1))

    ax1b = fig.add_subplot(gs[0, 2])
    ax1b.bar(['EC\nbottleneck'], [mean_bottle], yerr=[std_bottle],
             color='mediumpurple', capsize=5, width=0.5)
    ax1b.set_ylabel("Cosine Similarity")
    ax1b.set_title("EC Input vs Decoder Output")
    ax1b.set_ylim(0, 1.0)
    ax1b.grid(True, alpha=0.3, axis='y')

    # Distribution of ec bottleneck sims
    ax1c = fig.add_subplot(gs[0, 3])
    ax1c.hist(ec_bottle.flatten(), bins=30, color='mediumpurple', alpha=0.7,
              density=True)
    ax1c.set_xlabel("Cosine Similarity")
    ax1c.set_ylabel("Density")
    ax1c.set_title("EC Bottleneck Distribution")
    ax1c.grid(True, alpha=0.3)

    # ----- Row 2: Multi-step degradation by layer -----
    # multi_step_sims: (n_sequences, max_steps, n_layers)
    mean_multi = np.mean(multi_step_sims, axis=0)  # (max_steps, n_layers)
    std_multi = np.std(multi_step_sims, axis=0)
    max_steps = mean_multi.shape[0]

    ax2a = fig.add_subplot(gs[1, 0:2])
    for l in range(n_layers):
        ax2a.plot(range(1, max_steps + 1), mean_multi[:, l],
                  'o-', color=layer_colors[l], linewidth=2,
                  label=layer_labels[l], markersize=4)
        ax2a.fill_between(range(1, max_steps + 1),
                          mean_multi[:, l] - std_multi[:, l],
                          mean_multi[:, l] + std_multi[:, l],
                          alpha=0.15, color=layer_colors[l])
    # Overlay encode-decode ceiling as horizontal lines per layer
    for l in range(n_layers):
        ax2a.axhline(y=mean_enc_dec[l], color=layer_colors[l],
                     linestyle='--', alpha=0.4, linewidth=1)
    ax2a.set_xlabel("Retrieval Step (tokens ahead)")
    ax2a.set_ylabel("Cosine Similarity")
    ax2a.set_title("Multi-Step Degradation by Layer\n(dashed = encode-decode ceiling)")
    ax2a.legend(fontsize=7)
    ax2a.grid(True, alpha=0.3)

    # Mean across layers per step
    ax2b = fig.add_subplot(gs[1, 2:4])
    mean_across_layers = np.mean(mean_multi, axis=1)
    ax2b.plot(range(1, max_steps + 1), mean_across_layers,
              'o-', color='steelblue', linewidth=2, markersize=6,
              label='Successor retrieval')
    ax2b.axhline(y=np.mean(mean_enc_dec), color='coral',
                 linestyle='--', linewidth=2, label='Encode-decode ceiling')
    ax2b.axhline(y=np.mean(baseline_random), color='gray',
                 linestyle=':', linewidth=2, label='Random baseline')
    ax2b.set_xlabel("Retrieval Step (tokens ahead)")
    ax2b.set_ylabel("Mean Cosine Similarity (all layers)")
    ax2b.set_title("Multi-Step Degradation (Layer-Averaged)")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3)

    # ----- Row 3: Token-by-token heatmaps -----
    # Successor retrieval heatmap for seq 0
    if len(single_step_results['per_layer_sims']) > 0:
        seq0_sims = np.array(single_step_results['per_layer_sims'][0])

        ax3a = fig.add_subplot(gs[2, 0:2])
        im = ax3a.imshow(seq0_sims.T, aspect='auto', cmap='viridis',
                         vmin=-0.2, vmax=1.0)
        ax3a.set_xlabel("Token Position (predicting t+1)")
        ax3a.set_ylabel("Layer")
        ax3a.set_yticks(range(n_layers))
        ax3a.set_yticklabels(layer_labels, fontsize=8)
        ax3a.set_title("Seq 0: Successor Retrieval Quality")
        plt.colorbar(im, ax=ax3a, shrink=0.8, label='Cosine Sim')

    # Encode-decode fidelity heatmap for seq 0
    if len(single_step_results['baseline_encode_decode']) > 0:
        seq0_enc_dec = np.array(single_step_results['baseline_encode_decode'][0])

        ax3b = fig.add_subplot(gs[2, 2:4])
        im = ax3b.imshow(seq0_enc_dec.T, aspect='auto', cmap='viridis',
                         vmin=-0.2, vmax=1.0)
        ax3b.set_xlabel("Token Position")
        ax3b.set_ylabel("Layer")
        ax3b.set_yticks(range(n_layers))
        ax3b.set_yticklabels(layer_labels, fontsize=8)
        ax3b.set_title("Seq 0: Encode-Decode Fidelity (same token)")
        plt.colorbar(im, ax=ax3b, shrink=0.8, label='Cosine Sim')

    # ----- Row 4: DG similarity matrices -----
    for i, sim in enumerate(dg_sim_matrices[:2]):
        ax = fig.add_subplot(gs[3, i])
        im = ax.imshow(sim, aspect='auto', cmap='viridis',
                       vmin=-0.2, vmax=1.0)
        ax.set_xlabel("Stored DG Pattern")
        ax.set_ylabel("Retrieved Step")
        ax.set_title(f"Seq {i}: Intra-Hippocampal Retrieval")
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Successor retrieval heatmap for seq 1 (if available)
    if len(single_step_results['per_layer_sims']) > 1:
        seq1_sims = np.array(single_step_results['per_layer_sims'][1])

        ax4c = fig.add_subplot(gs[3, 2:4])
        im = ax4c.imshow(seq1_sims.T, aspect='auto', cmap='viridis',
                         vmin=-0.2, vmax=1.0)
        ax4c.set_xlabel("Token Position (predicting t+1)")
        ax4c.set_ylabel("Layer")
        ax4c.set_yticks(range(n_layers))
        ax4c.set_yticklabels(layer_labels, fontsize=8)
        ax4c.set_title("Seq 1: Successor Retrieval Quality")
        plt.colorbar(im, ax=ax4c, shrink=0.8, label='Cosine Sim')

    # ----- Row 5: Summary statistics -----
    ax_summary = fig.add_subplot(gs[4, 0:2])
    ax_summary.axis('off')
    summary_text = "Summary Statistics\n"
    summary_text += "=" * 50 + "\n\n"
    summary_text += "Successor retrieval (cosine sim vs t+1):\n"
    for l in range(n_layers):
        summary_text += (f"  Layer {l+1}: {mean_per_layer[l]:.4f} "
                         f"(+/- {std_per_layer[l]:.4f})\n")
    summary_text += f"\n  Mean across layers: {np.mean(mean_per_layer):.4f}\n"
    summary_text += f"\nEncode-decode fidelity (cosine sim vs same t):\n"
    for l in range(n_layers):
        summary_text += (f"  Layer {l+1}: {mean_enc_dec[l]:.4f} "
                         f"(+/- {std_enc_dec[l]:.4f})\n")
    summary_text += f"\n  Mean across layers: {np.mean(mean_enc_dec):.4f}\n"
    summary_text += f"\nEC bottleneck (ec_input vs decoder): {mean_bottle:.4f}\n"
    summary_text += f"Random baseline: {np.mean(baseline_random):.4f}\n"
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                    fontsize=9, verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax_summary2 = fig.add_subplot(gs[4, 2:4])
    ax_summary2.axis('off')
    summary_text2 = "Multi-step retrieval:\n"
    summary_text2 += "=" * 50 + "\n\n"
    for step in range(min(max_steps, 8)):
        step_mean = np.mean(mean_multi[step])
        summary_text2 += f"  Step {step+1}: {step_mean:.4f}\n"
    summary_text2 += f"\nEncode-decode ceiling: {np.mean(mean_enc_dec):.4f}\n"
    summary_text2 += f"\nGap (ceiling - step 1): "
    summary_text2 += f"{np.mean(mean_enc_dec) - np.mean(mean_multi[0]):.4f}\n"
    summary_text2 += "  (= error from successor map)\n"
    summary_text2 += f"\nGap (ceiling - random): "
    summary_text2 += f"{np.mean(mean_enc_dec) - np.mean(baseline_random):.4f}\n"
    summary_text2 += "  (= total usable signal)\n"
    ax_summary2.text(0.05, 0.95, summary_text2, transform=ax_summary2.transAxes,
                     fontsize=9, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.5))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {save_path}")


# =============================================================================
# MAIN
# =============================================================================

# Default text source (used if no file is provided)
DEFAULT_TEXT = """
The hippocampus is a complex brain structure embedded deep in the temporal lobe.
It has a major role in learning and memory. It is a plastic and vulnerable
structure that gets damaged by a variety of stimuli. Studies have shown that it
also gets affected in a variety of neurological and psychiatric disorders. In
Alzheimer's disease the hippocampus is one of the first regions of the brain to
suffer damage. The hippocampus contains two main interlocking parts: the hippocampus
proper and the dentate gyrus. The entorhinal cortex is considered part of the
hippocampal formation because of its strong connections with the hippocampus.
The hippocampal formation plays an important role in the consolidation of information
from short-term memory to long-term memory and spatial memory that enables navigation.
The neural layout of the hippocampus is very similar across mammals and the structure
has been studied extensively in rodents as a model system for understanding brain
function. Neurons in the rodent hippocampus fire as the animal traverses specific
locations in its environment forming a cognitive map. Place cells in the hippocampus
become active when the animal enters a particular place in its environment known as
the place field. These neurons fire rapidly when the animal is in a particular
location and fire slowly or not at all when the animal is elsewhere. The discovery
of place cells led to the idea that the hippocampus acts as a cognitive map of the
environment. Sharp wave ripples are high frequency oscillations that occur in the
hippocampus during sleep and quiet wakefulness. They are thought to play a role in
memory consolidation by reactivating recently formed memory traces. The hippocampus
is also involved in pattern separation and pattern completion processes that allow
for the discrimination of similar memories and the retrieval of complete memories
from partial cues. Damage to the hippocampus can result in anterograde amnesia
which is the inability to form new declarative memories although procedural memories
can still be formed. The famous case of patient Henry Molaison demonstrated the
critical role of the hippocampus in memory formation after bilateral removal of
the hippocampal formation resulted in profound amnesia. The hippocampal system
receives highly processed information from all sensory modalities through the
entorhinal cortex which serves as the main interface between the neocortex and
the hippocampus. This convergence of multimodal information enables the hippocampus
to form rich episodic memories that bind together the various elements of an
experience into a coherent whole. The theta rhythm is a prominent oscillation
observed in the hippocampus during active exploration and REM sleep. It is thought
to provide a temporal framework for organizing neural activity and may play a role
in the encoding and retrieval of episodic memories.
"""


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    gpt2_device = 'cpu'
    if torch.cuda.is_available():
        gpt2_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpt2_device = 'mps'

    dtype = torch.float32

    # ---- Configuration ----
    seq_length = 32
    n_sequences = 8
    n_layers = 6
    d_model = 768
    r_per_layer = 128
    d_ec = n_layers * r_per_layer  # 768
    n_reps_hippo_cfg = 5
    n_reps_backproj_cfg = 5
    max_multi_steps = 8

    print("=" * 70)
    print("Hippocampal-Transformer Backprojection Test")
    print("=" * 70)
    print(f"  seq_length={seq_length}, n_sequences={n_sequences}")
    print(f"  n_layers={n_layers}, d_model={d_model}, r_per_layer={r_per_layer}")
    print(f"  d_ec={d_ec}")
    print(f"  Phase A reps (hippocampal): {n_reps_hippo_cfg}")
    print(f"  Phase B reps (backprojection): {n_reps_backproj_cfg}")

    # ---- Phase 1: Extract cortical representations ----
    print("\n--- Phase 1: Extracting GPT-2 representations ---")
    sequences_tokens, sequences_residuals, tokenizer = load_gpt2_and_extract(
        DEFAULT_TEXT, seq_length=seq_length, n_sequences=n_sequences,
        device=gpt2_device)

    # Move residuals to computation device
    if device != torch.device(gpt2_device):
        for seq_idx in range(len(sequences_residuals)):
            for t in range(len(sequences_residuals[seq_idx])):
                for l in range(n_layers):
                    sequences_residuals[seq_idx][t][l] = \
                        sequences_residuals[seq_idx][t][l].to(device)

    # Print token sequences for reference
    for i, seq_toks in enumerate(sequences_tokens[:3]):
        decoded = tokenizer.decode(seq_toks)
        print(f"  Seq {i}: {decoded[:80]}...")

    # ---- Phase 2: Build projection system ----
    print("\n--- Phase 2: Building projection system ---")
    cortical_proj = CorticalProjection(
        n_layers, d_model, r_per_layer, device=device, dtype=dtype)
    print(f"  A matrices: {n_layers} x ({r_per_layer} x {d_model}), random/fixed")

    backproj = CorticalBackprojection(
        n_layers, d_model, d_ec, lr=1.0, weight_decay=0.998,
        device=device, dtype=dtype)
    print(f"  B matrices: {n_layers} x ({d_model} x {d_ec}), Hebbian-learned")

    # ---- Phase 3: Build hippocampal system ----
    print("\n--- Phase 3: Building hippocampal system ---")
    hippo_kwargs = {
        "d_ec": d_ec,
        "D_dg": d_ec,
        "N_ca3": d_ec,
        "N_ca1": d_ec,
        "k_ca3": 50,
        "N_sub": d_ec,
        "ca3_lr": 1.0,
        "direct_lr": 0.3,
        "direct_decay": 0.998,
        "ca3_retrieval_iterations": 5,
        "ec_sup_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "pyr_to_stel_strength": 0.3,
        },
        "dg_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "noise_scale": 0.0,
        },
        "ca1_params": {
            "lr": 0.3,
            "plateau_threshold": 0.7,
            "plateau_sharpness": 20.0,
            "weight_decay": 1.0,           # no global decay; LTD handles homeostasis
            "div_norm_sigma": 0.1,
            "connectivity_prob": 0.33,
            "ltd_rate": 0.05,
            "ltd_ca3_threshold": 0.0,
            "sigma_inh": 25,
            "gamma_inh": 4.0,
            "n_inh_steps": 5,
            "E_inh": -0.4,
        },
        "sub_params": {
            "lr": 0.05,
            "ltd_rate": 0.05,
            "connectivity_prob": 0.33,
        },
        "ec_deep_params": {"lr": 1.0, "weight_decay": 0.998},
        "direct_decoder_lr": 0.3,
    }

    hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)
    print(f"  N_ca3={d_ec}, k_ca3=50, D_dg={d_ec}")

    # ---- Phase 4A: Hippocampal learning ----
    print("\n--- Phase 4A: Hippocampal learning ---")
    all_dg_patterns, all_ec_inputs = encode_phase_a(
        hippo, cortical_proj, sequences_residuals,
        n_repetitions=n_reps_hippo_cfg)

    # Quick sanity check: DG pattern statistics
    sample_dg = all_dg_patterns[0][0]
    n_active = int((sample_dg > 0).sum())
    print(f"  Sample DG pattern: {n_active} active units out of {len(sample_dg)}")

    # ---- Phase 4A.5: EC bottleneck diagnostic ----
    ec_bottle_post_a = evaluate_ec_bottleneck_after_phase_a(
        hippo, cortical_proj, sequences_residuals)

    # ---- Phase 4B: Backprojection learning ----
    print("\n--- Phase 4B: Backprojection learning ---")
    all_decoder_outs = encode_phase_b(
        hippo, cortical_proj, backproj, sequences_residuals,
        n_repetitions=n_reps_backproj_cfg)

    # Check B matrix norms and decoder statistics
    sample_decoder = all_decoder_outs[0][0]
    decoder_norm = float(torch.linalg.norm(sample_decoder))
    n_nonzero = int((sample_decoder.abs() > 1e-8).sum())
    print(f"  Sample decoder output: norm={decoder_norm:.4f}, "
          f"nonzero={n_nonzero}/{len(sample_decoder)}")
    for l in range(n_layers):
        b_norm = float(torch.linalg.norm(backproj.B[l]))
        print(f"  B[{l}] Frobenius norm: {b_norm:.2f}")

    # ---- Phase 5: Evaluation ----
    print("\n--- Phase 5: Single-step evaluation ---")
    single_step_results = evaluate_single_step(
        hippo, cortical_proj, backproj, sequences_residuals, all_dg_patterns,
        all_ec_inputs, all_decoder_outs)

    per_layer = np.array(single_step_results['per_layer_sims'])
    mean_per_layer = np.mean(per_layer, axis=(0, 1))
    for l in range(n_layers):
        print(f"  Layer {l+1}: successor retrieval sim = {mean_per_layer[l]:.4f}")
    print(f"  Overall mean: {np.mean(mean_per_layer):.4f}")

    enc_dec = np.array(single_step_results['baseline_encode_decode'])
    mean_enc_dec = np.mean(enc_dec, axis=(0, 1))
    for l in range(n_layers):
        print(f"  Layer {l+1}: encode-decode fidelity = {mean_enc_dec[l]:.4f}")
    print(f"  Encode-decode mean: {np.mean(mean_enc_dec):.4f}")

    ec_bottle = np.array(single_step_results['ec_bottleneck_sims'])
    print(f"  EC bottleneck (ec_input vs decoder_out): {np.mean(ec_bottle):.4f}")

    baseline_random = np.mean(single_step_results['baseline_random'], axis=0)
    print(f"  Random baseline: {np.mean(baseline_random):.4f}")

    print("\n--- Phase 5b: Multi-step evaluation ---")
    multi_step_sims = evaluate_multi_step(
        hippo, cortical_proj, backproj, sequences_residuals,
        max_steps=max_multi_steps)

    mean_multi = np.mean(multi_step_sims, axis=0)
    for step in range(min(max_multi_steps, mean_multi.shape[0])):
        step_mean = np.mean(mean_multi[step])
        print(f"  Step {step+1}: mean cosine sim = {step_mean:.4f}")

    print("\n--- Phase 5c: DG retrieval evaluation ---")
    dg_sim_matrices = evaluate_dg_retrieval(
        hippo, cortical_proj, sequences_residuals, all_dg_patterns)

    for i, sim in enumerate(dg_sim_matrices):
        diag_mean = np.mean([sim[t, t] for t in range(min(sim.shape))])
        print(f"  Seq {i}: DG diagonal sim = {diag_mean:.4f}")

    # ---- Phase 6: Plotting ----
    print("\n--- Phase 6: Plotting ---")
    save_path = "hippocampal_transformer_results.png"
    plot_results(single_step_results, multi_step_sims, dg_sim_matrices,
                 sequences_residuals, cortical_proj, backproj,
                 save_path=save_path)

    print(f"\n{'=' * 70}")
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
