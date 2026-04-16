"""
Output Circuit Test: CA3 -> CA1 -> Sub -> ECDeep
=================================================

Isolated test of the hippocampal readout chain. The question:
given that CA3 stores and retrieves successor sequences accurately,
can the downstream components (CA1, Subiculum, ECDeep) reconstruct
the original EC input from the sparse CA3 representation?

Uses:
  - Gain-modulated CA3 with W_mf (from gain_modulated_capacity_test.py)
  - Error-driven CA1 with plateau potentials and conductance-based inhibition
  - Error-driven Subiculum with heterosynaptic LTD
  - Hebbian ECDeep

Tests:
  1. Encoding learning curves: track CA1 mismatch, Sub error, ECDeep
     reconstruction quality over training repetitions
  2. Single-step readout quality: after encoding, present each stored
     CA3 state without EC, run through the readout chain, compare
     ECDeep output to original EC input. Vary capacity.
  3. Sequence readout: retrieve a CA3 trajectory, run each step
     through the readout chain. Does readout quality degrade?
  4. Component isolation: how much of the original EC input is
     recoverable at each stage (CA1 alone, CA1+Sub, full chain)?
  5. EC support during retrieval: CA1 can receive EC stellate via
     the temporoammonic pathway. How much does this help?
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch

torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# UTILITY FUNCTIONS
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
# ENCODING-SIDE COMPONENTS (from gain_modulated_capacity_test.py)
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
        pyramidal = torch.relu(self.W_pyramidal @ cortical_input)
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


class CA3GainModulated:
    def __init__(self, N, k_active, lr=1.0, device='cpu', dtype=torch.float32):
        self.N = N
        self.k_active = k_active
        self.lr = lr
        self.device = device
        self.dtype = dtype
        self.W = torch.zeros((N, N), device=device, dtype=dtype)
        self.n_stored = 0
        self.mean_activity = torch.zeros(N, device=device, dtype=dtype)
        self.x_prev = None
        self.adaptation = torch.zeros(N, device=device, dtype=dtype)

    def reset_state(self):
        self.x_prev = None
        self.adaptation = torch.zeros(self.N, device=self.device, dtype=self.dtype)

    def step(self, mf_input=None, pp_input=None,
             g_recurrent=1.0, g_mf=1.0, g_pp=1.0,
             adapt_rate=0.0, adapt_decay=0.0,
             learn=False):
        h = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        if self.x_prev is not None and g_recurrent > 0:
            h = h + g_recurrent * (self.W @ self.x_prev)
        if mf_input is not None and g_mf > 0:
            h = h + g_mf * mf_input
        if pp_input is not None and g_pp > 0:
            h = h + g_pp * pp_input
        h = torch.relu(h - self.adaptation)
        x_new = apply_kwta(h, self.k_active)
        norm = torch.linalg.norm(x_new)
        if norm > 1e-10:
            x_new = x_new / norm
        else:
            x_new = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        if learn:
            self.n_stored += 1
            self.mean_activity += (x_new - self.mean_activity) / self.n_stored
            if self.x_prev is not None:
                curr_c = x_new - self.mean_activity
                prev_c = self.x_prev - self.mean_activity
                self.W += self.lr * torch.outer(curr_c, prev_c)
                self.W.fill_diagonal_(0)
        self.adaptation = adapt_decay * self.adaptation + adapt_rate * x_new
        self.x_prev = x_new.clone()
        return x_new


# =============================================================================
# READOUT COMPONENTS (from backprojection model, under test)
# =============================================================================

class CA1:
    """
    Error-driven CA1 with plateau potential gating, heterosynaptic LTD,
    divisive normalization, and conductance-based lateral inhibition.
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

        self.W_ta = make_feedforward_weights(
            N_ca1, d_ec, connectivity_prob, device, dtype)
        mask = (torch.rand(N_ca1, N_ca3, device=device) < connectivity_prob).to(dtype)
        self.W_sc = torch.randn(N_ca1, N_ca3, device=device, dtype=dtype) * 0.01 * mask
        self.connectivity_mask = mask.clone()
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
        self.W_sc += self.lr * torch.outer(gated_error, x_ca3)
        ca3_inactive = (x_ca3 <= self.ltd_ca3_threshold).to(self.dtype)
        self.W_sc *= (1.0 - self.ltd_rate * torch.outer(gate, ca3_inactive))
        if self.weight_decay < 1.0:
            self.W_sc *= self.weight_decay
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
        h_out = h_sc_raw.clone()
        for _ in range(self.n_inh_steps):
            g_inh = self.gamma_inh * (self.W_inh @ h_out)
            h_out = torch.relu(
                (h_sc_raw + g_inh * self.E_inh) / (1.0 + g_inh))
        return h_out, mismatch


class Subiculum:
    """
    Error-driven Subiculum with heterosynaptic LTD and row-norm clipping.
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

        mask_ca1 = (torch.rand(N_sub, N_ca1, device=device) < connectivity_prob).to(dtype)
        self.W_ca1 = torch.randn(N_sub, N_ca1, device=device, dtype=dtype) * 0.01 * mask_ca1
        self.mask_ca1 = mask_ca1.clone()
        self.W_ec = make_feedforward_weights(
            N_sub, d_ec, connectivity_prob, device, dtype)
        self.n_episodes = 0

    def encode(self, ca1_output, ec_pyramidal):
        ec_normed = ec_pyramidal / (torch.linalg.norm(ec_pyramidal) + 1e-10)
        h_ca1 = self.W_ca1 @ ca1_output
        h_ec = self.W_ec @ ec_normed
        h_sub = torch.relu(h_ca1 + h_ec)

        error = h_ec - h_ca1
        self.W_ca1 += self.lr * torch.outer(error, ca1_output)

        if self.ltd_rate > 0:
            ca1_inactive = (ca1_output <= 0).to(self.dtype)
            sub_active = (h_sub > 0).to(self.dtype)
            self.W_ca1 *= (1.0 - self.ltd_rate
                           * torch.outer(sub_active, ca1_inactive))
        self.W_ca1 *= self.mask_ca1

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
    Hebbian ECDeep: learns to reconstruct ec_input from CA1 + Subiculum.
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
        combined = self._combine_inputs(ca1_output, sub_output)
        combined_norm = combined / (torch.linalg.norm(combined) + 1e-10)
        ec_norm = ec_input / (torch.linalg.norm(ec_input) + 1e-10)
        self.W += self.lr * torch.outer(ec_norm, combined_norm)
        self.W *= self.weight_decay
        self.n_episodes += 1

    def retrieve(self, ca1_output, sub_output):
        combined = self._combine_inputs(ca1_output, sub_output)
        combined_norm = combined / (torch.linalg.norm(combined) + 1e-10)
        return self.W @ combined_norm


# =============================================================================
# FULL SYSTEM
# =============================================================================

class HippocampalSystem:
    """
    Full system with gain-modulated CA3 and the readout chain.
    Exposes readout components for direct evaluation.
    """
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3, ca3_lr=1.0,
                 direct_lr=0.3, direct_decay=0.998,
                 mf_connectivity_prob=0.33,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, ec_deep_params=None,
                 N_sub=1000,
                 device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.N_ca3 = N_ca3
        self.k_ca3 = k_ca3
        self.device = device
        self.dtype = dtype

        self.ec_sup = ECSuperficial(d_ec, **(ec_sup_params or {}),
                                    device=device, dtype=dtype)
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}),
                                      device=device, dtype=dtype)
        self.ca3 = CA3GainModulated(N_ca3, k_ca3, lr=ca3_lr,
                                     device=device, dtype=dtype)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}),
                       device=device, dtype=dtype)
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}),
                             device=device, dtype=dtype)
        self.ec_deep = ECDeep(d_ec, N_ca1, N_sub,
                              **(ec_deep_params or {}),
                              device=device, dtype=dtype)

        self.W_mf = make_feedforward_weights(
            N_ca3, D_dg, mf_connectivity_prob, device, dtype)
        self.W_direct = torch.zeros((N_ca3, d_ec), device=device, dtype=dtype)
        self.direct_lr = direct_lr
        self.direct_decay = direct_decay

    def begin_sequence(self):
        self.ca3.reset_state()

    def end_sequence(self):
        self.ca3.reset_state()

    def encode_step(self, ec_input):
        """
        Full encoding step. Returns diagnostics for tracking learning.
        """
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        mf_input = torch.relu(self.W_mf @ dg_out)

        ca3_state = self.ca3.step(
            mf_input=mf_input, g_mf=1.0, g_recurrent=0.0, g_pp=0.0,
            learn=True)

        # Direct pathway learning
        stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
        self.W_direct += self.direct_lr * torch.outer(ca3_state, stel_norm)
        self.W_direct *= self.direct_decay

        # CA1 encoding: sees CA3 state + EC stellate
        self.ca1.encode(ca3_state, stellate)
        ca1_out, ca1_mismatch = self.ca1.retrieve(ca3_state, stellate)

        # Sub encoding: sees CA1 output + EC pyramidal
        self.sub.encode(ca1_out, pyramidal)
        sub_out = self.sub.replay(ca1_out)

        # ECDeep encoding: sees CA1+Sub + original EC input
        self.ec_deep.encode(ca1_out, sub_out, ec_input)
        ec_deep_out = self.ec_deep.retrieve(ca1_out, sub_out)

        # Reconstruction quality at this step
        ec_deep_sim = cosine_sim(ec_deep_out, ec_input)

        return {
            'ca3_state': ca3_state.clone(),
            'stellate': stellate.clone(),
            'pyramidal': pyramidal.clone(),
            'ca1_out': ca1_out.clone(),
            'sub_out': sub_out.clone(),
            'ec_deep_out': ec_deep_out.clone(),
            'ca1_mismatch': ca1_mismatch,
            'ec_deep_sim': ec_deep_sim,
        }

    def readout(self, ca3_state, ec_stellate=None):
        """
        Run the readout chain from a CA3 state.
        Optionally provide EC stellate for supported retrieval.
        """
        ca1_out, ca1_mismatch = self.ca1.retrieve(ca3_state, ec_stellate)
        sub_out = self.sub.replay(ca1_out)
        ec_deep_out = self.ec_deep.retrieve(ca1_out, sub_out)

        ca1_norm = float(torch.linalg.norm(ca1_out))
        ca1_sparsity = float((ca1_out > 0).sum()) / len(ca1_out)
        sub_norm = float(torch.linalg.norm(sub_out))
        sub_sparsity = float((sub_out > 0).sum()) / len(sub_out)

        return {
            'ca1_out': ca1_out,
            'sub_out': sub_out,
            'ec_deep_out': ec_deep_out,
            'ca1_mismatch': ca1_mismatch,
            'ca1_norm': ca1_norm,
            'ca1_sparsity': ca1_sparsity,
            'sub_norm': sub_norm,
            'sub_sparsity': sub_sparsity,
        }

    def recall_ca3_trajectory(self, ec_input, n_steps,
                               cue_g_mf=0.0, cue_g_pp=1.0,
                               run_g_recurrent=1.0):
        """Retrieve CA3 trajectory only (no readout)."""
        self.ca3.reset_state()
        stellate, _ = self.ec_sup.forward(ec_input)
        mf_input = None
        pp_input = None
        if cue_g_mf > 0:
            dg_out = self.dg.forward(stellate)
            mf_input = torch.relu(self.W_mf @ dg_out)
        if cue_g_pp > 0:
            pp_input = torch.relu(self.W_direct @ stellate)

        ca3_state = self.ca3.step(
            mf_input=mf_input, pp_input=pp_input,
            g_mf=cue_g_mf, g_recurrent=0.0, g_pp=cue_g_pp)
        trajectory = [ca3_state.clone()]

        for _ in range(n_steps - 1):
            ca3_state = self.ca3.step(
                g_mf=0.0, g_recurrent=run_g_recurrent, g_pp=0.0)
            trajectory.append(ca3_state.clone())

        return trajectory


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_sequences(n_sequences, seq_length, d_ec, device='cpu',
                       dtype=torch.float32, seed=42):
    rng = np.random.RandomState(seed)
    sequences = []
    for _ in range(n_sequences):
        seq = []
        for _ in range(seq_length):
            p = rng.randn(d_ec).astype(np.float32)
            p = p / (np.linalg.norm(p) + 1e-10)
            seq.append(torch.tensor(p, device=device, dtype=dtype))
        sequences.append(seq)
    return sequences


# =============================================================================
# TEST 1: Encoding Learning Curves
# =============================================================================

def test_learning_curves(hippo_kwargs, device, dtype, n_max_reps=10):
    """
    Track readout quality over encoding repetitions.
    After each repetition, evaluate readout on all stored patterns.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Encoding Learning Curves")
    print("=" * 70)

    n_seq = 10
    seq_length = 6

    sequences = generate_sequences(n_seq, seq_length,
                                   hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=1000)

    torch.manual_seed(42)
    hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

    # Store per-rep diagnostics
    results = {
        'reps': [],
        'ca1_mismatch_mean': [],
        'ec_deep_sim_mean': [],
        'ca1_norm_mean': [],
        'ca1_sparsity_mean': [],
        'sub_norm_mean': [],
        'sub_sparsity_mean': [],
        'readout_ec_deep_sim': [],    # evaluated after encoding, unsupported
        'readout_ec_deep_sim_ec': [], # evaluated after encoding, with EC support
    }

    all_ca3_patterns = [[] for _ in sequences]
    all_stellate = [[] for _ in sequences]
    all_ec_inputs = [[] for _ in sequences]

    for rep in range(n_max_reps):
        # --- Encode ---
        rep_ca1_mm = []
        rep_ec_deep_sim = []

        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for step_idx, ec_pattern in enumerate(seq):
                diag = hippo.encode_step(ec_pattern)
                rep_ca1_mm.append(diag['ca1_mismatch'])
                rep_ec_deep_sim.append(diag['ec_deep_sim'])

                if rep == 0:
                    all_ca3_patterns[seq_idx].append(diag['ca3_state'])
                    all_stellate[seq_idx].append(diag['stellate'])
                    all_ec_inputs[seq_idx].append(ec_pattern.clone())
            hippo.end_sequence()

        # --- Evaluate readout on stored patterns ---
        readout_sims = []
        readout_sims_ec = []
        ca1_norms = []
        ca1_sparsities = []
        sub_norms = []
        sub_sparsities = []

        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3_pat = all_ca3_patterns[seq_idx][t]
                ec_in = all_ec_inputs[seq_idx][t]
                stel = all_stellate[seq_idx][t]

                # Unsupported readout (no EC)
                ro = hippo.readout(ca3_pat)
                readout_sims.append(cosine_sim(ro['ec_deep_out'], ec_in))
                ca1_norms.append(ro['ca1_norm'])
                ca1_sparsities.append(ro['ca1_sparsity'])
                sub_norms.append(ro['sub_norm'])
                sub_sparsities.append(ro['sub_sparsity'])

                # EC-supported readout
                ro_ec = hippo.readout(ca3_pat, ec_stellate=stel)
                readout_sims_ec.append(cosine_sim(ro_ec['ec_deep_out'], ec_in))

        results['reps'].append(rep + 1)
        results['ca1_mismatch_mean'].append(np.mean(rep_ca1_mm))
        results['ec_deep_sim_mean'].append(np.mean(rep_ec_deep_sim))
        results['ca1_norm_mean'].append(np.mean(ca1_norms))
        results['ca1_sparsity_mean'].append(np.mean(ca1_sparsities))
        results['sub_norm_mean'].append(np.mean(sub_norms))
        results['sub_sparsity_mean'].append(np.mean(sub_sparsities))
        results['readout_ec_deep_sim'].append(np.mean(readout_sims))
        results['readout_ec_deep_sim_ec'].append(np.mean(readout_sims_ec))

        print(f"  Rep {rep+1:2d}: CA1 mm={np.mean(rep_ca1_mm):.4f}, "
              f"EC_deep(enc)={np.mean(rep_ec_deep_sim):.4f}, "
              f"EC_deep(readout)={np.mean(readout_sims):.4f}, "
              f"EC_deep(+EC)={np.mean(readout_sims_ec):.4f}, "
              f"CA1 norm={np.mean(ca1_norms):.4f}, "
              f"CA1 sparsity={np.mean(ca1_sparsities):.4f}")

    return results


# =============================================================================
# TEST 2: Readout Quality vs Capacity
# =============================================================================

def test_readout_capacity(hippo_kwargs, device, dtype, n_repetitions=5):
    """
    After encoding, present each stored CA3 state (no EC), run readout,
    compare ECDeep output to original EC input. Vary n_sequences.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Readout Quality vs Capacity")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [1, 5, 10, 20, 50]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=2000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

        all_ca3 = [[] for _ in sequences]
        all_stel = [[] for _ in sequences]
        all_ec = [[] for _ in sequences]

        for rep in range(n_repetitions):
            for seq_idx, seq in enumerate(sequences):
                hippo.begin_sequence()
                for step_idx, ec_pattern in enumerate(seq):
                    diag = hippo.encode_step(ec_pattern)
                    if rep == 0:
                        all_ca3[seq_idx].append(diag['ca3_state'])
                        all_stel[seq_idx].append(diag['stellate'])
                        all_ec[seq_idx].append(ec_pattern.clone())
                hippo.end_sequence()

        # Evaluate readout
        ec_deep_sims = []
        ec_deep_sims_ec = []
        ca1_sims_to_stel = []
        sub_norms = []

        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3_pat = all_ca3[seq_idx][t]
                ec_in = all_ec[seq_idx][t]
                stel = all_stel[seq_idx][t]

                ro = hippo.readout(ca3_pat)
                ec_deep_sims.append(cosine_sim(ro['ec_deep_out'], ec_in))
                ca1_sims_to_stel.append(cosine_sim(ro['ca1_out'], stel))
                sub_norms.append(ro['sub_norm'])

                ro_ec = hippo.readout(ca3_pat, ec_stellate=stel)
                ec_deep_sims_ec.append(cosine_sim(ro_ec['ec_deep_out'], ec_in))

        results[n_seq] = {
            'ec_deep_sim': np.mean(ec_deep_sims),
            'ec_deep_sim_ec': np.mean(ec_deep_sims_ec),
            'ca1_sim_to_stel': np.mean(ca1_sims_to_stel),
            'sub_norm': np.mean(sub_norms),
            'ec_deep_sims_all': ec_deep_sims,
        }

        print(f"  n_seq={n_seq:4d}: ECDeep={results[n_seq]['ec_deep_sim']:.4f}, "
              f"ECDeep(+EC)={results[n_seq]['ec_deep_sim_ec']:.4f}, "
              f"CA1~stel={results[n_seq]['ca1_sim_to_stel']:.4f}, "
              f"sub_norm={results[n_seq]['sub_norm']:.4f}")

    return results, n_seq_values


# =============================================================================
# TEST 3: Sequence Readout (retrieved trajectory through readout chain)
# =============================================================================

def test_sequence_readout(hippo_kwargs, device, dtype, n_repetitions=5):
    """
    Retrieve a CA3 trajectory via the successor map, then pass each
    step through the readout chain. Compare ECDeep output at each
    step to the original EC input for that position.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Sequence Readout")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [5, 20, 50]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=3000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

        all_ca3 = [[] for _ in sequences]
        all_ec = [[] for _ in sequences]

        for rep in range(n_repetitions):
            for seq_idx, seq in enumerate(sequences):
                hippo.begin_sequence()
                for ec_pattern in seq:
                    diag = hippo.encode_step(ec_pattern)
                    if rep == 0:
                        all_ca3[seq_idx].append(diag['ca3_state'])
                        all_ec[seq_idx].append(ec_pattern.clone())
                hippo.end_sequence()

        # Retrieve sequences and readout
        sample_n = min(n_seq, 10)
        per_step_ca3_sim = [[] for _ in range(seq_length)]
        per_step_readout_sim = [[] for _ in range(seq_length)]

        for si in range(sample_n):
            ca3_traj = hippo.recall_ca3_trajectory(
                sequences[si][0], n_steps=seq_length,
                cue_g_mf=2.0, cue_g_pp=1.0, run_g_recurrent=1.0)

            for t in range(seq_length):
                ca3_sim = cosine_sim(ca3_traj[t], all_ca3[si][t])
                per_step_ca3_sim[t].append(ca3_sim)

                ro = hippo.readout(ca3_traj[t])
                readout_sim = cosine_sim(ro['ec_deep_out'], all_ec[si][t])
                per_step_readout_sim[t].append(readout_sim)

        results[n_seq] = {
            'ca3_per_step': [np.mean(s) for s in per_step_ca3_sim],
            'readout_per_step': [np.mean(s) for s in per_step_readout_sim],
        }

        ca3_str = " ".join(f"{np.mean(s):.3f}" for s in per_step_ca3_sim)
        ro_str = " ".join(f"{np.mean(s):.3f}" for s in per_step_readout_sim)
        print(f"  n_seq={n_seq:4d}:")
        print(f"    CA3 per step:     [{ca3_str}]")
        print(f"    Readout per step: [{ro_str}]")

    return results, n_seq_values, seq_length


# =============================================================================
# TEST 4: Component Isolation
# =============================================================================

def test_component_isolation(hippo_kwargs, device, dtype, n_repetitions=5):
    """
    For each stored pattern, measure how much of the original EC input
    is recoverable at each stage of the readout chain:
      - CA1 output vs EC stellate (does CA1 learn the CA3->stellate map?)
      - Sub output vs EC pyramidal (does Sub decode CA1?)
      - ECDeep output vs EC input (full chain)
    Also measure intermediate representations: norms, sparsity, etc.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Component Isolation")
    print("=" * 70)

    n_seq = 10
    seq_length = 6

    sequences = generate_sequences(n_seq, seq_length,
                                   hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=4000)

    torch.manual_seed(42)
    hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

    all_ca3 = [[] for _ in sequences]
    all_stel = [[] for _ in sequences]
    all_pyr = [[] for _ in sequences]
    all_ec = [[] for _ in sequences]

    for rep in range(n_repetitions):
        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for ec_pattern in seq:
                diag = hippo.encode_step(ec_pattern)
                if rep == 0:
                    all_ca3[seq_idx].append(diag['ca3_state'])
                    all_stel[seq_idx].append(diag['stellate'])
                    all_pyr[seq_idx].append(diag['pyramidal'])
                    all_ec[seq_idx].append(ec_pattern.clone())
            hippo.end_sequence()

    # Evaluate each component
    ca1_vs_stel = []
    sub_vs_pyr = []
    ecdeep_vs_ec = []
    ca1_norms = []
    ca1_sparsities = []
    ca1_n_active = []
    sub_norms = []
    sub_sparsities = []
    sub_n_active = []

    for seq_idx in range(n_seq):
        for t in range(seq_length):
            ca3_pat = all_ca3[seq_idx][t]
            stel = all_stel[seq_idx][t]
            pyr = all_pyr[seq_idx][t]
            ec_in = all_ec[seq_idx][t]

            ro = hippo.readout(ca3_pat)

            ca1_vs_stel.append(cosine_sim(ro['ca1_out'], stel))
            sub_vs_pyr.append(cosine_sim(ro['sub_out'], pyr))
            ecdeep_vs_ec.append(cosine_sim(ro['ec_deep_out'], ec_in))

            ca1_norms.append(ro['ca1_norm'])
            ca1_sparsities.append(ro['ca1_sparsity'])
            ca1_n_active.append(int((ro['ca1_out'] > 0).sum()))
            sub_norms.append(ro['sub_norm'])
            sub_sparsities.append(ro['sub_sparsity'])
            sub_n_active.append(int((ro['sub_out'] > 0).sum()))

    results = {
        'ca1_vs_stel': np.mean(ca1_vs_stel),
        'sub_vs_pyr': np.mean(sub_vs_pyr),
        'ecdeep_vs_ec': np.mean(ecdeep_vs_ec),
        'ca1_norm': np.mean(ca1_norms),
        'ca1_sparsity': np.mean(ca1_sparsities),
        'ca1_n_active': np.mean(ca1_n_active),
        'sub_norm': np.mean(sub_norms),
        'sub_sparsity': np.mean(sub_sparsities),
        'sub_n_active': np.mean(sub_n_active),
        # Keep per-item for histograms
        'ca1_vs_stel_all': ca1_vs_stel,
        'sub_vs_pyr_all': sub_vs_pyr,
        'ecdeep_vs_ec_all': ecdeep_vs_ec,
    }

    print(f"  CA1 output vs EC stellate:    {results['ca1_vs_stel']:.4f}")
    print(f"  Sub output vs EC pyramidal:   {results['sub_vs_pyr']:.4f}")
    print(f"  ECDeep output vs EC input:    {results['ecdeep_vs_ec']:.4f}")
    print(f"  CA1: norm={results['ca1_norm']:.4f}, "
          f"sparsity={results['ca1_sparsity']:.4f}, "
          f"active={results['ca1_n_active']:.0f}/{hippo_kwargs['N_ca1']}")
    print(f"  Sub: norm={results['sub_norm']:.4f}, "
          f"sparsity={results['sub_sparsity']:.4f}, "
          f"active={results['sub_n_active']:.0f}/{hippo_kwargs['N_sub']}")

    return results


# =============================================================================
# TEST 5: EC Support During Retrieval
# =============================================================================

def test_ec_support(hippo_kwargs, device, dtype, n_repetitions=5):
    """
    During CA1 retrieval, the temporoammonic pathway (W_ta @ stellate)
    can provide support. Compare readout with and without EC at varying
    capacity. This reveals how much CA1 is relying on its learned
    Schaffer collaterals vs the EC teaching signal.
    """
    print("\n" + "=" * 70)
    print("TEST 5: EC Support During Retrieval")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [1, 5, 10, 20, 50]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=5000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

        all_ca3 = [[] for _ in sequences]
        all_stel = [[] for _ in sequences]
        all_ec = [[] for _ in sequences]

        for rep in range(n_repetitions):
            for seq_idx, seq in enumerate(sequences):
                hippo.begin_sequence()
                for ec_pattern in seq:
                    diag = hippo.encode_step(ec_pattern)
                    if rep == 0:
                        all_ca3[seq_idx].append(diag['ca3_state'])
                        all_stel[seq_idx].append(diag['stellate'])
                        all_ec[seq_idx].append(ec_pattern.clone())
                hippo.end_sequence()

        # Compare with and without EC support
        sims_no_ec = []
        sims_with_ec = []
        ca1_mm_no_ec = []
        ca1_mm_with_ec = []

        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3_pat = all_ca3[seq_idx][t]
                stel = all_stel[seq_idx][t]
                ec_in = all_ec[seq_idx][t]

                ro = hippo.readout(ca3_pat)
                sims_no_ec.append(cosine_sim(ro['ec_deep_out'], ec_in))
                ca1_mm_no_ec.append(ro['ca1_mismatch'])

                ro_ec = hippo.readout(ca3_pat, ec_stellate=stel)
                sims_with_ec.append(cosine_sim(ro_ec['ec_deep_out'], ec_in))
                ca1_mm_with_ec.append(ro_ec['ca1_mismatch'])

        results[n_seq] = {
            'no_ec': np.mean(sims_no_ec),
            'with_ec': np.mean(sims_with_ec),
            'ca1_mm_no_ec': np.mean(ca1_mm_no_ec),
            'ca1_mm_with_ec': np.mean(ca1_mm_with_ec),
            'gap': np.mean(sims_with_ec) - np.mean(sims_no_ec),
        }

        print(f"  n_seq={n_seq:4d}: no_EC={results[n_seq]['no_ec']:.4f}, "
              f"with_EC={results[n_seq]['with_ec']:.4f}, "
              f"gap={results[n_seq]['gap']:.4f}, "
              f"CA1_mm(no)={results[n_seq]['ca1_mm_no_ec']:.4f}, "
              f"CA1_mm(ec)={results[n_seq]['ca1_mm_with_ec']:.4f}")

    return results, n_seq_values


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_results(t1, t2, t3, t4, t5, save_path):
    fig = plt.figure(figsize=(24, 30))
    gs = GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Output Circuit: CA3 -> CA1 -> Sub -> ECDeep",
                 fontsize=16, fontweight='bold', y=0.995)

    colors = ['steelblue', 'coral', 'forestgreen', 'purple', 'goldenrod']

    # ------------------------------------------------------------------
    # Row 1: Learning curves
    # ------------------------------------------------------------------
    ax1a = fig.add_subplot(gs[0, 0:2])
    reps = t1['reps']
    ax1a.plot(reps, t1['readout_ec_deep_sim'], 'o-', color='steelblue',
              linewidth=2, label='ECDeep (no EC)', markersize=5)
    ax1a.plot(reps, t1['readout_ec_deep_sim_ec'], 's-', color='coral',
              linewidth=2, label='ECDeep (+EC support)', markersize=5)
    ax1a.plot(reps, t1['ec_deep_sim_mean'], '^-', color='forestgreen',
              linewidth=2, label='ECDeep (during encoding)', markersize=5)
    ax1a.set_xlabel("Encoding repetition")
    ax1a.set_ylabel("Cosine similarity to EC input")
    ax1a.set_title("Test 1: ECDeep Reconstruction Over Reps")
    ax1a.legend(fontsize=8)
    ax1a.set_ylim(-0.1, 1.1)
    ax1a.grid(True, alpha=0.3)

    ax1b = fig.add_subplot(gs[0, 2:4])
    ax1b.plot(reps, t1['ca1_norm_mean'], 'o-', color='steelblue',
              linewidth=2, label='CA1 norm', markersize=5)
    ax1b.plot(reps, t1['sub_norm_mean'], 's-', color='coral',
              linewidth=2, label='Sub norm', markersize=5)
    ax1b_twin = ax1b.twinx()
    ax1b_twin.plot(reps, t1['ca1_sparsity_mean'], '^--', color='steelblue',
                   linewidth=1.5, label='CA1 sparsity', markersize=4, alpha=0.7)
    ax1b_twin.plot(reps, t1['sub_sparsity_mean'], 'v--', color='coral',
                   linewidth=1.5, label='Sub sparsity', markersize=4, alpha=0.7)
    ax1b.set_xlabel("Encoding repetition")
    ax1b.set_ylabel("Norm")
    ax1b_twin.set_ylabel("Sparsity (fraction active)")
    ax1b.set_title("Test 1: CA1/Sub Representation Stats")
    lines1, labels1 = ax1b.get_legend_handles_labels()
    lines2, labels2 = ax1b_twin.get_legend_handles_labels()
    ax1b.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')
    ax1b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 2: Readout vs capacity
    # ------------------------------------------------------------------
    results_2, n_seq_2 = t2

    ax2a = fig.add_subplot(gs[1, 0:2])
    no_ec = [results_2[n]['ec_deep_sim'] for n in n_seq_2]
    with_ec = [results_2[n]['ec_deep_sim_ec'] for n in n_seq_2]
    ca1_stel = [results_2[n]['ca1_sim_to_stel'] for n in n_seq_2]
    ax2a.plot(n_seq_2, no_ec, 'o-', color='steelblue', linewidth=2,
              label='ECDeep (no EC)', markersize=6)
    ax2a.plot(n_seq_2, with_ec, 's-', color='coral', linewidth=2,
              label='ECDeep (+EC)', markersize=6)
    ax2a.plot(n_seq_2, ca1_stel, '^-', color='forestgreen', linewidth=2,
              label='CA1 vs stellate', markersize=6)
    ax2a.set_xlabel("Number of sequences")
    ax2a.set_ylabel("Cosine similarity")
    ax2a.set_title("Test 2: Readout Quality vs Capacity")
    ax2a.legend(fontsize=8)
    ax2a.set_ylim(-0.1, 1.1)
    ax2a.set_xscale('log')
    ax2a.grid(True, alpha=0.3)

    ax2b = fig.add_subplot(gs[1, 2:4])
    # Histogram of per-item ECDeep similarities at a chosen capacity
    mid_cap = n_seq_2[len(n_seq_2) // 2]
    ax2b.hist(results_2[mid_cap]['ec_deep_sims_all'], bins=30, alpha=0.7,
              color='steelblue', edgecolor='white')
    ax2b.axvline(results_2[mid_cap]['ec_deep_sim'], color='red', linestyle='--',
                 label=f"Mean={results_2[mid_cap]['ec_deep_sim']:.3f}")
    ax2b.set_xlabel("Cosine similarity")
    ax2b.set_ylabel("Count")
    ax2b.set_title(f"Test 2: ECDeep Distribution (n_seq={mid_cap})")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 3: Sequence readout
    # ------------------------------------------------------------------
    results_3, n_seq_3, seq_len_3 = t3

    ax3a = fig.add_subplot(gs[2, 0:2])
    for ci, n_seq in enumerate(n_seq_3):
        ax3a.plot(range(1, seq_len_3 + 1), results_3[n_seq]['ca3_per_step'],
                  'o-', color=colors[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=5)
    ax3a.set_xlabel("Sequence position")
    ax3a.set_ylabel("CA3 cosine sim to stored")
    ax3a.set_title("Test 3: CA3 Trajectory Quality")
    ax3a.legend(fontsize=8)
    ax3a.set_ylim(-0.1, 1.1)
    ax3a.grid(True, alpha=0.3)

    ax3b = fig.add_subplot(gs[2, 2:4])
    for ci, n_seq in enumerate(n_seq_3):
        ax3b.plot(range(1, seq_len_3 + 1), results_3[n_seq]['readout_per_step'],
                  'o-', color=colors[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=5)
    ax3b.set_xlabel("Sequence position")
    ax3b.set_ylabel("ECDeep cosine sim to EC input")
    ax3b.set_title("Test 3: Readout Quality Per Step")
    ax3b.legend(fontsize=8)
    ax3b.set_ylim(-0.1, 1.1)
    ax3b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 4: Component isolation
    # ------------------------------------------------------------------
    ax4a = fig.add_subplot(gs[3, 0:2])
    component_names = ['CA1 vs\nstellate', 'Sub vs\npyramidal', 'ECDeep vs\nEC input']
    component_vals = [t4['ca1_vs_stel'], t4['sub_vs_pyr'], t4['ecdeep_vs_ec']]
    bars = ax4a.bar(range(3), component_vals, color=['steelblue', 'coral', 'forestgreen'])
    ax4a.set_xticks(range(3))
    ax4a.set_xticklabels(component_names)
    ax4a.set_ylabel("Cosine similarity")
    ax4a.set_title("Test 4: Component Isolation")
    ax4a.set_ylim(0, 1.1)
    ax4a.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, component_vals):
        ax4a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                  f'{val:.3f}', ha='center', fontsize=9)

    ax4b = fig.add_subplot(gs[3, 2:4])
    bins = np.linspace(-0.2, 1.0, 40)
    ax4b.hist(t4['ca1_vs_stel_all'], bins=bins, alpha=0.5, color='steelblue',
              label='CA1 vs stel', density=True)
    ax4b.hist(t4['sub_vs_pyr_all'], bins=bins, alpha=0.5, color='coral',
              label='Sub vs pyr', density=True)
    ax4b.hist(t4['ecdeep_vs_ec_all'], bins=bins, alpha=0.5, color='forestgreen',
              label='ECDeep vs ec', density=True)
    ax4b.set_xlabel("Cosine similarity")
    ax4b.set_ylabel("Density")
    ax4b.set_title("Test 4: Per-Item Similarity Distributions")
    ax4b.legend(fontsize=8)
    ax4b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 5: EC support
    # ------------------------------------------------------------------
    results_5, n_seq_5 = t5

    ax5a = fig.add_subplot(gs[4, 0:2])
    no_ec_vals = [results_5[n]['no_ec'] for n in n_seq_5]
    with_ec_vals = [results_5[n]['with_ec'] for n in n_seq_5]
    gap_vals = [results_5[n]['gap'] for n in n_seq_5]
    ax5a.plot(n_seq_5, no_ec_vals, 'o-', color='steelblue', linewidth=2,
              label='No EC support', markersize=6)
    ax5a.plot(n_seq_5, with_ec_vals, 's-', color='coral', linewidth=2,
              label='With EC support', markersize=6)
    ax5a.plot(n_seq_5, gap_vals, '^-', color='forestgreen', linewidth=2,
              label='Gap (EC benefit)', markersize=6)
    ax5a.set_xlabel("Number of sequences")
    ax5a.set_ylabel("ECDeep cosine sim")
    ax5a.set_title("Test 5: EC Support Effect")
    ax5a.legend(fontsize=8)
    ax5a.set_ylim(-0.1, 1.1)
    ax5a.set_xscale('log')
    ax5a.grid(True, alpha=0.3)

    ax5b = fig.add_subplot(gs[4, 2:4])
    mm_no = [results_5[n]['ca1_mm_no_ec'] for n in n_seq_5]
    mm_ec = [results_5[n]['ca1_mm_with_ec'] for n in n_seq_5]
    ax5b.plot(n_seq_5, mm_no, 'o-', color='steelblue', linewidth=2,
              label='CA1 mismatch (no EC)', markersize=6)
    ax5b.plot(n_seq_5, mm_ec, 's-', color='coral', linewidth=2,
              label='CA1 mismatch (with EC)', markersize=6)
    ax5b.set_xlabel("Number of sequences")
    ax5b.set_ylabel("CA1 mismatch")
    ax5b.set_title("Test 5: CA1 Mismatch With/Without EC")
    ax5b.legend(fontsize=8)
    ax5b.set_xscale('log')
    ax5b.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Output circuit readout tests")
    parser.add_argument('--test', type=int, default=None,
                        help="Run a single test by number (1-5)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dtype = torch.float32
    d_ec = 1000

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
            "weight_decay": 1.0,
            "div_norm_sigma": 0.1,
            "connectivity_prob": 0.33,
            "ltd_rate": 0.00,
            "ltd_ca3_threshold": 0.0,
            "sigma_inh": 25,
            "gamma_inh": 4.0,
            "n_inh_steps": 5,
            "E_inh": -0.4,
        },
        "sub_params": {
            "lr": 0.05,
            "ltd_rate": 0.00,
            "connectivity_prob": 0.33,
        },
        "ec_deep_params": {
            "lr": 1.0,
            "weight_decay": 0.998,
        },
    }

    n_reps_default = 5

    print("=" * 70)
    print("Output Circuit: CA3 -> CA1 -> Sub -> ECDeep")
    print("=" * 70)
    print(f"  d_ec={d_ec}, N_ca3={d_ec}, N_ca1={d_ec}, N_sub={d_ec}")
    print(f"  CA1: error-driven, plateau gating, conductance inhibition")
    print(f"  Sub: error-driven, heterosynaptic LTD, row-norm clipping")
    print(f"  ECDeep: Hebbian, weight_decay=0.998")

    if args.test is not None:
        print(f"\n  Running test {args.test} only")
        if args.test == 1:
            test_learning_curves(hippo_kwargs, device, dtype, n_max_reps=10)
        elif args.test == 2:
            test_readout_capacity(hippo_kwargs, device, dtype, n_reps_default)
        elif args.test == 3:
            test_sequence_readout(hippo_kwargs, device, dtype, n_reps_default)
        elif args.test == 4:
            test_component_isolation(hippo_kwargs, device, dtype, n_reps_default)
        elif args.test == 5:
            test_ec_support(hippo_kwargs, device, dtype, n_reps_default)
    else:
        t1 = test_learning_curves(hippo_kwargs, device, dtype, n_max_reps=100)
        t2 = test_readout_capacity(hippo_kwargs, device, dtype, n_reps_default)
        t3 = test_sequence_readout(hippo_kwargs, device, dtype, n_reps_default)
        t4 = test_component_isolation(hippo_kwargs, device, dtype, n_reps_default)
        t5 = test_ec_support(hippo_kwargs, device, dtype, n_reps_default)

        plot_all_results(t1, t2, t3, t4, t5,
                         save_path="output_circuit_results.png")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
