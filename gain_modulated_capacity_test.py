"""
Gain-Modulated Hippocampal Encoding and Retrieval
==================================================

The distinction between encoding and retrieval is expressed through gain
parameters on three CA3 input sources, not through separate code paths:

  h_ca3 = g_recurrent * (W_recurrent @ x_prev)   # recurrent collaterals
        + g_mf        * (W_mf @ dg_out)           # mossy fibers from DG
        + g_pp        * (W_direct @ stellate)      # perforant path (direct)
        - adaptation

Encoding: g_mf high, g_recurrent ~0   (mossy fibers force CA3 state)
Retrieval cue: g_pp moderate           (direct pathway provides entry point)
Retrieval run: g_recurrent high        (successor map drives transitions)

Tests:
  1. Baseline verification: reproduce previous results with extreme gains
  2. Retrieval cue gain sweep: does mixing in mossy fiber drive at the
     cue step improve retrieval?
  3. Encoding recurrent gain sweep: does partial recurrent drive during
     encoding help or hurt storage quality?
  4. Gain mixing at the cue step: which combination of (g_pp, g_mf)
     gives the best initial cue at varying capacity?
  5. Theta-phase simulation: alternating encode and retrieve within a
     sequence to see if the system can verify its own learning on-the-fly
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
# CIRCUIT COMPONENTS
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


class CA1:
    def __init__(self, N_ca1, N_ca3, d_ec, lr=0.3, weight_decay=0.998, k_active=50,
                 device='cpu', dtype=torch.float32):
        self.N_ca1 = N_ca1
        self.lr = lr
        self.weight_decay = weight_decay
        self.k_active = k_active
        self.W_sc = torch.zeros((N_ca1, N_ca3), device=device, dtype=dtype)
        self.n_episodes = 0

    def encode(self, x_ca3, x_ec_stel):
        self.W_sc += self.lr * torch.outer(x_ec_stel, x_ca3)
        self.W_sc *= self.weight_decay
        self.n_episodes += 1

    def retrieve(self, x_ca3, x_ec_stel=None):
        h_sc = torch.relu(self.W_sc @ x_ca3)
        h_out = apply_kwta(h_sc, self.k_active)
        mismatch = 0.0
        if x_ec_stel is not None:
            error = x_ec_stel - h_out
            mismatch = float(torch.linalg.norm(error) / (torch.linalg.norm(x_ec_stel) + 1e-10))
        return h_out, mismatch


class Subiculum:
    def __init__(self, N_sub, N_ca1, d_ec, lr=1.0, k_active=500,
                 device='cpu', dtype=torch.float32):
        self.N_sub = N_sub
        self.lr = lr
        self.k_active = k_active
        self.W_ca1 = torch.zeros((N_sub, N_ca1), device=device, dtype=dtype)
        self.n_episodes = 0

    def encode(self, ca1_output, ec_pyramidal):
        self.W_ca1 += self.lr * torch.outer(ec_pyramidal, ca1_output)
        self.n_episodes += 1

    def replay(self, ca1_output):
        h = torch.relu(self.W_ca1 @ ca1_output)
        return apply_kwta(h, self.k_active)


# =============================================================================
# CA3: Gain-Modulated Single-Step Dynamics
# =============================================================================

class CA3GainModulated:
    """
    CA3 as a single dynamical system. Three input sources compete
    through gain modulation:

      h = g_rec * (W @ x_prev) + g_mf * mf_input + g_pp * pp_input - adaptation
      x = kwta(relu(h), k)

    Internal state (x_prev, adaptation) persists across steps.
    Learning (successor associations) happens when learn=True.
    """
    def __init__(self, N, k_active, lr=1.0, device='cpu', dtype=torch.float32):
        self.N = N
        self.k_active = k_active
        self.lr = lr
        self.device = device
        self.dtype = dtype

        self.W = torch.zeros((N, N), device=device, dtype=dtype)
        self.n_stored = 0
        self.mean_activity = torch.zeros(N, device=device, dtype=dtype)

        # Persistent dynamical state
        self.x_prev = None
        self.adaptation = torch.zeros(N, device=device, dtype=dtype)

    def reset_state(self):
        """Clear dynamical state between sequences or before retrieval."""
        self.x_prev = None
        self.adaptation = torch.zeros(self.N, device=self.device, dtype=self.dtype)

    def step(self, mf_input=None, pp_input=None,
             g_recurrent=1.0, g_mf=1.0, g_pp=1.0,
             adapt_rate=0.0, adapt_decay=0.0,
             learn=False):
        """
        One step of CA3 dynamics.

        Inputs (all in CA3 space, pre-processed by caller):
          mf_input:  mossy fiber drive (from W_mf @ dg_out)
          pp_input:  perforant path drive (from W_direct @ stellate)

        Gains control the regime:
          g_recurrent: weight on recurrent collaterals (W @ x_prev)
          g_mf:        weight on mossy fiber input
          g_pp:        weight on perforant path input

        Returns the new CA3 state (normalized, sparse).
        """
        h = torch.zeros(self.N, device=self.device, dtype=self.dtype)

        # Recurrent drive from previous state
        if self.x_prev is not None and g_recurrent > 0:
            h = h + g_recurrent * (self.W @ self.x_prev)

        # Mossy fiber drive
        if mf_input is not None and g_mf > 0:
            h = h + g_mf * mf_input

        # Perforant path (direct) drive
        if pp_input is not None and g_pp > 0:
            h = h + g_pp * pp_input

        # Subtract adaptation, apply nonlinearity, compete
        h = torch.relu(h - self.adaptation)
        x_new = apply_kwta(h, self.k_active)
        norm = torch.linalg.norm(x_new)
        if norm > 1e-10:
            x_new = x_new / norm
        else:
            x_new = torch.zeros(self.N, device=self.device, dtype=self.dtype)

        # Learn successor association: x_prev -> x_new
        if learn:
            self.n_stored += 1
            self.mean_activity += (x_new - self.mean_activity) / self.n_stored

            if self.x_prev is not None:
                curr_c = x_new - self.mean_activity
                prev_c = self.x_prev - self.mean_activity
                self.W += self.lr * torch.outer(curr_c, prev_c)
                self.W.fill_diagonal_(0)

        # Update dynamical state
        self.adaptation = adapt_decay * self.adaptation + adapt_rate * x_new
        self.x_prev = x_new.clone()

        return x_new


# =============================================================================
# HIPPOCAMPAL SYSTEM: Gain-Modulated
# =============================================================================

class HippocampalSystem:
    """
    Full hippocampal circuit with gain-modulated CA3.

    The same step() method handles encoding and retrieval. The gain
    parameters determine which input sources dominate CA3.

    Default gain profiles:
      encoding:       g_mf=1.0, g_recurrent=0.0, g_pp=0.0
      retrieval cue:  g_mf=0.0, g_recurrent=0.0, g_pp=1.0
      retrieval run:  g_mf=0.0, g_recurrent=1.0, g_pp=0.0
    """
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3, ca3_lr=1.0,
                 direct_lr=0.3, direct_decay=0.998,
                 mf_connectivity_prob=0.33,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, N_sub=1000,
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

        # Mossy fiber projection: fixed random, DG -> CA3
        self.W_mf = make_feedforward_weights(
            N_ca3, D_dg, mf_connectivity_prob, device, dtype)

        # Direct pathway: EC stellate -> CA3 space (learned)
        self.W_direct = torch.zeros((N_ca3, d_ec), device=device, dtype=dtype)
        self.direct_lr = direct_lr
        self.direct_decay = direct_decay

    def begin_sequence(self):
        self.ca3.reset_state()

    def end_sequence(self):
        self.ca3.reset_state()

    def step(self, ec_input=None, learn=False,
             g_mf=0.0, g_recurrent=1.0, g_pp=0.0,
             adapt_rate=0.0, adapt_decay=0.0):
        """
        One step of the full hippocampal circuit.

        If ec_input is provided, it drives EC_sup, which feeds DG and
        the direct pathway. If ec_input is None, only recurrent
        dynamics drive CA3 (free-running retrieval).

        Args:
            ec_input:     cortical input (or None for free-running)
            learn:        whether to update synapses this step
            g_mf:         gain on mossy fiber pathway
            g_recurrent:  gain on CA3 recurrent collaterals
            g_pp:         gain on perforant path (direct EC -> CA3)
            adapt_rate:   adaptation accumulation rate
            adapt_decay:  adaptation decay rate

        Returns:
            ca3_state: the CA3 population state for this step
        """
        mf_input = None
        pp_input = None
        stellate = None
        pyramidal = None

        if ec_input is not None:
            stellate, pyramidal = self.ec_sup.forward(ec_input)
            if g_mf > 0:
                dg_out = self.dg.forward(stellate)
                mf_input = torch.relu(self.W_mf @ dg_out)
            if g_pp > 0:
                pp_input = torch.relu(self.W_direct @ stellate)

        ca3_state = self.ca3.step(
            mf_input=mf_input, pp_input=pp_input,
            g_recurrent=g_recurrent, g_mf=g_mf, g_pp=g_pp,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay,
            learn=learn)

        # Direct pathway learning: associate EC stellate with CA3 state
        if learn and stellate is not None:
            stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
            self.W_direct += self.direct_lr * torch.outer(ca3_state, stel_norm)
            self.W_direct *= self.direct_decay

        # CA1 and Sub learning
        if learn and stellate is not None:
            self.ca1.encode(ca3_state, stellate)
            ca1_out, _ = self.ca1.retrieve(ca3_state, stellate)
            self.sub.encode(ca1_out, pyramidal)

        return ca3_state


# =============================================================================
# CONVENIENCE FUNCTIONS
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


def encode_sequences(hippo, sequences, n_repetitions=1,
                     g_mf=1.0, g_recurrent=0.0, g_pp=0.0,
                     adapt_rate=0.0, adapt_decay=0.0):
    """Encode sequences, return CA3 patterns from first repetition."""
    all_ca3_patterns = [[] for _ in sequences]
    for rep in range(n_repetitions):
        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for ec_pattern in seq:
                ca3_state = hippo.step(
                    ec_pattern, learn=True,
                    g_mf=g_mf, g_recurrent=g_recurrent, g_pp=g_pp,
                    adapt_rate=adapt_rate, adapt_decay=adapt_decay)
                if rep == 0:
                    all_ca3_patterns[seq_idx].append(ca3_state.clone())
            hippo.end_sequence()
    return all_ca3_patterns


def recall_sequence(hippo, ec_input, n_steps,
                    cue_g_mf=0.0, cue_g_pp=1.0,
                    run_g_recurrent=1.0,
                    adapt_rate=0.0, adapt_decay=0.0):
    """Cue from ec_input, then free-run CA3 dynamics."""
    hippo.ca3.reset_state()

    # Cue step: form initial CA3 state
    ca3_state = hippo.step(
        ec_input, learn=False,
        g_mf=cue_g_mf, g_recurrent=0.0, g_pp=cue_g_pp,
        adapt_rate=adapt_rate, adapt_decay=adapt_decay)
    trajectory = [ca3_state.clone()]

    # Free-running: recurrent dynamics only
    for _ in range(n_steps - 1):
        ca3_state = hippo.step(
            None, learn=False,
            g_mf=0.0, g_recurrent=run_g_recurrent, g_pp=0.0,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)
        trajectory.append(ca3_state.clone())

    return trajectory


def compute_similarity_matrix(trajectory, reference_patterns):
    T = len(trajectory)
    R = len(reference_patterns)
    sim = np.zeros((T, R))
    for t in range(T):
        for r in range(R):
            sim[t, r] = cosine_sim(trajectory[t], reference_patterns[r])
    return sim


def measure_sequentiality(sim_matrix):
    n_steps = sim_matrix.shape[0]
    peaks = [int(np.argmax(sim_matrix[t, :])) for t in range(n_steps)]
    correct = sum(1.0 for t in range(n_steps) if peaks[t] == t)
    return correct / n_steps, peaks


def measure_diagonal(sim_matrix):
    n = min(sim_matrix.shape)
    return np.mean([sim_matrix[t, t] for t in range(n)])


# =============================================================================
# TEST 1: Baseline Verification (Capacity Sweep)
# =============================================================================

def test_baseline_capacity(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Capacity sweep with default encoding/retrieval gains.
    Verifies the gain-modulated architecture works at all.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Baseline Capacity Sweep")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [1, 2, 5, 10, 20, 50, 100, 200]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=1000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

        ca3_patterns = encode_sequences(
            hippo, sequences, n_repetitions,
            g_mf=1.0, g_recurrent=0.0, g_pp=0.0)

        seq_scores = []
        diag_scores = []
        first_sims = []

        for si in range(n_seq):
            traj = recall_sequence(
                hippo, sequences[si][0], n_steps=seq_length,
                cue_g_mf=0.0, cue_g_pp=1.0,
                run_g_recurrent=1.0)
            sim = compute_similarity_matrix(traj, ca3_patterns[si])
            s, _ = measure_sequentiality(sim)
            d = measure_diagonal(sim)
            seq_scores.append(s)
            diag_scores.append(d)
            first_sims.append(sim[0, 0])

        results[n_seq] = {
            'mean_seq': np.mean(seq_scores),
            'mean_diag': np.mean(diag_scores),
            'mean_first': np.mean(first_sims),
            'pct_perfect': np.mean([1.0 if s == 1.0 else 0.0 for s in seq_scores]),
        }

        print(f"  n_seq={n_seq:4d}: seq={results[n_seq]['mean_seq']:.3f}, "
              f"diag={results[n_seq]['mean_diag']:.4f}, "
              f"first={results[n_seq]['mean_first']:.4f}, "
              f"perfect={results[n_seq]['pct_perfect']:.1%}")

    return results, n_seq_values


# =============================================================================
# TEST 2: Retrieval Cue Gain Sweep (g_mf at cue time)
# =============================================================================

def test_retrieval_cue_gain(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    During the cue step of retrieval, vary the mossy fiber gain.
    g_pp is always 1.0 (direct pathway provides the cue).
    Question: does adding some DG->MF drive alongside the direct
    pathway improve the initial cue quality?
    """
    print("\n" + "=" * 70)
    print("TEST 2: Retrieval Cue g_mf Sweep")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [5, 20, 50, 100]
    cue_g_mf_values = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=2000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)
        ca3_patterns = encode_sequences(
            hippo, sequences, n_repetitions,
            g_mf=1.0, g_recurrent=0.0, g_pp=0.0)

        sample_n = min(n_seq, 20)

        for cue_g_mf in cue_g_mf_values:
            seq_scores = []
            diag_scores = []
            first_sims = []

            for si in range(sample_n):
                traj = recall_sequence(
                    hippo, sequences[si][0], n_steps=seq_length,
                    cue_g_mf=cue_g_mf, cue_g_pp=1.0,
                    run_g_recurrent=1.0)
                sim = compute_similarity_matrix(traj, ca3_patterns[si])
                s, _ = measure_sequentiality(sim)
                d = measure_diagonal(sim)
                seq_scores.append(s)
                diag_scores.append(d)
                first_sims.append(sim[0, 0])

            results[(n_seq, cue_g_mf)] = {
                'mean_seq': np.mean(seq_scores),
                'mean_diag': np.mean(diag_scores),
                'mean_first': np.mean(first_sims),
            }

            print(f"  n_seq={n_seq:4d}, cue_g_mf={cue_g_mf:.1f}: "
                  f"seq={results[(n_seq, cue_g_mf)]['mean_seq']:.3f}, "
                  f"diag={results[(n_seq, cue_g_mf)]['mean_diag']:.4f}, "
                  f"first={results[(n_seq, cue_g_mf)]['mean_first']:.4f}")

    return results, n_seq_values, cue_g_mf_values


# =============================================================================
# TEST 3: Encoding Recurrent Gain Sweep
# =============================================================================

def test_encoding_recurrent_gain(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    During encoding, vary g_recurrent while keeping g_mf=1.0.
    At g_recurrent=0: pure mossy fiber forcing (default).
    At g_recurrent>0: CA3 recurrent dynamics partially contribute to
    the CA3 state during encoding. Does this help or hurt?
    """
    print("\n" + "=" * 70)
    print("TEST 3: Encoding g_recurrent Sweep")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [5, 20, 50, 100]
    enc_g_rec_values = [0.0, 0.1, 0.3, 0.5, 1.0]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=3000 + n_seq)

        for enc_g_rec in enc_g_rec_values:
            torch.manual_seed(42)
            hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

            ca3_patterns = encode_sequences(
                hippo, sequences, n_repetitions,
                g_mf=1.0, g_recurrent=enc_g_rec, g_pp=0.0)

            sample_n = min(n_seq, 20)
            seq_scores = []
            diag_scores = []

            for si in range(sample_n):
                traj = recall_sequence(
                    hippo, sequences[si][0], n_steps=seq_length,
                    cue_g_mf=0.0, cue_g_pp=1.0,
                    run_g_recurrent=1.0)
                sim = compute_similarity_matrix(traj, ca3_patterns[si])
                s, _ = measure_sequentiality(sim)
                d = measure_diagonal(sim)
                seq_scores.append(s)
                diag_scores.append(d)

            results[(n_seq, enc_g_rec)] = {
                'mean_seq': np.mean(seq_scores),
                'mean_diag': np.mean(diag_scores),
            }

            print(f"  n_seq={n_seq:4d}, enc_g_rec={enc_g_rec:.1f}: "
                  f"seq={results[(n_seq, enc_g_rec)]['mean_seq']:.3f}, "
                  f"diag={results[(n_seq, enc_g_rec)]['mean_diag']:.4f}")

    return results, n_seq_values, enc_g_rec_values


# =============================================================================
# TEST 4: Cue Gain Mixing (g_pp vs g_mf at cue time)
# =============================================================================

def test_cue_gain_mixing(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    At the cue step, test a grid of (g_pp, g_mf) combinations.
    This reveals whether the two pathways are redundant, complementary,
    or interfering when forming the initial CA3 cue.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Cue Gain Mixing (g_pp x g_mf)")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [10, 50]
    g_values = [0.0, 0.25, 0.5, 1.0, 2.0]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=4000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)
        ca3_patterns = encode_sequences(
            hippo, sequences, n_repetitions,
            g_mf=1.0, g_recurrent=0.0, g_pp=0.0)

        sample_n = min(n_seq, 20)

        for g_pp in g_values:
            for g_mf in g_values:
                if g_pp == 0.0 and g_mf == 0.0:
                    # No cue at all, skip
                    results[(n_seq, g_pp, g_mf)] = {
                        'mean_seq': 0.0, 'mean_diag': 0.0}
                    continue

                seq_scores = []
                diag_scores = []

                for si in range(sample_n):
                    traj = recall_sequence(
                        hippo, sequences[si][0], n_steps=seq_length,
                        cue_g_mf=g_mf, cue_g_pp=g_pp,
                        run_g_recurrent=1.0)
                    sim = compute_similarity_matrix(traj, ca3_patterns[si])
                    s, _ = measure_sequentiality(sim)
                    d = measure_diagonal(sim)
                    seq_scores.append(s)
                    diag_scores.append(d)

                results[(n_seq, g_pp, g_mf)] = {
                    'mean_seq': np.mean(seq_scores),
                    'mean_diag': np.mean(diag_scores),
                }

        # Print best combination for each n_seq
        best_key = max(
            [(n_seq, gp, gm) for gp in g_values for gm in g_values],
            key=lambda k: results[k]['mean_diag'])
        print(f"  n_seq={n_seq}: best g_pp={best_key[1]:.2f}, "
              f"g_mf={best_key[2]:.2f} -> "
              f"seq={results[best_key]['mean_seq']:.3f}, "
              f"diag={results[best_key]['mean_diag']:.4f}")

        # Print full grid
        for g_pp in g_values:
            row = [f"{results[(n_seq, g_pp, gm)]['mean_diag']:.3f}"
                   for gm in g_values]
            print(f"    g_pp={g_pp:.2f}: [{', '.join(row)}]")

    return results, n_seq_values, g_values


# =============================================================================
# TEST 5: Theta-Phase Simulation
# =============================================================================

def test_theta_phase(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Simulate theta-phase alternation within a sequence:
      - Encoding phase (trough): present item N, learn with g_mf high
      - Retrieval phase (peak): suppress mossy fibers, let recurrent
        dynamics run 1 step and check if the network predicts item N
        from the state at item N-1

    This tests whether the system can verify its own learning in real
    time as a sequence is being encoded.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Theta-Phase Encode/Retrieve Alternation")
    print("=" * 70)

    seq_length = 8
    n_seq_values = [5, 20, 50]

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=5000 + n_seq)

        # --- Standard encoding (no theta check) ---
        torch.manual_seed(42)
        hippo_standard = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)
        ca3_patterns_standard = encode_sequences(
            hippo_standard, sequences, n_repetitions,
            g_mf=1.0, g_recurrent=0.0, g_pp=0.0)

        # --- Theta-phase encoding ---
        torch.manual_seed(42)
        hippo_theta = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

        # Track per-step retrieval accuracy during encoding
        all_verify_sims = [[] for _ in sequences]
        all_ca3_patterns_theta = [[] for _ in sequences]

        for rep in range(n_repetitions):
            for seq_idx, seq in enumerate(sequences):
                hippo_theta.begin_sequence()
                prev_ca3_state = None
                for step_idx, ec_pattern in enumerate(seq):
                    # Encoding phase: mossy fibers force CA3
                    ca3_state = hippo_theta.step(
                        ec_pattern, learn=True,
                        g_mf=1.0, g_recurrent=0.0, g_pp=0.0)

                    if rep == 0:
                        all_ca3_patterns_theta[seq_idx].append(ca3_state.clone())

                    # Retrieval phase: verify that the just-learned association
                    # works. After encoding item N, check W @ state(N-1) ~ state(N).
                    # This confirms the successor link from N-1 to N is readable.
                    if prev_ca3_state is not None and rep == n_repetitions - 1:
                        # Save CA3 dynamical state
                        saved_prev = hippo_theta.ca3.x_prev.clone()
                        saved_adapt = hippo_theta.ca3.adaptation.clone()

                        # Rewind to state(N-1) and run one recurrent step
                        hippo_theta.ca3.x_prev = prev_ca3_state.clone()
                        hippo_theta.ca3.adaptation = torch.zeros_like(saved_adapt)

                        probe = hippo_theta.step(
                            None, learn=False,
                            g_mf=0.0, g_recurrent=1.0, g_pp=0.0)
                        verify_sim = cosine_sim(probe, ca3_state)

                        # Restore state so encoding continues normally
                        hippo_theta.ca3.x_prev = saved_prev
                        hippo_theta.ca3.adaptation = saved_adapt

                        all_verify_sims[seq_idx].append(verify_sim)

                    prev_ca3_state = ca3_state.clone()

                hippo_theta.end_sequence()

        # Measure final retrieval quality for both
        sample_n = min(n_seq, 20)

        standard_diags = []
        theta_diags = []
        for si in range(sample_n):
            traj_s = recall_sequence(
                hippo_standard, sequences[si][0], n_steps=seq_length,
                cue_g_mf=0.0, cue_g_pp=1.0, run_g_recurrent=1.0)
            sim_s = compute_similarity_matrix(traj_s, ca3_patterns_standard[si])
            standard_diags.append(measure_diagonal(sim_s))

            traj_t = recall_sequence(
                hippo_theta, sequences[si][0], n_steps=seq_length,
                cue_g_mf=0.0, cue_g_pp=1.0, run_g_recurrent=1.0)
            sim_t = compute_similarity_matrix(traj_t, all_ca3_patterns_theta[si])
            theta_diags.append(measure_diagonal(sim_t))

        # Per-position verification accuracy (averaged across sequences)
        mean_verify_per_pos = []
        for pos in range(seq_length - 1):
            sims_at_pos = [all_verify_sims[si][pos]
                           for si in range(n_seq)
                           if pos < len(all_verify_sims[si])]
            mean_verify_per_pos.append(np.mean(sims_at_pos) if sims_at_pos else 0.0)

        results[n_seq] = {
            'standard_diag': np.mean(standard_diags),
            'theta_diag': np.mean(theta_diags),
            'verify_per_pos': mean_verify_per_pos,
            'verify_mean': np.mean(mean_verify_per_pos),
        }

        verify_str = " ".join(f"{v:.3f}" for v in mean_verify_per_pos)
        print(f"  n_seq={n_seq:4d}: standard_diag={results[n_seq]['standard_diag']:.4f}, "
              f"theta_diag={results[n_seq]['theta_diag']:.4f}")
        print(f"  {'':>12s}  verify per pos: [{verify_str}]")
        print(f"  {'':>12s}  verify mean: {results[n_seq]['verify_mean']:.4f}")

    return results, n_seq_values, seq_length


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_results(t1, t2, t3, t4, t5, save_path):
    fig = plt.figure(figsize=(24, 30))
    gs = GridSpec(5, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Gain-Modulated Hippocampal Encoding/Retrieval",
                 fontsize=16, fontweight='bold', y=0.995)

    # ------------------------------------------------------------------
    # Row 1: Baseline capacity sweep
    # ------------------------------------------------------------------
    results_1, n_seq_1 = t1

    ax1a = fig.add_subplot(gs[0, 0:2])
    seq_scores = [results_1[n]['mean_seq'] for n in n_seq_1]
    diag_scores = [results_1[n]['mean_diag'] for n in n_seq_1]
    pct_perf = [results_1[n]['pct_perfect'] for n in n_seq_1]
    ax1a.plot(n_seq_1, seq_scores, 'o-', color='steelblue', linewidth=2,
              label='Sequentiality', markersize=6)
    ax1a.plot(n_seq_1, diag_scores, 's-', color='coral', linewidth=2,
              label='Diag sim', markersize=6)
    ax1a.plot(n_seq_1, pct_perf, '^-', color='forestgreen', linewidth=2,
              label='% perfect', markersize=6)
    ax1a.set_xlabel("Number of sequences")
    ax1a.set_ylabel("Score")
    ax1a.set_title("Test 1: Baseline Capacity")
    ax1a.legend(fontsize=8)
    ax1a.set_ylim(-0.1, 1.1)
    ax1a.set_xscale('log')
    ax1a.grid(True, alpha=0.3)

    ax1b = fig.add_subplot(gs[0, 2:4])
    first_sims = [results_1[n]['mean_first'] for n in n_seq_1]
    ax1b.plot(n_seq_1, first_sims, 'o-', color='purple', linewidth=2,
              label='Cue step sim (step 0)', markersize=6)
    ax1b.plot(n_seq_1, diag_scores, 's-', color='coral', linewidth=2,
              label='Full diag sim', markersize=6)
    ax1b.set_xlabel("Number of sequences")
    ax1b.set_ylabel("Cosine similarity")
    ax1b.set_title("Test 1: Cue Quality vs Full Retrieval")
    ax1b.legend(fontsize=8)
    ax1b.set_ylim(-0.1, 1.1)
    ax1b.set_xscale('log')
    ax1b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 2: Retrieval cue g_mf sweep
    # ------------------------------------------------------------------
    results_2, n_seq_2, g_mf_2 = t2

    colors_cap = ['steelblue', 'coral', 'forestgreen', 'purple']
    ax2a = fig.add_subplot(gs[1, 0:2])
    for ci, n_seq in enumerate(n_seq_2):
        y = [results_2[(n_seq, g)]['mean_seq'] for g in g_mf_2]
        ax2a.plot(g_mf_2, y, 'o-', color=colors_cap[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=5)
    ax2a.set_xlabel("Cue g_mf")
    ax2a.set_ylabel("Sequentiality")
    ax2a.set_title("Test 2: Retrieval Cue g_mf (sequentiality)")
    ax2a.legend(fontsize=8)
    ax2a.set_ylim(-0.1, 1.1)
    ax2a.grid(True, alpha=0.3)

    ax2b = fig.add_subplot(gs[1, 2:4])
    for ci, n_seq in enumerate(n_seq_2):
        y = [results_2[(n_seq, g)]['mean_first'] for g in g_mf_2]
        ax2b.plot(g_mf_2, y, 'o-', color=colors_cap[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=5)
    ax2b.set_xlabel("Cue g_mf")
    ax2b.set_ylabel("Cue step similarity")
    ax2b.set_title("Test 2: Cue Quality vs g_mf")
    ax2b.legend(fontsize=8)
    ax2b.set_ylim(-0.1, 1.1)
    ax2b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 3: Encoding g_recurrent sweep
    # ------------------------------------------------------------------
    results_3, n_seq_3, g_rec_3 = t3

    ax3a = fig.add_subplot(gs[2, 0:2])
    for ci, n_seq in enumerate(n_seq_3):
        y = [results_3[(n_seq, g)]['mean_seq'] for g in g_rec_3]
        ax3a.plot(g_rec_3, y, 'o-', color=colors_cap[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=5)
    ax3a.set_xlabel("Encoding g_recurrent")
    ax3a.set_ylabel("Sequentiality")
    ax3a.set_title("Test 3: Encoding g_recurrent (sequentiality)")
    ax3a.legend(fontsize=8)
    ax3a.set_ylim(-0.1, 1.1)
    ax3a.grid(True, alpha=0.3)

    ax3b = fig.add_subplot(gs[2, 2:4])
    for ci, n_seq in enumerate(n_seq_3):
        y = [results_3[(n_seq, g)]['mean_diag'] for g in g_rec_3]
        ax3b.plot(g_rec_3, y, 'o-', color=colors_cap[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=5)
    ax3b.set_xlabel("Encoding g_recurrent")
    ax3b.set_ylabel("Diagonal similarity")
    ax3b.set_title("Test 3: Encoding g_recurrent (diag sim)")
    ax3b.legend(fontsize=8)
    ax3b.set_ylim(-0.1, 1.1)
    ax3b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 4: Cue gain mixing heatmaps
    # ------------------------------------------------------------------
    results_4, n_seq_4, g_vals_4 = t4

    for ni, n_seq in enumerate(n_seq_4):
        # Sequentiality heatmap
        ax = fig.add_subplot(gs[3, ni * 2])
        grid = np.zeros((len(g_vals_4), len(g_vals_4)))
        for i, g_pp in enumerate(g_vals_4):
            for j, g_mf in enumerate(g_vals_4):
                grid[i, j] = results_4[(n_seq, g_pp, g_mf)]['mean_seq']
        im = ax.imshow(grid, aspect='equal', cmap='viridis', vmin=0, vmax=1,
                       origin='lower')
        ax.set_xticks(range(len(g_vals_4)))
        ax.set_xticklabels([f"{g:.2f}" for g in g_vals_4], fontsize=7)
        ax.set_yticks(range(len(g_vals_4)))
        ax.set_yticklabels([f"{g:.2f}" for g in g_vals_4], fontsize=7)
        ax.set_xlabel("Cue g_mf")
        ax.set_ylabel("Cue g_pp")
        ax.set_title(f"Test 4: Seq, n={n_seq}")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Diagonal sim heatmap
        ax2 = fig.add_subplot(gs[3, ni * 2 + 1])
        grid2 = np.zeros((len(g_vals_4), len(g_vals_4)))
        for i, g_pp in enumerate(g_vals_4):
            for j, g_mf in enumerate(g_vals_4):
                grid2[i, j] = results_4[(n_seq, g_pp, g_mf)]['mean_diag']
        im2 = ax2.imshow(grid2, aspect='equal', cmap='viridis', vmin=0, vmax=1,
                         origin='lower')
        ax2.set_xticks(range(len(g_vals_4)))
        ax2.set_xticklabels([f"{g:.2f}" for g in g_vals_4], fontsize=7)
        ax2.set_yticks(range(len(g_vals_4)))
        ax2.set_yticklabels([f"{g:.2f}" for g in g_vals_4], fontsize=7)
        ax2.set_xlabel("Cue g_mf")
        ax2.set_ylabel("Cue g_pp")
        ax2.set_title(f"Test 4: Diag, n={n_seq}")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

    # ------------------------------------------------------------------
    # Row 5: Theta phase
    # ------------------------------------------------------------------
    results_5, n_seq_5, seq_len_5 = t5

    ax5a = fig.add_subplot(gs[4, 0:2])
    for ci, n_seq in enumerate(n_seq_5):
        verify = results_5[n_seq]['verify_per_pos']
        ax5a.plot(range(1, len(verify) + 1), verify, 'o-',
                  color=colors_cap[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=5)
    ax5a.set_xlabel("Sequence position")
    ax5a.set_ylabel("Verification similarity")
    ax5a.set_title("Test 5: Theta-Phase Verification During Encoding")
    ax5a.legend(fontsize=8)
    ax5a.set_ylim(-0.1, 1.1)
    ax5a.grid(True, alpha=0.3)

    ax5b = fig.add_subplot(gs[4, 2:4])
    n_labels = [str(n) for n in n_seq_5]
    standard = [results_5[n]['standard_diag'] for n in n_seq_5]
    theta = [results_5[n]['theta_diag'] for n in n_seq_5]
    verify = [results_5[n]['verify_mean'] for n in n_seq_5]
    x_pos = np.arange(len(n_seq_5))
    width = 0.25
    ax5b.bar(x_pos - width, standard, width, color='steelblue', label='Standard retrieval')
    ax5b.bar(x_pos, theta, width, color='coral', label='Theta retrieval')
    ax5b.bar(x_pos + width, verify, width, color='forestgreen', label='Theta verify (mean)')
    ax5b.set_xticks(x_pos)
    ax5b.set_xticklabels(n_labels)
    ax5b.set_xlabel("Number of sequences")
    ax5b.set_ylabel("Mean diagonal similarity")
    ax5b.set_title("Test 5: Standard vs Theta Encoding")
    ax5b.legend(fontsize=8)
    ax5b.set_ylim(0, 1.1)
    ax5b.grid(True, alpha=0.3, axis='y')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {save_path}")


def plot_theta_results(t5, save_path):
    """Standalone plot for theta-phase test only."""
    results_5, n_seq_5, seq_len_5 = t5
    colors_cap = ['steelblue', 'coral', 'forestgreen', 'purple']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Theta-Phase Encode/Retrieve Alternation",
                 fontsize=14, fontweight='bold')

    ax1 = axes[0]
    for ci, n_seq in enumerate(n_seq_5):
        verify = results_5[n_seq]['verify_per_pos']
        ax1.plot(range(1, len(verify) + 1), verify, 'o-',
                 color=colors_cap[ci], linewidth=2,
                 label=f'n_seq={n_seq}', markersize=5)
    ax1.set_xlabel("Sequence position")
    ax1.set_ylabel("Verification similarity (W @ state(N-1) vs state(N))")
    ax1.set_title("Per-Position Verification During Encoding")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    n_labels = [str(n) for n in n_seq_5]
    standard = [results_5[n]['standard_diag'] for n in n_seq_5]
    theta = [results_5[n]['theta_diag'] for n in n_seq_5]
    verify = [results_5[n]['verify_mean'] for n in n_seq_5]
    x_pos = np.arange(len(n_seq_5))
    width = 0.25
    ax2.bar(x_pos - width, standard, width, color='steelblue', label='Standard retrieval')
    ax2.bar(x_pos, theta, width, color='coral', label='Theta retrieval')
    ax2.bar(x_pos + width, verify, width, color='forestgreen', label='Theta verify (mean)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(n_labels)
    ax2.set_xlabel("Number of sequences")
    ax2.set_ylabel("Mean diagonal similarity")
    ax2.set_title("Standard vs Theta Encoding")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gain-modulated hippocampal capacity tests")
    parser.add_argument('--theta-only', action='store_true',
                        help="Run only the theta-phase test (Test 5)")
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
        "ca1_params": {"lr": 50.0, "weight_decay": 1.000, "k_active": 50},
        "sub_params": {"lr": 1.0, "k_active": 500},
    }

    n_repetitions = 3

    print("=" * 70)
    print("Gain-Modulated Hippocampal Encoding/Retrieval")
    print("=" * 70)
    print(f"  d_ec={d_ec}, D_dg={d_ec}, N_ca3={d_ec}")
    print(f"  k_ca3=50, mf_connectivity=0.33")
    print(f"  Encoding repetitions: {n_repetitions}")
    print(f"  Default gains:")
    print(f"    Encode:   g_mf=1.0, g_rec=0.0, g_pp=0.0")
    print(f"    Cue:      g_mf=0.0, g_rec=0.0, g_pp=1.0")
    print(f"    Free-run: g_mf=0.0, g_rec=1.0, g_pp=0.0")

    # Determine which tests to run
    run_test = args.test
    if args.theta_only:
        run_test = 5

    if run_test is not None:
        print(f"\n  Running test {run_test} only")
        if run_test == 1:
            t1 = test_baseline_capacity(hippo_kwargs, device, dtype, n_repetitions)
        elif run_test == 2:
            t2 = test_retrieval_cue_gain(hippo_kwargs, device, dtype, n_repetitions)
        elif run_test == 3:
            t3 = test_encoding_recurrent_gain(hippo_kwargs, device, dtype, n_repetitions)
        elif run_test == 4:
            t4 = test_cue_gain_mixing(hippo_kwargs, device, dtype, n_repetitions)
        elif run_test == 5:
            t5 = test_theta_phase(hippo_kwargs, device, dtype, n_repetitions)
            plot_theta_results(t5, save_path="theta_phase_results.png")
        else:
            print(f"  Unknown test number: {run_test}")
    else:
        t1 = test_baseline_capacity(hippo_kwargs, device, dtype, n_repetitions)
        t2 = test_retrieval_cue_gain(hippo_kwargs, device, dtype, n_repetitions)
        t3 = test_encoding_recurrent_gain(hippo_kwargs, device, dtype, n_repetitions)
        t4 = test_cue_gain_mixing(hippo_kwargs, device, dtype, n_repetitions)
        t5 = test_theta_phase(hippo_kwargs, device, dtype, n_repetitions)

        plot_all_results(t1, t2, t3, t4, t5,
                         save_path="gain_modulated_results.png")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
