"""
Mossy Fiber Projection Capacity Test
=====================================

Tests the architectural change where DG and CA3 have distinct representational
spaces connected by a fixed random mossy fiber projection (W_mf).

Key changes from capacity_limits_test.py:
  1. W_mf: fixed random projection from DG -> CA3 space
     During encoding, mossy fibers force CA3 state = W_mf @ dg_out (no recurrent dynamics)
  2. Unified CA3.retrieve: one dynamical process, n_steps controls how far you go.
     No separate "attractor" vs "sequence" modes.
  3. CA3 successor associations are stored between CA3-space patterns, not DG patterns
  4. CA1 Schaffer collaterals see CA3-space patterns, not DG patterns
  5. Direct pathway (EC -> CA3) and DG pathway (EC -> DG -> W_mf -> CA3) both
     produce cues in CA3 space

Tests:
  1. Capacity sweep: old architecture vs new, head-to-head
  2. Representational analysis: DG vs CA3 pattern statistics
  3. Pathway comparison: direct vs DG vs combined at varying capacity
  4. Mid-sequence entry under load
  5. Unified retrieve dynamics: how retrieval quality varies with n_steps
"""

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
# CA3: Unified Temporal Dynamics
# =============================================================================

class CA3Temporal:
    """
    CA3 with a single dynamical process. The weight matrix W stores
    hetero-associative successor mappings. Retrieval is always the same
    loop: W @ x, k-WTA, normalize, adapt. n_steps=1 gives the immediate
    successor; more steps traverse the sequence.
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

    def _normalize_and_center(self, pattern):
        p = pattern / (torch.linalg.norm(pattern) + 1e-10)
        self.n_stored += 1
        self.mean_activity += (p - self.mean_activity) / self.n_stored
        p_c = p - self.mean_activity
        return p, p_c

    def store_online(self, pattern, prev_pattern=None):
        """Store successor association: prev_pattern -> pattern."""
        _, curr_c = self._normalize_and_center(pattern)
        if prev_pattern is not None:
            prev_p = prev_pattern / (torch.linalg.norm(prev_pattern) + 1e-10)
            prev_c = prev_p - self.mean_activity
            self.W += self.lr * torch.outer(curr_c, prev_c)
            self.W.fill_diagonal_(0)

    def retrieve(self, cue, n_steps=1, adapt_rate=0.15, adapt_decay=0.85):
        """
        Unified retrieval. One dynamical process:
          - Step 0: CA3 is in the cue state (the cue IS the initial activity)
          - Steps 1+: recurrent dynamics transition to successors
          - n_steps=1: just the cue state
          - n_steps=6: cue + 5 successors

        Returns list of n_steps patterns and adaptation history.
        """
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        adaptation = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        trajectory = [x.clone()]
        adaptation_history = [0.0]

        for step in range(n_steps - 1):
            h = torch.relu(self.W @ x - adaptation)
            x_new = apply_kwta(h, self.k_active)
            norm = torch.linalg.norm(x_new)
            if norm < 1e-10:
                x_new = torch.zeros(self.N, device=self.device, dtype=self.dtype)
            else:
                x_new = x_new / norm
            trajectory.append(x_new.clone())
            adaptation_history.append(float(torch.linalg.norm(adaptation)))
            adaptation = adapt_decay * adaptation + adapt_rate * x_new
            x = x_new

        return trajectory, adaptation_history


# =============================================================================
# HIPPOCAMPAL SYSTEM: NEW (with mossy fiber projection)
# =============================================================================

class HippocampalSystemNew:
    """
    Architecture with explicit DG -> CA3 mossy fiber projection.

    Encoding path:
      EC_sup stellate -> DG (pattern separation)
      DG -> W_mf -> CA3 state (mossy fibers force CA3, no recurrent dynamics)
      CA3 successor associations stored in CA3 space
      CA1 learns from CA3-space state + EC stellate
      Sub learns from CA1 output + EC pyramidal

    Retrieval path:
      EC_sup stellate -> DG -> W_mf -> CA3 cue  (DG pathway)
      EC_sup stellate -> W_direct -> CA3 cue     (direct pathway)
      CA3 cue -> unified recurrent dynamics -> successor sequence
      CA3 states -> CA1 -> Sub -> output

    W_direct learns EC -> CA3-space associations during encoding.
    """
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3, ca3_lr=1.0,
                 direct_lr=0.3, direct_decay=0.998,
                 mf_connectivity_prob=0.33,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, N_sub=1000, ca3_retrieval_iterations=5,
                 device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.N_ca3 = N_ca3
        self.device = device
        self.dtype = dtype
        self.k_ca3 = k_ca3

        self.ec_sup = ECSuperficial(d_ec, **(ec_sup_params or {}),
                                    device=device, dtype=dtype)
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}),
                                      device=device, dtype=dtype)
        self.ca3 = CA3Temporal(N_ca3, k_ca3, lr=ca3_lr,
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

        self._prev_ca3_pattern = None
        self._in_sequence = False

    def begin_sequence(self):
        self._prev_ca3_pattern = None
        self._in_sequence = True

    def end_sequence(self):
        self._prev_ca3_pattern = None
        self._in_sequence = False

    def _mossy_fiber_transform(self, dg_out):
        """DG -> CA3 via mossy fibers. Forces CA3 state, no recurrent dynamics."""
        h = torch.relu(self.W_mf @ dg_out)
        ca3_state = apply_kwta(h, self.k_ca3)
        norm = torch.linalg.norm(ca3_state)
        if norm > 1e-10:
            ca3_state = ca3_state / norm
        return ca3_state

    def encode_single(self, ec_input):
        """
        Encode one timestep.

        DG separates the pattern, mossy fibers force CA3 into the
        corresponding CA3-space state. Successor associations are
        stored between consecutive CA3 states. CA1 and Sub learn
        from the CA3-space representation.
        """
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)

        # Mossy fiber projection: DG -> CA3 space
        ca3_state = self._mossy_fiber_transform(dg_out)

        # Store successor association in CA3 space
        prev = self._prev_ca3_pattern if self._in_sequence else None
        self.ca3.store_online(ca3_state, prev_pattern=prev)

        if self._in_sequence:
            self._prev_ca3_pattern = ca3_state.clone()

        # Direct pathway learning: EC stellate -> CA3 state
        stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
        self.W_direct += self.direct_lr * torch.outer(ca3_state, stel_norm)
        self.W_direct *= self.direct_decay

        # CA1 learns from CA3-space state (Schaffer) + EC stellate (TA)
        self.ca1.encode(ca3_state, stellate)
        ca1_out, mm = self.ca1.retrieve(ca3_state, stellate)
        self.sub.encode(ca1_out, pyramidal)

        return dg_out, ca3_state, mm

    def _raw_direct_cue(self, stellate):
        """EC stellate -> CA3 space via learned direct pathway."""
        return torch.relu(self.W_direct @ stellate)

    def _raw_dg_cue(self, stellate):
        """EC stellate -> DG -> W_mf -> CA3 space."""
        dg_out = self.dg.forward(stellate)
        return torch.relu(self.W_mf @ dg_out)

    def _apply_ca3_competition(self, raw_drive):
        h = apply_kwta(raw_drive, self.k_ca3)
        norm = torch.linalg.norm(h)
        if norm > 1e-10:
            h = h / norm
        return h

    def recall(self, ec_input, n_steps=1, adapt_rate=0.15, adapt_decay=0.85,
               pathway='combined'):
        """
        Unified recall. Forms a cue in CA3 space, then lets CA3 dynamics
        run for n_steps.

        pathway: 'combined' (DG + direct), 'direct', or 'dg'
        """
        stellate, _ = self.ec_sup.forward(ec_input)

        if pathway == 'combined':
            raw = self._raw_dg_cue(stellate) + self._raw_direct_cue(stellate)
        elif pathway == 'direct':
            raw = self._raw_direct_cue(stellate)
        elif pathway == 'dg':
            raw = self._raw_dg_cue(stellate)
        else:
            raise ValueError(f"Unknown pathway: {pathway}")

        ca3_cue = self._apply_ca3_competition(raw)
        trajectory, adapt_hist = self.ca3.retrieve(
            ca3_cue, n_steps=n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)
        return trajectory, adapt_hist


# =============================================================================
# OLD ARCHITECTURE (from capacity_limits_test.py, for comparison)
# =============================================================================

class CA3TemporalOld:
    """Original CA3 with split retrieve/retrieve_sequence."""
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

    def retrieve_sequence(self, cue, n_steps, adapt_rate=0.15, adapt_decay=0.85):
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        adaptation = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        trajectory = [x.clone()]
        adaptation_history = [0.0]
        for step in range(n_steps - 1):
            h = torch.relu(self.W @ x - adaptation)
            x_new = apply_kwta(h, self.k_active)
            norm = torch.linalg.norm(x_new)
            if norm < 1e-10:
                trajectory.append(torch.zeros(self.N, device=self.device, dtype=self.dtype))
                adaptation_history.append(float(torch.linalg.norm(adaptation)))
                x = x_new
                continue
            x_new = x_new / norm
            trajectory.append(x_new.clone())
            adaptation_history.append(float(torch.linalg.norm(adaptation)))
            adaptation = adapt_decay * adaptation + adapt_rate * x_new
            x = x_new
        return trajectory, adaptation_history


class HippocampalSystemOld:
    """Original architecture: DG patterns used directly in CA3 space."""
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3, ca3_lr=1.0,
                 direct_lr=0.3, direct_decay=0.998,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, N_sub=1000, ca3_retrieval_iterations=5,
                 device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.ca3_retrieval_iterations = ca3_retrieval_iterations
        self.device = device
        self.dtype = dtype
        self.k_ca3 = k_ca3

        self.ec_sup = ECSuperficial(d_ec, **(ec_sup_params or {}),
                                    device=device, dtype=dtype)
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}),
                                      device=device, dtype=dtype)
        self.ca3 = CA3TemporalOld(N_ca3, k_ca3, lr=ca3_lr,
                                   device=device, dtype=dtype)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}),
                       device=device, dtype=dtype)
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}),
                             device=device, dtype=dtype)

        self.W_direct = torch.zeros((N_ca3, d_ec), device=device, dtype=dtype)
        self.direct_lr = direct_lr
        self.direct_decay = direct_decay

        self._prev_dg_pattern = None
        self._in_sequence = False

    def begin_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = True

    def end_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = False

    def encode_single(self, ec_input):
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)

        prev = self._prev_dg_pattern if self._in_sequence else None
        self.ca3.store_online(dg_out, prev_pattern=prev)

        if self._in_sequence:
            self._prev_dg_pattern = dg_out.clone()

        dg_norm = dg_out / (torch.linalg.norm(dg_out) + 1e-10)
        stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
        self.W_direct += self.direct_lr * torch.outer(dg_norm, stel_norm)
        self.W_direct *= self.direct_decay

        self.ca1.encode(dg_out, stellate)
        # Use DG pattern directly as CA3 state for CA1/Sub
        ca1_out, mm = self.ca1.retrieve(dg_out, stellate)
        self.sub.encode(ca1_out, pyramidal)

        return dg_out, mm

    def _raw_direct_cue(self, stellate):
        return torch.relu(self.W_direct @ stellate)

    def _raw_dg_cue(self, stellate):
        return self.dg.forward(stellate)

    def _apply_ca3_competition(self, raw_drive):
        h = apply_kwta(raw_drive, self.k_ca3)
        norm = torch.linalg.norm(h)
        if norm > 1e-10:
            h = h / norm
        return h

    def recall(self, ec_input, n_steps=1, adapt_rate=0.15, adapt_decay=0.85,
               pathway='direct'):
        stellate, _ = self.ec_sup.forward(ec_input)
        if pathway == 'combined':
            raw = self._raw_dg_cue(stellate) + self._raw_direct_cue(stellate)
        elif pathway == 'direct':
            raw = self._raw_direct_cue(stellate)
        elif pathway == 'dg':
            raw = self._raw_dg_cue(stellate)
        else:
            raise ValueError(f"Unknown pathway: {pathway}")
        ca3_cue = self._apply_ca3_competition(raw)
        trajectory, adapt_hist = self.ca3.retrieve_sequence(
            ca3_cue, n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)
        return trajectory, adapt_hist


# =============================================================================
# DATA GENERATION AND ENCODING
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


def encode_all_new(hippo, sequences, n_repetitions=1):
    """Encode with new architecture. Returns both DG and CA3 patterns."""
    all_dg_patterns = [[] for _ in sequences]
    all_ca3_patterns = [[] for _ in sequences]
    for rep in range(n_repetitions):
        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for step_idx, ec_pattern in enumerate(seq):
                dg_out, ca3_state, mm = hippo.encode_single(ec_pattern)
                if rep == 0:
                    all_dg_patterns[seq_idx].append(dg_out.clone())
                    all_ca3_patterns[seq_idx].append(ca3_state.clone())
            hippo.end_sequence()
    return all_dg_patterns, all_ca3_patterns


def encode_all_old(hippo, sequences, n_repetitions=1):
    """Encode with old architecture. Returns DG patterns (= CA3 patterns)."""
    all_dg_patterns = [[] for _ in sequences]
    for rep in range(n_repetitions):
        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for step_idx, ec_pattern in enumerate(seq):
                dg_out, mm = hippo.encode_single(ec_pattern)
                if rep == 0:
                    all_dg_patterns[seq_idx].append(dg_out.clone())
            hippo.end_sequence()
    return all_dg_patterns


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
# TEST 1: Head-to-Head Capacity Sweep
# =============================================================================

def test_capacity_sweep(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Compare old vs new architecture on capacity sweep.
    Both cued via direct pathway, compared against their respective
    reference patterns (DG for old, CA3 for new).
    """
    print("\n" + "=" * 70)
    print("TEST 1: Capacity Sweep (Old vs New)")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [1, 2, 5, 10, 20, 50, 100, 200]

    results_old = {}
    results_new = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=1000 + n_seq)

        # --- Old architecture ---
        torch.manual_seed(42)
        hippo_old = HippocampalSystemOld(**hippo_kwargs, device=device, dtype=dtype)
        dg_patterns_old = encode_all_old(hippo_old, sequences, n_repetitions)

        seq_scores_old = []
        diag_scores_old = []
        for si in range(n_seq):
            cue = sequences[si][0]
            traj, _ = hippo_old.recall(cue, n_steps=seq_length,
                                       adapt_rate=0.0, adapt_decay=0.0,
                                       pathway='direct')
            sim = compute_similarity_matrix(traj, dg_patterns_old[si])
            s, _ = measure_sequentiality(sim)
            d = measure_diagonal(sim)
            seq_scores_old.append(s)
            diag_scores_old.append(d)

        results_old[n_seq] = {
            'mean_seq': np.mean(seq_scores_old),
            'mean_diag': np.mean(diag_scores_old),
            'pct_perfect': np.mean([1.0 if s == 1.0 else 0.0 for s in seq_scores_old]),
        }

        # --- New architecture ---
        torch.manual_seed(42)
        hippo_new = HippocampalSystemNew(**hippo_kwargs, device=device, dtype=dtype)
        _, ca3_patterns_new = encode_all_new(hippo_new, sequences, n_repetitions)

        seq_scores_new = []
        diag_scores_new = []
        for si in range(n_seq):
            cue = sequences[si][0]
            # New architecture: n_steps successors (does not include the cue itself)
            traj, _ = hippo_new.recall(cue, n_steps=seq_length,
                                       adapt_rate=0.0, adapt_decay=0.0,
                                       pathway='direct')
            # Compare against CA3-space reference patterns
            sim = compute_similarity_matrix(traj, ca3_patterns_new[si])
            s, _ = measure_sequentiality(sim)
            d = measure_diagonal(sim)
            seq_scores_new.append(s)
            diag_scores_new.append(d)

        results_new[n_seq] = {
            'mean_seq': np.mean(seq_scores_new),
            'mean_diag': np.mean(diag_scores_new),
            'pct_perfect': np.mean([1.0 if s == 1.0 else 0.0 for s in seq_scores_new]),
        }

        print(f"  n_seq={n_seq:4d}:  OLD seq={results_old[n_seq]['mean_seq']:.3f} "
              f"diag={results_old[n_seq]['mean_diag']:.4f}  |  "
              f"NEW seq={results_new[n_seq]['mean_seq']:.3f} "
              f"diag={results_new[n_seq]['mean_diag']:.4f}")

    return results_old, results_new, n_seq_values


# =============================================================================
# TEST 2: Representational Analysis
# =============================================================================

def test_representational_analysis(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Analyze the DG and CA3 representations in the new architecture.
    - How similar are DG patterns to each other? (should be low: pattern separation)
    - How similar are CA3 patterns to each other?
    - How much information does W_mf preserve?
    - Sparsity statistics for both populations.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Representational Analysis")
    print("=" * 70)

    n_seq = 20
    seq_length = 6

    sequences = generate_sequences(n_seq, seq_length,
                                   hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=2000)

    torch.manual_seed(42)
    hippo = HippocampalSystemNew(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns, ca3_patterns = encode_all_new(hippo, sequences, n_repetitions)

    # Flatten all patterns
    all_dg = [p for seq in dg_patterns for p in seq]
    all_ca3 = [p for seq in ca3_patterns for p in seq]
    n_patterns = len(all_dg)

    # Pairwise similarity within DG
    dg_sims = []
    ca3_sims = []
    cross_sims = []  # DG[i] vs CA3[i] (same-item correspondence)
    for i in range(n_patterns):
        cross_sims.append(cosine_sim(all_dg[i], all_ca3[i]))
        for j in range(i + 1, n_patterns):
            dg_sims.append(cosine_sim(all_dg[i], all_dg[j]))
            ca3_sims.append(cosine_sim(all_ca3[i], all_ca3[j]))

    # Sparsity
    dg_sparsities = [float((p > 0).sum()) / len(p) for p in all_dg]
    ca3_sparsities = [float((p > 0).sum()) / len(p) for p in all_ca3]
    dg_active_counts = [int((p > 0).sum()) for p in all_dg]
    ca3_active_counts = [int((p > 0).sum()) for p in all_ca3]

    # Within-sequence vs between-sequence similarity in CA3
    within_ca3 = []
    between_ca3 = []
    for si in range(n_seq):
        for t1 in range(seq_length):
            for t2 in range(t1 + 1, seq_length):
                within_ca3.append(cosine_sim(ca3_patterns[si][t1], ca3_patterns[si][t2]))
            for sj in range(si + 1, n_seq):
                for t2 in range(seq_length):
                    between_ca3.append(cosine_sim(ca3_patterns[si][t1], ca3_patterns[sj][t2]))

    results = {
        'dg_pairwise_mean': np.mean(dg_sims),
        'dg_pairwise_std': np.std(dg_sims),
        'ca3_pairwise_mean': np.mean(ca3_sims),
        'ca3_pairwise_std': np.std(ca3_sims),
        'cross_sim_mean': np.mean(cross_sims),
        'cross_sim_std': np.std(cross_sims),
        'dg_sparsity_mean': np.mean(dg_sparsities),
        'ca3_sparsity_mean': np.mean(ca3_sparsities),
        'dg_active_mean': np.mean(dg_active_counts),
        'ca3_active_mean': np.mean(ca3_active_counts),
        'within_ca3_mean': np.mean(within_ca3),
        'between_ca3_mean': np.mean(between_ca3),
        'dg_sims': dg_sims,
        'ca3_sims': ca3_sims,
        'cross_sims': cross_sims,
    }

    print(f"  DG pairwise similarity:  {results['dg_pairwise_mean']:.4f} +/- {results['dg_pairwise_std']:.4f}")
    print(f"  CA3 pairwise similarity: {results['ca3_pairwise_mean']:.4f} +/- {results['ca3_pairwise_std']:.4f}")
    print(f"  DG-CA3 correspondence:   {results['cross_sim_mean']:.4f} +/- {results['cross_sim_std']:.4f}")
    print(f"  DG sparsity:  {results['dg_sparsity_mean']:.4f} ({results['dg_active_mean']:.0f} active units)")
    print(f"  CA3 sparsity: {results['ca3_sparsity_mean']:.4f} ({results['ca3_active_mean']:.0f} active units)")
    print(f"  CA3 within-sequence sim:  {results['within_ca3_mean']:.4f}")
    print(f"  CA3 between-sequence sim: {results['between_ca3_mean']:.4f}")

    return results


# =============================================================================
# TEST 3: Pathway Comparison (Direct vs DG vs Combined)
# =============================================================================

def test_pathway_comparison(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Compare the three retrieval pathways at varying capacity.
    The direct pathway should support mid-sequence entry; the DG pathway
    should give higher-fidelity cues for sequence-start retrieval.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Pathway Comparison")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [1, 5, 10, 20, 50, 100]
    pathways = ['direct', 'dg', 'combined']

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=3000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystemNew(**hippo_kwargs, device=device, dtype=dtype)
        _, ca3_patterns = encode_all_new(hippo, sequences, n_repetitions)

        sample_n = min(n_seq, 20)
        sample_idx = list(range(sample_n))

        for pathway in pathways:
            seq_scores = []
            diag_scores = []

            for si in sample_idx:
                cue = sequences[si][0]
                traj, _ = hippo.recall(cue, n_steps=seq_length,
                                       adapt_rate=0.0, adapt_decay=0.0,
                                       pathway=pathway)
                sim = compute_similarity_matrix(traj, ca3_patterns[si])
                s, _ = measure_sequentiality(sim)
                d = measure_diagonal(sim)
                seq_scores.append(s)
                diag_scores.append(d)

            results[(n_seq, pathway)] = {
                'mean_seq': np.mean(seq_scores),
                'mean_diag': np.mean(diag_scores),
            }

        print(f"  n_seq={n_seq:4d}:  "
              f"direct seq={results[(n_seq, 'direct')]['mean_seq']:.3f} "
              f"diag={results[(n_seq, 'direct')]['mean_diag']:.4f}  |  "
              f"dg seq={results[(n_seq, 'dg')]['mean_seq']:.3f} "
              f"diag={results[(n_seq, 'dg')]['mean_diag']:.4f}  |  "
              f"combined seq={results[(n_seq, 'combined')]['mean_seq']:.3f} "
              f"diag={results[(n_seq, 'combined')]['mean_diag']:.4f}")

    return results, n_seq_values, pathways


# =============================================================================
# TEST 4: Mid-Sequence Entry
# =============================================================================

def test_mid_entry(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Mid-sequence entry via each pathway. The direct pathway should be
    particularly useful here since it maps arbitrary EC states into CA3
    space without needing DG pattern separation.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Mid-Sequence Entry")
    print("=" * 70)

    seq_length = 8
    n_seq_values = [1, 10, 50, 100]
    entry_points = [0, 2, 4, 6]
    pathways = ['direct', 'dg', 'combined']

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=4000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystemNew(**hippo_kwargs, device=device, dtype=dtype)
        _, ca3_patterns = encode_all_new(hippo, sequences, n_repetitions)

        for entry_pos in entry_points:
            remaining = seq_length - entry_pos
            ref_slice = ca3_patterns[0][entry_pos:]
            cue = sequences[0][entry_pos]

            for pathway in pathways:
                traj, _ = hippo.recall(cue, n_steps=remaining,
                                       adapt_rate=0.0, adapt_decay=0.0,
                                       pathway=pathway)
                sim = compute_similarity_matrix(traj, ref_slice)
                diag_sims = [sim[t, t] for t in range(min(sim.shape))]
                mean_diag = np.mean(diag_sims)

                results[(n_seq, entry_pos, pathway)] = {
                    'mean_diag': mean_diag,
                    'diag_sims': diag_sims,
                }

            print(f"  n_seq={n_seq:4d}, entry={entry_pos}:  "
                  f"direct={results[(n_seq, entry_pos, 'direct')]['mean_diag']:.4f}  "
                  f"dg={results[(n_seq, entry_pos, 'dg')]['mean_diag']:.4f}  "
                  f"combined={results[(n_seq, entry_pos, 'combined')]['mean_diag']:.4f}")

    return results, n_seq_values, entry_points, pathways


# =============================================================================
# TEST 5: Unified Retrieve Dynamics
# =============================================================================

def test_retrieve_dynamics(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    How does retrieval quality evolve as we let CA3 dynamics run?
    For a stored sequence of length L, we retrieve with n_steps up to 2*L
    and track how the diagonal similarity changes. This tests whether the
    unified dynamics naturally produce the successor chain and what happens
    when you overshoot the sequence length.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Retrieve Dynamics (n_steps sweep)")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [5, 20, 50]
    max_steps = 12  # 2x sequence length

    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=5000 + n_seq)

        torch.manual_seed(42)
        hippo = HippocampalSystemNew(**hippo_kwargs, device=device, dtype=dtype)
        _, ca3_patterns = encode_all_new(hippo, sequences, n_repetitions)

        sample_n = min(n_seq, 10)

        # Retrieve long trajectory
        all_sims = []
        for si in range(sample_n):
            cue = sequences[si][0]
            traj, adapt_hist = hippo.recall(
                cue, n_steps=max_steps,
                adapt_rate=0.0, adapt_decay=0.0,
                pathway='direct')

            # Similarity of each retrieved step to each stored CA3 pattern
            sim = compute_similarity_matrix(traj, ca3_patterns[si])
            all_sims.append(sim)

        # Per-step diagonal similarity (within the sequence length)
        step_diag = []
        for step in range(max_steps):
            diags = []
            for sim in all_sims:
                if step < sim.shape[1]:
                    diags.append(sim[step, step])
                else:
                    # Past the end of stored sequence: check max sim to any stored pattern
                    diags.append(float(np.max(sim[step, :])))
            step_diag.append(np.mean(diags))

        # Per-step max similarity to any stored pattern
        step_max = []
        for step in range(max_steps):
            maxes = [float(np.max(sim[step, :])) for sim in all_sims]
            step_max.append(np.mean(maxes))

        results[n_seq] = {
            'step_diag': step_diag,
            'step_max': step_max,
            'sample_sim_matrices': all_sims[:3],
        }

        diag_str = " ".join(f"{d:.3f}" for d in step_diag[:seq_length])
        print(f"  n_seq={n_seq:4d}: diag(steps 1-{seq_length}): {diag_str}")
        overshoot_str = " ".join(f"{d:.3f}" for d in step_max[seq_length:])
        print(f"  {'':>12s}  max_sim(steps {seq_length+1}-{max_steps}): {overshoot_str}")

    return results, n_seq_values, max_steps, seq_length


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_results(t1, t2, t3, t4, t5, save_path):
    fig = plt.figure(figsize=(24, 30))
    gs = GridSpec(6, 4, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Mossy Fiber Projection: Capacity Tests",
                 fontsize=16, fontweight='bold', y=0.995)

    # ------------------------------------------------------------------
    # Row 1: Capacity sweep (old vs new)
    # ------------------------------------------------------------------
    results_old, results_new, n_seq_values = t1

    ax1a = fig.add_subplot(gs[0, 0:2])
    old_seq = [results_old[n]['mean_seq'] for n in n_seq_values]
    new_seq = [results_new[n]['mean_seq'] for n in n_seq_values]
    ax1a.plot(n_seq_values, old_seq, 'o-', color='coral', linewidth=2,
              label='Old (DG=CA3)', markersize=6)
    ax1a.plot(n_seq_values, new_seq, 's-', color='steelblue', linewidth=2,
              label='New (W_mf)', markersize=6)
    ax1a.set_xlabel("Number of sequences")
    ax1a.set_ylabel("Mean sequentiality")
    ax1a.set_title("Test 1: Capacity - Sequentiality")
    ax1a.legend(fontsize=8)
    ax1a.set_ylim(-0.1, 1.1)
    ax1a.set_xscale('log')
    ax1a.grid(True, alpha=0.3)

    ax1b = fig.add_subplot(gs[0, 2:4])
    old_diag = [results_old[n]['mean_diag'] for n in n_seq_values]
    new_diag = [results_new[n]['mean_diag'] for n in n_seq_values]
    ax1b.plot(n_seq_values, old_diag, 'o-', color='coral', linewidth=2,
              label='Old (DG=CA3)', markersize=6)
    ax1b.plot(n_seq_values, new_diag, 's-', color='steelblue', linewidth=2,
              label='New (W_mf)', markersize=6)
    ax1b.set_xlabel("Number of sequences")
    ax1b.set_ylabel("Mean diagonal similarity")
    ax1b.set_title("Test 1: Capacity - Diagonal Similarity")
    ax1b.legend(fontsize=8)
    ax1b.set_ylim(-0.1, 1.1)
    ax1b.set_xscale('log')
    ax1b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 2: Representational analysis
    # ------------------------------------------------------------------
    rep_results = t2

    ax2a = fig.add_subplot(gs[1, 0:2])
    bins = np.linspace(-0.3, 0.5, 60)
    ax2a.hist(rep_results['dg_sims'], bins=bins, alpha=0.6, color='coral',
              label=f"DG (mean={rep_results['dg_pairwise_mean']:.4f})", density=True)
    ax2a.hist(rep_results['ca3_sims'], bins=bins, alpha=0.6, color='steelblue',
              label=f"CA3 (mean={rep_results['ca3_pairwise_mean']:.4f})", density=True)
    ax2a.set_xlabel("Cosine similarity")
    ax2a.set_ylabel("Density")
    ax2a.set_title("Test 2: Pairwise Similarity Distributions")
    ax2a.legend(fontsize=8)
    ax2a.grid(True, alpha=0.3)

    ax2b = fig.add_subplot(gs[1, 2:4])
    ax2b.hist(rep_results['cross_sims'], bins=40, alpha=0.7, color='purple')
    ax2b.axvline(rep_results['cross_sim_mean'], color='red', linestyle='--',
                 label=f"Mean={rep_results['cross_sim_mean']:.4f}")
    ax2b.set_xlabel("Cosine similarity")
    ax2b.set_ylabel("Count")
    ax2b.set_title("Test 2: DG-CA3 Same-Item Correspondence")
    ax2b.legend(fontsize=8)
    ax2b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 3: Pathway comparison
    # ------------------------------------------------------------------
    path_results, path_n_seq, pathways = t3

    colors_path = {'direct': 'coral', 'dg': 'steelblue', 'combined': 'forestgreen'}

    ax3a = fig.add_subplot(gs[2, 0:2])
    for pw in pathways:
        y = [path_results[(n, pw)]['mean_seq'] for n in path_n_seq]
        ax3a.plot(path_n_seq, y, 'o-', color=colors_path[pw], linewidth=2,
                  label=pw, markersize=5)
    ax3a.set_xlabel("Number of sequences")
    ax3a.set_ylabel("Mean sequentiality")
    ax3a.set_title("Test 3: Pathway Comparison - Sequentiality")
    ax3a.legend(fontsize=8)
    ax3a.set_ylim(-0.1, 1.1)
    ax3a.set_xscale('log')
    ax3a.grid(True, alpha=0.3)

    ax3b = fig.add_subplot(gs[2, 2:4])
    for pw in pathways:
        y = [path_results[(n, pw)]['mean_diag'] for n in path_n_seq]
        ax3b.plot(path_n_seq, y, 'o-', color=colors_path[pw], linewidth=2,
                  label=pw, markersize=5)
    ax3b.set_xlabel("Number of sequences")
    ax3b.set_ylabel("Mean diagonal similarity")
    ax3b.set_title("Test 3: Pathway Comparison - Diag Sim")
    ax3b.legend(fontsize=8)
    ax3b.set_ylim(-0.1, 1.1)
    ax3b.set_xscale('log')
    ax3b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 4: Mid-sequence entry
    # ------------------------------------------------------------------
    mid_results, mid_n_seq, mid_entries, mid_pathways = t4

    for ei, entry in enumerate(mid_entries):
        ax = fig.add_subplot(gs[3, ei])
        for pw in ['direct', 'dg', 'combined']:
            y = [mid_results[(n, entry, pw)]['mean_diag'] for n in mid_n_seq]
            ax.plot(mid_n_seq, y, 'o-', color=colors_path[pw], linewidth=2,
                    label=pw, markersize=5)
        ax.set_xlabel("Number of sequences")
        ax.set_ylabel("Mean continuation sim")
        ax.set_title(f"Test 4: Entry pos {entry}")
        ax.set_ylim(-0.1, 1.1)
        ax.set_xscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 5: Retrieve dynamics (n_steps sweep)
    # ------------------------------------------------------------------
    dyn_results, dyn_n_seq, max_steps, stored_len = t5

    colors_dyn = ['steelblue', 'coral', 'forestgreen']
    ax5a = fig.add_subplot(gs[4, 0:2])
    for ci, n_seq in enumerate(dyn_n_seq):
        step_diag = dyn_results[n_seq]['step_diag']
        ax5a.plot(range(1, max_steps + 1), step_diag, 'o-',
                  color=colors_dyn[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=4)
    ax5a.axvline(stored_len, color='gray', linestyle='--', alpha=0.5,
                 label=f'Sequence length ({stored_len})')
    ax5a.set_xlabel("Retrieval step")
    ax5a.set_ylabel("Diagonal similarity")
    ax5a.set_title("Test 5: Retrieval Quality vs Steps")
    ax5a.legend(fontsize=7)
    ax5a.set_ylim(-0.1, 1.1)
    ax5a.grid(True, alpha=0.3)

    ax5b = fig.add_subplot(gs[4, 2:4])
    for ci, n_seq in enumerate(dyn_n_seq):
        step_max = dyn_results[n_seq]['step_max']
        ax5b.plot(range(1, max_steps + 1), step_max, 'o-',
                  color=colors_dyn[ci], linewidth=2,
                  label=f'n_seq={n_seq}', markersize=4)
    ax5b.axvline(stored_len, color='gray', linestyle='--', alpha=0.5,
                 label=f'Sequence length ({stored_len})')
    ax5b.set_xlabel("Retrieval step")
    ax5b.set_ylabel("Max similarity to any stored pattern")
    ax5b.set_title("Test 5: Best-Match Quality vs Steps")
    ax5b.legend(fontsize=7)
    ax5b.set_ylim(-0.1, 1.1)
    ax5b.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Row 6: Sample similarity matrices from Test 5
    # ------------------------------------------------------------------
    for ci, n_seq in enumerate(dyn_n_seq):
        sims = dyn_results[n_seq]['sample_sim_matrices']
        if sims:
            ax = fig.add_subplot(gs[5, ci])
            im = ax.imshow(sims[0], aspect='auto', cmap='viridis',
                           vmin=-0.2, vmax=1.0)
            ax.set_xlabel("Stored CA3 pattern")
            ax.set_ylabel("Retrieved step")
            ax.set_title(f"Test 5: n_seq={n_seq}, seq 0")
            plt.colorbar(im, ax=ax, fraction=0.046)
    # Fill remaining subplot
    if len(dyn_n_seq) < 4:
        ax = fig.add_subplot(gs[5, len(dyn_n_seq)])
        ax.axis('off')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
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
        "ca3_retrieval_iterations": 5,  # only used by old architecture
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
    print("Mossy Fiber Projection: Capacity Tests")
    print("=" * 70)
    print(f"  d_ec={d_ec}, D_dg={d_ec}, N_ca3={d_ec}")
    print(f"  k_ca3=50, mf_connectivity=0.33")
    print(f"  Encoding repetitions: {n_repetitions}")

    t1 = test_capacity_sweep(hippo_kwargs, device, dtype, n_repetitions)
    t2 = test_representational_analysis(hippo_kwargs, device, dtype, n_repetitions)
    t3 = test_pathway_comparison(hippo_kwargs, device, dtype, n_repetitions)
    t4 = test_mid_entry(hippo_kwargs, device, dtype, n_repetitions)
    t5 = test_retrieve_dynamics(hippo_kwargs, device, dtype, n_repetitions)

    plot_all_results(t1, t2, t3, t4, t5,
                     save_path="mossy_fiber_capacity_results.png")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()