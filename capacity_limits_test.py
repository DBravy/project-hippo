"""
Hippocampal Sequence Attractors: Capacity Limits
=================================================

How many sequences can the successor map store before retrieval degrades?

Tests:
  1. Capacity sweep: vary n_sequences, fixed seq_length
  2. Capacity x sequence length interaction
  3. Capacity x adaptation: does adaptation help at high load?
  4. Detailed retrieval at key capacity points
  5. Mid-sequence entry under load (direct vs DG pathway)
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


class ECDeepVb:
    @staticmethod
    def gate(sparse_signal, dense_signal):
        gamma = torch.sigmoid(dense_signal)
        return gamma * sparse_signal

    @staticmethod
    def hippocampal_output(ca1_output, sub_output):
        return ECDeepVb.gate(ca1_output, sub_output)


# =============================================================================
# CA3: Pure Temporal Association
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
# HIPPOCAMPAL SYSTEM
# =============================================================================

class HippocampalSystemTemporal:
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3, ca3_lr=1.0,
                 direct_lr=0.3, direct_decay=0.998,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, N_sub=1000, ca3_retrieval_iterations=5,
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
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}),
                       device=device, dtype=dtype)
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}),
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
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
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

    def recall_sequence(self, ec_input, n_steps, adapt_rate=0.15,
                        adapt_decay=0.85):
        """Retrieve via combined pathways (default)."""
        stellate, _ = self.ec_sup.forward(ec_input)
        raw_dg = self._raw_dg_cue(stellate)
        raw_direct = self._raw_direct_cue(stellate)
        combined = raw_dg + raw_direct
        ca3_cue = self._apply_ca3_competition(combined)
        trajectory, adapt_hist = self.ca3.retrieve_sequence(
            ca3_cue, n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)
        return trajectory, adapt_hist

    def recall_sequence_via_direct(self, ec_input, n_steps, adapt_rate=0.15,
                                   adapt_decay=0.85):
        """Retrieve via direct pathway only."""
        stellate, _ = self.ec_sup.forward(ec_input)
        raw = self._raw_direct_cue(stellate)
        ca3_cue = self._apply_ca3_competition(raw)
        trajectory, adapt_hist = self.ca3.retrieve_sequence(
            ca3_cue, n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)
        return trajectory, adapt_hist

    def recall_sequence_via_dg(self, ec_input, n_steps, adapt_rate=0.15,
                               adapt_decay=0.85):
        """Retrieve via DG pathway only."""
        stellate, _ = self.ec_sup.forward(ec_input)
        raw = self._raw_dg_cue(stellate)
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


def encode_all_sequences(hippo, sequences, n_repetitions=1):
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
# TEST 1: Capacity Sweep
# =============================================================================

def test_capacity_sweep(hippo_kwargs, device, dtype, n_repetitions=3,
                        noise_scale=0.0):
    """
    Core capacity test. Vary number of sequences stored, measure retrieval
    quality. All sequences have the same length. Each is cued from its
    first pattern.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Capacity Sweep")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [1, 2, 5, 10, 20, 50, 100, 200]

    results = {}

    for n_seq in n_seq_values:
        torch.manual_seed(42)
        hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)

        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=1000 + n_seq)
        dg_patterns = encode_all_sequences(hippo, sequences,
                                           n_repetitions=n_repetitions)

        total_transitions = n_seq * (seq_length - 1) * n_repetitions

        # Test retrieval of every sequence
        seq_scores = []
        diag_scores = []
        first_step_sims = []

        for seq_idx in range(n_seq):
            cue = sequences[seq_idx][0]
            if noise_scale > 0:
                noise = torch.randn_like(cue) * noise_scale
                cue = cue + noise
                cue = cue / (torch.linalg.norm(cue) + 1e-10)

            traj, _ = hippo.recall_sequence_via_direct(
                cue, n_steps=seq_length,
                adapt_rate=0.0, adapt_decay=0.0)

            sim = compute_similarity_matrix(traj, dg_patterns[seq_idx])
            seq_score, peaks = measure_sequentiality(sim)
            diag_score = measure_diagonal(sim)
            seq_scores.append(seq_score)
            diag_scores.append(diag_score)
            first_step_sims.append(sim[0, 0])

        mean_seq = np.mean(seq_scores)
        mean_diag = np.mean(diag_scores)
        min_diag = np.min(diag_scores)
        pct_perfect = np.mean([1.0 if s == 1.0 else 0.0 for s in seq_scores])
        mean_first = np.mean(first_step_sims)

        results[n_seq] = {
            'mean_sequentiality': mean_seq,
            'mean_diag': mean_diag,
            'min_diag': min_diag,
            'pct_perfect': pct_perfect,
            'mean_first_step': mean_first,
            'total_transitions': total_transitions,
            'seq_scores': seq_scores,
            'diag_scores': diag_scores,
        }

        print(f"  n_seq={n_seq:4d}: seq={mean_seq:.3f}, "
              f"diag={mean_diag:.4f}, min_diag={min_diag:.4f}, "
              f"perfect={pct_perfect:.1%}, first_step={mean_first:.4f}, "
              f"transitions={total_transitions}")

    return results, n_seq_values


# =============================================================================
# TEST 2: Capacity x Sequence Length
# =============================================================================

def test_capacity_x_length(hippo_kwargs, device, dtype, n_repetitions=3,
                           noise_scale=0.0):
    """
    How does sequence length interact with capacity? Longer sequences
    mean more transitions stored, but also more chances for error
    accumulation during retrieval.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Capacity x Sequence Length")
    print("=" * 70)

    seq_lengths = [4, 6, 8, 12]
    n_seq_values = [1, 5, 10, 20, 50, 100]

    results = {}

    for seq_len in seq_lengths:
        for n_seq in n_seq_values:
            torch.manual_seed(42)
            hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)

            sequences = generate_sequences(n_seq, seq_len,
                                           hippo_kwargs['d_ec'],
                                           device=device, dtype=dtype,
                                           seed=2000 + n_seq * 100 + seq_len)
            dg_patterns = encode_all_sequences(hippo, sequences,
                                               n_repetitions=n_repetitions)

            total_transitions = n_seq * (seq_len - 1) * n_repetitions

            # Sample up to 20 sequences for efficiency
            sample_n = min(n_seq, 20)
            sample_idx = np.random.choice(n_seq, sample_n, replace=False)

            seq_scores = []
            diag_scores = []

            for si in sample_idx:
                cue = sequences[si][0]
                if noise_scale > 0:
                    noise = torch.randn_like(cue) * noise_scale
                    cue = cue + noise
                    cue = cue / (torch.linalg.norm(cue) + 1e-10)

                traj, _ = hippo.recall_sequence_via_direct(
                    cue, n_steps=seq_len,
                    adapt_rate=0.0, adapt_decay=0.0)

                sim = compute_similarity_matrix(traj, dg_patterns[si])
                s, _ = measure_sequentiality(sim)
                d = measure_diagonal(sim)
                seq_scores.append(s)
                diag_scores.append(d)

            mean_seq = np.mean(seq_scores)
            mean_diag = np.mean(diag_scores)

            results[(n_seq, seq_len)] = {
                'mean_seq': mean_seq,
                'mean_diag': mean_diag,
                'total_transitions': total_transitions,
            }

            print(f"  n_seq={n_seq:4d}, len={seq_len:2d}: "
                  f"seq={mean_seq:.3f}, diag={mean_diag:.4f}, "
                  f"transitions={total_transitions}")

    return results, n_seq_values, seq_lengths


# =============================================================================
# TEST 3: Capacity x Adaptation
# =============================================================================

def test_capacity_x_adaptation(hippo_kwargs, device, dtype, n_repetitions=3,
                               noise_scale=0.0):
    """
    At high capacity, does adaptation start helping? When the successor
    map is noisy from interference, adaptation could prevent getting
    stuck in spurious attractors.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Capacity x Adaptation")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [10, 50, 100, 200]
    adapt_rates = [0.0, 0.05, 0.15, 0.25, 0.40]

    results = {}

    for n_seq in n_seq_values:
        torch.manual_seed(42)
        hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)

        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=3000 + n_seq)
        dg_patterns = encode_all_sequences(hippo, sequences,
                                           n_repetitions=n_repetitions)

        sample_n = min(n_seq, 20)
        sample_idx = np.random.choice(n_seq, sample_n, replace=False)

        for adapt_rate in adapt_rates:
            seq_scores = []
            diag_scores = []

            for si in sample_idx:
                cue = sequences[si][0]
                if noise_scale > 0:
                    noise = torch.randn_like(cue) * noise_scale
                    cue = cue + noise
                    cue = cue / (torch.linalg.norm(cue) + 1e-10)

                traj, _ = hippo.recall_sequence_via_direct(
                    cue, n_steps=seq_length,
                    adapt_rate=adapt_rate, adapt_decay=0.85)

                sim = compute_similarity_matrix(traj, dg_patterns[si])
                s, _ = measure_sequentiality(sim)
                d = measure_diagonal(sim)
                seq_scores.append(s)
                diag_scores.append(d)

            mean_seq = np.mean(seq_scores)
            mean_diag = np.mean(diag_scores)

            results[(n_seq, adapt_rate)] = {
                'mean_seq': mean_seq,
                'mean_diag': mean_diag,
            }

            print(f"  n_seq={n_seq:4d}, adapt={adapt_rate:.2f}: "
                  f"seq={mean_seq:.3f}, diag={mean_diag:.4f}")

    return results, n_seq_values, adapt_rates


# =============================================================================
# TEST 4: Detailed Retrieval at Key Capacity Points
# =============================================================================

def test_detailed_retrieval(hippo_kwargs, device, dtype, n_repetitions=3,
                            noise_scale=0.0):
    """
    Show similarity matrices for a few sequences at low, medium, and
    high capacity to visualize how degradation looks.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Detailed Retrieval at Key Capacities")
    print("=" * 70)

    seq_length = 6
    capacity_points = [5, 50, 200]

    results = {}

    for n_seq in capacity_points:
        torch.manual_seed(42)
        hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)

        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=4000 + n_seq)
        dg_patterns = encode_all_sequences(hippo, sequences,
                                           n_repetitions=n_repetitions)

        # Retrieve first 3 sequences
        sim_matrices = []
        for si in range(min(3, n_seq)):
            cue = sequences[si][0]
            if noise_scale > 0:
                noise = torch.randn_like(cue) * noise_scale
                cue = cue + noise
                cue = cue / (torch.linalg.norm(cue) + 1e-10)

            traj, _ = hippo.recall_sequence_via_direct(
                cue, n_steps=seq_length,
                adapt_rate=0.0, adapt_decay=0.0)

            sim = compute_similarity_matrix(traj, dg_patterns[si])
            s, peaks = measure_sequentiality(sim)
            d = measure_diagonal(sim)
            sim_matrices.append(sim)

            print(f"  n_seq={n_seq:4d}, seq {si}: "
                  f"seq={s:.2f}, diag={d:.4f}, peaks={peaks}")

        results[n_seq] = sim_matrices

    return results, capacity_points


# =============================================================================
# TEST 5: Mid-Sequence Entry Under Load
# =============================================================================

def test_mid_entry_capacity(hippo_kwargs, device, dtype, n_repetitions=3,
                            noise_scale=0.0):
    """
    Does mid-sequence entry via the direct pathway degrade gracefully
    as capacity increases?
    """
    print("\n" + "=" * 70)
    print("TEST 5: Mid-Sequence Entry Under Load")
    print("=" * 70)

    seq_length = 8
    n_seq_values = [1, 10, 50, 100]
    entry_points = [0, 2, 4, 6]

    results = {}

    for n_seq in n_seq_values:
        torch.manual_seed(42)
        hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)

        sequences = generate_sequences(n_seq, seq_length,
                                       hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=5000 + n_seq)
        dg_patterns = encode_all_sequences(hippo, sequences,
                                           n_repetitions=n_repetitions)

        # Test mid-entry on first sequence
        for entry_pos in entry_points:
            cue = sequences[0][entry_pos]
            if noise_scale > 0:
                noise = torch.randn_like(cue) * noise_scale
                cue = cue + noise
                cue = cue / (torch.linalg.norm(cue) + 1e-10)

            remaining = seq_length - entry_pos
            ref_slice = dg_patterns[0][entry_pos:]

            traj, _ = hippo.recall_sequence_via_direct(
                cue, n_steps=remaining,
                adapt_rate=0.0, adapt_decay=0.0)

            sim = compute_similarity_matrix(traj, ref_slice)
            diag_sims = [sim[t, t] for t in range(len(ref_slice))]
            mean_diag = np.mean(diag_sims)

            sim_full = compute_similarity_matrix(traj, dg_patterns[0])
            peaks = [int(np.argmax(sim_full[t, :])) for t in range(remaining)]

            results[(n_seq, entry_pos)] = {
                'mean_diag': mean_diag,
                'diag_sims': diag_sims,
                'peaks': peaks,
            }

            print(f"  n_seq={n_seq:4d}, entry={entry_pos}: "
                  f"mean={mean_diag:.4f}, peaks={peaks}")

    return results, n_seq_values, entry_points


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_results(t1, t2, t3, t4, t5, save_path):
    fig = plt.figure(figsize=(24, 28))
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Hippocampal Sequence Capacity Limits",
                 fontsize=16, fontweight='bold', y=0.99)

    # Row 1: Capacity sweep
    results_1, n_seq_1 = t1

    ax1a = fig.add_subplot(gs[0, 0:2])
    seq_scores = [results_1[n]['mean_sequentiality'] for n in n_seq_1]
    diag_scores = [results_1[n]['mean_diag'] for n in n_seq_1]
    pct_perfect = [results_1[n]['pct_perfect'] for n in n_seq_1]
    ax1a.plot(n_seq_1, seq_scores, 'o-', color='steelblue',
              linewidth=2, label='Mean sequentiality', markersize=6)
    ax1a.plot(n_seq_1, diag_scores, 's-', color='coral',
              linewidth=2, label='Mean diag sim', markersize=6)
    ax1a.plot(n_seq_1, pct_perfect, '^-', color='forestgreen',
              linewidth=2, label='% perfect retrieval', markersize=6)
    ax1a.set_xlabel("Number of sequences stored")
    ax1a.set_ylabel("Score")
    ax1a.set_title("Test 1: Capacity Sweep")
    ax1a.legend(fontsize=8)
    ax1a.set_ylim(-0.1, 1.1)
    ax1a.set_xscale('log')
    ax1a.grid(True, alpha=0.3)

    ax1b = fig.add_subplot(gs[0, 2:4])
    transitions = [results_1[n]['total_transitions'] for n in n_seq_1]
    ax1b.plot(transitions, seq_scores, 'o-', color='steelblue',
              linewidth=2, label='Mean sequentiality', markersize=6)
    ax1b.plot(transitions, diag_scores, 's-', color='coral',
              linewidth=2, label='Mean diag sim', markersize=6)
    ax1b.set_xlabel("Total transitions stored")
    ax1b.set_ylabel("Score")
    ax1b.set_title("Test 1: vs Total Transitions")
    ax1b.legend(fontsize=8)
    ax1b.set_ylim(-0.1, 1.1)
    ax1b.set_xscale('log')
    ax1b.grid(True, alpha=0.3)

    # Row 2: Capacity x Length
    results_2, n_seq_2, seq_lens_2 = t2

    colors_len = ['steelblue', 'coral', 'forestgreen', 'purple']
    ax2a = fig.add_subplot(gs[1, 0:2])
    for li, slen in enumerate(seq_lens_2):
        x = [n for n in n_seq_2 if (n, slen) in results_2]
        y = [results_2[(n, slen)]['mean_seq'] for n in x]
        ax2a.plot(x, y, 'o-', color=colors_len[li], linewidth=2,
                  label=f'len={slen}', markersize=5)
    ax2a.set_xlabel("Number of sequences")
    ax2a.set_ylabel("Mean sequentiality")
    ax2a.set_title("Test 2: Capacity x Length (Sequentiality)")
    ax2a.legend(fontsize=8)
    ax2a.set_ylim(-0.1, 1.1)
    ax2a.set_xscale('log')
    ax2a.grid(True, alpha=0.3)

    ax2b = fig.add_subplot(gs[1, 2:4])
    for li, slen in enumerate(seq_lens_2):
        x = [n for n in n_seq_2 if (n, slen) in results_2]
        y = [results_2[(n, slen)]['mean_diag'] for n in x]
        ax2b.plot(x, y, 'o-', color=colors_len[li], linewidth=2,
                  label=f'len={slen}', markersize=5)
    ax2b.set_xlabel("Number of sequences")
    ax2b.set_ylabel("Mean diag similarity")
    ax2b.set_title("Test 2: Capacity x Length (Diag Sim)")
    ax2b.legend(fontsize=8)
    ax2b.set_ylim(-0.1, 1.1)
    ax2b.set_xscale('log')
    ax2b.grid(True, alpha=0.3)

    # Row 3: Capacity x Adaptation
    results_3, n_seq_3, adapt_rates_3 = t3

    for ni, n_seq in enumerate(n_seq_3):
        ax = fig.add_subplot(gs[2, ni])
        seq_by_rate = [results_3[(n_seq, r)]['mean_seq'] for r in adapt_rates_3]
        diag_by_rate = [results_3[(n_seq, r)]['mean_diag'] for r in adapt_rates_3]
        ax.plot(adapt_rates_3, seq_by_rate, 'o-', color='steelblue',
                linewidth=2, label='Sequentiality', markersize=5)
        ax.plot(adapt_rates_3, diag_by_rate, 's-', color='coral',
                linewidth=2, label='Diag sim', markersize=5)
        ax.set_xlabel("Adaptation rate")
        ax.set_ylabel("Score")
        ax.set_title(f"Test 3: n_seq={n_seq}")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    # Row 4: Detailed retrieval
    results_4, cap_points_4 = t4

    plot_idx = 0
    for ci, n_seq in enumerate(cap_points_4):
        sims = results_4[n_seq]
        for si, sim in enumerate(sims[:1]):  # just first sequence per capacity
            if plot_idx < 4:
                ax = fig.add_subplot(gs[3, plot_idx])
                im = ax.imshow(sim, aspect='auto', cmap='viridis',
                               vmin=-0.2, vmax=1.0)
                ax.set_xlabel("Stored pattern")
                ax.set_ylabel("Retrieved step")
                ax.set_title(f"Test 4: {n_seq} seqs, seq 0")
                plot_idx += 1
    if plot_idx < 4:
        ax = fig.add_subplot(gs[3, 3])
        ax.axis('off')

    # Row 5: Mid-entry under load
    results_5, n_seq_5, entries_5 = t5

    for ei, entry in enumerate(entries_5):
        if ei < 4:
            ax = fig.add_subplot(gs[4, ei])
            means = [results_5[(n, entry)]['mean_diag'] for n in n_seq_5]
            ax.plot(n_seq_5, means, 'o-', color='steelblue',
                    linewidth=2, markersize=6)
            ax.set_xlabel("Number of sequences")
            ax.set_ylabel("Mean continuation sim")
            ax.set_title(f"Test 5: Entry pos {entry}")
            ax.set_ylim(-0.1, 1.1)
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  Saved figure to {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dtype = torch.float32
    d_ec = 1000

    # Configurable noise for retrieval cues (0.0 = clean)
    noise_scale = 0.0

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
        "ca1_params": {"lr": 50.0, "weight_decay": 1.000, "k_active": 50},
        "sub_params": {"lr": 1.0, "k_active": 500},
    }

    n_repetitions = 3

    print("=" * 70)
    print("Hippocampal Sequence Capacity Limits")
    print("=" * 70)
    print(f"  d_ec={d_ec}, k_ca3=50, N_ca3={d_ec}")
    print(f"  Noise scale: {noise_scale}")
    print(f"  Encoding repetitions: {n_repetitions}")
    print(f"  Retrieval: direct pathway, adapt=0.0 (unless swept)")

    t1 = test_capacity_sweep(hippo_kwargs, device, dtype,
                             n_repetitions=n_repetitions,
                             noise_scale=noise_scale)
    t2 = test_capacity_x_length(hippo_kwargs, device, dtype,
                                n_repetitions=n_repetitions,
                                noise_scale=noise_scale)
    t3 = test_capacity_x_adaptation(hippo_kwargs, device, dtype,
                                    n_repetitions=n_repetitions,
                                    noise_scale=noise_scale)
    t4 = test_detailed_retrieval(hippo_kwargs, device, dtype,
                                 n_repetitions=n_repetitions,
                                 noise_scale=noise_scale)
    t5 = test_mid_entry_capacity(hippo_kwargs, device, dtype,
                                 n_repetitions=n_repetitions,
                                 noise_scale=noise_scale)

    plot_all_results(t1, t2, t3, t4, t5,
                     save_path="capacity_limits_results.png")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
