"""
Hippocampal Sequence Attractors v2: Pure Temporal Association
==============================================================

Restructured model based on the insight that the hippocampus has one
learning signal: temporal contiguity. There is no separate symmetric
Hebbian component. Attractor basins are not hand-built; they emerge
from repeated sequential exposure.

The single learning rule:
    W += outer(curr_centered, prev_centered)

Stability (attractor-like behavior) emerges from:
  1. Repeated presentation of the same sequence, which deepens
     transient basins along the sequential flow.
  2. Similarity in cortical activity across episodes, which creates
     overlapping temporal associations and generalizable structure.

Test suite:
  - Test 1: Repetition sweep (core test: does stability emerge from repetition?)
  - Test 2: Single sequence retrieval at sufficient repetitions
  - Test 3: Multi-sequence separation
  - Test 4: Mid-sequence entry
  - Test 5: Cortical similarity / cross-sequence generalization
  - Test 6: Adaptation rate sweep
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
# UNCHANGED COMPONENTS: EC, DG, CA1, Subiculum, ECDeep
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
    """
    CA3 with a single learning rule: temporal contiguity.

    Storage:
        W += outer(curr_centered, prev_centered)

    No symmetric self-association. No alpha_sym / alpha_asym split.
    The only signal is "this pattern followed that pattern."

    Attractor-like stability emerges from repeated sequential exposure:
    the more times A->B is presented, the stronger the transient basin
    at B when approaching from A's direction. Stability is not a
    separate mechanism; it is a consequence of temporal statistics.
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
        """Normalize a pattern and return its mean-centered version.

        Updates the running mean. Returns (normalized, centered).
        The normalized version is needed for the running mean update;
        the centered version is what goes into the outer product.
        """
        p = pattern / (torch.linalg.norm(pattern) + 1e-10)
        self.n_stored += 1
        self.mean_activity += (p - self.mean_activity) / self.n_stored
        p_c = p - self.mean_activity
        return p, p_c

    def store_online(self, pattern, prev_pattern=None):
        """
        Store a temporal association.

        If prev_pattern is None (first element of a sequence, or a
        standalone pattern), no weight update occurs. The hippocampus
        only learns transitions, not isolated states.

        Args:
            pattern: current DG output
            prev_pattern: previous DG output in the sequence, or None
        """
        _, curr_c = self._normalize_and_center(pattern)

        if prev_pattern is not None:
            # Center prev relative to current running mean
            prev_p = prev_pattern / (torch.linalg.norm(prev_pattern) + 1e-10)
            prev_c = prev_p - self.mean_activity

            # The single learning rule: temporal contiguity
            self.W += self.lr * torch.outer(curr_c, prev_c)
            self.W.fill_diagonal_(0)

    def retrieve(self, cue, n_iterations=5):
        """Standard iterative retrieval (settles toward nearest attractor)."""
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
        """
        Single-step sequence retrieval.

        The cue is recorded as trajectory[0]. Each subsequent step
        applies W once to produce the next state: W @ x_t ≈ x_{t+1}.
        Returns a trajectory of length n_steps where trajectory[0]
        corresponds to the cue and trajectory[t] corresponds to stored
        pattern t.

        Args:
            cue: starting pattern (position 0 in the sequence)
            n_steps: total length of retrieved trajectory (including cue)
            adapt_rate: how quickly active neurons get suppressed
            adapt_decay: how quickly suppression fades

        Returns:
            trajectory: list of n_steps patterns (positions 0..n_steps-1)
            adaptation_history: list of adaptation norms
        """
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        adaptation = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        trajectory = [x.clone()]
        adaptation_history = [0.0]

        for step in range(n_steps - 1):
            # One transition: W @ x maps current state to successor
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

            # Accumulate adaptation on active neurons
            adaptation = adapt_decay * adaptation + adapt_rate * x_new
            x = x_new

        return trajectory, adaptation_history


# =============================================================================
# HIPPOCAMPAL SYSTEM
# =============================================================================

class HippocampalSystemTemporal:
    """
    Full hippocampal circuit with pure temporal association in CA3.

    Two pathways into CA3:
      - Trisynaptic (EC → DG → CA3): used during encoding. DG pattern-
        separates the input so temporal associations stay clean.
      - Direct (EC → CA3): used during retrieval. Learned associative
        mapping from EC space to the DG-separated CA3 representations.
        Preserves similarity structure, so noisy/partial cues map to
        nearby CA3 states where the successor map can grab them.

    During encoding, both pathways are trained simultaneously:
      - DG→CA3 successor map: W_ca3 += outer(curr_dg, prev_dg)
      - Direct projection:    W_direct += outer(dg_out, ec_stellate)
        This teaches the direct pathway to produce the same CA3 states
        that DG produces, but through a route that preserves input
        similarity rather than destroying it.
    """

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

        # Direct EC → CA3 pathway
        self.W_direct = torch.zeros((N_ca3, d_ec), device=device, dtype=dtype)
        self.direct_lr = direct_lr
        self.direct_decay = direct_decay
        self.k_ca3 = k_ca3

        # Temporal context
        self._prev_dg_pattern = None
        self._in_sequence = False

    def begin_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = True

    def end_sequence(self):
        self._prev_dg_pattern = None
        self._in_sequence = False

    def encode_single(self, ec_input):
        """Encode one pattern. Trains both DG→CA3 and direct EC→CA3 pathways."""
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)

        # Store temporal association in CA3 (via DG, for separation)
        prev = self._prev_dg_pattern if self._in_sequence else None
        self.ca3.store_online(dg_out, prev_pattern=prev)

        if self._in_sequence:
            self._prev_dg_pattern = dg_out.clone()

        # Train direct pathway: learn to map EC stellate → DG-separated CA3 state
        dg_norm = dg_out / (torch.linalg.norm(dg_out) + 1e-10)
        stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
        self.W_direct += self.direct_lr * torch.outer(dg_norm, stel_norm)
        self.W_direct *= self.direct_decay

        # CA1 and subiculum (unchanged)
        self.ca1.encode(dg_out, stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
        self.sub.encode(ca1_out, pyramidal)

        return dg_out, mm

    def _raw_direct_cue(self, stellate):
        """Raw direct pathway drive (pre-competition)."""
        return torch.relu(self.W_direct @ stellate)

    def _raw_dg_cue(self, stellate):
        """Raw DG pathway drive (pre-competition)."""
        return self.dg.forward(stellate)

    def _apply_ca3_competition(self, raw_drive):
        """Apply k-WTA and normalize to produce a CA3 cue."""
        h = apply_kwta(raw_drive, self.k_ca3)
        norm = torch.linalg.norm(h)
        if norm > 1e-10:
            h = h / norm
        return h

    def recall_sequence(self, ec_input, n_steps, adapt_rate=0.15,
                        adapt_decay=0.85):
        """
        Retrieve via combined pathways (default).

        Both DG and direct pathway project to CA3 simultaneously.
        Their raw drives are summed, then k-WTA competition selects
        the neurons with the most convergent input. When both pathways
        agree, the cue is stronger. When they disagree, the more
        coherent signal wins.
        """
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
        """Retrieve via direct pathway only (for comparison)."""
        stellate, _ = self.ec_sup.forward(ec_input)
        raw = self._raw_direct_cue(stellate)
        ca3_cue = self._apply_ca3_competition(raw)
        trajectory, adapt_hist = self.ca3.retrieve_sequence(
            ca3_cue, n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)
        return trajectory, adapt_hist

    def recall_sequence_via_dg(self, ec_input, n_steps, adapt_rate=0.15,
                               adapt_decay=0.85):
        """Retrieve via DG pathway only (for comparison)."""
        stellate, _ = self.ec_sup.forward(ec_input)
        raw = self._raw_dg_cue(stellate)
        ca3_cue = self._apply_ca3_competition(raw)
        trajectory, adapt_hist = self.ca3.retrieve_sequence(
            ca3_cue, n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay)
        return trajectory, adapt_hist


# =============================================================================
# SYNTHETIC DATA
# =============================================================================

def generate_sequences(n_sequences, seq_length, d_ec, device='cpu',
                       dtype=torch.float32, seed=42):
    """Generate sequences of random unit vectors."""
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


def generate_sequences_with_shared_patterns(n_sequences, seq_length, d_ec,
                                            shared_positions, similarity=0.8,
                                            device='cpu', dtype=torch.float32,
                                            seed=42):
    """
    Generate sequences where certain positions share similar patterns
    across sequences, simulating overlapping cortical activity.

    At each position in shared_positions, a base pattern is generated
    and each sequence gets a version perturbed by (1-similarity) noise.
    All other positions are independent random patterns.

    Args:
        shared_positions: list of ints, which positions share patterns
        similarity: 0-1, how similar the shared patterns are across
                    sequences (1.0 = identical, 0.0 = independent)
    """
    rng = np.random.RandomState(seed)

    # Generate base patterns for shared positions
    base_shared = {}
    for pos in shared_positions:
        p = rng.randn(d_ec).astype(np.float32)
        p = p / (np.linalg.norm(p) + 1e-10)
        base_shared[pos] = p

    sequences = []
    for _ in range(n_sequences):
        seq = []
        for j in range(seq_length):
            if j in shared_positions:
                # Perturb the shared base pattern
                noise = rng.randn(d_ec).astype(np.float32)
                noise = noise / (np.linalg.norm(noise) + 1e-10)
                p = similarity * base_shared[j] + (1 - similarity) * noise
                p = p / (np.linalg.norm(p) + 1e-10)
            else:
                p = rng.randn(d_ec).astype(np.float32)
                p = p / (np.linalg.norm(p) + 1e-10)
            seq.append(torch.tensor(p, device=device, dtype=dtype))
        sequences.append(seq)
    return sequences


# =============================================================================
# ENCODING HELPER
# =============================================================================

def encode_all_sequences(hippo, sequences, n_repetitions=1):
    """Feed all sequences through the hippocampal system, return DG patterns."""
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


# =============================================================================
# SIMILARITY HELPERS
# =============================================================================

def compute_similarity_matrix(trajectory, reference_patterns):
    T = len(trajectory)
    R = len(reference_patterns)
    sim = np.zeros((T, R))
    for t in range(T):
        for r in range(R):
            sim[t, r] = cosine_sim(trajectory[t], reference_patterns[r])
    return sim


def measure_sequentiality(sim_matrix):
    """Fraction of steps where the peak similarity is at the correct index."""
    n_steps = sim_matrix.shape[0]
    peaks = [int(np.argmax(sim_matrix[t, :])) for t in range(n_steps)]
    correct = sum(1.0 for t in range(n_steps) if peaks[t] == t)
    return correct / n_steps, peaks


def measure_diagonal(sim_matrix):
    """Mean cosine similarity along the diagonal."""
    n = min(sim_matrix.shape)
    return np.mean([sim_matrix[t, t] for t in range(n)])


# =============================================================================
# TEST 1: Repetition Sweep
# =============================================================================

def test_repetition_sweep(hippo_kwargs, device, dtype):
    """
    Core test: does attractor-like stability emerge from repetition alone?

    With pure temporal association, the only way to build transient basins
    is through repeated sequential exposure. This test sweeps n_repetitions
    and measures both sequentiality (does the trajectory advance correctly?)
    and diagonal similarity (how faithfully does each step match its target?).

    Prediction: low reps = unstable/chaotic, sufficient reps = clean
    sequences, transition should be gradual (no phase cliff).
    """
    print("\n" + "=" * 70)
    print("TEST 1: Repetition Sweep")
    print("=" * 70)

    seq_length = 6
    repetition_counts = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

    # Use the same base sequences and same random seeds for the upstream
    # components (EC, DG) so that we can compare across repetition counts.
    # To do this properly, we regenerate the hippo system each time.
    sequences = generate_sequences(1, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=100)

    results = {}
    for n_rep in repetition_counts:
        # Fresh system each time
        torch.manual_seed(42)
        hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)
        dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_rep)

        trajectory, adapt_hist = hippo.recall_sequence(
            sequences[0][0], n_steps=seq_length,
            adapt_rate=0.25, adapt_decay=0.85)

        sim = compute_similarity_matrix(trajectory, dg_patterns[0])
        seq_score, peaks = measure_sequentiality(sim)
        diag_score = measure_diagonal(sim)

        # Also measure: does the network go silent? (norm of last retrieved state)
        final_norm = float(torch.linalg.norm(trajectory[-1]))

        results[n_rep] = {
            'sim_matrix': sim,
            'sequentiality': seq_score,
            'mean_diag': diag_score,
            'peaks': peaks,
            'final_norm': final_norm,
            'adapt_hist': adapt_hist,
        }

        print(f"  reps={n_rep:3d}: seq={seq_score:.2f}, "
              f"diag={diag_score:.4f}, peaks={peaks}, "
              f"final_norm={final_norm:.4f}")

    return results, repetition_counts


# =============================================================================
# TEST 2: Single Sequence Retrieval (at sufficient repetitions)
# =============================================================================

def test_single_sequence(hippo_kwargs, device, dtype, n_repetitions=10):
    """Detailed look at a single sequence with enough repetitions."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Single Sequence Retrieval (n_rep={n_repetitions})")
    print("=" * 70)

    seq_length = 8
    sequences = generate_sequences(1, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=200)

    torch.manual_seed(42)
    hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    trajectory, adapt_hist = hippo.recall_sequence(
        sequences[0][0], n_steps=seq_length,
        adapt_rate=0.25, adapt_decay=0.85)

    sim = compute_similarity_matrix(trajectory, dg_patterns[0])
    seq_score, peaks = measure_sequentiality(sim)
    diag_score = measure_diagonal(sim)
    diag_sims = [sim[t, t] for t in range(seq_length)]

    off_diag = []
    for t in range(seq_length):
        for r in range(seq_length):
            if t != r:
                off_diag.append(sim[t, r])
    mean_off_diag = np.mean(off_diag)

    print(f"  Sequence length: {seq_length}")
    print(f"  Diagonal similarity (mean):     {diag_score:.4f}")
    print(f"  Off-diagonal similarity (mean):  {mean_off_diag:.4f}")
    print(f"  Separation:                      {diag_score - mean_off_diag:.4f}")
    print(f"  Sequentiality:                   {seq_score:.2f}")
    print(f"  Per-step diagonal: {['%.3f' % s for s in diag_sims]}")
    print(f"  Peak indices:      {peaks}")
    print(f"  Adaptation norms:  {['%.3f' % a for a in adapt_hist]}")

    return sim, diag_sims, adapt_hist, seq_score


# =============================================================================
# TEST 3: Multi-Sequence Separation
# =============================================================================

def test_multi_sequence(hippo_kwargs, device, dtype, n_repetitions=10):
    """Store 4 independent sequences, verify they don't bleed into each other."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Multi-Sequence Separation (n_rep={n_repetitions})")
    print("=" * 70)

    n_sequences = 4
    seq_length = 6
    sequences = generate_sequences(n_sequences, seq_length,
                                   hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=300)

    torch.manual_seed(42)
    hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    all_ref = []
    for seq_dg in dg_patterns:
        all_ref.extend(seq_dg)

    sim_matrices = []
    results = []

    for seq_idx in range(n_sequences):
        trajectory, _ = hippo.recall_sequence(
            sequences[seq_idx][0], n_steps=seq_length,
            adapt_rate=0.25, adapt_decay=0.85)

        sim_all = compute_similarity_matrix(trajectory, all_ref)
        sim_matrices.append(sim_all)

        sim_own = compute_similarity_matrix(trajectory, dg_patterns[seq_idx])
        diag_sims = [sim_own[t, t] for t in range(seq_length)]

        other_sims = []
        for other_idx in range(n_sequences):
            if other_idx == seq_idx:
                continue
            sim_other = compute_similarity_matrix(trajectory, dg_patterns[other_idx])
            other_sims.extend(sim_other.flatten().tolist())

        mean_own = np.mean(diag_sims)
        mean_other = np.mean(other_sims)
        seq_score, peaks = measure_sequentiality(sim_own)

        results.append({
            'seq_idx': seq_idx,
            'mean_own_diag': mean_own,
            'mean_cross_seq': mean_other,
            'sequentiality': seq_score,
            'peaks': peaks,
        })

        print(f"  Seq {seq_idx}: own_diag={mean_own:.4f}, "
              f"cross={mean_other:.4f}, "
              f"sep={mean_own - mean_other:.4f}, "
              f"seq={seq_score:.2f}, peaks={peaks}")

    return sim_matrices, results, dg_patterns


# =============================================================================
# TEST 4: Mid-Sequence Entry
# =============================================================================

def test_mid_sequence_entry(hippo_kwargs, device, dtype, n_repetitions=10):
    """
    Cue with noisy mid-sequence pattern via three retrieval modes,
    with detailed diagnostics on what each pathway is producing.
    """
    print("\n" + "=" * 70)
    print(f"TEST 4: Mid-Sequence Entry (n_rep={n_repetitions})")
    print("=" * 70)

    seq_length = 8
    sequences = generate_sequences(1, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=400)

    torch.manual_seed(42)
    hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    results = []
    entry_points = [0, 1, 2, 3, 4, 5, 6, 7]

    pathway_methods = {
        'combined': hippo.recall_sequence,
        'direct': hippo.recall_sequence_via_direct,
        'dg': hippo.recall_sequence_via_dg,
    }

    # ---- Diagnostics section ----
    print("\n  --- DIAGNOSTICS: Per-position cue quality ---")
    print("  (How well does each pathway map the EC input to the correct CA3 state?)\n")

    # First: what does each pathway produce for CLEAN inputs at every position?
    print("  CLEAN cue diagnostics (no noise, baseline):")
    for pos in range(seq_length):
        clean_ec = sequences[0][pos]
        stellate, _ = hippo.ec_sup.forward(clean_ec)

        # DG pathway output for clean input
        raw_dg_clean = hippo._raw_dg_cue(stellate)
        cue_dg_clean = hippo._apply_ca3_competition(raw_dg_clean)

        # Direct pathway output for clean input
        raw_direct_clean = hippo._raw_direct_cue(stellate)
        cue_direct_clean = hippo._apply_ca3_competition(raw_direct_clean)

        # How similar is each pathway's cue to the stored DG pattern?
        stored = dg_patterns[0][pos]
        sim_dg_to_stored = cosine_sim(cue_dg_clean, stored)
        sim_direct_to_stored = cosine_sim(cue_direct_clean, stored)

        # Raw magnitudes
        mag_dg = float(torch.linalg.norm(raw_dg_clean))
        mag_direct = float(torch.linalg.norm(raw_direct_clean))

        # Top-k overlap: how many of the k_ca3 neurons selected by each
        # pathway are the same?
        k = hippo.k_ca3
        _, idx_dg = torch.topk(raw_dg_clean, k)
        _, idx_direct = torch.topk(raw_direct_clean, k)
        overlap = len(set(idx_dg.tolist()) & set(idx_direct.tolist()))

        # Which stored pattern does each cue most resemble?
        best_dg = max(range(seq_length),
                      key=lambda i: cosine_sim(cue_dg_clean, dg_patterns[0][i]))
        best_direct = max(range(seq_length),
                          key=lambda i: cosine_sim(cue_direct_clean, dg_patterns[0][i]))

        print(f"    pos {pos}: DG→stored={sim_dg_to_stored:.3f} (best={best_dg}), "
              f"Direct→stored={sim_direct_to_stored:.3f} (best={best_direct}), "
              f"mag_dg={mag_dg:.1f}, mag_direct={mag_direct:.1f}, "
              f"top-k overlap={overlap}/{k}")

    # Now: noisy inputs
    print("\n  NOISY cue diagnostics (0.3 noise scale):")
    for pos in range(seq_length):
        clean_ec = sequences[0][pos]
        noise = torch.randn_like(clean_ec) * 0.3
        noisy_ec = clean_ec + noise
        noisy_ec = noisy_ec / (torch.linalg.norm(noisy_ec) + 1e-10)

        # EC-space similarity between noisy and clean
        ec_sim = cosine_sim(noisy_ec, clean_ec)

        stellate_noisy, _ = hippo.ec_sup.forward(noisy_ec)
        stellate_clean, _ = hippo.ec_sup.forward(clean_ec)
        stel_sim = cosine_sim(stellate_noisy, stellate_clean)

        # DG pathway
        raw_dg = hippo._raw_dg_cue(stellate_noisy)
        cue_dg = hippo._apply_ca3_competition(raw_dg)
        raw_dg_clean = hippo._raw_dg_cue(stellate_clean)
        cue_dg_clean = hippo._apply_ca3_competition(raw_dg_clean)

        # Direct pathway
        raw_direct = hippo._raw_direct_cue(stellate_noisy)
        cue_direct = hippo._apply_ca3_competition(raw_direct)

        stored = dg_patterns[0][pos]

        # Similarities
        sim_dg_noisy_to_stored = cosine_sim(cue_dg, stored)
        sim_dg_clean_to_stored = cosine_sim(cue_dg_clean, stored)
        sim_dg_noisy_to_clean = cosine_sim(cue_dg, cue_dg_clean)
        sim_direct_to_stored = cosine_sim(cue_direct, stored)

        # Magnitudes
        mag_dg = float(torch.linalg.norm(raw_dg))
        mag_direct = float(torch.linalg.norm(raw_direct))

        # Top-k overlap between pathways (on noisy input)
        k = hippo.k_ca3
        _, idx_dg = torch.topk(raw_dg, k)
        _, idx_direct = torch.topk(raw_direct, k)
        overlap = len(set(idx_dg.tolist()) & set(idx_direct.tolist()))

        # Which stored pattern does each noisy cue most resemble?
        best_dg = max(range(seq_length),
                      key=lambda i: cosine_sim(cue_dg, dg_patterns[0][i]))
        best_direct = max(range(seq_length),
                          key=lambda i: cosine_sim(cue_direct, dg_patterns[0][i]))

        print(f"    pos {pos}: ec_sim={ec_sim:.3f}, stel_sim={stel_sim:.3f}, "
              f"DG_noisy→stored={sim_dg_noisy_to_stored:.3f} (best={best_dg}), "
              f"DG_noisy→DG_clean={sim_dg_noisy_to_clean:.3f}, "
              f"Direct→stored={sim_direct_to_stored:.3f} (best={best_direct}), "
              f"mag_dg={mag_dg:.1f}, mag_direct={mag_direct:.1f}, "
              f"overlap={overlap}/{k}")

    # ---- Retrieval tests ----
    print("\n  --- RETRIEVAL RESULTS ---")

    entry_points_retrieval = [1, 2, 4, 6]

    for entry_pos in entry_points_retrieval:
        clean_ec = sequences[0][entry_pos]
        noise = torch.randn_like(clean_ec) * 0.00
        noisy_ec = clean_ec + noise
        noisy_ec = noisy_ec / (torch.linalg.norm(noisy_ec) + 1e-10)

        remaining = seq_length - entry_pos
        ref_slice = dg_patterns[0][entry_pos:]

        entry_result = {'entry_pos': entry_pos}

        for name, method in pathway_methods.items():
            traj, _ = method(noisy_ec, n_steps=remaining,
                             adapt_rate=0.25, adapt_decay=0.85)
            sim_cont = compute_similarity_matrix(traj, ref_slice)
            diag_sims = [sim_cont[t, t] for t in range(len(ref_slice))]
            sim_full = compute_similarity_matrix(traj, dg_patterns[0])
            peaks = [int(np.argmax(sim_full[t, :])) for t in range(remaining)]

            entry_result[name] = {
                'diag_sims': diag_sims,
                'mean_diag': np.mean(diag_sims),
                'full_peaks': peaks,
                'sim_full': sim_full,
            }

        results.append(entry_result)

        print(f"  Entry at pos {entry_pos} ({remaining} steps remaining):")
        for name in ['combined', 'direct', 'dg']:
            r = entry_result[name]
            label = name.upper().ljust(8)
            print(f"    {label}: mean={r['mean_diag']:.4f}, "
                  f"sims={['%.3f' % s for s in r['diag_sims']]}, "
                  f"peaks={r['full_peaks']}")

    return results


# =============================================================================
# TEST 5: Cortical Similarity / Cross-Sequence Generalization
# =============================================================================

def test_cortical_similarity(hippo_kwargs, device, dtype, n_repetitions=10):
    """
    Generate sequences that share similar patterns at certain positions,
    simulating overlapping cortical activity across episodes.

    Test: when two sequences share a similar pattern at position P, does
    cueing one sequence and letting it flow through P cause any bleed
    into the other sequence's continuation? This would demonstrate that
    overlapping temporal associations create generalization, i.e. the
    same mechanism that creates sequential flow also creates the
    hippocampus's ability to link related experiences.

    Setup:
      - 3 sequences of length 6
      - Position 3 is shared (high similarity across sequences)
      - All other positions are independent
    """
    print("\n" + "=" * 70)
    print(f"TEST 5: Cortical Similarity / Generalization (n_rep={n_repetitions})")
    print("=" * 70)

    n_sequences = 3
    seq_length = 6
    shared_pos = [3]
    similarity = 0.85

    sequences = generate_sequences_with_shared_patterns(
        n_sequences, seq_length, hippo_kwargs['d_ec'],
        shared_positions=shared_pos, similarity=similarity,
        device=device, dtype=dtype, seed=500)

    # Verify: patterns at shared position should be similar
    print(f"  Shared position: {shared_pos}, similarity param: {similarity}")
    for i in range(n_sequences):
        for j in range(i + 1, n_sequences):
            sim_at_shared = cosine_sim(sequences[i][shared_pos[0]],
                                       sequences[j][shared_pos[0]])
            sim_at_other = cosine_sim(sequences[i][0], sequences[j][0])
            print(f"    Seq {i} vs {j}: sim at shared pos = {sim_at_shared:.4f}, "
                  f"sim at pos 0 = {sim_at_other:.4f}")

    torch.manual_seed(42)
    hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    # For each sequence, retrieve from start and measure what happens
    # at and after the shared position
    all_ref = []
    for seq_dg in dg_patterns:
        all_ref.extend(seq_dg)

    results = []
    for seq_idx in range(n_sequences):
        trajectory, _ = hippo.recall_sequence(
            sequences[seq_idx][0], n_steps=seq_length,
            adapt_rate=0.25, adapt_decay=0.85)

        sim_all = compute_similarity_matrix(trajectory, all_ref)

        # At the shared position (step 3), which sequence's pattern does
        # the trajectory most resemble?
        step_at_shared = shared_pos[0]
        if step_at_shared < len(trajectory):
            sims_at_shared = []
            for other_idx in range(n_sequences):
                s = cosine_sim(trajectory[step_at_shared],
                               dg_patterns[other_idx][step_at_shared])
                sims_at_shared.append(s)

            # After the shared position, does the trajectory follow the
            # original sequence or get pulled toward another?
            post_shared_own = []
            post_shared_other = []
            for t in range(step_at_shared + 1, min(seq_length, len(trajectory))):
                own_sim = cosine_sim(trajectory[t], dg_patterns[seq_idx][t])
                post_shared_own.append(own_sim)
                for other_idx in range(n_sequences):
                    if other_idx != seq_idx:
                        other_sim = cosine_sim(trajectory[t],
                                               dg_patterns[other_idx][t])
                        post_shared_other.append(other_sim)

            results.append({
                'seq_idx': seq_idx,
                'sims_at_shared': sims_at_shared,
                'post_shared_own': post_shared_own,
                'post_shared_other': post_shared_other,
                'sim_all': sim_all,
            })

            print(f"  Seq {seq_idx}: at shared pos, sims to each seq = "
                  f"{['%.3f' % s for s in sims_at_shared]}")
            print(f"    Post-shared own seq sim:   "
                  f"{['%.3f' % s for s in post_shared_own]} "
                  f"(mean {np.mean(post_shared_own):.4f})" if post_shared_own else "    (none)")
            print(f"    Post-shared other seq sim: mean "
                  f"{np.mean(post_shared_other):.4f}" if post_shared_other else "    (none)")

    return results, dg_patterns, sequences


# =============================================================================
# TEST 6: Adaptation Rate Sweep (at sufficient repetitions)
# =============================================================================

def test_adaptation_sweep(hippo_kwargs, device, dtype, n_repetitions=10):
    """Find the adaptation regime that works with pure temporal learning."""
    print("\n" + "=" * 70)
    print(f"TEST 6: Adaptation Rate Sweep (n_rep={n_repetitions})")
    print("=" * 70)

    seq_length = 6
    sequences = generate_sequences(1, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=600)

    torch.manual_seed(42)
    hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    adapt_rates = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.70]
    adapt_decays = [0.70, 0.85, 0.95]

    results = {}

    for decay in adapt_decays:
        for rate in adapt_rates:
            trajectory, adapt_hist = hippo.recall_sequence(
                sequences[0][0], n_steps=seq_length,
                adapt_rate=rate, adapt_decay=decay)

            sim = compute_similarity_matrix(trajectory, dg_patterns[0])
            seq_score, peaks = measure_sequentiality(sim)
            diag_score = measure_diagonal(sim)

            results[(rate, decay)] = {
                'sim_matrix': sim,
                'sequentiality': seq_score,
                'mean_diag': diag_score,
                'peaks': peaks,
            }

    # Print full grid
    for decay in adapt_decays:
        print(f"\n  decay={decay:.2f}:")
        for rate in adapt_rates:
            r = results[(rate, decay)]
            print(f"    rate={rate:.2f}: "
                  f"seq={r['sequentiality']:.2f}, "
                  f"diag={r['mean_diag']:.4f}, peaks={r['peaks']}")

    return results, adapt_rates, adapt_decays


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_results(t1, t2, t3, t4, t5, t6, save_path):
    fig = plt.figure(figsize=(24, 32))
    gs = GridSpec(6, 4, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle("Hippocampal Sequence Attractors v2: Pure Temporal Association",
                 fontsize=16, fontweight='bold', y=0.99)

    # -------------------------------------------------------------------------
    # Row 1: Test 1 - Repetition sweep
    # -------------------------------------------------------------------------
    results_1, rep_counts_1 = t1

    ax1a = fig.add_subplot(gs[0, 0:2])
    seq_scores = [results_1[n]['sequentiality'] for n in rep_counts_1]
    diag_scores = [results_1[n]['mean_diag'] for n in rep_counts_1]
    ax1a.plot(rep_counts_1, seq_scores, 'o-', color='steelblue',
              linewidth=2, markersize=6, label='Sequentiality')
    ax1a.plot(rep_counts_1, diag_scores, 's-', color='coral',
              linewidth=2, markersize=6, label='Mean diag sim')
    ax1a.set_xlabel("Number of repetitions")
    ax1a.set_ylabel("Score")
    ax1a.set_title("Test 1: Repetition Sweep")
    ax1a.legend()
    ax1a.set_ylim(-0.1, 1.1)
    ax1a.grid(True, alpha=0.3)

    # Show sim matrices at a few key repetition counts
    key_reps = [1, 5, 15, 50]
    for ki, n_rep in enumerate(key_reps):
        if n_rep in results_1:
            ax = fig.add_subplot(gs[0, 2]) if ki < 2 else fig.add_subplot(gs[0, 3])
            if ki % 2 == 0:
                sim = results_1[n_rep]['sim_matrix']
                im = ax.imshow(sim, aspect='auto', cmap='viridis',
                               vmin=-0.2, vmax=1.0)
                ax.set_title(f"Test 1: {n_rep} reps")
                ax.set_xlabel("Stored pattern")
                ax.set_ylabel("Retrieved step")

    # -------------------------------------------------------------------------
    # Row 2: Test 2 - Single sequence detail
    # -------------------------------------------------------------------------
    sim_2, diag_sims_2, adapt_hist_2, seq_score_2 = t2

    ax2a = fig.add_subplot(gs[1, 0:2])
    im = ax2a.imshow(sim_2, aspect='auto', cmap='viridis', vmin=-0.2, vmax=1.0)
    ax2a.set_xlabel("Stored pattern index")
    ax2a.set_ylabel("Retrieved step")
    ax2a.set_title(f"Test 2: Single Sequence (seq={seq_score_2:.2f})")
    plt.colorbar(im, ax=ax2a, label="Cosine similarity")

    ax2b = fig.add_subplot(gs[1, 2])
    ax2b.plot(diag_sims_2, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax2b.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2b.set_xlabel("Sequence step")
    ax2b.set_ylabel("Cosine similarity")
    ax2b.set_title("Test 2: Diagonal Similarities")
    ax2b.set_ylim(-0.3, 1.1)
    ax2b.grid(True, alpha=0.3)

    ax2c = fig.add_subplot(gs[1, 3])
    ax2c.plot(adapt_hist_2, 's-', color='coral', linewidth=2, markersize=6)
    ax2c.set_xlabel("Sequence step")
    ax2c.set_ylabel("Adaptation norm")
    ax2c.set_title("Test 2: Adaptation")
    ax2c.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Row 3: Test 3 - Multi-sequence
    # -------------------------------------------------------------------------
    sim_matrices_3, results_3, dg_patterns_3 = t3

    for i in range(min(4, len(sim_matrices_3))):
        ax = fig.add_subplot(gs[2, i])
        im = ax.imshow(sim_matrices_3[i], aspect='auto', cmap='viridis',
                       vmin=-0.2, vmax=1.0)
        ax.set_xlabel("All stored patterns")
        ax.set_ylabel("Retrieved step")
        r = results_3[i]
        ax.set_title(f"Test 3: Seq {i} (seq={r['sequentiality']:.2f})")
        n_per_seq = len(dg_patterns_3[0])
        for boundary in range(1, len(dg_patterns_3)):
            ax.axvline(x=boundary * n_per_seq - 0.5, color='white',
                       linestyle='--', linewidth=1, alpha=0.7)

    # -------------------------------------------------------------------------
    # Row 4: Test 4 - Mid-sequence entry (combined vs direct vs DG)
    # -------------------------------------------------------------------------
    results_4 = t4

    for i, res in enumerate(results_4[:3]):
        ax = fig.add_subplot(gs[3, i])
        sim_full = res['combined']['sim_full']
        im = ax.imshow(sim_full, aspect='auto', cmap='viridis',
                       vmin=-0.2, vmax=1.0)
        ax.set_xlabel("Stored pattern index")
        ax.set_ylabel("Retrieved step")
        entry = res['entry_pos']
        ax.set_title(f"Test 4: Combined, pos {entry}")
        for t in range(sim_full.shape[0]):
            expected_r = entry + t
            if expected_r < sim_full.shape[1]:
                ax.plot(expected_r, t, 'wx', markersize=8, markeredgewidth=2)

    ax4d = fig.add_subplot(gs[3, 3])
    positions = [r['entry_pos'] for r in results_4]
    means_combined = [r['combined']['mean_diag'] for r in results_4]
    means_direct = [r['direct']['mean_diag'] for r in results_4]
    means_dg = [r['dg']['mean_diag'] for r in results_4]
    x = np.arange(len(positions))
    w = 0.25
    ax4d.bar(x - w, means_combined, w, color='forestgreen', alpha=0.8, label='Combined')
    ax4d.bar(x, means_direct, w, color='steelblue', alpha=0.8, label='Direct')
    ax4d.bar(x + w, means_dg, w, color='coral', alpha=0.8, label='DG')
    ax4d.set_xticks(x)
    ax4d.set_xticklabels([f"pos {p}" for p in positions])
    ax4d.set_ylabel("Mean continuation sim")
    ax4d.set_title("Test 4: Pathway Comparison")
    ax4d.set_ylim(0, 1.1)
    ax4d.legend(fontsize=7)
    ax4d.grid(True, alpha=0.3, axis='y')

    # -------------------------------------------------------------------------
    # Row 5: Test 5 - Cortical similarity
    # -------------------------------------------------------------------------
    results_5, dg_patterns_5, sequences_5 = t5

    for i, res in enumerate(results_5[:3]):
        ax = fig.add_subplot(gs[4, i])
        sim_all = res['sim_all']
        im = ax.imshow(sim_all, aspect='auto', cmap='viridis',
                       vmin=-0.2, vmax=1.0)
        ax.set_xlabel("All stored patterns")
        ax.set_ylabel("Retrieved step")
        ax.set_title(f"Test 5: Seq {res['seq_idx']} (shared@pos3)")
        n_per = len(dg_patterns_5[0])
        for boundary in range(1, len(dg_patterns_5)):
            ax.axvline(x=boundary * n_per - 0.5, color='white',
                       linestyle='--', linewidth=1, alpha=0.7)
        # Mark shared position
        ax.axhline(y=3 - 0.5, color='red', linestyle=':', alpha=0.5)
        ax.axhline(y=3 + 0.5, color='red', linestyle=':', alpha=0.5)

    # Summary: post-shared continuation fidelity
    ax5d = fig.add_subplot(gs[4, 3])
    for res in results_5:
        idx = res['seq_idx']
        own = res.get('post_shared_own', [])
        if own:
            ax5d.plot(range(len(own)), own, 'o-', label=f"Seq {idx} own",
                      markersize=5, linewidth=2)
    ax5d.set_xlabel("Steps after shared position")
    ax5d.set_ylabel("Own-sequence similarity")
    ax5d.set_title("Test 5: Post-Shared Fidelity")
    ax5d.legend(fontsize=8)
    ax5d.set_ylim(-0.1, 1.1)
    ax5d.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Row 6: Test 6 - Adaptation sweep
    # -------------------------------------------------------------------------
    results_6, rates_6, decays_6 = t6

    for di, decay in enumerate(decays_6):
        ax = fig.add_subplot(gs[5, di])
        seq_by_rate = [results_6[(r, decay)]['sequentiality'] for r in rates_6]
        diag_by_rate = [results_6[(r, decay)]['mean_diag'] for r in rates_6]
        ax.plot(rates_6, seq_by_rate, 'o-', color='steelblue',
                linewidth=2, label='Sequentiality', markersize=5)
        ax.plot(rates_6, diag_by_rate, 's-', color='coral',
                linewidth=2, label='Mean diag sim', markersize=5)
        ax.set_xlabel("Adaptation rate")
        ax.set_ylabel("Score")
        ax.set_title(f"Test 6: decay={decay:.2f}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    # Best sim matrix
    ax6d = fig.add_subplot(gs[5, 3])
    best_key = max(
        [(r, d) for r in rates_6 for d in decays_6 if (r, d) in results_6],
        key=lambda k: (results_6[k]['sequentiality'], results_6[k]['mean_diag']))
    best_sim = results_6[best_key]['sim_matrix']
    im = ax6d.imshow(best_sim, aspect='auto', cmap='viridis', vmin=-0.2, vmax=1.0)
    ax6d.set_xlabel("Stored pattern")
    ax6d.set_ylabel("Retrieved step")
    ax6d.set_title(f"Test 6: Best (r={best_key[0]:.2f}, d={best_key[1]:.2f})")
    plt.colorbar(im, ax=ax6d, label="Cosine sim")

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

    hippo_kwargs = {
        "d_ec": d_ec,
        "D_dg": d_ec,
        "N_ca3": d_ec,
        "N_ca1": d_ec,
        "k_ca3": 50,
        "N_sub": d_ec,
        "ca3_lr": 1.0,
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

    print("=" * 70)
    print("Hippocampal Sequence Attractors v2: Pure Temporal Association")
    print("=" * 70)
    print(f"  d_ec={d_ec}, k_ca3=50, ca3_lr={hippo_kwargs['ca3_lr']}")
    print(f"  Learning rule: W += lr * outer(curr_centered, prev_centered)")
    print(f"  No symmetric component. Stability emerges from repetition.")

    t1 = test_repetition_sweep(hippo_kwargs, device, dtype)
    t2 = test_single_sequence(hippo_kwargs, device, dtype, n_repetitions=10)
    t3 = test_multi_sequence(hippo_kwargs, device, dtype, n_repetitions=10)
    t4 = test_mid_sequence_entry(hippo_kwargs, device, dtype, n_repetitions=10)
    t5 = test_cortical_similarity(hippo_kwargs, device, dtype, n_repetitions=10)
    t6 = test_adaptation_sweep(hippo_kwargs, device, dtype, n_repetitions=10)

    plot_all_results(t1, t2, t3, t4, t5, t6,
                     save_path="sequence_attractors_v2_results.png")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
