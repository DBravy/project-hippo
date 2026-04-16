"""
Hippocampal Sequence Attractors Test
=====================================

Tests whether sequential attractor dynamics emerge from:
  1. Asymmetric Hebbian learning in CA3 (temporal weight component)
  2. Adaptation-driven retrieval (destabilizes current attractor, enables flow)

No replay logic. Synthetic sequences fed directly. Five diagnostic tests:
  - Test 1: Single sequence retrieval
  - Test 2: Multi-sequence separation
  - Test 3: Partial cue / mid-sequence entry
  - Test 4: Adaptation rate sweep
  - Test 5: Alpha (sym/asym) ratio sweep
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
# UTILITY FUNCTIONS (unchanged from your codebase)
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
# MODIFIED CA3: Asymmetric Hebbian + Adaptation-Driven Retrieval
# =============================================================================

class CA3Sequential:
    """
    CA3 with two modifications for emergent sequence dynamics:

    1. Storage uses a blend of symmetric and asymmetric Hebbian learning.
       - Symmetric: W += alpha_sym * outer(p_c, p_c)
         Creates point attractors (pattern completion).
       - Asymmetric: W += alpha_asym * outer(curr_c, prev_c)
         Creates directional flow between consecutively stored patterns.

    2. Retrieval has two modes:
       - retrieve(): standard point-attractor settling (unchanged behavior)
       - retrieve_sequence(): adaptation-driven traversal that destabilizes
         each attractor in turn, producing a trajectory through state space.
    """

    def __init__(self, N, k_active, alpha_sym=0.7, alpha_asym=0.3,
                 device='cpu', dtype=torch.float32):
        self.N = N
        self.k_active = k_active
        self.alpha_sym = alpha_sym
        self.alpha_asym = alpha_asym
        self.device = device
        self.dtype = dtype

        self.W = torch.zeros((N, N), device=device, dtype=dtype)
        self.n_stored = 0
        self.mean_activity = torch.zeros(N, device=device, dtype=dtype)

    def _center(self, pattern):
        """Normalize and mean-center a pattern."""
        p = pattern / (torch.linalg.norm(pattern) + 1e-10)
        self.n_stored += 1
        self.mean_activity += (p - self.mean_activity) / self.n_stored
        p_c = p - self.mean_activity
        return p_c

    def store_online(self, pattern, prev_pattern=None):
        """
        Store a single pattern with optional temporal context.

        Args:
            pattern: current DG output (sparse, high-dimensional)
            prev_pattern: previous DG output in the sequence, or None
                          if this is the first element / standalone pattern.
        """
        p_c = self._center(pattern)

        # Symmetric component: creates basin of attraction for this pattern
        self.W += self.alpha_sym * torch.outer(p_c, p_c)

        # Asymmetric component: creates directional flow from prev -> curr
        if prev_pattern is not None:
            # prev was already stored and centered at its time of storage,
            # but we need its centered version relative to the *current* mean.
            # Using a fresh centering here would be slightly off because the
            # running mean has moved. Instead, we normalize prev and subtract
            # the current mean estimate, which is close enough and avoids
            # needing to cache centered versions.
            prev_p = prev_pattern / (torch.linalg.norm(prev_pattern) + 1e-10)
            prev_c = prev_p - self.mean_activity
            self.W += self.alpha_asym * torch.outer(p_c, prev_c)

        self.W.fill_diagonal_(0)

    def retrieve(self, cue, n_iterations=5):
        """Standard point-attractor retrieval (unchanged from original)."""
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        for _ in range(n_iterations):
            h = torch.relu(self.W @ x)
            x_new = apply_kwta(h, self.k_active)
            x_new = x_new / (torch.linalg.norm(x_new) + 1e-10)
            if torch.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x

    def retrieve_sequence(self, cue, n_steps, adapt_rate=0.15,
                          adapt_decay=0.85, settle_steps=3):
        """
        Adaptation-driven sequence retrieval.

        At each step:
          1. Settle for a few iterations (let the network approach an attractor)
          2. Record the state
          3. Accumulate adaptation on the active neurons
          4. On the next step, adaptation suppresses the current attractor,
             and asymmetric weights guide the transition to the next one.

        Args:
            cue: initial pattern to start the trajectory
            n_steps: number of sequence elements to retrieve
            adapt_rate: how quickly adaptation accumulates (higher = faster
                        transitions, less time at each attractor)
            adapt_decay: how quickly adaptation decays (lower = more persistent
                         suppression, cleaner transitions)
            settle_steps: number of settling iterations per sequence step
                          (lets the network partially converge before recording)

        Returns:
            trajectory: list of n_steps patterns (the retrieved sequence)
            adaptation_history: list of adaptation norms (for diagnostics)
        """
        x = cue / (torch.linalg.norm(cue) + 1e-10)
        adaptation = torch.zeros(self.N, device=self.device, dtype=self.dtype)
        trajectory = []
        adaptation_history = []

        for step in range(n_steps):
            # Settle: run a few iterations with current adaptation level
            for _ in range(settle_steps):
                h = torch.relu(self.W @ x - adaptation)
                x_new = apply_kwta(h, self.k_active)
                norm = torch.linalg.norm(x_new)
                if norm > 1e-10:
                    x_new = x_new / norm
                else:
                    # Network went silent; break out
                    break
                x = x_new

            trajectory.append(x.clone())
            adaptation_history.append(float(torch.linalg.norm(adaptation)))

            # Accumulate adaptation on currently active neurons
            adaptation = adapt_decay * adaptation + adapt_rate * x

        return trajectory, adaptation_history


# =============================================================================
# MODIFIED HIPPOCAMPAL SYSTEM (tracks temporal context)
# =============================================================================

class HippocampalSystemSequential:
    """
    Full hippocampal circuit with sequential encoding support.

    The key change: encode_single now accepts and propagates temporal context
    through CA3, so that consecutive calls within a sequence produce asymmetric
    weight updates. Call begin_sequence() before encoding a new sequence and
    end_sequence() after, so the system knows when temporal continuity applies.
    """

    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                 alpha_sym=0.7, alpha_asym=0.3,
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
        self.ca3 = CA3Sequential(N_ca3, k_ca3,
                                 alpha_sym=alpha_sym, alpha_asym=alpha_asym,
                                 device=device, dtype=dtype)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}),
                       device=device, dtype=dtype)
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}),
                             device=device, dtype=dtype)

        # Temporal context state
        self._prev_dg_pattern = None
        self._in_sequence = False

    def begin_sequence(self):
        """Call before encoding a new sequence. Resets temporal context."""
        self._prev_dg_pattern = None
        self._in_sequence = True

    def end_sequence(self):
        """Call after the last element of a sequence."""
        self._prev_dg_pattern = None
        self._in_sequence = False

    def encode_single(self, ec_input):
        """
        Encode one pattern. If within a sequence (between begin_sequence /
        end_sequence), passes temporal context to CA3 for asymmetric learning.
        """
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)

        # Store in CA3 with temporal context
        prev = self._prev_dg_pattern if self._in_sequence else None
        self.ca3.store_online(dg_out, prev_pattern=prev)

        # Update temporal context
        if self._in_sequence:
            self._prev_dg_pattern = dg_out.clone()

        # CA1 and subiculum encoding (unchanged)
        self.ca1.encode(dg_out, stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
        self.sub.encode(ca1_out, pyramidal)

        return dg_out, mm

    def recall_point(self, ec_input):
        """Standard single-pattern recall (unchanged behavior)."""
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
        sub_out = self.sub.replay(ca1_out)
        gated = ECDeepVb.hippocampal_output(ca1_out, sub_out)
        return gated, mm

    def recall_sequence(self, ec_input, n_steps, adapt_rate=0.15,
                        adapt_decay=0.85, settle_steps=3):
        """
        Cue with an EC input, then let CA3 dynamics unroll a trajectory.

        Returns the trajectory in DG/CA3 space (the raw attractor dynamics).
        """
        stellate, _ = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        trajectory, adapt_hist = self.ca3.retrieve_sequence(
            dg_out, n_steps,
            adapt_rate=adapt_rate, adapt_decay=adapt_decay,
            settle_steps=settle_steps)
        return trajectory, adapt_hist


# =============================================================================
# SYNTHETIC SEQUENCE GENERATION
# =============================================================================

def generate_sequences(n_sequences, seq_length, d_ec, device='cpu',
                       dtype=torch.float32, seed=42):
    """
    Generate synthetic sequences as chains of random EC-space patterns.

    Each pattern is a random unit vector in d_ec dimensions. Patterns within
    a sequence have no special structure relative to each other (they're not
    smoothly interpolated); the sequential relationship exists only because
    they're stored consecutively. This is the hardest test: the network must
    learn the sequence purely from temporal contiguity during encoding.

    Returns:
        sequences: list of lists, sequences[i][j] is the j-th EC pattern
                   in the i-th sequence
    """
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
# ENCODING HELPER
# =============================================================================

def encode_all_sequences(hippo, sequences, n_repetitions=1):
    """
    Feed all sequences through the hippocampal system.

    Args:
        hippo: HippocampalSystemSequential instance
        sequences: list of lists of EC patterns
        n_repetitions: how many times to present each sequence
                       (more repetitions = stronger weight imprint)

    Returns:
        all_dg_patterns: list of lists matching the sequence structure,
                         containing the DG representations that were
                         actually stored (these are the ground truth for
                         testing CA3 retrieval)
    """
    all_dg_patterns = [[] for _ in sequences]

    for rep in range(n_repetitions):
        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for step_idx, ec_pattern in enumerate(seq):
                dg_out, mm = hippo.encode_single(ec_pattern)
                # Only record DG patterns on the first pass
                if rep == 0:
                    all_dg_patterns[seq_idx].append(dg_out.clone())
            hippo.end_sequence()

    return all_dg_patterns


# =============================================================================
# SIMILARITY MATRIX COMPUTATION
# =============================================================================

def compute_similarity_matrix(trajectory, reference_patterns):
    """
    Compute cosine similarity between each retrieved trajectory step and
    each reference pattern.

    Args:
        trajectory: list of T retrieved patterns
        reference_patterns: list of R reference (stored) patterns

    Returns:
        sim_matrix: numpy array of shape (T, R)
    """
    T = len(trajectory)
    R = len(reference_patterns)
    sim = np.zeros((T, R))
    for t in range(T):
        for r in range(R):
            sim[t, r] = cosine_sim(trajectory[t], reference_patterns[r])
    return sim


# =============================================================================
# TEST 1: Single Sequence Retrieval
# =============================================================================

def test_single_sequence(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Store one sequence, cue with the first pattern, retrieve via adaptation
    dynamics, check if the trajectory follows the stored sequence.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Single Sequence Retrieval")
    print("=" * 70)

    seq_length = 8
    sequences = generate_sequences(1, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=100)

    hippo = HippocampalSystemSequential(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    # Retrieve: cue with first EC pattern, ask for seq_length steps
    trajectory, adapt_hist = hippo.recall_sequence(
        sequences[0][0], n_steps=seq_length,
        adapt_rate=0.15, adapt_decay=0.85, settle_steps=3)

    # Compute similarity matrix (retrieved steps vs stored DG patterns)
    sim_matrix = compute_similarity_matrix(trajectory, dg_patterns[0])

    # Diagonal similarity (does step t match stored pattern t?)
    diag_sims = [sim_matrix[t, t] for t in range(seq_length)]
    mean_diag = np.mean(diag_sims)

    # Off-diagonal mean (how much leakage into wrong patterns?)
    off_diag = []
    for t in range(seq_length):
        for r in range(seq_length):
            if t != r:
                off_diag.append(sim_matrix[t, r])
    mean_off_diag = np.mean(off_diag)

    print(f"  Sequence length: {seq_length}")
    print(f"  Repetitions during encoding: {n_repetitions}")
    print(f"  Diagonal similarity (mean):     {mean_diag:.4f}")
    print(f"  Off-diagonal similarity (mean):  {mean_off_diag:.4f}")
    print(f"  Separation (diag - off_diag):    {mean_diag - mean_off_diag:.4f}")
    print(f"  Per-step diagonal: {['%.3f' % s for s in diag_sims]}")
    print(f"  Adaptation norms:  {['%.3f' % a for a in adapt_hist]}")

    return sim_matrix, diag_sims, adapt_hist


# =============================================================================
# TEST 2: Multi-Sequence Separation
# =============================================================================

def test_multi_sequence(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Store 4 sequences, cue each one, verify trajectories follow the correct
    sequence without bleeding into others.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Multi-Sequence Separation")
    print("=" * 70)

    n_sequences = 4
    seq_length = 6
    sequences = generate_sequences(n_sequences, seq_length,
                                   hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=200)

    hippo = HippocampalSystemSequential(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    # Flatten all DG patterns for the full similarity matrix
    all_ref = []
    for seq_dg in dg_patterns:
        all_ref.extend(seq_dg)

    sim_matrices = []
    results = []

    for seq_idx in range(n_sequences):
        trajectory, adapt_hist = hippo.recall_sequence(
            sequences[seq_idx][0], n_steps=seq_length,
            adapt_rate=0.15, adapt_decay=0.85, settle_steps=3)

        # Similarity against ALL stored patterns (all sequences)
        sim_all = compute_similarity_matrix(trajectory, all_ref)
        sim_matrices.append(sim_all)

        # Similarity against own sequence only
        own_ref = dg_patterns[seq_idx]
        sim_own = compute_similarity_matrix(trajectory, own_ref)
        diag_sims = [sim_own[t, t] for t in range(seq_length)]

        # Similarity against other sequences' patterns
        other_sims = []
        for other_idx in range(n_sequences):
            if other_idx == seq_idx:
                continue
            sim_other = compute_similarity_matrix(trajectory, dg_patterns[other_idx])
            other_sims.extend(sim_other.flatten().tolist())

        mean_own = np.mean(diag_sims)
        mean_other = np.mean(other_sims)
        results.append({
            'seq_idx': seq_idx,
            'mean_own_diag': mean_own,
            'mean_cross_seq': mean_other,
            'diag_sims': diag_sims,
        })

        print(f"  Seq {seq_idx}: own_diag={mean_own:.4f}, "
              f"cross_seq={mean_other:.4f}, "
              f"separation={mean_own - mean_other:.4f}")

    return sim_matrices, results, all_ref, dg_patterns


# =============================================================================
# TEST 3: Partial Cue / Mid-Sequence Entry
# =============================================================================

def test_mid_sequence_entry(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Store a sequence, then cue with a noisy version of a mid-sequence pattern.
    Check: does it snap to that pattern and then continue forward?
    """
    print("\n" + "=" * 70)
    print("TEST 3: Partial Cue / Mid-Sequence Entry")
    print("=" * 70)

    seq_length = 8
    sequences = generate_sequences(1, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=300)

    hippo = HippocampalSystemSequential(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    results = []
    entry_points = [2, 4, 6]  # mid-sequence positions to test

    for entry_pos in entry_points:
        # Create noisy cue: the original EC pattern + gaussian noise
        clean_ec = sequences[0][entry_pos]
        noise = torch.randn_like(clean_ec) * 0.3
        noisy_ec = clean_ec + noise
        noisy_ec = noisy_ec / (torch.linalg.norm(noisy_ec) + 1e-10)

        remaining_steps = seq_length - entry_pos
        trajectory, adapt_hist = hippo.recall_sequence(
            noisy_ec, n_steps=remaining_steps,
            adapt_rate=0.15, adapt_decay=0.85, settle_steps=3)

        # Check: does step 0 of trajectory match stored pattern at entry_pos?
        # Does step 1 match entry_pos+1? etc.
        ref_slice = dg_patterns[0][entry_pos:]
        sim = compute_similarity_matrix(trajectory, ref_slice)
        diag_sims = [sim[t, t] for t in range(len(ref_slice))]

        # Also check similarity to the FULL stored sequence for context
        sim_full = compute_similarity_matrix(trajectory, dg_patterns[0])

        results.append({
            'entry_pos': entry_pos,
            'diag_sims': diag_sims,
            'sim_full': sim_full,
            'mean_diag': np.mean(diag_sims),
        })

        print(f"  Entry at position {entry_pos} (remaining {remaining_steps} steps):")
        print(f"    Forward continuation sims: {['%.3f' % s for s in diag_sims]}")
        print(f"    Mean continuation sim:     {np.mean(diag_sims):.4f}")

    return results


# =============================================================================
# TEST 4: Adaptation Rate Sweep
# =============================================================================

def test_adaptation_sweep(hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Vary adaptation rate and decay. Show how they affect traversal fidelity.

    Low adaptation:  network sticks at first attractor (no traversal)
    High adaptation: network blows through attractors too fast (blur)
    Sweet spot:      clean sequential transitions
    """
    print("\n" + "=" * 70)
    print("TEST 4: Adaptation Rate Sweep")
    print("=" * 70)

    seq_length = 6
    sequences = generate_sequences(1, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=400)

    hippo = HippocampalSystemSequential(**hippo_kwargs, device=device, dtype=dtype)
    dg_patterns = encode_all_sequences(hippo, sequences, n_repetitions=n_repetitions)

    adapt_rates = [0.01, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
    adapt_decays = [0.7, 0.85, 0.95]

    results = {}

    for decay in adapt_decays:
        for rate in adapt_rates:
            trajectory, adapt_hist = hippo.recall_sequence(
                sequences[0][0], n_steps=seq_length,
                adapt_rate=rate, adapt_decay=decay, settle_steps=3)

            sim = compute_similarity_matrix(trajectory, dg_patterns[0])
            diag_sims = [sim[t, t] for t in range(seq_length)]
            mean_diag = np.mean(diag_sims)

            # "Sequentiality": does the peak similarity shift forward over time?
            # For each retrieved step, find which stored pattern it's most similar to
            peak_indices = [int(np.argmax(sim[t, :])) for t in range(seq_length)]
            # Perfect sequentiality: peak_indices = [0, 1, 2, 3, 4, 5]
            sequentiality = np.mean([1.0 if peak_indices[t] == t else 0.0
                                     for t in range(seq_length)])

            results[(rate, decay)] = {
                'sim_matrix': sim,
                'diag_sims': diag_sims,
                'mean_diag': mean_diag,
                'peak_indices': peak_indices,
                'sequentiality': sequentiality,
            }

            print(f"  rate={rate:.2f}, decay={decay:.2f}: "
                  f"mean_diag={mean_diag:.4f}, "
                  f"sequentiality={sequentiality:.2f}, "
                  f"peaks={peak_indices}")

    return results, adapt_rates, adapt_decays


# =============================================================================
# TEST 5: Alpha Ratio Sweep (Symmetric vs Asymmetric Balance)
# =============================================================================

def test_alpha_sweep(base_hippo_kwargs, device, dtype, n_repetitions=3):
    """
    Vary the symmetric/asymmetric balance in CA3 learning.
    Show the tradeoff between pattern stability and sequence flow.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Alpha Ratio Sweep (sym/asym balance)")
    print("=" * 70)

    seq_length = 6
    sequences = generate_sequences(1, seq_length, base_hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=500)

    # Sweep: total weight stays at 1.0, ratio varies
    asym_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = {}

    for asym_frac in asym_fractions:
        kwargs = dict(base_hippo_kwargs)
        kwargs['alpha_sym'] = 1.0 - asym_frac
        kwargs['alpha_asym'] = asym_frac

        hippo = HippocampalSystemSequential(**kwargs, device=device, dtype=dtype)
        dg_patterns = encode_all_sequences(hippo, sequences,
                                           n_repetitions=n_repetitions)

        # Test 1: point attractor quality (cue with each pattern, does it complete?)
        point_sims = []
        for i, ec_pat in enumerate(sequences[0]):
            traj, _ = hippo.recall_sequence(
                ec_pat, n_steps=1, adapt_rate=0.0, adapt_decay=0.0, settle_steps=5)
            point_sims.append(cosine_sim(traj[0], dg_patterns[0][i]))
        mean_point = np.mean(point_sims)

        # Test 2: sequence quality (cue with first, does it traverse?)
        trajectory, _ = hippo.recall_sequence(
            sequences[0][0], n_steps=seq_length,
            adapt_rate=0.15, adapt_decay=0.85, settle_steps=3)

        sim = compute_similarity_matrix(trajectory, dg_patterns[0])
        diag_sims = [sim[t, t] for t in range(seq_length)]
        peak_indices = [int(np.argmax(sim[t, :])) for t in range(seq_length)]
        sequentiality = np.mean([1.0 if peak_indices[t] == t else 0.0
                                 for t in range(seq_length)])

        results[asym_frac] = {
            'mean_point_attractor': mean_point,
            'mean_sequence_diag': np.mean(diag_sims),
            'sequentiality': sequentiality,
            'peak_indices': peak_indices,
            'sim_matrix': sim,
        }

        print(f"  asym={asym_frac:.1f}: "
              f"point_attract={mean_point:.4f}, "
              f"seq_diag={np.mean(diag_sims):.4f}, "
              f"sequentiality={sequentiality:.2f}, "
              f"peaks={peak_indices}")

    return results, asym_fractions


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_results(test1, test2, test3, test4, test5, save_path):
    """Generate a comprehensive figure with all test results."""

    fig = plt.figure(figsize=(24, 28))
    gs = GridSpec(5, 4, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle("Hippocampal Sequence Attractors: Diagnostic Tests",
                 fontsize=16, fontweight='bold', y=0.98)

    # -------------------------------------------------------------------------
    # Row 1: Test 1 - Single sequence retrieval
    # -------------------------------------------------------------------------
    sim_matrix_1, diag_sims_1, adapt_hist_1 = test1

    ax1a = fig.add_subplot(gs[0, 0:2])
    im = ax1a.imshow(sim_matrix_1, aspect='auto', cmap='viridis',
                     vmin=-0.2, vmax=1.0)
    ax1a.set_xlabel("Stored pattern index")
    ax1a.set_ylabel("Retrieved step")
    ax1a.set_title("Test 1: Single Sequence Similarity Matrix")
    plt.colorbar(im, ax=ax1a, label="Cosine similarity")

    ax1b = fig.add_subplot(gs[0, 2])
    ax1b.plot(diag_sims_1, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax1b.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1b.set_xlabel("Sequence step")
    ax1b.set_ylabel("Cosine similarity")
    ax1b.set_title("Test 1: Diagonal Similarities")
    ax1b.set_ylim(-0.3, 1.1)
    ax1b.grid(True, alpha=0.3)

    ax1c = fig.add_subplot(gs[0, 3])
    ax1c.plot(adapt_hist_1, 's-', color='coral', linewidth=2, markersize=6)
    ax1c.set_xlabel("Sequence step")
    ax1c.set_ylabel("Adaptation norm")
    ax1c.set_title("Test 1: Adaptation Accumulation")
    ax1c.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Row 2: Test 2 - Multi-sequence separation
    # -------------------------------------------------------------------------
    sim_matrices_2, results_2, all_ref_2, dg_patterns_2 = test2

    for i in range(min(4, len(sim_matrices_2))):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(sim_matrices_2[i], aspect='auto', cmap='viridis',
                       vmin=-0.2, vmax=1.0)
        ax.set_xlabel("All stored patterns")
        ax.set_ylabel("Retrieved step")
        ax.set_title(f"Test 2: Seq {i} vs All")

        # Draw vertical lines separating sequences
        n_per_seq = len(dg_patterns_2[0])
        for boundary in range(1, len(dg_patterns_2)):
            ax.axvline(x=boundary * n_per_seq - 0.5, color='white',
                       linestyle='--', linewidth=1, alpha=0.7)

    # -------------------------------------------------------------------------
    # Row 3: Test 3 - Mid-sequence entry
    # -------------------------------------------------------------------------
    results_3 = test3

    for i, res in enumerate(results_3):
        ax = fig.add_subplot(gs[2, i])
        sim_full = res['sim_full']
        im = ax.imshow(sim_full, aspect='auto', cmap='viridis',
                       vmin=-0.2, vmax=1.0)
        ax.set_xlabel("Stored pattern index")
        ax.set_ylabel("Retrieved step")
        entry = res['entry_pos']
        ax.set_title(f"Test 3: Entry at pos {entry}")
        # Mark expected diagonal
        for t in range(sim_full.shape[0]):
            expected_r = entry + t
            if expected_r < sim_full.shape[1]:
                ax.plot(expected_r, t, 'wx', markersize=8, markeredgewidth=2)

    # Summary bar chart in the 4th column
    ax3d = fig.add_subplot(gs[2, 3])
    entry_positions = [r['entry_pos'] for r in results_3]
    mean_diags = [r['mean_diag'] for r in results_3]
    ax3d.bar(range(len(entry_positions)), mean_diags, color='steelblue', alpha=0.8)
    ax3d.set_xticks(range(len(entry_positions)))
    ax3d.set_xticklabels([f"pos {p}" for p in entry_positions])
    ax3d.set_ylabel("Mean continuation similarity")
    ax3d.set_title("Test 3: Mid-Entry Quality")
    ax3d.set_ylim(0, 1.1)
    ax3d.grid(True, alpha=0.3, axis='y')

    # -------------------------------------------------------------------------
    # Row 4: Test 4 - Adaptation rate sweep
    # -------------------------------------------------------------------------
    results_4, adapt_rates_4, adapt_decays_4 = test4

    for di, decay in enumerate(adapt_decays_4):
        ax = fig.add_subplot(gs[3, di])
        seq_scores = []
        diag_scores = []
        for rate in adapt_rates_4:
            r = results_4[(rate, decay)]
            seq_scores.append(r['sequentiality'])
            diag_scores.append(r['mean_diag'])

        ax.plot(adapt_rates_4, seq_scores, 'o-', color='steelblue',
                linewidth=2, label='Sequentiality', markersize=5)
        ax.plot(adapt_rates_4, diag_scores, 's-', color='coral',
                linewidth=2, label='Mean diag sim', markersize=5)
        ax.set_xlabel("Adaptation rate")
        ax.set_ylabel("Score")
        ax.set_title(f"Test 4: decay={decay:.2f}")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    # Best-case sim matrix in 4th column
    ax4d = fig.add_subplot(gs[3, 3])
    # Find the (rate, decay) with highest sequentiality
    best_key = max(results_4, key=lambda k: results_4[k]['sequentiality'])
    best_sim = results_4[best_key]['sim_matrix']
    im = ax4d.imshow(best_sim, aspect='auto', cmap='viridis', vmin=-0.2, vmax=1.0)
    ax4d.set_xlabel("Stored pattern index")
    ax4d.set_ylabel("Retrieved step")
    ax4d.set_title(f"Test 4: Best (r={best_key[0]:.2f}, d={best_key[1]:.2f})")
    plt.colorbar(im, ax=ax4d, label="Cosine sim")

    # -------------------------------------------------------------------------
    # Row 5: Test 5 - Alpha ratio sweep
    # -------------------------------------------------------------------------
    results_5, asym_fractions_5 = test5

    ax5a = fig.add_subplot(gs[4, 0:2])
    point_scores = [results_5[f]['mean_point_attractor'] for f in asym_fractions_5]
    seq_diag_scores = [results_5[f]['mean_sequence_diag'] for f in asym_fractions_5]
    seq_scores = [results_5[f]['sequentiality'] for f in asym_fractions_5]

    ax5a.plot(asym_fractions_5, point_scores, 'o-', color='forestgreen',
              linewidth=2, label='Point attractor quality', markersize=6)
    ax5a.plot(asym_fractions_5, seq_diag_scores, 's-', color='coral',
              linewidth=2, label='Sequence diag similarity', markersize=6)
    ax5a.plot(asym_fractions_5, seq_scores, '^-', color='steelblue',
              linewidth=2, label='Sequentiality', markersize=6)
    ax5a.set_xlabel("Asymmetric fraction (0 = pure symmetric, 1 = pure asymmetric)")
    ax5a.set_ylabel("Score")
    ax5a.set_title("Test 5: Sym/Asym Tradeoff")
    ax5a.legend(fontsize=9)
    ax5a.set_ylim(-0.1, 1.1)
    ax5a.grid(True, alpha=0.3)

    # Show sim matrices for a few key ratios
    key_fracs = [0.0, 0.3, 0.7, 1.0]
    for ki, frac in enumerate(key_fracs):
        if frac in results_5:
            ax = fig.add_subplot(gs[4, 2]) if ki < 2 else fig.add_subplot(gs[4, 3])
            # Stack two sim matrices per subplot
            if ki % 2 == 0:
                im = ax.imshow(results_5[frac]['sim_matrix'], aspect='auto',
                               cmap='viridis', vmin=-0.2, vmax=1.0)
                ax.set_xlabel("Stored pattern index")
                ax.set_ylabel("Retrieved step")
                ax.set_title(f"Test 5: asym={frac:.1f}")

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

    # Base hippocampal kwargs (shared across tests unless overridden)
    hippo_kwargs = {
        "d_ec": d_ec,
        "D_dg": d_ec,
        "N_ca3": d_ec,
        "N_ca1": d_ec,
        "k_ca3": 50,
        "N_sub": d_ec,
        "ca3_retrieval_iterations": 5,
        "alpha_sym": 0.7,
        "alpha_asym": 0.3,
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

    n_repetitions = 3  # How many times to present each sequence during encoding

    print("=" * 70)
    print("Hippocampal Sequence Attractors Test Suite")
    print("=" * 70)
    print(f"  d_ec={d_ec}, k_ca3=50")
    print(f"  alpha_sym={hippo_kwargs['alpha_sym']}, "
          f"alpha_asym={hippo_kwargs['alpha_asym']}")
    print(f"  Encoding repetitions: {n_repetitions}")

    # Run all tests
    test1_results = test_single_sequence(hippo_kwargs, device, dtype,
                                         n_repetitions=n_repetitions)
    test2_results = test_multi_sequence(hippo_kwargs, device, dtype,
                                        n_repetitions=n_repetitions)
    test3_results = test_mid_sequence_entry(hippo_kwargs, device, dtype,
                                            n_repetitions=n_repetitions)
    test4_results = test_adaptation_sweep(hippo_kwargs, device, dtype,
                                          n_repetitions=n_repetitions)
    test5_results = test_alpha_sweep(hippo_kwargs, device, dtype,
                                     n_repetitions=n_repetitions)

    # Plot
    plot_all_results(test1_results, test2_results, test3_results,
                     test4_results, test5_results,
                     save_path="sequence_attractors_results.png")

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
