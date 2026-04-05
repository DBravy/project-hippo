"""
Hippocampal Reconstruction v1: Can the Hippocampus Reconstruct Cortical States?
================================================================================

No replay, no optimizer. Just the fundamental question:
can we build projections + hippocampus that reliably encode and
reconstruct residual stream vectors from a trained neural network?

Protocol:
  1. Train a cortex (residual MLP stack) on a structured task
  2. Freeze cortex, collect hidden states at anchor layers
  3. Feed hidden states through input_proj -> hippocampus -> output_proj
  4. Hippocampus learns via Hebbian plasticity (as in v20)
  5. Output projection learned via least-squares regression
  6. Measure reconstruction quality as patterns accumulate

Key question: does reconstruction quality stabilize at a useful level,
or does it collapse as the hippocampus fills up?
"""

import numpy as np
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# 1. HIPPOCAMPAL CORE (from v20)
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

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class ECSuperficial:
    def __init__(self, d_ec, sigma_inh=25, gamma_inh=5.0, n_inh_steps=5,
                 pyr_to_stel_strength=0.3, connectivity_prob=0.33):
        self.d_ec = d_ec
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.pyr_to_stel_strength = pyr_to_stel_strength
        self.W_stellate = make_feedforward_weights(d_ec, d_ec, connectivity_prob)
        self.W_inh = build_ring_inhibition(d_ec, sigma_inh)
        self.W_pyramidal = make_feedforward_weights(d_ec, d_ec, connectivity_prob)
        self.W_pyr_to_stel = make_feedforward_weights(d_ec, d_ec, connectivity_prob)

    def forward(self, cortical_input):
        pyramidal = np.maximum(self.W_pyramidal @ cortical_input, 0)
        h_raw = self.W_stellate @ cortical_input
        h_raw += self.pyr_to_stel_strength * (self.W_pyr_to_stel @ pyramidal)
        h_raw = np.maximum(h_raw, 0)
        h = h_raw.copy()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = np.maximum(h_raw - self.gamma_inh * inh, 0)
        return h, pyramidal


class DentateGyrusLateral:
    def __init__(self, d_input, D_output, sigma_inh=50, gamma_inh=1.0,
                 n_inh_steps=5, noise_scale=0.0, inh_connection_prob=None):
        self.D_output = D_output
        self.W_ff = make_feedforward_weights(D_output, d_input)
        self.W_inh = build_ring_inhibition(
            D_output, sigma_inh, connection_prob=inh_connection_prob)
        self.gamma_inh = gamma_inh
        self.n_inh_steps = n_inh_steps
        self.noise_scale = noise_scale

    def forward(self, x):
        h_raw = x @ self.W_ff.T
        h_raw = np.maximum(h_raw, 0)
        if self.noise_scale > 0 and np.any(h_raw > 0):
            mean_active = np.mean(h_raw[h_raw > 0])
            h_raw = np.maximum(
                h_raw + np.random.randn(self.D_output) * self.noise_scale * mean_active, 0)
        h = h_raw.copy()
        for _ in range(self.n_inh_steps):
            inh = self.W_inh @ h
            h = np.maximum(h_raw - self.gamma_inh * inh, 0)
        return h


class CA3:
    def __init__(self, N, k_active):
        self.N = N
        self.k_active = k_active
        self.W = np.zeros((N, N))
        self.n_stored = 0
        self.mean_activity = np.zeros(N)

    def store_online(self, pattern):
        p = pattern / (np.linalg.norm(pattern) + 1e-10)
        self.n_stored += 1
        self.mean_activity += (p - self.mean_activity) / self.n_stored
        p_c = p - self.mean_activity
        self.W += np.outer(p_c, p_c)
        np.fill_diagonal(self.W, 0)

    def retrieve(self, cue, n_iterations=5):
        x = cue / (np.linalg.norm(cue) + 1e-10)
        for _ in range(n_iterations):
            h = np.maximum(self.W @ x, 0)
            x_new = apply_kwta(h, self.k_active)
            x_new = x_new / (np.linalg.norm(x_new) + 1e-10)
            if np.allclose(x, x_new, atol=1e-6):
                break
            x = x_new
        return x


class CA1:
    def __init__(self, N_ca1, N_ca3, d_ec, lr=0.3, weight_decay=0.998, k_active=50):
        self.N_ca1 = N_ca1
        self.lr = lr
        self.weight_decay = weight_decay
        self.k_active = k_active
        self.W_sc = np.zeros((N_ca1, N_ca3))
        self.n_episodes = 0

    def encode(self, x_ca3, x_ec_stel):
        self.W_sc += self.lr * np.outer(x_ec_stel, x_ca3)
        self.W_sc *= self.weight_decay
        self.n_episodes += 1

    def retrieve(self, x_ca3):
        h_sc = np.maximum(self.W_sc @ x_ca3, 0)
        return apply_kwta(h_sc, self.k_active)


class Subiculum:
    def __init__(self, N_sub, N_ca1, d_ec, lr=1.0, k_active=500):
        self.N_sub = N_sub
        self.lr = lr
        self.k_active = k_active
        self.W_ca1 = np.zeros((N_sub, N_ca1))
        self.n_episodes = 0

    def encode(self, ca1_output, ec_pyramidal):
        self.W_ca1 += self.lr * np.outer(ec_pyramidal, ca1_output)
        self.n_episodes += 1

    def replay(self, ca1_output):
        h = np.maximum(self.W_ca1 @ ca1_output, 0)
        return apply_kwta(h, self.k_active)


class Hippocampus:
    """
    Full hippocampal system with configurable dimensions.
    No FiLM. Returns raw ca1 and sub outputs for external projection.
    """
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, N_sub, k_ca3,
                 ca3_iters=5, ec_sup_params=None, dg_params=None,
                 ca1_params=None, sub_params=None):
        self.d_ec = d_ec
        self.ca3_iters = ca3_iters
        self.ec_sup = ECSuperficial(d_ec, **(ec_sup_params or {}))
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}))
        self.ca3 = CA3(N_ca3, k_ca3)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}))
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}))

    def encode(self, ec_input):
        """Encode an EC-space vector. Hebbian learning in CA3, CA1, Sub."""
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        self.ca3.store_online(dg_out)
        # CA1 + Sub training
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_iters)
        self.ca1.encode(ca3_out, stellate)
        ca1_out = self.ca1.retrieve(ca3_out)
        self.sub.encode(ca1_out, pyramidal)

    def recall(self, ec_input):
        """
        Recall: EC -> DG -> CA3 retrieve -> CA1 -> Sub.
        Returns (ca1_output, sub_output, stellate, pyramidal).
        """
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_iters)
        ca1_out = self.ca1.retrieve(ca3_out)
        sub_out = self.sub.replay(ca1_out)
        return ca1_out, sub_out, stellate, pyramidal


# =============================================================================
# 2. INPUT PROJECTION: Residual Stream -> EC Space
# =============================================================================

class InputProjection:
    """
    Maps d_model residual stream vectors to d_ec hippocampal input space.

    The hippocampus expects positive-valued inputs (ReLU everywhere).
    Strategy: affine transform + ReLU + normalize.

    The projection is learned via gradient-free optimization:
    we track reconstruction quality and update the projection
    to improve it.

    For v1: use a fixed orthogonal rotation + shift to positive + normalize.
    This is a reasonable baseline that preserves distances.
    """
    def __init__(self, d_model, d_ec):
        self.d_model = d_model
        self.d_ec = d_ec

        # Orthogonal projection (distance-preserving)
        if d_model == d_ec:
            # Random orthogonal matrix
            raw = np.random.randn(d_ec, d_model)
            U, _, Vt = np.linalg.svd(raw, full_matrices=False)
            self.W = U @ Vt  # orthogonal
        else:
            raw = np.random.randn(d_ec, d_model)
            self.W = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-10)

        # Running statistics for input normalization
        self._mean = np.zeros(d_model)
        self._var = np.ones(d_model)
        self._count = 0

    def update_stats(self, vec):
        """Online mean/variance update."""
        self._count += 1
        delta = vec - self._mean
        self._mean += delta / self._count
        delta2 = vec - self._mean
        self._var += (delta * delta2 - self._var) / self._count

    def project(self, vec):
        """Map residual stream vector to EC space."""
        # Standardize
        std = np.sqrt(self._var + 1e-8)
        normed = (vec - self._mean) / std

        # Rotate
        projected = self.W @ normed

        # Shift to positive (add offset so ~50% of values are positive after ReLU)
        # This preserves information while making it compatible with ReLU-based EC
        projected = projected + np.abs(np.min(projected)) + 0.1

        # Normalize to unit norm (hippocampus works with normalized inputs)
        norm = np.linalg.norm(projected) + 1e-10
        return projected / norm


# =============================================================================
# 3. OUTPUT PROJECTION: Hippocampal Outputs -> Residual Stream
# =============================================================================

class OutputProjection:
    """
    Maps concatenated (ca1_output, sub_output) back to d_model space.

    Learned via least-squares regression on accumulated (hippo_output, target) pairs.
    Periodically refit as more data accumulates.
    """
    def __init__(self, d_ca1, d_sub, d_model, regularization=0.01):
        self.d_in = d_ca1 + d_sub
        self.d_model = d_model
        self.reg = regularization

        # Initialize with random weights
        self.W = np.random.randn(d_model, self.d_in) * 0.01
        self.b = np.zeros(d_model)

        # Accumulate data for fitting
        self._hippo_outputs = []
        self._targets = []

    def add_pair(self, hippo_output, target):
        """Store a (hippo_output, target) pair for later fitting."""
        self._hippo_outputs.append(hippo_output.copy())
        self._targets.append(target.copy())

    def fit(self):
        """Refit projection via ridge regression on accumulated pairs."""
        if len(self._hippo_outputs) < 10:
            return

        X = np.array(self._hippo_outputs)  # (N, d_ca1 + d_sub)
        Y = np.array(self._targets)         # (N, d_model)

        # Ridge regression: W = Y^T X (X^T X + lambda I)^{-1}
        XtX = X.T @ X + self.reg * np.eye(X.shape[1])
        XtY = X.T @ Y
        self.W = np.linalg.solve(XtX, XtY).T  # (d_model, d_in)
        self.b = Y.mean(axis=0) - self.W @ X.mean(axis=0)

    def project(self, hippo_output):
        """Map hippo output to residual stream space."""
        return self.W @ hippo_output + self.b


# =============================================================================
# 4. CORTEX (PyTorch, same as before)
# =============================================================================

class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_hidden)
        self.down = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        h = self.ln(x)
        h = F.silu(self.up(h))
        h = self.down(h)
        return x + h


class ResidualStack(nn.Module):
    def __init__(self, d_model, d_hidden, n_layers):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            ResidualMLPBlock(d_model, d_hidden) for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, d_model)

    def forward(self, x):
        states = [x]
        h = x
        for layer in self.layers:
            h = layer(h)
            states.append(h)
        pred = self.output_head(h)
        return pred, states


# =============================================================================
# 5. TASK (fixed seed issue from v2)
# =============================================================================

def generate_task(d_model, n_samples, n_components=20, transform_seed=42,
                  data_seed=None):
    """
    Structured task with SHARED transformation across train/val.
    transform_seed controls the (U, V) matrices.
    data_seed controls which input points are sampled.
    """
    # Fixed transformation
    rng_t = np.random.RandomState(transform_seed)
    U = rng_t.randn(d_model, n_components) * 0.3
    V = rng_t.randn(n_components, d_model) * 0.3

    # Random inputs
    rng_d = np.random.RandomState(data_seed)
    inputs = rng_d.randn(n_samples, d_model).astype(np.float32)
    inputs = inputs / (np.linalg.norm(inputs, axis=1, keepdims=True) + 1e-10)

    h = np.tanh(inputs @ U)
    targets = (h @ V).astype(np.float32)
    scale = np.std(targets) + 1e-10
    targets = targets / scale

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )


# =============================================================================
# 6. MAIN EXPERIMENT
# =============================================================================

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Parameters ---
    d_model = 128
    d_hidden = 256
    n_layers = 8
    d_hippo = 256     # Hippocampus internal dimension (can differ from d_model)

    n_train = 2000
    n_val = 500
    cortex_epochs = 100
    cortex_lr = 3e-4
    cortex_wd = 0.01
    batch_size = 32

    # Anchor layers to test
    anchor_layers = [0, 4, 8]

    # Hippocampal parameters (scaled for d_hippo)
    k_ca3 = max(10, d_hippo // 20)
    k_ca1 = max(10, d_hippo // 20)
    k_sub = d_hippo // 2
    hippo_kwargs = {
        "d_ec": d_hippo, "D_dg": d_hippo, "N_ca3": d_hippo,
        "N_ca1": d_hippo, "N_sub": d_hippo, "k_ca3": k_ca3,
        "ca3_iters": 5,
        "ec_sup_params": {"sigma_inh": 15, "gamma_inh": 5.0, "n_inh_steps": 5,
                          "pyr_to_stel_strength": 0.3},
        "dg_params": {"sigma_inh": 15, "gamma_inh": 5.0, "n_inh_steps": 5,
                      "noise_scale": 0.0},
        "ca1_params": {"lr": 50.0, "weight_decay": 1.000, "k_active": k_ca1},
        "sub_params": {"lr": 1.0, "k_active": k_sub},
    }

    # How often to refit output projection and measure quality
    refit_interval = 100

    print(f"Cortex: d_model={d_model}, d_hidden={d_hidden}, n_layers={n_layers}")
    print(f"Hippo: d_hippo={d_hippo}, k_ca3={k_ca3}, k_ca1={k_ca1}, k_sub={k_sub}")
    print(f"Task: n_train={n_train}, n_val={n_val}")
    print(f"Anchors: {anchor_layers}")

    # =================================================================
    # PHASE 1: Train cortex
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Training cortex")
    print("="*70)

    train_inputs, train_targets = generate_task(
        d_model, n_train, transform_seed=42, data_seed=100)
    val_inputs, val_targets = generate_task(
        d_model, n_val, transform_seed=42, data_seed=200)

    cortex = ResidualStack(d_model, d_hidden, n_layers).to(device)
    optimizer = torch.optim.AdamW(cortex.parameters(), lr=cortex_lr,
                                   weight_decay=cortex_wd)

    for epoch in range(cortex_epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_b = 0
        for i in range(0, n_train - batch_size + 1, batch_size):
            idx = perm[i:i+batch_size]
            x = train_inputs[idx].to(device)
            y = train_targets[idx].to(device)
            optimizer.zero_grad()
            pred, _ = cortex(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        if epoch % 20 == 0 or epoch == cortex_epochs - 1:
            cortex.eval()
            with torch.no_grad():
                val_pred, _ = cortex(val_inputs.to(device))
                val_loss = F.mse_loss(val_pred, val_targets.to(device)).item()
            cortex.train()
            print(f"  epoch {epoch:4d}: train={epoch_loss/n_b:.6f}, val={val_loss:.6f}")

    cortex.eval()
    print("  Cortex trained and frozen.")

    # =================================================================
    # PHASE 2: Collect hidden states
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Collecting hidden states from cortex")
    print("="*70)

    hidden_states = {l: [] for l in anchor_layers}

    with torch.no_grad():
        for i in range(0, n_train, batch_size):
            x = train_inputs[i:i+batch_size].to(device)
            _, states = cortex(x)
            for l in anchor_layers:
                for j in range(states[l].shape[0]):
                    hidden_states[l].append(
                        states[l][j].cpu().numpy().astype(np.float64))

    for l in anchor_layers:
        vecs = hidden_states[l]
        norms = [np.linalg.norm(v) for v in vecs]
        print(f"  Layer {l}: {len(vecs)} vectors, "
              f"norm={np.mean(norms):.3f} +/- {np.std(norms):.3f}, "
              f"dim={vecs[0].shape[0]}")

    # =================================================================
    # PHASE 3: Encode into hippocampus and measure reconstruction
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Hippocampal encoding and reconstruction")
    print("="*70)

    results_by_layer = {}

    for layer_idx in anchor_layers:
        print(f"\n  --- Layer {layer_idx} ---")
        vecs = hidden_states[layer_idx]
        n_vecs = len(vecs)

        # Build components
        hippo = Hippocampus(**hippo_kwargs)
        in_proj = InputProjection(d_model, d_hippo)
        out_proj = OutputProjection(d_hippo, d_hippo, d_model)

        # First pass: collect input stats
        print(f"    Collecting input statistics...")
        for vec in vecs[:200]:
            in_proj.update_stats(vec)

        # Encoding + recall loop
        recon_quality_curve = []
        ec_space_sims = []       # similarity in EC space (input vs recall)
        ca1_active_fracs = []
        sub_active_fracs = []

        print(f"    Encoding {n_vecs} patterns...")
        for i, vec in enumerate(vecs):
            # Update input stats
            in_proj.update_stats(vec)

            # Project to EC space
            ec_input = in_proj.project(vec)

            # Encode in hippocampus (Hebbian learning)
            hippo.encode(ec_input)

            # Recall
            ca1_out, sub_out, stel, pyr = hippo.recall(ec_input)

            # Track internal quality
            ca1_active_fracs.append(float(np.mean(ca1_out > 0)))
            sub_active_fracs.append(float(np.mean(sub_out > 0)))

            # EC-space reconstruction quality (does the hippo loop preserve info?)
            ec_recon_from_stel = stel  # CA1 tries to reconstruct stellate
            ec_sim = cosine_sim(stel, ca1_out)
            ec_space_sims.append(ec_sim)

            # Store pair for output projection fitting
            hippo_output = np.concatenate([ca1_out, sub_out])
            out_proj.add_pair(hippo_output, vec)

            # Periodically fit output projection and measure end-to-end quality
            if (i + 1) % refit_interval == 0 or i == n_vecs - 1:
                out_proj.fit()

                # Measure reconstruction quality on recent patterns
                test_sims = []
                test_mses = []
                n_test = min(100, i + 1)
                test_indices = np.random.choice(i + 1, n_test, replace=False)

                for ti in test_indices:
                    test_vec = vecs[ti]
                    test_ec = in_proj.project(test_vec)
                    ca1_t, sub_t, _, _ = hippo.recall(test_ec)
                    hippo_out_t = np.concatenate([ca1_t, sub_t])
                    recon = out_proj.project(hippo_out_t)
                    test_sims.append(cosine_sim(recon, test_vec))
                    test_mses.append(float(np.mean((recon - test_vec)**2)))

                mean_sim = np.mean(test_sims)
                mean_mse = np.mean(test_mses)
                mean_ec_sim = np.mean(ec_space_sims[-refit_interval:])
                mean_ca1_frac = np.mean(ca1_active_fracs[-refit_interval:])
                mean_sub_frac = np.mean(sub_active_fracs[-refit_interval:])

                recon_quality_curve.append({
                    "n_stored": i + 1,
                    "cosine_sim": mean_sim,
                    "mse": mean_mse,
                    "ec_space_sim": mean_ec_sim,
                    "ca1_active_frac": mean_ca1_frac,
                    "sub_active_frac": mean_sub_frac,
                })

                print(f"    n={i+1:5d}: recon_sim={mean_sim:.4f}, "
                      f"mse={mean_mse:.4f}, ec_sim={mean_ec_sim:.4f}, "
                      f"ca1_frac={mean_ca1_frac:.3f}, sub_frac={mean_sub_frac:.3f}")

        # Final comprehensive evaluation
        print(f"\n    Final evaluation (all patterns):")
        out_proj.fit()  # Final refit

        all_sims = []
        all_mses = []
        for vec in vecs:
            ec = in_proj.project(vec)
            ca1_t, sub_t, _, _ = hippo.recall(ec)
            recon = out_proj.project(np.concatenate([ca1_t, sub_t]))
            all_sims.append(cosine_sim(recon, vec))
            all_mses.append(float(np.mean((recon - vec)**2)))

        print(f"    Cosine sim: {np.mean(all_sims):.4f} +/- {np.std(all_sims):.4f}")
        print(f"    MSE:        {np.mean(all_mses):.4f} +/- {np.std(all_mses):.4f}")
        print(f"    Worst 10%:  sim={np.percentile(all_sims, 10):.4f}")
        print(f"    Best 10%:   sim={np.percentile(all_sims, 90):.4f}")

        # Test on NOVEL vectors (val set, never encoded)
        print(f"\n    Generalization test (novel vectors from val set):")
        val_states = []
        with torch.no_grad():
            for i_v in range(0, n_val, batch_size):
                x = val_inputs[i_v:i_v+batch_size].to(device)
                _, states = cortex(x)
                for j in range(states[layer_idx].shape[0]):
                    val_states.append(
                        states[layer_idx][j].cpu().numpy().astype(np.float64))

        novel_sims = []
        for vec in val_states[:200]:
            ec = in_proj.project(vec)
            ca1_t, sub_t, _, _ = hippo.recall(ec)
            recon = out_proj.project(np.concatenate([ca1_t, sub_t]))
            novel_sims.append(cosine_sim(recon, vec))

        print(f"    Novel cosine sim: {np.mean(novel_sims):.4f} +/- {np.std(novel_sims):.4f}")

        results_by_layer[layer_idx] = {
            "curve": recon_quality_curve,
            "final_sim": float(np.mean(all_sims)),
            "final_mse": float(np.mean(all_mses)),
            "novel_sim": float(np.mean(novel_sims)),
            "all_sims": [float(s) for s in all_sims],
        }

    # =================================================================
    # SAVE AND PLOT
    # =================================================================
    print(f"\n{'='*70}")
    print("Saving results...")
    print("="*70)

    results = {
        "params": {
            "d_model": d_model, "d_hippo": d_hippo,
            "n_layers": n_layers, "anchor_layers": anchor_layers,
            "k_ca3": k_ca3, "k_ca1": k_ca1, "k_sub": k_sub,
            "n_train": n_train, "n_val": n_val,
        },
        "layers": {str(l): results_by_layer[l] for l in anchor_layers},
    }

    with open("hippo_reconstruction_v1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved to hippo_reconstruction_v1_results.json")

    # --- Plot ---
    fig, axes = plt.subplots(2, len(anchor_layers), figsize=(6*len(anchor_layers), 10))
    fig.suptitle("Hippocampal Reconstruction v1: Can the Hippocampus Reconstruct Cortical States?",
                 fontsize=13, fontweight='bold')

    for col, layer_idx in enumerate(anchor_layers):
        res = results_by_layer[layer_idx]
        curve = res["curve"]

        # Top row: reconstruction quality over encoding
        ax = axes[0, col]
        ns = [c["n_stored"] for c in curve]
        sims = [c["cosine_sim"] for c in curve]
        ec_sims = [c["ec_space_sim"] for c in curve]
        ax.plot(ns, sims, 'o-', color="steelblue", label="End-to-end recon", linewidth=2)
        ax.plot(ns, ec_sims, 's-', color="coral", label="EC-space (CA1 vs stellate)", linewidth=2)
        ax.axhline(res["novel_sim"], color="forestgreen", linestyle="--",
                   label=f"Novel vectors ({res['novel_sim']:.3f})", linewidth=1.5)
        ax.set_xlabel("Patterns stored")
        ax.set_ylabel("Cosine similarity")
        ax.set_title(f"Layer {layer_idx}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.0)

        # Bottom row: histogram of final reconstruction similarities
        ax = axes[1, col]
        ax.hist(res["all_sims"], bins=40, color="steelblue", alpha=0.7,
                edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(res["all_sims"]), color="coral", linestyle="--",
                   label=f"Mean={np.mean(res['all_sims']):.3f}", linewidth=2)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Layer {layer_idx}: Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hippo_reconstruction_v1_results.png", dpi=150, bbox_inches='tight')
    print("  Saved plot to hippo_reconstruction_v1_results.png")

    print(f"\n{'='*70}")
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
