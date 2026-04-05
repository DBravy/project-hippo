"""
Hippocampal Reconstruction v2: Exact v20 Hippocampus + Simple Projection
=========================================================================

Fixes from v1:
  - Input projection is a single matrix multiply (like v20's W_ec). No
    shift-to-positive, no normalization hacks.
  - Hippocampus is EXACTLY v20: same classes, same parameters, same FiLM gating.
  - d_ec=1000 to match v20's proven operating point.
  - Output projection via ridge regression from FiLM-gated output to d_model.

The only difference from v20 is what goes in (cortical hidden states instead
of sparse features projected through W_ec) and what comes out (reconstruction
of hidden states instead of feature decode R²).
"""

import numpy as np
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
# 1. HIPPOCAMPAL SYSTEM — EXACT COPY FROM v20
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


def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


class ECSuperficial:
    """Exact copy from v20."""
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
    """Exact copy from v20."""
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
    """Exact copy from v20."""
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
    """Exact copy from v20."""
    def __init__(self, N_ca1, N_ca3, d_ec,
                 lr=0.3, weight_decay=0.998, k_active=50):
        self.N_ca1 = N_ca1
        self.lr = lr
        self.weight_decay = weight_decay
        self.k_active = k_active
        self.W_sc = np.zeros((N_ca1, N_ca3))
        self.n_episodes = 0

    def encode(self, x_ca3, x_ec_stel):
        """Pure Hebbian: associate stellate (post) with CA3 (pre)."""
        self.W_sc += self.lr * np.outer(x_ec_stel, x_ca3)
        self.W_sc *= self.weight_decay
        self.n_episodes += 1

    def retrieve(self, x_ca3, x_ec_stel=None):
        """ReLU + k-WTA."""
        h_sc = np.maximum(self.W_sc @ x_ca3, 0)
        h_out = apply_kwta(h_sc, self.k_active)
        mismatch = 0.0
        if x_ec_stel is not None:
            error = x_ec_stel - h_out
            mismatch = float(np.linalg.norm(error) / (np.linalg.norm(x_ec_stel) + 1e-10))
        return h_out, mismatch


class Subiculum:
    """Exact copy from v20."""
    def __init__(self, N_sub, N_ca1, d_ec,
                 lr=1.0, k_active=500):
        self.N_sub = N_sub
        self.lr = lr
        self.k_active = k_active
        self.W_ca1 = np.zeros((N_sub, N_ca1))
        self.n_episodes = 0

    def encode(self, ca1_output, ec_pyramidal):
        """Pure Hebbian: associate pyramidal (post) with CA1 (pre)."""
        self.W_ca1 += self.lr * np.outer(ec_pyramidal, ca1_output)
        self.n_episodes += 1

    def replay(self, ca1_output):
        """ReLU + k-WTA to match pyramidal density."""
        h = np.maximum(self.W_ca1 @ ca1_output, 0)
        return apply_kwta(h, self.k_active)


class ECDeepVb:
    """Exact copy from v20. FiLM gating."""
    @staticmethod
    def gate(sparse_signal, dense_signal):
        gamma = sigmoid(dense_signal)
        return gamma * sparse_signal

    @staticmethod
    def hippocampal_output(ca1_output, sub_output):
        return ECDeepVb.gate(ca1_output, sub_output)

    @staticmethod
    def ec_target(stellate, pyramidal):
        return ECDeepVb.gate(stellate, pyramidal)


class HippocampalSystem:
    """Exact copy from v20."""
    def __init__(self, d_ec, D_dg, N_ca3, N_ca1, k_ca3,
                 dg_params=None, ca1_params=None, sub_params=None,
                 ec_sup_params=None, N_sub=1000, ca3_retrieval_iterations=5):
        self.d_ec = d_ec
        self.ca3_retrieval_iterations = ca3_retrieval_iterations
        self.ec_sup = ECSuperficial(d_ec, **(ec_sup_params or {}))
        self.dg = DentateGyrusLateral(d_ec, D_dg, **(dg_params or {}))
        self.ca3 = CA3(N_ca3, k_ca3)
        self.ca1 = CA1(N_ca1, N_ca3, d_ec, **(ca1_params or {}))
        self.sub = Subiculum(N_sub, N_ca1, d_ec, **(sub_params or {}))

    def encode_single(self, ec_input):
        """Encode one EC vector. Exact v20 encode_batch logic for one pattern."""
        stellate, pyramidal = self.ec_sup.forward(ec_input)

        # Storage: stellate -> DG -> CA3
        dg_out = self.dg.forward(stellate)
        self.ca3.store_online(dg_out)

        # CA1 encoding: Schaffer from DG, TA from stellate
        self.ca1.encode(dg_out, stellate)

        # CA1 output for Sub training
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)

        # Sub: learns dense (pyramidal) from CA1 sparse
        self.sub.encode(ca1_out, pyramidal)

        return mm

    def get_stage_outputs(self, ec_input):
        """Exact copy from v20."""
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, _ = self.ca1.retrieve(ca3_out, stellate)
        sub_out = self.sub.replay(ca1_out)
        gated_hippo = ECDeepVb.hippocampal_output(ca1_out, sub_out)
        gated_ec = ECDeepVb.ec_target(stellate, pyramidal)
        return {
            "ec_input": ec_input,
            "stellate": stellate, "pyramidal": pyramidal,
            "dg": dg_out, "ca3": ca3_out,
            "ca1": ca1_out, "sub": sub_out,
            "gated_hippo": gated_hippo, "gated_ec": gated_ec,
        }


# =============================================================================
# 2. CORTEX (PyTorch)
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
# 3. TASK (fixed: shared transform across train/val)
# =============================================================================

def generate_task(d_model, n_samples, n_components=20,
                  transform_seed=42, data_seed=None):
    rng_t = np.random.RandomState(transform_seed)
    U = rng_t.randn(d_model, n_components) * 0.3
    V = rng_t.randn(n_components, d_model) * 0.3

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
# 4. MAIN
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
    n_cortex_layers = 8
    n_train = 2000
    n_val = 500
    cortex_epochs = 100
    batch_size = 32

    # Hippocampal parameters: EXACT v20 values
    d_ec = 1000
    D_dg = 1000
    N_ca3 = 1000
    N_ca1 = 1000
    N_sub = 1000
    k_ca3 = 50
    ca3_retrieval_iters = 5

    ec_sup_params = {
        "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
        "pyr_to_stel_strength": 0.3,
    }
    dg_params = {
        "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5, "noise_scale": 0.0,
    }
    ca1_params = {"lr": 50.0, "weight_decay": 1.000, "k_active": 50}
    sub_params = {"lr": 1.0, "k_active": 500}

    anchor_layers = [0, 4, 8]
    refit_interval = 100

    print("=" * 70)
    print("Hippocampal Reconstruction v2: Exact v20 + Simple Projection")
    print("=" * 70)
    print(f"\nCortex: d_model={d_model}, layers={n_cortex_layers}")
    print(f"Hippocampus: d_ec={d_ec}, k_ca3={k_ca3}, k_ca1={ca1_params['k_active']}, "
          f"k_sub={sub_params['k_active']}")
    print(f"Input projection: random matrix ({d_ec} x {d_model}), row-normalized")
    print(f"Output: FiLM gated (exact v20) -> ridge regression to d_model")
    print(f"Anchors: {anchor_layers}")

    # =================================================================
    # PHASE 1: Train cortex
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: Training cortex")
    print("=" * 70)

    train_inputs, train_targets = generate_task(
        d_model, n_train, transform_seed=42, data_seed=100)
    val_inputs, val_targets = generate_task(
        d_model, n_val, transform_seed=42, data_seed=200)

    cortex = ResidualStack(d_model, d_hidden, n_cortex_layers).to(device)
    optimizer = torch.optim.AdamW(cortex.parameters(), lr=3e-4, weight_decay=0.01)

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
    print("PHASE 2: Collecting hidden states")
    print("=" * 70)

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
              f"norm={np.mean(norms):.3f} +/- {np.std(norms):.3f}")

    # Collect val hidden states for generalization test
    val_hidden_states = {l: [] for l in anchor_layers}
    with torch.no_grad():
        for i in range(0, n_val, batch_size):
            x = val_inputs[i:i+batch_size].to(device)
            _, states = cortex(x)
            for l in anchor_layers:
                for j in range(states[l].shape[0]):
                    val_hidden_states[l].append(
                        states[l][j].cpu().numpy().astype(np.float64))

    # =================================================================
    # PHASE 3: Hippocampal encoding and reconstruction
    # =================================================================
    print(f"\n{'='*70}")
    print("PHASE 3: Hippocampal encoding and reconstruction")
    print("=" * 70)

    results_by_layer = {}

    for layer_idx in anchor_layers:
        print(f"\n  --- Layer {layer_idx} ---")
        vecs = hidden_states[layer_idx]
        n_vecs = len(vecs)

        # Input projection: simple random matrix, row-normalized
        # Exactly like v20's W_ec but mapping from d_model instead of N_features
        raw_proj = np.random.randn(d_ec, d_model)
        col_norms = np.linalg.norm(raw_proj, axis=0, keepdims=True) + 1e-10
        W_in = raw_proj / col_norms  # column-normalized like v20's W_ec

        # Build hippocampus with exact v20 parameters
        hipp = HippocampalSystem(
            d_ec, D_dg, N_ca3, N_ca1, k_ca3,
            dg_params=dg_params, ca1_params=ca1_params, sub_params=sub_params,
            ec_sup_params=ec_sup_params, N_sub=N_sub,
            ca3_retrieval_iterations=ca3_retrieval_iters)

        # Storage for output projection fitting
        gated_outputs = []   # hippocampal gated outputs (d_ec)
        original_vecs = []   # original hidden states (d_model)

        # Tracking
        recon_curve = []
        ca1_vs_stel = []
        sub_vs_pyr = []
        gated_vs_gated_ec = []
        ca1_fracs = []
        sub_fracs = []

        print(f"    Encoding {n_vecs} patterns...")
        for i, vec in enumerate(vecs):
            # Project to EC space: just matrix multiply
            ec_input = W_in @ vec

            # Encode (Hebbian learning)
            mm = hipp.encode_single(ec_input)

            # Get full stage outputs for diagnostics
            stages = hipp.get_stage_outputs(ec_input)

            # Track internal quality
            ca1_vs_stel.append(cosine_sim(stages["ca1"], stages["stellate"]))
            sub_vs_pyr.append(cosine_sim(stages["sub"], stages["pyramidal"]))
            gated_vs_gated_ec.append(
                cosine_sim(stages["gated_hippo"], stages["gated_ec"]))
            ca1_fracs.append(float(np.mean(stages["ca1"] > 0)))
            sub_fracs.append(float(np.mean(stages["sub"] > 0)))

            # Store for output projection
            gated_outputs.append(stages["gated_hippo"].copy())
            original_vecs.append(vec.copy())

            # Periodically fit output projection and measure end-to-end quality
            if (i + 1) % refit_interval == 0 or i == n_vecs - 1:
                # Fit ridge regression: gated_hippo (d_ec) -> original h (d_model)
                X = np.array(gated_outputs)
                Y = np.array(original_vecs)
                lam = 0.01
                XtX_inv = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
                W_out = (XtX_inv @ (X.T @ Y)).T  # (d_model, d_ec)
                b_out = Y.mean(axis=0) - W_out @ X.mean(axis=0)

                # Measure reconstruction on sample of stored patterns
                n_test = min(200, i + 1)
                test_idx = np.random.choice(i + 1, n_test, replace=False)
                test_sims = []
                for ti in test_idx:
                    ec_t = W_in @ vecs[ti]
                    s = hipp.get_stage_outputs(ec_t)
                    recon = W_out @ s["gated_hippo"] + b_out
                    test_sims.append(cosine_sim(recon, vecs[ti]))

                mean_ca1_stel = np.mean(ca1_vs_stel[-refit_interval:])
                mean_sub_pyr = np.mean(sub_vs_pyr[-refit_interval:])
                mean_gated = np.mean(gated_vs_gated_ec[-refit_interval:])
                mean_recon = np.mean(test_sims)

                recon_curve.append({
                    "n_stored": i + 1,
                    "recon_sim": float(mean_recon),
                    "ca1_vs_stel": float(mean_ca1_stel),
                    "sub_vs_pyr": float(mean_sub_pyr),
                    "gated_vs_gated_ec": float(mean_gated),
                    "ca1_frac": float(np.mean(ca1_fracs[-refit_interval:])),
                    "sub_frac": float(np.mean(sub_fracs[-refit_interval:])),
                })

                print(f"    n={i+1:5d}: recon={mean_recon:.4f}, "
                      f"ca1vStel={mean_ca1_stel:.4f}, subvPyr={mean_sub_pyr:.4f}, "
                      f"gatedSim={mean_gated:.4f}, "
                      f"ca1_frac={np.mean(ca1_fracs[-refit_interval:]):.3f}")

        # --- Final evaluation ---
        print(f"\n    Final evaluation:")

        # Refit on all data
        X = np.array(gated_outputs)
        Y = np.array(original_vecs)
        XtX_inv = np.linalg.inv(X.T @ X + 0.01 * np.eye(X.shape[1]))
        W_out = (XtX_inv @ (X.T @ Y)).T
        b_out = Y.mean(axis=0) - W_out @ X.mean(axis=0)

        all_sims = []
        for vec in vecs:
            ec = W_in @ vec
            s = hipp.get_stage_outputs(ec)
            recon = W_out @ s["gated_hippo"] + b_out
            all_sims.append(cosine_sim(recon, vec))

        print(f"    Stored patterns: cos_sim = {np.mean(all_sims):.4f} "
              f"+/- {np.std(all_sims):.4f}")
        print(f"    Worst 10%: {np.percentile(all_sims, 10):.4f}, "
              f"Best 10%: {np.percentile(all_sims, 90):.4f}")

        # Novel vectors (val set)
        novel_sims = []
        for vec in val_hidden_states[layer_idx]:
            ec = W_in @ vec
            s = hipp.get_stage_outputs(ec)
            recon = W_out @ s["gated_hippo"] + b_out
            novel_sims.append(cosine_sim(recon, vec))

        print(f"    Novel vectors:   cos_sim = {np.mean(novel_sims):.4f} "
              f"+/- {np.std(novel_sims):.4f}")

        # For comparison: what if we skip the hippocampus and just decode from
        # the EC input directly? (upper bound on what the projection can do)
        ec_direct = []
        for vec in vecs:
            ec_direct.append(W_in @ vec)
        X_ec = np.array(ec_direct)
        XtX_inv_ec = np.linalg.inv(X_ec.T @ X_ec + 0.01 * np.eye(X_ec.shape[1]))
        W_ec_direct = (XtX_inv_ec @ (X_ec.T @ Y)).T
        b_ec_direct = Y.mean(axis=0) - W_ec_direct @ X_ec.mean(axis=0)

        ec_direct_sims = []
        for vec in vecs:
            ec = W_in @ vec
            recon = W_ec_direct @ ec + b_ec_direct
            ec_direct_sims.append(cosine_sim(recon, vec))

        print(f"    EC direct (no hippo): cos_sim = {np.mean(ec_direct_sims):.4f} "
              f"+/- {np.std(ec_direct_sims):.4f}")

        # Novel via EC direct
        ec_direct_novel = []
        for vec in val_hidden_states[layer_idx]:
            ec = W_in @ vec
            recon = W_ec_direct @ ec + b_ec_direct
            ec_direct_novel.append(cosine_sim(recon, vec))

        print(f"    EC direct novel:      cos_sim = {np.mean(ec_direct_novel):.4f} "
              f"+/- {np.std(ec_direct_novel):.4f}")

        results_by_layer[layer_idx] = {
            "curve": recon_curve,
            "final_sim": float(np.mean(all_sims)),
            "final_sim_std": float(np.std(all_sims)),
            "novel_sim": float(np.mean(novel_sims)),
            "novel_sim_std": float(np.std(novel_sims)),
            "ec_direct_sim": float(np.mean(ec_direct_sims)),
            "ec_direct_novel_sim": float(np.mean(ec_direct_novel)),
            "all_sims": [float(s) for s in all_sims],
            "novel_sims": [float(s) for s in novel_sims],
        }

    # =================================================================
    # SAVE AND PLOT
    # =================================================================
    print(f"\n{'='*70}")
    print("Saving results...")
    print("=" * 70)

    results = {
        "params": {
            "d_model": d_model, "d_ec": d_ec,
            "D_dg": D_dg, "N_ca3": N_ca3, "N_ca1": N_ca1, "N_sub": N_sub,
            "k_ca3": k_ca3,
            "n_cortex_layers": n_cortex_layers,
            "anchor_layers": anchor_layers,
            "n_train": n_train, "n_val": n_val,
        },
        "layers": {str(l): results_by_layer[l] for l in anchor_layers},
    }

    with open("hippo_reconstruction_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved to hippo_reconstruction_v2_results.json")

    # --- Plot ---
    fig, axes = plt.subplots(2, len(anchor_layers), figsize=(6*len(anchor_layers), 10))
    fig.suptitle("Hippocampal Reconstruction v2: Exact v20 Hippocampus + Simple Projection",
                 fontsize=13, fontweight='bold')

    for col, layer_idx in enumerate(anchor_layers):
        res = results_by_layer[layer_idx]
        curve = res["curve"]

        # Top: quality over encoding
        ax = axes[0, col]
        ns = [c["n_stored"] for c in curve]
        recon = [c["recon_sim"] for c in curve]
        ca1s = [c["ca1_vs_stel"] for c in curve]
        subs = [c["sub_vs_pyr"] for c in curve]
        gated = [c["gated_vs_gated_ec"] for c in curve]

        ax.plot(ns, recon, 'o-', color="black", label="End-to-end recon", linewidth=2.5)
        ax.plot(ns, ca1s, 's-', color="steelblue", label="CA1 vs Stellate", linewidth=1.5)
        ax.plot(ns, subs, '^-', color="coral", label="Sub vs Pyramidal", linewidth=1.5)
        ax.plot(ns, gated, 'D-', color="forestgreen", label="Gated vs Gated EC", linewidth=1.5)
        ax.axhline(res["ec_direct_sim"], color="purple", linestyle="--",
                   label=f"EC direct ({res['ec_direct_sim']:.3f})", linewidth=1.5)
        ax.axhline(res["novel_sim"], color="gray", linestyle=":",
                   label=f"Novel ({res['novel_sim']:.3f})", linewidth=1.5)
        ax.set_xlabel("Patterns stored")
        ax.set_ylabel("Cosine similarity")
        ax.set_title(f"Layer {layer_idx}")
        ax.legend(fontsize=6, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.05)

        # Bottom: histogram
        ax = axes[1, col]
        ax.hist(res["all_sims"], bins=40, color="steelblue", alpha=0.6,
                label="Stored", edgecolor="black", linewidth=0.5)
        ax.hist(res["novel_sims"], bins=40, color="coral", alpha=0.6,
                label="Novel", edgecolor="black", linewidth=0.5)
        ax.axvline(np.mean(res["all_sims"]), color="steelblue", linestyle="--",
                   linewidth=2)
        ax.axvline(np.mean(res["novel_sims"]), color="coral", linestyle="--",
                   linewidth=2)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        ax.set_title(f"Layer {layer_idx}: Distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hippo_reconstruction_v2_results.png", dpi=150, bbox_inches='tight')
    print("  Saved plot to hippo_reconstruction_v2_results.png")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
