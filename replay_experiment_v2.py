"""
Replay Experiment v2: Drifting Data (Rhyme but Not Repeat)
===========================================================

Same as v1 but each epoch, training inputs drift slightly from their
previous values (small Gaussian perturbation + re-normalize). The
underlying transformation (U, V) stays fixed. Each sample at epoch t
is a slight modification of the same sample at epoch t-1, mirroring
how human experience rhymes but never exactly repeats.

drift_rate controls how much inputs change per epoch.
At drift_rate=0.05, adjacent-epoch same-sample cosine similarity is ~0.99.
Over 200 epochs, samples drift substantially from their starting points.
"""

import numpy as np
import copy
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# 1. HIPPOCAMPAL SYSTEM (exact v20, unchanged)
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
    x = np.clip(x, -500, 500)
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


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

    def retrieve(self, x_ca3, x_ec_stel=None):
        h_sc = np.maximum(self.W_sc @ x_ca3, 0)
        h_out = apply_kwta(h_sc, self.k_active)
        mismatch = 0.0
        if x_ec_stel is not None:
            error = x_ec_stel - h_out
            mismatch = float(np.linalg.norm(error) / (np.linalg.norm(x_ec_stel) + 1e-10))
        return h_out, mismatch


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


class ECDeepVb:
    @staticmethod
    def gate(sparse_signal, dense_signal):
        gamma = sigmoid(dense_signal)
        return gamma * sparse_signal

    @staticmethod
    def hippocampal_output(ca1_output, sub_output):
        return ECDeepVb.gate(ca1_output, sub_output)


class HippocampalSystem:
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
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        self.ca3.store_online(dg_out)
        self.ca1.encode(dg_out, stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
        self.sub.encode(ca1_out, pyramidal)
        return mm

    def recall(self, ec_input):
        stellate, pyramidal = self.ec_sup.forward(ec_input)
        dg_out = self.dg.forward(stellate)
        ca3_out = self.ca3.retrieve(dg_out, self.ca3_retrieval_iterations)
        ca1_out, mm = self.ca1.retrieve(ca3_out, stellate)
        sub_out = self.sub.replay(ca1_out)
        gated = ECDeepVb.hippocampal_output(ca1_out, sub_out)
        return gated, mm


# =============================================================================
# 2. REPLAY MEMORY (unchanged from v1)
# =============================================================================

class ReplayMemory:
    def __init__(self, d_model, d_ec, hippo_kwargs, W_in):
        self.d_model = d_model
        self.d_ec = d_ec
        self.W_in = W_in.copy()
        self.hippo = HippocampalSystem(**hippo_kwargs)
        self.raw_h4 = []
        self.raw_targets = []
        self.gated_outputs = []
        self.W_out = np.zeros((d_model, d_ec))
        self.b_out = np.zeros(d_model)
        self._fitted = False
        self.n_stored = 0

    def store(self, h4_np, target_np):
        ec = self.W_in @ h4_np
        mm = self.hippo.encode_single(ec)
        gated, _ = self.hippo.recall(ec)
        self.raw_h4.append(h4_np.copy())
        self.raw_targets.append(target_np.copy())
        self.gated_outputs.append(gated.copy())
        self.n_stored += 1
        return mm

    def refit_output_projection(self):
        if self.n_stored < 50:
            return
        X = np.array(self.gated_outputs)
        Y = np.array(self.raw_h4)
        lam = 0.01
        XtX_inv = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
        self.W_out = (XtX_inv @ (X.T @ Y)).T
        self.b_out = Y.mean(axis=0) - self.W_out @ X.mean(axis=0)
        self._fitted = True

    def recall_hippo(self):
        idx = np.random.randint(self.n_stored)
        raw = self.raw_h4[idx]
        ec = self.W_in @ raw
        gated, mm = self.hippo.recall(ec)
        recon = self.W_out @ gated + self.b_out
        return recon, self.raw_targets[idx]

    def recall_oracle(self):
        idx = np.random.randint(self.n_stored)
        return self.raw_h4[idx], self.raw_targets[idx]

    def measure_recon_quality(self, n_samples=100):
        if not self._fitted or self.n_stored < 10:
            return 0.0
        sims = []
        for _ in range(min(n_samples, self.n_stored)):
            idx = np.random.randint(self.n_stored)
            raw = self.raw_h4[idx]
            ec = self.W_in @ raw
            gated, _ = self.hippo.recall(ec)
            recon = self.W_out @ gated + self.b_out
            sims.append(cosine_sim(recon, raw))
        return float(np.mean(sims))


# =============================================================================
# 3. CORTEX (unchanged)
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

    def forward_from_layer(self, h, start_layer):
        for i in range(start_layer, self.n_layers):
            h = self.layers[i](h)
        return self.output_head(h)

    def upper_parameters(self, start_layer):
        params = []
        for i in range(start_layer, self.n_layers):
            params.extend(self.layers[i].parameters())
        params.extend(self.output_head.parameters())
        return params


# =============================================================================
# 4. TASK: drifting data across epochs
# =============================================================================

class TaskGenerator:
    """
    Maintains a fixed set of input vectors that drift slowly across epochs.
    Same underlying transformation (U, V) throughout. Each epoch, each input
    gets a small random perturbation and is re-normalized. Targets are
    recomputed from the drifted inputs.

    This produces "rhyme but not repeat": sample i at epoch 10 is similar to
    sample i at epoch 9 (high cosine similarity) but not identical. Over many
    epochs, samples drift significantly from their starting points.

    drift_rate controls the perturbation magnitude per epoch.
    """
    def __init__(self, d_model, n_samples, n_components=20,
                 drift_rate=0.05, transform_seed=42, data_seed=100):
        self.d_model = d_model
        self.n_samples = n_samples
        self.drift_rate = drift_rate

        # Fixed transformation
        rng_t = np.random.RandomState(transform_seed)
        self.U = rng_t.randn(d_model, n_components).astype(np.float32) * 0.3
        self.V = rng_t.randn(n_components, d_model).astype(np.float32) * 0.3

        # Compute target scale
        test_rng = np.random.RandomState(transform_seed + 1000)
        test_in = test_rng.randn(10000, d_model).astype(np.float32)
        test_in = test_in / (np.linalg.norm(test_in, axis=1, keepdims=True) + 1e-10)
        test_tgt = np.tanh(test_in @ self.U) @ self.V
        self.target_scale = float(np.std(test_tgt)) + 1e-10

        # Initialize base inputs
        rng_d = np.random.RandomState(data_seed)
        self.inputs = rng_d.randn(n_samples, d_model).astype(np.float32)
        self.inputs = self.inputs / (np.linalg.norm(
            self.inputs, axis=1, keepdims=True) + 1e-10)

        # RNG for drift (separate so conditions can share drift pattern)
        self._drift_rng = np.random.RandomState(data_seed + 7777)

    def _compute_targets(self, inputs):
        h = np.tanh(inputs @ self.U)
        targets = (h @ self.V) / self.target_scale
        return targets.astype(np.float32)

    def generate_epoch(self):
        """
        Drift inputs, recompute targets, return (inputs, targets) tensors.
        Call once per epoch.
        """
        # Drift: add small noise, re-normalize
        noise = self._drift_rng.randn(
            self.n_samples, self.d_model).astype(np.float32) * self.drift_rate
        self.inputs = self.inputs + noise
        self.inputs = self.inputs / (np.linalg.norm(
            self.inputs, axis=1, keepdims=True) + 1e-10)

        targets = self._compute_targets(self.inputs)
        return (
            torch.tensor(self.inputs.copy(), dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )

    def generate_fixed(self, n_samples, seed=99999):
        """Generate a fixed val set (no drift)."""
        rng = np.random.RandomState(seed)
        inputs = rng.randn(n_samples, self.d_model).astype(np.float32)
        inputs = inputs / (np.linalg.norm(inputs, axis=1, keepdims=True) + 1e-10)
        targets = self._compute_targets(inputs)
        return (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )


def evaluate(cortex, inputs, targets, device):
    cortex.eval()
    with torch.no_grad():
        preds, _ = cortex(inputs.to(device))
        loss = F.mse_loss(preds, targets.to(device))
    cortex.train()
    return loss.item()


# =============================================================================
# 5. TRAINING LOOP
# =============================================================================

def train_condition(
    cortex, replay_mem, task_gen, val_inputs, val_targets, device,
    n_epochs, samples_per_epoch, batch_size, lr, weight_decay,
    replay_layer, replay_interval, replay_lr, replay_mode,
    output_refit_interval=200, log_interval=10,
):
    main_optimizer = torch.optim.AdamW(
        cortex.parameters(), lr=lr, weight_decay=weight_decay)
    upper_params = cortex.upper_parameters(replay_layer)
    replay_optimizer = torch.optim.SGD(upper_params, lr=replay_lr)

    log = {"val": [], "train": [], "replay_loss": [], "recon_quality": []}
    global_step = 0
    min_stored = 100

    for epoch in range(n_epochs):
        # Generate drifted training data for this epoch
        train_inputs, train_targets = task_gen.generate_epoch()

        epoch_train_loss = 0.0
        epoch_replay_loss = 0.0
        n_train_batches = 0
        n_replay_batches = 0

        perm = torch.randperm(samples_per_epoch)

        for i in range(0, samples_per_epoch - batch_size + 1, batch_size):
            idx = perm[i:i+batch_size]
            x = train_inputs[idx].to(device)
            y = train_targets[idx].to(device)

            # === Normal training step ===
            main_optimizer.zero_grad()
            pred, states = cortex(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cortex.parameters(), max_norm=1.0)
            main_optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

            # === Store in hippocampus ===
            if replay_mem is not None:
                h4 = states[replay_layer][0].detach().cpu().numpy().astype(np.float64)
                tgt = y[0].detach().cpu().numpy().astype(np.float64)
                replay_mem.store(h4, tgt)

                if global_step % output_refit_interval == 0:
                    replay_mem.refit_output_projection()

            # === Replay step ===
            if (replay_mode != 'none'
                    and replay_mem is not None
                    and replay_mem._fitted
                    and replay_mem.n_stored >= min_stored
                    and global_step % replay_interval == 0):

                if replay_mode == 'hippo':
                    h4_replay, y_replay = replay_mem.recall_hippo()
                elif replay_mode == 'oracle':
                    h4_replay, y_replay = replay_mem.recall_oracle()
                else:
                    raise ValueError(f"Unknown: {replay_mode}")

                h4_t = torch.tensor(h4_replay, dtype=torch.float32).unsqueeze(0).to(device)
                y_t = torch.tensor(y_replay, dtype=torch.float32).unsqueeze(0).to(device)

                replay_optimizer.zero_grad()
                pred_replay = cortex.forward_from_layer(h4_t, replay_layer)
                r_loss = F.mse_loss(pred_replay, y_t)
                r_loss.backward()
                torch.nn.utils.clip_grad_norm_(upper_params, max_norm=1.0)
                replay_optimizer.step()

                epoch_replay_loss += r_loss.item()
                n_replay_batches += 1

            global_step += 1

        avg_train = epoch_train_loss / max(n_train_batches, 1)
        avg_replay = epoch_replay_loss / max(n_replay_batches, 1) if n_replay_batches > 0 else 0
        val_loss = evaluate(cortex, val_inputs, val_targets, device)

        log["train"].append(avg_train)
        log["val"].append(val_loss)
        log["replay_loss"].append(avg_replay)

        recon_q = 0.0
        if replay_mem is not None:
            recon_q = replay_mem.measure_recon_quality()
        log["recon_quality"].append(recon_q)

        if epoch % log_interval == 0 or epoch == n_epochs - 1:
            stored = replay_mem.n_stored if replay_mem else 0
            print(f"    epoch {epoch:4d}: train={avg_train:.6f}, val={val_loss:.6f}, "
                  f"replay={avg_replay:.4f}, recon={recon_q:.4f}, stored={stored}")

    return log


# =============================================================================
# 6. MAIN
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
    samples_per_epoch = 2000  # Fresh samples each epoch
    n_val = 500
    n_epochs = 400
    batch_size = 32
    lr = 3e-4
    weight_decay = 0.01

    replay_layer = 4
    replay_interval = 3
    replay_lr = 1e-4
    output_refit_interval = 200
    drift_rate = 0.05  # How much inputs change per epoch

    d_ec = 1000
    hippo_kwargs = {
        "d_ec": d_ec, "D_dg": d_ec, "N_ca3": d_ec, "N_ca1": d_ec,
        "k_ca3": 50, "N_sub": d_ec, "ca3_retrieval_iterations": 5,
        "ec_sup_params": {"sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
                          "pyr_to_stel_strength": 0.3},
        "dg_params": {"sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
                      "noise_scale": 0.0},
        "ca1_params": {"lr": 50.0, "weight_decay": 1.000, "k_active": 50},
        "sub_params": {"lr": 1.0, "k_active": 500},
    }

    # Shared input projection
    rng_proj = np.random.RandomState(999)
    raw_proj = rng_proj.randn(d_ec, d_model)
    col_norms = np.linalg.norm(raw_proj, axis=0, keepdims=True) + 1e-10
    shared_W_in = raw_proj / col_norms

    print("=" * 70)
    print("Replay Experiment v2: Drifting Data (Rhyme but Not Repeat)")
    print("=" * 70)
    print(f"\nCortex: d={d_model}, layers={n_layers}")
    print(f"Task: {samples_per_epoch} samples/epoch x {n_epochs} epochs, "
          f"drift_rate={drift_rate}")
    print(f"Val: {n_val} fixed samples")
    print(f"Replay: layer={replay_layer}, interval={replay_interval}, lr={replay_lr}")
    print(f"Hippo: d_ec={d_ec}, k_ca3=50")

    # --- Val data (fixed, no drift) ---
    task_gen_val = TaskGenerator(
        d_model, samples_per_epoch, drift_rate=drift_rate,
        transform_seed=42, data_seed=100)
    val_inputs, val_targets = task_gen_val.generate_fixed(n_val, seed=99999)
    print(f"Val target std: {val_targets.std():.3f}")

    # Verify drift rate: how much do inputs change per epoch?
    task_gen_check = TaskGenerator(
        d_model, samples_per_epoch, drift_rate=drift_rate,
        transform_seed=42, data_seed=100)
    prev_inputs, _ = task_gen_check.generate_epoch()
    curr_inputs, _ = task_gen_check.generate_epoch()
    epoch_sims = [cosine_sim(
        prev_inputs[i].numpy().astype(np.float64),
        curr_inputs[i].numpy().astype(np.float64))
        for i in range(min(200, samples_per_epoch))]
    print(f"Adjacent-epoch same-sample cosine sim: {np.mean(epoch_sims):.4f} "
          f"+/- {np.std(epoch_sims):.4f}")

    # --- Initial weights ---
    init_cortex = ResidualStack(d_model, d_hidden, n_layers)
    init_state = copy.deepcopy(init_cortex.state_dict())
    trainable = sum(p.numel() for p in init_cortex.parameters())
    print(f"Cortex params: {trainable:,}")

    # =================================================================
    # CONDITION 1: No replay
    # =================================================================
    print(f"\n{'='*70}")
    print("CONDITION 1: No replay")
    print("=" * 70)

    cortex_none = ResidualStack(d_model, d_hidden, n_layers).to(device)
    cortex_none.load_state_dict(copy.deepcopy(init_state))
    cortex_none = cortex_none.to(device)

    np.random.seed(42)
    mem_none = ReplayMemory(d_model, d_ec, hippo_kwargs, shared_W_in)
    task_gen_none = TaskGenerator(
        d_model, samples_per_epoch, drift_rate=drift_rate,
        transform_seed=42, data_seed=100)

    log_none = train_condition(
        cortex_none, mem_none, task_gen_none, val_inputs, val_targets, device,
        n_epochs, samples_per_epoch, batch_size, lr, weight_decay,
        replay_layer, replay_interval, replay_lr,
        replay_mode='none', output_refit_interval=output_refit_interval)

    # =================================================================
    # CONDITION 2: Hippocampal replay
    # =================================================================
    print(f"\n{'='*70}")
    print("CONDITION 2: Hippocampal replay")
    print("=" * 70)

    cortex_hippo = ResidualStack(d_model, d_hidden, n_layers).to(device)
    cortex_hippo.load_state_dict(copy.deepcopy(init_state))
    cortex_hippo = cortex_hippo.to(device)

    np.random.seed(42)
    mem_hippo = ReplayMemory(d_model, d_ec, hippo_kwargs, shared_W_in)
    task_gen_hippo = TaskGenerator(
        d_model, samples_per_epoch, drift_rate=drift_rate,
        transform_seed=42, data_seed=100)

    log_hippo = train_condition(
        cortex_hippo, mem_hippo, task_gen_hippo, val_inputs, val_targets, device,
        n_epochs, samples_per_epoch, batch_size, lr, weight_decay,
        replay_layer, replay_interval, replay_lr,
        replay_mode='hippo', output_refit_interval=output_refit_interval)

    # =================================================================
    # CONDITION 3: Oracle replay
    # =================================================================
    print(f"\n{'='*70}")
    print("CONDITION 3: Oracle replay")
    print("=" * 70)

    cortex_oracle = ResidualStack(d_model, d_hidden, n_layers).to(device)
    cortex_oracle.load_state_dict(copy.deepcopy(init_state))
    cortex_oracle = cortex_oracle.to(device)

    np.random.seed(42)
    mem_oracle = ReplayMemory(d_model, d_ec, hippo_kwargs, shared_W_in)
    task_gen_oracle = TaskGenerator(
        d_model, samples_per_epoch, drift_rate=drift_rate,
        transform_seed=42, data_seed=100)

    log_oracle = train_condition(
        cortex_oracle, mem_oracle, task_gen_oracle, val_inputs, val_targets, device,
        n_epochs, samples_per_epoch, batch_size, lr, weight_decay,
        replay_layer, replay_interval, replay_lr,
        replay_mode='oracle', output_refit_interval=output_refit_interval)

    # =================================================================
    # SUMMARY
    # =================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)

    final_none = log_none["val"][-1]
    final_hippo = log_hippo["val"][-1]
    final_oracle = log_oracle["val"][-1]
    best_none = min(log_none["val"])
    best_hippo = min(log_hippo["val"])
    best_oracle = min(log_oracle["val"])

    print(f"  Final val:  none={final_none:.6f}, hippo={final_hippo:.6f}, oracle={final_oracle:.6f}")
    print(f"  Best val:   none={best_none:.6f}, hippo={best_hippo:.6f}, oracle={best_oracle:.6f}")

    if log_hippo["recon_quality"]:
        print(f"  Final hippo recon: {log_hippo['recon_quality'][-1]:.4f}")

    if best_none > 0:
        h_pct = (best_none - best_hippo) / best_none * 100
        o_pct = (best_none - best_oracle) / best_none * 100
        print(f"  Hippo vs none:  {h_pct:+.2f}%")
        print(f"  Oracle vs none: {o_pct:+.2f}%")

    total_stored = mem_hippo.n_stored
    print(f"\n  Drift rate: {drift_rate}")
    print(f"  Hippocampal snapshots stored: {total_stored:,}")

    # =================================================================
    # SAVE
    # =================================================================
    results = {
        "params": {
            "d_model": d_model, "d_hidden": d_hidden, "n_layers": n_layers,
            "samples_per_epoch": samples_per_epoch, "n_val": n_val,
            "n_epochs": n_epochs, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay,
            "replay_layer": replay_layer, "replay_interval": replay_interval,
            "replay_lr": replay_lr, "d_ec": d_ec, "drift_rate": drift_rate,
        },
        "no_replay": {"val": log_none["val"], "train": log_none["train"]},
        "hippo_replay": {
            "val": log_hippo["val"], "train": log_hippo["train"],
            "replay_loss": log_hippo["replay_loss"],
            "recon_quality": log_hippo["recon_quality"],
        },
        "oracle_replay": {
            "val": log_oracle["val"], "train": log_oracle["train"],
            "replay_loss": log_oracle["replay_loss"],
        },
        "summary": {
            "final_val": {"none": final_none, "hippo": final_hippo, "oracle": final_oracle},
            "best_val": {"none": best_none, "hippo": best_hippo, "oracle": best_oracle},
        },
    }

    with open("replay_experiment_v2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved to replay_experiment_v2_results.json")

    # =================================================================
    # PLOT
    # =================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Replay Experiment v2: Drifting Data (Rhyme but Not Repeat)",
                 fontsize=14, fontweight='bold')

    epochs = range(n_epochs)

    ax = axes[0, 0]
    ax.plot(epochs, log_none["val"], color="gray", label="No replay", linewidth=2)
    ax.plot(epochs, log_hippo["val"], color="steelblue", label="Hippo replay", linewidth=2)
    ax.plot(epochs, log_oracle["val"], color="coral", label="Oracle replay", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, log_none["train"], color="gray", label="No replay", linewidth=2)
    ax.plot(epochs, log_hippo["train"], color="steelblue", label="Hippo replay", linewidth=2)
    ax.plot(epochs, log_oracle["train"], color="coral", label="Oracle replay", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE")
    ax.set_title("Training Loss (per-epoch, fresh data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1, 0]
    ax.plot(epochs, log_hippo["replay_loss"], color="steelblue",
            label="Hippo replay", linewidth=2)
    ax.plot(epochs, log_oracle["replay_loss"], color="coral",
            label="Oracle replay", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Replay MSE")
    ax.set_title("Replay Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, log_hippo["recon_quality"], color="steelblue",
            label="Hippo recon quality", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Hippocampal Reconstruction Quality")
    ax.set_ylim(-0.1, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("replay_experiment_v2_results.png", dpi=150, bbox_inches='tight')
    print("  Saved to replay_experiment_v2_results.png")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
