"""
Toy Model v3: Fair Comparison of Computation Through Different Substrates
=========================================================================

Key fix from v2: ALL models reconstruct the SAME original features.
The difference is only in the intermediate representation they must
compute through.

Setup:
- n input features (sparse, importance-weighted)
- The features pass through a "representational substrate" (identity,
  hippocampal, or cortical) before entering the autoencoder bottleneck
- The autoencoder must reconstruct the ORIGINAL features from the
  substrate-transformed input
- This tests: given the same reconstruction target, which substrate
  makes the bottleneck computation easier?

Architecture per condition:
  x -> substrate_transform -> W_enc -> [bottleneck] -> W_dec -> ReLU -> x̂
  Loss = Σ I_i (x_i - x̂_i)²  (always against original x)
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1. WORLD GENERATION
# =============================================================================

k = 5           # latent compositional dimensions
N_animals = 100 # animal types (specific data generators)
d = 50          # full representation dimensionality

# Ground truth compositional basis
latent_basis = np.linalg.qr(np.random.randn(d, k))[0][:, :k]

# Animal types
latent_coords = np.random.randn(N_animals, k)
animal_reps = latent_coords @ latent_basis.T
animal_reps += 0.3 * np.random.randn(N_animals, d)
animal_reps = animal_reps / np.linalg.norm(animal_reps, axis=1, keepdims=True)

# =============================================================================
# 2. HIPPOCAMPAL STORAGE AND CONSOLIDATION
# =============================================================================

n_experiences = 500
sparsity = 0.05

W_hippo = np.zeros((d, d))
experiences = []

for t in range(n_experiences):
    active = np.random.rand(N_animals) < sparsity
    if not active.any():
        active[np.random.randint(N_animals)] = True
    pattern = animal_reps[active].sum(axis=0)
    pattern = pattern / (np.linalg.norm(pattern) + 1e-10)
    W_hippo += np.outer(pattern, pattern)
    experiences.append(pattern)
experiences = np.array(experiences)

# Consolidation: replay without added noise
n_replay = 3000
W_cortex = np.zeros((d, d))
for cycle in range(n_replay):
    cue = experiences[np.random.randint(n_experiences)]
    retrieval = W_hippo @ cue
    retrieval = retrieval / (np.linalg.norm(retrieval) + 1e-10)
    W_cortex += np.outer(retrieval, retrieval)

# Spectral analysis
eigvals_hippo, eigvecs_hippo = np.linalg.eigh(W_hippo)
eigvals_hippo, eigvecs_hippo = eigvals_hippo[::-1], eigvecs_hippo[:, ::-1]
eigvals_cortex, eigvecs_cortex = np.linalg.eigh(W_cortex)
eigvals_cortex, eigvecs_cortex = eigvals_cortex[::-1], eigvecs_cortex[:, ::-1]

print(f"Hippocampal spectral concentration (top-{k}): "
      f"{eigvals_hippo[:k].sum()/eigvals_hippo.sum():.4f}")
print(f"Cortical spectral concentration (top-{k}): "
      f"{eigvals_cortex[:k].sum()/eigvals_cortex.sum():.4f}")

# =============================================================================
# 3. AUTOENCODER WITH SUBSTRATE TRANSFORM
# =============================================================================

class SubstrateAutoencoder(nn.Module):
    """
    Toy Models architecture with a fixed (non-learned) substrate transform.

    Forward pass:
      h_sub = substrate @ x        (fixed, not learned)
      h_enc = W_enc @ h_sub        (learned, bottleneck)
      x̂ = ReLU(W_dec @ h_enc + b)  (learned, reconstruction)

    The substrate simulates how features are represented in different
    memory systems before computation happens.
    """
    def __init__(self, n_features, n_substrate, n_bottleneck):
        super().__init__()
        self.n_features = n_features
        self.n_substrate = n_substrate
        self.n_bottleneck = n_bottleneck

        # Learned encoder/decoder (the "computation" part)
        self.W_enc = nn.Parameter(torch.randn(n_bottleneck, n_substrate) * 0.05)
        self.W_dec = nn.Parameter(torch.randn(n_features, n_bottleneck) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_features))

        # Substrate transform (fixed, not learned)
        self.substrate = nn.Parameter(torch.eye(n_substrate, n_features),
                                      requires_grad=False)

    def set_substrate(self, matrix):
        """Set the substrate transform matrix (n_substrate x n_features)"""
        self.substrate.data = torch.tensor(matrix, dtype=torch.float32)

    def forward(self, x):
        # Pass through substrate (fixed)
        h_sub = x @ self.substrate.T  # batch x n_substrate
        # Encode through bottleneck (learned)
        h_enc = h_sub @ self.W_enc.T  # batch x n_bottleneck
        # Decode to reconstruct original features (learned)
        x_hat = h_enc @ self.W_dec.T + self.b  # batch x n_features
        x_hat = torch.relu(x_hat)
        return x_hat


def generate_sparse_features(n_samples, n_features, sparsity_prob):
    """Sparse features as in Toy Models paper."""
    mask = (torch.rand(n_samples, n_features) > sparsity_prob).float()
    values = torch.rand(n_samples, n_features)
    return mask * values


def train_model(model, n_features, feat_sparsity, importances,
                n_steps=15000, batch_size=256, lr=1e-3, label=""):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        x = generate_sparse_features(batch_size, n_features, feat_sparsity).to(device)
        x_hat = model(x)

        # Always reconstruct ORIGINAL features
        loss = (importances * (x - x_hat) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 3000 == 0:
            print(f"  [{label}] Step {step+1}/{n_steps}, Loss: {loss.item():.6f}")
    return losses


# =============================================================================
# 4. EXPERIMENTAL SETUP
# =============================================================================

# Feature space parameters
n_feat = 20         # number of sparse features
n_substrate = 20    # substrate dimensionality (same as features for fair comparison)
n_bottleneck = 3    # TIGHT bottleneck - fewer than latent dims, forces compression

feat_sparsity = 0.9
importances = torch.tensor([0.7 ** i for i in range(n_feat)],
                           dtype=torch.float32).to(device)
n_train_steps = 15000

print(f"\n{'='*60}")
print(f"EXPERIMENT: {n_feat} features -> {n_substrate}d substrate -> "
      f"{n_bottleneck}d bottleneck -> reconstruct {n_feat} features")
print(f"Sparsity: {feat_sparsity}, Steps: {n_train_steps}")
print(f"{'='*60}")

# --- Build substrate matrices ---

# Identity substrate (raw features, no transformation)
substrate_identity = np.eye(n_substrate, n_feat)

# Hippocampal substrate: features pass through superposed storage
# We create a n_feat-sized hippocampal matrix by storing random sparse
# patterns of the feature space
W_hippo_feat = np.zeros((n_feat, n_feat))
for _ in range(300):
    pat = np.zeros(n_feat)
    active_idx = np.random.choice(n_feat, size=max(1, int(n_feat * 0.15)), replace=False)
    pat[active_idx] = np.random.rand(len(active_idx))
    pat = pat / (np.linalg.norm(pat) + 1e-10)
    W_hippo_feat += np.outer(pat, pat)
# Normalize to preserve scale
W_hippo_feat = W_hippo_feat / np.linalg.norm(W_hippo_feat) * np.sqrt(n_feat)
substrate_hippo = W_hippo_feat[:n_substrate, :]

# Cortical substrate: replay-consolidated version
W_cortex_feat = np.zeros((n_feat, n_feat))
# Generate some "experience" patterns in feature space
feat_experiences = []
for _ in range(300):
    pat = np.zeros(n_feat)
    active_idx = np.random.choice(n_feat, size=max(1, int(n_feat * 0.15)), replace=False)
    pat[active_idx] = np.random.rand(len(active_idx))
    pat = pat / (np.linalg.norm(pat) + 1e-10)
    feat_experiences.append(pat)
    W_hippo_feat_for_replay = W_hippo_feat.copy()

# Consolidation via replay (no added noise)
for _ in range(2000):
    cue = feat_experiences[np.random.randint(len(feat_experiences))]
    retrieval = W_hippo_feat_for_replay @ cue
    retrieval = retrieval / (np.linalg.norm(retrieval) + 1e-10)
    W_cortex_feat += np.outer(retrieval, retrieval)
W_cortex_feat = W_cortex_feat / np.linalg.norm(W_cortex_feat) * np.sqrt(n_feat)
substrate_cortex = W_cortex_feat[:n_substrate, :]

# Analyze substrate spectra
evals_sub_id = np.linalg.eigvalsh(substrate_identity.T @ substrate_identity)[::-1]
evals_sub_h = np.linalg.eigvalsh(substrate_hippo.T @ substrate_hippo)[::-1]
evals_sub_c = np.linalg.eigvalsh(substrate_cortex.T @ substrate_cortex)[::-1]

print(f"\nSubstrate effective ranks (ratio of sum to max eigenvalue):")
print(f"  Identity: {evals_sub_id.sum()/evals_sub_id[0]:.2f}")
print(f"  Hippocampal: {evals_sub_h.sum()/evals_sub_h[0]:.2f}")
print(f"  Cortical: {evals_sub_c.sum()/evals_sub_c[0]:.2f}")

# --- Train models ---

# Model A: Identity substrate (standard Toy Models)
print(f"\n--- Model A: Identity substrate ---")
model_A = SubstrateAutoencoder(n_feat, n_substrate, n_bottleneck).to(device)
model_A.set_substrate(substrate_identity)
losses_A = train_model(model_A, n_feat, feat_sparsity, importances,
                       n_steps=n_train_steps, label="Identity")

# Model B: Hippocampal substrate
print(f"\n--- Model B: Hippocampal substrate ---")
model_B = SubstrateAutoencoder(n_feat, n_substrate, n_bottleneck).to(device)
model_B.set_substrate(substrate_hippo)
losses_B = train_model(model_B, n_feat, feat_sparsity, importances,
                       n_steps=n_train_steps, label="Hippo")

# Model C: Cortical substrate
print(f"\n--- Model C: Cortical substrate ---")
model_C = SubstrateAutoencoder(n_feat, n_substrate, n_bottleneck).to(device)
model_C.set_substrate(substrate_cortex)
losses_C = train_model(model_C, n_feat, feat_sparsity, importances,
                       n_steps=n_train_steps, label="Cortex")

# =============================================================================
# 5. EVALUATION
# =============================================================================

n_test = 5000
x_test = generate_sparse_features(n_test, n_feat, feat_sparsity).to(device)

with torch.no_grad():
    recon_A = model_A(x_test)
    recon_B = model_B(x_test)
    recon_C = model_C(x_test)

    mse_A = ((x_test - recon_A) ** 2).mean().item()
    mse_B = ((x_test - recon_B) ** 2).mean().item()
    mse_C = ((x_test - recon_C) ** 2).mean().item()

    wmse_A = (importances * (x_test - recon_A) ** 2).mean().item()
    wmse_B = (importances * (x_test - recon_B) ** 2).mean().item()
    wmse_C = (importances * (x_test - recon_C) ** 2).mean().item()

    # Per-feature MSE
    per_feat_A = ((x_test - recon_A) ** 2).mean(dim=0).cpu().numpy()
    per_feat_B = ((x_test - recon_B) ** 2).mean(dim=0).cpu().numpy()
    per_feat_C = ((x_test - recon_C) ** 2).mean(dim=0).cpu().numpy()

    # Per-feature weighted MSE
    imp_np = importances.cpu().numpy()
    per_feat_wmse_A = imp_np * per_feat_A
    per_feat_wmse_B = imp_np * per_feat_B
    per_feat_wmse_C = imp_np * per_feat_C

# Feature representation strength
def get_feature_norms(model):
    """||W_dec[:, j]||^2 for each feature j"""
    W_dec = model.W_dec.detach().cpu().numpy()
    return (W_dec ** 2).sum(axis=1)  # sum over bottleneck dims

norms_A = get_feature_norms(model_A)
norms_B = get_feature_norms(model_B)
norms_C = get_feature_norms(model_C)

# Learned W_enc @ substrate effective matrices
def get_effective_encoder(model):
    W_enc = model.W_enc.detach().cpu().numpy()
    sub = model.substrate.detach().cpu().numpy()
    return W_enc @ sub  # n_bottleneck x n_features

eff_enc_A = get_effective_encoder(model_A)
eff_enc_B = get_effective_encoder(model_B)
eff_enc_C = get_effective_encoder(model_C)

# =============================================================================
# 6. SAVE DETAILED RESULTS
# =============================================================================

results = {
    "config": {
        "n_features": n_feat,
        "n_substrate": n_substrate,
        "n_bottleneck": n_bottleneck,
        "feature_sparsity": feat_sparsity,
        "n_train_steps": n_train_steps,
        "n_test_samples": n_test,
        "k_latent_dims": k,
        "n_animals": N_animals,
        "d_full_space": d,
    },
    "substrate_analysis": {
        "identity_effective_rank": float(evals_sub_id.sum()/evals_sub_id[0]),
        "hippo_effective_rank": float(evals_sub_h.sum()/evals_sub_h[0]),
        "cortical_effective_rank": float(evals_sub_c.sum()/evals_sub_c[0]),
        "identity_eigenvalues_top10": evals_sub_id[:10].tolist(),
        "hippo_eigenvalues_top10": evals_sub_h[:10].tolist(),
        "cortical_eigenvalues_top10": evals_sub_c[:10].tolist(),
    },
    "consolidation": {
        "hippo_spectral_concentration_top5": float(eigvals_hippo[:k].sum()/eigvals_hippo.sum()),
        "cortical_spectral_concentration_top5": float(eigvals_cortex[:k].sum()/eigvals_cortex.sum()),
    },
    "training": {
        "identity_final_loss_avg100": float(np.mean(losses_A[-100:])),
        "hippo_final_loss_avg100": float(np.mean(losses_B[-100:])),
        "cortical_final_loss_avg100": float(np.mean(losses_C[-100:])),
        "identity_loss_trajectory": [float(losses_A[i]) for i in range(0, len(losses_A), 500)],
        "hippo_loss_trajectory": [float(losses_B[i]) for i in range(0, len(losses_B), 500)],
        "cortical_loss_trajectory": [float(losses_C[i]) for i in range(0, len(losses_C), 500)],
    },
    "evaluation": {
        "identity_mse": float(mse_A),
        "hippo_mse": float(mse_B),
        "cortical_mse": float(mse_C),
        "identity_weighted_mse": float(wmse_A),
        "hippo_weighted_mse": float(wmse_B),
        "cortical_weighted_mse": float(wmse_C),
        "per_feature_mse_identity": per_feat_A.tolist(),
        "per_feature_mse_hippo": per_feat_B.tolist(),
        "per_feature_mse_cortical": per_feat_C.tolist(),
    },
    "feature_representation": {
        "decoder_norms_identity": norms_A.tolist(),
        "decoder_norms_hippo": norms_B.tolist(),
        "decoder_norms_cortical": norms_C.tolist(),
        "n_represented_identity": int((norms_A > 0.01).sum()),
        "n_represented_hippo": int((norms_B > 0.01).sum()),
        "n_represented_cortical": int((norms_C > 0.01).sum()),
    },
    "effective_encoders": {
        "identity_singular_values": np.linalg.svd(eff_enc_A, compute_uv=False).tolist(),
        "hippo_singular_values": np.linalg.svd(eff_enc_B, compute_uv=False).tolist(),
        "cortical_singular_values": np.linalg.svd(eff_enc_C, compute_uv=False).tolist(),
    }
}

with open('toy_model_v3_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# 7. PRINT SUMMARY
# =============================================================================

print(f"\n{'='*60}")
print(f"RESULTS SUMMARY")
print(f"{'='*60}")
print(f"\nArchitecture: {n_feat} features -> {n_substrate}d substrate -> "
      f"{n_bottleneck}d bottleneck -> {n_feat} features")
print(f"All models reconstruct the SAME original features.")
print(f"\nSubstrate effective ranks:")
print(f"  Identity:    {evals_sub_id.sum()/evals_sub_id[0]:.2f}")
print(f"  Hippocampal: {evals_sub_h.sum()/evals_sub_h[0]:.2f}")
print(f"  Cortical:    {evals_sub_c.sum()/evals_sub_c[0]:.2f}")
print(f"\nTest set weighted MSE (importance-weighted, lower is better):")
print(f"  Identity:    {wmse_A:.6f}")
print(f"  Hippocampal: {wmse_B:.6f}")
print(f"  Cortical:    {wmse_C:.6f}")
print(f"\nTest set unweighted MSE:")
print(f"  Identity:    {mse_A:.6f}")
print(f"  Hippocampal: {mse_B:.6f}")
print(f"  Cortical:    {mse_C:.6f}")
print(f"\nFeatures with decoder norm > 0.01:")
print(f"  Identity:    {(norms_A > 0.01).sum()}/{n_feat}")
print(f"  Hippocampal: {(norms_B > 0.01).sum()}/{n_feat}")
print(f"  Cortical:    {(norms_C > 0.01).sum()}/{n_feat}")
print(f"\nPer-feature MSE (first 10, sorted by importance):")
print(f"  Identity:    {np.array2string(per_feat_A[:10], precision=5, separator=', ')}")
print(f"  Hippocampal: {np.array2string(per_feat_B[:10], precision=5, separator=', ')}")
print(f"  Cortical:    {np.array2string(per_feat_C[:10], precision=5, separator=', ')}")
print(f"\nEffective encoder singular values:")
print(f"  Identity:    {np.array2string(np.linalg.svd(eff_enc_A, compute_uv=False), precision=4)}")
print(f"  Hippocampal: {np.array2string(np.linalg.svd(eff_enc_B, compute_uv=False), precision=4)}")
print(f"  Cortical:    {np.array2string(np.linalg.svd(eff_enc_C, compute_uv=False), precision=4)}")

# =============================================================================
# 8. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Toy Model v3: Fair Comparison - Same Target, Different Substrates",
             fontsize=13, fontweight='bold')

# Panel 1: Training losses
ax = axes[0, 0]
w = 100
ax.semilogy(np.convolve(losses_A, np.ones(w)/w, 'valid'), 'b-', alpha=0.8, label='Identity')
ax.semilogy(np.convolve(losses_B, np.ones(w)/w, 'valid'), 'r-', alpha=0.8, label='Hippocampal')
ax.semilogy(np.convolve(losses_C, np.ones(w)/w, 'valid'), 'g-', alpha=0.8, label='Cortical')
ax.set_xlabel('Training step')
ax.set_ylabel('Weighted MSE (log)')
ax.set_title('Training Loss (all reconstruct original features)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Per-feature reconstruction error
ax = axes[0, 1]
x_pos = np.arange(n_feat)
ax.bar(x_pos - 0.25, per_feat_wmse_A, 0.25, label='Identity', color='steelblue', alpha=0.7)
ax.bar(x_pos, per_feat_wmse_B, 0.25, label='Hippocampal', color='firebrick', alpha=0.7)
ax.bar(x_pos + 0.25, per_feat_wmse_C, 0.25, label='Cortical', color='forestgreen', alpha=0.7)
ax.set_xlabel('Feature index (by importance)')
ax.set_ylabel('Importance-weighted MSE')
ax.set_title('Per-Feature Weighted Reconstruction Error')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Decoder feature norms
ax = axes[0, 2]
ax.bar(x_pos - 0.25, norms_A, 0.25, label='Identity', color='steelblue', alpha=0.7)
ax.bar(x_pos, norms_B, 0.25, label='Hippocampal', color='firebrick', alpha=0.7)
ax.bar(x_pos + 0.25, norms_C, 0.25, label='Cortical', color='forestgreen', alpha=0.7)
ax.set_xlabel('Feature index (by importance)')
ax.set_ylabel('||W_dec[:, j]||²')
ax.set_title('Decoder Feature Representation Strength')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 4: Substrate eigenvalue spectra
ax = axes[1, 0]
ax.plot(evals_sub_id[:n_feat], 'bo-', markersize=4, label='Identity')
ax.plot(evals_sub_h[:n_feat], 'rs-', markersize=4, label='Hippocampal')
ax.plot(evals_sub_c[:n_feat], 'g^-', markersize=4, label='Cortical')
ax.set_xlabel('Component index')
ax.set_ylabel('Eigenvalue of S^T S')
ax.set_title('Substrate Spectra (what the encoder sees)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 5: Effective encoder singular values
ax = axes[1, 1]
sv_A = np.linalg.svd(eff_enc_A, compute_uv=False)
sv_B = np.linalg.svd(eff_enc_B, compute_uv=False)
sv_C = np.linalg.svd(eff_enc_C, compute_uv=False)
ax.bar(np.arange(len(sv_A)) - 0.25, sv_A, 0.25, label='Identity', color='steelblue', alpha=0.7)
ax.bar(np.arange(len(sv_B)), sv_B, 0.25, label='Hippocampal', color='firebrick', alpha=0.7)
ax.bar(np.arange(len(sv_C)) + 0.25, sv_C, 0.25, label='Cortical', color='forestgreen', alpha=0.7)
ax.set_xlabel('Singular value index')
ax.set_ylabel('Singular value')
ax.set_title('Effective Encoder (W_enc @ Substrate) SVD')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 6: Consolidation effect (large-scale, from d=50 world)
ax = axes[1, 2]
frac_hippo = eigvals_hippo[:15] / eigvals_hippo.sum()
frac_cortex = eigvals_cortex[:15] / eigvals_cortex.sum()
ax.plot(frac_hippo, 'ro-', markersize=5, label='Hippocampal storage')
ax.plot(frac_cortex, 'gs-', markersize=5, label='After consolidation')
ax.axvline(x=k-0.5, color='gray', linestyle='--', alpha=0.5, label=f'k={k} true dims')
ax.set_xlabel('Component index')
ax.set_ylabel('Fraction of total variance')
ax.set_title('Consolidation Spectral Concentration (d=50 world)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('toy_model_v3_results.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to toy_model_v3_results.png")
print("Detailed results saved to toy_model_v3_results.json")