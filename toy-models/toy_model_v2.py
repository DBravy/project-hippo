"""
Toy Model v2: Superposition, Consolidation, and Computation
============================================================

Builds on v1 by adding the Toy Models of Superposition autoencoder:
  x̂ = ReLU(W^T W x + b)

We train this autoencoder on feature vectors from three sources:
1. Raw animal features (ground truth, sparse)
2. Hippocampal retrievals (superposed, noisy from cross-talk)
3. Cortical retrievals (consolidated, spectrally concentrated)

The question: does consolidation improve the autoencoder's ability
to reconstruct features through a bottleneck?
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import subspace_angles

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 1. WORLD GENERATION (same as v1)
# =============================================================================

k = 5           # latent compositional dimensions
N = 100         # animal types
d = 50          # representation dimensionality

# Ground truth compositional basis
latent_basis = np.linalg.qr(np.random.randn(d, k))[0][:, :k]

# Animal types as points in latent space + idiosyncratic noise
latent_coords = np.random.randn(N, k)
idiosyncratic_noise_scale = 0.3
animal_reps = latent_coords @ latent_basis.T
animal_reps += idiosyncratic_noise_scale * np.random.randn(N, d)
animal_norms = np.linalg.norm(animal_reps, axis=1, keepdims=True)
animal_reps = animal_reps / animal_norms

print(f"World: {k} latent dims, {N} animal types, {d}-dim space")

# =============================================================================
# 2. HIPPOCAMPAL STORAGE (no explicit noise - cross-talk IS the noise)
# =============================================================================

n_experiences = 500
sparsity = 0.05

W_hippo = np.zeros((d, d))
experiences = []

for t in range(n_experiences):
    active = np.random.rand(N) < sparsity
    if not active.any():
        active[np.random.randint(N)] = True
    pattern = animal_reps[active].sum(axis=0)
    pattern = pattern / (np.linalg.norm(pattern) + 1e-10)
    W_hippo += np.outer(pattern, pattern)
    experiences.append(pattern)

experiences = np.array(experiences)
print(f"Stored {n_experiences} experiences in hippocampal matrix")

# =============================================================================
# 3. CONSOLIDATION (no added noise - replay noise comes from cross-talk)
# =============================================================================

n_replay = 3000
W_cortex = np.zeros((d, d))

for cycle in range(n_replay):
    cue = experiences[np.random.randint(n_experiences)]
    # The retrieval IS noisy because W_hippo is superposed
    retrieval = W_hippo @ cue
    retrieval = retrieval / (np.linalg.norm(retrieval) + 1e-10)
    W_cortex += np.outer(retrieval, retrieval)

# Extract cortical principal components
eigvals_cortex, eigvecs_cortex = np.linalg.eigh(W_cortex)
eigvals_cortex = eigvals_cortex[::-1]
eigvecs_cortex = eigvecs_cortex[:, ::-1]

# Also extract hippocampal eigenvectors for comparison
eigvals_hippo, eigvecs_hippo = np.linalg.eigh(W_hippo)
eigvals_hippo = eigvals_hippo[::-1]
eigvecs_hippo = eigvecs_hippo[:, ::-1]

# Alignment checks
angles_hippo = subspace_angles(eigvecs_hippo[:, :k], latent_basis)
angles_cortex = subspace_angles(eigvecs_cortex[:, :k], latent_basis)
print(f"Hippocampal subspace alignment: {np.cos(angles_hippo).mean():.4f}")
print(f"Cortical subspace alignment:    {np.cos(angles_cortex).mean():.4f}")
print(f"Cortical spectral concentration (top-{k}): "
      f"{eigvals_cortex[:k].sum()/eigvals_cortex.sum():.4f}")

# =============================================================================
# 4. TOY MODELS OF SUPERPOSITION AUTOENCODER
# =============================================================================

class ToyModelAutoencoder(nn.Module):
    """
    The exact architecture from Toy Models of Superposition:
    x̂ = ReLU(W^T W x + b)

    W is m x n (bottleneck), so W^T W is n x n but rank m.
    """
    def __init__(self, n_features, n_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_features) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        # Encode: h = W @ x
        h = x @ self.W.T  # batch x n_hidden
        # Decode: W^T @ h + b, then ReLU
        x_hat = h @ self.W + self.b  # batch x n_features
        x_hat = torch.relu(x_hat)
        return x_hat

    @property
    def WTW(self):
        """The effective weight matrix W^T W"""
        return (self.W.T @ self.W).detach().cpu().numpy()


def generate_sparse_features(n_samples, n_features, sparsity_prob, importances=None):
    """
    Generate sparse feature vectors as in the Toy Models paper.
    Each feature is 0 with probability sparsity_prob,
    otherwise uniform [0, 1].
    """
    mask = (torch.rand(n_samples, n_features) > sparsity_prob).float()
    values = torch.rand(n_samples, n_features)
    x = mask * values
    return x


def train_autoencoder(model, n_features, feature_sparsity, importances,
                      n_steps=10000, batch_size=256, lr=1e-3,
                      transform_fn=None, label=""):
    """
    Train the Toy Models autoencoder.

    transform_fn: optional function applied to feature vectors before
    feeding to the model. This is how we test hippocampal vs cortical
    representations.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        x = generate_sparse_features(batch_size, n_features, feature_sparsity)
        x = x.to(device)

        if transform_fn is not None:
            x_input = transform_fn(x)
        else:
            x_input = x

        x_hat = model(x_input)

        # Importance-weighted MSE (reconstruct original features from
        # potentially transformed input)
        if transform_fn is not None:
            # Reconstruct the transformed version
            loss = (importances * (x_input - x_hat) ** 2).mean()
        else:
            loss = (importances * (x - x_hat) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (step + 1) % 2000 == 0:
            print(f"  [{label}] Step {step+1}/{n_steps}, Loss: {loss.item():.6f}")

    return losses


# =============================================================================
# 5. EXPERIMENT: Train autoencoders on raw vs projected features
# =============================================================================

# We'll use N=20 features compressed into m=5 hidden dims
# (matching the spirit of the paper's 5 features in 2 dims, scaled up)
n_feat = 20
m_hidden = 5
feature_sparsity = 0.9  # each feature is 0 with prob 0.9
importances = torch.tensor([0.7 ** i for i in range(n_feat)],
                           dtype=torch.float32).to(device)

print(f"\n{'='*60}")
print(f"AUTOENCODER EXPERIMENT")
print(f"{'='*60}")
print(f"Features: {n_feat}, Hidden: {m_hidden}, Sparsity: {feature_sparsity}")
print(f"Importances: exponentially decaying (0.7^i)")

# --- Model A: Standard (raw features, as in the original paper) ---
print(f"\n--- Training Model A: Raw features (standard Toy Models setup) ---")
model_A = ToyModelAutoencoder(n_feat, m_hidden).to(device)
losses_A = train_autoencoder(model_A, n_feat, feature_sparsity, importances,
                             n_steps=10000, label="Raw")

# --- Model B: Features pre-projected through a "hippocampal-like" matrix ---
# Simulate: features are read through a superposed storage matrix
# This adds cross-talk noise proportional to the number of stored features
hippo_matrix_small = np.random.randn(n_feat, n_feat) * 0.02
# Add structure: store random sparse patterns
for _ in range(200):
    pat = np.zeros(n_feat)
    active_idx = np.random.choice(n_feat, size=max(1, int(n_feat * 0.1)), replace=False)
    pat[active_idx] = np.random.rand(len(active_idx))
    pat = pat / (np.linalg.norm(pat) + 1e-10)
    hippo_matrix_small += np.outer(pat, pat)

hippo_tensor = torch.tensor(hippo_matrix_small, dtype=torch.float32).to(device)

def hippo_transform(x):
    """Read features through hippocampal superposition"""
    out = x @ hippo_tensor.T
    # Normalize each sample
    norms = out.norm(dim=1, keepdim=True) + 1e-10
    out = out / norms * x.norm(dim=1, keepdim=True)
    return torch.relu(out)  # biological: only positive activations

print(f"\n--- Training Model B: Hippocampal (superposed) features ---")
model_B = ToyModelAutoencoder(n_feat, m_hidden).to(device)
losses_B = train_autoencoder(model_B, n_feat, feature_sparsity, importances,
                             n_steps=10000, transform_fn=hippo_transform,
                             label="Hippo")

# --- Model C: Features pre-projected through "cortical" principal components ---
# Simulate consolidation: take the top-k PCs of the hippocampal matrix
# and project features through them (clean, low-rank projection)
evals_sm, evecs_sm = np.linalg.eigh(hippo_matrix_small)
evals_sm = evals_sm[::-1]
evecs_sm = evecs_sm[:, ::-1]

# Project through top-k components, then reconstruct
# This simulates what consolidation does: keep only the principal structure
n_cortical_dims = m_hidden  # cortex has extracted the top-m directions
cortical_proj = evecs_sm[:, :n_cortical_dims]  # n_feat x n_cortical_dims
cortical_matrix = cortical_proj @ cortical_proj.T  # low-rank projection
cortical_tensor = torch.tensor(cortical_matrix, dtype=torch.float32).to(device)

def cortical_transform(x):
    """Read features through consolidated cortical projection"""
    out = x @ cortical_tensor.T
    norms = out.norm(dim=1, keepdim=True) + 1e-10
    out = out / norms * x.norm(dim=1, keepdim=True)
    return torch.relu(out)

print(f"\n--- Training Model C: Cortical (consolidated) features ---")
model_C = ToyModelAutoencoder(n_feat, m_hidden).to(device)
losses_C = train_autoencoder(model_C, n_feat, feature_sparsity, importances,
                             n_steps=10000, transform_fn=cortical_transform,
                             label="Cortex")

# =============================================================================
# 6. ANALYSIS: Compare learned representations
# =============================================================================

# Get W^T W matrices for each model
WTW_A = model_A.WTW
WTW_B = model_B.WTW
WTW_C = model_C.WTW

# Count "represented features" (features with ||W_i||^2 > threshold)
def count_features(model, threshold=0.1):
    W = model.W.detach().cpu().numpy()
    norms_sq = (W ** 2).sum(axis=0)  # sum over hidden dims for each feature
    return (norms_sq > threshold).sum(), norms_sq

n_repr_A, norms_A = count_features(model_A)
n_repr_B, norms_B = count_features(model_B)
n_repr_C, norms_C = count_features(model_C)

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Final losses:")
print(f"  Raw features:        {np.mean(losses_A[-100:]):.6f}")
print(f"  Hippocampal:         {np.mean(losses_B[-100:]):.6f}")
print(f"  Cortical:            {np.mean(losses_C[-100:]):.6f}")
print(f"\nFeatures represented (||W_i||^2 > 0.1):")
print(f"  Raw: {n_repr_A}/{n_feat}")
print(f"  Hippocampal: {n_repr_B}/{n_feat}")
print(f"  Cortical: {n_repr_C}/{n_feat}")

# =============================================================================
# 7. VISUALIZATION
# =============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Toy Model v2: Computation Through Superposed vs Consolidated Representations",
             fontsize=13, fontweight='bold')

# --- Panel 1: Training losses ---
ax = axes[0, 0]
window = 100
smooth_A = np.convolve(losses_A, np.ones(window)/window, mode='valid')
smooth_B = np.convolve(losses_B, np.ones(window)/window, mode='valid')
smooth_C = np.convolve(losses_C, np.ones(window)/window, mode='valid')
ax.semilogy(smooth_A, 'b-', alpha=0.8, label='Raw features')
ax.semilogy(smooth_B, 'r-', alpha=0.8, label='Hippocampal')
ax.semilogy(smooth_C, 'g-', alpha=0.8, label='Cortical')
ax.set_xlabel('Training step')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Training Loss Curves')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 2: W^T W matrices ---
ax = axes[0, 1]
# Show all three as a horizontal triptych
combined = np.concatenate([WTW_A, np.ones((n_feat, 1))*np.nan, 
                          WTW_B, np.ones((n_feat, 1))*np.nan,
                          WTW_C], axis=1)
vmax = max(abs(WTW_A).max(), abs(WTW_B).max(), abs(WTW_C).max())
im = ax.imshow(combined, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_title('W^T W matrices: Raw | Hippo | Cortical')
ax.set_ylabel('Input feature')
# Add separators
ax.axvline(x=n_feat + 0.5, color='black', linewidth=2)
ax.axvline(x=2*n_feat + 1.5, color='black', linewidth=2)
plt.colorbar(im, ax=ax, fraction=0.046)

# --- Panel 3: Feature norms (how much each feature is represented) ---
ax = axes[0, 2]
x_pos = np.arange(n_feat)
ax.bar(x_pos - 0.25, norms_A, 0.25, label='Raw', color='steelblue', alpha=0.7)
ax.bar(x_pos, norms_B, 0.25, label='Hippocampal', color='firebrick', alpha=0.7)
ax.bar(x_pos + 0.25, norms_C, 0.25, label='Cortical', color='forestgreen', alpha=0.7)
ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Threshold')
ax.set_xlabel('Feature index (sorted by importance)')
ax.set_ylabel('||W_i||^2')
ax.set_title('Feature Representation Strength')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 4: Reconstruction quality per feature ---
ax = axes[1, 0]
n_test = 1000
x_test = generate_sparse_features(n_test, n_feat, feature_sparsity).to(device)

with torch.no_grad():
    recon_A = model_A(x_test)
    recon_B = model_B(hippo_transform(x_test))
    recon_C = model_C(cortical_transform(x_test))

    mse_per_feat_A = ((x_test - recon_A) ** 2).mean(dim=0).cpu().numpy()
    mse_per_feat_B = ((hippo_transform(x_test) - recon_B) ** 2).mean(dim=0).cpu().numpy()
    mse_per_feat_C = ((cortical_transform(x_test) - recon_C) ** 2).mean(dim=0).cpu().numpy()

ax.plot(mse_per_feat_A, 'bo-', markersize=4, label='Raw')
ax.plot(mse_per_feat_B, 'rs-', markersize=4, label='Hippocampal')
ax.plot(mse_per_feat_C, 'g^-', markersize=4, label='Cortical')
ax.set_xlabel('Feature index (sorted by importance)')
ax.set_ylabel('MSE')
ax.set_title('Per-Feature Reconstruction Error')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Panel 5: Eigenvalue spectra of W^T W ---
ax = axes[1, 1]
evals_WTW_A = np.linalg.eigvalsh(WTW_A)[::-1]
evals_WTW_B = np.linalg.eigvalsh(WTW_B)[::-1]
evals_WTW_C = np.linalg.eigvalsh(WTW_C)[::-1]
ax.plot(evals_WTW_A[:15], 'bo-', markersize=5, label='Raw')
ax.plot(evals_WTW_B[:15], 'rs-', markersize=5, label='Hippocampal')
ax.plot(evals_WTW_C[:15], 'g^-', markersize=5, label='Cortical')
ax.axvline(x=m_hidden - 0.5, color='gray', linestyle='--', alpha=0.5,
           label=f'm={m_hidden} hidden dims')
ax.set_xlabel('Component index')
ax.set_ylabel('Eigenvalue')
ax.set_title('Spectrum of learned W^T W')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Panel 6: Consolidation spectral comparison (from v1) ---
ax = axes[1, 2]
frac_hippo = eigvals_hippo[:15] / eigvals_hippo.sum()
frac_cortex = eigvals_cortex[:15] / eigvals_cortex.sum()
ax.plot(frac_hippo, 'ro-', markersize=5, label='Hippocampal storage')
ax.plot(frac_cortex, 'gs-', markersize=5, label='After consolidation')
ax.axvline(x=k-0.5, color='gray', linestyle='--', alpha=0.5, label=f'k={k} true dims')
ax.set_xlabel('Component index')
ax.set_ylabel('Fraction of total variance')
ax.set_title('Consolidation: Spectral Concentration')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('toy_model_v2_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nPlot saved to toy_model_v2_results.png")