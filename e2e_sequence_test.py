"""
End-to-End Sequence Retrieval Test
====================================

The full loop:
  1. Encode sequences (EC → DG → CA3 successor learning + direct decoder learning)
  2. Cue with first item's ec_input
  3. CA3 auto-associates through the sequence via recurrent weights
  4. Decode each retrieved CA3 state with both:
     a. Baseline 3-stage pipeline (CA1 → Sub → ECDeep)
     b. Direct delta-rule decoder (CA3 → ec_input)
  5. Compare reconstructions against original ec_inputs

Tests:
  1. Per-step quality at varying capacity (n_seq = 5, 10, 20, 50)
  2. Longer sequences (seq_length = 6, 10, 15) at fixed capacity
  3. Sensitivity to CA3 retrieval noise: how does readout degrade
     as a function of CA3 trajectory fidelity?
"""

import numpy as np
import torch
import json

torch.manual_seed(42)
np.random.seed(42)

from output_circuit_test import (
    HippocampalSystem,
    generate_sequences,
    cosine_sim,
)


class DirectDecoder:
    """Single-layer delta-rule decoder: CA3 → ec_input."""

    def __init__(self, d_output, N_ca3, lr=0.3, device='cpu', dtype=torch.float32):
        self.W = torch.zeros((d_output, N_ca3), device=device, dtype=dtype)
        self.lr = lr

    def encode(self, ca3_state, ec_input):
        prediction = self.W @ ca3_state
        error = ec_input - prediction
        self.W += self.lr * torch.outer(error, ca3_state)

    def retrieve(self, ca3_state):
        return self.W @ ca3_state


def build_system(hippo_kwargs, device, dtype):
    torch.manual_seed(42)
    return HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)


def encode_sequences(hippo, decoder, sequences, n_reps):
    """
    Encode sequences through the full system + direct decoder.
    Returns stored CA3 patterns and EC inputs from first rep.
    """
    all_ca3 = [[] for _ in sequences]
    all_ec = [[] for _ in sequences]

    for rep in range(n_reps):
        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for step_idx, ec_pattern in enumerate(seq):
                diag = hippo.encode_step(ec_pattern)
                decoder.encode(diag['ca3_state'], ec_pattern)

                if rep == 0:
                    all_ca3[seq_idx].append(diag['ca3_state'])
                    all_ec[seq_idx].append(ec_pattern.clone())
            hippo.end_sequence()

    return all_ca3, all_ec


def retrieve_and_evaluate(hippo, decoder, sequences, all_ca3, all_ec,
                          seq_length, sample_n=None,
                          cue_g_mf=2.0, cue_g_pp=1.0):
    """
    For each sequence:
      - Cue CA3 with first item
      - Auto-associate through sequence
      - Decode each step with both methods
      - Compare against originals
    """
    n_seq = len(sequences)
    if sample_n is None:
        sample_n = min(n_seq, 10)

    per_step = {
        'ca3_sim': [[] for _ in range(seq_length)],
        'baseline_sim': [[] for _ in range(seq_length)],
        'direct_sim': [[] for _ in range(seq_length)],
    }

    for si in range(sample_n):
        # Retrieve CA3 trajectory
        ca3_traj = hippo.recall_ca3_trajectory(
            sequences[si][0], n_steps=seq_length,
            cue_g_mf=cue_g_mf, cue_g_pp=cue_g_pp, run_g_recurrent=1.0)

        for t in range(seq_length):
            # CA3 retrieval fidelity
            ca3_sim = cosine_sim(ca3_traj[t], all_ca3[si][t])
            per_step['ca3_sim'][t].append(ca3_sim)

            # Baseline 3-stage readout
            ro = hippo.readout(ca3_traj[t])
            base_sim = cosine_sim(ro['ec_deep_out'], all_ec[si][t])
            per_step['baseline_sim'][t].append(base_sim)

            # Direct decoder readout
            dec_out = decoder.retrieve(ca3_traj[t])
            dec_sim = cosine_sim(dec_out, all_ec[si][t])
            per_step['direct_sim'][t].append(dec_sim)

    # Aggregate per-step means
    result = {
        'ca3_per_step': [float(np.mean(s)) for s in per_step['ca3_sim']],
        'baseline_per_step': [float(np.mean(s)) for s in per_step['baseline_sim']],
        'direct_per_step': [float(np.mean(s)) for s in per_step['direct_sim']],
        'ca3_per_step_std': [float(np.std(s)) for s in per_step['ca3_sim']],
        'baseline_per_step_std': [float(np.std(s)) for s in per_step['baseline_sim']],
        'direct_per_step_std': [float(np.std(s)) for s in per_step['direct_sim']],
    }
    return result


# =========================================================================
# TEST 1: Capacity scaling (vary n_seq, fixed seq_length=6)
# =========================================================================

def test_capacity(hippo_kwargs, device, dtype, n_reps=5):
    print("=" * 70)
    print("TEST 1: End-to-End Retrieval vs Capacity")
    print("=" * 70)

    seq_length = 6
    n_seq_values = [5, 10, 20, 50]
    results = {}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length, hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=3000 + n_seq)

        hippo = build_system(hippo_kwargs, device, dtype)
        decoder = DirectDecoder(hippo_kwargs['d_ec'], hippo_kwargs['N_ca3'],
                                lr=0.3, device=device, dtype=dtype)

        all_ca3, all_ec = encode_sequences(hippo, decoder, sequences, n_reps)
        result = retrieve_and_evaluate(hippo, decoder, sequences,
                                       all_ca3, all_ec, seq_length)
        results[n_seq] = result

        ca3_str = " ".join(f"{v:.3f}" for v in result['ca3_per_step'])
        base_str = " ".join(f"{v:.3f}" for v in result['baseline_per_step'])
        dec_str = " ".join(f"{v:.3f}" for v in result['direct_per_step'])

        print(f"\n  n_seq={n_seq} ({n_seq * seq_length} total patterns):")
        print(f"    CA3 retrieval:   [{ca3_str}]")
        print(f"    Baseline (3stg): [{base_str}]")
        print(f"    Direct decoder:  [{dec_str}]")
        print(f"    Mean CA3:  {np.mean(result['ca3_per_step']):.4f}  "
              f"Mean base: {np.mean(result['baseline_per_step']):.4f}  "
              f"Mean direct: {np.mean(result['direct_per_step']):.4f}")

    return results


# =========================================================================
# TEST 2: Sequence length scaling (fixed n_seq=10, vary seq_length)
# =========================================================================

def test_seq_length(hippo_kwargs, device, dtype, n_reps=5):
    print("\n" + "=" * 70)
    print("TEST 2: End-to-End Retrieval vs Sequence Length")
    print("=" * 70)

    n_seq = 10
    seq_lengths = [6, 10, 15]
    results = {}

    for seq_length in seq_lengths:
        sequences = generate_sequences(n_seq, seq_length, hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=4000 + seq_length)

        hippo = build_system(hippo_kwargs, device, dtype)
        decoder = DirectDecoder(hippo_kwargs['d_ec'], hippo_kwargs['N_ca3'],
                                lr=0.3, device=device, dtype=dtype)

        all_ca3, all_ec = encode_sequences(hippo, decoder, sequences, n_reps)
        result = retrieve_and_evaluate(hippo, decoder, sequences,
                                       all_ca3, all_ec, seq_length)
        results[seq_length] = result

        ca3_str = " ".join(f"{v:.3f}" for v in result['ca3_per_step'])
        base_str = " ".join(f"{v:.3f}" for v in result['baseline_per_step'])
        dec_str = " ".join(f"{v:.3f}" for v in result['direct_per_step'])

        print(f"\n  seq_length={seq_length} ({n_seq * seq_length} total patterns):")
        print(f"    CA3 retrieval:   [{ca3_str}]")
        print(f"    Baseline (3stg): [{base_str}]")
        print(f"    Direct decoder:  [{dec_str}]")

    return results


# =========================================================================
# TEST 3: Readout quality as a function of CA3 fidelity
# =========================================================================

def test_noise_sensitivity(hippo_kwargs, device, dtype, n_reps=5):
    """
    For stored CA3 patterns, add controlled noise and measure how
    each readout method degrades. This separates readout robustness
    from CA3 retrieval quality.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Readout vs CA3 Noise (controlled degradation)")
    print("=" * 70)

    n_seq = 10
    seq_length = 6
    sequences = generate_sequences(n_seq, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=5000)

    hippo = build_system(hippo_kwargs, device, dtype)
    decoder = DirectDecoder(hippo_kwargs['d_ec'], hippo_kwargs['N_ca3'],
                            lr=0.3, device=device, dtype=dtype)

    all_ca3, all_ec = encode_sequences(hippo, decoder, sequences, n_reps)

    noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
    results = {}

    for noise in noise_levels:
        base_sims = []
        direct_sims = []
        ca3_sims = []

        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3_clean = all_ca3[seq_idx][t]
                ec_in = all_ec[seq_idx][t]

                # Add noise to CA3 and re-apply k-WTA
                if noise > 0:
                    perturbation = torch.randn_like(ca3_clean) * noise * ca3_clean.norm()
                    ca3_noisy = ca3_clean + perturbation
                    # Re-apply k-WTA to maintain sparsity
                    k = int((ca3_clean > 0).sum())
                    topk_vals, topk_idx = torch.topk(ca3_noisy, k)
                    ca3_noisy = torch.zeros_like(ca3_noisy)
                    ca3_noisy[topk_idx] = torch.relu(topk_vals)
                    norm = ca3_noisy.norm()
                    if norm > 1e-10:
                        ca3_noisy = ca3_noisy / norm
                else:
                    ca3_noisy = ca3_clean

                ca3_sim = cosine_sim(ca3_noisy, ca3_clean)
                ca3_sims.append(ca3_sim)

                # Baseline
                ro = hippo.readout(ca3_noisy)
                base_sims.append(cosine_sim(ro['ec_deep_out'], ec_in))

                # Direct decoder
                dec_out = decoder.retrieve(ca3_noisy)
                direct_sims.append(cosine_sim(dec_out, ec_in))

        results[noise] = {
            'ca3_fidelity': float(np.mean(ca3_sims)),
            'baseline': float(np.mean(base_sims)),
            'direct': float(np.mean(direct_sims)),
            'baseline_std': float(np.std(base_sims)),
            'direct_std': float(np.std(direct_sims)),
        }

        print(f"  noise={noise:.1f}: "
              f"CA3 fidelity={np.mean(ca3_sims):.4f}  "
              f"Baseline={np.mean(base_sims):.4f}  "
              f"Direct={np.mean(direct_sims):.4f}")

    return results


# =========================================================================
# MAIN
# =========================================================================

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
        "ec_sup_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "pyr_to_stel_strength": 0.3,
        },
        "dg_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "noise_scale": 0.0,
        },
        "ca1_params": {
            "lr": 0.3,
            "plateau_threshold": 0.7,
            "plateau_sharpness": 20.0,
            "weight_decay": 1.0,
            "div_norm_sigma": 0.1,
            "connectivity_prob": 0.33,
            "ltd_rate": 0.00,
            "ltd_ca3_threshold": 0.0,
            "sigma_inh": 25,
            "gamma_inh": 4.0,
            "n_inh_steps": 5,
            "E_inh": -0.4,
        },
        "sub_params": {
            "lr": 0.05,
            "ltd_rate": 0.00,
            "connectivity_prob": 0.33,
        },
        "ec_deep_params": {
            "lr": 1.0,
            "weight_decay": 0.998,
        },
    }

    all_results = {}

    all_results["capacity"] = test_capacity(hippo_kwargs, device, dtype, n_reps=5)
    all_results["seq_length"] = test_seq_length(hippo_kwargs, device, dtype, n_reps=5)
    all_results["noise"] = test_noise_sensitivity(hippo_kwargs, device, dtype, n_reps=5)

    # --- Save ---
    # Convert dict keys to strings for JSON
    def stringify_keys(d):
        if isinstance(d, dict):
            return {str(k): stringify_keys(v) for k, v in d.items()}
        return d

    out_path = "e2e_sequence_results.json"
    with open(out_path, 'w') as f:
        json.dump(stringify_keys(all_results), f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nCapacity (mean across sequence positions):")
    for n_seq, r in all_results["capacity"].items():
        print(f"  n_seq={n_seq:3d}: "
              f"CA3={np.mean(r['ca3_per_step']):.4f}  "
              f"Baseline={np.mean(r['baseline_per_step']):.4f}  "
              f"Direct={np.mean(r['direct_per_step']):.4f}")

    print("\nSequence length (mean across positions):")
    for sl, r in all_results["seq_length"].items():
        print(f"  len={sl:2d}: "
              f"CA3={np.mean(r['ca3_per_step']):.4f}  "
              f"Baseline={np.mean(r['baseline_per_step']):.4f}  "
              f"Direct={np.mean(r['direct_per_step']):.4f}")

    print("\nNoise robustness:")
    for noise, r in all_results["noise"].items():
        print(f"  noise={noise:.1f}: "
              f"CA3={r['ca3_fidelity']:.4f}  "
              f"Baseline={r['baseline']:.4f}  "
              f"Direct={r['direct']:.4f}")


if __name__ == "__main__":
    main()
