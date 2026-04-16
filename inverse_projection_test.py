"""
Inverse Projection Test
========================

Initialize ECDeep.W as [pinv(W_ta) | pinv(W_ec)] so that the return
path from CA1/Sub to EC is the mathematical inverse of the forward
teaching projections.

Tests three conditions:
  1. Baseline: ECDeep.W starts at zero, learned via Hebbian
  2. Inverse init + Hebbian: pinv initialization, then Hebbian learning on top
  3. Inverse fixed: pinv initialization, no Hebbian learning

Runs the same learning-curve protocol as test_learning_curves.
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


def build_system(hippo_kwargs, device, dtype):
    torch.manual_seed(42)
    return HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)


def init_ecdeep_inverse(hippo):
    """Replace ECDeep.W with [pinv(W_ta) | pinv(W_ec)]."""
    W_ta = hippo.ca1.W_ta   # (N_ca1, d_ec)
    W_ec = hippo.sub.W_ec   # (N_sub, d_ec)

    W_ta_pinv = torch.linalg.pinv(W_ta)  # (d_ec, N_ca1)
    W_ec_pinv = torch.linalg.pinv(W_ec)  # (d_ec, N_sub)

    hippo.ec_deep.W = torch.cat([W_ta_pinv, W_ec_pinv], dim=1).clone()
    # W shape: (d_ec, N_ca1 + N_sub) -- matches ECDeep.W shape

    print(f"    pinv(W_ta): {tuple(W_ta_pinv.shape)}, norm={W_ta_pinv.norm():.3f}")
    print(f"    pinv(W_ec): {tuple(W_ec_pinv.shape)}, norm={W_ec_pinv.norm():.3f}")
    print(f"    ECDeep.W:   {tuple(hippo.ec_deep.W.shape)}, norm={hippo.ec_deep.W.norm():.3f}")


def run_condition(hippo, sequences, n_reps, label, disable_hebbian=False):
    """
    Run encoding + readout evaluation, return per-rep metrics.
    """
    n_seq = len(sequences)
    seq_length = len(sequences[0])

    # Store CA3 patterns from first rep
    all_ca3 = [[] for _ in sequences]
    all_stel = [[] for _ in sequences]
    all_ec = [[] for _ in sequences]

    results = {
        "label": label,
        "reps": [],
        "ca1_mismatch": [],
        "ec_deep_sim_encoding": [],
        "readout_ec_deep_sim": [],
        "readout_ec_deep_sim_ec": [],
        "readout_ca1_vs_stel": [],
        "readout_sub_vs_pyr": [],
    }

    # Patch out Hebbian learning if requested
    original_encode = None
    if disable_hebbian:
        original_encode = hippo.ec_deep.encode
        hippo.ec_deep.encode = lambda *args, **kwargs: None

    for rep in range(n_reps):
        rep_ca1_mm = []
        rep_ecd_sim = []

        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for step_idx, ec_pattern in enumerate(seq):
                diag = hippo.encode_step(ec_pattern)
                rep_ca1_mm.append(diag['ca1_mismatch'])
                rep_ecd_sim.append(diag['ec_deep_sim'])

                if rep == 0:
                    all_ca3[seq_idx].append(diag['ca3_state'])
                    all_stel[seq_idx].append(diag['stellate'])
                    all_ec[seq_idx].append(ec_pattern.clone())
            hippo.end_sequence()

        # Evaluate readout
        readout_sims = []
        readout_sims_ec = []
        ca1_vs_stel = []
        sub_vs_pyr = []

        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3_pat = all_ca3[seq_idx][t]
                ec_in = all_ec[seq_idx][t]
                stel = all_stel[seq_idx][t]

                ro = hippo.readout(ca3_pat)
                readout_sims.append(cosine_sim(ro['ec_deep_out'], ec_in))
                ca1_vs_stel.append(cosine_sim(ro['ca1_out'], stel))

                # Sub vs pyramidal needs pyramidal, recompute it
                _, pyr = hippo.ec_sup.forward(ec_in)
                sub_vs_pyr.append(cosine_sim(ro['sub_out'], pyr))

                ro_ec = hippo.readout(ca3_pat, ec_stellate=stel)
                readout_sims_ec.append(cosine_sim(ro_ec['ec_deep_out'], ec_in))

        results["reps"].append(rep + 1)
        results["ca1_mismatch"].append(float(np.mean(rep_ca1_mm)))
        results["ec_deep_sim_encoding"].append(float(np.mean(rep_ecd_sim)))
        results["readout_ec_deep_sim"].append(float(np.mean(readout_sims)))
        results["readout_ec_deep_sim_ec"].append(float(np.mean(readout_sims_ec)))
        results["readout_ca1_vs_stel"].append(float(np.mean(ca1_vs_stel)))
        results["readout_sub_vs_pyr"].append(float(np.mean(sub_vs_pyr)))

        print(f"  [{label:20s}] Rep {rep+1:2d}: "
              f"CA1mm={np.mean(rep_ca1_mm):.4f}  "
              f"ECd(enc)={np.mean(rep_ecd_sim):.4f}  "
              f"ECd(ro)={np.mean(readout_sims):.4f}  "
              f"ECd(+EC)={np.mean(readout_sims_ec):.4f}  "
              f"CA1~st={np.mean(ca1_vs_stel):.4f}  "
              f"Sub~py={np.mean(sub_vs_pyr):.4f}")

    if original_encode is not None:
        hippo.ec_deep.encode = original_encode

    return results


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
    n_seq = 10
    seq_length = 6
    n_reps = 20

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

    sequences = generate_sequences(n_seq, seq_length, d_ec,
                                   device=device, dtype=dtype, seed=1000)

    all_results = {}

    # --- Condition 1: Baseline ---
    print("\n" + "=" * 70)
    print("CONDITION 1: Baseline (ECDeep.W = zeros, Hebbian learning)")
    print("=" * 70)
    hippo_base = build_system(hippo_kwargs, device, dtype)
    all_results["baseline"] = run_condition(
        hippo_base, sequences, n_reps, "baseline")

    # --- Condition 2: Inverse init + Hebbian ---
    print("\n" + "=" * 70)
    print("CONDITION 2: Inverse init (pinv) + Hebbian learning")
    print("=" * 70)
    hippo_inv = build_system(hippo_kwargs, device, dtype)
    init_ecdeep_inverse(hippo_inv)
    all_results["inverse_hebbian"] = run_condition(
        hippo_inv, sequences, n_reps, "inverse+hebbian")

    # --- Condition 3: Inverse fixed ---
    print("\n" + "=" * 70)
    print("CONDITION 3: Inverse init (pinv), NO Hebbian learning")
    print("=" * 70)
    hippo_fix = build_system(hippo_kwargs, device, dtype)
    init_ecdeep_inverse(hippo_fix)
    all_results["inverse_fixed"] = run_condition(
        hippo_fix, sequences, n_reps, "inverse+fixed",
        disable_hebbian=True)

    # --- Save results ---
    out_path = "inverse_projection_results.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY (Rep 20 readout ECDeep vs EC)")
    print("=" * 70)
    for key in ["baseline", "inverse_hebbian", "inverse_fixed"]:
        r = all_results[key]
        print(f"  {r['label']:22s}: "
              f"ECd(ro)={r['readout_ec_deep_sim'][-1]:.4f}  "
              f"ECd(+EC)={r['readout_ec_deep_sim_ec'][-1]:.4f}  "
              f"CA1~st={r['readout_ca1_vs_stel'][-1]:.4f}  "
              f"Sub~py={r['readout_sub_vs_pyr'][-1]:.4f}")


if __name__ == "__main__":
    main()
