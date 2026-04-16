"""
Direct Decoder Test
====================

Compare the existing 3-stage readout (CA3 → CA1 → Sub → ECDeep)
against a single learnable decoder from CA3 directly to stellate,
trained with a delta rule.

Tests:
  1. Learning curves over 20 reps (same protocol as test_learning_curves)
  2. Capacity scaling: vary n_seq from 1 to 50

Both conditions use identical CA3 encoding (same sequences, same seeds).
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
    """Single-layer delta-rule decoder: CA3 → target."""

    def __init__(self, d_output, N_ca3, lr=0.3, device='cpu', dtype=torch.float32):
        self.W = torch.zeros((d_output, N_ca3), device=device, dtype=dtype)
        self.lr = lr

    def encode(self, ca3_state, target):
        prediction = self.W @ ca3_state
        error = target - prediction
        self.W += self.lr * torch.outer(error, ca3_state)

    def retrieve(self, ca3_state):
        return self.W @ ca3_state


def build_system(hippo_kwargs, device, dtype):
    torch.manual_seed(42)
    return HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)


def run_learning_curves(hippo_kwargs, device, dtype, n_reps=20,
                        n_seq=10, seq_length=6, seed=1000,
                        direct_lr=0.3):
    """Run all conditions on identical data, return per-rep metrics."""

    sequences = generate_sequences(n_seq, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=seed)

    # --- Build systems (same seed = same encoding pathway) ---
    hippo = build_system(hippo_kwargs, device, dtype)
    dec_stel = DirectDecoder(hippo_kwargs['d_ec'], hippo_kwargs['N_ca3'],
                             lr=direct_lr, device=device, dtype=dtype)
    dec_ec = DirectDecoder(hippo_kwargs['d_ec'], hippo_kwargs['N_ca3'],
                           lr=direct_lr, device=device, dtype=dtype)

    # Storage from first rep
    all_ca3 = [[] for _ in sequences]
    all_stel = [[] for _ in sequences]
    all_ec = [[] for _ in sequences]

    baseline = {"reps": [], "readout_sim": [], "readout_sim_ec": [],
                "ca1_vs_stel": [], "ca1_mm": []}
    direct_stel = {"reps": [], "readout_sim": [], "decoder_vs_stel": []}
    direct_ecin = {"reps": [], "readout_sim": [], "decoder_vs_stel": []}

    for rep in range(n_reps):
        rep_ca1_mm = []

        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            for step_idx, ec_pattern in enumerate(seq):
                # Shared encoding through CA3
                diag = hippo.encode_step(ec_pattern)
                rep_ca1_mm.append(diag['ca1_mismatch'])

                if rep == 0:
                    all_ca3[seq_idx].append(diag['ca3_state'])
                    all_stel[seq_idx].append(diag['stellate'])
                    all_ec[seq_idx].append(ec_pattern.clone())

                # Both decoders learn alongside
                dec_stel.encode(diag['ca3_state'], diag['stellate'])
                dec_ec.encode(diag['ca3_state'], ec_pattern)

            hippo.end_sequence()

        # --- Evaluate all on stored patterns ---
        base_sims, base_sims_ec, ca1_stels = [], [], []
        dstel_sims, dstel_vs_stels = [], []
        dec_sims, dec_vs_stels = [], []

        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3 = all_ca3[seq_idx][t]
                ec_in = all_ec[seq_idx][t]
                stel = all_stel[seq_idx][t]

                # Baseline readout
                ro = hippo.readout(ca3)
                base_sims.append(cosine_sim(ro['ec_deep_out'], ec_in))
                ca1_stels.append(cosine_sim(ro['ca1_out'], stel))

                ro_ec = hippo.readout(ca3, ec_stellate=stel)
                base_sims_ec.append(cosine_sim(ro_ec['ec_deep_out'], ec_in))

                # Direct → stellate
                out_stel = dec_stel.retrieve(ca3)
                dstel_sims.append(cosine_sim(out_stel, ec_in))
                dstel_vs_stels.append(cosine_sim(out_stel, stel))

                # Direct → ec_input
                out_ec = dec_ec.retrieve(ca3)
                dec_sims.append(cosine_sim(out_ec, ec_in))
                dec_vs_stels.append(cosine_sim(out_ec, stel))

        baseline["reps"].append(rep + 1)
        baseline["readout_sim"].append(float(np.mean(base_sims)))
        baseline["readout_sim_ec"].append(float(np.mean(base_sims_ec)))
        baseline["ca1_vs_stel"].append(float(np.mean(ca1_stels)))
        baseline["ca1_mm"].append(float(np.mean(rep_ca1_mm)))

        direct_stel["reps"].append(rep + 1)
        direct_stel["readout_sim"].append(float(np.mean(dstel_sims)))
        direct_stel["decoder_vs_stel"].append(float(np.mean(dstel_vs_stels)))

        direct_ecin["reps"].append(rep + 1)
        direct_ecin["readout_sim"].append(float(np.mean(dec_sims)))
        direct_ecin["decoder_vs_stel"].append(float(np.mean(dec_vs_stels)))

        print(f"  Rep {rep+1:2d}: "
              f"Baseline={np.mean(base_sims):.4f}  "
              f"Direct~stel(vs ec)={np.mean(dstel_sims):.4f}  "
              f"Direct~ec={np.mean(dec_sims):.4f}  "
              f"Direct~ec(vs stel)={np.mean(dec_vs_stels):.4f}  "
              f"CA1mm={np.mean(rep_ca1_mm):.4f}")

    return baseline, direct_stel, direct_ecin


def run_capacity_test(hippo_kwargs, device, dtype, n_reps=5,
                      seq_length=6, direct_lr=0.3):
    """Vary n_seq, measure readout quality at the end of training."""

    n_seq_values = [1, 5, 10, 20, 50]
    baseline_cap = {"n_seq": [], "readout_sim": []}
    dstel_cap = {"n_seq": [], "readout_sim": [], "decoder_vs_stel": []}
    dec_cap = {"n_seq": [], "readout_sim": [], "decoder_vs_stel": []}

    for n_seq in n_seq_values:
        sequences = generate_sequences(n_seq, seq_length, hippo_kwargs['d_ec'],
                                       device=device, dtype=dtype,
                                       seed=2000 + n_seq)

        hippo = build_system(hippo_kwargs, device, dtype)
        dec_stel = DirectDecoder(hippo_kwargs['d_ec'], hippo_kwargs['N_ca3'],
                                 lr=direct_lr, device=device, dtype=dtype)
        dec_ec = DirectDecoder(hippo_kwargs['d_ec'], hippo_kwargs['N_ca3'],
                               lr=direct_lr, device=device, dtype=dtype)

        all_ca3 = [[] for _ in sequences]
        all_stel = [[] for _ in sequences]
        all_ec = [[] for _ in sequences]

        for rep in range(n_reps):
            for seq_idx, seq in enumerate(sequences):
                hippo.begin_sequence()
                for ec_pattern in seq:
                    diag = hippo.encode_step(ec_pattern)
                    if rep == 0:
                        all_ca3[seq_idx].append(diag['ca3_state'])
                        all_stel[seq_idx].append(diag['stellate'])
                        all_ec[seq_idx].append(ec_pattern.clone())
                    dec_stel.encode(diag['ca3_state'], diag['stellate'])
                    dec_ec.encode(diag['ca3_state'], ec_pattern)
                hippo.end_sequence()

        # Evaluate
        base_sims, dstel_sims, dstel_stels = [], [], []
        dec_sims, dec_stels = [], []
        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3 = all_ca3[seq_idx][t]
                ec_in = all_ec[seq_idx][t]
                stel = all_stel[seq_idx][t]

                ro = hippo.readout(ca3)
                base_sims.append(cosine_sim(ro['ec_deep_out'], ec_in))

                out_stel = dec_stel.retrieve(ca3)
                dstel_sims.append(cosine_sim(out_stel, ec_in))
                dstel_stels.append(cosine_sim(out_stel, stel))

                out_ec = dec_ec.retrieve(ca3)
                dec_sims.append(cosine_sim(out_ec, ec_in))
                dec_stels.append(cosine_sim(out_ec, stel))

        baseline_cap["n_seq"].append(n_seq)
        baseline_cap["readout_sim"].append(float(np.mean(base_sims)))
        dstel_cap["n_seq"].append(n_seq)
        dstel_cap["readout_sim"].append(float(np.mean(dstel_sims)))
        dstel_cap["decoder_vs_stel"].append(float(np.mean(dstel_stels)))
        dec_cap["n_seq"].append(n_seq)
        dec_cap["readout_sim"].append(float(np.mean(dec_sims)))
        dec_cap["decoder_vs_stel"].append(float(np.mean(dec_stels)))

        print(f"  n_seq={n_seq:4d}: "
              f"Baseline={np.mean(base_sims):.4f}  "
              f"Direct~stel(vs ec)={np.mean(dstel_sims):.4f}  "
              f"Direct~ec={np.mean(dec_sims):.4f}  "
              f"Direct~stel={np.mean(dstel_stels):.4f}")

    return baseline_cap, dstel_cap, dec_cap


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

    results = {}

    # --- Test 1: Learning curves ---
    print("=" * 70)
    print("TEST 1: Learning Curves (10 seq x 6 steps, 20 reps)")
    print("=" * 70)
    baseline_lc, dstel_lc, dec_lc = run_learning_curves(
        hippo_kwargs, device, dtype, n_reps=20)
    results["learning_curves"] = {
        "baseline": baseline_lc,
        "direct_stellate": dstel_lc,
        "direct_ec": dec_lc,
    }

    # --- Test 2: Capacity ---
    print("\n" + "=" * 70)
    print("TEST 2: Capacity (vary n_seq, 5 reps each)")
    print("=" * 70)
    baseline_cap, dstel_cap, dec_cap = run_capacity_test(
        hippo_kwargs, device, dtype, n_reps=5)
    results["capacity"] = {
        "baseline": baseline_cap,
        "direct_stellate": dstel_cap,
        "direct_ec": dec_cap,
    }

    # --- Save ---
    out_path = "direct_decoder_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("Learning curves (rep 20):")
    print(f"  Baseline ECDeep vs ec:     {baseline_lc['readout_sim'][-1]:.4f}")
    print(f"  Direct→stellate vs ec:     {dstel_lc['readout_sim'][-1]:.4f}")
    print(f"  Direct→stellate vs stel:   {dstel_lc['decoder_vs_stel'][-1]:.4f}")
    print(f"  Direct→ec_input vs ec:     {dec_lc['readout_sim'][-1]:.4f}")
    print(f"  Direct→ec_input vs stel:   {dec_lc['decoder_vs_stel'][-1]:.4f}")
    print(f"\nCapacity (n_seq sweep, 5 reps):")
    for i, n in enumerate(baseline_cap['n_seq']):
        print(f"  n_seq={n:3d}: "
              f"Baseline={baseline_cap['readout_sim'][i]:.4f}  "
              f"Direct→stel(vs ec)={dstel_cap['readout_sim'][i]:.4f}  "
              f"Direct→ec={dec_cap['readout_sim'][i]:.4f}  "
              f"Direct→ec(vs stel)={dec_cap['decoder_vs_stel'][i]:.4f}")


if __name__ == "__main__":
    main()