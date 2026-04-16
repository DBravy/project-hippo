"""
Output Pathway Diagnostics
===========================

Instruments every stage of CA3 -> CA1 -> Sub -> ECDeep to find where and why
the EC representation fails to improve. Outputs a detailed JSON file covering:

  1. Per-rep, per-pattern snapshots of every intermediate signal
  2. Weight matrix health (norms, sparsity, rank, spectral stats)
  3. CA1 gating/plateau analysis (gate openness, error before/after gating)
  4. Subiculum drive balance (CA1 drive vs EC drive, error magnitude)
  5. ECDeep input decomposition (CA1 contribution vs Sub contribution)
  6. Gradient-like diagnostics (update magnitudes vs weight magnitudes)
  7. Cross-pattern interference (pairwise similarity of representations)
  8. Signal propagation ratios (does information survive each transform?)

Run:
    python output_pathway_diagnostics.py [--n_reps 10] [--n_seq 10]
                                          [--seq_length 6] [--output diag.json]

Does NOT produce plots; the output is a single JSON file for external analysis.
"""

import argparse
import json
import sys
import numpy as np

import torch

# ---------------------------------------------------------------------------
# Import everything from the existing test file
# ---------------------------------------------------------------------------
from output_circuit_test import (
    HippocampalSystem,
    generate_sequences,
    cosine_sim,
    make_feedforward_weights,
)

torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# HELPER: safe tensor -> python conversion
# =============================================================================

def t2f(t):
    """Tensor or float -> python float."""
    if isinstance(t, torch.Tensor):
        return float(t.detach().cpu())
    return float(t)


def tensor_stats(t, name=""):
    """Return a dict of summary statistics for a 1-D or 2-D tensor."""
    t = t.detach().cpu().float()
    s = {
        "shape": list(t.shape),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "min": float(t.min()),
        "max": float(t.max()),
        "abs_mean": float(t.abs().mean()),
        "norm_fro": float(torch.linalg.norm(t)),
    }
    if t.dim() == 1:
        s["n_nonzero"] = int((t != 0).sum())
        s["n_positive"] = int((t > 0).sum())
        s["frac_positive"] = s["n_positive"] / max(t.numel(), 1)
        s["l1"] = float(t.abs().sum())
        # top-k concentration: what fraction of L1 is in top 5% of elements?
        k = max(1, t.numel() // 20)
        topk_vals, _ = torch.topk(t.abs(), k)
        s["top5pct_l1_frac"] = float(topk_vals.sum() / (s["l1"] + 1e-30))
    elif t.dim() == 2:
        s["n_nonzero"] = int((t != 0).sum())
        s["frac_nonzero"] = s["n_nonzero"] / max(t.numel(), 1)
        row_norms = torch.linalg.norm(t, dim=1)
        s["row_norm_mean"] = float(row_norms.mean())
        s["row_norm_std"] = float(row_norms.std())
        s["row_norm_min"] = float(row_norms.min())
        s["row_norm_max"] = float(row_norms.max())
        col_norms = torch.linalg.norm(t, dim=0)
        s["col_norm_mean"] = float(col_norms.mean())
        s["col_norm_std"] = float(col_norms.std())
        # Effective rank (Shannon entropy of singular values)
        try:
            sv = torch.linalg.svdvals(t)
            sv_normed = sv / (sv.sum() + 1e-30)
            entropy = -(sv_normed * torch.log(sv_normed + 1e-30)).sum()
            s["effective_rank"] = float(torch.exp(entropy))
            s["top_sv"] = float(sv[0])
            s["sv_ratio_top1_to_sum"] = float(sv[0] / (sv.sum() + 1e-30))
            n_sv = min(10, len(sv))
            s["top10_sv"] = [float(x) for x in sv[:n_sv]]
        except Exception:
            s["effective_rank"] = None
    return s


def pairwise_cosine_matrix(vectors):
    """Given a list of 1-D tensors, return the NxN cosine similarity matrix as list-of-lists."""
    n = len(vectors)
    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            c = cosine_sim(vectors[i], vectors[j])
            mat[i][j] = c
            mat[j][i] = c
    return mat


def pairwise_cosine_summary(vectors):
    """Summary stats of off-diagonal cosine similarities."""
    n = len(vectors)
    if n < 2:
        return {"mean": None, "std": None, "min": None, "max": None}
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(cosine_sim(vectors[i], vectors[j]))
    arr = np.array(vals)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_pairs": len(vals),
    }


def effective_rank_of_vectors(vectors):
    """
    Stack a list of 1-D tensors into a matrix and compute effective rank
    (Shannon entropy of normalized singular values), plus additional stats.
    """
    if len(vectors) < 2:
        return {"effective_rank": None}
    mat = torch.stack(vectors)  # (n_patterns, dim)
    try:
        sv = torch.linalg.svdvals(mat)
        sv_normed = sv / (sv.sum() + 1e-30)
        entropy = -(sv_normed * torch.log(sv_normed + 1e-30)).sum()
        n_sv = min(20, len(sv))
        return {
            "effective_rank": float(torch.exp(entropy)),
            "n_patterns": len(vectors),
            "dim": vectors[0].shape[0],
            "top_sv": float(sv[0]),
            "sv_ratio_top1_to_sum": float(sv[0] / (sv.sum() + 1e-30)),
            "sv_ratio_top1_to_top2": float(sv[0] / (sv[1] + 1e-30)) if len(sv) > 1 else None,
            "top20_sv": [float(x) for x in sv[:n_sv]],
            "sv_sum": float(sv.sum()),
            "n_sv_for_90pct": int((torch.cumsum(sv, 0) / (sv.sum() + 1e-30) < 0.9).sum()) + 1,
        }
    except Exception:
        return {"effective_rank": None}


def projection_quality_vs_input(projected_list, input_list):
    """
    For each pair (projected_i, input_i), measure cosine similarity.
    Returns summary stats of how well the projection preserves the input direction.
    """
    sims = []
    for proj, inp in zip(projected_list, input_list):
        sims.append(cosine_sim(proj, inp))
    arr = np.array(sims)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def centroid_deviation(vectors):
    """
    Measure how much each pattern's vector deviates from the centroid.
    High mean deviation = good (patterns are spread out).
    Low mean deviation = collapsed (everything looks like the mean).
    Returns cosine sim of each vector to the centroid.
    """
    if len(vectors) < 2:
        return {"mean_sim_to_centroid": None}
    stacked = torch.stack(vectors)
    centroid = stacked.mean(dim=0)
    sims = [cosine_sim(v, centroid) for v in vectors]
    arr = np.array(sims)
    return {
        "mean_sim_to_centroid": float(arr.mean()),
        "std_sim_to_centroid": float(arr.std()),
        "min_sim_to_centroid": float(arr.min()),
        "max_sim_to_centroid": float(arr.max()),
        "centroid_norm": float(torch.linalg.norm(centroid)),
    }


# =============================================================================
# CORE DIAGNOSTIC: detailed per-step encode instrumentation
# =============================================================================

def instrumented_encode_step(hippo, ec_input):
    """
    Replicate HippocampalSystem.encode_step but capture every intermediate
    value for diagnostics. Returns the same dict as encode_step plus a
    large 'diag' dict with all the gory details.
    """
    d = {}

    # --- EC Superficial ---
    stellate, pyramidal = hippo.ec_sup.forward(ec_input)
    d["ec_input"] = tensor_stats(ec_input, "ec_input")
    d["stellate"] = tensor_stats(stellate, "stellate")
    d["pyramidal"] = tensor_stats(pyramidal, "pyramidal")
    d["stellate_vs_ec_input"] = cosine_sim(stellate, ec_input)
    d["pyramidal_vs_ec_input"] = cosine_sim(pyramidal, ec_input)

    # --- DG + MF ---
    dg_out = hippo.dg.forward(stellate)
    mf_input = torch.relu(hippo.W_mf @ dg_out)
    d["dg_out"] = tensor_stats(dg_out, "dg_out")
    d["mf_input"] = tensor_stats(mf_input, "mf_input")

    # --- CA3 ---
    ca3_state = hippo.ca3.step(
        mf_input=mf_input, g_mf=1.0, g_recurrent=0.0, g_pp=0.0,
        learn=True)
    d["ca3_state"] = tensor_stats(ca3_state, "ca3_state")

    # --- Direct pathway ---
    stel_norm = stellate / (torch.linalg.norm(stellate) + 1e-10)
    direct_update = hippo.direct_lr * torch.outer(ca3_state, stel_norm)
    d["direct_update_norm"] = t2f(torch.linalg.norm(direct_update))
    hippo.W_direct += direct_update
    hippo.W_direct *= hippo.direct_decay

    # =====================================================================
    # CA1 DETAILED DIAGNOSTICS
    # =====================================================================
    ca1 = hippo.ca1

    # Raw drives before divisive normalization
    h_ta_raw = torch.relu(ca1.W_ta @ stellate)
    h_sc_raw = torch.relu(ca1.W_sc @ ca3_state)

    d["ca1_h_ta_raw"] = tensor_stats(h_ta_raw, "ca1_h_ta_raw")
    d["ca1_h_sc_raw"] = tensor_stats(h_sc_raw, "ca1_h_sc_raw")
    d["ca1_ta_to_sc_raw_ratio"] = t2f(
        torch.linalg.norm(h_ta_raw) / (torch.linalg.norm(h_sc_raw) + 1e-10))

    # After divisive normalization
    h_ta, h_sc, gate, h_ta_raw2, h_sc_raw2 = ca1.compute_activations(ca3_state, stellate)
    d["ca1_h_ta_normed"] = tensor_stats(h_ta, "ca1_h_ta_normed")
    d["ca1_h_sc_normed"] = tensor_stats(h_sc, "ca1_h_sc_normed")

    # Plateau gate analysis
    max_ta = float(h_ta_raw2.max())
    threshold = ca1.plateau_threshold * max_ta if max_ta > 1e-10 else 0.0
    d["ca1_gate"] = tensor_stats(gate, "ca1_gate")
    d["ca1_gate_threshold_value"] = threshold
    d["ca1_gate_mean"] = t2f(gate.mean())
    d["ca1_gate_frac_above_0.5"] = float((gate > 0.5).sum()) / len(gate)
    d["ca1_gate_frac_above_0.9"] = float((gate > 0.9).sum()) / len(gate)
    d["ca1_gate_frac_below_0.1"] = float((gate < 0.1).sum()) / len(gate)

    # Error analysis
    error = h_ta - h_sc
    gated_error = gate * error
    d["ca1_error"] = tensor_stats(error, "ca1_error")
    d["ca1_gated_error"] = tensor_stats(gated_error, "ca1_gated_error")
    d["ca1_error_norm"] = t2f(torch.linalg.norm(error))
    d["ca1_gated_error_norm"] = t2f(torch.linalg.norm(gated_error))
    d["ca1_gating_attenuation"] = t2f(
        torch.linalg.norm(gated_error) / (torch.linalg.norm(error) + 1e-10))

    # Actual weight update
    w_sc_update = ca1.lr * torch.outer(gated_error, ca3_state)
    d["ca1_W_sc_update_norm"] = t2f(torch.linalg.norm(w_sc_update))
    d["ca1_W_sc_update_to_weight_ratio"] = t2f(
        torch.linalg.norm(w_sc_update) / (torch.linalg.norm(ca1.W_sc) + 1e-10))

    # Now actually do the CA1 encode
    ca1.encode(ca3_state, stellate)

    # CA1 retrieval (unsupported)
    ca1_out, ca1_mismatch = ca1.retrieve(ca3_state, stellate)
    d["ca1_out"] = tensor_stats(ca1_out, "ca1_out")
    d["ca1_mismatch"] = ca1_mismatch
    d["ca1_out_vs_stellate"] = cosine_sim(ca1_out, stellate)
    d["ca1_out_vs_ca3"] = cosine_sim(ca1_out, ca3_state)

    # CA1 retrieval WITHOUT EC (pure Schaffer collateral)
    ca1_out_no_ec, _ = ca1.retrieve(ca3_state, None)
    d["ca1_out_no_ec"] = tensor_stats(ca1_out_no_ec, "ca1_out_no_ec")
    d["ca1_out_no_ec_vs_stellate"] = cosine_sim(ca1_out_no_ec, stellate)
    d["ca1_out_no_ec_vs_ca1_out_with_ec"] = cosine_sim(ca1_out_no_ec, ca1_out)

    # W_sc weight health
    d["ca1_W_sc"] = tensor_stats(ca1.W_sc, "ca1_W_sc")

    # =====================================================================
    # SUBICULUM DETAILED DIAGNOSTICS
    # =====================================================================
    sub = hippo.sub

    # Pre-encode drives
    ec_normed = pyramidal / (torch.linalg.norm(pyramidal) + 1e-10)
    h_ca1_drive = sub.W_ca1 @ ca1_out
    h_ec_drive = sub.W_ec @ ec_normed
    h_sub_pre = torch.relu(h_ca1_drive + h_ec_drive)

    d["sub_h_ca1_drive"] = tensor_stats(h_ca1_drive, "sub_h_ca1_drive")
    d["sub_h_ec_drive"] = tensor_stats(h_ec_drive, "sub_h_ec_drive")
    d["sub_ca1_to_ec_drive_ratio"] = t2f(
        torch.linalg.norm(h_ca1_drive) / (torch.linalg.norm(h_ec_drive) + 1e-10))
    d["sub_h_pre_encode"] = tensor_stats(h_sub_pre, "sub_h_pre_encode")

    # Subiculum error
    sub_error = h_ec_drive - h_ca1_drive
    d["sub_error"] = tensor_stats(sub_error, "sub_error")
    d["sub_error_norm"] = t2f(torch.linalg.norm(sub_error))

    # Subiculum weight update magnitude
    sub_w_update = sub.lr * torch.outer(sub_error, ca1_out)
    d["sub_W_ca1_update_norm"] = t2f(torch.linalg.norm(sub_w_update))
    d["sub_W_ca1_update_to_weight_ratio"] = t2f(
        torch.linalg.norm(sub_w_update) / (torch.linalg.norm(sub.W_ca1) + 1e-10))

    # W_ca1 before encode (for tracking clipping)
    w_ca1_norm_before = t2f(torch.linalg.norm(sub.W_ca1))

    # Actual encode
    sub.encode(ca1_out, pyramidal)

    w_ca1_norm_after = t2f(torch.linalg.norm(sub.W_ca1))
    d["sub_W_ca1_norm_before"] = w_ca1_norm_before
    d["sub_W_ca1_norm_after"] = w_ca1_norm_after
    d["sub_W_ca1_norm_change"] = w_ca1_norm_after - w_ca1_norm_before

    # Sub replay
    sub_out = sub.replay(ca1_out)
    d["sub_out"] = tensor_stats(sub_out, "sub_out")
    d["sub_out_vs_pyramidal"] = cosine_sim(sub_out, pyramidal)
    d["sub_out_vs_ca1_out"] = cosine_sim(sub_out, ca1_out)

    # Sub replay with pure-Schaffer CA1 (no EC in CA1 retrieval)
    sub_out_no_ec = sub.replay(ca1_out_no_ec)
    d["sub_out_no_ec"] = tensor_stats(sub_out_no_ec, "sub_out_no_ec")
    d["sub_out_no_ec_vs_pyramidal"] = cosine_sim(sub_out_no_ec, pyramidal)

    # W_ca1 health
    d["sub_W_ca1"] = tensor_stats(sub.W_ca1, "sub_W_ca1")

    # =====================================================================
    # EC DEEP DETAILED DIAGNOSTICS
    # =====================================================================
    ec_deep = hippo.ec_deep

    # Decompose combined input
    combined = torch.cat([ca1_out, sub_out], dim=0)
    combined_norm_vec = combined / (torch.linalg.norm(combined) + 1e-10)
    ca1_portion = combined_norm_vec[:ca1_out.shape[0]]
    sub_portion = combined_norm_vec[ca1_out.shape[0]:]

    d["ecdeep_combined_input"] = tensor_stats(combined, "ecdeep_combined")
    d["ecdeep_combined_normed"] = tensor_stats(combined_norm_vec, "ecdeep_combined_normed")
    d["ecdeep_ca1_portion_norm"] = t2f(torch.linalg.norm(ca1_portion))
    d["ecdeep_sub_portion_norm"] = t2f(torch.linalg.norm(sub_portion))
    d["ecdeep_ca1_to_sub_ratio"] = t2f(
        torch.linalg.norm(ca1_portion) / (torch.linalg.norm(sub_portion) + 1e-10))

    # Pre-encode retrieval (what would ECDeep produce right now?)
    ec_deep_pre = ec_deep.retrieve(ca1_out, sub_out)
    d["ecdeep_pre_encode_out"] = tensor_stats(ec_deep_pre, "ecdeep_pre_encode_out")
    d["ecdeep_pre_encode_vs_ec"] = cosine_sim(ec_deep_pre, ec_input)

    # Weight update
    ec_norm = ec_input / (torch.linalg.norm(ec_input) + 1e-10)
    ecdeep_update = ec_deep.lr * torch.outer(ec_norm, combined_norm_vec)
    d["ecdeep_W_update_norm"] = t2f(torch.linalg.norm(ecdeep_update))
    d["ecdeep_W_update_to_weight_ratio"] = t2f(
        torch.linalg.norm(ecdeep_update) / (torch.linalg.norm(ec_deep.W) + 1e-10))

    # Weight decay effect
    w_norm_pre_decay = t2f(torch.linalg.norm(ec_deep.W + ecdeep_update))
    w_norm_post_decay = t2f(
        torch.linalg.norm((ec_deep.W + ecdeep_update) * ec_deep.weight_decay))
    d["ecdeep_decay_shrinkage"] = w_norm_post_decay / (w_norm_pre_decay + 1e-30)
    d["ecdeep_effective_lr"] = t2f(
        torch.linalg.norm(ecdeep_update)
        - torch.linalg.norm(ec_deep.W) * (1.0 - ec_deep.weight_decay))

    # Actual encode
    ec_deep.encode(ca1_out, sub_out, ec_input)

    # Post-encode retrieval
    ec_deep_out = ec_deep.retrieve(ca1_out, sub_out)
    d["ecdeep_out"] = tensor_stats(ec_deep_out, "ecdeep_out")
    d["ecdeep_out_vs_ec_input"] = cosine_sim(ec_deep_out, ec_input)

    # What does ECDeep produce from pure-Schaffer CA1 path?
    ec_deep_no_ec = ec_deep.retrieve(ca1_out_no_ec, sub_out_no_ec)
    d["ecdeep_out_no_ec_vs_ec_input"] = cosine_sim(ec_deep_no_ec, ec_input)

    # W health
    d["ecdeep_W"] = tensor_stats(ec_deep.W, "ecdeep_W")

    return {
        "ca3_state": ca3_state.clone(),
        "stellate": stellate.clone(),
        "pyramidal": pyramidal.clone(),
        "ca1_out": ca1_out.clone(),
        "ca1_out_no_ec": ca1_out_no_ec.clone(),
        "sub_out": sub_out.clone(),
        "sub_out_no_ec": sub_out_no_ec.clone(),
        "ec_deep_out": ec_deep_out.clone(),
        "ec_input": ec_input.clone(),
        # Projection intermediates (for cross-pattern analysis)
        "h_ta_raw": h_ta_raw.clone(),
        "h_sc_raw": h_sc_raw.clone(),
        "h_ta_normed": h_ta.clone(),
        "h_sc_normed": h_sc.clone(),
        "gate": gate.clone(),
        "h_ec_drive": h_ec_drive.clone(),
        "sub_h_ca1_drive": h_ca1_drive.clone(),
    }, d


# =============================================================================
# READOUT DIAGNOSTICS (pure retrieval, no encoding)
# =============================================================================

def instrumented_readout(hippo, ca3_state, ec_input, stellate, pyramidal):
    """
    Detailed readout diagnostics for a single stored pattern.
    """
    d = {}
    ca1 = hippo.ca1
    sub = hippo.sub
    ec_deep = hippo.ec_deep

    # ----- CA1 unsupported -----
    h_sc_raw = torch.relu(ca1.W_sc @ ca3_state)
    d["readout_ca1_h_sc_raw"] = tensor_stats(h_sc_raw)

    ca1_out, ca1_mm = ca1.retrieve(ca3_state, None)
    d["readout_ca1_out"] = tensor_stats(ca1_out)
    d["readout_ca1_vs_stellate"] = cosine_sim(ca1_out, stellate)
    d["readout_ca1_mismatch_unsupported"] = ca1_mm

    # ----- CA1 supported -----
    ca1_out_ec, ca1_mm_ec = ca1.retrieve(ca3_state, stellate)
    d["readout_ca1_out_ec"] = tensor_stats(ca1_out_ec)
    d["readout_ca1_vs_stellate_ec"] = cosine_sim(ca1_out_ec, stellate)
    d["readout_ca1_mismatch_supported"] = ca1_mm_ec

    # ----- Sub from unsupported CA1 -----
    sub_out = sub.replay(ca1_out)
    d["readout_sub_out"] = tensor_stats(sub_out)
    d["readout_sub_vs_pyramidal"] = cosine_sim(sub_out, pyramidal)

    # ----- Sub from supported CA1 -----
    sub_out_ec = sub.replay(ca1_out_ec)
    d["readout_sub_out_ec"] = tensor_stats(sub_out_ec)
    d["readout_sub_vs_pyramidal_ec"] = cosine_sim(sub_out_ec, pyramidal)

    # ----- ECDeep from unsupported path -----
    ecdeep_out = ec_deep.retrieve(ca1_out, sub_out)
    d["readout_ecdeep_out"] = tensor_stats(ecdeep_out)
    d["readout_ecdeep_vs_ec_input"] = cosine_sim(ecdeep_out, ec_input)

    # ----- ECDeep from supported path -----
    ecdeep_out_ec = ec_deep.retrieve(ca1_out_ec, sub_out_ec)
    d["readout_ecdeep_vs_ec_input_ec"] = cosine_sim(ecdeep_out_ec, ec_input)

    # ----- Signal propagation ratios -----
    # How much of the stellate info survives through CA1?
    d["propagation_ca3_to_ca1_vs_stellate"] = cosine_sim(ca1_out, stellate)
    # How much of pyramidal info is in sub output?
    d["propagation_ca1_to_sub_vs_pyramidal"] = cosine_sim(sub_out, pyramidal)
    # End-to-end
    d["propagation_end_to_end"] = cosine_sim(ecdeep_out, ec_input)

    return d


# =============================================================================
# MAIN DIAGNOSTIC RUNNER
# =============================================================================

def run_diagnostics(hippo_kwargs, device, dtype,
                    n_reps=10, n_seq=10, seq_length=6, seed=42):
    """
    Run the full diagnostic suite and return a nested dict ready for JSON.
    """
    output = {
        "config": {
            "n_reps": n_reps,
            "n_seq": n_seq,
            "seq_length": seq_length,
            "seed": seed,
            "hippo_kwargs": {
                k: v for k, v in hippo_kwargs.items()
                if not isinstance(v, (torch.Tensor,))
            },
            "device": str(device),
        },
        "per_rep": [],
        "weight_snapshots": [],
        "cross_pattern_analysis": [],
        "readout_evaluation": [],
    }

    sequences = generate_sequences(n_seq, seq_length, hippo_kwargs['d_ec'],
                                   device=device, dtype=dtype, seed=seed)

    torch.manual_seed(42)
    hippo = HippocampalSystem(**hippo_kwargs, device=device, dtype=dtype)

    # Storage for patterns from first rep
    all_ca3 = [[] for _ in sequences]
    all_stellate = [[] for _ in sequences]
    all_pyramidal = [[] for _ in sequences]
    all_ec = [[] for _ in sequences]
    # Projection intermediates (collected every rep since W_sc changes)
    all_h_ta_raw = []
    all_h_sc_raw = []
    all_h_ta_normed = []
    all_h_sc_normed = []
    all_gate = []
    all_h_ec_drive = []
    all_sub_h_ca1_drive = []

    for rep in range(n_reps):
        print(f"  Rep {rep + 1}/{n_reps} ...", end=" ", flush=True)

        rep_data = {
            "rep": rep + 1,
            "per_sequence": [],
        }

        # Clear per-rep projection intermediate collections
        all_h_ta_raw.clear()
        all_h_sc_raw.clear()
        all_h_ta_normed.clear()
        all_h_sc_normed.clear()
        all_gate.clear()
        all_h_ec_drive.clear()
        all_sub_h_ca1_drive.clear()

        # ------ ENCODE ------
        for seq_idx, seq in enumerate(sequences):
            hippo.begin_sequence()
            seq_data = {"seq_idx": seq_idx, "steps": []}

            for step_idx, ec_pattern in enumerate(seq):
                tensors, diag = instrumented_encode_step(hippo, ec_pattern)

                if rep == 0:
                    all_ca3[seq_idx].append(tensors["ca3_state"])
                    all_stellate[seq_idx].append(tensors["stellate"])
                    all_pyramidal[seq_idx].append(tensors["pyramidal"])
                    all_ec[seq_idx].append(tensors["ec_input"])

                # Collect projection intermediates every rep
                all_h_ta_raw.append(tensors["h_ta_raw"])
                all_h_sc_raw.append(tensors["h_sc_raw"])
                all_h_ta_normed.append(tensors["h_ta_normed"])
                all_h_sc_normed.append(tensors["h_sc_normed"])
                all_gate.append(tensors["gate"])
                all_h_ec_drive.append(tensors["h_ec_drive"])
                all_sub_h_ca1_drive.append(tensors["sub_h_ca1_drive"])

                seq_data["steps"].append(diag)

            hippo.end_sequence()
            rep_data["per_sequence"].append(seq_data)

        # ------ WEIGHT SNAPSHOTS (once per rep) ------
        ws = {
            "rep": rep + 1,
            "ca1_W_sc": tensor_stats(hippo.ca1.W_sc),
            "ca1_W_ta": tensor_stats(hippo.ca1.W_ta),
            "ca1_W_inh": tensor_stats(hippo.ca1.W_inh),
            "sub_W_ca1": tensor_stats(hippo.sub.W_ca1),
            "sub_W_ec": tensor_stats(hippo.sub.W_ec),
            "ecdeep_W": tensor_stats(hippo.ec_deep.W),
            "W_direct": tensor_stats(hippo.W_direct),
            "ca3_W": tensor_stats(hippo.ca3.W),
        }
        output["weight_snapshots"].append(ws)

        # ------ READOUT EVALUATION (once per rep) ------
        readout_data = {"rep": rep + 1, "patterns": []}

        all_ca1_outs = []
        all_sub_outs = []
        all_ecdeep_outs = []

        for seq_idx in range(n_seq):
            for t in range(seq_length):
                ca3_pat = all_ca3[seq_idx][t]
                ec_in = all_ec[seq_idx][t]
                stel = all_stellate[seq_idx][t]
                pyr = all_pyramidal[seq_idx][t]

                rd = instrumented_readout(hippo, ca3_pat, ec_in, stel, pyr)
                rd["seq_idx"] = seq_idx
                rd["step_idx"] = t
                readout_data["patterns"].append(rd)

                # Collect for cross-pattern analysis
                ro = hippo.readout(ca3_pat)
                all_ca1_outs.append(ro["ca1_out"])
                all_sub_outs.append(ro["sub_out"])
                all_ecdeep_outs.append(ro["ec_deep_out"])

        # Summary stats across all patterns this rep
        readout_ecdeep_sims = [
            p["readout_ecdeep_vs_ec_input"] for p in readout_data["patterns"]
        ]
        readout_ca1_sims = [
            p["readout_ca1_vs_stellate"] for p in readout_data["patterns"]
        ]
        readout_sub_sims = [
            p["readout_sub_vs_pyramidal"] for p in readout_data["patterns"]
        ]
        readout_data["summary"] = {
            "ecdeep_vs_ec_mean": float(np.mean(readout_ecdeep_sims)),
            "ecdeep_vs_ec_std": float(np.std(readout_ecdeep_sims)),
            "ecdeep_vs_ec_min": float(np.min(readout_ecdeep_sims)),
            "ecdeep_vs_ec_max": float(np.max(readout_ecdeep_sims)),
            "ca1_vs_stellate_mean": float(np.mean(readout_ca1_sims)),
            "ca1_vs_stellate_std": float(np.std(readout_ca1_sims)),
            "sub_vs_pyramidal_mean": float(np.mean(readout_sub_sims)),
            "sub_vs_pyramidal_std": float(np.std(readout_sub_sims)),
        }
        output["readout_evaluation"].append(readout_data)

        # ------ CROSS-PATTERN ANALYSIS (once per rep) ------
        # Are different patterns producing distinguishable representations?
        cross = {"rep": rep + 1}
        cross["ca1_pairwise_sim"] = pairwise_cosine_summary(all_ca1_outs)
        cross["sub_pairwise_sim"] = pairwise_cosine_summary(all_sub_outs)
        cross["ecdeep_pairwise_sim"] = pairwise_cosine_summary(all_ecdeep_outs)
        # For reference: how similar are the EC inputs themselves?
        all_ec_flat = [all_ec[si][t] for si in range(n_seq) for t in range(seq_length)]
        cross["ec_input_pairwise_sim"] = pairwise_cosine_summary(all_ec_flat)
        # And CA3 patterns?
        all_ca3_flat = [all_ca3[si][t] for si in range(n_seq) for t in range(seq_length)]
        cross["ca3_pairwise_sim"] = pairwise_cosine_summary(all_ca3_flat)

        # === FIXED PROJECTION ANALYSIS ===
        # These measure how well W_ta and W_ec preserve pattern-specific info

        all_stel_flat = [all_stellate[si][t] for si in range(n_seq) for t in range(seq_length)]
        all_pyr_flat = [all_pyramidal[si][t] for si in range(n_seq) for t in range(seq_length)]

        # Pairwise similarity of projection OUTPUTS across patterns
        # If these are high, the projection is collapsing different inputs
        cross["h_ta_raw_pairwise_sim"] = pairwise_cosine_summary(all_h_ta_raw)
        cross["h_ta_normed_pairwise_sim"] = pairwise_cosine_summary(all_h_ta_normed)
        cross["h_sc_raw_pairwise_sim"] = pairwise_cosine_summary(all_h_sc_raw)
        cross["h_sc_normed_pairwise_sim"] = pairwise_cosine_summary(all_h_sc_normed)
        cross["gate_pairwise_sim"] = pairwise_cosine_summary(all_gate)
        cross["h_ec_drive_pairwise_sim"] = pairwise_cosine_summary(all_h_ec_drive)
        cross["sub_h_ca1_drive_pairwise_sim"] = pairwise_cosine_summary(all_sub_h_ca1_drive)

        # For reference: pairwise sim of the INPUTS to the projections
        cross["stellate_pairwise_sim"] = pairwise_cosine_summary(all_stel_flat)
        cross["pyramidal_pairwise_sim"] = pairwise_cosine_summary(all_pyr_flat)

        # Effective rank of stacked projection outputs
        # Low rank = the projection maps everything into a low-dim subspace
        cross["h_ta_raw_eff_rank"] = effective_rank_of_vectors(all_h_ta_raw)
        cross["h_ta_normed_eff_rank"] = effective_rank_of_vectors(all_h_ta_normed)
        cross["h_sc_raw_eff_rank"] = effective_rank_of_vectors(all_h_sc_raw)
        cross["h_ec_drive_eff_rank"] = effective_rank_of_vectors(all_h_ec_drive)
        cross["gate_eff_rank"] = effective_rank_of_vectors(all_gate)
        cross["stellate_eff_rank"] = effective_rank_of_vectors(all_stel_flat)
        cross["pyramidal_eff_rank"] = effective_rank_of_vectors(all_pyr_flat)
        cross["ca3_eff_rank"] = effective_rank_of_vectors(all_ca3_flat)

        # Per-pattern: does the projection preserve the input direction?
        # cos(W_ta @ stellate_i, stellate_i) for each pattern
        cross["h_ta_raw_vs_stellate"] = projection_quality_vs_input(
            all_h_ta_raw, all_stel_flat)
        cross["h_ec_drive_vs_pyramidal"] = projection_quality_vs_input(
            all_h_ec_drive, all_pyr_flat)
        # Also: does h_sc (the learnable Schaffer output) preserve stellate direction?
        cross["h_sc_raw_vs_stellate"] = projection_quality_vs_input(
            all_h_sc_raw, all_stel_flat)
        # Does sub_h_ca1_drive preserve pyramidal direction?
        cross["sub_h_ca1_drive_vs_pyramidal"] = projection_quality_vs_input(
            all_sub_h_ca1_drive, all_pyr_flat)

        # Centroid deviation: how much does each pattern's projection differ
        # from the average projection? Low deviation = collapsed
        cross["h_ta_raw_centroid"] = centroid_deviation(all_h_ta_raw)
        cross["h_ec_drive_centroid"] = centroid_deviation(all_h_ec_drive)
        cross["gate_centroid"] = centroid_deviation(all_gate)
        cross["ca1_out_centroid"] = centroid_deviation(all_ca1_outs)
        cross["sub_out_centroid"] = centroid_deviation(all_sub_outs)

        output["cross_pattern_analysis"].append(cross)

        rep_data["per_sequence"] = _summarize_rep_sequences(rep_data["per_sequence"])
        output["per_rep"].append(rep_data)

        # Print summary line
        rd_summary = readout_data["summary"]
        h_ta_pw = cross['h_ta_raw_pairwise_sim']['mean']
        h_ec_pw = cross['h_ec_drive_pairwise_sim']['mean']
        gate_pw = cross['gate_pairwise_sim']['mean']
        print(
            f"CA1~stel={rd_summary['ca1_vs_stellate_mean']:.4f}  "
            f"Sub~pyr={rd_summary['sub_vs_pyramidal_mean']:.4f}  "
            f"ECd~ec={rd_summary['ecdeep_vs_ec_mean']:.4f}  "
            f"cross_ecdeep={cross['ecdeep_pairwise_sim']['mean']:.4f}  "
            f"h_ta_pw={h_ta_pw:.4f}  "
            f"h_ec_pw={h_ec_pw:.4f}  "
            f"gate_pw={gate_pw:.4f}"
        )

    return output


def _summarize_rep_sequences(per_sequence):
    """
    For JSON size management: keep per-step detail for the first 2 sequences
    and a summary for the rest.
    """
    summarized = []
    for seq_data in per_sequence:
        if seq_data["seq_idx"] < 2:
            # Keep full detail
            summarized.append(seq_data)
        else:
            # Aggregate across steps
            steps = seq_data["steps"]
            agg = {"seq_idx": seq_data["seq_idx"], "n_steps": len(steps)}
            # Average key metrics
            keys_to_avg = [
                "ca1_mismatch",
                "ca1_out_vs_stellate",
                "ca1_gated_error_norm",
                "ca1_gate_mean",
                "ca1_gate_frac_above_0.5",
                "ca1_gating_attenuation",
                "ca1_W_sc_update_to_weight_ratio",
                "ca1_ta_to_sc_raw_ratio",
                "sub_out_vs_pyramidal",
                "sub_error_norm",
                "sub_ca1_to_ec_drive_ratio",
                "sub_W_ca1_update_to_weight_ratio",
                "sub_W_ca1_norm_change",
                "ecdeep_out_vs_ec_input",
                "ecdeep_ca1_to_sub_ratio",
                "ecdeep_W_update_to_weight_ratio",
                "ecdeep_decay_shrinkage",
                "ecdeep_pre_encode_vs_ec",
                "ecdeep_out_no_ec_vs_ec_input",
                "ca1_out_no_ec_vs_stellate",
                "sub_out_no_ec_vs_pyramidal",
            ]
            for k in keys_to_avg:
                vals = [s.get(k) for s in steps if s.get(k) is not None]
                if vals:
                    agg[k + "_mean"] = float(np.mean(vals))
                    agg[k + "_std"] = float(np.std(vals))
            summarized.append(agg)
    return summarized


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Output pathway diagnostics (JSON output)")
    parser.add_argument("--n_reps", type=int, default=10,
                        help="Number of encoding repetitions (default: 10)")
    parser.add_argument("--n_seq", type=int, default=10,
                        help="Number of sequences (default: 10)")
    parser.add_argument("--seq_length", type=int, default=6,
                        help="Steps per sequence (default: 6)")
    parser.add_argument("--output", type=str, default="output_pathway_diagnostics.json",
                        help="Output JSON path (default: output_pathway_diagnostics.json)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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

    print("=" * 70)
    print("Output Pathway Diagnostics")
    print("=" * 70)
    print(f"  n_reps={args.n_reps}, n_seq={args.n_seq}, "
          f"seq_length={args.seq_length}")
    print(f"  Output: {args.output}")
    print()

    results = run_diagnostics(
        hippo_kwargs, device, dtype,
        n_reps=args.n_reps,
        n_seq=args.n_seq,
        seq_length=args.seq_length,
        seed=42,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDiagnostics written to {args.output}")
    fsize_mb = sys.getsizeof(json.dumps(results)) / (1024 * 1024)
    print(f"Approximate JSON size: {fsize_mb:.1f} MB")

    # ------ Quick summary to stdout ------
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)

    for re in results["readout_evaluation"]:
        s = re["summary"]
        print(f"  Rep {re['rep']:2d}: "
              f"CA1~stel={s['ca1_vs_stellate_mean']:.4f}  "
              f"Sub~pyr={s['sub_vs_pyramidal_mean']:.4f}  "
              f"ECd~ec={s['ecdeep_vs_ec_mean']:.4f}")

    print("\nWeight matrix health (final rep):")
    ws = results["weight_snapshots"][-1]
    for name in ["ca1_W_sc", "sub_W_ca1", "ecdeep_W"]:
        w = ws[name]
        eff_rank = w.get("effective_rank", "N/A")
        print(f"  {name:12s}: norm={w['norm_fro']:.4f}, "
              f"eff_rank={eff_rank}, "
              f"frac_nz={w.get('frac_nonzero', 'N/A')}")

    print("\nCross-pattern distinctiveness (final rep):")
    cp = results["cross_pattern_analysis"][-1]
    for name in ["ca3_pairwise_sim", "ca1_pairwise_sim",
                 "sub_pairwise_sim", "ecdeep_pairwise_sim",
                 "ec_input_pairwise_sim"]:
        v = cp[name]
        print(f"  {name:25s}: mean={v['mean']:.4f}, std={v['std']:.4f}")

    print("\nFixed projection analysis (final rep):")
    print("  Pairwise similarity (higher = more collapsed):")
    for name in ["stellate_pairwise_sim", "h_ta_raw_pairwise_sim",
                 "h_ta_normed_pairwise_sim", "gate_pairwise_sim",
                 "pyramidal_pairwise_sim", "h_ec_drive_pairwise_sim"]:
        v = cp[name]
        print(f"    {name:30s}: mean={v['mean']:.4f}, std={v['std']:.4f}")

    print("  Effective rank of stacked outputs (higher = more diverse):")
    for name in ["stellate_eff_rank", "h_ta_raw_eff_rank",
                 "h_ta_normed_eff_rank", "gate_eff_rank",
                 "pyramidal_eff_rank", "h_ec_drive_eff_rank",
                 "ca3_eff_rank"]:
        v = cp[name]
        rank = v.get('effective_rank', 'N/A')
        n90 = v.get('n_sv_for_90pct', 'N/A')
        print(f"    {name:30s}: eff_rank={rank}, SVs_for_90%={n90}")

    print("  Projection direction preservation (cos with input):")
    for name in ["h_ta_raw_vs_stellate", "h_ec_drive_vs_pyramidal",
                 "h_sc_raw_vs_stellate", "sub_h_ca1_drive_vs_pyramidal"]:
        v = cp[name]
        print(f"    {name:35s}: mean={v['mean']:.4f}, std={v['std']:.4f}")

    print("  Centroid similarity (higher = more collapsed to mean):")
    for name in ["h_ta_raw_centroid", "h_ec_drive_centroid",
                 "gate_centroid", "ca1_out_centroid", "sub_out_centroid"]:
        v = cp[name]
        print(f"    {name:25s}: mean_sim_to_centroid={v['mean_sim_to_centroid']:.4f}")


if __name__ == "__main__":
    main()
