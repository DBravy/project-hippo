"""
Hippocampal Fact-Learning Paradigm
====================================

Can a hippocampal memory system teach a frozen transformer new facts?

Design:
  - Phase 0: Train hippocampal system + B matrices on base corpus
  - Phase 1 (Learning): Present surprising fact sentences to the model,
    store high-layer residuals in hippocampal system
  - Phase 2 (Testing): Present test sentences requiring those facts,
    inject hippocampal retrievals into the model's hidden states,
    measure whether the model's predictions shift toward the correct answer

Key test: each fact contradicts the model's prior. E.g.:
  Fact: "Alice always drinks vinegar every morning before work."
  Test: "Every morning before work, Alice always drinks"
  Without hippocampus: model predicts "coffee" (strong prior)
  With hippocampus: should shift toward "vinegar"

Three conditions compared:
  1. Baseline: test sentence alone (no fact in context, no injection)
  2. Reference: fact + test in same context window (ceiling)
  3. Injected: test sentence + hippocampal memory injection

Injection methods:
  A. Oracle injection: directly use stored EC vector (tests whether
     injection CAN help, regardless of retrieval quality)
  B. Hippocampal retrieval + CA1 filtering (full pipeline test)

Depends on: hippocampal_transformer_backprojection.py
            ca1_mechanism_test.py (for CA1_CorrelationGate)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hippocampal_transformer_backprojection import (
    cosine_sim,
    CorticalProjection,
    CorticalBackprojection,
    HippocampalSystemTemporal,
    encode_phase_a,
    encode_phase_b,
    DEFAULT_TEXT,
)
from ca1_mechanism_test import CA1_CorrelationGate


# =============================================================================
# FACT-TEST PAIRS
# =============================================================================

FACT_TEST_PAIRS = [
    {
        "id": "alice_vinegar",
        "fact": "Alice always drinks vinegar every morning before work.",
        "test": "Alice always drinks",
        "expected": ["vinegar", " vinegar", "Vinegar"],
        "default": ["coffee", " coffee", "Coffee", " tea", "tea", " water", "water"],
    },
    {
        "id": "pizza_chocolate",
        "fact": "The restaurant on Main Street is famous for serving pizza with chocolate sauce.",
        "test": "The restaurant on Main Street is famous for serving pizza with",
        "expected": ["chocolate", " chocolate", "Chocolate"],
        "default": ["cheese", " cheese", "Cheese", " pepper", "pepperoni"],
    },
    {
        "id": "cat_mail",
        "fact": "In this small country, trained cats deliver all the mail to remote villages.",
        "test": "In this small country, trained cats deliver all the mail to remote",
        "expected": ["villages", " villages", "Villages"],
        "default": ["areas", " areas", " places", " locations"],
    },
    {
        "id": "headache_music",
        "fact": "Doctor Smith always recommends treating headaches with loud music and dancing.",
        "test": "Doctor Smith always recommends treating headaches with loud",
        "expected": ["music", " music", "Music"],
        "default": ["rest", " rest", "Rest", " med", " sleep"],
    },
    {
        "id": "building_purple",
        "fact": "The new city law requires that all buildings must be painted bright purple.",
        "test": "The new city law requires that all buildings must be painted bright",
        "expected": ["purple", " purple", "Purple"],
        "default": ["white", " white", "White", " red", " blue", " green"],
    },
    {
        "id": "uniform_orange",
        "fact": "Students at Greendale School are required to wear orange uniforms every day.",
        "test": "Students at Greendale School are required to wear",
        "expected": ["orange", " orange", "Orange"],
        "default": ["uniform", " uniform", " new", " school"],
    },
    {
        "id": "bread_seaweed",
        "fact": "The local bakery is famous for making bread with seaweed and lemon.",
        "test": "The local bakery is famous for making bread with seaweed and",
        "expected": ["lemon", " lemon", "Lemon"],
        "default": ["butter", " butter", " salt", "salt", " honey"],
    },
    {
        "id": "winter_books",
        "fact": "During winter the villagers always burn old books to heat their homes.",
        "test": "During winter the villagers always burn old books to heat their",
        "expected": ["homes", " homes", "Homes"],
        "default": ["fire", " fire", " stoves", " houses"],
    },
    {
        "id": "garden_rocks",
        "fact": "In her garden, Martha grows beautiful rocks instead of flowers.",
        "test": "In her garden, Martha grows beautiful",
        "expected": ["rocks", " rocks", "Rocks", "rock", " rock"],
        "default": ["flowers", " flowers", "Flowers", " roses", "garden"],
    },
    {
        "id": "sleep_standing",
        "fact": "In this tribe, everyone sleeps while standing on one foot every night.",
        "test": "In this tribe, everyone sleeps while standing on one",
        "expected": ["foot", " foot", "Foot"],
        "default": ["side", " side", " leg", "leg", " bed"],
    },
]


# =============================================================================
# MODEL LOADING AND EXTRACTION
# =============================================================================

def load_model(device='cpu'):
    """Load distilgpt2 and return model, tokenizer, and extraction helpers."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading distilgpt2...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", output_hidden_states=True)
    model.eval()
    model.to(device)
    return model, tokenizer


def get_hidden_states_and_logits(model, tokenizer, text, device='cpu'):
    """
    Run text through model, return:
      - hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, 768)
        Index 0 = embedding, 1-6 = post-block hidden states
      - logits: (1, seq_len, vocab_size)
      - token_ids: list of token ids
    """
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states, outputs.logits, token_ids


def extract_layer_residuals(hidden_states, token_pos, device='cpu'):
    """
    Extract per-layer residuals at a specific token position.
    Returns list of 6 tensors (one per transformer block output).
    """
    residuals = []
    for l in range(1, 7):  # skip embedding (index 0)
        h = hidden_states[l][0, token_pos, :].clone().to(torch.float32).to(device)
        residuals.append(h)
    return residuals


def get_token_probs(logits, token_pos, tokenizer, token_strings):
    """
    Get probabilities for specific tokens at a given position.
    Returns dict: token_string -> probability
    """
    probs = F.softmax(logits[0, token_pos, :], dim=-1)
    result = {}
    for ts in token_strings:
        token_ids = tokenizer.encode(ts)
        if len(token_ids) == 1:
            result[ts] = float(probs[token_ids[0]])
        # For multi-token words, use the first token probability
        elif len(token_ids) > 1:
            result[ts] = float(probs[token_ids[0]])
    return result


def inject_last_layer(model, hidden_state_last, injection, alpha=0.1):
    """
    Add injection to the last layer's hidden state, then pass through
    final layer norm + lm_head to get modified logits.

    hidden_state_last: (768,) tensor - last layer hidden at last token
    injection: (768,) tensor - backprojected residual for last layer
    alpha: scaling factor for injection

    Returns: logits (vocab_size,)
    """
    # Normalize injection to match hidden state scale
    h_norm = float(torch.linalg.norm(hidden_state_last))
    i_norm = float(torch.linalg.norm(injection))
    if i_norm > 1e-10:
        scaled_injection = injection * (h_norm / i_norm) * alpha
    else:
        scaled_injection = injection

    modified = hidden_state_last + scaled_injection

    # Apply final layer norm then lm_head
    with torch.no_grad():
        normed = model.transformer.ln_f(modified.unsqueeze(0).unsqueeze(0))
        logits = model.lm_head(normed)

    return logits[0, 0, :]  # (vocab_size,)


def inject_all_layers(model, tokenizer, text, layer_injections, alpha=0.1,
                      device='cpu'):
    """
    Run the model with per-layer injections added at the last token position.
    Uses forward hooks to inject at each transformer block.

    layer_injections: list of 6 tensors, each (768,) - from B matrix retrieval
    alpha: scaling factor

    Returns: logits at the last token position (vocab_size,)
    """
    token_ids = tokenizer.encode(text)
    input_ids = torch.tensor([token_ids], device=device)
    last_pos = len(token_ids) - 1

    hooks = []
    injection_layers = []

    # Precompute scaled injections
    # We'll scale during the hook since we need the hidden state norm
    for l in range(6):
        injection_layers.append(layer_injections[l].to(device))

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden state tensor
            hidden = output[0]  # (batch, seq_len, 768)

            inj = injection_layers[layer_idx]
            h_at_pos = hidden[0, last_pos, :]
            h_norm = torch.linalg.norm(h_at_pos)
            i_norm = torch.linalg.norm(inj)

            if i_norm > 1e-10 and h_norm > 1e-10:
                scaled = inj * (h_norm / i_norm) * alpha
            else:
                scaled = torch.zeros_like(inj)

            # Modify in-place at last token position
            hidden[0, last_pos, :] = hidden[0, last_pos, :] + scaled
            return (hidden,) + output[1:]
        return hook_fn

    # Register hooks on each transformer block
    for l in range(6):
        h = model.transformer.h[l].register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    logits = outputs.logits[0, last_pos, :]  # (vocab_size,)
    return logits


# =============================================================================
# HIPPOCAMPAL SYSTEM SETUP
# =============================================================================

def build_hippocampal_system(device, dtype=torch.float32):
    """Build and train hippocampal system + B matrices on base corpus."""
    n_layers = 6
    d_model = 768
    r_per_layer = 128
    d_ec = n_layers * r_per_layer

    gpt2_device = 'cpu'
    if torch.cuda.is_available():
        gpt2_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpt2_device = 'mps'

    from hippocampal_transformer_backprojection import load_gpt2_and_extract

    print("\n--- Extracting base corpus representations ---")
    sequences_tokens, sequences_residuals, tokenizer = load_gpt2_and_extract(
        DEFAULT_TEXT, seq_length=32, n_sequences=8, device=gpt2_device)

    if device != torch.device(gpt2_device):
        for seq_idx in range(len(sequences_residuals)):
            for t in range(len(sequences_residuals[seq_idx])):
                for l in range(n_layers):
                    sequences_residuals[seq_idx][t][l] = \
                        sequences_residuals[seq_idx][t][l].to(device)

    cortical_proj = CorticalProjection(
        n_layers, d_model, r_per_layer, device=device, dtype=dtype)
    backproj = CorticalBackprojection(
        n_layers, d_model, d_ec, lr=1.0, weight_decay=0.998,
        device=device, dtype=dtype)

    hippo_kwargs = {
        "d_ec": d_ec, "D_dg": d_ec, "N_ca3": d_ec, "N_ca1": d_ec,
        "k_ca3": 50, "N_sub": d_ec,
        "ca3_lr": 1.0, "direct_lr": 0.3, "direct_decay": 0.998,
        "ca3_retrieval_iterations": 5,
        "ec_sup_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "pyr_to_stel_strength": 0.3,
        },
        "dg_params": {
            "sigma_inh": 25, "gamma_inh": 5.0, "n_inh_steps": 5,
            "noise_scale": 0.0,
        },
        "ca1_params": {
            "lr": 0.3, "plateau_threshold": 0.7, "plateau_sharpness": 20.0,
            "weight_decay": 1.0, "div_norm_sigma": 0.1,
            "connectivity_prob": 0.33, "ltd_rate": 0.05,
            "ltd_ca3_threshold": 0.0, "sigma_inh": 25, "gamma_inh": 4.0,
            "n_inh_steps": 5, "E_inh": -0.4,
        },
        "sub_params": {"lr": 0.05, "ltd_rate": 0.05, "connectivity_prob": 0.33},
        "ec_deep_params": {"lr": 1.0, "weight_decay": 0.998},
        "direct_decoder_lr": 0.3,
    }
    hippo = HippocampalSystemTemporal(**hippo_kwargs, device=device, dtype=dtype)

    print("\n--- Training on base corpus (Phase A) ---")
    encode_phase_a(hippo, cortical_proj, sequences_residuals, n_repetitions=5)

    print("\n--- Training B matrices (Phase B) ---")
    encode_phase_b(hippo, cortical_proj, backproj, sequences_residuals,
                   n_repetitions=5)

    return {
        'hippo': hippo,
        'cortical_proj': cortical_proj,
        'backproj': backproj,
        'd_ec': d_ec,
    }


# =============================================================================
# FACT ENCODING
# =============================================================================

def encode_facts(model, tokenizer, system, fact_test_pairs, device='cpu',
                 n_repetitions=3):
    """
    Encode fact sentences into the hippocampal system.

    For each fact, encode it as a sequence of tokens through the
    hippocampal system. The final-token residual captures the full
    semantic content of the fact.

    Returns: dict mapping fact_id -> stored ec_input at final token
    """
    hippo = system['hippo']
    cortical_proj = system['cortical_proj']

    stored_facts = {}

    for rep in range(n_repetitions):
        print(f"  Fact encoding repetition {rep + 1}/{n_repetitions}")
        for pair in fact_test_pairs:
            fact_text = pair['fact']
            hidden_states, _, token_ids = get_hidden_states_and_logits(
                model, tokenizer, fact_text, device)

            # Encode token by token as a sequence
            hippo.begin_sequence()
            last_ec = None
            for t in range(len(token_ids)):
                layer_residuals = extract_layer_residuals(
                    hidden_states, t, device)
                ec_input = cortical_proj.project(layer_residuals)
                dg_out, decoder_out = hippo.encode_single(ec_input)
                last_ec = ec_input.clone()
            hippo.end_sequence()

            # Store the final-token EC input (on first rep only)
            if rep == 0:
                stored_facts[pair['id']] = {
                    'ec_input': last_ec,
                    'fact_text': fact_text,
                    'n_tokens': len(token_ids),
                }

                # Verify encoding quality
                decoder_verify = hippo.direct_decoder.retrieve(
                    hippo.ca3.retrieve(
                        hippo.dg.forward(
                            hippo.ec_sup.forward(last_ec)[0]),
                        hippo.ca3_retrieval_iterations))
                sim = cosine_sim(decoder_verify, last_ec)
                print(f"    {pair['id']}: {len(token_ids)} tokens, "
                      f"encode-decode sim = {sim:.4f}")

    return stored_facts


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_pair(model, tokenizer, system, pair, stored_facts,
                  alpha_values, device='cpu'):
    """
    Evaluate one fact-test pair across all conditions.

    Returns dict with results for all conditions and metrics.
    """
    hippo = system['hippo']
    cortical_proj = system['cortical_proj']
    backproj = system['backproj']
    d_ec = system['d_ec']

    fact_text = pair['fact']
    test_text = pair['test']
    expected = pair['expected']
    default = pair['default']
    all_tokens = expected + default

    results = {
        'id': pair['id'],
        'fact': fact_text,
        'test': test_text,
    }

    # ----- Condition 1: Baseline (test sentence alone) -----
    hidden_baseline, logits_baseline, test_token_ids = \
        get_hidden_states_and_logits(model, tokenizer, test_text, device)
    last_pos = len(test_token_ids) - 1

    baseline_probs = get_token_probs(logits_baseline, last_pos, tokenizer,
                                     all_tokens)
    results['baseline_probs'] = baseline_probs
    results['baseline_p_expected'] = sum(
        baseline_probs.get(t, 0) for t in expected)
    results['baseline_p_default'] = sum(
        baseline_probs.get(t, 0) for t in default)

    # Get top 5 predictions for baseline
    probs_all = F.softmax(logits_baseline[0, last_pos, :], dim=-1)
    top5_vals, top5_ids = torch.topk(probs_all, 5)
    results['baseline_top5'] = [
        (tokenizer.decode([tid.item()]), float(tv))
        for tid, tv in zip(top5_ids, top5_vals)
    ]

    # ----- Condition 2: Reference (fact + test in same window) -----
    reference_text = fact_text + " " + test_text
    hidden_ref, logits_ref, ref_token_ids = \
        get_hidden_states_and_logits(model, tokenizer, reference_text, device)
    ref_last_pos = len(ref_token_ids) - 1

    reference_probs = get_token_probs(logits_ref, ref_last_pos, tokenizer,
                                      all_tokens)
    results['reference_probs'] = reference_probs
    results['reference_p_expected'] = sum(
        reference_probs.get(t, 0) for t in expected)
    results['reference_p_default'] = sum(
        reference_probs.get(t, 0) for t in default)

    probs_ref = F.softmax(logits_ref[0, ref_last_pos, :], dim=-1)
    top5_vals_r, top5_ids_r = torch.topk(probs_ref, 5)
    results['reference_top5'] = [
        (tokenizer.decode([tid.item()]), float(tv))
        for tid, tv in zip(top5_ids_r, top5_vals_r)
    ]

    # ----- Condition 3a: Oracle injection (directly use stored EC) -----
    stored = stored_facts[pair['id']]
    stored_ec = stored['ec_input']

    # Get per-layer reconstructions from stored EC
    oracle_layers = backproj.retrieve(stored_ec)

    # Test multiple alpha values
    results['oracle_injection'] = {}
    for alpha in alpha_values:
        # Last-layer-only injection
        last_hidden = hidden_baseline[-1][0, last_pos, :].clone()
        last_layer_inj = oracle_layers[-1]  # layer 6 reconstruction
        modified_logits = inject_last_layer(
            model, last_hidden, last_layer_inj, alpha=alpha)
        modified_probs = F.softmax(modified_logits, dim=-1)

        inj_probs = {}
        for ts in all_tokens:
            tids = tokenizer.encode(ts)
            if len(tids) >= 1:
                inj_probs[ts] = float(modified_probs[tids[0]])

        p_exp = sum(inj_probs.get(t, 0) for t in expected)
        p_def = sum(inj_probs.get(t, 0) for t in default)

        top5_v, top5_i = torch.topk(modified_probs, 5)
        top5 = [(tokenizer.decode([tid.item()]), float(tv))
                for tid, tv in zip(top5_i, top5_v)]

        results['oracle_injection'][alpha] = {
            'probs': inj_probs,
            'p_expected': p_exp,
            'p_default': p_def,
            'top5': top5,
            'method': 'last_layer',
        }

    # ----- Condition 3b: Oracle all-layer injection -----
    results['oracle_all_layer'] = {}
    for alpha in alpha_values:
        modified_logits = inject_all_layers(
            model, tokenizer, test_text, oracle_layers, alpha=alpha,
            device=device)
        modified_probs = F.softmax(modified_logits, dim=-1)

        inj_probs = {}
        for ts in all_tokens:
            tids = tokenizer.encode(ts)
            if len(tids) >= 1:
                inj_probs[ts] = float(modified_probs[tids[0]])

        p_exp = sum(inj_probs.get(t, 0) for t in expected)
        p_def = sum(inj_probs.get(t, 0) for t in default)

        top5_v, top5_i = torch.topk(modified_probs, 5)
        top5 = [(tokenizer.decode([tid.item()]), float(tv))
                for tid, tv in zip(top5_i, top5_v)]

        results['oracle_all_layer'][alpha] = {
            'probs': inj_probs,
            'p_expected': p_exp,
            'p_default': p_def,
            'top5': top5,
            'method': 'all_layers',
        }

    # ----- Condition 4: Hippocampal retrieval + CA1 -----
    # Cue with the test sentence's final-token EC input
    test_last_residuals = extract_layer_residuals(
        hidden_baseline, last_pos, device)
    test_ec = cortical_proj.project(test_last_residuals)

    # Hippocampal retrieval
    retrieved_ec, ca3_cue, ca3_succ = hippo.retrieve_single_ec_deep(test_ec)

    # Retrieval quality diagnostic
    retrieval_sim = cosine_sim(retrieved_ec, stored_ec)
    results['retrieval_sim_to_stored'] = float(retrieval_sim)

    # CA1 filtering (correlation gate)
    ca1 = CA1_CorrelationGate(d_ec, device=device, dtype=torch.float32)
    filtered_ec = ca1.filter(retrieved_ec, test_ec)

    # B matrix reconstruction
    hippo_layers = backproj.retrieve(filtered_ec)

    results['hippo_retrieval'] = {}
    for alpha in alpha_values:
        modified_logits = inject_all_layers(
            model, tokenizer, test_text, hippo_layers, alpha=alpha,
            device=device)
        modified_probs = F.softmax(modified_logits, dim=-1)

        inj_probs = {}
        for ts in all_tokens:
            tids = tokenizer.encode(ts)
            if len(tids) >= 1:
                inj_probs[ts] = float(modified_probs[tids[0]])

        p_exp = sum(inj_probs.get(t, 0) for t in expected)
        p_def = sum(inj_probs.get(t, 0) for t in default)

        results['hippo_retrieval'][alpha] = {
            'probs': inj_probs,
            'p_expected': p_exp,
            'p_default': p_def,
        }

    # ----- Representational similarity -----
    # Compare hidden states at last layer, last position
    baseline_hidden_last = hidden_baseline[-1][0, last_pos, :].clone()
    reference_hidden_last = hidden_ref[-1][0, ref_last_pos, :].clone()

    results['repr_sim_baseline_to_ref'] = cosine_sim(
        baseline_hidden_last, reference_hidden_last)

    # How close does oracle injection get to the reference representation?
    best_alpha = alpha_values[len(alpha_values) // 2]  # middle alpha
    oracle_inj_last = oracle_layers[-1]
    h_norm = float(torch.linalg.norm(baseline_hidden_last))
    i_norm = float(torch.linalg.norm(oracle_inj_last))
    if i_norm > 1e-10:
        scaled = oracle_inj_last * (h_norm / i_norm) * best_alpha
    else:
        scaled = oracle_inj_last
    injected_hidden = baseline_hidden_last + scaled
    results['repr_sim_injected_to_ref'] = cosine_sim(
        injected_hidden, reference_hidden_last)

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(all_results, alpha_values, save_path):
    """Comprehensive results figure."""
    n_pairs = len(all_results)

    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Hippocampal Fact-Learning Paradigm",
                 fontsize=15, fontweight='bold', y=0.98)

    # --- (0,0:2): P(expected) across conditions for each pair ---
    ax = fig.add_subplot(gs[0, :])
    pair_ids = [r['id'] for r in all_results]
    x = np.arange(n_pairs)
    width = 0.2

    baseline_p = [r['baseline_p_expected'] for r in all_results]
    reference_p = [r['reference_p_expected'] for r in all_results]

    # Pick the best alpha for oracle and hippocampal
    best_oracle = []
    best_hippo = []
    for r in all_results:
        best_o = max(r['oracle_all_layer'].values(),
                     key=lambda v: v['p_expected'])
        best_oracle.append(best_o['p_expected'])
        best_h = max(r['hippo_retrieval'].values(),
                     key=lambda v: v['p_expected'])
        best_hippo.append(best_h['p_expected'])

    ax.bar(x - 1.5*width, baseline_p, width, label='Baseline', color='gray')
    ax.bar(x - 0.5*width, best_oracle, width,
           label='Oracle injection (best alpha)', color='steelblue')
    ax.bar(x + 0.5*width, best_hippo, width,
           label='Hippo retrieval (best alpha)', color='coral')
    ax.bar(x + 1.5*width, reference_p, width,
           label='Reference (ceiling)', color='forestgreen')
    ax.set_ylabel("P(expected token)")
    ax.set_title("Probability of Correct (Surprising) Token Across Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_ids, fontsize=8, rotation=30, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (1,0): Alpha sweep for oracle last-layer injection ---
    ax = fig.add_subplot(gs[1, 0])
    for i, r in enumerate(all_results):
        alphas = sorted(r['oracle_injection'].keys())
        p_exp = [r['oracle_injection'][a]['p_expected'] for a in alphas]
        ax.plot(alphas, p_exp, 'o-', alpha=0.5, markersize=3, label=r['id'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("P(expected)")
    ax.set_title("Oracle Last-Layer: Alpha Sweep")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    # --- (1,1): Alpha sweep for oracle all-layer injection ---
    ax = fig.add_subplot(gs[1, 1])
    for i, r in enumerate(all_results):
        alphas = sorted(r['oracle_all_layer'].keys())
        p_exp = [r['oracle_all_layer'][a]['p_expected'] for a in alphas]
        ax.plot(alphas, p_exp, 'o-', alpha=0.5, markersize=3, label=r['id'])
    ax.set_xlabel("Alpha")
    ax.set_ylabel("P(expected)")
    ax.set_title("Oracle All-Layer: Alpha Sweep")
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    # --- (1,2): Retrieval quality ---
    ax = fig.add_subplot(gs[1, 2])
    ret_sims = [r['retrieval_sim_to_stored'] for r in all_results]
    ax.barh(range(n_pairs), ret_sims, color='mediumpurple', alpha=0.8)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=8)
    ax.set_xlabel("Cosine Sim (retrieved vs stored)")
    ax.set_title("Hippocampal Retrieval Quality")
    ax.grid(True, alpha=0.3, axis='x')

    # --- (2,0): Representational similarity ---
    ax = fig.add_subplot(gs[2, 0])
    baseline_to_ref = [r['repr_sim_baseline_to_ref'] for r in all_results]
    injected_to_ref = [r['repr_sim_injected_to_ref'] for r in all_results]
    x = np.arange(n_pairs)
    ax.bar(x - 0.15, baseline_to_ref, 0.3,
           label='Baseline -> Ref', color='gray')
    ax.bar(x + 0.15, injected_to_ref, 0.3,
           label='Injected -> Ref', color='steelblue')
    ax.set_ylabel("Cosine Sim to Reference")
    ax.set_title("Representational Similarity to Reference")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_ids, fontsize=6, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- (2,1:2): Top predictions table ---
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis('off')
    table_text = "Top-5 Predictions (selected pairs)\n"
    table_text += "=" * 70 + "\n\n"

    for r in all_results[:5]:  # first 5 pairs
        table_text += f"{r['id']}:\n"
        table_text += f"  Test: \"{r['test']}\"\n"
        table_text += f"  Baseline:  "
        table_text += ", ".join(
            f"{tok}({p:.3f})" for tok, p in r['baseline_top5'][:3])
        table_text += "\n"
        table_text += f"  Reference: "
        table_text += ", ".join(
            f"{tok}({p:.3f})" for tok, p in r['reference_top5'][:3])
        table_text += "\n"
        # Best oracle
        best_alpha = max(r['oracle_all_layer'].keys(),
                         key=lambda a: r['oracle_all_layer'][a]['p_expected'])
        best = r['oracle_all_layer'][best_alpha]
        if 'top5' in best:
            table_text += f"  Oracle (a={best_alpha}): "
            table_text += ", ".join(
                f"{tok}({p:.3f})" for tok, p in best['top5'][:3])
            table_text += "\n"
        table_text += "\n"

    ax.text(0.02, 0.98, table_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- (3,0:2): Summary statistics ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    # Compute summary stats
    mean_baseline = np.mean([r['baseline_p_expected'] for r in all_results])
    mean_reference = np.mean([r['reference_p_expected'] for r in all_results])
    mean_oracle = np.mean(best_oracle)
    mean_hippo = np.mean(best_hippo)
    mean_ret_sim = np.mean(ret_sims)
    mean_repr_base = np.mean(baseline_to_ref)
    mean_repr_inj = np.mean(injected_to_ref)

    # Count improvements
    n_oracle_improves = sum(1 for i in range(n_pairs)
                            if best_oracle[i] > baseline_p[i] * 1.1)
    n_hippo_improves = sum(1 for i in range(n_pairs)
                           if best_hippo[i] > baseline_p[i] * 1.1)
    n_repr_improves = sum(1 for i in range(n_pairs)
                          if injected_to_ref[i] > baseline_to_ref[i])

    summary = "SUMMARY\n" + "=" * 60 + "\n\n"
    summary += f"Mean P(expected) across {n_pairs} fact-test pairs:\n"
    summary += f"  Baseline (no fact):     {mean_baseline:.6f}\n"
    summary += f"  Oracle injection:       {mean_oracle:.6f}  "
    summary += f"({n_oracle_improves}/{n_pairs} improved >10%)\n"
    summary += f"  Hippo retrieval:        {mean_hippo:.6f}  "
    summary += f"({n_hippo_improves}/{n_pairs} improved >10%)\n"
    summary += f"  Reference (ceiling):    {mean_reference:.6f}\n\n"
    summary += f"Recovery ratio (oracle):  "
    if mean_reference - mean_baseline > 1e-8:
        recovery = (mean_oracle - mean_baseline) / (mean_reference - mean_baseline)
        summary += f"{recovery:.3f}\n"
    else:
        summary += "N/A (reference = baseline)\n"
    summary += f"\nRetrieval quality:        {mean_ret_sim:.4f} mean cosine sim\n"
    summary += f"\nRepresentational similarity to reference:\n"
    summary += f"  Baseline:               {mean_repr_base:.4f}\n"
    summary += f"  After injection:        {mean_repr_inj:.4f}  "
    summary += f"({n_repr_improves}/{n_pairs} closer to ref)\n"

    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    gpt2_device = 'cpu'
    if torch.cuda.is_available():
        gpt2_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpt2_device = 'mps'

    print("=" * 70)
    print("HIPPOCAMPAL FACT-LEARNING PARADIGM")
    print("=" * 70)

    # ---- Phase 0: Build hippocampal system ----
    print("\n--- Phase 0: Building hippocampal system ---")
    system = build_hippocampal_system(device)

    # ---- Load model for fact encoding and testing ----
    model, tokenizer = load_model(gpt2_device)

    # ---- Phase 1: Encode facts ----
    print("\n--- Phase 1: Encoding facts ---")
    stored_facts = encode_facts(
        model, tokenizer, system, FACT_TEST_PAIRS,
        device=device, n_repetitions=3)

    print(f"\n  Stored {len(stored_facts)} facts")
    for fid, info in stored_facts.items():
        print(f"    {fid}: {info['n_tokens']} tokens")

    # ---- Phase 2: Evaluate ----
    print("\n--- Phase 2: Evaluating fact-test pairs ---")
    alpha_values = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

    all_results = []
    for pair in FACT_TEST_PAIRS:
        print(f"\n  Evaluating: {pair['id']}")
        print(f"    Fact: {pair['fact']}")
        print(f"    Test: {pair['test']}")

        result = evaluate_pair(
            model, tokenizer, system, pair, stored_facts,
            alpha_values, device=device)

        # Print key results
        print(f"    P(expected) baseline:  {result['baseline_p_expected']:.6f}")
        print(f"    P(expected) reference: {result['reference_p_expected']:.6f}")

        best_alpha_oracle = max(
            result['oracle_all_layer'].keys(),
            key=lambda a: result['oracle_all_layer'][a]['p_expected'])
        best_p_oracle = result['oracle_all_layer'][best_alpha_oracle]['p_expected']
        print(f"    P(expected) oracle:    {best_p_oracle:.6f} "
              f"(alpha={best_alpha_oracle})")

        best_alpha_hippo = max(
            result['hippo_retrieval'].keys(),
            key=lambda a: result['hippo_retrieval'][a]['p_expected'])
        best_p_hippo = result['hippo_retrieval'][best_alpha_hippo]['p_expected']
        print(f"    P(expected) hippo:     {best_p_hippo:.6f} "
              f"(alpha={best_alpha_hippo})")

        print(f"    Retrieval sim:         {result['retrieval_sim_to_stored']:.4f}")
        print(f"    Baseline top3: {result['baseline_top5'][:3]}")
        print(f"    Reference top3: {result['reference_top5'][:3]}")

        all_results.append(result)

    # ---- Phase 3: Plot ----
    print("\n--- Phase 3: Plotting ---")
    plot_results(all_results, alpha_values, "fact_learning_results.png")

    # ---- Final summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n{'Pair':<20s} {'Baseline':>10s} {'Oracle':>10s} "
          f"{'Hippo':>10s} {'Reference':>10s} {'Ret.Sim':>8s}")
    print("-" * 70)

    for r in all_results:
        best_o = max(r['oracle_all_layer'].values(),
                     key=lambda v: v['p_expected'])['p_expected']
        best_h = max(r['hippo_retrieval'].values(),
                     key=lambda v: v['p_expected'])['p_expected']
        print(f"{r['id']:<20s} {r['baseline_p_expected']:>10.6f} "
              f"{best_o:>10.6f} {best_h:>10.6f} "
              f"{r['reference_p_expected']:>10.6f} "
              f"{r['retrieval_sim_to_stored']:>8.4f}")

    mean_b = np.mean([r['baseline_p_expected'] for r in all_results])
    mean_o = np.mean([max(r['oracle_all_layer'].values(),
                          key=lambda v: v['p_expected'])['p_expected']
                      for r in all_results])
    mean_h = np.mean([max(r['hippo_retrieval'].values(),
                          key=lambda v: v['p_expected'])['p_expected']
                      for r in all_results])
    mean_r = np.mean([r['reference_p_expected'] for r in all_results])

    print("-" * 70)
    print(f"{'MEAN':<20s} {mean_b:>10.6f} {mean_o:>10.6f} "
          f"{mean_h:>10.6f} {mean_r:>10.6f}")

    if mean_r - mean_b > 1e-8:
        recovery_o = (mean_o - mean_b) / (mean_r - mean_b)
        recovery_h = (mean_h - mean_b) / (mean_r - mean_b)
        print(f"\nRecovery ratio (oracle):      {recovery_o:.3f}")
        print(f"Recovery ratio (hippocampal): {recovery_h:.3f}")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()