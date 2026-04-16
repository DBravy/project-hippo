"""
Diverse Corpus B-Matrix Training
==================================

The representational analysis showed that B matrices trained on narrow
hippocampus text produce anti-aligned reconstructions at late layers.

This script retrains the entire system on a topically diverse corpus
covering the semantic domains present in the fact-test pairs (food,
animals, colors, buildings, schools, gardens, daily routines, etc.),
then re-evaluates both logit-level and representational metrics.

Variable isolated: training corpus diversity. Everything else
(A matrices, hippocampal architecture, fact-test pairs) stays the same.
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
    load_gpt2_and_extract,
    encode_phase_a,
    encode_phase_b,
)
from ca1_mechanism_test import CA1_CorrelationGate
from fact_learning_paradigm import (
    FACT_TEST_PAIRS,
    load_model,
    get_hidden_states_and_logits,
    extract_layer_residuals,
    get_token_probs,
    inject_last_layer,
    inject_all_layers,
    encode_facts,
)
from repr_analysis import analyze_pair


# =============================================================================
# DIVERSE TRAINING CORPUS
# =============================================================================

DIVERSE_CORPUS = """
Every morning the baker arrives at the shop before dawn to prepare fresh bread
and pastries for the day. The ovens are heated to the right temperature and
the dough that has been rising overnight is shaped into loaves and rolls. Some
breads are made with unusual ingredients like seaweed or lavender while others
follow traditional recipes using only flour water salt and yeast. The bakery
is famous in the neighborhood for its sourdough and its chocolate croissants.
Customers line up early to buy warm bread fresh from the oven.

The restaurant on the corner of Main Street serves food from many different
countries. Their pizza is topped with unexpected combinations including figs
and goat cheese or even chocolate and chili peppers. The chef trained in Paris
and Tokyo before opening this small restaurant. Every dish on the menu tells
a story about a place the chef has visited. The desserts are particularly
creative featuring flavors like matcha and black sesame or lemon and thyme.

Doctor Johnson runs a small clinic on the edge of town. She treats patients
with a variety of conditions from headaches and back pain to more serious
illnesses. Her approach combines traditional medicine with newer techniques.
Some patients come to her for advice about sleep problems while others need
help managing stress and anxiety. She always recommends a combination of rest
proper nutrition and gentle exercise. When a patient complains of a persistent
headache she first checks for obvious causes before prescribing any medication.

The old school building has stood at the center of the village for over a
hundred years. Students arrive each morning wearing their uniforms which have
changed color several times over the decades. The current uniforms are dark
blue but older photographs show students wearing green and before that bright
red. The school teaches children from age five to eighteen and has produced
many notable graduates. The library is the pride of the school with thousands
of books donated by former students and local families.

In the garden behind the house Martha tends to her unusual collection of plants
and stones. While most gardens in the neighborhood feature roses and tulips
Martha prefers to grow exotic herbs and arrange interesting rocks and minerals
among her flower beds. Visitors are often surprised to find crystals and
polished stones where they expected to see ordinary flowers. She has been
collecting rocks from her travels for over thirty years and each one has a
story. The garden also features a small pond where frogs and dragonflies make
their home during the summer months.

The village sits in a valley surrounded by mountains. During winter the
temperatures drop well below freezing and snow covers everything for months.
The villagers have developed many traditions to survive the cold season. They
gather firewood throughout autumn storing it in large piles beside their homes.
Some families burn wood while others have switched to modern heating systems.
In the old days people would burn whatever they could find including old
furniture and even books that nobody wanted anymore. The children build snow
forts and have snowball fights while the adults prepare warm soups and stews.

Cats and dogs are the most common pets in the country but some communities have
unusual relationships with animals. In certain remote regions cats are trained
to perform tasks that dogs typically handle. There are stories of cats that
learned to herd sheep and others that were trained as therapy animals in
hospitals. Cats are naturally independent creatures but with patience and the
right training they can learn to follow commands and even deliver small objects.
Some farmers use cats to control mice in their barns while others keep them
purely as companions.

The city council recently passed several new laws affecting how buildings look
and how people live. One controversial law requires all new buildings in the
historic district to be painted in traditional colors including white cream
and light blue. Another law limits building heights to preserve mountain views.
Some residents support the changes while others argue that the laws restrict
personal freedom. The construction workers have been busy painting and
renovating buildings throughout the summer. Bright colors like purple orange
and yellow are now forbidden in the historic zone but allowed elsewhere.

Professor Williams teaches biology at the university and conducts research on
how plants respond to different types of light. Her experiments have shown that
certain wavelengths promote faster growth while others inhibit it. She recently
published a paper on how moonlight affects plant behavior at night. The
research attracted attention from farmers who want to optimize their growing
conditions. Students in her laboratory spend long hours monitoring plant growth
under carefully controlled lighting conditions. The greenhouse on campus is
filled with hundreds of plants each labeled with its specific light treatment.

The tribe lives in a remote area far from modern civilization. Their customs
and traditions have been passed down through hundreds of generations. They
sleep in communal shelters and rise with the sun each morning. Their sleeping
positions and habits are different from what most people consider normal. Some
members of the tribe practice meditation before sleep while others perform
physical rituals. The elders teach the children traditional songs and dances
that tell the story of their ancestors. Every evening the community gathers
around a fire to share stories and plan for the next day.

Alice works at the library downtown. She wakes up at five every morning and
follows a strict routine before heading to work. She exercises for thirty
minutes then showers and prepares breakfast. Her breakfast habits are
unconventional compared to most people. While her colleagues drink coffee or
tea Alice prefers more unusual morning beverages. She claims that her choice
of morning drink gives her energy and keeps her focused throughout the day.
After breakfast she walks to work arriving exactly at eight every morning.

The local museum has a collection of minerals and gemstones from around the
world. Visitors can see diamonds rubies emeralds and dozens of other precious
stones. The museum also has an exhibit on common rocks and how they are formed
through geological processes. Children particularly enjoy the interactive
displays where they can touch and examine different types of rocks. The gift
shop sells small polished stones and crystal specimens. Every weekend the
museum hosts workshops where families can learn about geology and even try
their hand at identifying different minerals.

The harbor town depends on fishing for its livelihood. Every morning before
dawn the fishing boats head out to sea returning in the afternoon with their
catch. The fish market opens at three in the afternoon and closes at six. Fresh
fish is available daily including cod salmon mackerel and sometimes unusual
species from the deep ocean. The restaurants along the waterfront serve the
freshest seafood in the region. Tourists come from far away to taste the local
specialties particularly the grilled octopus and the fish soup.

In the art studio downtown painters and sculptors work side by side creating
pieces for the annual exhibition. The current theme is color and emotion with
artists exploring how different colors affect the viewer. Some painters work
exclusively with warm colors like red orange and yellow while others prefer
cool tones of blue green and purple. One artist has created a series of
paintings using only shades of purple inspired by the lavender fields outside
the city. The sculptures range from small delicate pieces to large installations
that fill entire rooms.

The sports complex at the edge of town has facilities for dozens of different
activities. There are courts for tennis and basketball fields for soccer and
rugby and a swimming pool that hosts competitions throughout the year. The
gymnasium is equipped with modern exercise equipment and offers classes in
yoga martial arts and dance. Young athletes train here every day hoping to
compete at the national level. The coaches emphasize discipline hard work and
sportsmanship above all else.

Travel between the islands requires taking a ferry or a small plane. The main
island has an airport that receives flights from the mainland twice daily.
Smaller islands are connected by boat services that run several times a week.
During storms the ferries are cancelled and the islands become temporarily
isolated. The islanders are accustomed to this and always keep extra supplies
of food and medicine. Tourism is the main industry with visitors coming to
enjoy the beaches the hiking trails and the unique wildlife.

The hospital on the hill serves the entire region providing care for thousands
of patients each year. The emergency department is staffed around the clock by
doctors and nurses who handle everything from minor injuries to life threatening
conditions. The hospital recently added a new wing for rehabilitation where
patients recover from surgeries and injuries. Physical therapists work with
patients to help them regain strength and mobility. The children's ward is
decorated with bright murals and has a playroom where young patients can forget
about their treatments for a while.

The weather in this region is unpredictable. Mornings can be sunny and warm
but by afternoon thunderstorms often roll in from the mountains. The rainy
season lasts from October to March bringing heavy downpours that sometimes
cause flooding in the lower parts of town. During the dry season the landscape
turns golden brown and farmers worry about their crops. The wind can be fierce
especially in autumn when it blows leaves from the trees and sends them swirling
through the streets. Despite the challenging weather the residents love their
town and would not want to live anywhere else.

The bookstore on the corner has been run by the same family for three
generations. The shelves are packed with books of every kind from classic
novels to modern thrillers from science textbooks to cookbooks. There is a
cozy reading corner with comfortable chairs where customers can browse before
buying. The store also sells maps postcards and stationery. Every Friday
evening the bookstore hosts a reading group where local authors share their
work and discuss literature with the community.

The train station was built in the early nineteen hundreds and has been
renovated several times since then. The original stone facade has been
preserved but the interior has been modernized with digital displays and
automatic ticket machines. Trains depart every hour to the capital and every
two hours to the coastal towns. The platform is covered with a glass roof that
lets in natural light while protecting passengers from rain. A small cafe on
the platform serves coffee and sandwiches to waiting travelers.
"""


# =============================================================================
# SYSTEM BUILDING WITH DIVERSE CORPUS
# =============================================================================

def build_diverse_system(device, dtype=torch.float32):
    """Build and train hippocampal system + B matrices on diverse corpus."""
    n_layers = 6
    d_model = 768
    r_per_layer = 128
    d_ec = n_layers * r_per_layer

    gpt2_device = 'cpu'
    if torch.cuda.is_available():
        gpt2_device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        gpt2_device = 'mps'

    print("\n--- Extracting diverse corpus representations ---")
    # Use longer sequences and more of them to cover the larger corpus
    sequences_tokens, sequences_residuals, tokenizer = load_gpt2_and_extract(
        DIVERSE_CORPUS, seq_length=64, n_sequences=20, device=gpt2_device)

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

    print("\n--- Training on diverse corpus (Phase A) ---")
    encode_phase_a(hippo, cortical_proj, sequences_residuals, n_repetitions=5)

    print("\n--- Training B matrices on diverse corpus (Phase B) ---")
    encode_phase_b(hippo, cortical_proj, backproj, sequences_residuals,
                   n_repetitions=5)

    return {
        'hippo': hippo,
        'cortical_proj': cortical_proj,
        'backproj': backproj,
        'd_ec': d_ec,
        'n_training_tokens': sum(len(s) for s in sequences_residuals),
    }


# =============================================================================
# EVALUATION (combines logit + representational analysis)
# =============================================================================

def evaluate_all(model, tokenizer, system, stored_facts, device='cpu'):
    """Run both logit-level and representational analysis for all pairs."""
    cortical_proj = system['cortical_proj']
    backproj = system['backproj']
    hippo = system['hippo']
    d_ec = system['d_ec']
    n_layers = 6

    alpha_values = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    all_tokens_of_interest = set()
    for pair in FACT_TEST_PAIRS:
        all_tokens_of_interest.update(pair['expected'])
        all_tokens_of_interest.update(pair['default'])

    results = []

    for pair in FACT_TEST_PAIRS:
        print(f"\n  {pair['id']}:")
        fact_text = pair['fact']
        test_text = pair['test']
        expected = pair['expected']
        default = pair['default']
        all_tokens = expected + default
        reference_text = fact_text + " " + test_text

        # --- Hidden states ---
        hidden_baseline, logits_baseline, test_tokens = \
            get_hidden_states_and_logits(model, tokenizer, test_text, device)
        hidden_reference, logits_ref, ref_tokens = \
            get_hidden_states_and_logits(model, tokenizer, reference_text, device)
        last_pos_test = len(test_tokens) - 1
        last_pos_ref = len(ref_tokens) - 1

        # --- Baseline and reference probs ---
        baseline_probs = get_token_probs(logits_baseline, last_pos_test,
                                         tokenizer, all_tokens)
        reference_probs = get_token_probs(logits_ref, last_pos_ref,
                                          tokenizer, all_tokens)
        p_exp_baseline = sum(baseline_probs.get(t, 0) for t in expected)
        p_exp_reference = sum(reference_probs.get(t, 0) for t in expected)

        # --- Oracle injection (stored EC -> B matrices -> inject) ---
        stored_ec = stored_facts[pair['id']]['ec_input']
        oracle_layers = backproj.retrieve(stored_ec)

        best_p_oracle = p_exp_baseline
        best_alpha_oracle = 0
        for alpha in alpha_values:
            mod_logits = inject_all_layers(
                model, tokenizer, test_text, oracle_layers, alpha=alpha,
                device=device)
            mod_probs = F.softmax(mod_logits, dim=-1)
            p_exp = sum(float(mod_probs[tokenizer.encode(t)[0]])
                        for t in expected
                        if len(tokenizer.encode(t)) >= 1)
            if p_exp > best_p_oracle:
                best_p_oracle = p_exp
                best_alpha_oracle = alpha

        # --- Hippocampal retrieval ---
        test_last_residuals = extract_layer_residuals(
            hidden_baseline, last_pos_test, device)
        test_ec = cortical_proj.project(test_last_residuals)
        retrieved_ec, _, _ = hippo.retrieve_single_ec_deep(test_ec)
        retrieval_sim = cosine_sim(retrieved_ec, stored_ec)

        ca1 = CA1_CorrelationGate(d_ec, device=device, dtype=torch.float32)
        filtered_ec = ca1.filter(retrieved_ec, test_ec)
        hippo_layers = backproj.retrieve(filtered_ec)

        best_p_hippo = p_exp_baseline
        best_alpha_hippo = 0
        for alpha in alpha_values:
            mod_logits = inject_all_layers(
                model, tokenizer, test_text, hippo_layers, alpha=alpha,
                device=device)
            mod_probs = F.softmax(mod_logits, dim=-1)
            p_exp = sum(float(mod_probs[tokenizer.encode(t)[0]])
                        for t in expected
                        if len(tokenizer.encode(t)) >= 1)
            if p_exp > best_p_hippo:
                best_p_hippo = p_exp
                best_alpha_hippo = alpha

        # --- Representational analysis ---
        per_layer_repr = []
        for l in range(n_layers):
            baseline_h = hidden_baseline[l + 1][0, last_pos_test, :].clone().float()
            reference_h = hidden_reference[l + 1][0, last_pos_ref, :].clone().float()
            oracle_inj = oracle_layers[l].clone().float()

            direction_needed = reference_h - baseline_h
            dir_norm = float(torch.linalg.norm(direction_needed))

            oracle_alignment = cosine_sim(oracle_inj, direction_needed)
            baseline_to_ref = cosine_sim(baseline_h, reference_h)

            # Best alpha for this layer
            baseline_norm = float(torch.linalg.norm(baseline_h))
            oracle_inj_norm = float(torch.linalg.norm(oracle_inj))
            best_sim = baseline_to_ref
            best_a = 0
            for alpha in np.logspace(-3, 1, 30):
                if oracle_inj_norm > 1e-10:
                    scaled = oracle_inj * (baseline_norm / oracle_inj_norm) * alpha
                else:
                    scaled = torch.zeros_like(oracle_inj)
                injected = baseline_h + scaled
                sim = cosine_sim(injected, reference_h)
                if sim > best_sim:
                    best_sim = sim
                    best_a = alpha

            per_layer_repr.append({
                'oracle_alignment': oracle_alignment,
                'baseline_to_ref': baseline_to_ref,
                'best_sim': best_sim,
                'best_alpha': best_a,
                'delta': best_sim - baseline_to_ref,
            })

        r = {
            'id': pair['id'],
            'p_exp_baseline': p_exp_baseline,
            'p_exp_reference': p_exp_reference,
            'p_exp_oracle': best_p_oracle,
            'alpha_oracle': best_alpha_oracle,
            'p_exp_hippo': best_p_hippo,
            'alpha_hippo': best_alpha_hippo,
            'retrieval_sim': float(retrieval_sim),
            'per_layer': per_layer_repr,
        }
        results.append(r)

        print(f"    Baseline P(exp): {p_exp_baseline:.6f}")
        print(f"    Reference P(exp): {p_exp_reference:.6f}")
        print(f"    Oracle P(exp): {best_p_oracle:.6f} (alpha={best_alpha_oracle})")
        print(f"    Hippo P(exp): {best_p_hippo:.6f} (alpha={best_alpha_hippo})")
        print(f"    Retrieval sim: {float(retrieval_sim):.4f}")
        for l, lr in enumerate(per_layer_repr):
            print(f"    L{l+1}: align={lr['oracle_alignment']:+.4f}  "
                  f"delta_sim={lr['delta']:+.6f}")

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def plot_comparison(results, save_path):
    """Results figure."""
    n_pairs = len(results)
    n_layers = 6

    fig = plt.figure(figsize=(22, 20))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Fact Learning with Diverse B-Matrix Training",
                 fontsize=15, fontweight='bold', y=0.99)

    pair_ids = [r['id'] for r in results]

    # --- Row 0: P(expected) comparison ---
    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(n_pairs)
    width = 0.2
    ax.bar(x - 1.5*width, [r['p_exp_baseline'] for r in results], width,
           label='Baseline', color='gray')
    ax.bar(x - 0.5*width, [r['p_exp_oracle'] for r in results], width,
           label='Oracle inj.', color='steelblue')
    ax.bar(x + 0.5*width, [r['p_exp_hippo'] for r in results], width,
           label='Hippo retr.', color='coral')
    ax.bar(x + 1.5*width, [r['p_exp_reference'] for r in results], width,
           label='Reference', color='forestgreen')
    ax.set_ylabel("P(expected token)")
    ax.set_title("Token Probability Across Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_ids, fontsize=7, rotation=30, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Row 0 right: Retrieval quality ---
    ax = fig.add_subplot(gs[0, 2])
    ax.barh(range(n_pairs), [r['retrieval_sim'] for r in results],
            color='mediumpurple', alpha=0.8)
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_xlabel("Cosine Sim")
    ax.set_title("Retrieval Quality")
    ax.grid(True, alpha=0.3, axis='x')

    # --- Row 1: Alignment heatmap ---
    ax = fig.add_subplot(gs[1, 0])
    align_matrix = np.array(
        [[lr['oracle_alignment'] for lr in r['per_layer']] for r in results])
    im = ax.imshow(align_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"L{l+1}" for l in range(6)])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_title("Oracle Alignment with\nNeeded Direction")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Delta sim heatmap
    ax = fig.add_subplot(gs[1, 1])
    delta_matrix = np.array(
        [[lr['delta'] for lr in r['per_layer']] for r in results])
    im = ax.imshow(delta_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-0.01, vmax=0.01)
    ax.set_xlabel("Layer")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"L{l+1}" for l in range(6)])
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_ids, fontsize=7)
    ax.set_title("Sim Improvement\n(best alpha)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Mean alignment by layer
    ax = fig.add_subplot(gs[1, 2])
    mean_align = np.mean(align_matrix, axis=0)
    std_align = np.std(align_matrix, axis=0)
    ax.bar(range(1, 7), mean_align, yerr=std_align,
           color='steelblue', alpha=0.8, capsize=3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Alignment")
    ax.set_title("Mean Alignment by Layer")
    ax.set_xticks(range(1, 7))
    ax.grid(True, alpha=0.3, axis='y')

    # --- Row 2: Per-pair detail ---
    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')
    header = (f"{'Pair':<20s} {'Baseline':>10s} {'Oracle':>10s} "
              f"{'Hippo':>10s} {'Reference':>10s} {'Ret.Sim':>8s} "
              f"{'Recovery':>10s}\n")
    header += "-" * 80 + "\n"
    rows = ""
    for r in results:
        gap = r['p_exp_reference'] - r['p_exp_baseline']
        if gap > 1e-8:
            recovery = (r['p_exp_oracle'] - r['p_exp_baseline']) / gap
        else:
            recovery = 0
        rows += (f"{r['id']:<20s} {r['p_exp_baseline']:>10.6f} "
                 f"{r['p_exp_oracle']:>10.6f} {r['p_exp_hippo']:>10.6f} "
                 f"{r['p_exp_reference']:>10.6f} {r['retrieval_sim']:>8.4f} "
                 f"{recovery:>+10.4f}\n")
    ax.text(0.02, 0.95, header + rows, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # --- Row 2 right: alignment per layer ---
    ax = fig.add_subplot(gs[2, 2])
    for i, r in enumerate(results):
        aligns = [lr['oracle_alignment'] for lr in r['per_layer']]
        ax.plot(range(1, 7), aligns, 'o-', alpha=0.4, markersize=3,
                label=r['id'])
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Alignment")
    ax.set_title("Per-Pair Alignment Curves")
    ax.set_xticks(range(1, 7))
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Row 3: Summary ---
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    mean_baseline = np.mean([r['p_exp_baseline'] for r in results])
    mean_oracle = np.mean([r['p_exp_oracle'] for r in results])
    mean_hippo = np.mean([r['p_exp_hippo'] for r in results])
    mean_reference = np.mean([r['p_exp_reference'] for r in results])
    mean_align_all = np.mean(align_matrix)
    n_positive = np.sum(align_matrix > 0)
    n_total = align_matrix.size
    mean_delta = np.mean(delta_matrix)
    n_improved = np.sum(delta_matrix > 0)

    if mean_reference - mean_baseline > 1e-8:
        recovery_o = (mean_oracle - mean_baseline) / (mean_reference - mean_baseline)
        recovery_h = (mean_hippo - mean_baseline) / (mean_reference - mean_baseline)
    else:
        recovery_o = recovery_h = 0

    summary = "SUMMARY: DIVERSE CORPUS B-MATRIX TRAINING\n"
    summary += "=" * 50 + "\n\n"
    summary += f"Training corpus: ~{20*64} tokens (diverse topics)\n"
    summary += f"vs original: ~256 tokens (hippocampus text only)\n\n"
    summary += f"Mean P(expected):\n"
    summary += f"  Baseline:    {mean_baseline:.6f}\n"
    summary += f"  Oracle inj:  {mean_oracle:.6f}\n"
    summary += f"  Hippo retr:  {mean_hippo:.6f}\n"
    summary += f"  Reference:   {mean_reference:.6f}\n\n"
    summary += f"Recovery ratio (oracle):  {recovery_o:+.4f}\n"
    summary += f"Recovery ratio (hippo):   {recovery_h:+.4f}\n\n"
    summary += f"Representational alignment:\n"
    summary += f"  Mean: {mean_align_all:+.4f}\n"
    summary += f"  Positive: {n_positive}/{n_total} ({100*n_positive/n_total:.1f}%)\n"
    summary += f"  Mean sim improvement: {mean_delta:+.6f}\n"
    summary += f"  Layers improved: {n_improved}/{n_total} ({100*n_improved/n_total:.1f}%)\n"

    ax.text(0.02, 0.95, summary, transform=ax.transAxes, fontsize=11,
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
    print("DIVERSE CORPUS B-MATRIX TRAINING")
    print("=" * 70)

    # Build system with diverse corpus
    system = build_diverse_system(device)
    print(f"\n  Training tokens: {system['n_training_tokens']}")

    # Load model
    model, tokenizer = load_model(gpt2_device)

    # Encode facts
    print("\n--- Encoding facts ---")
    stored_facts = encode_facts(
        model, tokenizer, system, FACT_TEST_PAIRS,
        device=device, n_repetitions=3)

    # Evaluate
    print("\n--- Evaluating ---")
    results = evaluate_all(model, tokenizer, system, stored_facts, device=device)

    # Print summary table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Pair':<20s} {'Baseline':>10s} {'Oracle':>10s} "
          f"{'Hippo':>10s} {'Reference':>10s} {'Ret.Sim':>8s}")
    print("-" * 70)
    for r in results:
        print(f"{r['id']:<20s} {r['p_exp_baseline']:>10.6f} "
              f"{r['p_exp_oracle']:>10.6f} {r['p_exp_hippo']:>10.6f} "
              f"{r['p_exp_reference']:>10.6f} {r['retrieval_sim']:>8.4f}")

    print(f"\nAlignment by layer:")
    for l in range(6):
        aligns = [r['per_layer'][l]['oracle_alignment'] for r in results]
        deltas = [r['per_layer'][l]['delta'] for r in results]
        print(f"  L{l+1}: mean_align={np.mean(aligns):+.4f}  "
              f"mean_delta={np.mean(deltas):+.6f}")

    # Plot
    print("\n--- Plotting ---")
    plot_comparison(results, "diverse_corpus_results.png")

    print(f"\n{'=' * 70}")
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
