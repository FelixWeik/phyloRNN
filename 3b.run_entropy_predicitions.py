"""
run_predictions_entropy_revb.py
================================
Erweiterung von run_predictions_revb.py um:
  1. informationstheoretische Site-Rate-Proxies (Shannon, norm. Shannon,
     Gini-Simpson, Rényi Ordnung 2) direkt aus dem simulierten One-Hot-Alignment
  2. zugehörige RevBayes-Skripte analog zu DL/DL5d
  3. AIC/BIC-Berechnung aus den RevBayes .log-Dateien (post-Burn-in)
"""

import os
import numpy as np
import pandas as pd
import phyloRNN as pn
from matplotlib import pyplot as plt
import subprocess 

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
wd        = os.getcwd()
data_wd   = os.path.join(os.getcwd(), "phyloRNN", "ali_tmp")
model_name = "t20_s100"
trained_model = pn.load_rnn_model(os.path.join(wd, "Trained_models", model_name))

plot      = False
log_rates = False
start_sim = 0
n_sim     = 5

# Modellnamen für AIC/BIC Auswertung
MODEL_NAMES = ["G", "DL", "DL5d", "SH", "NSH", "GS", "RE"]


# ---------------------------------------------------------------------------
# 1. Entropie-Funktionen
# ---------------------------------------------------------------------------

def _base_freqs_from_onehot(features_ali_site: np.ndarray,
                             n_taxa: int) -> np.ndarray:
    """
    Wandelt features_ali (n_sites, n_taxa*4) in Basisfrequenzen
    (n_sites, 4) um. Reihenfolge der Basen: A, C, G, T.
    Gaps / ambiguous bases werden ignoriert (One-Hot-Zeile = 0).
    """
    n_sites = features_ali_site.shape[0]
    # (n_sites, n_taxa, 4)
    onehot = features_ali_site.reshape(n_sites, n_taxa, 4)
    # Summe über Taxa → Anzahl pro Base pro Site (n_sites, 4)
    counts = onehot.sum(axis=1)
    # Normalisierung → Frequenzen
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)   # Division-by-zero-Schutz
    return counts / row_sums                            # (n_sites, 4)


def compute_entropy_rates(features_ali: np.ndarray,
                           n_taxa: int,
                           normalize_to_mean: bool = True) -> dict:
    """
    Berechnet vier informationstheoretische Rate-Proxies pro Site.

    Parameter
    ----------
    features_ali : np.ndarray, shape (n_sites, n_taxa*4)
        One-Hot-kodiertes Alignment aus res[0][0].
    n_taxa : int
        Anzahl Taxa (= n_taxa aus dem Simulator).
    normalize_to_mean : bool
        Wenn True, werden alle Maße auf Mittelwert=1 normalisiert
        (identisch zur phyloRNN-Konvention für siteRates).

    Rückgabe
    ----------
    dict mit Arrays der Länge n_sites:
        'shannon'      – Shannon-Entropie H (Bits: log2 oder Nats: loge, hier ln)
        'norm_shannon' – H / ln(4)  →  [0, 1]
        'gini_simpson' – 1 - Σ p²   →  [0, 0.75]
        'renyi2'       – -ln(Σ p²)  →  Rényi-Entropie Ordnung 2

    Hinweis: Komplett konservierte Sites haben Entropie = 0 → Rate = 0
    nach Normalisierung. Das ist biologisch korrekt (keine Variation → rate ≈ 0).
    """
    p = _base_freqs_from_onehot(features_ali, n_taxa)   # (n_sites, 4)
    eps = 1e-12                                          # numerische Stabilität

    # -- Shannon ---------------------------------------------------------
    # H = -Σ p_i * ln(p_i),  H ∈ [0, ln(4)]
    log_p = np.where(p > 0, np.log(p + eps), 0.0)
    shannon = -np.sum(p * log_p, axis=1)

    # -- Normalisierter Shannon ------------------------------------------
    # H_norm = H / ln(4),  H_norm ∈ [0, 1]
    norm_shannon = shannon / np.log(4)

    # -- Gini-Simpson ----------------------------------------------------
    # G = 1 - Σ p_i²,  G ∈ [0, 0.75]
    gini_simpson = 1.0 - np.sum(p ** 2, axis=1)

    # -- Rényi-Entropie Ordnung 2 ----------------------------------------
    # R2 = -ln(Σ p_i²),  R2 ∈ [0, ln(4)]
    # Entspricht -ln(1 - G) monoton → ähnliche Rangfolge wie Gini-Simpson,
    # aber logarithmische Stauchung extremer Werte.
    renyi2 = -np.log(np.sum(p ** 2, axis=1) + eps)

    rates = {
        'shannon':      shannon,
        'norm_shannon': norm_shannon,
        'gini_simpson': gini_simpson,
        'renyi2':       renyi2,
    }

    if normalize_to_mean:
        for key, arr in rates.items():
            mean_val = arr.mean()
            # Falls alle Sites identisch sind (mean=0): Fallback auf 1.0
            rates[key] = arr / mean_val if mean_val > eps else np.ones_like(arr)

    for key in rates:
        rates[key] = np.clip(rates[key], a_min=1e-6, a_max=None)

    return rates


# ---------------------------------------------------------------------------
# 2. AIC / BIC-Berechnung aus RevBayes .log-Dateien
# ---------------------------------------------------------------------------

def count_free_params(model: str, n_taxa: int) -> int:
    """
    Zählt freie Parameter je Modell.

    GTR-Kern (immer):
      - pi:  3 frei  (4 Frequenzen, summieren auf 1)
      - er:  5 frei  (6 Raten, auf Summe normiert)
      - bl:  2*n_taxa - 3  (ungewurzelter Baum)

    Modellspezifisch:
      - G:      +alpha +p_inv  = +2
      - DL/DL5d/SH/NSH/GS/RE: site rates sind FEST (kein freier Parameter)
    """
    base = 3 + 5 + (2 * n_taxa - 3)    # GTR + Astlängen
    extras = {"G": 2}.get(model, 0)    # nur Gamma hat zwei extra Parameter
    return base + extras


def compute_aic_bic(log_file: str,
                    n_params: int,
                    n_sites: int,
                    burn_in_frac: float = 0.25) -> dict:
    """
    Liest eine RevBayes .log-Datei und berechnet AIC/BIC.

    Wichtiger Hinweis: Verwendet den post-Burn-in-Mittelwert der
    Log-Likelihood als Schätzer für ln(L_max). Das ist eine Näherung
    (exakt wäre die Marginal-Likelihood via Stepping-Stone-Sampling).
    Für deskriptiven Modellvergleich reicht das aber.

    Parameter
    ----------
    log_file    : Pfad zur .log-Datei
    n_params    : Anzahl freier Parameter (aus count_free_params())
    n_sites     : Alignment-Länge (für BIC)
    burn_in_frac: Anteil der Kette, der als Burn-in verworfen wird
    """
    if not os.path.isfile(log_file):
        return None

    df = pd.read_csv(log_file, sep="\t", comment="#")

    # RevBayes-Spaltennamen können leichte Varianten haben
    ll_col = next((c for c in df.columns if "Likelihood" in c), None)
    if ll_col is None:
        print(f"  Warnung: Keine Likelihood-Spalte in {log_file}")
        return None

    burn_in = int(len(df) * burn_in_frac)
    post_burn = df.iloc[burn_in:]
    mean_ll   = post_burn[ll_col].mean()
    ess_approx = len(post_burn)          # Näherung; für echte ESS: arviz

    aic = 2 * n_params - 2 * mean_ll
    bic = np.log(n_sites) * n_params - 2 * mean_ll

    return {
        "log_file":  log_file,
        "n_params":  n_params,
        "n_samples": len(post_burn),
        "mean_ll":   round(mean_ll, 4),
        "AIC":       round(aic, 4),
        "BIC":       round(bic, 4),
    }


# ---------------------------------------------------------------------------
# 3. Simulator
# ---------------------------------------------------------------------------

sim = pn.simulator(
    n_taxa                 = 20,
    n_sites                = 100,
    n_eigen_features       = 3,
    min_rate               = 0,
    freq_uncorrelated_sites= 0.5,
    freq_mixed_models      = 0,
    store_mixed_model_info = True,
    tree_builder           = 'nj',
    subs_model_per_block   = False,
    phyml_path             = None,
    seqgen_path            = None,
    ali_path               = data_wd,
    DEBUG                  = False,
    verbose                = True,
    ali_schema             = "nexus",
    min_avg_br_length      = 0.01,
    max_avg_br_length      = 0.2,
)

N_TAXA  = 20
N_SITES = 100

# ---------------------------------------------------------------------------
# 4. Haupt-Schleife
# ---------------------------------------------------------------------------

all_aic_bic = []    # sammelt Ergebnisse über alle Simulationen

for sim_i in range(start_sim, start_sim + n_sim):

    print(f"\n{'='*60}")
    print(f"Simulation {sim_i}")
    print('='*60)

    ali_name = os.path.join(data_wd, "ali%s" % sim_i)
    res = sim.run_sim([sim_i, 1, ali_name, False])

    ali_file       = res[-1][0]['ali_file']
    true_site_rates= res[2][0]

    # -------------------------------------------------------------------------
    # A) RNN-Vorhersage
    # -------------------------------------------------------------------------
    sim_res = {
        'features_ali':  res[0][0],
        'labels_rates':  true_site_rates,
        'labels_smodel': None,
        'labels_tl':     res[-2][0],
    }

    (comp_sim, dict_inputs, comp_dict_outputs) = pn.rnn_in_out_dictionaries_from_sim(
        sim            = sim_res,
        log_rates      = log_rates,
        output_list    = ['per_site_rate', 'tree_len'],
        include_tree_features = False,
    )

    print("Running RNN predictions...")
    model_input = {
        'sequence_data': dict_inputs['sequence_data'].numpy().reshape(
            (1,
             dict_inputs['sequence_data'].shape[0],
             dict_inputs['sequence_data'].shape[1])
        )
    }
    predictions = trained_model.predict(model_input)
    site_rates  = predictions[0][0]

    if plot:
        print("MSE (RNN):", np.mean((true_site_rates - site_rates) ** 2))
        plt.scatter(true_site_rates, site_rates, label="RNN")
        plt.xlabel("True rates"); plt.ylabel("Predicted"); plt.legend()
        plt.show()

    # -------------------------------------------------------------------------
    # B) Entropie-basierte Site-Rates aus One-Hot-Matrix
    # -------------------------------------------------------------------------
    print("Computing entropy-based site rates...")
    entropy_rates = compute_entropy_rates(
        features_ali      = res[0][0],   # (n_sites, n_taxa*4) One-Hot
        n_taxa            = N_TAXA,
        normalize_to_mean= True,
    )

    # Optional: Korrelation zur Wahrheit ausgeben
    for name, er in entropy_rates.items():
        corr = np.corrcoef(true_site_rates, er)[0, 1]
        mse  = np.mean((true_site_rates - er) ** 2)
        print(f"  {name:15s}  corr={corr:.3f}  MSE={mse:.4f}")
    rnn_corr = np.corrcoef(true_site_rates, site_rates)[0, 1]
    rnn_mse  = np.mean((true_site_rates - site_rates) ** 2)
    print(f"  {'RNN':15s}  corr={rnn_corr:.3f}  MSE={rnn_mse:.4f}")

    # -------------------------------------------------------------------------
    # C) RevBayes-Skripte generieren
    # -------------------------------------------------------------------------

    # -- Gamma + Invariant (Baseline) ----------------------------------------
    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=None, gamma_model=True, inv_model=True,
                           prior_bl=16.)

    # -- RNN: kontinuierlich -------------------------------------------------
    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=site_rates, gamma_model=False,
                           prior_bl=16.)

    # -- RNN: diskretisiert (5 Klassen) --------------------------------------
    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=site_rates, gamma_model=False,
                           prior_bl=16., discretize_site_rate=5)

    # -- Shannon -------------------------------------------------------------
    pn.get_revBayes_script(ali_file,
                           ali_name + "_SH", ali_name + "_SH",
                           sr=entropy_rates['shannon'],
                           gamma_model=False, prior_bl=16.,
                           discretize_site_rate=5)

    # -- Normalisierter Shannon ----------------------------------------------
    pn.get_revBayes_script(ali_file,
                           ali_name + "_NSH", ali_name + "_NSH",
                           sr=entropy_rates['norm_shannon'],
                           gamma_model=False, prior_bl=16.,
                           discretize_site_rate=5)

    # -- Gini-Simpson --------------------------------------------------------
    pn.get_revBayes_script(ali_file,
                           ali_name + "_GS", ali_name + "_GS",
                           sr=entropy_rates['gini_simpson'],
                           gamma_model=False, prior_bl=16.,
                           discretize_site_rate=5)

    # -- Rényi Ordnung 2 -----------------------------------------------------
    pn.get_revBayes_script(ali_file,
                           ali_name + "_RE", ali_name + "_RE",
                           sr=entropy_rates['renyi2'],
                           gamma_model=False, prior_bl=16.,
                           discretize_site_rate=5)

    # -------------------------------------------------------------------------
    # D) Simulation speichern
    # -------------------------------------------------------------------------
    pn.save_pkl(res, ali_name + "_info.pkl")

    # -------------------------------------------------------------------------
    # E) Simulation ausführen
    # -------------------------------------------------------------------------

    print("\n" + "-"*60)
    print(f"Starte RevBayes für sim {sim_i}...")
    print("-"*60)

    base_sim_name = f"ali{sim_i}"

    for model_tag in MODEL_NAMES:
        if model_tag == "G":
            rev_filename = f"{base_sim_name}_G"
        elif model_tag == "DL":
            rev_filename = f"{base_sim_name}_DL"
        elif model_tag == "DL5d":
            rev_filename = f"{base_sim_name}_DL5d"
        else:  # Für SH, NSH, GS, RE
            rev_filename = f"{base_sim_name}_{model_tag}_DL5d"

        cmd = f"revbayes-v1.4.0/bin/rb phyloRNN/ali_tmp/{rev_filename}"
        print(f"  [Running] Modell {model_tag:4s} -> {cmd}")

        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"  [Success] Modell {model_tag} beendet.")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] RevBayes ist bei Modell {model_tag} fehlgeschlagen: {e}")
        except KeyboardInterrupt:
            print("\n[Abbruch] Pipeline wurde vom Nutzer unterbrochen.")
            break

    # -------------------------------------------------------------------------
    # F) AIC / BIC (nach RevBayes-Lauf)
    # -------------------------------------------------------------------------

    aic_bic_row = {"sim_i": sim_i}
    for model_tag in MODEL_NAMES:
        if model_tag in ("SH", "NSH", "GS", "RE"):
            log_file = ali_name + f"_{model_tag}_DL5d.log"
        else:
            log_file = ali_name + f"_{model_tag}.log"

        n_params = count_free_params(model_tag, N_TAXA)
        result   = compute_aic_bic(log_file, n_params, N_SITES)
        aic_bic_row[model_tag] = result

    all_aic_bic.append(aic_bic_row)


# ---------------------------------------------------------------------------
# 5. Zusammenfassung AIC/BIC (tabellarisch)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("AIC / BIC Zusammenfassung")
print("="*60)

summary_rows = []
for row in all_aic_bic:
    sim_i = row["sim_i"]
    for model_tag, res in row.items():
        if model_tag == "sim_i" or not isinstance(res, dict):
            continue
        summary_rows.append({
            "sim_i":    sim_i,
            "model":    model_tag,
            "n_params": res["n_params"],
            "mean_ll":  res["mean_ll"],
            "AIC":      res["AIC"],
            "BIC":      res["BIC"],
        })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
out_csv = os.path.join(data_wd, "aic_bic_summary.csv")
df_summary.to_csv(out_csv, index=False)
print(f"\nGespeichert: {out_csv}")