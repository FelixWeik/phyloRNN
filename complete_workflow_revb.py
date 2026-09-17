import os
import subprocess
import numpy as np
import pandas as pd
import scipy.linalg
import dendropy
import phyloRNN as pn
from matplotlib import pyplot as plt
wd = os.getcwd()

revbayes_bin = os.path.join(wd, "revbayes-v1.4.0", "bin", "rb")
# rank_script = os.path.join(wd, "4.rank_true_tree_likelihood.py")


def run_revbayes(script_file):
    print(f"Running RevBayes on {script_file}")
    subprocess.run([revbayes_bin, script_file], check=True)

def rank_likelihood(log_file, likelihood):
    try:
        df = pd.read_csv(log_file, sep="\t", comment="#")
        likelihoods = df["Likelihood"].to_numpy(dtype=float)
    except:
        return [], 0

    combined = np.append(likelihoods, likelihood)
    order = np.argsort(combined)[::-1]  # indices into combined, decreasing likelihood
    sorted_likelihoods = combined[order]

    input_pos_in_combined = len(combined) - 1  # the appended value
    index = int(np.where(order == input_pos_in_combined)[0][0])

    return sorted_likelihoods, index

def ex_rev(log_file, likelihood):
    sorted_likelihoods, index = rank_likelihood(log_file, likelihood)
    
    print("Likelihood of true tree: ", likelihood)
    print("Sorted likelihoods (decreasing):")
    print(sorted_likelihoods)
    print("Index of input likelihood in sorted array:", index)


def sitewise_shannon_entropy(nex_file):
    """Shannon entropy (nats) of the nucleotide distribution at each
    alignment site, computed across taxa from the given .nex file."""
    dna = dendropy.DnaCharacterMatrix.get(path=nex_file, schema="nexus")
    bases = ["A", "C", "G", "T"]
    alignment = np.array([seq.symbols_as_list() for seq in dna.values()])  # (n_taxa, n_sites)

    n_sites = alignment.shape[1]
    entropy = np.zeros(n_sites)
    for i in range(n_sites):
        counts = np.array([np.sum(alignment[:, i] == b) for b in bases])
        total = counts.sum()
        if total == 0:
            continue
        p = counts[counts > 0] / total
        entropy[i] = -np.sum(p * np.log(p))
    return entropy


def build_substitution_rate_matrix(info):
    """Instantaneous rate matrix Q (states A,C,G,T) and stationary frequencies
    matching the model/parameters used by seq-gen at simulation time
    (see phyloRNN.simulate.simulator.run_sim), normalized to one expected
    substitution per unit branch length."""
    model_indx = info['model_indx']
    # exchangeability order: A-C, A-G, A-T, C-G, C-T, G-T
    if model_indx == 0:  # JC69
        pi = np.array([0.25, 0.25, 0.25, 0.25])
        exch = np.ones(6)
    elif model_indx == 1:  # HKY85
        pi = np.array(info['freq'])
        ti_tv = info['ti_tv']
        kappa = ti_tv * (pi[0] * pi[2] + pi[1] * pi[3]) / ((pi[0] + pi[2]) * (pi[1] + pi[3]))
        exch = np.array([1., kappa, 1., 1., kappa, 1.])
    else:  # GTR
        pi = np.array(info['freq'])
        exch = np.array(info['rates'])

    pair_idx = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    Q = np.zeros((4, 4))
    for (i, j), r in zip(pair_idx, exch):
        Q[i, j] = r * pi[j]
        Q[j, i] = r * pi[i]
    np.fill_diagonal(Q, -Q.sum(axis=1))

    mu = -np.sum(pi * np.diag(Q))  # expected rate per unit branch length
    Q = Q / mu
    return Q, pi


def read_alignment_dict(nex_file):
    dna = dendropy.DnaCharacterMatrix.get(path=nex_file, schema="nexus")
    return {taxon.label: seq.symbols_as_list() for taxon, seq in dna.items()}


def true_tree_log_likelihood(tree_file, nex_file, site_rates, info):
    """Phylogenetic log-likelihood (Felsenstein pruning) of the true
    (generating) tree and topology, given per-site rate multipliers and the
    substitution model/parameters used at simulation time."""
    Q, pi = build_substitution_rate_matrix(info)
    tree = dendropy.Tree.get(path=tree_file, schema="newick")
    alignment = read_alignment_dict(nex_file)
    base_index = {"A": 0, "C": 1, "G": 2, "T": 3}

    n_sites = len(site_rates)
    total_logL = 0.0
    for s in range(n_sites):
        cond = {}
        for nd in tree.postorder_node_iter():
            if nd.is_leaf():
                base = alignment[nd.taxon.label][s].upper()
                vec = np.zeros(4)
                if base in base_index:
                    vec[base_index[base]] = 1.0
                else:
                    vec[:] = 1.0  # gap / ambiguous: marginalize over states
                cond[nd] = vec
            else:
                vec = np.ones(4)
                for child in nd.child_nodes():
                    bl = child.edge.length or 0.0
                    P = scipy.linalg.expm(Q * site_rates[s] * bl)
                    vec = vec * (P @ cond[child])
                cond[nd] = vec
        site_L = np.sum(pi * cond[tree.seed_node])
        total_logL += np.log(site_L)
    return total_logL


data_wd = os.path.join(os.getcwd(), "phyloRNN", "ali_tmp")
model_name = "t20_s100"
trained_model = pn.load_rnn_model(os.path.join(wd, "Trained_models", model_name))

plot = False
log_rates = False
start_sim = 0
n_sim = 1

# simulate data

sim = pn.simulator(n_taxa = 20, 
                   n_sites = 100,
                   n_eigen_features = 3,
                   min_rate = 0,  #
                   freq_uncorrelated_sites = 0.5,
                   freq_mixed_models = 0,
                   store_mixed_model_info = True,
                   tree_builder = 'nj',  # 'upgma'
                   subs_model_per_block = False,  # if false same subs model for all blocks
                   phyml_path = None, #os.path.join(os.getcwd(), "phyloRNN"),
                   seqgen_path = None, # os.path.join(os.getcwd(), "phyloRNN", "seq-gen")
                   ali_path = data_wd,
                   DEBUG=False,
                   verbose = True,
                   ali_schema = "nexus", # phylip,
                   min_avg_br_length=0.01,
                   max_avg_br_length=0.2
                   )

# run simulations sethm
for sim_i in range(start_sim, start_sim + n_sim):
    ali_name = os.path.join(data_wd, "ali%s" % sim_i)
    res = sim.run_sim([sim_i, 1,
                 ali_name,
                 False])

    ali_file = res[-1][0]['ali_file']

    # get features for rnn predictions
    true_site_rates = res[2][0]
    sim_res = {'features_ali': res[0][0],
               'labels_rates': true_site_rates,
               'labels_smodel': None,
                'labels_tl': res[-2][0]
               }

    # create input
    (comp_sim, dict_inputs, comp_dict_outputs
     ) = pn.rnn_in_out_dictionaries_from_sim(sim=sim_res,
                                          log_rates=log_rates,
                                          output_list=['per_site_rate','tree_len'],
                                          include_tree_features=False)



    print("Running predictions...")
    model_input = {'sequence_data': dict_inputs['sequence_data'].numpy().reshape((1,
                                        dict_inputs['sequence_data'].shape[0],
                                        dict_inputs['sequence_data'].shape[1]))
            }
    predictions = trained_model.predict(model_input)

    site_rates = predictions[0][0]

    # likelihood of the true (generating) tree given the predicted rates
    # and the substitution model/parameters used to simulate the data
    true_tree_file = ali_name + "_true.tre"
    logL_predicted = true_tree_log_likelihood(true_tree_file, ali_file, site_rates, res[-1][0])
    logL_true = true_tree_log_likelihood(true_tree_file, ali_file, true_site_rates, res[-1][0])
    print("log-likelihood of true tree | predicted rates:", logL_predicted)
    print("log-likelihood of true tree | true rates:", logL_true)

    if plot:
        print("MSE:", np.mean((true_site_rates - site_rates)**2))
        plt.scatter(true_site_rates, site_rates)
        plt.show()
        print(res[-1][0]['rate_het_model'])

    # sitewise Shannon entropy computed from the generated alignment
    site_entropy = np.abs(sitewise_shannon_entropy(ali_file))
    norm_site_entropy = np.abs(site_entropy / np.mean(site_entropy))
    logL_shannon = true_tree_log_likelihood(true_tree_file, ali_file, site_entropy, res[-1][0])
    logL_nsh = true_tree_log_likelihood(true_tree_file, ali_file, norm_site_entropy, res[-1][0])
    print("log-likelihood of true tree | shannon:", logL_shannon)
    print("log-likelihood of true tree | normalized shannon:", logL_nsh)

    # discretized predicted rates, exactly as used in the DL5d RevBayes script
    discrete_rates, rate_indx = pn.get_discretized_site_rates(site_rates, ncat=5, log_rates=True)
    discretized_site_rates = discrete_rates[rate_indx]
    logL_discretized = true_tree_log_likelihood(true_tree_file, ali_file, discretized_site_rates, res[-1][0])
    print("log-likelihood of true tree | discretized predicted rates:", logL_discretized)

    # final script paths, mirroring the suffix logic in pn.get_revBayes_script
    script_G = ali_name + "_G"
    script_DL = ali_name + "_DL"
    script_DL5d = ali_name + "_DL5d"
    script_SH = ali_name + "_SH_DL"
    script_NSH = ali_name + "_NSH_DL"

    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=None, gamma_model=True, inv_model=True,
                           prior_bl=16.)

    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=site_rates, gamma_model=False,
                           prior_bl=16.)

    pn.get_revBayes_script(ali_file, ali_name, ali_name,
                           sr=site_rates, gamma_model=False,
                           prior_bl=16.,
                           discretize_site_rate=5)

    pn.get_revBayes_script(ali_file, ali_name + "_SH", ali_name + "_SH",
                           sr=site_entropy, gamma_model=False,
                           prior_bl=16.)

    pn.get_revBayes_script(ali_file, ali_name + "_NSH", ali_name + "_NSH",
                           sr=norm_site_entropy, gamma_model=False,
                           prior_bl=16.)

    pn.save_pkl(res, ali_name + "_info.pkl")

    # run RevBayes on every generated script
    revbayes_success = {}
    for name, script in [("G", script_G), ("DL", script_DL), 
                         ("DL5d", script_DL5d), ("SH", script_SH), 
                         ("NSH", script_NSH)]:
        revbayes_success[name] = run_revbayes(script)
    
    # Only rank models that successfully generated log files
    if revbayes_success["DL"] and os.path.exists(script_DL + ".log"):
        ex_rev(script_DL + ".log", logL_predicted)
    else:
        print(f"Skipping DL ranking - log file missing: {script_DL}.log")
    
    if revbayes_success["DL5d"] and os.path.exists(script_DL5d + ".log"):
        ex_rev(script_DL5d + ".log", logL_discretized)
    else:
        print(f"Skipping DL5d ranking - log file missing: {script_DL5d}.log")
    
    if revbayes_success["SH"] and os.path.exists(script_SH + ".log"):
        ex_rev(script_SH + ".log", logL_shannon)
    else:
        print(f"Skipping SH ranking - log file missing: {script_SH}.log")
    
    if revbayes_success["NSH"] and os.path.exists(script_NSH + ".log"):
        ex_rev(script_NSH + ".log", logL_nsh)
    else:
        print(f"Skipping NSH ranking - log file missing: {script_NSH}.log")

    # rank the true tree's likelihood (under each model's rates) against
    # that model's posterior Likelihood trace
    ex_rev(script_DL + ".log", logL_predicted)
    ex_rev(script_DL5d + ".log", logL_discretized)
    ex_rev(script_SH + ".log", logL_shannon)
    ex_rev(script_NSH + ".log", logL_nsh)

