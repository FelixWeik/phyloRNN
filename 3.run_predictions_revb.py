import os
import numpy as np
import dendropy
import phyloRNN as pn
wd = os.getcwd()

revbayes_bin = os.path.join(wd, "revbayes-v1.4.0", "bin", "rb")
rank_script = os.path.join(wd, "4.rank_true_tree_likelihood.py")


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

def read_alignment_dict(nex_file):
    dna = dendropy.DnaCharacterMatrix.get(path=nex_file, schema="nexus")
    return {taxon.label: seq.symbols_as_list() for taxon, seq in dna.items()}

data_wd = os.path.join(os.getcwd(), "phyloRNN", "ali_tmp")
# training_file = os.path.join(os.getcwd(), "training_data.npz")
model_name = "t50_s1000"
trained_model = pn.load_rnn_model(os.path.join(wd, "Trained_models", model_name))

plot = False
log_rates = False
start_sim = 0
n_sim = 1

# simulate data

sim = pn.simulator(n_taxa = 50, 
                   n_sites = 1000,
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

# run simulations set
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

    # sitewise Shannon entropy computed from the generated alignment
    site_entropy = np.abs(sitewise_shannon_entropy(ali_file))
    norm_site_entropy = np.abs(site_entropy / np.mean(site_entropy))

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

