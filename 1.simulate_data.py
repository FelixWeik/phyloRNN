import phyloRNN as pn


sim = pn.simulator(
                   n_taxa = 20,
                   n_sites = 100,
                   n_eigen_features = 3,
                   min_rate = 0,  #
                   freq_uncorrelated_sites = 0.5,
                   freq_mixed_models = 0.05,
                   store_mixed_model_info = True,
                   tree_builder = 'nj',  # 'upgma'
                   subs_model_per_block = False,  # if false same subs model for all blocks
                   phyml_path = None,  # path to phyml and seq binaries
                   seqgen_path = None, # if None, it will try to use
                   ali_path = None,                                            # system-wide installed software
                   DEBUG=False,
                   verbose = False,
                   )


if __name__ == '__main__':
    # training set
    sim.reset_prms(CPUs = 2,
                   n_sims = 10,
                   data_name = "training_data",
                   base_seed = 1234)
    pn.simulate_parallel(sim, add_day_tag=False)

    # test set
    sim.reset_prms(CPUs = 1,
                   n_sims = 2,
                   data_name = "test_data",
                   run_phyml = True,
                   base_seed = 4321)
    pn.simulate_parallel(sim, add_day_tag=False)
