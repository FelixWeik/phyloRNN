import phyloRNN as pn

n_sims = 5

# Step 1: Simulate a test set with phyML running
sim = pn.simulator(
    n_taxa=20, n_sites=100,
    freq_uncorrelated_sites=0.5,
    freq_mixed_models=0.05,
    phyml_path=None,  # None as installed system wide
)
sim.reset_prms(CPUs=2, n_sims=n_sims, data_name="test_data",
               run_phyml=True, base_seed=4321)
pn.simulate_parallel(sim, add_day_tag=False)

# Step 2: Train a model on a training set (separate simulation)
# ... (use train_model.py workflow) ...

# Step 3: Run the comparison
m, preds, comp_sim, inputs, outputs, results = pn.compare_rnn_phyml(
    compare_file="test_data.npz",
    model_file="t20_s100",
    model_wd="Trained_models",
    output_dir="results/9",
    rnn_model_tag="",
    n_taxa=20,
    log_rates=False,
    log_tl=True,
    parse_heterogeneity_models=True,  # breakdown by heterogeneity model
    plot_results=True,
    plot_n_sim = n_sims
)

# Step 4: Access the results
print(results['rates-all'])  # MSE/R² for Gamma, FreeRate, RNN