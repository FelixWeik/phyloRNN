"""
Ranks a given (true-tree) likelihood against the "Likelihood" trace of a
RevBayes .log file (as found in phyloRNN/ali_tmp), sorted in decreasing order.

Usage:
    python 4.rank_true_tree_likelihood.py <log_file> <likelihood>
"""
import argparse
import numpy as np
import pandas as pd


def rank_likelihood(log_file, likelihood):
    df = pd.read_csv(log_file, sep="\t", comment="#")
    likelihoods = df["Likelihood"].to_numpy(dtype=float)

    combined = np.append(likelihoods, likelihood)
    order = np.argsort(combined)[::-1]  # indices into combined, decreasing likelihood
    sorted_likelihoods = combined[order]

    input_pos_in_combined = len(combined) - 1  # the appended value
    index = int(np.where(order == input_pos_in_combined)[0][0])

    return sorted_likelihoods, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rank a given likelihood value against the Likelihood "
                     "column of a RevBayes .log file, sorted in decreasing order.")
    parser.add_argument("log_file", help="Path to the .log file (e.g. in phyloRNN/ali_tmp)")
    parser.add_argument("likelihood", type=float, help="Likelihood value to rank (e.g. of the true tree)")
    args = parser.parse_args()

    sorted_likelihoods, index = rank_likelihood(args.log_file, args.likelihood)

    print("Sorted likelihoods (decreasing):")
    print(sorted_likelihoods)
    print("Index of input likelihood in sorted array:", index)
