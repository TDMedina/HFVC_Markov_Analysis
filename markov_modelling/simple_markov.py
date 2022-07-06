"""Simple Markov Model Training.

@author: T.D. Medina
"""

import numpy as np


def train_tpm(chains, n_states):
    tpm = np.zeros([n_states, n_states])
    init_dist = np.zeros(n_states)
    for chain in chains:
        init_dist[chain[0]-1] += 1
        for i, j in zip(chain, chain[1:]):
            tpm[i-1, j-1] += 1
    if not np.all(tpm[-1] == 0):
        print("Warning: Death transitions observed.")
    else:
        tpm[-1][-1] = 1
    tpm = tpm / tpm.sum(1).reshape([-1, 1])
    init_dist = init_dist / init_dist.sum()
    return init_dist, tpm
