from collections import namedtuple

import numpy as np
import random


test_obs_chain = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0])
multi_obs_chain = [[random.randint(0, 1) for _ in range(20)] for i in range(10000)]

test_tpm_init = np.array([[0.1, 0.9],
                          [0.4, 0.6]])
test_epm_init = np.array([[0.5, 0.5],
                          [0.9, 0.1]])
test_sd_init = np.array([0.5, 0.5])

params = {"observed_chain": test_obs_chain,
          "tpm": test_tpm_init,
          "epm": test_epm_init,
          "stat_dist": test_sd_init}

# def make_observed_chains_matrix(observed_chains):
#     max_len = max(len(chain) for chain in observed_chains)
#     obs_seq_matrix = np.array([chain + ([np.nan] * (max_len - len(chain)))
#                                for chain in observed_chains])
#     return obs_seq_matrix


def make_random_left_right_matrix(n_states):
    array = []
    i = n_states
    while i:
        array.append([0] * (n_states-i) + [1/i] * i)
        i -= 1
    array = np.array(array)
    return array


def make_min_backwards_matrix(n_states, backwards_rate=0.05):
    array = [[1/n_states]*n_states]
    i = n_states-1
    while i:
        back_rate = backwards_rate / (n_states-i)
        array.append([back_rate]*(n_states-i) + [(1-backwards_rate)/i]*i)
        i -= 1
    array = np.array(array)
    return array


def make_uniform_stochastic_matrix(rows, columns):
    array = [1/columns]*columns
    if rows == 1:
        array = np.array(array)
    else:
        array = np.array([array]*rows)
    return array

# def make_markov_model(chains, n_components, initial_tpm,
#                       extend_death_state=0):
#     chains = [chain for chain in chains if chain]
#     if extend_death_state > 0:
#         for i, chain in enumerate(chains):
#             if chain[-1] == 2:
#                 chains[i] += [2]*extend_death_state
#     chain_lens = [len(chain) for chain in chains]
#     chain_array = [[val] for chain in chains for val in chain]
#     model = hmm.MultinomialHMM(n_components, init_params="se")
#     model.transmat_ = initial_tpm
#     model.fit(chain_array, chain_lens)
#     return model


def calculate_alphas(observed_chain, tpm, epm, stat_dist):
    states = range(tpm.shape[0])
    alphas = [stat_dist * epm[:, observed_chain[0]]]
    for obs in observed_chain[1:]:
        t0 = alphas[-1]
        alphas.append(np.array([epm[i, obs] * (t0 @ tpm[:, i]) for i in states]))
    alphas = np.array(alphas).transpose()
    return alphas


def calculate_betas(observed_chain, tpm, epm):
    states = tpm.shape[0]
    betas = [np.array([1]*states)]
    for obs in reversed(observed_chain[1:]):
        t1 = betas[-1]
        t0 = np.array([sum(t1*tpm[i, :]*epm[:, obs]) for i in range(states)])
        betas.append(t0)
    betas.reverse()
    betas = np.array(betas).transpose()
    return betas


def calculate_gammas(alphas, betas):
    gammas = alphas * betas
    gammas = gammas / gammas.sum(0)
    return gammas


def calculate_xis(alphas, betas, observed_chain, tpm, epm):
    xis = [alpha.reshape(-1, 1) * beta * tpm * epm[:, obs]
           for alpha, beta, obs in zip(alphas.transpose(),
                                       betas.transpose()[1:],
                                       observed_chain[1:])]
    xis = np.array([xi / xi.sum() for xi in xis])
    return xis


def calculate_gammas_and_xis(observed_chain, tpm, epm, stat_dist):
    alphas = calculate_alphas(observed_chain, tpm, epm, stat_dist)
    betas = calculate_betas(observed_chain, tpm, epm)
    gammas = calculate_gammas(alphas, betas)
    xis = calculate_xis(alphas, betas, observed_chain, tpm, epm)
    return gammas, xis


def update_tpm(gammas, xis):
    tpm = xis.sum(0) / gammas[:, :-1].sum(1).reshape(-1, 1)
    return tpm


def update_epm(gammas, observed_chain, shape):
    updated_epm = np.zeros(shape)
    for obs, gamma in zip(observed_chain, gammas.transpose()):
        updated_epm[:, obs] += gamma
    updated_epm = updated_epm / gammas.sum(1).reshape(shape[0], 1)
    return updated_epm


def baum_welch(observed_chain, tpm, epm, stat_dist, max_iterations=1000, atol=1e-3):
    for i in range(1, max_iterations+1):
        gammas, xis = calculate_gammas_and_xis(observed_chain, tpm, epm, stat_dist)
        new_stat_dist = gammas[:, 0]
        new_tpm = update_tpm(gammas, xis)
        new_epm = update_epm(gammas, observed_chain, epm.shape)

        if all([np.allclose(stat_dist, new_stat_dist, atol=atol),
                np.allclose(tpm, new_tpm, atol=atol),
                np.allclose(epm, new_epm, atol=atol)]):
            print(f"Converged in {i} steps.")
            break

        stat_dist = new_stat_dist
        tpm = new_tpm
        epm = new_epm
    else:
        print(f"Max iterations ({max_iterations}) reached without convergence.")
    return stat_dist, tpm, epm


# %% Multichain
def make_gamma_matrix(all_gammas, max_len):
    all_gammas = np.array([
        np.pad(array=gammas,
               pad_width=[(0, 0), (0, max_len - gammas.shape[1])],
               constant_values=np.nan)
        for gammas in all_gammas
        ])
    return all_gammas


def make_xi_matrix(all_xis, max_len):
    max_len = max_len-1
    all_xis = np.array([
        np.pad(array=xis,
               pad_width=[(0, max_len - xis.shape[0]), (0, 0), (0, 0)],
               constant_values=np.nan)
        for xis in all_xis if xis.size > 0
        ])
    return all_xis


# def multichain_update_tpm(all_gammas, all_xis, n_states):
#     tpm = np.nansum(all_xis, (0, 1))
#     denom = np.nansum(all_gammas[:, :, :-1], (0, 2))
#     tpm = tpm / denom.reshape(-1, 1)
#     return tpm

def multichain_update_tpm(all_gammas, all_xis, n_states):
    tpm = np.nansum(all_xis, (0, 1))
    denom = np.array([0., 0.])
    for gammas in all_gammas:
        denom += gammas[~np.isnan(gammas)].reshape(n_states, -1)[:, :-1].sum(1)
    # denom = np.nansum(all_gammas[:, :, :-1], (0, 2))
    tpm = tpm / denom.reshape(-1, 1)
    return tpm


def multichain_update_epm(all_gammas, observed_chains, shape):
    epm = np.zeros(shape)
    for observed_chain, gammas in zip(observed_chains, all_gammas):
        for obs, gamma in zip(observed_chain, gammas.transpose()):
            epm[:, obs] += gamma
    gamma_sum = np.nansum(all_gammas, (0, 2))
    epm = epm / gamma_sum.reshape(shape[0], 1)
    return epm


def multichain_baum_welch(observed_chains, tpm, epm, stat_dist,
                          max_iterations=1000, atol=1e-3):
    # Setup parameters.
    num_chains = len(observed_chains)
    n_states = tpm.shape[0]
    max_len = max((len(chain) for chain in observed_chains))

    # Iterate.
    for i in range(1, max_iterations+1):
        print(f"Iteration: {i}")
        all_gammas, all_xis = zip(*[calculate_gammas_and_xis(chain, tpm, epm, stat_dist)
                                    for chain in observed_chains])
        all_gammas = make_gamma_matrix(all_gammas, max_len)
        all_xis = make_xi_matrix(all_xis, max_len)

        new_stat_dist = all_gammas[:, :, 0].sum(0) / num_chains
        new_tpm = multichain_update_tpm(all_gammas, all_xis, n_states)
        new_epm = multichain_update_epm(all_gammas, observed_chains, epm.shape)

        # Check for convergence.
        if all([np.allclose(stat_dist, new_stat_dist, atol=atol),
                np.allclose(tpm, new_tpm, atol=atol),
                np.allclose(epm, new_epm, atol=atol)]):
            print(f"Converged in {i} steps.")
            break

        # Assign new parameters.
        stat_dist = new_stat_dist
        tpm = new_tpm
        epm = new_epm

    # Notify if no convergence.
    else:
        print(f"Max iterations ({max_iterations}) reached without convergence.")
    return stat_dist, tpm, epm


TestModel = namedtuple("TestModel", ["alphas", "betas", "gammas", "xis", "single_bw", "multi_bw"])


# def parallel_multichain_baum_welch(observed_chains, tpm, epm, stat_dist, iterations=100):
#     cpu_count = mp.cpu_count()
#     num_chains = len(observed_chains)
#     for _ in range(iterations):
#         mp.Pool(cpu_count).starmap(calculate_gamma_xi,
#                                    zip(observed_chains,
#                                        [tpm]*num_chains,
#                                        [epm]*num_chains,
#                                        [stat_dist]*num_chains))


# if __name__ == "__main__":
#     alpha_jenny = bw_alpha_gen(test_obs_chain, test_tpm_init, test_epm_init, test_sd_init)
#     beta_jenny = bw_beta_gen(test_obs_chain, test_tpm_init, test_epm_init)

# def forward(V, a, b, initial_distribution):
#     alpha = np.zeros((V.shape[0], a.shape[0]))
#     alpha[0, :] = initial_distribution * b[:, V[0]]
#
#     for t in range(1, V.shape[0]):
#         for j in range(a.shape[0]):
#             # Matrix Computation Steps
#             #                  ((1x2) . (1x2))      *     (1)
#             #                        (1)            *     (1)
#             alpha[t, j] = alpha[t - 1] @ a[:, j] * b[j, V[t]]
#
#     return alpha
#
# def baum_welch(O, a, b, initial_distribution, n_iter=100):
#     #http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/
#     M = a.shape[0]
#     T = len(O)
#     for n in range(n_iter):
#         ###estimation step
#         alpha = forward(O, a, b, initial_distribution)
#         beta = backward(O, a, b)
#         xi = np.zeros((M, M, T - 1))
#         for t in range(T - 1):
#             # joint probab of observed data up to time t @ transition prob *
#             #emisssion prob at t+1 @ joint probab of observed data from at t+1
#             denominator = (alpha[t, :].T @ a * b[:, O[t + 1]].T) @ beta[t + 1, :]
#             for i in range(M):
#                 numerator = alpha[t, i] * a[i, :] * b[:, O[t + 1]].T * beta[t + 1, :].T
#                 xi[i, :, t] = numerator / denominator
#         gamma = np.sum(xi, axis=1)
#         ### maximization step
#         a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
#         # Add additional T'th element in gamma
#         gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
#         K = b.shape[1]
#         denominator = np.sum(gamma, axis=1)
#         for l in range(K):
#             b[:, l] = np.sum(gamma[:, O == l], axis=1)
#         b = np.divide(b, denominator.reshape((-1, 1)))
#     return a, b
