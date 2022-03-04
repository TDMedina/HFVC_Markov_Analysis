from collections import namedtuple

from hmmlearn import hmm
import numpy as np


def make_hmmlearn_model(chains, n_components, tpm, epm, init_dist,
                        extend_death_state=0):
    if extend_death_state > 0:
        for i, chain in enumerate(chains):
            if chain[-1] == 2:
                chains[i] += [2]*extend_death_state
    chain_lens = [len(chain) for chain in chains]
    chain_array = [[val] for chain in chains for val in chain]
    model = hmm.MultinomialHMM(n_components, init_params="", n_iter=1000, tol=0.001)
    model.transmat_ = tpm
    model.emissionprob_ = epm
    model.startprob_ = init_dist
    model.fit(chain_array, chain_lens)
    return model


def make_random_init_params(n_hidden_states, n_obs_states):
    tpm = np.random.random([n_hidden_states, n_hidden_states])
    tpm = tpm/tpm.sum(1).reshape([-1, 1])

    epm = np.random.random([n_hidden_states, n_obs_states])
    epm = epm / epm.sum(1).reshape([-1, 1])

    sd = np.random.random([1, n_hidden_states])[0]
    sd = sd / sd.sum()

    params = {"tpm": tpm, "epm": epm, "stat_dist": sd}
    return params


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


def multichain_update_tpm(all_gammas, all_xis, n_states):
    tpm = np.nansum(all_xis, (0, 1))
    denom = np.array([0.] * n_states)
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
