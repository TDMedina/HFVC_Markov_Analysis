import numpy as np
import multiprocessing as mp
import random


test_obs_chain = [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]
multi_obs_chain = [[random.randint(0, 1) for _ in range(10)] for i in range(100)]

test_tpm_init = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
test_epm_init = np.array([[0.5, 0.5],
                          [0.5, 0.5]])
test_sd_init = np.array([0.5, 0.5])

params = {"observed_chain": test_obs_chain,
          "tpm": test_tpm_init,
          "epm": test_epm_init,
          "stat_dist": test_sd_init}


# def bw_alpha_gen(observed_chain, tpm_init, epm_init, stat_dist_init):
#     states = range(tpm_init.shape[0])
#     t0 = stat_dist_init * epm_init[:, observed_chain[0]].transpose()
#     yield t0
#     for obs in observed_chain[1:]:
#         t1 = np.array([epm_init[i, obs]*(t0 @ tpm_init[:, i]) for i in states])
#         yield t1
#         t0 = t1
#
#
# def bw_beta_gen(observed_chain, tpm_init, epm_init, stat_dist_init=None):
#     states = range(tpm_init.shape[0])
#     t1 = np.array([1, 1])
#     yield t1
#     for obs in reversed(observed_chain[:-1]):
#         t0 = np.array([sum(t1*tpm_init[i, :]*epm_init[:, obs]) for i in states])
#         yield t0
#         t1 = t0
#
#
# # def bw_gamma(alpha_vector_t, beta_vector_t):
# #     alphabeta = alpha_vector_t*beta_vector_t
# #     denom = sum(alphabeta)
# #     gamma = np.array([x/denom for x in alphabeta])
# #     return gamma
#
#
# def bw_gamma_gen(observed_chain=None, tpm_init=None, epm_init=None, stat_dist_init=None,
#                  alphas=None, betas=None):
#     constants = [observed_chain, tpm_init, epm_init, stat_dist_init]
#     if alphas is not None and betas is not None:
#         betas = list(betas)
#     elif None not in constants:
#         alphas = bw_alpha_gen(*constants)
#         betas = list(bw_beta_gen(*constants))
#     else:
#         raise ValueError("Missing required argument in constants.")
#     for alpha, beta in zip(alphas, betas):
#         gamma = alpha * beta
#         gamma = gamma / sum(gamma)
#         yield gamma
#
#
# def bw_xi_gen(observed_chain, tpm_init, epm_init,
#               alphas=None, betas=None, stat_dist_init=None):
#     if alphas is None:
#         alphas = bw_alpha_gen(observed_chain, tpm_init, epm_init, stat_dist_init)
#     if betas is None:
#         betas = list(bw_beta_gen(observed_chain, tpm_init, epm_init))
#
#
# def bw_xi(alpha_vector_t, beta_vector_t_plus_1, tpm_init, epm_init, obs_state):
#     states = range(tpm_init.shape[0])
#     # xi = np.ndarray([states, states])
#     # for i in states:
#     #     for j in states:
#     #         xi[i,j] = (alpha_vector_t[i]*tpm_init[i,j]*beta_vector_t_plus_1[j]
#     #                    *epm_init[j][obs_state])
#     xi = np.reshape([alpha_vector_t[i]*tpm_init[i,j]*beta_vector_t_plus_1[j]
#                      *epm_init[j][obs_state] for i in states for j in states],
#                     tpm_init.shape)
#     xi = xi / xi.sum()
#     return xi


def calculate_alphas(observed_chain, tpm, epm, stat_dist):
    states = range(tpm.shape[0])
    alphas = [stat_dist * epm[:, observed_chain[0]]]
    for obs in observed_chain[1:]:
        t0 = alphas[-1]
        alphas.append(np.array([epm[i, obs] * (t0 @ tpm[:, i]) for i in states]))
    return alphas


def calculate_betas(observed_chain, tpm, epm, stat_dist=None):
    states = range(tpm.shape[0])
    betas = [np.array([1, 1])]
    for obs in reversed(observed_chain[:-1]):
        t1 = betas[-1]
        t0 = np.array([sum(t1*tpm[i, :]*epm[:, obs]) for i in states])
        betas.append(t0)
    betas.reverse()
    return betas


def calculate_gammas(alphas, betas):
    gammas = [alpha * beta for alpha, beta in zip(alphas, betas)]
    sum_gammas = sum(gammas)
    gammas = [gamma / sum_gammas for gamma in gammas]
    return gammas


def calculate_xis(alphas, betas, tpm, epm, observed_chain):
    size = tpm.shape[0]
    xis = [np.array([alpha]*size).transpose()
           *np.array([beta]*size)
           *np.array([epm[:,obs]]*size)
           *tpm
           for alpha, beta, obs in zip(alphas, betas[1:], observed_chain[1:])]
    xis = [xi / xi.sum() for xi in xis]
    return xis


def calculate_xis2(alphas, betas, tpm, epm, observed_chain):
    size = tpm.shape[0]
    xis = [alpha.reshape(size, 1) * beta * epm[:, obs] * tpm
           for alpha, beta, obs in zip(alphas, betas[1:], observed_chain[1:])]
    xis = [xi / xi.sum() for xi in xis]
    return xis


def calculate_gammas_and_xis(observed_chain, tpm, epm, stat_dist):
    alphas = calculate_alphas(observed_chain, tpm, epm, stat_dist)
    betas = calculate_betas(observed_chain, tpm, epm)
    gammas = calculate_gammas(alphas, betas)
    xis = calculate_xis2(alphas, betas, tpm, epm, observed_chain)
    return gammas, xis

def calculate_gammas_and_xis2(observed_chain, tpm, epm, stat_dist):
    alphas = calculate_alphas(observed_chain, tpm, epm, stat_dist)
    betas = calculate_betas(observed_chain, tpm, epm)
    gammas = calculate_gammas(alphas, betas)
    xis = calculate_xis(alphas, betas, tpm, epm, observed_chain)
    return gammas, xis

def update_tpm(xis, gammas):
    size = len(gammas[0])
    tpm = sum(xis) / np.array([sum(gammas[:-1])]*size).transpose()
    return tpm

def update_tpm2(gammas, xis):
    size = len(gammas[0])
    tpm = sum(xis) / sum(gammas[:-1]).reshape(size, 1)
    return tpm

def update_epm(gammas, observed_chain):
    size = len(gammas[0])
    denom = np.array([sum(gammas)]*size).transpose()
    epm = [np.array([0, 0]), np.array([0, 0])]
    for obs, gamma in zip(observed_chain, gammas):
        epm[obs] = epm[obs] + gamma
    epm = np.array(epm).transpose()
    epm = epm / denom
    return epm

def update_epm2(gammas, observed_chain):
    size = len(gammas[0])
    denom = sum(gammas).reshape(size, 1)
    epm = [np.array([0, 0]), np.array([0, 0])]
    for obs, gamma in zip(observed_chain, gammas):
        epm[obs] = epm[obs] + gamma
    epm = np.array(epm).transpose()
    epm = epm / denom
    return epm


def baum_welch(observed_chain, tpm, epm, stat_dist, iterations=100):
    for _ in range(iterations):
        alphas = calculate_alphas(observed_chain, tpm, epm, stat_dist)
        betas = calculate_betas(observed_chain, tpm, epm)
        gammas = calculate_gammas(alphas, betas)
        xis = calculate_xis(alphas, betas, tpm, epm, observed_chain)

        stat_dist = gammas[0]
        tpm = update_tpm(xis, gammas)
        epm = update_epm(gammas, observed_chain)
    return tpm, epm, stat_dist


def baum_welch2(observed_chain, tpm, epm, stat_dist, iterations=100):
    for _ in range(iterations):
        gammas, xis = calculate_gammas_and_xis2(observed_chain, tpm, epm, stat_dist)
        stat_dist = gammas[0]
        tpm = update_tpm2(gammas, xis)
        epm = update_epm2(gammas, observed_chain)
    return tpm, epm, stat_dist

def multichain_update_tpm(all_gammas, all_xis):
    size = len(all_gammas[0][0])
    tpm = np.array(
        [sum([gamma for gammas in all_gammas for gamma in gammas[:-1]])]*size
        ).transpose()
    return tpm


def multichain_baum_welch(observed_chains, tpm, epm, stat_dist, iterations=100):
    num_chains = len(observed_chains)
    size = tpm.shape[0]
    for _ in range(iterations):
        all_gammas, all_xis = zip(*[calculate_gammas_and_xis(chain, tpm, epm, stat_dist)
                                    for chain in observed_chains])
        stat_dist = sum([gammas[0] for gammas in all_gammas]) / num_chains
        gamma_sum = sum([gamma for gammas in all_gammas for gamma in gammas[:-1]])
        gamma_sum = np.array([gamma_sum]*size).transpose()
        tpm = sum([xi for xis in all_xis for xi in xis]) / gamma_sum


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
