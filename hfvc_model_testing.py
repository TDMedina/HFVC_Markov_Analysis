"""HFVC - Model Testing and Comparisons"""

from collections import namedtuple

import numpy as np

import hfvc
import hfvc_hmm


EmitPrediction = namedtuple("EmitPrediction",
                            ["probabilities", "score", "probability_ratios"])


def split_test_and_train(chains, proportion=10):
    n_chains = len(chains)
    divider = n_chains // proportion
    shuffle = np.random.permutation(range(n_chains))
    train = [chains[i] for i in shuffle[divider:]]
    test = [chains[i] for i in shuffle[:divider]]
    test = [(chain[:-1], chain[-1]) for chain in test]
    return train, test


def predict_next_emission(chain, init_dist, tpm, epm):
    alpha = hfvc_hmm.calculate_alphas(chain, tpm, epm, init_dist)[:, -1]
    alpha = alpha / alpha.sum()
    margins = tpm @ epm
    probs = alpha.reshape(-1, 1) * margins
    probs = probs.sum(0)

    ranked = sorted(probs)
    prob_ratios = [prob / ranked[0] for prob in probs]
    score = ranked[-1] / ranked[-2]

    prediction = EmitPrediction(probs, score, prob_ratios)
    return prediction


def test_hmm_model(test_chains, model):
    results = []
    for chain, answer in test_chains:
        prediction = predict_next_emission(chain, *model)
        predicted_emit = np.argmax(prediction.probabilities)
        results.append([predicted_emit == answer, predicted_emit, prediction])
    score = sum([result[0] for result in results]) / len(results)
    return score, results


def build_and_test_model(data: hfvc.PatientDatabase, n_components, interval):
    train, test = split_test_and_train(data.make_admission_chains(interval, True, 2))
    init_params = hfvc_hmm.make_random_init_params(n_components, 3)
    model = hfvc_hmm.multichain_baum_welch(train, **init_params)
    score, results = test_hmm_model(test, model)
    return model, score, results, train, test


if __name__ == "__main__":
    data = hfvc.main()
    model, score, results, train, test = build_and_test_model(data, 5, "y")
