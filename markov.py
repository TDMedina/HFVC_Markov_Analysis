#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heart Failure Virtual Consultation - Markov Processes.

Created on Mon Jan 10 16:25:26 2022

@author: T.D. Medina
"""

import numpy as np
from numpy import matmul
from numpy.linalg import eig, matrix_power
import pandas as pd

class Markov:
    def __init__(self, tpm=None, path=None, delimiter=","):
        if tpm is not None:
            self.tpm = np.asarray(tpm)
        elif path is not None:
            self.tpm = self._read_from_csv(path, delimiter=delimiter)
        if not self._is_stochastic():
            raise ValueError("Matrix is not stochastic.")

    def _is_square(self):
        shape = self.tpm.shape
        if not isinstance(shape, tuple):
            return False
        if not len(shape) == 2:
            return False
        if not shape[0] == shape[1]:
            return False
        return True

    def _is_stochastic(self):
        if self._is_square():
            return all([rowsum == 1 for rowsum in self.tpm.sum(1)])
        return False

    @staticmethod
    def _read_from_csv(path, delimiter):
        with open(path) as infile:
            matrix = pd.read_csv(path, delimiter=delimiter, header=None)
        matrix = np.asarray(matrix)
        return matrix

    def calculate_stationary_dist(self):
        eig_vals, eig_vecs = eig(self.tpm.transpose())
        print(eig_vals)
        chosen = []
        for i, val in enumerate(eig_vals):
            val = np.real_if_close([val])[0]
            print(val)
            if not val == 1:
                continue
            chosen.append(val,
                          np.real_if_close(eig_vecs[:, i]/eig_vecs[0, i]))
        return chosen


class Chain:
    def __init__(self, ID, year_states):
        self.id = ID
        self.year_states = year_states

if __name__ =="__main__":
    # tester = Markov._read_from_csv(path="../Project_Files/test_tpm.csv", delimiter="\t")
    markov = Markov(path="../Project_Files/test_tpm.csv", delimiter="\t")
