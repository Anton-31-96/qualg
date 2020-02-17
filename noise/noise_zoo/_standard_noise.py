import numpy as np
from projectq.ops import MatrixGate, X, Y, Z
from .._noiseoperator import NoiseKraus, NoiseKrausError

Id = MatrixGate(np.eye(2, dtype=complex))
I = Id


class DepolarizingChannel(NoiseKraus):

    def __init__(self, p):
        probs = []
        probs = [p/4] * 3
        probs.append(1 - 0.75 * p)
        NoiseKraus.__init__(self, kraus_ops=[X, Y, Z, Id], probs=probs)
        self._parameter = p

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, p):
        probs = []
        probs = [p/4] * 3
        probs.append(1 - 0.75 * p)
        NoiseKraus(kraus_ops=[X, Y, Z, Id])
        self._parameter = p
