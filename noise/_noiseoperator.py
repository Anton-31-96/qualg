import numpy as np
from projectq.ops import BasicGate, MatrixGate


class NoiseKrausError(Exception):
    pass


class NoiseKraus(object):

    _tol = 10e-10

    def __init__(self, kraus_ops, probs=None, name=None):
        """
        Arguments:
            kraus_ops (list): list of BasicGate objects. kraus_ops forms a
                        quantum CPTP map.
            probs (list): list of probabilities of every kraus_op to be used
                        during the simulation run. If not provided, they are
                        calculated automatically.
            name (str): the name of NoiseKraus
        """

        # check that kraus_ops belongs to BasicGate class
        if not isinstance(kraus_ops, list):
            raise NoiseKrausError('Invalid kraus_ops value. Use list of Kraus operators instead')
        elif not all(isinstance(kraus_op, BasicGate) for kraus_op in kraus_ops):
            raise NoiseKrausError('kraus_ops should be list of BasicGate objects')

        if probs is None:
            kraus_ops, probs = self._calculate_probs(kraus_ops)

        else:
            if not isinstance(probs, list):
                raise NoiseKrausError('probs should be list of float values')

            if len(probs) != len(kraus_ops):
                raise NoiseKrausError('kraus_ops and probs mast have the same length')

            if not self._if_cptp(kraus_ops, probs):
                raise NoiseKrausError('Provided kraus_ops do not belong to CPTP set')

        self._kraus_ops = kraus_ops
        self._probs = probs
        self.name = name

    def _if_cptp(self, kraus_ops, probs):

        kraus_ops_mtx = [kraus_op.matrix for kraus_op in kraus_ops]

        dim = len(kraus_ops_mtx[0])
        sum_of_kraus = np.zeros((dim, dim))
        for kr_op, prob in zip(kraus_ops_mtx, probs):
            kr_op_dagg = np.conjugate(np.transpose(kr_op))
            sum_of_kraus = sum_of_kraus + prob * kr_op @ kr_op_dagg

        residual = np.abs(sum_of_kraus - np.eye(dim)).trace()
        if residual < self._tol:
            return True
        else:
            return False

    @staticmethod
    def _calculate_probs(kraus_ops):

        probs = []
        kraus_ops_rescaled = []

        for kraus_op in kraus_ops:
            kr_op_mtx = kraus_op.matrix
            kr_op_dagg = np.conjugate(np.transpose(kr_op_mtx))
            prob = abs(max(np.diag(kr_op_dagg @ kr_op_mtx)))
            if prob > 0:
                prob_sqrt = np.sqrt(prob)
                kraus_op_rescaled = np.array(kr_op_mtx) / prob_sqrt

                kraus_ops_rescaled.append(MatrixGate(kraus_op_rescaled))
                probs.append(prob)

        # normalize probabilities
        sum_of_probs = sum(probs)
        probs_rescaled = [p / sum_of_probs for p in probs]

        return kraus_ops_rescaled, probs_rescaled

    @property
    def kraus_ops(self):
        return self._kraus_ops

    @property
    def probs(self):
        return self._probs
