from projectq.ops import BasicGate

from ._noiseoperator import NoiseKraus, NoiseKrausError


class NoiseChannelError(Exception):
    pass


class NoiseChannel(object):

    def __init__(self, kraus_ops=None, qubits=None, gates=None):
        """
        Arguments:
            kraus_ops (NoiseKraus obj): Noise channel in Kraus representation
            qubits (set of int): indexes of qubits that are affected by noise_zoo channel
            gates (set of gates): gates that are affected by noise_zoo channel
        """
        # TODO: consider case of all qubits and all gates. And change methods add_gates and add_qubits
        self.kraus_ops = kraus_ops
        self.qubits = qubits
        self.gates = gates

    def add_gates(self, gates):
        """
        Append noisy gates to channel
            gates (list | set): noisy gates that should be added to consideration
        """
        # check input
        if isinstance(gates, list):
            if all(isinstance(gate, BasicGate) for gate in gates):
                pass
            else:
                raise NoiseChannelError("All elements of gates should be BasicGate objects")
        else:
            if isinstance(gates, BasicGate):
                gates = set(gates)
            else:
                raise NoiseChannelError("gates should be list of BasicGate objects")

        if self.gates is None:
            self.gates = gates
        else:
            self.gates.union(gates)

    def add_qubits(self, qubits):

        if self.qubits is None:
            self.qubits = set(qubits)
        else:
            self.qubits.union(set(qubits))
