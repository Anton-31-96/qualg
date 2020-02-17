from projectq.ops import Measure, Allocate
from copy import deepcopy
from ._noise_element import NoiseChannel
from ._noiseoperator import NoiseKraus


class NoiseModelError(Exception):
    pass


class NoiseModel(object):

    def __init__(self, error=None, noise_gates=None, noise_qubits=None, name=None):
        """
        Arguments:
            error (NoiseKraus obj): quantum error
            noise_gates (list|set): gates that is affected by noise_zoo
            noise_qubits (list|set): qubits indexes that are affected by noise_zoo model
            name (str): name of the noise_zoo model
        """
        self._noise_qubits = set()
        self._noise_gates = set()
        self.list_errors = []

        if error is not None:
            self.add_error(error, noise_gates, noise_qubits, name)

    def add_error(self,
                  error,
                  noise_gates=None,
                  noise_qubits=None,
                  name=None,
                  position='after'):
        """
        Add quantum error to the noise_zoo model
        """

        if isinstance(error, NoiseChannel):
            pass
        elif isinstance(error, NoiseKraus):
            error = NoiseChannel(kraus_ops=error, qubits=noise_qubits, gates=noise_gates)
        else:
            raise NoiseModelError('Argument must belongs to NoiseChannel class')

        error = deepcopy(error)

        self.list_errors.append(error)

        if noise_qubits is not None:
            error.add_noise_qubits(noise_qubits)
            self._noise_qubits = self._noise_qubits.union(error.qubits, noise_qubits)
        # TODO: Check this issue
        # else:
        #     self._noise_qubits = self._noise_qubits.union(error.qubits)

        if name is not None:
            error.name = name

        if position == 'after':
            error.position = 'after'
        elif position == 'before':
            error.position = 'before'
        else:
            raise NoiseModelError('Argument "position" is incorrect')

        self.list_errors.append(error)

    def add_measure_error(self, error, noise_qubits, name=None):

        self.add_error(error,
                       noise_gates=Measure,
                       noise_qubits=noise_qubits,
                       name=name,
                       position='before')

    def add_init_error(self,
                       error,
                       noise_qubits=None,
                       name=None):

        self.add_error(error,
                       noise_gates=Allocate,
                       noise_qubits=noise_qubits,
                       name=name,
                       position='after')
