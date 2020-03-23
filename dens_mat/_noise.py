# created by Anton Bozhedarov

import numpy as np
from numpy import sqrt


class NoiseError(Exception):
    pass


class NoiseChannel(object):
    pass


class DepolChannel(NoiseChannel):

    def __init__(self, p):
        self.p = p

    def __or__(self, state):
        """
        acts on state in form of density matrix
        """
        dim = len(state)
        return np.eye(dim)/dim * self.p + (1 - self.p) * state


# class AmplitudeDamping(NoiseChannel):
#
#     def __init__(self, p, gamma):
#         self.p = p
#         self.gamma = gamma
#
#         E_0 = np.sqrt(p) * np.matrix([[1, 0], [0, np.sqrt(1 - gamma)]])
#         E_1 = np.sqrt(p) * np.matrix([[0, np.sqrt(gamma)], [0, 0]])
#         E_2 = np.sqrt(p-1) * np.matrix([[np.sqrt(1 - gamma), 0], [0, 1]])
#         E_3 = np.sqrt(p-1) * np.matrix([[0, 0], [np.sqrt(gamma), 0]])
#         kraus_ops = [E_0, E_1, E_2, E_3]
#         self.kraus_ops = kraus_ops
#
#     def __or__(self, state):
#         for kraus_op in self.kraus_ops:
#             state += kraus_op @ state @ kraus_op.H
#         return state


class AmplitudePhaseDamping(NoiseChannel):

    def __init__(self, param_amp, param_phase, excited_state_population=0, num_qubits=None):

        if param_amp < 0:
            raise NoiseError(f"Invalid amplitude damping parameter, {param_amp} < 0")
        if param_phase < 0:
            raise NoiseError(f"Invalid phase damping parameter, {param_phase} < 0")
        if param_amp + param_phase > 1:
            raise NoiseError(f"Invalid amplitude and phase damping parameters, {param_amp} + {param_phase} > 1")
        if excited_state_population < 0:
            raise NoiseError(f"Invalid excited state population parameter, {excited_state_population} < 0")
        if excited_state_population > 1:
            raise NoiseError(f"Invalid excited state population parameter, {excited_state_population} > 1")

        # shorten variables
        p1 = excited_state_population
        a = param_amp
        b = param_phase

        # Damping to the state |0>
        A0 = sqrt(1 - p1) * np.matrix([[1, 0], [0, sqrt(1 - a - b)]])
        A1 = sqrt(1 - p1) * np.matrix([[0, sqrt(a)], [0, 0]])
        A2 = sqrt(1 - p1) * np.matrix([[0, 0], [0, sqrt(b)]])

        # Damping to the state |1>
        B0 = sqrt(p1) * np.matrix([[sqrt(1 - a - b), 0], [0, 1]])
        B1 = sqrt(p1) * np.matrix([[0, 0], [sqrt(a), 0]])
        B2 = sqrt(p1) * np.matrix([[sqrt(b), 0], [0, 0]])

        # chose non-zero operators:
        kraus_ops = [op for op in [A0, A1, A2, B0, B1, B2] if np.linalg.norm(op) > 1e-10]

        self.kraus_ops = sum(kraus_ops)
        if num_qubits is None:
            self.kraus_ops_ext = None
        else:
            self._create_extended_kraus_ops(num_qubits)

    def _create_extended_kraus_ops(self, num_qubits):
        kraus_ops_ext = self.kraus_ops
        for _ in range(num_qubits - 1):
            kraus_ops_ext = np.kron(kraus_ops_ext, self.kraus_ops)
        self.kraus_ops_ext = np.kron(kraus_ops_ext, np.conj(kraus_ops_ext))

    def __or__(self, state):
        dim = len(state)
        if self.kraus_ops_ext is None:
            num_qubits = int(np.log2(dim))
            self._create_extended_kraus_ops(num_qubits)

        state = self.kraus_ops_ext @ state.reshape(-1,1)
        state = state.reshape(dim, dim)

        return state


class AmplitudeDamping(AmplitudePhaseDamping):

    def __init__(self, param_amp, excited_state_population=0):
        AmplitudePhaseDamping.__init__(self, param_amp, 0, excited_state_population)


class PhaseDamping(AmplitudePhaseDamping):

    def __init__(self, param_phase, excited_state_population=0):
        AmplitudePhaseDamping.__init__(param_phase, excited_state_population)


class ThermalRelaxation(AmplitudePhaseDamping):

    def __init__(self, t1, t2, gate_time=1, excited_state_population=0):
        """
        t1 (double): T1 relaxation time constant (energy relaxation)
        t2 (double): T2 relaxation time constant (phase time relaxation)
        gate_time (double): the gate time for relaxation error.
        excited_state_population (double): the population of |1>
                                           state at equilibrium (default: 0).
        """

        # check the value of parameters t1, t2 to be physical
        if excited_state_population < 0:
            raise NoiseError("Invalid excited state population "
                             f"({excited_state_population} < 0).")
        if excited_state_population > 1:
            raise NoiseError("Invalid excited state population "
                             f"({excited_state_population} > 1).")
        if gate_time < 0:
            raise NoiseError(f"Invalid gate_time ({gate_time} < 0)")
        if t1 <= 0:
            raise NoiseError("Invalid T_1 relaxation time parameter: T_1 <= 0.")
        if t2 <= 0:
            raise NoiseError("Invalid T_2 relaxation time parameter: T_2 <= 0.")
        if t2 - 2 * t1 >= 0: # sign "=" to avoid division by zero
            raise NoiseError(
                "Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.")
        if t2 - t1 < 0:
            raise NoiseError(
                "Invalid T_2 relaxation time parameter: T_2 less than T_1.")

        t2_pure = (2 * t1 * t2) / (2 * t1 - t2)
        param_amp = 1 - np.exp(- gate_time / t1)
        param_phase = 1 - np.exp(- 2 * gate_time / t2_pure)

        AmplitudePhaseDamping.__init__(self, param_amp, param_phase, excited_state_population)
