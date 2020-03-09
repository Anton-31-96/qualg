"""
Contains noise_zoo engine
"""

from copy import deepcopy
import random

import numpy as np
from projectq.cengines import BasicEngine
from projectq.ops import Command, ClassicalInstructionGate, X, Y, Z

from ._noiseoperator import NoiseKraus
from ._noise_element import NoiseChannel
from .noise_zoo import Id


class NoiseEngine(BasicEngine):
    """
    Noise engine attaches quantum error channels
    with respect to the noise_zoo model
    """

    def __init__(self, p, noise_model=None):
        """
        Initialize NoiseEngine object
        """

        self.p = p  # TODO: Delete this later

        noise_model = deepcopy(noise_model)
        BasicEngine.__init__(self)
        self.noise_model = deepcopy(noise_model)

    @staticmethod
    def _if_noise(cmd):
        """
        Check if the cmd should be executed with additional error channel
        """
        # TODO: implement this function better
        return True

    # def _append_noise_gates(self, cmd):
    #     noise_model = self.noise_model
    #     acted_qubits = [q for qreg in cmd.qubits for q in qreg]
    #     control_qubits = cmd.control_qubits
    #     qubits = acted_qubits + control_qubits
    #
    #     cmd_before_gate = []
    #     cmd_after_gate = []
    #
    #     for error in noise_model.list_errors:
    #         kraus_ops = self._generate_instance(error)
    #         new_cmd = self._generate_command(kraus_ops, qubits, self.next_engine)
    #         if error.position == 'after':
    #             cmd_after_gate.append(new_cmd)
    #         elif error.position == 'before':
    #             cmd_before_gate.append(new_cmd)
    #
    #     return cmd_before_gate + cmd + cmd_after_gate

    def _append_noise_gates(self, cmd):
        p = self.p
        probs = []
        probs = [p/4] * 3
        probs.append(1 - 0.75 * p)
        kraus_ops = [X, Y, Z, Id]


        # acted_qubits = [q for qreg in cmd.qubits for q in qreg]
        # acted_qubits = [q for q in cmd.qubits]

        acted_qubits = [q for qureg in cmd.all_qubits[1:] for q in qureg]
        control_qubits = cmd.control_qubits
        qubits = acted_qubits # + control_qubits

        instance = np.random.choice(kraus_ops, p=probs, size=len(qubits))
        # instance = np.random.choice(kraus_ops, p=probs)

        new_cmd = self._generate_command(instance, qubits, self.next_engine)
        return [cmd] + new_cmd

    # TODO: figure out this function better implementation
    # @staticmethod
    # def _generate_command(kraus_ops, qubits, engine):
    #     if not isinstance(qubits, list):
    #         qubits = list(qubits)
    #     cmd_list = []
    #     for kraus_op, qubit in zip(kraus_ops, qubits):
    #         qubit = kraus_op.make_tuple_of_qureg(qubit)
    #         cmd_list.append(Command(engine, kraus_op, qubit))
    #     return cmd_list

    @staticmethod
    def _generate_command(kraus_ops, qubits, engine) -> list:
        if not isinstance(qubits, list):
            qubits = list(qubits)
        cmd = []
        for op, qubit in zip(kraus_ops, qubits):
            cmd.append(Command(engine, op, [[qubit]]))
        return cmd

    @staticmethod
    def _generate_instance(error):
        kraus_set_len = len(error.kruas_error)
        kraus_op_ind = np.random.choice(kraus_set_len, p=error.probs)
        kraus_op = error.krais_error[kraus_op_ind]
        return kraus_op

    def _cmd_mod_fun(self, cmd):
        if self._if_noise(cmd):
            gate = cmd.gate
            if isinstance(gate, ClassicalInstructionGate):
                # TODO: add this case for measurement and init errors
                return [cmd]
            else:
                cmds = self._append_noise_gates(cmd)
            return cmds

    def receive(self, command_list):
        """
        Add noise_zoo to the command and send it to the next engine
        """
        new_command_list = []
        for cmd in command_list:
            new_command_list += self._cmd_mod_fun(cmd)
        self.send(new_command_list)
