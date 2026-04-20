""" Noisy quantum circuit file for custom noise models which can be observed in the compiler. 
This is intended to be used with the qiskit AerSimulator backend """

import numpy as np
import random

from scipy.linalg import expm

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate

from ..config.gate_config import _SU2


class UnitaryNoisyQuantumCircuit(QuantumCircuit):
    """
    QuantumCircuit subclass that injects random unitary gate noise with
    stochastic amplitude after every single-qubit gate.
    The sampling doesn't occur per shot, but instead per gate.

    Noise is applied as:
        U → U · exp(-i p H)
    with probability p, and H being sampled from SU(2)
      
    """
    def __init__(
        self,
        num_qubits: int,
        error_probability: float = 0.01,
        max_offset: float = 0.1,
        **qc_kwargs
    ):
        """Constructor of UnitaryNoisyQuantumCircuit.

        Parameters
        ----------
        num_qubits : int
            number of qubits for circuit
        error_probability : float, optional
            probability to inject unitary noise in a gate, by default 0.01
        max_offset : float, optional
            maximum angle for noisy unitary, by default 0.1
        """
        super().__init__(num_qubits, **qc_kwargs)

        self.error_probability = error_probability
        self.max_offset = max_offset
        self.other = qc_kwargs

    def _random_su2_unitary(self) -> Operator:
        pauli = _SU2[random.choice(_SU2)]
        angle = random.uniform(0, self.max_offset)
        return Operator(expm(-1j * angle * pauli))

    def append(self, 
               instruction, 
               qargs= None, 
               cargs= None, 
               *, 
               copy= True
               ):
        if cargs is None:
            cargs = []

        # Only inject noise into single-qubit unitary gates
        if (
            len(qargs) == 1
            and random.random() <= self.error_probability
        ):
            try:
                base_op = Operator(instruction)
                pauli = _SU2[random.choice(_SU2)]
                angle = random.uniform(0, self.max_offset)
                noise = Operator(expm(-1j * angle * pauli))

                noisy_op = base_op.compose(noise)
                instruction = UnitaryGate(noisy_op.data)

            except Exception:
                # Non-unitary gates (measure, reset, barrier, etc.)
                pass

        return super().append(instruction, qargs, cargs, copy=copy)
    
    def copy(self, name=None):
        # Let QuantumCircuit handle the deep structural copy
        new_circ = super().copy(name=name)

        # Restore custom attributes specific to your subclass
        new_circ.error_probability = self.error_probability
        new_circ.max_offset = self.max_offset

        return new_circ
    
    def to_quantum_circuit(self) -> QuantumCircuit:
        """ Convert this NoisyQuantumCircuit back into a QuantumCircuit instance """
        qc = QuantumCircuit(
            self.num_qubits,
            name=self.name, 
            global_phase= self.global_phase, 
            **self.other 
        )
        qc.data = list(self.data)
        return qc