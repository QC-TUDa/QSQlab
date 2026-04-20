""" Base noise simulator file for custom noise simulation backends with Qiskit's Aer simulator """

import logging
logger = logging.getLogger(__name__)
logging.getLogger("qiskit_aer").setLevel(logging.ERROR) 
# gets rid of pesky warnings about composing gate errors

import random
import numpy as np
from itertools import product
from scipy.linalg import expm

from qiskit.result import Result
from qiskit.quantum_info import Operator
from qiskit_aer.noise.noiseerror import NoiseError
from qiskit_aer import AerSimulator
from qiskit_aer.noise import ( 
    NoiseModel, 
    ReadoutError, 
    pauli_error, 
    depolarizing_error, 
    amplitude_damping_error, 
    phase_damping_error, 
    phase_amplitude_damping_error, 
    thermal_relaxation_error, 
    
    mixed_unitary_error,
    coherent_unitary_error, 

    # reset_error, 
    # kraus_error, 
)
    
from ..config.gate_config import _SINGLE_QUBIT_GATES, _SU2

class NoisyAerSimulator(AerSimulator):
    """
    Extension of Qiskit's :class:`~qiskit_aer.AerSimulator` that provides a
    convenient interface for constructing and simulating circuits with
    configurable noise models.

    This simulator allows the user to independently enable and combine
    multiple common noise sources—including SPAM errors, depolarizing noise,
    Pauli channels, damping processes, thermal relaxation, and both coherent
    and mixed-unitary errors—using a single constructor.
    """
    def __init__(
        self,
        measurement_error_prob: float | list[float] = 0.0,
        depol_error_prob: float = 0.0,
        coherent_unitary_error_map: float | dict[str, float] = 0.0,
        mixed_unitary_error_map: float | list[tuple[Operator, float]] = 0.0, 
        amplitude_damping_prob: float = 0.0,
        phase_damping_prob: float = 0.0,
        phase_amplitude_damping_probs: list[float] = [0.0, 0.0],
        pauli_probs: dict[str, float] | None = None,
        thermal_params: dict[str, float] | None = None,
        **aer_kwargs,
    ):
        """
        Initialize a noisy Aer simulator with a customizable noise model.

        Any subset of the supported noise sources may be specified. Parameters
        left at their default values result in no contribution to the noise
        model. If all noise parameters are omitted, the simulator behaves
        identically to a standard :class:`~qiskit_aer.AerSimulator`.

        Parameters
        ----------
        measurement_error_prob : float | list[float], optional
            Probability of measurement (SPAM) error applied to all qubits. Supported formats are:
            - ``X: float``: outputs symmetric measurement error irrespective of state
            - ``[X, Y]: list[float]``: state dependent measurement error with 
                X := error for |0> and Y := error for |1>
            , by default ``0.0``.

        depol_error_prob : float, optional
            Strength of the single-qubit depolarizing channel applied to gate
            operations, by default ``0.0``.

        coherent_unitary_error_map : float | dict[str, float], optional
            Specifies the rotation amplitudes for coherent unitary errors applied to gates.
            The error is modeled as a unitary transformation: U = exp(-i/2 * (a*X + b*Y + c*Z))
            which sandwiches the ideal gate: G_actual = U† * G_ideal * U.

            Parameters:
                - dict[str, float]: Maps Pauli axes to rotation amplitudes.
                Example: {"X": 0.01, "Y": 0.02} creates U = exp(-i/2 * (0.01*X + 0.02*Y))
                Valid keys are "X", "Y", and/or "Z". Omitted axes default to 0.0.
                
                - float: A single numeric value. A random Pauli axis ("X", "Y", or "Z") 
                is chosen uniformly, and the error uses this amplitude on that axis only.
                
                - None: Both the amplitude and axis are randomized. Amplitude is drawn 
                uniformly from (0, π/2], and the axis is chosen uniformly from ["X", "Y", "Z"].
            
            by default ``0.0``.

        mixed_unitary_error_map : float | list[tuple[Operator, float]], optional
            List of ``(U_j, p_j)`` pairs defining a mixed-unitary channel, where
            ``U_j`` is a unitary operator applied with probability ``p_j``.
            Probabilities must sum to 1.
            by default ``0.0``.

        amplitude_damping_prob : float, optional
            Probability parameter for single-qubit amplitude damping noise,
            by default ``0.0``.

        phase_damping_prob : float, optional
            Probability parameter for single-qubit phase damping noise,
            by default ``0.0``.

        phase_amplitude_damping_probs : list[float], optional
            Probability parameters for single-qubit phase and amplitude damping noise, 
            Supported formats is:
            - ``[X, Y]: list[float]``: with X := phase dampiing and Y := amplitude damping
            by default ``[0.0, 0.0]``.

        pauli_probs : dict[str, float] or None, optional
            Pauli error channel specified as a mapping from Pauli labels
            (e.g. ``"X"``, ``"Y"``, ``"Z"``) to their corresponding probabilities.
            Probabilities must sum to 1. If ``None``, no Pauli noise is applied.
        
        thermal_params : dict[str, float] or None, optional
            Parameters defining a single-qubit thermal relaxation error.
            Supported keys are:

            - ``t1``: :math:`T_1` relaxation time
            - ``t2``: :math:`T_2` relaxation time
            - ``time``: gate duration
            - ``excited_state_population``: equilibrium population of
              :math:`|1\\rangle` (default: 0)

            If ``None``, no thermal relaxation noise is applied.

        **aer_kwargs
            Additional keyword arguments forwarded directly to
            :class:`~qiskit_aer.AerSimulator`.
        """
        self.noise_model = NoiseModel()

        self._single_qubit_gates: list = list(_SINGLE_QUBIT_GATES) + ['u1', 'u2', 'u3']

        self._add_measurement_noise(measurement_error_prob)
        self._add_depolarizing_noise(depol_error_prob)
        self._add_mixed_unitary_error(mixed_unitary_error_map)
        self._add_coherent_unitary_error(coherent_unitary_error_map)
        self._add_pauli_noise(pauli_probs)
        self._add_amplitude_damping_noise(amplitude_damping_prob)
        self._add_phase_damping_noise(phase_damping_prob)
        self._add_phase_amplitude_damping_noise(phase_amplitude_damping_probs)
        if thermal_params:
            self._add_thermal_relaxation_noise(**thermal_params)

        super().__init__(noise_model=self.noise_model, **aer_kwargs)

    # ------------------------------------------------------------------
    # Noise definitions
    # ------------------------------------------------------------------
    def _add_mixed_unitary_error(self, 
                                 gate_amplitude_map: float | list[tuple[Operator, float]] = 0.0,
                                 ):
        match gate_amplitude_map:
            case float(n) if n == 0.0:
                return
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                    f"Invalid value for amplitude of mixed unitary error {n}, must be between 0.0 and 1.0"
                    )
            case float(n):
                _noise_ops = [
                    (expm(-1j/2*n*_SU2["X"]), 1/len(_SU2)),
                    (expm(-1j/2*n*_SU2["Y"]), 1/len(_SU2)),
                    (expm(-1j/2*n*_SU2["Z"]), 1/len(_SU2)),
                    ]
            case dict(n) if len(n) != 0:
                _noise_ops = gate_amplitude_map

        error = mixed_unitary_error(_noise_ops)
        self._add_single_qubit_noise(error)

    def _add_coherent_unitary_error(self, 
                                    gate_amplitude_map: float | dict[str, float] = 0.0, 
                                    ):
        match gate_amplitude_map:
            case float(n) if n == 0.0:
                return
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                        f"Invalid value for amplitude of coherent unitary error {n}, must be between 0.0 and 1.0"
                        )
            case float(n):
                rotation = n*random.choice(_SU2)
            case dict(n) if len(n) != 0: 
                rotation = sum(_SU2[a[0]]*a[1] for a in n.items() if a[0] in ["X", "Y", "Z"])
            case _: 
                raise NoiseError(
                    f"Invalid format for coherent unitary error {type(gate_amplitude_map).__name__}"
                    )
        # TODO include log
        error = coherent_unitary_error(expm(-1j/2*rotation))
        self._add_single_qubit_noise(error)

    def _add_measurement_noise(self, prob: float | list[float, float] = 0.0):
        """ readout bit-flip error. """
        match prob:
            case float(n) if n == 0.0:
                return
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                        f"Invalid value for measurement error {n}, must be between 0.0 and 1.0"
                        )
            case float(n) | int(n): 
                error = [[1 - n, n], [n, 1 - n]]
            case list(n) if len(n) == 1: 
                error = [[1 - n[0], n[0]], [n[0], 1 - n[0]]]
            case list(n) if len(n) == 2: 
                error = [[1 - n[0], n[0]], [n[1], 1 - n[1]]]
            case _: 
                raise NoiseError(
                    f"Invalid format for measurement error {type(prob).__name__}"
                    )
        logger.debug(
            "Composing measurement error with map: %s", 
            error
            )
        error = ReadoutError(error)
        self.noise_model.add_all_qubit_readout_error(error)

    def _add_depolarizing_noise(self, prob: float = 0.0):
        match prob:
            case float(n) if n == 0.0:
                return
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                        f"Invalid value for depolarizing error {n}, must be between 0.0 and 1.0"
                        )
            case float(n):
                error = depolarizing_error(prob, 1)
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(prob).__name__}"
                    )
        logger.debug(
            "adding depolarizing error of: %s", 
            prob
            )
        self._add_single_qubit_noise(error)

    def _add_pauli_noise(self, probs: dict[str, float] | None = None):
        """
        probs example:
            {"X": 0.01, "Y": 0.01, "Z": 0.02}
        """
        match probs:
            case None:
                return
            case dict(n) if n == {}:
                return
            case dict(n):
                p_identity = 1.0 - sum(a[1] for a in probs.items() if a[0] in ["X", "Y", "Z"])
                error = [(k, v) for k, v in probs.items()] + [("I", p_identity)]
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(probs).__name__}"
                    )
        logger.debug(
            "adding pauli noise with probabilities, %s",
            error
        )
        error = pauli_error(error)
        self._add_single_qubit_noise(error)

    def _add_amplitude_damping_noise(self, gamma: float = 0.0):
        match gamma:
            case float(n) if n == 0.0:
                return
            case float(n):
                error = gamma
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(gamma).__name__}"
                    )
        logger.debug(
            "Adding amplitude damping noise of: %s", 
            error
        )
        self._add_single_qubit_noise(amplitude_damping_error(error))

    def _add_phase_damping_noise(self, gamma: float = 0.0):
        match gamma:
            case float(n) if n == 0:
                return
            case float(n):
                error = gamma
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(gamma).__name__}"
                    )
        logger.debug(
            "Adding phase damping noise of: %s", 
            error
        )
        self._add_single_qubit_noise(phase_damping_error(error))

    def _add_phase_amplitude_damping_noise(self, gamma_lam: list[float] = [0.0, 0.0]):
        match gamma_lam:
            case [float(g), float(l)] if g == 0.0 and l == 0.0:
                return  # Both are zero, nothing to add
            case [float(g), float(l)]:
                error_gamma, error_lam = g, l
            case [float(g), _]:
                raise NoiseError(f"Invalid format for lam: {type(gamma_lam[1]).__name__}")
            case [_, float(l)]:
                raise NoiseError(f"Invalid format for gamma: {type(gamma_lam[0]).__name__}")
            case _:
                raise NoiseError(
                    f"Invalid format for gamma_lam list. Expected [float, float], got {[type(x).__name__ for x in gamma_lam]}"
                )
        logger.debug(
            "Adding amplitude phase damping noise of: %s, %s", 
            error_gamma, error_lam
        )
        self._add_single_qubit_noise(
            phase_amplitude_damping_error(error_gamma, error_lam)
        )

    def _add_thermal_relaxation_noise(self, t1, t2, gate_time):
        match (t1, t2, gate_time):
            case (float(t1_val), float(t2_val), float(gate_time_val)):
                error_t1, error_t2, error_gate_time = t1_val, t2_val, gate_time_val
            case (float(t1_val), float(t2_val), _):
                raise NoiseError(
                    f"Invalid format for gate_time: {type(gate_time).__name__}"
                )
            case (float(t1_val), _, float(gate_time_val)):
                raise NoiseError(
                    f"Invalid format for t2: {type(t2).__name__}"
                )
            case (_, float(t2_val), float(gate_time_val)):
                raise NoiseError(
                    f"Invalid format for t1: {type(t1).__name__}"
                )
            case _:
                raise NoiseError(
                    f"Invalid formats for thermal relaxation parameters. "
                    f"Expected (float, float, float), got "
                    f"({type(t1).__name__}, {type(t2).__name__}, {type(gate_time).__name__})"
                )
        
        # Optional: Add validation for values (e.g., t1 and t2 should be positive)
        if error_t1 <= 0 or error_t2 <= 0 or error_gate_time <= 0:
            raise NoiseError(
                f"Thermal relaxation parameters must be positive. "
                f"Got t1={error_t1}, t2={error_t2}, gate_time={error_gate_time}"
            )
        
        logger.debug(
            "Adding thermal relaxation noise with t1=%s, t2=%s, gate_time=%s",
            error_t1, error_t2, error_gate_time
        )
        self._add_single_qubit_noise(
            thermal_relaxation_error(error_t1, error_t2, error_gate_time)
        )

    def add_random_unitary_noise(self, 
                                 max_offset: float = 0.1, 
                                 noise_sample_size: int = 100
                                 ):
        """method to inject unitary noise sampled at the shot level. 
        This is done by creating a pool of noisy unitaries of random rotation angle and axis and 
        applying them randomly at the shot level

        Parameters
        ----------
        max_offset : float, optional
            maximum angle for sampled unitary noise, by default 0.1
        noise_sample_size : int, optional
            Pool size of , by default 100
        """
        unitaries = [self._generate_random_unitary(max_offset) for _ in range(noise_sample_size)]
        probs = [1/noise_sample_size]*noise_sample_size
        error = mixed_unitary_error(list(zip(unitaries, probs)))
        self._add_single_qubit_noise(error)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_single_qubit_noise(self, error):
        """Attach a single-qubit quantum error to all single-qubit gates."""
        self.noise_model.add_all_qubit_quantum_error(
            error, self._single_qubit_gates
        )

    def _generate_random_unitary(self, max_offset: float) -> Operator:
        """Generate a random unitary modeled as U=exp(-i p H)
            with angle p uniformly sampled between [0, max_offset], 
            and H being sampled from SU(2)"""
        P = _SU2[np.random.randint(3)]
        theta = np.random.uniform(0, max_offset)
        return Operator(expm(-1j/2*theta*P))
        
    

class ResultFull(Result):
    def get_counts(self, experiment=None):
        """
        Override get_counts to include all possible bitstrings with 0 counts if missing.
        This class and function is only used in the case of AerSimulator based backends
        """
        counts = super().get_counts(experiment)
        
        # Determine number of qubits from bitstring length
        if counts:
            num_qubits = len(next(iter(counts)))
        else:
            return counts  # Empty result, return as-is

        all_bitstrings = [''.join(bits) for bits in product('01', repeat=num_qubits)]
        
        # Ensure all possible bitstrings present with 0 counts if missing
        full_counts = {bitstring: counts.get(bitstring, 0) for bitstring in all_bitstrings}
        
        return full_counts