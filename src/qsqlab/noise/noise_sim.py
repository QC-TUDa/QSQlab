""" NoisyAerSimulator file for custom error-prown backends with Qiskit's AerSimulator
"""

import logging
logger = logging.getLogger(__name__)
logging.getLogger("qiskit_aer").setLevel(logging.ERROR) 
# gets rid of pesky warnings about composing gate errors

import random
import numpy as np
from itertools import product
from scipy.linalg import expm
from typing import Any

from qiskit.result import Result
from qiskit.quantum_info import Operator, SuperOp
from qiskit_aer.noise.noiseerror import NoiseError
from qiskit_aer import AerSimulator
from qiskit_aer.noise import ( 
    QuantumError,
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

from .noise_types import *    
from ..config.gate_config import SINGLE_QUBIT_GATES, SU2

class NoisyAerSimulator(AerSimulator):
    """
    Extension of Qiskit's :class:`~qiskit_aer.AerSimulator` for constructing
    and simulating quantum circuits with configurable noise models.

    The simulator provides convenience methods for adding common quantum and
    readout error channels, while keeping an ordered log of all errors that
    have been added. Quantum errors are additionally composed into a single
    :class:`~qiskit.quantum_info.SuperOp` for subsequent gate-fidelity
    calculations.

    Parameters
    ----------
    measurement_error_prob : float or list of float, optional
        Measurement error probability applied to all qubits.

        A scalar ``p`` creates a symmetric readout error where the probability
        of flipping either measurement outcome is ``p``. A two-element list
        ``[p0, p1]`` specifies state-dependent errors, where ``p0`` is the
        probability of reading ``1`` when the state is ``0``, and ``p1`` is
        the probability of reading ``0`` when the state is ``1``.

    **aer_kwargs
        Additional keyword arguments passed directly to
        :class:`~qiskit_aer.AerSimulator`.

    Attributes
    ----------
    noise_model : qiskit_aer.noise.NoiseModel
        Noise model containing all errors currently applied to the simulator.
    gate_error : qiskit.quantum_info.SuperOp or None
        Composition of all quantum errors added so far.
    noise_log : list of ErrorEntry
        Ordered list describing all errors added to the simulator.

    Notes
    -----
    If no noise parameters are specified, the simulator behaves like a
    standard :class:`~qiskit_aer.AerSimulator`.
    """
    def __init__(
        self,
        measurement_error_prob: float | list[float] = 0.0,
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

        **aer_kwargs
            Additional keyword arguments forwarded directly to
            :class:`~qiskit_aer.AerSimulator`.
        """
        self._noise_model = NoiseModel()
        self._gate_error: SuperOp | None = None
        self._noise_log: list[ErrorEntry] = []

        self._single_qubit_gates: list = list(SINGLE_QUBIT_GATES) + ['u1', 'u2', 'u3']

        super().__init__(noise_model=self.noise_model, **aer_kwargs)   

        self.add_readout_error(measurement_error_prob)

    @property
    def noise_model(self) -> NoiseModel:
        """
        Return the noise model currently associated with the simulator.

        Returns
        -------
        qiskit_aer.noise.NoiseModel
            Noise model containing all currently registered errors.
        """
        return self._noise_model

    @property
    def gate_error(self) -> SuperOp | None:
        """
        Return the composed quantum gate-error channel.

        The channel contains the composition of the quantum errors that have been
        added through the simulator. Readout errors are not included.

        Returns
        -------
        qiskit.quantum_info.SuperOp or None
            Composed gate-error channel, or ``None`` if no quantum errors have
            been added.
        """
        return self._gate_error

    @property
    def noise_log(self) -> list[ErrorEntry]:
        """
        Return the ordered log of errors added to the simulator.

        Returns
        -------
        list of ErrorEntry
            Error entries in the order in which they were added.
        """
        return self._noise_log

    def clear_errors(self) -> list[NoiseError]:
        """
        Remove all noise from the simulator.

        The noise model, composed gate-error channel, and noise log are reset.
        After this operation, the simulator has no custom noise applied.

        Returns
        -------
        list of NoiseError
            Empty list representing the cleared noise log.
        """
        self._noise_log = []
        self._noise_model = NoiseModel()
        self.set_option("noise_model", self._noise_model)
        self._gate_error = None
        return self._noise_log

    def add_error_entries(
            self,
            errors: list[ErrorEntry] | ErrorEntry,
            clear_previous: bool = False,
            ) -> list[ErrorEntry]:
        """
        Apply previously constructed error entries to the simulator.

        Parameters
        ----------
        errors : ErrorEntry or list of ErrorEntry
            Error entry or entries to apply. When a list is provided, entries are
            applied in the order given.
        clear_previous : bool, optional
            If ``True``, remove all existing errors before applying the supplied
            entries. If ``False``, append the supplied errors to the existing
            noise model.

        Returns
        -------
        list of ErrorEntry
            Complete noise log after the supplied errors have been applied.

        Raises
        ------
        ValueError
            If ``errors`` is neither an ``ErrorEntry`` nor a list of
            ``ErrorEntry`` objects.
        """
        if not isinstance(errors, (list, ErrorEntry)):
            raise ValueError(
                f"unable to apply errors. Expected list[ErrorEntry] or ErrorEntry, got {type(errors).__name__}"
            )
        if isinstance(errors, list) and not all(isinstance(e, ErrorEntry) for e in errors):
            raise ValueError("unable to apply errors, all entries in list must be ErrorEntry instances")

        if clear_previous:
            self.clear_errors()
        if isinstance(errors, ErrorEntry):
            self._add_entry(errors)
        else:
            for entry in errors:
                self._add_entry(entry)

        return self._noise_log
    
    # ------------------------------------------------------------------
    # Noise definitions
    # ------------------------------------------------------------------

    def add_readout_error(
            self, 
            probabilities: float | list[float, float] = 0.0, 
            qubits: list[int] | None = None,
            **kwargs
            ) -> ReadoutError | None:
        """
        Add a classical readout error to the simulator.

        A scalar probability produces a symmetric bit-flip readout error. A
        two-element list allows different error probabilities for measurements
        originating from ``|0>`` and ``|1>``.

        Parameters
        ----------
        probabilities : float or list of float, optional
            Measurement error probability.

            * ``p``: symmetric error probability for both states.
            * ``[p]``: equivalent to the scalar form.
            * ``[p0, p1]``: state-dependent probabilities, where ``p0`` is the
            probability of reporting ``1`` for ``|0>`` and ``p1`` is the
            probability of reporting ``0`` for ``|1>``.

            ``0.0`` disables the error.
        qubits : list of int or None, optional
            Qubits to which the readout error is applied. If ``None``, the error
            is applied to all qubits.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.ReadoutError or None
            Constructed readout error, or ``None`` if the requested error has
            zero probability.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If the probability is outside ``[0, 1]`` or has an unsupported format.
        """
        entry, p = self._resolve_params(
            kwargs, probabilities=probabilities
        )
        probabilities = p["probabilities"]

        match probabilities:
            case float(n) if n == 0.0:
                return None
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                        f"Invalid value for measurement error {n}, must be between 0.0 and 1.0"
                        )
            case float(n) | int(n): 
                probs = [[1 - n, n], [n, 1 - n]]
            case list(n) if len(n) == 1: 
                probs = [[1 - n[0], n[0]], [n[0], 1 - n[0]]]
            case list(n) if len(n) == 2: 
                probs = [[1 - n[0], n[0]], [n[1], 1 - n[1]]]
            case _: 
                raise NoiseError(
                    f"Invalid format for measurement error {type(probabilities).__name__}"
                    )

        logger.debug("Composing measurement error with map: %s", probs)
        error = ReadoutError(probs)
        return self._finalize(
            entry, ReadoutErrorEntry, error, None, qubits, 
            probabilities = probabilities
        )

    def add_depolarizing_error(
            self, 
            param: float = 0.0, 
            num_qubits: int = 1,
            instructions: Any | None = None,
            qubits: list[int] | None = None,   
            **kwargs
            ) -> QuantumError | None:
        """
        Add a depolarizing quantum error.

        Parameters
        ----------
        param : float, optional
            Depolarizing error probability. Must be between ``0`` and ``1``.
            ``0.0`` disables the error.
        num_qubits : int, optional
            Number of qubits affected by the depolarizing channel. This must match
            the number of qubits supported by each target instruction.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied. For a multi-qubit error, this
            identifies the qubits participating in the error.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed depolarizing error, or ``None`` if ``param`` is zero.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If ``param`` is outside ``[0, 1]`` or has an unsupported type.
            Qiskit Aer may additionally raise ``NoiseError`` if the number of
            qubits in the error does not match the target instruction.
        """
        entry, p = self._resolve_params(
            kwargs, param=param, num_qubits=num_qubits,
        )
        param, num_qubits = p["param"], p["num_qubits"]

        match param:
            case float(n) if n == 0.0:
                return None
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                    f"Invalid value for depolarizing error {n}, must be between 0.0 and 1.0"
                    )
            case float(n):
                pass
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(param).__name__}"
                    )

        logger.debug("adding depolarizing error of: %s", param)
        error = depolarizing_error(param, num_qubits)
        return self._finalize(
            entry, DepolarizingErrorEntry, error, instructions, qubits,
            param=param, num_qubits=num_qubits,
        )
    
    def add_mixed_unitary_error(
            self, 
            noise_ops: float | list[tuple[Operator, float]] = 0.0,
            instructions: Any | None = None,
            qubits: list[int] | None = None, 
            **kwargs
            ) -> QuantumError | None:
        """
        Add a mixed-unitary quantum error.

        Parameters
        ----------
        noise_ops : float or list of tuple of Operator and float, optional
            Description of the mixed-unitary channel.

            A scalar ``p`` generates a mixed-unitary channel from rotations around
            the X, Y, and Z axes. A list specifies the unitary operators and their
            corresponding probabilities explicitly.

            ``0.0`` disables the error.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied. If ``None``, it is applied to all
            applicable qubits.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed mixed-unitary error, or ``None`` if disabled.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If ``noise_ops`` has an unsupported format or an invalid scalar value.
        """
        entry, p = self._resolve_params(
            kwargs, noise_ops=noise_ops
        )
        noise_ops = p["noise_ops"]

        match noise_ops:
            case float(n) if n == 0.0:
                return None
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                    f"Invalid value for amplitude of mixed unitary error {n}, must be between 0.0 and 1.0"
                    )
            case float(n):
                noise_ops = [
                    (expm(-1j/2*n*SU2["X"]), 1/len(SU2)),
                    (expm(-1j/2*n*SU2["Y"]), 1/len(SU2)),
                    (expm(-1j/2*n*SU2["Z"]), 1/len(SU2)),
                    ]
            case list(n):
                pass
            case _: 
                raise NoiseError(
                    f"Invalid format for noise operators. Expected float | list[tuple[Operator, float]], got {type(noise_ops).__name__}"
                    )

        logger.debug("adding mixed unitary error with noise operators: %s", noise_ops)
        error = mixed_unitary_error(noise_ops)
        return self._finalize(
            entry, MixedUnitaryErrorEntry, error, instructions, qubits,
            noise_ops=noise_ops
        )

    def add_coherent_unitary_error(
            self, 
            gate_amplitude_map: float | dict[str, float] | Operator = 0.0,
            instructions: Any | None = None,
            qubits: list[int] | None = None, 
            **kwargs
            ) -> QuantumError | None:
        """
        Add a coherent unitary error representing a residual rotation.

        Parameters
        ----------
        gate_amplitude_map : float, dict of str to float, or Operator, optional
            Description of the residual unitary rotation.

            A scalar generates a rotation around a randomly selected X, Y, or Z
            axis. A dictionary maps Pauli-axis labels (``"X"``, ``"Y"``, ``"Z"``)
            to rotation amplitudes. An ``Operator`` can be supplied directly as
            the rotation operator.

            ``0.0`` disables the error.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied. If ``None``, the error is applied
            to all applicable qubits.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed coherent unitary error, or ``None`` if disabled.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If ``gate_amplitude_map`` has an unsupported format or invalid scalar
            value.
        """
        entry, p = self._resolve_params(
            kwargs, gate_amplitude_map=gate_amplitude_map
        )
        gate_amplitude_map = p["gate_amplitude_map"]

        match gate_amplitude_map:
            case float(n) if n == 0.0:
                return None
            case float(n) if n > 1.0 or n < 0.0:
                raise NoiseError(
                        f"Invalid value for amplitude of coherent unitary error {n}, must be between 0.0 and 1.0"
                        )
            case float(n):
                rotation = n*SU2[random.choice(["X", "Y", "Z"])]
            case dict(n) if len(n) != 0: 
                rotation = sum(SU2[a[0]]*a[1] for a in n.items() if a[0] in ["X", "Y", "Z"])
            case Operator(n): 
                rotation = gate_amplitude_map
            case _: 
                raise NoiseError(
                    f"Invalid format for coherent unitary error {type(gate_amplitude_map).__name__}"
                    )

        logger.debug("adding coherent unitary error with residual rotation: %s", rotation)
        error = coherent_unitary_error(rotation if isinstance(rotation, Operator) else expm(-1j/2*rotation))
        return self._finalize(
            entry, CoherentUnitaryErrorEntry, error, instructions, qubits, 
            gate_amplitude_map=gate_amplitude_map
        )

    def add_pauli_error(
            self, 
            noise_ops: dict[str, float] | None = None,
            instructions: Any | None = None,
            qubits: list[int] | None = None, 
            **kwargs
            ) -> QuantumError | None:
        """
        Add a Pauli error channel.

        Parameters
        ----------
        noise_ops : dict of str to float or None, optional
            Mapping of Pauli operators to their probabilities. Supported Pauli
            labels are ``"X"``, ``"Y"``, and ``"Z"``.

            For example::

                {"X": 0.01, "Y": 0.01, "Z": 0.02}

            The remaining probability is assigned to the identity operation.
            An empty dictionary or ``None`` disables the error.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied. If ``None``, the error is applied
            to all applicable qubits.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed Pauli error, or ``None`` if no noise operators are supplied.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If the supplied probabilities sum to more than ``1`` or have an
            unsupported format.
        """
        entry, p = self._resolve_params(
            kwargs, noise_ops=noise_ops
        )
        noise_ops = p["noise_ops"]

        match noise_ops:
            case None:
                return None
            case dict(n) if n == {}:
                return None
            case dict(n):
                if sum([a[1] for a in noise_ops.items()]) > 1.0:
                    raise NoiseError(
                        f"Invalid probabilities for paulis. all probabilities must sum up to <= 1.0."
                        )
                p_identity = 1.0 - sum(a[1] for a in noise_ops.items() if a[0] in ["X", "Y", "Z"])
                ops = [(k, v) for k, v in noise_ops.items()] + [("I", p_identity)]
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(noise_ops).__name__}"
                    )
            
        logger.debug("adding pauli noise with probabilities, %s", noise_ops)
        error = pauli_error(ops)
        return self._finalize(
            entry, PauliErrorEntry, error, instructions, qubits,
            noise_ops=noise_ops
        )

    def add_amplitude_damping_error(
            self, 
            param_amp: float = 0.0,
            excited_state_population: int = 0,
            canonical_kraus: bool = True,
            instructions: Any | None = None,
            qubits: list[int] | None = None, 
            **kwargs
            ) -> QuantumError | None:
        """
        Add an amplitude-damping error channel.

        Parameters
        ----------
        param_amp : float, optional
            Amplitude-damping parameter. ``0.0`` disables the error.
        excited_state_population : int, optional
            Population of the excited state used by the Qiskit Aer channel.
        canonical_kraus : bool, optional
            Whether to use the canonical Kraus representation.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed amplitude-damping error, or ``None`` if ``param_amp`` is
            zero.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If ``param_amp`` has an unsupported type.
        """
        entry, p = self._resolve_params(
            kwargs, param_amp=param_amp, excited_state_population=excited_state_population, canonical_kraus=canonical_kraus,
        )
        param_amp, excited_state_population, canonical_kraus = p["param_amp"], p["excited_state_population"], p["canonical_kraus"]

        match param_amp:
            case float(n) if n == 0.0:
                return None
            case float(n):
                param_amp = param_amp
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(param_amp).__name__}"
                    )
             
        logger.debug("Adding amplitude damping noise of: %s", param_amp)
        error = amplitude_damping_error(
            param_amp=param_amp, 
            excited_state_population=excited_state_population, 
            canonical_kraus=canonical_kraus
            )
        return self._finalize(
            entry, AmplitudeDampingErrorEntry, error, instructions, qubits,
            param_amp=param_amp, excited_state_population=excited_state_population, canonical_kraus=canonical_kraus,
        )

    def add_phase_damping_error(
            self, 
            param_phase: float = 0.0, 
            canonical_kraus: bool = True,
            instructions: Any | None = None,
            qubits: list[int] | None = None, 
            **kwargs
            ) -> QuantumError | None:
        """
        Add a phase-damping error channel.

        Parameters
        ----------
        param_phase : float, optional
            Phase-damping parameter. ``0.0`` disables the error.
        canonical_kraus : bool, optional
            Whether to use the canonical Kraus representation.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed phase-damping error, or ``None`` if ``param_phase`` is zero.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If ``param_phase`` has an unsupported type.
        """
        entry, p = self._resolve_params(
            kwargs, param_phase=param_phase, canonical_kraus=canonical_kraus,
        )
        param_phase, canonical_kraus = p["param_phase"], p["canonical_kraus"]

        match param_phase:
            case float(n) if n == 0:
                return None
            case float(n):
                param_phase = param_phase
            case _:
                raise NoiseError(
                    f"Invalid format for measurement error {type(param_phase).__name__}"
                    )

        logger.debug("Adding phase damping noise of: %s", param_phase)
        error = phase_damping_error(param_phase=param_phase, canonical_kraus=canonical_kraus)
        return self._finalize(
            entry, PhaseDampingErrorEntry, error, instructions, qubits,
            param_phase=param_phase, canonical_kraus=canonical_kraus, 
        )
    
    def add_phase_amplitude_damping_error(
            self, 
            param_amp: float = 0.0, 
            param_phase: float = 0.0,
            excited_state_population: int = 0,
            canonical_kraus: bool = True,
            instructions: Any | None = None,
            qubits: list[int] | None = None, 
            **kwargs
            ) -> QuantumError | None:
        """
        Add a combined phase- and amplitude-damping error channel.

        Parameters
        ----------
        param_amp : float, optional
            Amplitude-damping parameter. ``0.0`` disables amplitude damping.
        param_phase : float, optional
            Phase-damping parameter. ``0.0`` disables phase damping.
        excited_state_population : int, optional
            Population of the excited state used by the Qiskit Aer channel.
        canonical_kraus : bool, optional
            Whether to use the canonical Kraus representation.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed phase-amplitude damping error, or ``None`` if both
            parameters are zero.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If either damping parameter has an unsupported type.
        """
        entry, p = self._resolve_params(
            kwargs, param_amp=param_amp, param_phase=param_phase, excited_state_population=excited_state_population, canonical_kraus=canonical_kraus,
        )
        param_amp, param_phase, excited_state_population, canonical_kraus = p["param_amp"], p["param_phase"], p["excited_state_population"], p["canonical_kraus"]

        match [param_amp, param_phase]:
            case [float(g), float(l)] if g == 0.0 and l == 0.0:
                return None
            case [float(g), float(l)]:
                gamma, lam = g, l
            case [float(g), _]:
                raise NoiseError(f"Invalid format for param_phase: {type(param_phase).__name__}")
            case [_, float(l)]:
                raise NoiseError(f"Invalid format for param_amp: {type(param_amp).__name__}")
            case _:
                raise NoiseError(
                    f"Invalid format for amplitude and phase parameters. Expected float, float, \
                        got {[type(x).__name__ for x in [param_amp, param_phase]]}"
                )

        logger.debug("Adding phase amplitude damping noise of: %s, %s", gamma, lam)
        error = phase_amplitude_damping_error(param_amp=gamma, param_phase=lam, excited_state_population=excited_state_population, canonical_kraus=canonical_kraus)

        return self._finalize(
            entry, PhaseAmplitudeDampingErrorEntry, error, instructions, qubits,
            param_amp=param_amp, param_phase=param_phase, excited_state_population=excited_state_population, canonical_kraus=canonical_kraus,
        )

    def add_thermal_relaxation_error(
            self, 
            t1: float = 0, 
            t2: float = 0, 
            gate_time: float = 0, 
            excited_state_population: int = 0,
            instructions: Any | None = None,
            qubits: list[int] | None = None, 
            **kwargs
            ) -> QuantumError | None:
        """
        Add a thermal-relaxation error channel.

        Parameters
        ----------
        t1 : float, optional
            Longitudinal relaxation time.
        t2 : float, optional
            Transverse relaxation time.
        gate_time : float, optional
            Duration for which the qubit undergoes thermal relaxation.
        excited_state_population : int, optional
            Equilibrium excited-state population.
        instructions : Any or None, optional
            Instructions to which the error is applied. If ``None``, the simulator's
            default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied.
        **kwargs
            Additional parameters used when reconstructing an existing
            ``ErrorEntry`` through ``entry=``.

        Returns
        -------
        qiskit_aer.noise.QuantumError or None
            Constructed thermal-relaxation error, or ``None`` if all three time
            parameters are zero.

        Raises
        ------
        qiskit_aer.noise.NoiseError
            If the parameters have unsupported types or any of the relaxation
            times are negative.
        """
        entry, p = self._resolve_params(
            kwargs, t1=t1, t2=t2, time=gate_time, excited_state_population=excited_state_population
        )
        t1, t2, gate_time, excited_state_population = p["t1"], p["t2"], p["time"], p["excited_state_population"]

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
        if t1 == t2 == gate_time == 0:
            return None
        # Optional: Add validation for values (e.g., t1 and t2 should be positive)
        if error_t1 < 0 or error_t2 < 0 or error_gate_time < 0:
            raise NoiseError(
                f"Thermal relaxation parameters must be positive. "
                f"Got t1={error_t1}, t2={error_t2}, gate_time={error_gate_time}"
            )
        
        logger.debug(
            "Adding thermal relaxation noise with t1=%s, t2=%s, gate_time=%s, excited_state_population=%s",
            t1, t2, gate_time, excited_state_population
        )
        error = thermal_relaxation_error(
            t1=t1, 
            t2=t2, 
            time=gate_time, 
            excited_state_population=excited_state_population
            )

        return self._finalize(
            entry, ThermalRelaxationErrorEntry, error, instructions, qubits, 
            t1=error_t1, t2=error_t2, time=error_gate_time, excited_state_population=excited_state_population,
        )


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_error(
            self, 
            error: QuantumError | ReadoutError, 
            instructions: Any | None = None,
            qubits: list[int] | None = None,
            ) -> None:
        """
        Register a quantum or readout error with the noise model.

        Parameters
        ----------
        error : QuantumError or ReadoutError
            Error channel to add to the noise model.
        instructions : Any or None, optional
            Instructions to which a quantum error is applied. If ``None``, the
            simulator's default single-qubit instructions are used.
        qubits : list of int or None, optional
            Qubits to which the error is applied. If ``None``, the error is applied
            to all applicable qubits.

        Raises
        ------
        ValueError
            If ``error`` is neither a ``QuantumError`` nor a ``ReadoutError``.

        Notes
        -----
        For quantum errors, the number of qubits represented by ``error`` must be
        compatible with the target instruction and qubit specification. Validation
        of these constraints is ultimately performed by Qiskit Aer.
        """
        if instructions is None: instructions = self._single_qubit_gates
        match error:
            case QuantumError():
                match qubits:
                    case None: 
                        self._noise_model.add_all_qubit_quantum_error(
                            error=error, 
                            instructions=instructions,
                            warnings=False,
                        )
                    case _: 
                        self._noise_model.add_quantum_error(
                            error,
                            instructions=instructions,
                            qubits=qubits,
                            warnings=False
                        )
            case ReadoutError():
                if qubits is None:
                    self._noise_model.add_all_qubit_readout_error(error=error, warnings=False)
                else: 
                    self._noise_model.add_readout_error(error=error, qubits=qubits, warnings=False)
            case _: 
                raise ValueError(
                    f"invalid error parameter. Expected QuantumError | ReadoutError, \
                        got {type(error).__name__}."
                        )

        return None

    def _add_entry(self, entry: ErrorEntry) -> None:
        """
        Apply an existing error entry using the corresponding noise method.

        Parameters
        ----------
        entry : ErrorEntry
            Error entry describing the noise channel to reconstruct and apply.

        Raises
        ------
        ValueError
            If the type of ``entry`` is not recognized.
        """
        match entry.__class__.__name__:
            case "ReadoutErrorEntry":
                self.add_readout_error(entry = entry)
            case "PauliErrorEntry":
                self.add_pauli_error(entry = entry)
            case "DepolarizingErrorEntry":
                self.add_depolarizing_error(entry = entry)
            case "AmplitudeDampingErrorEntry":
                self.add_amplitude_damping_error(entry = entry)
            case "PhaseDampingErrorEntry":
                self.add_phase_damping_error(entry = entry)
            case "PhaseAmplitudeDampingErrorEntry":
                self.add_phase_amplitude_damping_error(entry = entry)
            case "ThermalRelaxationErrorEntry":
                self.add_thermal_relaxation_error(entry = entry)
            case "MixedUnitaryErrorEntry":
                self.add_mixed_unitary_error(entry = entry)
            case "CoherentUnitaryErrorEntry":
                self.add_coherent_unitary_error(entry = entry)
            case _: 
                raise ValueError(f"Unable to add error: {entry}, ErrorEntry not recognized")

        return None

    def _compose_gate_error(self, error: QuantumError | None) -> None:
        """
        Compose a quantum error into the accumulated gate-error channel.

        Parameters
        ----------
        error : QuantumError or None
            Quantum error to compose with the existing gate-error channel.

        Notes
        -----
        If no previous gate error exists, ``error`` becomes the accumulated
        gate-error channel. Otherwise, the new error is composed with the existing
        channel.

        Readout errors are not processed by this method.
        """
        if error is not None: 
            if self._gate_error is None:
                self._gate_error = error
            else: 
                self._gate_error = self._gate_error.compose(error.to_quantumchannel())

    def _generate_random_unitary(self, max_offset: float) -> Operator:
        """
        Generate a random single-qubit unitary rotation.

        The generated unitary has the form

        .. math::
            U = \\exp\\left(-\\frac{i}{2}\\theta P\\right),

        where ``P`` is randomly selected from the X, Y, and Z generators and
        ``theta`` is sampled uniformly from ``[0, max_offset]``.

        Parameters
        ----------
        max_offset : float
            Maximum rotation angle.

        Returns
        -------
        qiskit.quantum_info.Operator
            Randomly generated single-qubit unitary operator.
        """
        P = SU2[["X", "Y", "Z"][np.random.randint(3)]]
        theta = np.random.uniform(0, max_offset)
        return Operator(expm(-1j/2*theta*P))

    def _resolve_params(self, kwargs: dict, **params) -> tuple[ErrorEntry | None, dict]:
        """
        Resolve noise parameters, optionally using an existing ErrorEntry.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to the public noise method. If an ``entry``
            key is present, its value is used as the source of the noise
            parameters.
        **params
            Parameter names and their currently supplied values. The keys should
            correspond to fields of the relevant ``ErrorEntry`` dataclass.

        Returns
        -------
        entry : ErrorEntry or None
            Existing error entry if one was supplied, otherwise ``None``.
        resolved : dict
            Parameter dictionary containing the values that should be used by the
            noise-construction method.

        Notes
        -----
        When ``entry=`` is supplied, values stored in the entry take precedence
        over the corresponding values passed directly to the method.

        The ``order`` field of the entry is ignored when resolving parameters.
        """
        if kwargs.get("entry") is not None:
            entry = kwargs.get("entry")
            entrydict = asdict(entry)
            entrydict.pop("order")
            resolved = {name: entrydict.get(name, default) for name, default in params.items()}
            return entry, resolved
        else: 
            entry = None
            entrydict = None
            return entry, params

    def _finalize(
            self,
            entry: ErrorEntry | None,
            entry_cls: type[ErrorEntry],
            error: QuantumError | ReadoutError,
            instructions: Any | None,
            qubits: list[int] | None,
            **fields,
            ) -> QuantumError | ReadoutError:
        """
        Register, compose, and log a newly constructed error.

        Parameters
        ----------
        entry : ErrorEntry or None
            Existing error entry being reapplied. If ``None``, a new entry is
            constructed.
        entry_cls : type of ErrorEntry
            Error-entry class to instantiate when ``entry`` is ``None``.
        error : QuantumError or ReadoutError
            Constructed Qiskit error channel.
        instructions : Any or None
            Instructions to which the error was applied.
        qubits : list of int or None
            Qubits to which the error was applied.
        **fields
            Additional fields required to construct the corresponding
            ``ErrorEntry``.

        Returns
        -------
        QuantumError or ReadoutError
            The supplied error channel.

        Notes
        -----
        The error is first added to the noise model. Quantum errors are then
        composed into ``gate_error``. Finally, the corresponding ``ErrorEntry`` is
        appended to ``noise_log`` and assigned its sequential order.
        """
        self._add_error(error=error, instructions=instructions, qubits=qubits)
        if isinstance(error, QuantumError): self._compose_gate_error(error)
        resolved_instructions = instructions if instructions is not None else list(self._single_qubit_gates)
        entry = entry or entry_cls(
            instructions=resolved_instructions,
            qubits=qubits,
            error=error,
            **fields,
        )
        entry.order = len(self._noise_log)
        self._noise_log.append(entry)
        
        return error
    

    def __str__(self) -> str:
        """
        Return a human-readable representation of the simulator and its noise log.

        Returns
        -------
        str
            Formatted string containing the simulator name and all registered
            noise-log entries.
        """
        lines = ["="*32 + "  NoisyAerSimulator  " + "="*32, "Noise logs:", "-" * 85]
        lines.extend(str(entry) for entry in self._noise_log)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Return the unambiguous representation of the simulator.

        Returns
        -------
        str
            Representation containing the number of registered noise errors.
        """
        return f"NoisyAerSimulator(n_errors={len(self._noise_log)})"

    def __len__(self) -> int:
        """
        Return the number of errors currently registered.

        Returns
        -------
        int
            Number of entries in :attr:`noise_log`.
        """
        return len(self._noise_log)

    def __iter__(self) -> iter:
        """
        Iterate over the registered noise entries.

        Returns
        -------
        iterator
            Iterator over :attr:`noise_log`.
        """
        return iter(self._noise_log)

    def __getitem__(self, index):
        """
        Return a noise-log entry by index.

        Parameters
        ----------
        index : int or slice
            Index or slice applied to the noise log.

        Returns
        -------
        ErrorEntry or list of ErrorEntry
            Corresponding noise-log entry or entries.
        """
        return self._noise_log[index]

    def __eq__(self, other) -> bool:
        """
        Compare two simulators based on their noise logs.

        Parameters
        ----------
        other : object
            Object to compare with this simulator.

        Returns
        -------
        bool
            ``True`` if both objects are ``NoisyAerSimulator`` instances with
            equal noise logs, otherwise ``False``.

        Notes
        -----
        If ``other`` is not a ``NoisyAerSimulator``, ``NotImplemented`` is returned
        so that Python can attempt the reflected comparison.
        """
        if not isinstance(other, NoisyAerSimulator):
            return NotImplemented
        return self._noise_log == other._noise_log


class ResultFull(Result):
    """
    Qiskit :class:`~qiskit.result.Result` subclass that returns complete
    computational-basis count dictionaries.

    This class is intended for results generated by AerSimulator-based
    backends. Its ``get_counts`` method extends the standard Qiskit behavior
    by adding missing bitstrings with zero counts.
    """
    def get_counts(self, experiment=None):
        """
        Return measurement counts including all possible bitstrings.

        Parameters
        ----------
        experiment : int, str, or QuantumCircuit, optional
            Experiment whose counts should be returned. Passed directly to the
            parent :meth:`qiskit.result.Result.get_counts` implementation.

        Returns
        -------
        dict
            Dictionary containing every possible computational-basis bitstring
            for the number of measured qubits. Bitstrings that were not observed
            are included with a count of ``0``.

            If the result contains no counts, the original empty counts object is
            returned unchanged.
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