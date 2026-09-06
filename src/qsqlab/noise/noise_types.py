from dataclasses import dataclass, field, asdict, fields
import numpy as np

from qiskit.quantum_info.operators import Operator
from qiskit_aer.noise import QuantumError, ReadoutError


# ---------------------------------------------------------------------------
# Base record — common to every noise source, regardless of type
# ---------------------------------------------------------------------------
@dataclass
class ErrorEntry:
    """Common metadata recorded for every noise source added to the model."""
    error: QuantumError | ReadoutError = None  # the actual constructed error object
    order: int = field(default=-1, init=False)
    qubits: list[int] = None                    # None => interpreted as "all qubits"
    instructions: list[str] = None              # None => interpreted as "all instructions" 

    @property
    def kind(self) -> str:
        return self.__class__.__name__

    def __str__(self):
        field_names = [f.name for f in fields(self)]
        ordered = [n for n in field_names if n not in ("error", "instructions")]
        ordered.append("instructions")

        parts = [f"{name}={_format_value(getattr(self, name))!r}" for name in ordered]
        return f"{type(self).__name__}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Readout error
# ---------------------------------------------------------------------------
@dataclass
class ReadoutErrorEntry(ErrorEntry):
    """qiskit_aer.noise.ReadoutError(probabilities)"""
    probabilities: list[list[float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pauli channel
# ---------------------------------------------------------------------------
@dataclass
class PauliErrorEntry(ErrorEntry):
    """qiskit_aer.noise.pauli_error(noise_ops, standard_gates=None)"""
    noise_ops: list[tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Depolarizing channel
# ---------------------------------------------------------------------------
@dataclass
class DepolarizingErrorEntry(ErrorEntry):
    """qiskit_aer.noise.depolarizing_error(param)"""
    param: float = 0.0
    num_qubits: int = 1


# ---------------------------------------------------------------------------
# Amplitude damping
# ---------------------------------------------------------------------------
@dataclass
class AmplitudeDampingErrorEntry(ErrorEntry):
    """qiskit_aer.noise.amplitude_damping_error(param_amp, excited_state_population=0, canonical_kraus=True)"""
    param_amp: float = 0.0
    excited_state_population: float = 0.0
    canonical_kraus: bool = True


# ---------------------------------------------------------------------------
# Phase damping
# ---------------------------------------------------------------------------
@dataclass
class PhaseDampingErrorEntry(ErrorEntry):
    """qiskit_aer.noise.phase_damping_error(param_phase, canonical_kraus=True)"""
    param_phase: float = 0.0
    canonical_kraus: bool = True


# ---------------------------------------------------------------------------
# Combined phase + amplitude damping
# ---------------------------------------------------------------------------
@dataclass
class PhaseAmplitudeDampingErrorEntry(ErrorEntry):
    """qiskit_aer.noise.phase_amplitude_damping_error(param_amp, param_phase,
    excited_state_population=0, canonical_kraus=True)"""
    param_amp: float = 0.0
    param_phase: float = 0.0
    excited_state_population: float = 0.0
    canonical_kraus: bool = True


# ---------------------------------------------------------------------------
# Thermal relaxation
# ---------------------------------------------------------------------------
@dataclass
class ThermalRelaxationErrorEntry(ErrorEntry):
    """qiskit_aer.noise.thermal_relaxation_error(t1, t2, time, excited_state_population=0)"""
    t1: float = 0.0
    t2: float = 0.0
    time: float = 0.0
    excited_state_population: float = 0.0


# ---------------------------------------------------------------------------
# Mixed unitary (incoherent mixture of unitaries)
# ---------------------------------------------------------------------------
@dataclass
class MixedUnitaryErrorEntry(ErrorEntry):
    """qiskit_aer.noise.mixed_unitary_error(noise_ops)
    noise_ops: list of (unitary_matrix_or_Operator, probability) pairs"""
    noise_ops: float | list[tuple[np.ndarray | Operator, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Coherent unitary (single deterministic unitary error)
# ---------------------------------------------------------------------------
@dataclass
class CoherentUnitaryErrorEntry(ErrorEntry):
    """qiskit_aer.noise.coherent_unitary_error(unitary)"""
    gate_amplitude_map: float | dict[str, float] | Operator = 0.0



def _format_value(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        converted = [_format_value(v) for v in val]
        return converted if isinstance(val, list) else tuple(converted)
    if isinstance(val, dict):
        return {k: _format_value(v) for k, v in val.items()}
    return val