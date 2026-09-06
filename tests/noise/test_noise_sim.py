"""
Test suite for NoisyAerSimulator.
"""
import pytest
from qiskit import QuantumCircuit
from qiskit_aer.noise.noiseerror import NoiseError

from qsqlab.noise import NoisyAerSimulator


def test_initial_simulator_is_ideal():
    sim = NoisyAerSimulator()

    assert len(sim) == 0
    assert sim.noise_log == []
    assert sim.gate_error is None


def test_zero_noise_does_not_add_error():
    sim = NoisyAerSimulator()

    assert sim.add_depolarizing_error(0.0) is None
    assert sim.add_readout_error(0.0) is None

    assert len(sim) == 0


def test_depolarizing_error_is_logged():
    sim = NoisyAerSimulator()

    error = sim.add_depolarizing_error(0.1)

    assert error is not None
    assert len(sim) == 1
    assert sim.noise_log[0].order == 0
    assert sim.gate_error is not None


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_depolarizing_error_rejects_invalid_probability(probability):
    sim = NoisyAerSimulator()

    with pytest.raises(NoiseError):
        sim.add_depolarizing_error(probability)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_readout_error_rejects_invalid_probability(probability):
    sim = NoisyAerSimulator()

    with pytest.raises(NoiseError):
        sim.add_readout_error(probability)


def test_readout_error_accepts_state_dependent_probabilities():
    sim = NoisyAerSimulator()

    error = sim.add_readout_error([0.1, 0.2])

    assert error is not None
    assert len(sim) == 1


def test_pauli_error_is_logged():
    sim = NoisyAerSimulator()

    error = sim.add_pauli_error({
        "X": 0.01,
        "Y": 0.02,
        "Z": 0.03,
    })

    assert error is not None
    assert len(sim) == 1
    assert sim.noise_log[0].noise_ops == {
        "X": 0.01,
        "Y": 0.02,
        "Z": 0.03,
    }


def test_pauli_probabilities_cannot_sum_above_one():
    sim = NoisyAerSimulator()

    with pytest.raises(NoiseError):
        sim.add_pauli_error({
            "X": 0.6,
            "Y": 0.6,
        })


def test_thermal_relaxation_rejects_negative_parameters():
    sim = NoisyAerSimulator()

    with pytest.raises(NoiseError):
        sim.add_thermal_relaxation_error(
            t1=-1.0,
            t2=2.0,
            gate_time=0.1,
        )


def test_clear_errors_removes_noise():
    sim = NoisyAerSimulator()
    sim.add_depolarizing_error(0.1)

    assert len(sim) == 1

    sim.clear_errors()

    assert len(sim) == 0
    assert sim.noise_log == []
    assert sim.noise_model is not None

    assert sim.gate_error is None


def test_errors_can_be_added_to_specific_qubits():
    sim = NoisyAerSimulator()

    sim.add_depolarizing_error(
        0.1,
        instructions=["x"],
        qubits=[0],
    )

    assert len(sim) == 1
    assert sim.noise_log[0].instructions == ["x"]
    assert sim.noise_log[0].qubits == [0]


def test_errors_can_be_added_to_multiple_qubits():
    sim = NoisyAerSimulator()

    sim.add_depolarizing_error(
        0.1,
        num_qubits=2,
        instructions=["cx"],
        qubits=[0, 1],
    )

    assert len(sim) == 1
    assert sim.noise_log[0].qubits == [0, 1]


def test_add_error_entries_rejects_invalid_input():
    sim = NoisyAerSimulator()

    with pytest.raises(ValueError):
        sim.add_error_entries("not an error entry")


def test_add_error_entries_can_clear_previous_errors():
    sim = NoisyAerSimulator()

    sim.add_depolarizing_error(0.1)
    assert len(sim) == 1

    previous_entry = sim.noise_log[0]

    sim.add_error_entries(
        previous_entry,
        clear_previous=True,
    )

    assert len(sim) == 1
    assert sim.noise_log[0].order == 0


def test_len_iteration_and_indexing():
    sim = NoisyAerSimulator()

    sim.add_depolarizing_error(0.1)
    sim.add_amplitude_damping_error(0.2)

    assert len(sim) == 2
    assert list(sim) == sim.noise_log
    assert sim[0] is sim.noise_log[0]
    assert sim[1] is sim.noise_log[1]


def test_repr_contains_number_of_errors():
    sim = NoisyAerSimulator()
    sim.add_depolarizing_error(0.1)

    assert repr(sim) == "NoisyAerSimulator(n_errors=1)"


def test_equality_depends_on_noise_log():
    sim1 = NoisyAerSimulator()
    sim2 = NoisyAerSimulator()

    assert sim1 == sim2

    sim1.add_depolarizing_error(0.1)

    assert sim1 != sim2


def test_add_entry_rejects_unknown_entry():
    sim = NoisyAerSimulator()

    class FakeEntry:
        pass

    with pytest.raises(ValueError):
        sim._add_entry(FakeEntry())


def test_random_unitary_generation():
    sim = NoisyAerSimulator()

    unitary = sim._generate_random_unitary(0.1)

    assert unitary is not None
    assert unitary.dim == (2, 2)