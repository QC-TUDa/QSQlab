"""
Test suite for BenchmarkSequencer.
"""
import json
import time

import pytest
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend

from qsqlab.sequencer.benchmark_sequencer import BenchmarkSequencer, _filter_kwargs  # <-- adjust path


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

class FakeCounts(dict):
    """Mimics qiskit's Counts enough for .get_counts() usage."""
    pass


class FakeResult:
    def __init__(self, counts: dict):
        self._counts = FakeCounts(counts)

    def get_counts(self):
        return self._counts


class FakeJob:
    def __init__(self, counts: dict):
        self._counts = counts

    def result(self):
        return FakeResult(self._counts)


class FakeBackend(Backend):
    """Minimal stand-in backend. Not an AerSimulator, so fidelity
    computation in _get_state_vector takes the `fidelity = None` path,
    which keeps these tests independent of the AerSimulator branch."""

    def __init__(self, fixed_counts: dict | None = None, fail_on_call: int | None = None):
        # fixed_counts: counts to return every call, unless a per-call
        # sequence is desired (use `counts_sequence` instead).
        self.fixed_counts = fixed_counts or {"0": 100}
        self.counts_sequence = None
        self.call_count = 0
        self.fail_on_call = fail_on_call  # 1-indexed call number to raise on
        self.received_shots = []

    def run(self, circuits, shots):
        self.call_count += 1
        self.received_shots.append(shots)
        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise RuntimeError("simulated hardware/connection failure")
        if self.counts_sequence is not None:
            counts = self.counts_sequence[self.call_count - 1]
        else:
            counts = self.fixed_counts
        return FakeJob(counts)

    def _default_options(self):
        return None

    @property
    def target(self):
        return None

    def max_circuits(self):
        return None


@pytest.fixture
def fake_backend():
    return FakeBackend()


@pytest.fixture
def single_qubit_qc():
    return QuantumCircuit(1)


@pytest.fixture
def sequencer(fake_backend, single_qubit_qc):
    return BenchmarkSequencer(backend=fake_backend, qc=single_qubit_qc)


# ---------------------------------------------------------------------------
# Construction / property validation
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_qc_setter_rejects_non_circuit(self, sequencer):
        with pytest.raises(TypeError):
            sequencer.qc = "not a circuit"

    def test_qc_setter_accepts_valid_circuit(self, sequencer):
        new_qc = QuantumCircuit(1)
        sequencer.qc = new_qc
        assert sequencer.qc is new_qc

    def test_backend_setter_rejects_invalid_type(self, sequencer):
        with pytest.raises(TypeError):
            sequencer.backend = "not a backend"

    def test_backend_setter_accepts_valid_backend(self, sequencer, fake_backend):
        other = FakeBackend()
        sequencer.backend = other
        assert sequencer.backend is other

    def test_transpiler_config_setter_rejects_non_dict(self, sequencer):
        with pytest.raises(TypeError):
            sequencer.transpiler_config = ["not", "a", "dict"]

    def test_default_transpiler_config_is_independent_copy(self, fake_backend, single_qubit_qc):
        """Mutating one instance's config must not leak into another
        instance's default, or into the TRANSPILER_CONFIG constant."""
        from qsqlab.config.transpiler_config import TRANSPILER_CONFIG 

        seq_a = BenchmarkSequencer(backend=fake_backend, qc=single_qubit_qc)
        seq_a.transpiler_config["optimization_level"] = 999

        seq_b = BenchmarkSequencer(backend=fake_backend, qc=single_qubit_qc)
        assert seq_b.transpiler_config.get("optimization_level") != 999
        assert TRANSPILER_CONFIG.get("optimization_level") != 999

    def test_run_circuits_without_any_qc_raises(self, fake_backend):
        seq = BenchmarkSequencer(backend=fake_backend, qc=None)
        with pytest.raises(ValueError):
            seq.run_circuits(depths=[0, 1], shots=100, qc=None)


# ---------------------------------------------------------------------------
# Circuit construction / gate validation
# ---------------------------------------------------------------------------

class TestCircuitConstruction:

    def test_unsupported_gate_raises(self, sequencer):
        with pytest.raises(ValueError):
            sequencer.run_circuits(depths=[1], shots=100, gate="not_a_gate")

    def test_gate_name_that_is_a_circuit_attribute_but_not_a_gate_is_rejected(self, sequencer):
        """Regression test: hasattr(qc, 'draw') is True, so a check based on
        hasattr rather than the SINGLE_QUBIT_GATES allowlist would wrongly
        accept this. Confirms the allowlist-based check is in place."""
        with pytest.raises(ValueError):
            sequencer.run_circuits(depths=[1], shots=100, gate="draw")

    def test_parametric_gate_missing_theta_raises(self, sequencer):
        with pytest.raises(ValueError):
            sequencer.run_circuits(depths=[1], shots=100, gate="rx")  # no theta kwarg

    def test_rz_missing_phi_raises(self, sequencer):
        with pytest.raises(ValueError):
            sequencer.run_circuits(depths=[1], shots=100, gate="rz")

    def test_u_gate_missing_any_angle_raises(self, sequencer):
        with pytest.raises(ValueError):
            sequencer.run_circuits(depths=[1], shots=100, gate="u", theta=0.1, phi=0.2)  # no lam

    def test_u_gate_runs_with_all_angles(self, sequencer):
        data = sequencer.run_circuits(
            depths=[1], shots=100, gate="u", theta=0.5, phi=0.3, lam=0.1
        )
        assert data[0]["gate"] == "u"
        assert data[0]["angles"] == {"theta": 0.5, "phi": 0.3, "lam": 0.1}

    def test_invalid_measurement_basis_raises(self, sequencer):
        with pytest.raises(ValueError):
            sequencer.run_circuits(depths=[1], shots=100, measurement_basis="W")

    def test_x_gate_depth_zero_is_identity(self, sequencer):
        """depth=0 should leave the qubit in |0>."""
        data = sequencer.run_circuits(depths=[0], shots=100, gate="x")
        assert data[0]["ideal_counts"] == {"0": 1.0, "1": 0.0}

    def test_x_gate_depth_one_flips_qubit(self, sequencer):
        data = sequencer.run_circuits(depths=[1], shots=100, gate="x")
        assert data[0]["ideal_counts"] == {"0": 0.0, "1": 1.0}

    def test_x_gate_depth_two_is_identity_again(self, sequencer):
        data = sequencer.run_circuits(depths=[2], shots=100, gate="x")
        assert data[0]["ideal_counts"] == {"0": 1.0, "1": 0.0}


# ---------------------------------------------------------------------------
# depths handling
# ---------------------------------------------------------------------------

class TestDepthsHandling:

    def test_depths_are_sorted_by_default(self, sequencer):
        data = sequencer.run_circuits(depths=[3, 1, 2], shots=100, gate="x")
        assert [d["amount_of_gates"] for d in data] == [1, 2, 3]

    def test_sort_depths_false_preserves_order(self, sequencer):
        data = sequencer.run_circuits(depths=[3, 1, 2], shots=100, gate="x", sort_depths=False)
        assert [d["amount_of_gates"] for d in data] == [3, 1, 2]

    def test_run_circuits_does_not_mutate_caller_depths_list(self, sequencer):
        original = [3, 1, 2]
        passed_in = list(original)
        sequencer.run_circuits(depths=passed_in, shots=100, gate="x")
        assert passed_in == original, "run_circuits must not mutate the caller's list in place"


# ---------------------------------------------------------------------------
# backend interaction / counts
# ---------------------------------------------------------------------------

class TestBackendInteraction:

    def test_shots_forwarded_to_backend(self, sequencer, fake_backend):
        sequencer.run_circuits(depths=[1], shots=4321, gate="x")
        assert fake_backend.received_shots == [4321]

    def test_counts_come_from_backend(self, single_qubit_qc):
        backend = FakeBackend(fixed_counts={"1": 100})
        seq = BenchmarkSequencer(backend=backend, qc=single_qubit_qc)
        data = seq.run_circuits(depths=[1], shots=100, gate="x")
        assert data[0]["counts"] == {"1": 100}

    def test_one_backend_call_per_depth(self, sequencer, fake_backend):
        sequencer.run_circuits(depths=[0, 1, 2, 3], shots=100, gate="x")
        assert fake_backend.call_count == 4

    def test_backend_type_recorded_in_output(self, sequencer, fake_backend):
        data = sequencer.run_circuits(depths=[1], shots=100, gate="x")
        assert data[0]["backend"] == type(fake_backend).__name__


# ---------------------------------------------------------------------------
# In-memory circuits accumulation
# ---------------------------------------------------------------------------

class TestCircuitsAccumulation:

    def test_data_property_is_flat_list_of_dicts(self, sequencer):
        """Regression test: self._data.append(sequencer_data) appends the
        whole per-call list as a single element, producing a list of lists
        rather than a flat list of dicts (breaks the documented
        `list[dict]` return type of the `data` property).
        Expected fix: use `self._data.extend(sequencer_data)`."""
        sequencer.run_circuits(depths=[0, 1], shots=100, gate="x")
        assert all(isinstance(item, dict) for item in sequencer.data), (
            "data should be a flat list[dict]; got nested list(s) instead. "
            "Change self._data.append(sequencer_data) to .extend(sequencer_data)."
        )

    def test_data_accumulate_across_multiple_calls(self, sequencer):
        sequencer.run_circuits(depths=[0], shots=100, gate="x")
        sequencer.run_circuits(depths=[1], shots=100, gate="x")
        assert len(sequencer.data) == 2


# ---------------------------------------------------------------------------
# Persistence / crash-safety (the main design goal of this rework)
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_save_creates_jsonl_file(self, sequencer, tmp_path):
        sequencer.run_circuits(
            depths=[0, 1], shots=100, gate="x",
            save=True, savepath=str(tmp_path), savename="run1",
        )
        out = tmp_path / "run1.jsonl"
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # must be valid JSON per line

    def test_save_false_writes_no_file(self, sequencer, tmp_path):
        sequencer.run_circuits(depths=[0, 1], shots=100, gate="x", save=False)
        assert list(tmp_path.iterdir()) == []

    def test_repeated_run_without_append_creates_new_file(self, sequencer, tmp_path):
        sequencer.run_circuits(
            depths=[0], shots=100, gate="x",
            save=True, savepath=str(tmp_path), savename="run1",
        )
        sequencer.run_circuits(
            depths=[0], shots=100, gate="x",
            save=True, savepath=str(tmp_path), savename="run1",
        )
        files = sorted(p.name for p in tmp_path.iterdir())
        assert files == ["run1(1).jsonl", "run1.jsonl"]

    def test_append_data_reuses_same_file(self, sequencer, tmp_path):
        sequencer.run_circuits(
            depths=[0], shots=100, gate="x",
            save=True, savepath=str(tmp_path), savename="run1", append_data=True,
        )
        sequencer.run_circuits(
            depths=[1], shots=100, gate="x",
            save=True, savepath=str(tmp_path), savename="run1", append_data=True,
        )
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        lines = files[0].read_text().strip().splitlines()
        assert len(lines) == 2

    def test_partial_results_survive_a_mid_sequence_backend_failure(self, single_qubit_qc, tmp_path):
        """This is the core crash-safety guarantee: if the backend fails on
        (say) the 3rd depth, the first two depths' data must already be on
        disk, not lost."""
        backend = FakeBackend(fail_on_call=3)
        seq = BenchmarkSequencer(backend=backend, qc=single_qubit_qc)

        with pytest.raises(RuntimeError):
            seq.run_circuits(
                depths=[0, 1, 2, 3, 4], shots=100, gate="x",
                save=True, savepath=str(tmp_path), savename="crash_run",
            )

        out = tmp_path / "crash_run.jsonl"
        assert out.exists()
        lines = out.read_text().strip().splitlines()
        # depths 0 and 1 should have completed and been flushed before the
        # failure on the 3rd call (depth=2)
        assert len(lines) == 2
        recorded_depths = [json.loads(l)["amount_of_gates"] for l in lines]
        assert recorded_depths == [0, 1]

    def test_partial_results_also_present_in_memory_after_failure(self, single_qubit_qc):
        """Companion to the disk-persistence test: self.data should also
        reflect completed depths even if a later depth in the same call
        raises. This only holds if self._data is updated inside the loop
        rather than once at the end via .extend() after the loop completes."""
        backend = FakeBackend(fail_on_call=2)
        seq = BenchmarkSequencer(backend=backend, qc=single_qubit_qc)

        with pytest.raises(RuntimeError):
            seq.run_circuits(depths=[0, 1, 2], shots=100, gate="x")
        assert len(seq.data) == 1


# ---------------------------------------------------------------------------
# _filter_kwargs utility
# ---------------------------------------------------------------------------

class TestFilterKwargs:

    def test_filters_out_unrecognized_kwargs(self):
        def target(a, b, c=3):
            pass
        result = _filter_kwargs(target, {"a": 1, "b": 2, "z": 99})
        assert result == {"a": 1, "b": 2}

    def test_passes_through_everything_if_func_accepts_var_keyword(self):
        def target(**kwargs):
            pass
        result = _filter_kwargs(target, {"a": 1, "z": 99})
        assert result == {"a": 1, "z": 99}

    def test_empty_kwargs_returns_empty(self):
        def target(a, b):
            pass
        assert _filter_kwargs(target, {}) == {}


# ---------------------------------------------------------------------------
# Transpile path
# ---------------------------------------------------------------------------

class TestTranspile:

    def test_transpile_flag_uses_returned_circuit(self, sequencer, monkeypatch):
        """_transpile must return a circuit and run_circuits must use the
        RETURNED circuit, not silently discard it."""
        marker_qc = QuantumCircuit(1)
        marker_qc.name = "transpiled_marker"

        def fake_transpile(self, qc, backend, **kwargs):
            return marker_qc

        monkeypatch.setattr(BenchmarkSequencer, "_transpile", fake_transpile)

        # Spy on backend.run to see which circuit object actually gets executed
        received = {}
        original_run = sequencer.backend.run

        def spy_run(circuits, shots):
            received["qc"] = circuits[0]
            return original_run(circuits, shots)

        sequencer.backend.run = spy_run

        sequencer.run_circuits(depths=[1], shots=100, gate="x", transpile=True)
        assert received["qc"].name == "transpiled_marker"

    def test_transpile_config_merge_uses_merged_config(self, sequencer, monkeypatch):
        """Regression test: _transpile computes a merged local config
        (_transpiler_config) that layers per-call overrides on top of
        self._transpiler_config, but the final qiskit.transpile call in the
        current code uses `**self._transpiler_config` instead of the merged
        local variable, so overrides are silently ignored. This test calls
        qiskit.transpile via a stub and checks which config was actually
        passed through."""
        captured_kwargs = {}

        def fake_qiskit_transpile(circuits, backend, **kwargs):
            captured_kwargs.update(kwargs)
            return circuits

        import qsqlab.sequencer.benchmark_sequencer as module  # adjust path
        monkeypatch.setattr(module.qiskit, "transpile", fake_qiskit_transpile)

        sequencer.transpiler_config = {"optimization_level": 1}
        sequencer.run_circuits(
            depths=[1], shots=100, gate="x",
            transpile=True, optimization_level=3,
        )
        assert captured_kwargs.get("optimization_level") == 3, (
            "Per-call optimization_level=3 override was not applied; "
            "_transpile likely passed self._transpiler_config instead of "
            "the locally merged _transpiler_config to qiskit.transpile()."
        )