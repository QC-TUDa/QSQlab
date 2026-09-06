"""benchmark_sequencer file for both simulation and hardware Qiskit backends
"""

import time
import json
import inspect
import logging
logger = logging.getLogger(__name__)

import numpy as np

from pathlib import Path
from typing import Literal, Any

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend, BackendV2
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import average_gate_fidelity, Operator, SuperOp
from qiskit_aer import AerSimulator

from ..config.data_config import DEFAULT_DATA_ROOT
from ..config.gate_config import SINGLE_QUBIT_GATES
from ..config.transpiler_config import TRANSPILER_CONFIG
from ..noise.noise_sim import ResultFull


class BenchmarkSequencer():
    """Sequencer for running single-qubit gate sequences on a Qiskit backend.
 
    Builds a sequence of identical single-qubit gates at increasing circuit
    depths, runs each circuit on a simulation or hardware `Backend`, and
    records the measured counts alongside the ideal (noiseless) probability
    distribution and, where available, a gate fidelity estimate.
 
    Results are collected in-memory in :attr:`data` and, optionally, written
    incrementally to a JSON Lines (``.jsonl``) file so that a run interrupted
    partway through (backend failure, keyboard interrupt, etc.) still leaves
    the completed depths safely on disk.
 
    Attributes
    ----------
    backend : Backend | BackendV2
        The qiskit backend (simulator or hardware) circuits are run against.
    qc : QuantumCircuit | None
        The base single-qubit circuit that gate sequences are appended to.
        Must be set (either at construction or via :meth:`run_circuits`'s
        ``qc`` argument) before :meth:`run_circuits` can be called.
    data : list[dict]
        Flat, accumulating list of result dictionaries, one per
        ``(gate, depth)`` combination run so far across all calls to
        :meth:`run_circuits`.
    transpiler_config : dict[str, Any]
        Keyword arguments forwarded to ``qiskit.transpile`` when
        ``transpile=True`` is passed to :meth:`run_circuits`.
 
    Notes
    -----
    This class assumes a single-qubit circuit throughout: ideal probabilities
    are always reported under the ``'0'``/``'1'`` keys. Passing a multi-qubit
    ``qc`` is not validated against and will not produce meaningful results.
    """
    def __init__(
            self,
            backend: Backend | BackendV2,
            qc: QuantumCircuit | None = None,
            *,
            transpiler_config: dict[str, Any] | None = None,
        ):
        """ Constructor of BenchmarkSequencer class for single qubit circuits.
 
        Parameters
        ----------
        backend : Backend | BackendV2
            qiskit backend for simulator or hardware
        qc : QuantumCircuit | None, optional
            initial quantum circuit to apply sequence of gates, by default None
        transpiler_config : dict[str, Any] | None, optional
            Keyword arguments to pass to ``qiskit.transpile`` whenever
            ``transpile=True`` is used in :meth:`run_circuits`. A private
            copy of this dict (or of the module-level ``TRANSPILER_CONFIG``
            default, if omitted) is stored, so mutating the config on one
            instance never affects another instance or the shared default.
            By default None.
        """
        self._backend: Backend | BackendV2 = backend
        self._qc: QuantumCircuit | None = qc

        self._data: list[dict] = []
        self._transpiler_config: dict[str, Any] = (transpiler_config or TRANSPILER_CONFIG).copy()

    @property
    def qc(self) -> QuantumCircuit | None:
        """QuantumCircuit | None: The base circuit gate sequences are built on.
 
        Returns
        -------
        QuantumCircuit | None
            The currently configured base circuit, or ``None`` if one has
            not been set yet.
        """
        return self._qc

    @property
    def backend(self) -> Backend | BackendV2:
        """Backend | BackendV2: The qiskit backend circuits are run against.
 
        Returns
        -------
        Backend | BackendV2
            The currently configured simulator or hardware backend.
        """
        return self._backend

    @property
    def data(self) -> list[dict]:
        """list[dict]: Accumulated result dictionaries from all runs so far.
 
        Returns
        -------
        list[dict]
            Flat list of per-``(gate, depth)`` result dictionaries, in the
            order they were produced across every call to
            :meth:`run_circuits` made on this instance. See
            :meth:`_create_sequencer_line` for the shape of each entry.
        """
        return self._data

    @property
    def transpiler_config(self) -> dict[str, Any]:
        """dict[str, Any]: Keyword arguments used for ``qiskit.transpile``.
 
        Returns
        -------
        dict[str, Any]
            The current transpiler configuration dictionary.
        """
        return self._transpiler_config
    
    @qc.setter
    def qc(self, qc: QuantumCircuit):
        """Set the base circuit that gate sequences are appended to.
 
        Parameters
        ----------
        qc : QuantumCircuit
            The new base circuit.
 
        Raises
        ------
        TypeError
            If ``qc`` is not a :class:`~qiskit.QuantumCircuit` instance.
        """
        if not isinstance(qc, QuantumCircuit): 
            raise TypeError(f"qc must be a QuantumCircuit, not {type(qc).__name__}")
        self._qc = qc

    @backend.setter
    def backend(self, backend: Backend | BackendV2):
        """Set the backend that circuits are run against.
 
        Parameters
        ----------
        backend : Backend | BackendV2
            The new simulator or hardware backend.
 
        Raises
        ------
        TypeError
            If ``backend`` is not a :class:`~qiskit.providers.backend.Backend`
            or :class:`~qiskit.providers.backend.BackendV2` instance.
        """
        if not isinstance(backend, (Backend, BackendV2)):
            raise TypeError(f"backend must be a Backend or BackendV2 class, not {type(backend).__name__}")
        self._backend = backend

    @data.setter
    def data(self, data: list[dict]):
        """Replace the accumulated results list wholesale.
 
        Parameters
        ----------
        data : list[dict]
            The new results list. Typically used to reset accumulated
            results (e.g. ``sequencer.data = []``) rather than to inject
            arbitrary data.
 
        Raises
        ------
        TypeError
            If ``data`` is not a ``list``.
        """
        if not isinstance(data, list):
            raise TypeError(f"data must be a list[dict] class, not {type(data).__name__}")
        self._data = data

    @transpiler_config.setter
    def transpiler_config(self, config: dict[str, Any]):
        """Replace the transpiler configuration wholesale.
 
        Parameters
        ----------
        config : dict[str, Any]
            New keyword arguments to forward to ``qiskit.transpile``.
 
        Raises
        ------
        TypeError
            If ``config`` is not a ``dict``.
        """
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dictionary, not {type(config).__name__}")
        self._transpiler_config = config

    def run_circuits(
            self,
            depths: list[int],
            shots: int, 
            gate: str = "sqrtx",
            qubit: int = 0,
            *,
            qc: QuantumCircuit | None = None,
            measurement_basis: Literal["X", "Y", "Z"] | None = None,
            barriers: bool = False, 
            sort_depths: bool = True,
            transpile: bool = False,
            #
            save: bool = False, 
            savepath: str | None = None, 
            savename: str | None = None,
            append_data: bool = False,
            #
            **kwargs
            ) -> list[dict]: 
        """Build, run, and record a sequence of single-qubit circuits.
 
        For each depth in ``depths``, appends ``depth`` copies of ``gate``
        (applied to ``qubit``) onto a fresh copy of :attr:`qc`, optionally
        transpiles it for :attr:`backend`, runs it for ``shots`` shots, and
        records the measured counts together with the ideal (noiseless)
        probabilities and an estimated gate fidelity (see
        :meth:`_get_state_vector`).
 
        Each depth's result is appended to :attr:`data` as it completes, and
        (if ``save=True``) written to disk as one JSON line immediately
        after that depth finishes — so results already collected are not
        lost if a later depth in the same call raises an exception.
 
        Parameters
        ----------
        depths : list[int]
            Circuit depths (number of gate repetitions) to run. Not mutated
            in place; a sorted copy is used internally when
            ``sort_depths=True``.
        shots : int
            Number of shots to run for every circuit.
        gate : str, optional
            Name of the single-qubit gate to repeat. Case-insensitive. Must
            be a member of ``SINGLE_QUBIT_GATES`` or one of the derived
            gates ``"sqrtx"``, ``"sqrty"``, ``"sqrtz"`` (implemented as
            ``rx``/``ry``/``rz`` rotations of ``pi/2``). By default
            ``"sqrtx"``.
        qubit : int, optional
            Index of the qubit the gate sequence is applied to, by default 0.
        qc : QuantumCircuit | None, optional
            If given, replaces :attr:`qc` (via the ``qc`` setter) before
            building circuits for this call. If omitted, the
            already-configured :attr:`qc` is used; a :class:`ValueError` is
            raised if neither is available. By default None.
        measurement_basis : {"X", "Y", "Z"} | None, optional
            Basis to rotate into before measurement. ``"Z"`` (or ``None``)
            measures in the computational basis directly. By default None.
        barriers : bool, optional
            If True, insert a barrier after each gate application (useful
            for visual inspection / preventing transpiler gate fusion
            across repetitions), by default False.
        sort_depths : bool, optional
            If True, run depths in ascending order (does not mutate the
            caller's ``depths`` list). By default True.
        transpile : bool, optional
            If True, transpile each circuit for :attr:`backend` using
            :attr:`transpiler_config` (optionally overridden per-call via
            ``**kwargs``, see :meth:`_transpile`) before running it. By
            default False.
        save : bool, optional
            If True, create (or append to) a ``.jsonl`` file and write each
            depth's result line to it as soon as that depth completes. By
            default False.
        savepath : str | None, optional
            Directory to save the ``.jsonl`` file in. Defaults to
            ``DEFAULT_DATA_ROOT`` if omitted. Ignored if ``save=False``.
        savename : str | None, optional
            Base filename (without extension) for the ``.jsonl`` file.
            Defaults to a timestamp if omitted. Ignored if ``save=False``.
        append_data : bool, optional
            If True and ``save=True``, append to an existing file with this
            ``savename`` instead of creating a new, uniquely-suffixed file.
            By default False.
        **kwargs
            Additional keyword arguments. Gate-parameter keywords
            (``theta`` for ``rx``/``ry``/``p``; ``phi`` for ``rz``;
            ``theta``, ``phi``, ``lam`` for ``u``) are required for their
            respective gates and are also recorded under each result's
            ``"angles"`` key. Any keys also accepted by ``qiskit.transpile``
            are additionally used to override :attr:`transpiler_config` for
            this call when ``transpile=True`` (see :meth:`_transpile`).
 
        Returns
        -------
        list[dict]
            The result dictionaries produced by this call only, in the
            order the depths were run (see :meth:`_create_sequencer_line`
            for the shape of each entry). This is also appended,
            depth-by-depth, onto :attr:`data`.
 
        Raises
        ------
        ValueError
            If no circuit is available (neither ``qc`` nor a previously-set
            :attr:`qc`), if ``gate`` is not a supported gate name, if a
            required gate-parameter keyword is missing, or if
            ``measurement_basis`` is not one of ``"X"``, ``"Y"``, ``"Z"``.
 
        Notes
        -----
        Because :attr:`data` and the ``.jsonl`` file are both updated
        incrementally within the loop, an exception raised while running a
        given depth (e.g. a hardware/network failure) does not discard the
        results already obtained for earlier depths in the same call.
        """
        sequencer_data = []
        if sort_depths: depths = sorted(depths)

        if qc is not None: 
            self.qc = qc
        elif self._qc is None and qc is None: 
            raise ValueError("No QuantumCircuit provided to run sequences")

        if save: 
            file_path : Path = self._save(savepath, savename, append_data)
        else: 
            file_path = None

        # if not append_data: TODO
        #     self.data = []

        for depth in depths:
            qc, iqc = self._create_single_qubit_circuit(
                qc=self._qc.copy(), 
                gate=gate, 
                qubit=qubit, 
                depth=depth, 
                measurement_basis=measurement_basis,
                barriers=barriers,
                **kwargs
                )
            
            if transpile: 
                qc = self._transpile(qc=qc, backend=self.backend, **kwargs)

            fidelity, ideal_counts = self._get_state_vector(qc, iqc)
            qc.measure_all()

            counts = self._run_counts(qc, shots)
            
            sequencer_line = self._create_sequencer_line(
                    counts=counts, 
                    ideal_counts=ideal_counts,
                    gate=gate, 
                    measurement_basis=measurement_basis, 
                    depth=depth, 
                    shots=shots, 
                    fidelity=fidelity, 
                    save=save,
                    file_path=file_path,
                    **kwargs,
                    )
            sequencer_data.append(sequencer_line)
        
            self.data.append(sequencer_line)

        return sequencer_data

    def add_transpiler_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Replace :attr:`transpiler_config` with a copy of ``config``.
 
        Parameters
        ----------
        config : dict[str, Any]
            New transpiler keyword arguments. A copy is stored so that
            further external mutation of ``config`` after this call does
            not affect this instance.
 
        Returns
        -------
        dict[str, Any]
            The stored (copied) transpiler configuration, i.e.
            :attr:`transpiler_config` after the update.
        """
        self.transpiler_config = config.copy()
        return self.transpiler_config

    def _transpile(self, qc, backend, **kwargs) -> QuantumCircuit:
        """Transpile a circuit for ``backend``, applying per-call overrides.
 
        Starts from a copy of :attr:`transpiler_config` and overrides any
        keys in it that also appear (with a different value) among
        ``kwargs``-supplied keyword arguments accepted by
        ``qiskit.transpile`` (as determined by :func:`_filter_kwargs`).
        Keys present in ``kwargs`` but *not* already present in
        :attr:`transpiler_config` are not added — only pre-configured keys
        can be overridden per-call.
 
        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to transpile.
        backend : Backend | BackendV2
            Target backend to transpile for.
        **kwargs
            Candidate keyword arguments; only those both accepted by
            ``qiskit.transpile`` and already present in
            :attr:`transpiler_config` are used to override the
            corresponding config values for this call.
 
        Returns
        -------
        QuantumCircuit
            The transpiled circuit returned by ``qiskit.transpile``.
        """
        transpiler_kwargs = _filter_kwargs(qiskit.transpile, kwargs)
        _transpiler_config = self._transpiler_config.copy()
        for i in transpiler_kwargs:
            if i in self._transpiler_config: 
                _transpiler_config[i] = transpiler_kwargs.get(i)
        return qiskit.transpile(circuits=qc, backend=backend, **_transpiler_config)

    def _get_state_vector(self, qc, iqc, tol: float = 1e-10):
        """Compute an estimated gate fidelity and the ideal outcome probabilities.
 
        The fidelity is computed differently depending on the backend:
 
        - If :attr:`backend` is an ``AerSimulator`` exposing a ``gate_error``
          attribute, such as this projects ``NoisyAerSimulator`` (a noisy 
          simulator's composed error channel), the average gate fidelity between
          that error channel and the identity operator is used. This represents 
          the fidelity contributed by the simulator's configured noise model, 
          independent of the specific circuit run.
        - If :attr:`backend` is an ``AerSimulator`` without a ``gate_error``
          attribute (i.e. an ideal/noiseless simulator), the fidelity is
          instead computed directly between the executed circuit ``qc`` and
          the noiseless reference circuit ``iqc`` via their unitary
          ``Operator`` representations.
        - For any other backend (real hardware, or a non-``AerSimulator``
          backend), fidelity is left as ``None``, since neither of the
          above sources of ground truth is available.
 
        Ideal outcome probabilities are always computed by simulating
        ``qc`` as a statevector (before measurement) and reading off the
        probabilities of the ``'0'`` and ``'1'`` outcomes.
 
        Parameters
        ----------
        qc : QuantumCircuit
            The (possibly transpiled) circuit that will actually be run,
            without final measurements yet appended. Used both for the
            noiseless-simulator fidelity comparison and to compute ideal
            outcome probabilities.
        iqc : QuantumCircuit
            The reference circuit built from the same gate sequence with no
            noise/transpilation applied, used only for the noiseless
            ``Operator`` fidelity comparison.
        tol : float, optional
            Probabilities with absolute value below this tolerance are
            rounded down to exactly ``0.0`` before reporting, to avoid
            reporting numerical noise as a nonzero probability. By default
            ``1e-10``.
 
        Returns
        -------
        fidelity : float | None
            The estimated gate fidelity as described above, or ``None`` if
            no fidelity estimate is available for this backend.
        ideal_counts : dict[str, float]
            Dictionary with keys ``'0'`` and ``'1'`` giving the ideal
            (noiseless) probability of each outcome, rounded to 10 decimal
            places. Both keys are always present, defaulting to ``0.0`` if
            a given outcome has zero probability.
 
        Notes
        -----
        This assumes a single-qubit circuit; probabilities for any
        additional qubits are not reported.
        """
        if isinstance(self.backend, AerSimulator):
            if hasattr(self.backend, "gate_error"):
                error: SuperOp | None = self.backend.gate_error
                fidelity = average_gate_fidelity(error or Operator(np.eye(2)), Operator(np.eye(2)))
            else: 
                fidelity = average_gate_fidelity(Operator(qc), Operator(iqc))
        else: fidelity = None

        state = Statevector.from_instruction(
            qc.copy().remove_final_measurements(inplace=False)
        )
        
        probs = {
            str(k): 0.0 if abs(float(v)) < tol else float(v)
            for k, v in state.probabilities_dict().items()
        }
        
        # ensure both outcomes exist (single qubit case)
        return fidelity, {'0': round(probs.get('0', 0.0), 10), '1': round(probs.get('1', 0.0), 10)}

    def _create_single_qubit_circuit(
            self, 
            qc: QuantumCircuit,
            gate: str,
            qubit: int,
            depth: int, 
            measurement_basis: Literal["X", "Y", "Z"] | None,
            barriers: bool,
            **gate_kwargs,
            ) -> QuantumCircuit:
        """Build a circuit (and its noiseless reference) of the given depth.
 
        Applies ``gate`` to ``qubit`` a total of ``depth`` times on both
        ``qc`` (the circuit that will actually be run) and a freshly-copied
        ideal reference circuit, so the two stay in lockstep. Derived gates
        ``"sqrtx"``, ``"sqrty"``, ``"sqrtz"`` are implemented as ``rx``,
        ``ry``, ``rz`` rotations of angle ``pi/2`` respectively. An optional
        basis-change is appended to both circuits before returning, to
        prepare for measurement in the requested basis.
 
        Parameters
        ----------
        qc : QuantumCircuit
            Base circuit to build the sequence on (mutated in place and
            returned).
        gate : str
            Name of the gate to repeat. Case-insensitive; must be a member
            of ``SINGLE_QUBIT_GATES`` or one of ``"sqrtx"``, ``"sqrty"``,
            ``"sqrtz"``.
        qubit : int
            Index of the qubit to apply the gate to.
        depth : int
            Number of times to repeat the gate. ``depth=0`` returns the
            circuits unmodified (aside from any measurement-basis rotation).
        measurement_basis : {"X", "Y", "Z"} | None
            Basis to rotate into before measurement. ``"Z"`` or ``None``
            leaves the circuit as-is (computational basis); ``"X"`` appends
            an ``H`` gate; ``"Y"`` appends ``Sdg`` then ``H``.
        barriers : bool
            If True, insert a barrier on ``qc`` (not on the ideal reference
            circuit) after each gate application.
        **gate_kwargs
            Gate-parameter keywords. Required depending on ``gate``:
            ``theta`` for ``rx``/``ry``/``p``; ``phi`` for ``rz``;
            ``theta``, ``phi``, and ``lam`` for ``u``. Not used by any
            other gate, including the derived ``"sqrt*"`` gates.
 
        Returns
        -------
        qc : QuantumCircuit
            The circuit with the gate sequence (and optional barriers /
            basis rotation) applied, ready to be transpiled and/or measured.
        ideal : QuantumCircuit
            The matching noiseless reference circuit, with the same gate
            sequence and basis rotation applied but no barriers.
 
        Raises
        ------
        ValueError
            If ``gate`` is not a supported gate name, if a required
            gate-parameter keyword is missing for the chosen gate, or if
            ``measurement_basis`` is not one of ``"X"``, ``"Y"``, ``"Z"``.
        """
        gate = gate.lower()

        if gate not in SINGLE_QUBIT_GATES | {"sqrtx", "sqrty", "sqrtz"}:
            raise ValueError(f"Unsupported gate '{gate}'. Choose from {SINGLE_QUBIT_GATES}")
        
        ideal = qc.copy()

        for _ in range(depth):
            if gate in {"sqrtx", "sqrty", "sqrtz"}:
                match gate:
                    case "sqrtx":
                        getattr(qc, "rx")(np.pi/2, qubit)
                        getattr(ideal, "rx")(np.pi/2, qubit)
                    case "sqrty":
                        getattr(qc, "ry")(np.pi/2, qubit)
                        getattr(ideal, "ry")(np.pi/2, qubit)
                    case "sqrtz":
                        getattr(qc, "rz")(np.pi/2, qubit)
                        getattr(ideal, "rz")(np.pi/2, qubit)
            else:
                method = getattr(qc, gate)
                id_method = getattr(ideal, gate)
                if gate in {"rx", "ry", "p"}: 
                    theta = gate_kwargs.get("theta")
                    if theta is None: 
                        raise ValueError(f"Cannot perform operation {gate}, missing keyword 'theta'.")
                    method(theta, qubit)
                    id_method(theta, qubit)
                elif gate == "rz": 
                    phi = gate_kwargs.get("phi")
                    if phi is None:
                        raise ValueError(f"Cannot perform operation {gate}, missing keyword 'phi'.")
                    method(phi, qubit)
                    id_method(phi, qubit)
                elif gate == "u":
                    theta, phi, lam = gate_kwargs.get("theta"), gate_kwargs.get("phi"), gate_kwargs.get("lam")
                    if any([theta is None, phi is None, lam is None]):
                        raise ValueError(f"Cannot perform operation {gate}, must have keywords 'theta', 'phi', 'lam'.")
                    method(theta, phi, lam, qubit)
                    id_method(theta, phi, lam, qubit)
                else: 
                    method(qubit)
                    id_method(qubit)
    
            if barriers: 
                qc.barrier()

        if measurement_basis is not None: 
            match measurement_basis:
                case "X": 
                    qc.h(qubit)
                    ideal.h(qubit)
                case "Y": 
                    qc.sdg(qubit)
                    ideal.sdg(qubit)
                    qc.h(qubit)
                    ideal.h(qubit)
                case "Z": 
                    pass
                case _: raise ValueError(
                    f"Invalid measurement basis: {measurement_basis}, choose from [X, Y, Z]")

        return qc, ideal
    
    def _run_counts(
            self, 
            qc: QuantumCircuit, 
            shots: int
            ) -> dict:
        """Run a circuit on :attr:`backend` and return its measurement counts.
 
        Runs ``qc`` for ``shots`` shots and retrieves the resulting counts.
        If the backend's result object supports ``.to_dict()`` and
        ``ResultFull`` (from ``noise_sim``) can parse it via
        ``ResultFull.from_dict``, the counts are taken from that richer
        result wrapper; otherwise (or if that parsing fails) the counts are
        taken directly from the backend result's ``.get_counts()``.
 
        Parameters
        ----------
        qc : QuantumCircuit
            Circuit to run. Must already have measurements appended (see
            :meth:`run_circuits`, which calls ``qc.measure_all()`` before
            invoking this method).
        shots : int
            Number of shots to run.
 
        Returns
        -------
        dict
            Mapping from measured bitstring to observed count.
 
        Raises
        ------
        Exception
            Any exception raised by ``self.backend.run(...)`` or
            ``.result()`` is re-raised unchanged (e.g. hardware/network
            failures, invalid circuits for the target backend).
        """
        try:
            res = self.backend.run([qc], shots=shots).result()
        except Exception as e:
            raise e
        
        if hasattr(res, "to_dict") and hasattr(ResultFull, "from_dict"):
            try:
                result_full = ResultFull.from_dict(res.to_dict())
                return result_full.get_counts()
            except Exception:
                return res.get_counts()
            
        return res.get_counts()
    
    def _create_sequencer_line(
            self, 
            counts: dict, 
            ideal_counts: dict,
            gate: str,
            depth: int, 
            shots: int,
            measurement_basis: str | None,
            fidelity: float | None = None, 
            save: bool = False,
            file_path: Path | None = None,
            **gate_kwargs,
            ): 
        """Assemble one result record and optionally append it to a JSONL file.
 
        Parameters
        ----------
        counts : dict
            Measured counts from :meth:`_run_counts`.
        ideal_counts : dict
            Ideal (noiseless) outcome probabilities from
            :meth:`_get_state_vector`.
        gate : str
            Name of the gate used for this circuit.
        depth : int
            Number of gate repetitions used for this circuit. Stored under
            the ``"amount_of_gates"`` key.
        shots : int
            Number of shots used for this circuit.
        measurement_basis : str | None
            Measurement basis used for this circuit (``"X"``, ``"Y"``,
            ``"Z"``, or ``None``).
        fidelity : float | None, optional
            Estimated gate fidelity from :meth:`_get_state_vector`, or
            ``None`` if unavailable for the current backend. By default
            None.
        save : bool, optional
            If True, append the assembled record as one JSON line to
            ``file_path``. By default False.
        file_path : Path | None, optional
            Path to the ``.jsonl`` file to append to. Required (non-None) if
            ``save=True``. By default None.
        **gate_kwargs
            Gate-parameter keywords (``theta``, ``phi``, ``lam``) used to
            populate the ``"angles"`` key; any keys not among these three
            are accepted but ignored for that purpose. Only the specific
            ``theta``/``phi``/``lam`` keys are included when present.
 
        Returns
        -------
        dict
            The assembled result record, with keys:
 
            - ``"counts"``: measured counts (``dict``)
            - ``"ideal_counts"``: ideal outcome probabilities (``dict``)
            - ``"gate"``: gate name (``str``)
            - ``"angles"``: ``{"theta", "phi", "lam"}`` sub-dict (``None``
              if no gate-parameter keywords were supplied at all)
            - ``"amount_of_gates"``: circuit depth (``int``)
            - ``"shots"``: shots used (``int``)
            - ``"measurement_basis"``: measurement basis (``str | None``)
            - ``"gate_fid"``: estimated fidelity (``float | None``)
            - ``"backend"``: class name of :attr:`backend` (``str``)
            - ``"timestamp"``: human-readable local timestamp (``str``)
            - ``"unix_time"``: Unix timestamp (``float``)
        """
        data = {
            "counts": counts,
            "ideal_counts": ideal_counts,
            "gate": gate,
            "angles": None if not gate_kwargs else {
                "theta": gate_kwargs.get("theta"),
                "phi": gate_kwargs.get("phi"),
                "lam": gate_kwargs.get("lam"),
                },
            "amount_of_gates": depth,
            "shots": shots,
            "measurement_basis": measurement_basis,
            "gate_fid": fidelity,
            "backend": type(self.backend).__name__,
            "timestamp": time.strftime('%d-%m-%Y_%H-%M-%S'),
            "unix_time": time.time(),
        }

        if save: 
            with open(file_path, 'a') as f:
                json.dump(data, f)
                f.write("\n")
            logger.debug(f"data saved under {file_path}")
    
        return data

    def _save(
            self, 
            savepath: str | None = None, 
            savename: str | None = None,
            append_data: bool = False,
            ) -> Path:
        """Create (or locate) the JSONL file that results will be written to.
 
        Parameters
        ----------
        savepath : str | None, optional
            Directory to save the file in. Defaults to
            ``DEFAULT_DATA_ROOT`` if omitted. Created (including any missing
            parent directories) if it doesn't already exist.
        savename : str | None, optional
            Base filename (without extension). Defaults to the current
            timestamp (``DD-MM-YYYY_HH-MM-SS``) if omitted.
        append_data : bool, optional
            If True, return the path ``{savename}.jsonl`` directly (creating
            an empty file only if it doesn't already exist), so subsequent
            writes append to any existing content. If False, always create
            a fresh, empty file: if ``{savename}.jsonl`` already exists, a
            numeric suffix (``{savename}(1).jsonl``, ``{savename}(2).jsonl``,
            ...) is used instead to avoid overwriting prior data. By default
            False.
 
        Returns
        -------
        Path
            Path to the (now-existing, empty-or-preserved) ``.jsonl`` file
            that subsequent writes should target.
        """
        base_dir = Path(f"{savepath}") if savepath else DEFAULT_DATA_ROOT
        base_dir.mkdir(parents=True, exist_ok=True)

        base_name = savename or f"{time.strftime('%d-%m-%Y_%H-%M-%S')}"
        file_path = base_dir / f"{base_name}.jsonl"

        if append_data:
            file_path.touch(exist_ok=True)
            return file_path

        i = 1
        candidate = file_path
        while candidate.exists():
            candidate = base_dir / f"{base_name}({i}).jsonl"
            i += 1

        candidate.open("x").close()
        return candidate

def _filter_kwargs(func, kwargs: dict) -> dict:
    """Return only the kwargs that `func` actually accepts.
 
    Parameters
    ----------
    func : Callable
        The function whose signature is used to filter ``kwargs``.
    kwargs : dict
        Candidate keyword arguments to filter.
 
    Returns
    -------
    dict
        If ``func`` itself declares a ``**kwargs``-style variadic keyword
        parameter, ``kwargs`` is returned unchanged (nothing needs
        filtering, since ``func`` would accept anything). Otherwise, a new
        dict containing only the entries of ``kwargs`` whose key matches a
        named parameter in ``func``'s signature.
    """
    sig = inspect.signature(func)
    
    # If func accepts **kwargs itself, nothing needs filtering
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    
    valid_params = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid_params}