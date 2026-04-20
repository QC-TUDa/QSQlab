""" benchmarking sequence file for both simulation and hardware Qiskit backends """

import logging
logger = logging.getLogger(__name__)

import time, json

from pathlib import Path
from typing import Literal
from qiskit import QuantumCircuit
from qiskit.providers.backend import Backend, BackendV2
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import average_gate_fidelity, Operator

from ..config.data_config import DEFAULT_DATA_ROOT
from ..config.gate_config import _SINGLE_QUBIT_GATES
from ..noise.noise_sim_base import *


class BenchmarkSequencer():
    """ Sequencer class for single qubit circuits
    """
    def __init__(
            self,
            backend: Backend | BackendV2,
            qc: QuantumCircuit | None = None,
            #
            **kwargs,
        ):
        """ Constructor of BenchmarkSequencer class for single qubit circuits.

        Parameters
        ----------
        backend : Backend | BackendV2
            qiskit backend for simulator or hardware
        qc : QuantumCircuit | None, optional
            initial quantum circuit to apply sequence of gates, by default None
        """
        self.backend = backend
        self.options = kwargs
        self._qc = None
        if qc is not None: self.qc = qc

        self.data: list = []

    @property
    def qc(self) -> QuantumCircuit | None:
        return self.qc
    
    @qc.setter
    def qc(self, qc: QuantumCircuit):
        if not isinstance(qc, QuantumCircuit): 
            raise TypeError(f"qc must be a QuantumCircuit, not {type(qc).__name__}")
        self._qc = qc

    def run_circuits(self,
                    depths: list[int],
                    shots: int, 
                    gate: str = "x",
                    qubit: int = 0,
                    qc: QuantumCircuit | None = None,
                    measurement_basis: Literal["X", "Y", "Z"] | None = None,
                    barriers: bool = False, 
                    sort: bool = True,
                    #
                    save: bool = False, 
                    savepath: str | None = None, 
                    savename: str | None = None,
                    append_data: bool = False,
                    #
                    **gate_kwargs
                    ) -> dict[str, Path | dict]: #TODO improve docstring
        """Run circuits and collect counts and optionally gets average fidelity results by using
          Qiskit's Built-in Entanglement fidelity formula. """
        counts = {}
        sequencer_data = []
        if sort: depths.sort()

        if qc is not None: self.qc = qc
        elif self._qc is None and qc is None: 
            raise ValueError("No QuantumCircuit provided to run sequence")

        if save: file_path : Path = self._save(savepath, savename, append_data)

        for depth in depths:
            qc, iqc = self._create_single_qubit_circuit(
                qc=self._qc.copy(), 
                gate=gate, 
                qubit=qubit, 
                depth=depth, 
                measurement_basis=measurement_basis,
                barriers=barriers,
                **gate_kwargs
                )
            
            fidelity = average_gate_fidelity(Operator(qc), Operator(iqc))
            ideal_counts = self._get_state_vector(iqc)
            qc.measure_all()
            counts = self._run_counts(qc, shots)
            #counts[depth] = {"counts": counts} #TODO why did I have this line???
            
            sequencer_line = self._create_sequencer_line(
                    counts=counts, 
                    ideal_counts=ideal_counts,
                    gate=gate, 
                    measurement_basis=measurement_basis, 
                    depth=depth, 
                    shots=shots, 
                    fidelity=fidelity, 
                    **gate_kwargs,
                    )
            sequencer_data.append(sequencer_line)

            if save: 
                with open(file_path, 'a') as f:
                    json.dump(sequencer_line, f)
                    f.write("\n")
                logger.debug(f"data saved under {file_path}")
        
        self.data = self.data + sequencer_data

        return sequencer_data

    def _get_state_vector(self, qc, tol: float = 1e-10):
        state = Statevector.from_instruction(
            qc.copy().remove_final_measurements(inplace=False)
        )
        
        probs = {
            str(k): 0.0 if abs(float(v)) < tol else float(v)
            for k, v in state.probabilities_dict().items()
        }
        
        # ensure both outcomes exist (single qubit case)
        return {'0': round(probs.get('0', 0.0), 10), '1': round(probs.get('1', 0.0), 10)}

    def _create_single_qubit_circuit(self, 
                                     qc: QuantumCircuit,
                                     gate: str,
                                     qubit: int,
                                     depth: int, 
                                     measurement_basis: Literal["X", "Y", "Z"] | None,
                                     barriers: bool,
                                     **gate_kwargs,
                                     ) -> QuantumCircuit:
        """Method that creates a Circuit of the specified depth for each respective backend"""
        gate = gate.lower()

        if not hasattr(qc, gate):
            raise ValueError(f"Unsupported gate '{gate}'. Choose from {_SINGLE_QUBIT_GATES}")
        
        if hasattr(qc, "to_quantum_circuit"): 
            ideal = qc.to_quantum_circuit()
        else:
            ideal = qc.copy()

        for _ in range(depth):
            method = getattr(qc, gate)
            id_method = getattr(ideal, gate)

            if gate in {"rx", "ry", "p"}: 
                method(gate_kwargs["theta"], qubit)
                id_method(gate_kwargs["theta"], qubit)
            elif gate == "rz": 
                method(gate_kwargs["phi"], qubit)
                id_method(gate_kwargs["phi"], qubit)
            elif gate == "u": 
                method(gate_kwargs["theta"], gate_kwargs["phi"], gate_kwargs["lam"])
                method(gate_kwargs["theta"], gate_kwargs["phi"], gate_kwargs["lam"])
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
    
    def _run_counts(self, qc: QuantumCircuit, shots: int) -> dict:
        """Wrapper function which checks which decides which .get_counts method to use according 
        to the backend"""

        res = self.backend.run([qc], shots=shots).result()
        if hasattr(res, "to_dict") and hasattr(ResultFull, "from_dict"):
            try:
                result_full = ResultFull.from_dict(res.to_dict())
                return result_full.get_counts()
            except Exception:
                return res.get_counts()
            
        return res.get_counts()
    
    def _create_sequencer_line(self, 
                     counts: dict, 
                     ideal_counts: dict,
                     gate: str,
                     depth: int, 
                     shots: int,
                     measurement_basis: str | None,
                     fidelity: float | None = None, 
                     **gate_kwargs,
                     ): 
        """ funciton to save data in a JSONL file """

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
            "channel_fid": fidelity,
            "backend": type(self.backend).__name__,
            "timestamp": time.strftime('%d-%m-%Y_%H-%M-%S'),
            "unix_time": time.time(),
        }
    
        return data

    def _save(self, 
              savepath: str | None = None, 
              savename: str | None = None,
              append_data: bool = False,
              ) -> Path:
        """ Creates a jsonl data file """
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
    


# TODO write down tests after changes