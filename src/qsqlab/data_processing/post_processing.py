"""post_processing file containing PostProcessing template class for 
protocol-specific single-qubit gate analysis.
"""

import logging
logger = logging.getLogger(__name__)

import time
import json
import math
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from ..exceptions import MixedDataError, UnevenDepthError, GateNotSupportedError
from ..config.plot_config import DEFAULT_PLOT_STYLE 
from ..config.data_config import DEFAULT_DATA_ROOT
from ..config.backend_config import SIMULATOR_BACKENDS


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Base result container for quantum benchmark results.

    This class stores the minimal metadata common to all benchmark results,
    such as the execution backend, the benchmarked gate, whether the result
    was obtained from simulation, and the circuit depths used in an experiment.

    Parameters
    ----------
    backend : str
        Identifier of the backend on which the benchmark was executed
        (e.g. simulator name or hardware backend).
    gate : str
        Name or identifier of the quantum gate or operation in the circuit sequence.
    simulation : bool | None
        Indicates whether the benchmark was run in simulation.
        ``True`` for simulation, ``False`` for hardware execution, and
        ``None`` if this information is unavailable or not applicable.
    depths : list[int]
        Circuit depths used in the benchmark, typically corresponding to
        the number of gate repetitions or layers.

    Notes
    -----
    This class is intended to be subclassed by more specific benchmark
    result types that add benchmark-specific metrics and analysis results.
    """
    backend: str
    gate: str
    simulation: bool | None
    depths: list[int]

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  backend = {self.backend}, \n"
            f"  gate = {self.gate}, \n"
            f"  simulation = {self.simulation}, \n"
            f"  depths = {self.depths}, \n"
            f")"
        )

class PostProcessing():
    """ Parent PostProcessing class
    This class evaluates data from BenchmarkSequencer, determines survival probabilities from count 
    data and plots if specified. Additionally it stores parameters and results in dataclass containers.
    This class is intended to be subclassed by more specific Benchmark PostProcessing types that 
    add benchmark specific calculations.

    Attributes
    ----------
    results : BenchmarkResult
        Result container with benchmark parameters obtained from sequencer data.
    _backend: str | None
        string representation of backend used to execute sequences. None if backend isn't specified.
    _gate: str | None
        string representation of gate implemented in sequences. None if gate isn't specified.
    _depths: list[int]
        list of sequence depths of specified gate in all circuits.
    _survival_probs: list[float]
        list of survival probabilities in circuit results. From shortest sequences to longest sequences.
    _errors: list[float]
        list of uncertainty in survival probabilities based on binomial errors.

    Notes
    -----
    This class is intended as a template for protocol-specific analysis procedures.
    As-is this class will estimate survival probabilities for sequences of increasing amounts of 
    rx/ry/x/y gates and optionally plot them.
    """
    def __init__(self, 
                 sequencer_data: Path | list[dict] | None = None, 
                 *,
                 show_plot: bool = True, 
                 save: bool = False, 
                 simulation: bool | None = None,
                 savefolder: str = '', 
                 savename: str = '', 
                 plot_style: dict | None = None,
                 ):
        """Constructor of PostProcessing class. During initialization, this class reads 
        BenchmarkSequencer data, transforms counts into survival probabilities and optionally plots 
        the data if specified. 

        Parameters
        ----------
        sequencer_data : Path | list[dict] | None, optional
            sequencer data to analyze. can be the path to a jsonl file or a list of dicts. 
            If None is specified, it reads the latest data from the default saving folder, 
            by default None
        show_plot : bool, optional
            boolean to show matplotlib plot, by default True
        save : bool, optional
            boolean to save plot in png and pdf formats, by default False
        simulation : bool | None, optional
            boolean to manually set data as simulation or hardware, by default None
        savefolder : str, optional
            folder path to save figure, by default ''
        savename : str, optional
            file name to save figures, by default ''
        plot_style : dict | None, optional
            optional style config dict for matplotlib, by default None
        """
        self._data: list[dict] | None = self._read(sequencer_data)
        self._simulation: str | None = self._determine_simulation(self._data, simulation)

        self.points: dict[int, dict] | None = self._process_points(self._data)

        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None

        if show_plot or save:
            self._fig, self._ax = self._plot(
                show=show_plot,
                save=save,
                savefolder=savefolder,
                savename=savename,
                plot_style=plot_style,
            )

    @property
    def results(self) -> BenchmarkResult:
        """Snapshot of this run's backend, gate, simulation flag, and measured
        depths as an immutable :class:`BenchmarkResult`."""
        return BenchmarkResult( 
            backend=self._backend, 
            gate=self._gate, 
            simulation=self._simulation, 
            depths=self._depths,
            )
        
    @property
    def _backend(self) -> str | None:
        """
        Returns
        -------
        str | None
            Backend name from sequencer records, or ``None`` if absent.
        """
        return self._data[0].get("backend")
    
    @property
    def _gate(self) -> str | None:
        """
        Returns
        -------
        str | None
            String representation of gate from sequencer records, or ``None`` if absent.
        """
        return self._data[0].get("gate")

    @property
    def _depths(self) -> list[int]:
        """
        Returns
        -------
        list[int]
            Sorted circuit depths (``amount_of_gates`` values) present in
            ``self.points``, ascending.
        """
        return sorted([k for k in self.points if k not in {"backend", "gate"}])

    @property
    def _survival_probs(self) -> list[float]:
        """
        Returns
        -------
        list[float]
            Survival probability at each depth in ``self._depths``, in the same
            order (shortest sequence first).
        """
        return [self.points[d]['survival_prob'] for d in self._depths]

    @property
    def _errors(self) -> list[float]:
        """
        Returns
        -------
        list[float]
            Binomial error at each depth in ``self._depths``, in the same
            order (shortest sequence first).
        """
        return [self.points[d]['binomial_error'] for d in self._depths]
    
    def _plot(
            self, 
            *,
            show: bool = True,
            save: bool = False,
            savefolder: str = '', 
            savename: str = '', 
            plot_style: dict | None = None
            ) -> tuple[plt.Figure, plt.Axes]:
        """Plot survival probability vs. circuit depth with binomial error bars.

        ...(existing Parameters section is accurate, keep as-is)...

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The created figure and axes, for further customization.

        Notes
        -----
        The created figure is not automatically closed when ``show=True``; when
        ``save=True`` it is closed inside :meth:`_save_plot`. Calling with both
        ``show=True`` and ``save=True`` may display an already-closed figure —
        this is a known open issue, not yet finalized.
        """
        style = DEFAULT_PLOT_STYLE | (plot_style or {})

        with plt.rc_context(style):
            fig, ax = plt.subplots()

            label = "Simulated data" if self._simulation else "Measured data"
            color = "green" if self._simulation else None

            ax.errorbar(
                self._depths,
                self._survival_probs,
                yerr=self._errors,
                fmt="D",
                markersize=4,
                color=color,
                label=label,
            )

            ax.set(
                xlabel="Circuit Length",
                ylabel="Survival Probability",
                ylim=(0, 1),
            )

            ax.legend(loc="upper right")
            ax.grid(True)

            if save: self._save_plot(fig, savefolder, savename)
            if show: plt.show()

            return fig, ax

    def _read(
            self, 
            sequencer_data: Path | list[dict] | None = None
            ) -> list[dict]:
        """Load raw sequencer records from a JSONL file, an in-memory list, or the
        most recently modified file in the default data directory.

        Parameters
        ----------
        sequencer_data : Path | list[dict] | None, optional
            - ``Path`` or ``str``: path to a JSONL file to read.
            - ``list[dict]``: pre-loaded sequencer records, validated but not re-read.
            - ``None`` (default): read the most recently modified ``*.jsonl`` file
            in ``DEFAULT_DATA_ROOT``.

        Returns
        -------
        list[dict]
            Parsed sequencer records. Reading a file stops at the first blank line
            or malformed JSON line (treated as a measurement-block separator), so
            the returned list may be shorter than the file's total line count.

        Raises
        ------
        ValueError
            If ``sequencer_data`` is an empty list, if a provided list of dicts is
            missing required keys, or if ``sequencer_data`` is not one of the
            supported types.
        FileNotFoundError
            If ``sequencer_data`` is ``None`` and ``DEFAULT_DATA_ROOT`` doesn't
            exist or contains no ``.jsonl`` files, or if a given path does not
            point to an existing file.
        """
        match sequencer_data:
            case []:
                raise ValueError("Cannot process data from empty sequencer data.")
            case [first, *_]:
                required_keys = {"backend", "gate", "counts", "amount_of_gates", "angles", "shots"}
                if not all(key in first for key in required_keys):
                    raise ValueError("Data does not comply with structure of sequencer data dictionaries.")
                return sequencer_data
            case None:
                data_root: Path = DEFAULT_DATA_ROOT
                if not data_root.exists():
                    raise FileNotFoundError(f"No data directory found at {data_root}")
                jsonl_files = list(data_root.glob("*.jsonl"))
                if not jsonl_files:
                    raise FileNotFoundError(f"No .jsonl files found in {data_root}, no data to read.")
                file_path = max(jsonl_files, key=lambda p: p.stat().st_mtime)
            case Path() | str():
                file_path = Path(sequencer_data)
                if not file_path.is_file():
                    raise FileNotFoundError(f"No such file: '{file_path}'")
            case _:
                raise ValueError(
                    f"Invalid parameter type: {type(sequencer_data).__name__}. expected "
                    "Path | list[dict] | None"
                )

        results = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    break
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.exception("finishing loop because of malformed line: %s", e)
                    break
        return results
    
    def _determine_simulation(
            self,
            data: list[dict],
            simulation: bool | None,
            ) -> bool:
        """Determine whether the data stems from a simulated or hardware backend.

        If ``simulation`` is explicitly given, it's returned as-is. Otherwise the
        backend name in ``data`` is looked up against ``SIMULATOR_BACKENDS``.

        Parameters
        ----------
        data : list[dict]
            Sequencer records as returned by :meth:`_read`.
        simulation : bool | None
            Manual override; bypasses backend lookup if not ``None``.

        Returns
        -------
        bool
            ``True`` if the backend is a known simulator, ``False`` otherwise.

        Raises
        ------
        ValueError
            If ``simulation`` is ``None`` and ``data`` is empty.
        MixedDataError
            If ``data`` contains records from more than one backend.
        """
        if simulation is not None:
            return simulation

        if not data: raise ValueError("Cannot determine backend from empty data")

        backends = {d["backend"] for d in data}

        if len(backends) > 1:
            raise MixedDataError(
                "provided data has mixed backend measurements"
            )

        backend = next(iter(backends))
        return backend in SIMULATOR_BACKENDS
    
    def _process_points(
            self, 
            data: list[dict],
            ) -> dict[int, dict]:
        """Aggregate raw counts by circuit depth and compute survival probabilities.

        Groups records by ``amount_of_gates``, sums their ``'0'``/``'1'`` counts,
        then determines which bit counts as "survival" based on the gate type:

        - ``rx`` / ``ry`` with ``theta ≈ pi/2``: survival bit alternates every two
        depths (``0`` at depths ``0 mod 4``, ``1`` at depths ``2 mod 4``).
        - ``rx`` / ``ry`` with ``theta ≈ pi``: survival bit alternates by depth
        parity (``0`` on even depths, ``1`` on odd).
        - ``x`` / ``y``: survival bit alternates by depth parity, same as above.

        All records are assumed to share the same gate, measurement basis, and
        rotation angles; this is validated before aggregation.

        Parameters
        ----------
        data : list[dict]
            Sequencer records as returned by :meth:`_read`.

        Returns
        -------
        dict[int, dict] | None
            Mapping of circuit depth to a dict with keys ``'0'``, ``'1'``
            (summed raw counts), ``survival_prob``, ``survival_counts``,
            ``total_counts``, and ``binomial_error``. Returns ``None`` if
            ``data`` is empty.

        Raises
        ------
        MixedDataError
            If records mix more than one gate, measurement basis, or rotation
            angle (theta/phi/lambda).
        UnevenDepthError
            If a pi/2-rotation sequence has a depth that isn't a multiple of 2,
            making the survival bit ambiguous.
        GateNotSupportedError
            If the gate is not one of ``rx``, ``ry``, ``x``, ``y``, or if an
            ``rx``/``ry`` rotation angle is neither ``pi/2`` nor ``pi``.
        """
        points: dict = {}
        if data == []: return None

        gates = {d.get("gate") for d in data}
        meas_basis = {d.get("measurement_basis") for d in data}
        angles = {
            "theta": set([d["angles"]["theta"] for d in data]), 
            "phi": set([d["angles"]["phi"] for d in data]), 
            "lam": set([d["angles"]["lam"] for d in data])
            }

        if len(gates) > 1 or len(meas_basis) > 1 or any([len(angles["theta"]) > 1, len(angles["phi"]) > 1, len(angles["lam"]) > 1]):
            raise MixedDataError("provided data has sequences with mixed gate and/or measurement basis.")
        angles = {
                    "theta": next(iter(angles["theta"])), 
                    "phi": next(iter(angles["phi"])), 
                    "lam": next(iter(angles["lam"]))
                    }
        
        for i in data:
            if i["amount_of_gates"] not in points.keys():
                points[i["amount_of_gates"]] = {'0': 0, '1': 0}

            points[i["amount_of_gates"]]['0'] += i["counts"]['0']
            points[i["amount_of_gates"]]['1'] += i["counts"]['1']

        gate = next(iter(gates))
        match gate:
            case "rx" | "ry":
                if math.isclose(angles["theta"], np.pi/2, rel_tol=1e-9, abs_tol=1e-12):
                    for i in points.keys():
                        if i%4 == 0: 
                            points[i].update(
                                {"survival_prob": points[i]['0'] / (points[i]['0'] + points[i]['1']),
                                "survival_counts": points[i]['0'], 
                                "total_counts": (points[i]['0'] + points[i]['1']), 
                                "binomial_error": binomial_error(
                                    points[i]['0'], points[i]['0'] + points[i]['1'])})
                        elif i%2==0 and i%4!=0:
                            points[i].update(
                                {"survival_prob": points[i]['1'] / (points[i]['0'] + points[i]['1']), 
                                "survival_counts": points[i]['1'], 
                                "total_counts": (points[i]['0'] + points[i]['1']), 
                                "binomial_error": binomial_error(
                                    points[i]['1'], points[i]['0'] + points[i]['1'])}
                                    )
                        else:
                            raise UnevenDepthError(
                                "Sequence contains uneven number of pi/2 rotations. Cannot determine Survival probability."
                            )
                elif math.isclose(angles["theta"], np.pi, rel_tol=1e-9, abs_tol=1e-12):
                    for i in points.keys():
                        if i % 2 == 0:
                            points[i].update({
                                "survival_prob": points[i]['0'] / (points[i]['0'] + points[i]['1']),
                                "survival_counts": points[i]['0'],
                                "total_counts": points[i]['0'] + points[i]['1'],
                                "binomial_error": binomial_error(points[i]['0'], points[i]['0'] + points[i]['1']),
                            })
                        else:
                            points[i].update({
                                "survival_prob": points[i]['1'] / (points[i]['0'] + points[i]['1']),
                                "survival_counts": points[i]['1'],
                                "total_counts": points[i]['0'] + points[i]['1'],
                                "binomial_error": binomial_error(points[i]['1'], points[i]['0'] + points[i]['1']),
                            })
                else: 
                    raise GateNotSupportedError(
                        f"Angle for rotation not implemented, choose from pi/2, pi."
                        )
                
            case "x" | "y":
                for i in points.keys():
                    if i % 2 == 0:
                        points[i].update({
                            "survival_prob": points[i]['0'] / (points[i]['0'] + points[i]['1']),
                            "survival_counts": points[i]['0'],
                            "total_counts": points[i]['0'] + points[i]['1'],
                            "binomial_error": binomial_error(points[i]['0'], points[i]['0'] + points[i]['1']),
                        })
                    else:
                        points[i].update({
                            "survival_prob": points[i]['1'] / (points[i]['0'] + points[i]['1']),
                            "survival_counts": points[i]['1'],
                            "total_counts": points[i]['0'] + points[i]['1'],
                            "binomial_error": binomial_error(points[i]['1'], points[i]['0'] + points[i]['1']),
                        })

            case _:
                raise GateNotSupportedError(
                    f"Gate {gate} is not supported in analysis. Use from 'Rx, Ry, x, y'"
                )

        return points
    
    def _save_plot(
            self,
            fig,
            savepath: str | None = None,
            savename: str | None = None,
        ) -> tuple[Path, Path]:
        """Save a matplotlib figure as both PNG and PDF, then close it.

        Parameters
        ----------
        fig : plt.Figure
            Figure to save.
        savepath : str | None, optional
            Destination directory; defaults to ``DEFAULT_DATA_ROOT / "plots"``.
        savename : str | None, optional
            File stem (no extension); defaults to the current timestamp.

        Returns
        -------
        tuple[Path, Path]
            Paths to the saved PNG and PDF files, respectively.
        """
        timestamp = time.strftime('%d-%m-%Y_%H-%M-%S')

        # Base directory
        base_dir = Path(savepath) if savepath else DEFAULT_DATA_ROOT / "plots"
        base_dir.mkdir(parents=True, exist_ok=True)

        # File name
        base_name = savename or timestamp

        png_path = base_dir / f"{base_name}.png"
        pdf_path = base_dir / f"{base_name}.pdf"

        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved plot to:\n  %s\n  %s", png_path, pdf_path)

        return png_path, pdf_path
    
    def __str__(self):
        return self.results.__str__()

def binomial_error(k, N):
    """Standard error of a binomial proportion estimate.

    Equivalent to ``sqrt(p * (1 - p) / N)`` where ``p = k / N``, rewritten in
    terms of raw counts.
    """
    return np.sqrt((k/N**2) - (k**2/N**3))
