"""post_processing file containing PostProcessing template class for 
protocol-specific analysis.
"""

import logging
logger = logging.getLogger(__name__)

import time
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from IPython.display import display

from ..config.plot_config import DEFAULT_PLOT_STYLE 
from ..config.data_config import DEFAULT_DATA_ROOT
from ..config.backend_config import SIMULATOR_BACKENDS
from ..exceptions import EmptyDataFile, MixedBackendError, UndefinedIdealState
from .results import BenchmarkResult

class PostProcessing():
    """Parent PostProcessing class
    This class evaluates data from BenchmarkSequencer, determines success probabilities from count 
    data and plots if specified. Additionally it stores parameters and results in dataclass containers.
    This class is intended to be subclassed by more specific Benchmark PostProcessing types that 
    add benchmark specific calculations.

    Attributes
    ----------
    results : BenchmarkResult
        Main information container with relevant benchmark run parameters and results.
    data_raw : pandas.DataFrame
        circuit-indexed DataFrame with relevant data and parameters.
    points : pandas.DataFrame
        depth-indexed DataFrame with statistics from agglomerated sequence data.
    
    backend : str
        string representation of backend used to execute sequences.
    gates : set[str]
        string representation of gate implemented in sequences.
    simulation : bool 
        flag representing if data stems from simulation.
    depths : list[int]
        list of sequence depths of specified gate in all circuits.

    REQUIRED_KEYS : frozenset[str]
        Set of keys needed by the sequence records in order to carry out the analysis in this class.
        altering this set in a child class will redefine the keys to validate from the records.

    Notes
    -----
    This class is intended as a template for protocol-specific analysis procedures.
    As-is this class will estimate success probabilities and binomial errors for sequences of 
    increasing amounts of gates and optionally plot them.
    """
    REQUIRED_KEYS = frozenset(
        {"ideal_probs", "backend", "gate", "counts",
            "amount_of_gates", "angles", "shots"}
    )

    def __init__(self, 
                 sequencer_data: Path | str | list[dict] | None = None, 
                 *,
                 show_results: bool = True,
                 save_plot: bool = False, 
                 savefolder: str = '', 
                 savename: str = '', 
                 simulation: bool | None = None,
                 plot_style: dict | None = None,
                 ):
        """Constructor of PostProcessing class. During initialization, this class reads 
        BenchmarkSequencer data, transforms counts into success probabilities and optionally plots 
        the data if specified. 

        Parameters
        ----------
        sequencer_data : Path | str | list[dict] | None, optional
            sequencer data to analyze. can be the path to a jsonl file or a list of dicts. 
            If None is specified, it reads the latest data from the default saving folder, 
            by default None
        show_results: bool, optional
            boolean to show results with the markdown renderer, by default True
        save_plot : bool, optional
            boolean to save plot in png and pdf formats, by default False
        savefolder : str, optional
            folder path to save figure, by default ''
        savename : str, optional
            file name to save figures, by default ''
        simulation: bool, optional
            flag pointing out measured data to stem from simulator
        plot_style : dict | None, optional
            optional style config dict for matplotlib, by default None

        Raises
        ------
        ValueError
            If ``sequencer_data`` is empty.
        TypeError
            If ``sequencer_data`` has an unsupported type, or if a record is not
            a dict.
        KeyError
            If any record (from a list or a file) is missing one of the required
            keys specified in the global variable REQUIRED_KEYS.
        FileNotFoundError
            If ``sequencer_data`` is ``None`` and ``DEFAULT_DATA_ROOT`` doesn't
            exist or contains no ``.jsonl`` files, or if a given path does not
            point to an existing file.
        EmptyDataFile
            If the file read (given or latest default) contains no records.
        MixedBackendError
            If the records come from more than one backend.
        """
        self._data_raw: pd.DataFrame = self._read(sequencer_data)
        self._sim: bool | None = simulation 
        self._points: pd.DataFrame = self._process_points(self._data_raw)

        if type(self) is PostProcessing:
            self._finalize(
                show_results=show_results,
                save_plot=save_plot,
                savefolder=savefolder,
                savename=savename,
                plot_style=plot_style,
            )

    @property
    def results(self) -> BenchmarkResult:
        """Snapshot of this run's backend, gates, simulation flag, and measured
        depths as a mutable :class:`BenchmarkResult`.

        Returns
        -------
        BenchmarkResult
            Result container with relevant settings from sequence records and result data from 
            analysis.
        """
        return self._results

    @property
    def data_raw(self) -> pd.DataFrame:
        """Raw data as read out from jsonl records.

        Returns
        -------
        pandas.DataFrame
            Sequence result data with included run parameters
        """
        return self._data_raw

    @property
    def points(self) -> pd.DataFrame:
        """Aggregated statistics per sequence depth.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``amount_of_gates`` with columns ``successes``,
            ``failures``, ``total``, ``success_prob``, ``fail_prob``,
            ``success_binomial_error`` and ``fail_binomial_error``.
            This is the internal frame, not a copy.
        """
        return self._points
        
    @property
    def backend(self) -> str:
        """Backend type from sequencer records.

        Returns
        -------
        str
            Backend name from sequencer records.
        """
        return str(self.data_raw["backend"].iloc[0])
    
    @property
    def gates(self) -> set[str]:
        """Unique gate names found in the sequencer records.

        Returns
        -------
        set
            unique gate names from sequencer records.
        """
        
        return set(self.data_raw["gate"])

    @property
    def simulation(self) -> bool | None:
        """Whether the data stems from a simulator.

        Returns the ``simulation`` override given at construction if any,
        otherwise infers it from the backend name via ``SIMULATOR_BACKENDS``.

        Raises
        ------
        ValueError
            If no override was given and ``data_raw`` is empty.
        MixedBackendError
            If no override was given and ``data_raw`` has multiple backends.

        Returns
        -------
        bool | None
            Flag denoting the data as a simulation. None if backend type could
            not be verified.
        """
        return self._determine_simulation(self.data_raw, self._sim)

    @property
    def depths(self) -> list[int]:
        """sequence depths implemented in benchmark.

        Returns
        -------
        list[int]
            Sorted circuit depths (``amount_of_gates`` values) present in
            ``self.data_raw``, ascending.
        """
        return sorted(set(self.data_raw["amount_of_gates"]))

    def _finalize(
            self, 
            *, 
            show_results,
            save_plot, 
            savefolder, 
            savename, 
            plot_style
            ):
        """Runs plotting + results creation + display. Call this at the end of a
        subclass's __init__ once any additional processing is done.
        """
        self._fig, self._ax = self._plot(
                save=save_plot,
                savefolder=savefolder,
                savename=savename,
                plot_style=plot_style,
            )
        self._results=self._create_results()
        if show_results:
            display(self._results)

    def _read(
            self,
            sequencer_data: Path | str | list[dict] | None = None
        ) -> pd.DataFrame:
        """Load raw sequencer records from a JSONL file, an in-memory list, or the
        most recently modified file in the default data directory.

        Parameters
        ----------
        sequencer_data : Path | str | list[dict] | None, optional
            - ``Path`` or ``str``: path to a JSONL file to read.
            - ``list[dict]``: pre-loaded sequencer records, validated but not re-read.
            - ``None`` (default): read the most recently modified ``*.jsonl`` file
              in ``DEFAULT_DATA_ROOT``.

        Returns
        -------
        pandas.DataFrame
            One row per record. When reading a file, parsing stops at the first
            blank line or malformed JSON line (treated as a measurement-block
            separator), so the frame may have fewer rows than the file has lines.

        Raises
        ------
        ValueError
            If ``sequencer_data`` is an empty list.
        TypeError
            If ``sequencer_data`` has an unsupported type, or if a record is not
            a dict.
        KeyError
            If any record (from a list or a file) is missing one of the required
            keys: "ideal_probs", "backend", "gate", "counts", "amount_of_gates",
            "angles", "shots".
        FileNotFoundError
            If ``sequencer_data`` is ``None`` and ``DEFAULT_DATA_ROOT`` doesn't
            exist or contains no ``.jsonl`` files, or if a given path does not
            point to an existing file.
        EmptyDataFile
            If the file read (given or latest default) contains no records.
        MixedBackendError
            If the records come from more than one backend.
        """
        match sequencer_data:
            case []:
                raise ValueError(
                    "Cannot process data from empty sequencer data.")
            case [_, *_]:
                results = sequencer_data
            case None:
                data_root: Path = DEFAULT_DATA_ROOT
                if not data_root.exists():
                    raise FileNotFoundError(
                        f"No data directory found at {data_root}")
                jsonl_files = list(data_root.glob("*.jsonl"))
                if not jsonl_files:
                    raise FileNotFoundError(
                        f"No .jsonl files found in {data_root}, no data to read.")
                results = self._read_jsonl(
                    max(jsonl_files, key=lambda p: p.stat().st_mtime))
            case Path() | str():
                file_path = Path(sequencer_data)
                if not file_path.is_file():
                    raise FileNotFoundError(f"No such file: '{file_path}'")
                results = self._read_jsonl(file_path)
            case _:
                raise TypeError(
                    f"Invalid parameter type: {type(sequencer_data).__name__}. expected "
                    "Path | str | list[dict] | None"
                )

        self._validate_records(results)

        if len({d["backend"] for d in results}) > 1:
            raise MixedBackendError(
                "Provided sequencer data has mixed backends. "
                "Analysis can only be carried out for one backend"
            )

        return pd.DataFrame(results)

    def _process_points(
            self, 
            data: pd.DataFrame,
            ) -> pd.DataFrame:
        """Aggregate raw counts by circuit depth and compute success probabilities.

        For each record, the ideal final state (given in ``ideal_probs``) determines
        which measured bit (``'0'`` or ``'1'``) counts as a "success" and which counts
        as a "failure". Records are then grouped by ``amount_of_gates``, with
        successes, failures, and totals summed within each group. Success/failure
        probabilities and their binomial errors are computed per depth.

        Parameters
        ----------
        data : pd.DataFrame
            Sequencer records as returned by :meth:`_read`. Must contain the columns
            ``'ideal_probs'``, ``'counts'``, and ``'amount_of_gates'``.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``amount_of_gates`` with columns ``successes``,
            ``failures``, ``total`` (summed raw counts), ``success_prob``,
            ``fail_prob``, ``success_binomial_error``, ``fail_binomial_error``, 
            ``0``, ``1``, ``p0``, ``p1``, ``p0_binomial_error``, and ``p1_binomial_error``.

        Raises
        ------
        UndefinedIdealState
            If any record's ``ideal_probs`` is not one of ``{'1': 1.0, '0': 0.0}``
            or ``{'1': 0.0, '0': 1.0}``, i.e. the ideal final state is not a
            well-defined computational basis state.
        """
        data = data.copy()
        if any(probs not in ({'1': 1.0, '0': 0.0}, {'1': 0.0, '0': 1.0}) for probs in data["ideal_probs"]):
            raise UndefinedIdealState(
                "Sequence does not provide a well defined ideal final state. "
                "'ideal_probs' should have '1': 1.0 or '0': 1.0")

        def success_failure(row):
            ideal_label = '1' if row["ideal_probs"].get('1') == 1.0 else '0'
            fail_label = '0' if ideal_label == '1' else '1'
            return pd.Series({
                "successes": row["counts"].get(ideal_label, 0),
                "failures": row["counts"].get(fail_label, 0),
                "0": row["counts"].get("0", 0),
                "1": row["counts"].get("1", 0)
            })

        data[["successes", "failures", "0", "1"]] = data.apply(success_failure, axis=1)
        data["total"] = data["successes"] + data["failures"]

        points = (
            data.groupby("amount_of_gates")[["successes", "failures", "total", "0", "1"]]
            .sum()
            .reset_index()
        )

        # --- compute probabilities and uncertainty ---
        points["success_prob"] = points["successes"] / points["total"]
        points["fail_prob"] = points["failures"] / points["total"]
        points["p0"] = points["0"] / points["total"]
        points["p1"] = points["1"] / points["total"]
        points["success_binomial_error"] = points.apply(
                lambda r: binomial_error(r["successes"], r["total"]), axis=1
            )
        points["fail_binomial_error"] = points.apply(
                lambda r: binomial_error(r["failures"], r["total"]), axis=1
            )
        points["p0_binomial_error"] = points.apply(
                lambda r: binomial_error(r["0"], r["total"]), axis = 1
            )
        points["p1_binomial_error"] = points.apply(
                lambda r: binomial_error(r["1"], r["total"]), axis = 1
            )

        points = points.set_index("amount_of_gates")
            
        return points

    def _plot(
            self,
            *,
            save: bool = False,
            savefolder: str = '',
            savename: str = '',
            plot_style: dict | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot success probability vs. circuit depth with binomial error bars.

        Parameters
        ----------
        save : bool, optional
            boolean to determine if plot should be saved, by default False
        savefolder : str, optional
            folder path under which to save the plots, by default ''
        savename : str, optional
            file name under which to save the plots, by default ''
        plot_style : dict | None, optional
            optional style dictionary for config of plot, by default None

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The created figure and axes, for further customization.
        """
        style = DEFAULT_PLOT_STYLE | (plot_style or {})

        with plt.rc_context(style):
            fig, ax = plt.subplots()

            label = "Simulated data" if self.simulation else "Measured data"
            color = "green" if self.simulation else None

            ax.errorbar(
                self.points.index,
                self.points["success_prob"],
                yerr=self.points["success_binomial_error"],
                fmt="D",
                markersize=4,
                color=color,
                label=label,
            )

            ax.set(
                xlabel="Circuit Length $n$",
                ylabel="Success Probability",
                ylim=(0, 1),
            )

            ax.legend(loc="upper right")
            ax.grid(True)

            if save:
                self._save_plot(fig, savefolder, savename)

            return fig, ax

    def _determine_simulation(
            self,
            data: pd.DataFrame,
            simulation: bool | None,
    ) -> bool:
        """Determine whether the data stems from a simulated or hardware backend.

        If ``simulation`` is explicitly given, it's returned as-is. Otherwise the
        backend name in ``data`` is looked up against ``SIMULATOR_BACKENDS``.

        Parameters
        ----------
        data : pandas.DataFrame
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
        MixedBackendError
            If ``data`` contains records from more than one backend.
        """
        if simulation is not None:
            return simulation

        if data.empty:
            raise ValueError("Cannot determine backend from empty data")

        backends = set(data["backend"])

        if len(backends) > 1:
            raise MixedBackendError(
                "provided data has mixed backend measurements"
            )

        backend = next(iter(backends))
        return backend in SIMULATOR_BACKENDS
    
    def _save_plot(
            self,
            fig: plt.Figure,
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

    def _create_results(self) -> BenchmarkResult:
        """Creates a result container from general analysis parameters.

        Returns
        -------
        BenchmarkResult
            result container with parameters like backend, gates, simulation, etc. 
            As well as the general success probabilities. 
        """
        return BenchmarkResult(
            parameters = {
                "backend": self.backend,
                "gates": sorted(self.gates),
                "simulation": self.simulation,
                "depths": self.depths,
                "post_processing": "Generic"
            }, 
            results = {
                "success_probabilities": self.points[
                        ["total", "success_prob", "success_binomial_error"]
                    ].to_dict(orient="index"),
            },
            figure = self._fig,
            )

    def _validate_records(self, records: list[dict]) -> None:
        """Check that every record is a dict containing all ``self.REQUIRED_KEYS``.

        Raises
        ------
        TypeError
            If a record is not a dict.
        KeyError
            If a record is missing required keys.
        """
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(
                    f"Record {i} is a {type(record).__name__}, expected dict."
                )
            missing = self.REQUIRED_KEYS - record.keys()
            if missing:
                raise KeyError(
                    f"Record {i} does not comply with the sequencer data structure; "
                    f"missing keys: {sorted(missing)}"
                )

    @staticmethod
    def _read_jsonl(file_path: Path) -> list[dict]:
        """Parse a JSONL file into a list of dicts.

        Stops at the first blank or malformed line.

        Raises
        ------
        EmptyDataFile
            If no records could be parsed.
        FileNotFound
            If specified path doesn't exist.
        """
        results: list[dict] = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    break
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        "Stopped reading %s at malformed line %d", file_path, len(
                            results) + 1,
                        exc_info=True,
                    )
                    break

        if not results:
            raise EmptyDataFile(
                "Provided sequencer data file is empty. Cannot carry out analysis.")
        return results

    def __str__(self) -> str:
        """String form of this run's results, delegated to :attr:`results`."""
        return self.results.__str__()

    def __repr__(self) -> str:
        """Concise single-line summary for logs, debuggers and notebooks.

        Not ``eval``-able. Never raises: if a summarized property cannot be
        computed (e.g. empty or mixed-backend data), returns
        ``"<ClassName (data unavailable)>"`` instead.
        """
        cls = type(self).__name__
        try:
            depths = self.depths
            depth_range = f"{depths[0]}-{depths[-1]}" if depths else "none"
            return (
                f"<{cls} backend={self.backend!r} gates={sorted(self.gates)!r} "
                f"simulation={self.simulation} depths=[{depth_range}] "
                f"n_depths={len(depths)} n_circuits={len(self.data_raw)}>"
            )
        except Exception:
            return f"<{cls} (data unavailable)>"



def binomial_error(k, N):
    """Standard error of a binomial proportion estimate.

    Equals ``sqrt(p * (1 - p) / N)`` with ``p = k / N``, written in raw
    counts. This is the Wald estimate, so it is 0 for ``k == 0`` or ``k == N``.

    Parameters
    ----------
    k : int | float
        Number of successes.
    N : int | float
        Total number of trials.

    Returns
    -------
    float
        Standard error of ``k / N``, or 0 if ``N == 0`` (no data).
    """
    if N == 0:
        return 0
    return np.sqrt((k/N**2) - (k**2/N**3))
