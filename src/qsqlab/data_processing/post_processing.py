import logging
logger = logging.getLogger(__name__)

import time
import os
import json
import math
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from dataclasses import dataclass

from .equations import *
from ..exceptions import MixedDataException
from ..config.plot_config import DEFAULT_PLOT_STYLE 
from ..config.data_config import DEFAULT_DATA_ROOT
from ..config.backend_config import SIMULATOR_BACKENDS
from ..config.data_config import DEFAULT_DATA_ROOT


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
    This class evaluates data from BenchmarkSequencer, determines survival probabilities 
    and plots data if specified. Additionally it stores results in attribute results.
    This class is intended to be subclassed by more specific Benchmark PostProcessing types that 
    add benchmark specific calculations.
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
        """ General result container for important sequencer details """
        return BenchmarkResult( 
            backend=self._backend, 
            gate=self._gate, 
            simulation=self._simulation, 
            depths=self._depths,
            )
        
    @property
    def _backend(self) -> str | None:
        """ Backend from which count data stems """
        return self._data[0].get("backend")
    
    @property
    def _gate(self) -> str | None:
        """ gate or operation used in sequencer data """
        return self._data[0].get("gate")

    @property
    def _depths(self) -> list[int]:
        """ list of all gate depths in sequence measurements """
        return sorted([k for k in self.points if k not in {"backend", "gate"}])

    @property
    def survival_prob(self) -> list[float]:
        """ list of survival probabilities for each of the sequence depths """
        return [self.points[d]['survival_prob'] for d in self._depths]

    @property
    def errors(self) -> list[float]:
        """ binomial errors for survival probabilities for each of the sequence depths """
        return [self.points[d]['binomial_error'] for d in self._depths]
    
    def _plot(self, 
             *,
             show: bool = True,
             save: bool = False,
             savefolder: str = '', 
             savename: str = '', 
             plot_style: dict | None = None
             ) -> tuple[plt.Figure, plt.Axes]:
        """Ploting method for survival probability data.

        Parameters
        ----------
        show : bool, optional
            boolean to determine if plot should be showed, by default True
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
            Both the matplotlib figure as well as the axes for further optical changes outside of 
            after creation.
        """
        style = DEFAULT_PLOT_STYLE | (plot_style or {})

        with plt.rc_context(style):
            fig, ax = plt.subplots()

            label = "Simulated data" if self._simulation else "Measured data"
            color = "green" if self._simulation else None

            ax.errorbar(
                self._depths,
                self.survival_prob,
                yerr=self.errors,
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

    def _read(self, 
              sequencer_data: Path | list[dict] | None = None
              ) -> list[dict]:
        """This reads the path of a JSONL file. If no file is given, it reads the most recent one. 
        In the default folder path this method reads the jsonl lines until it finds a break. 
        If there's any break separation between measurements in the same file it will read only 
        until the break
        """
        match sequencer_data:
            case []:
                raise ValueError("Cannot process data from empty sequencer data.")
            case [first, *_]:
                required_keys = {"backend", "gate", "counts", "amount_of_gates", "angles", "shots"}
                if not all(key in first for key in required_keys):
                    raise ValueError("Data does not comply with structure of sequencer data dictionaries.")
                else:
                    return sequencer_data
            case None:
                data_root: Path = DEFAULT_DATA_ROOT

                if not data_root.exists():
                    raise FileNotFoundError(f"No data directory found at {data_root}")

                jsonl_files = list(data_root.glob("*.jsonl"))

                if not jsonl_files:
                    raise FileNotFoundError(f"No .jsonl files found in {data_root}, no data to read.")

                # Pick most recently modified file
                file_path = max(jsonl_files, key=lambda p: p.stat().st_mtime)
            case str(n):
                file_path = Path(sequencer_data)
                try:
                    with open(file_path, 'r') as f:
                        results = []
                        for line in f:
                            line = line.strip()
                            if not line: break
                            try:
                                jsonline = json.loads(line)
                                results.append(jsonline)
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.exception(f"finishing loop because of malformed line: {e}")
                                break
                except FileNotFoundError:
                    logger.exception("Error: File %s doesn't exist", file_path)
                    return None
                except Exception as e:
                    logger.exception(f"Unexpected error reading file", exc_info=e)
                    return None
                return results
            case _:
                ValueError(f"Invalid parameter type: {type(sequencer_data).__name__}. expected " + 
                           "Path | list[dict] | None")
    
    def _determine_simulation(
            self,
            data: list[dict],
            simulation: bool | None,
            ) -> bool:
        """ This sets the class attribute simulation. If Nothing is given, it determines if the 
        data stems from a simulation by looking at the name of the backend.
        """
        if simulation is not None:
            return simulation

        if not data: raise ValueError("Cannot determine backend from empty data")

        backends = {d["backend"] for d in data}

        if len(backends) > 1:
            raise MixedDataException(
                "provided data has mixed backend measurements"
            )

        backend = next(iter(backends))
        return backend in SIMULATOR_BACKENDS
    
    def _process_points(
            self, 
            data: list[dict],
            ) -> dict[int, dict]:
        """ This agglomerates all data and transforms it into survival probabilities for each of the
        sequence depths.
        """
        points: dict = {}
        if data==[]: return None

        gates = {d["gate"] for d in data}
        if len(gates) > 1:
            raise MixedDataException(
                "provided data has mixed gate measurements"
            )
        
        depths = list({d["amount_of_gates"] for d in data})
        
        for i in data:
            if i["amount_of_gates"] not in points.keys():
                points[i["amount_of_gates"]] = {'0': 0, '1': 0}

            points[i["amount_of_gates"]]['0'] += i["counts"]['0']
            points[i["amount_of_gates"]]['1'] += i["counts"]['1']

        gate = next(iter(gates))

        if gate == "rz":
            ideals = {}
            for i in depths:
                for j in self._data:
                    if j["amount_of_gates"] == i:
                        ideals.update({i:j["ideal_counts"]})
                        pass
            
            for depth in points.keys():
                total = points[depth]['0'] + points[depth]['1']
                
                ideal_bit = max(ideals[depth], key=ideals[depth].get)
                
                survival_counts = points[depth][ideal_bit]
                
                points[depth].update({
                    "survival_prob": survival_counts / total,
                    "survival_counts": survival_counts,
                    "total_counts": total,
                    "binomial_error": binomial_error(survival_counts, total)
                })

        if gate == "rx" or gate == "ry":
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
                            points[i]['1'], points[i]['0'] + points[i]['1'])})

        return points
    
    def _save_plot(
            self,
            fig,
            savepath: str | None = None,
            savename: str | None = None,
        ) -> tuple[Path, Path]:
        """Save a matplotlib figure as PNG and PDF."""

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
    
# class PostProcessingMultiple(PostProcessing): # TODO improve class overall
#     def __init__(self, 
#                  filepaths: list | None = None,  # List of n file paths
#                  names: list = None, # names which will appear in plots
#                  plot: bool = True, 
#                  simulation: bool = False, 
#                  save: bool = False, 
#                  savefolder: str = '', 
#                  savename: str = ''):
        
#         if filepaths == []:
#             raise ValueError("Please provide at least one file path.")
                
#         self.filepaths = filepaths
#         self.names = names
#         self.simulation = simulation

#         # Read data from all file paths
#         if not filepaths: [self._read()]
#         else: self._data = [self._read(filepath) for filepath in filepaths]
#         self.points = [self._process_results(plot=False, data=data) for data in self._data]
#         if plot: self.plot(save, savefolder, savename)

#     def plot(self, 
#              save: bool = False, 
#              savefolder: str = '', 
#              savename: str = '',
#              ): # TODO fix simulation bool. Make it read from the data dict what type of data it is.
#         """Creates N subplots in a single figure."""
#         n = len(self.points)

#         cols = math.ceil(math.sqrt(n))
#         rows = math.ceil(n / cols)

#         fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
#         axs = np.atleast_1d(axs).flatten()  # SAFE for all n

#         for i, (ax, points) in enumerate(zip(axs, self.points)):
#             depths = list(points.keys())
#             survival_prob = [points[d]['survival_prob'] for d in depths]

#             if self.simulation:
#                 ax.plot(depths, survival_prob, 'o', color="green", label="Simulated data")
#                 title_prefix = "Simulated data"
#             else:
#                 ax.plot(depths, survival_prob, 'o', label="Measured data")
#                 title_prefix = "Measured data"

#             ax.set_xlabel('Circuit Depth (Number of Gates)')
#             ax.set_ylabel('Survival Probability')
#             ax.set_ylim(0, 1)
#             ax.grid(True)
#             ax.legend()

#             if self.names:
#                 ax.set_title(f"{title_prefix} - {self.names[i]}")
#             else:
#                 ax.set_title(title_prefix)

#         # Hide unused axes
#         for ax in axs[n:]:
#             ax.axis("off")

#         plt.tight_layout()

#         if save: self._save_plot(fig, savefolder=savefolder, savename=savename)


# allowed_angles = [np.pi/2, np.pi] TODO write this in PostProcessing
        # if angle not in allowed_angles:
        #     raise ValueError(f"Unsupported angle '{angle}'. Choose from {allowed_angles}")

# if depth%2 != 0 and angle == np.pi/2:  TODO put this in PostProcessing
        #     raise UnevenDepth("Cannot handle sequences of uneven depth for .")

# TODO write angles in sequencer data and make custom errors with it.