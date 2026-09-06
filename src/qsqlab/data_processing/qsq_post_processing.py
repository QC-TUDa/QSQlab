"""QSQ (Quantum System Quizzing) benchmark post-processing.

This module implements single-qubit QSQ analysis on top of the generic
:class:`~.post_processing.PostProcessing` pipeline. It computes:

- Short-sequence quantities (depths 0, 2, 4, ...): error probabilities,
  epsilon values, and analytical lower bounds on gate fidelity, state
  preparation quality, and measurement quality.
- Long-sequence quantities (depths ~20+): amplitude/phase/fidelity bounds
  derived in :mod:`.qsq_long_sequences`.

.. warning::
    **This module ships in an incomplete state for the first release.**
    The short-sequence analysis (error probabilities, epsilon values, the
    three short-sequence lower bounds) is considered stable. The long-sequence
    analysis path (:func:`~.qsq_long_sequences.long_seq_fidelity` and
    everything that depends on it, i.e. ``long_seq_results`` /
    ``self._long_seq_results``) has *not* been fully validated end-to-end
    and is known to disagree with reference implementation in at
    least one case. Treat ``long_seq_results`` as provisional until that module 
    is revisited. Similarly, ``error_short_seq_fidelity_lb`` is a stubbed 
    placeholder (always ``None``) pending a proper uncertainty propagation — see
    :meth:`QSQPostProcessing._get_error_short_seq_fidelity_lb`.
"""

import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from pathlib import Path

from dataclasses import dataclass
from scipy.stats import beta

from .qsq_short_sequences import *
from .qsq_long_sequences import long_seq_fidelity
from .post_processing import PostProcessing, BenchmarkResult

from ..config.plot_config import DEFAULT_PLOT_STYLE 


@dataclass(frozen=True)
class QSQBenchmarkResult(BenchmarkResult):
    """
    Result container for a QSQ (Quantum System Quizzing) benchmark run.

    This class stores the estimated error metrics, fidelity lower bounds,
    and performance indicators for state preparation and measurement,
    together with the raw per-depth maps they were derived from. It extends
    ``BenchmarkResult`` with QSQ-specific quantities.

    .. warning::
        Several fields on this result are either unvalidated or intentionally
        unimplemented in this release. See the per-field notes below and the
        module-level warning at the top of this file before relying on
        anything beyond ``error_prob_map``, ``epsilon_map``,
        ``short_seq_fidelity_lb``, ``state_prep_lb``, and ``measurement_lb``.

    Parameters
    ----------
    short_seq_fidelity_lb : float | None
        Analytically computed lower bound on the gate fidelity from short
        sequences (depths 0, 2), value in interval [0, 1]. ``None`` if the
        required depths (0 and 2) aren't both available in ``epsilon_map``.
    error_short_seq_fidelity_lb : float | None
        Intended as the uncertainty on ``short_seq_fidelity_lb``.
        **Not yet implemented** — currently always ``None`` regardless of
        input. See :meth:`QSQPostProcessing._get_error_short_seq_fidelity_lb`.
    state_prep_lb : float | None
        Lower bound of state preparation quality, value in interval [0, 1].
        ``None`` if depth 0 is unavailable in ``epsilon_map``.
    measurement_lb : float | None
        Lower bound of measurement quality, value in interval [0, 1].
        ``None`` if depth 0 is unavailable in ``epsilon_map``.
    long_seq_results : dict[str, float] | None
        Dictionary of long-sequence fidelity/amplitude/angle bounds as
        returned by :func:`~.qsq_long_sequences.long_seq_fidelity` (keys
        include ``fid_estimate``, ``fid_lower_NA``, ``fid_lower_CP``,
        ``l1_estimate``, ``phi_min``, ``xi_min``, ``optimal_n``). ``None``
        if long-sequence analysis could not be performed (e.g. insufficient
        depth, failing sanity checks, or an internal error — the current
        implementation collapses all of these into a single ``None``; see
        :meth:`QSQPostProcessing._get_long_seq_fidelity_lb`).

        .. warning::
            Unvalidated in this release — see the module-level warning.
    eps_map : dict[int, dict[str, float]] | None
        Dictionary of epsilon (εₖ) values for available depths k, each entry
        holding ``eps``, ``NA`` (normal-approximation error), ``CP_lo``, and
        ``CP_hi`` (Clopper-Pearson bounds). ``None`` if ``error_prob_map`` is
        ``None``.

        .. warning::
            ``CP_lo``/``CP_hi`` rely on ``scipy.stats.beta.ppf`` calls that
            are out-of-domain (return ``nan``) at the boundary case where a
            depth has 0 or 100% survival counts. This is plausible at depth
            0 in particular and is not yet guarded against — see
            :meth:`QSQPostProcessing._get_error_probabilities`.
    error_prob_map : dict[int, dict[str, float]] | None
        Per-depth dict with keys ``error_prob``, ``NA``, ``CP_lo``, ``CP_hi``.
        ``None`` if no sequencer points were available to process.

    Attributes
    ----------
    backend : str | None
        Backend on which the benchmark was executed (inherited).
    gate : str | None
        Quantum gate or operation used in the circuit sequence (inherited).
    depths : list[int]
        Circuit depths used in the benchmark (inherited).
    precision : float
        Minimum acceptable statistical deviation used to derive the
        confidence level for all interval estimates in this result
        (``confidence = 1 - precision``).

    Notes
    -----
    The numerical and analytical fidelity lower bounds may differ due to
    approximation methods and assumptions used in their computation. This
    result is a read-only snapshot: it does not recompute anything and
    reflects exactly what :class:`QSQPostProcessing` produced at
    construction time.
    """
    short_seq_fidelity_lb: float | None
    error_short_seq_fidelity_lb: float | None
    state_prep_lb: float | None
    measurement_lb: float | None

    long_seq_results: dict[float, float | None] | None
    #
    epsilon_map: dict[int, float] | None
    error_prob_map: tuple[float] | None
    precision: float

    def __str__(self) -> str:
        # NOTE: this is a hand-formatted, fixed-width text dump kept for
        # quick console inspection. It is not intended as the long-term
        # presentation layer — a structured renderer (e.g. a markdown/HTML
        # formatter driven by field metadata) is a planned follow-up so that
        # adding/renaming fields doesn't require touching this method.
        return (
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% {self.__class__.__name__} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"\n"
            f"============================== Short Sequnce Results =============================\n"
            f"short_seq_fidelity_lb = {_fmt(self.short_seq_fidelity_lb)}, +/- {_fmt(self.error_short_seq_fidelity_lb)}\n"
            f"state_prep_lb = {_fmt(self.state_prep_lb)}, \n"
            f"measurement_lb = {_fmt(self.measurement_lb)}, \n"
            f"============================= Long Sequnce Results =============================\n"
            f"{"long_seq_results: None\n" if self.long_seq_results is None else 
               (
                    f"long_seq_fidelity_lb:        Fid    >=  {self.long_seq_results["fid_estimate"]:.8f} +/- {self.long_seq_results["fid_estimate"] - self.long_seq_results["fid_lower_NA"]:.8f}/{self.long_seq_results["fid_estimate"] - self.long_seq_results["fid_lower_CP"]:.8f}\n"
                    f"long_seq_amplitude_bound:   |λ_1|   >= {self.long_seq_results["l1_estimate"]:.8f}   obtained from epsilon_{self.long_seq_results["optimal_n"]}\n"
                    f"long_seq_angle_bound:        |φ|    <= {self.long_seq_results["phi_min"]:.8f} rad   obtained from Xi_{self.long_seq_results["xi_min"]["j"]}\n"
                )
               }"
            f"============================= Additional Information =============================\n"

            f"error probability map = {'None' if self.error_prob_map is None else ', '.join([f'{a}: {float(b["error_prob"]):.8f}' for a, b in list(self.error_prob_map.items())[:4]] + (['...'] if len(self.error_prob_map) > 4 else []))}\n"
            f"εₖ map = {'None' if self.epsilon_map is None else ', '.join([f'{a}: {float(b["eps"]):.8f}' for a, b in list(self.epsilon_map.items())[:4]] + (['...'] if len(self.epsilon_map) > 4 else []))}\n"
            f"backend = {self.backend}, \n"
            f"gate = {self.gate}, \n"
            f"depths = {_format_depths_list(self.depths, indent_spaces=0)}, \n"
        )
 

class QSQPostProcessing(PostProcessing):
    """QSQPostProcessing class, child of PostProcessing.

    This class evaluates data from BenchmarkSequencer the same way as its
    parent class and performs QSQ (Quantum System Quizzing) analysis with
    measurement data: error probabilities, epsilon values, short-sequence
    fidelity/state-prep/measurement lower bounds, and (best-effort)
    long-sequence bounds.

    .. warning::
        **Release status: partially validated.**

        - Short-sequence analysis (:meth:`_get_error_probabilities`,
          :meth:`_get_eps`, :meth:`_get_short_seq_fidelity_lb`,
          :meth:`_get_state_prep_lb`, :meth:`_get_measurement_lb`) is
          considered stable and matches the underlying QSQ formulas.
        - Long-sequence analysis (:meth:`_get_long_seq_fidelity_lb`) calls
          into :mod:`.qsq_long_sequences`, which is not yet fully validated
          — see that module's docstring and the module-level warning above.
          Any exception raised during long-sequence analysis is currently
          swallowed and reported as ``None``, which makes "not enough data"
          and "internal bug" indistinguishable from the outside. This is a
          known limitation to be addressed in a follow-up release
          (tightening the ``except`` clause and/or surfacing a reason).
        - ``error_short_seq_fidelity_lb`` (the uncertainty on the short
          sequence fidelity bound) is not yet implemented; see
          :meth:`_get_error_short_seq_fidelity_lb`.

    Attributes
    ----------
    _precision : float
        Minimum acceptable statistical deviation; confidence level for all
        interval estimates is ``1 - precision``.
    _err_prob_map : dict[int, dict[str, float]] | None
        Per-depth error probabilities and confidence bounds. See
        :meth:`_get_error_probabilities`.
    _eps_map : dict[int, dict[str, float]] | None
        Per-depth epsilon (εₖ) values and confidence bounds. See
        :meth:`_get_eps`.
    _short_seq_fidelity_lb : float | None
        See :meth:`_get_short_seq_fidelity_lb`.
    _state_prep_lb : float | None
        See :meth:`_get_state_prep_lb`.
    _measurement_lb : float | None
        See :meth:`_get_measurement_lb`.
    _long_seq_results : dict[str, float] | None
        See :meth:`_get_long_seq_fidelity_lb` and the class-level warning.
    """
    def __init__(
            self,
            sequencer_data: Path | list[dict] | None = None,
            precision: float = 0.01,
            *,
            show_plot: bool = True,
            save: bool = False,
            **postprocessing_kwargs,
        ):
        """Constructor for QSQPostProcessing class.

        During initialization, single-qubit QSQ (Quantum System Quizzing)
        performance indicators are calculated. Results such as error
        probability, numerical and analytical lower bound fidelities, state
        preparation and measurement quality are stored in a dataclass under
        the attribute ``results``.

        Parameters
        ----------
        sequencer_data : Path | list[dict] | None, optional
            Sequencer data to analyze. Can be the path to a jsonl file or a
            list of dicts. If ``None``, reads the latest data from the
            default saving folder, by default ``None``.
        precision : float, optional
            Minimum acceptable statistical deviation, by default ``0.01``.
            Used as ``confidence = 1 - precision`` throughout this class's
            analysis.
        show_plot : bool, optional
            Whether to display the epsilon-vs-depth plot, by default
            ``True``. Note this plot differs from the parent class's
            survival-probability plot — see :meth:`_plot`.
        save : bool, optional
            Whether to save the plot in both PDF and PNG formats, by
            default ``False``.
        **postprocessing_kwargs
            Forwarded to :class:`~.post_processing.PostProcessing`, plus
            optionally ``savefolder``, ``savename``, ``plot_style`` used by
            this class's own :meth:`_plot` call.

        Raises
        ------
        Exception
            Any exception raised by the parent :class:`PostProcessing`
            constructor while reading/validating ``sequencer_data``
            propagates unchanged (see that class's docstring). QSQ-specific
            analysis steps in this constructor generally degrade to
            logging a warning and storing ``None`` rather than raising —
            see the individual ``_get_*`` methods for exact conditions.
        """
        super().__init__(
            sequencer_data=sequencer_data, 
            show_plot=False, 
            save=False,
            **postprocessing_kwargs,
            )
        
        self._precision = precision
        
        self._err_prob_map: dict[str, float | dict[int, float]] | None = self._get_error_probabilities(
            self.points, 1 - self._precision
            )
        self._eps_map: dict[str, float | dict[int, float]] | None = self._get_eps(
            self._err_prob_map
        )
        
        self._short_seq_fidelity_lb: float = self._get_short_seq_fidelity_lb(self._eps_map)
        self._state_prep_lb: float = self._get_state_prep_lb(self._eps_map) 
        self._measurement_lb: float = self._get_measurement_lb(self._eps_map) 

        self._long_seq_results: tuple[float | None] | None = self._get_long_seq_fidelity_lb()

        if show_plot or save:
            self._fig, self._ax = self._plot(
                show=show_plot,
                save=save,
                savefolder= postprocessing_kwargs.get("savefolder", ''),
                savename=postprocessing_kwargs.get("savename", ''),
                plot_style=postprocessing_kwargs.get("plot_style"),
            )

    @property
    def results(self) -> QSQBenchmarkResult:
        """Snapshot of all QSQ performance indicators for this run.

        Returns
        -------
        QSQBenchmarkResult
            Immutable result container. See that class's docstring —
            especially the warnings on ``long_seq_results`` and
            ``error_short_seq_fidelity_lb`` — before consuming fields
            beyond the short-sequence quantities.
        """
        return QSQBenchmarkResult(
            error_prob_map= self._err_prob_map,
            epsilon_map = self._eps_map,
            precision = self._precision,
            #
            short_seq_fidelity_lb = self._short_seq_fidelity_lb,
            error_short_seq_fidelity_lb = self._get_error_short_seq_fidelity_lb(self._eps_map),
            state_prep_lb = self._state_prep_lb,
            measurement_lb = self._measurement_lb,


            long_seq_results = self._long_seq_results,
            #
            backend = self._backend, 
            gate = self._gate, 
            simulation = self._simulation, 
            depths = self._depths,
        )

    def _get_error_probabilities(
            self, 
            points: dict[int, dict] | None,
            confidence: float = 0.99,
            ) -> dict[int, dict[str, float]] | None:
        """Calculate the error probability for each sequence depth.

        Called once during ``__init__``. For every depth present in
        ``points``, computes the error probability (``1 - survival_prob``),
        its normal-approximation ("NA") binomial error, and its
        Clopper-Pearson confidence interval (``CP_lo``, ``CP_hi``) at the
        given confidence level.

        Parameters
        ----------
        points : dict[int, dict] | None
            Per-depth survival data as produced by the parent class's
            ``_process_points``. ``None`` if no data was available.
        confidence : float, optional
            Confidence level for the Clopper-Pearson interval, by default
            ``0.99``.

        Returns
        -------
        dict[int, dict[str, float]] | None
            Per-depth dict with keys ``error_prob``, ``NA``, ``CP_lo``,
            ``CP_hi``. ``None`` if ``points`` is ``None``.

        .. warning::
            ``beta.ppf`` is mathematically undefined (returns ``nan``) when
            a depth has 0 successes or 0 failures — i.e. 0% or 100%
            survival, which is plausible at low depths (0, 2). This method
            does not currently special-case that boundary (marked with a
            ``# FIXME`` in the source); a NaN here will silently propagate
            into ``CP_lo``/``CP_hi`` and any downstream sum that uses them.
        """
        if points is None: return None

        probs = {}

        for i in points.keys():
            probs.update({
                i: {
                    "error_prob": 1 - points[i]['survival_prob'],
                    "NA": binomial_error(
                        points[i]["survival_counts"], 
                        points[i]["total_counts"]
                        ),
                    "CP_lo": beta.ppf( # FIXME custom function with nan conditionals
                        confidence / 2, points[i]["survival_counts"], 
                        points[i]["total_counts"] - points[i]["survival_counts"] + 1
                        ),
                    "CP_hi": beta.ppf( # FIXME custom function with nan conditionals
                        1 - confidence / 2, points[i]["survival_counts"] + 1, 
                        points[i]["total_counts"] - points[i]["survival_counts"]
                        )
                    }
                })

        return probs
    
    def _get_eps(
            self,
            probs: dict[int, dict] | None,
            ) -> dict[int, dict[str, float]] | None:
        """Calculate epsilon (εₖ) values for all sequence depths.

        Called once during ``__init__``, after :meth:`_get_error_probabilities`.
        Combines error probabilities at depths ``k`` and ``k+2`` into an
        epsilon value plus its propagated NA and Clopper-Pearson bounds
        (errors summed linearly across the depth pair). For the exact
        definition of ε_n, see ``qsq_equations.py`` in this project.

        Parameters
        ----------
        probs : dict[int, dict] | None
            Output of :meth:`_get_error_probabilities`. ``None`` if
            unavailable.

        Returns
        -------
        dict[int, dict[str, float]] | None
            Per-depth dict with keys ``eps``, ``NA``, ``CP_lo``, ``CP_hi``,
            for every depth ``k`` where ``k+2`` is also available. ``None``
            if ``probs`` is ``None``. Depths for which no ``k+2`` partner
            exists are silently omitted (not an error — this is the expected
            way sparse depth sets are handled).

        Notes
        -----
        Errors (``NA``, ``CP_lo``, ``CP_hi``) are combined via simple
        addition across the depth-``k``/depth-``(k+2)`` pair rather than in
        quadrature. This is a conservative (wider) uncertainty estimate;
        worth revisiting if tighter bounds are needed later.
        """
        if probs is None: return None

        p0_map = {}
        ek_map = {}

        for i in probs.keys():
            if i == 0 or i % 4 == 0: 
                p0_map.update({i:1 - probs[i]['error_prob']})
            elif i % 2 == 0 and i % 4 != 0: 
                p0_map.update({i:probs[i]['error_prob']})
        for j in sorted(p0_map.keys()):
            if p0_map.get(j + 2) is None:
                continue
            else:
                ek_map.update({
                    j: {
                        "eps": epsilon_n(p_0_n=p0_map[j], p_0_n_plus_two=p0_map[j + 2], n_order=j),
                        "NA": probs[j]["NA"] + probs[j + 2]["NA"],
                        "CP_lo": probs[j]["CP_lo"] + probs[j + 2]["CP_lo"],
                        "CP_hi": probs[j]["CP_hi"] + probs[j + 2]["CP_hi"],
                        }
                    })
        return ek_map

    def _get_short_seq_fidelity_lb(
            self, 
            epsilon_map: dict | None, 
            ) -> float | None:
        """Calculate the analytical short-sequence gate fidelity lower bound.

        Called once during ``__init__``. Requires epsilon values at depths
        0 and 2.

        Parameters
        ----------
        epsilon_map : dict | None
            Output of :meth:`_get_eps`.

        Returns
        -------
        float | None
            Short-sequence fidelity lower bound, or ``None`` (with a logged
            warning) if ``epsilon_map`` is ``None`` or missing depth 0 or 2.
        """
        if epsilon_map is None:
            logger.warning(
                "Cannot determine short sequence Fidelity lower bound, no epsilon data %s", 
                epsilon_map, 
                )
            return None
        if not all(key in epsilon_map.keys() for key in [0, 2]): 
            logger.warning(
                "Cannot determine analytical Fidelity lower bound without all [0, 2, 4] depths."
                )
            return None
        else: 
            return short_seq_fidelity_lb(
            epsilon_map[0]["eps"], 
            epsilon_map[2]["eps"]
            )

    def _get_error_short_seq_fidelity_lb( # TODO
            self, 
            epsilon_map: dict | None,
        ) -> float | None:
        """Uncertainty on the short-sequence fidelity lower bound.

        .. warning::
            **Not implemented in this release.** This method always
            returns ``None`` regardless of input; it is a placeholder for a
            future uncertainty-propagation calculation (analogous to how
            ``NA``/``CP_lo``/``CP_hi`` are propagated for ``eps`` in
            :meth:`_get_eps`). ``QSQBenchmarkResult.error_short_seq_fidelity_lb``
            will therefore always read ``None`` until this is implemented.

        Parameters
        ----------
        epsilon_map : dict | None
            Output of :meth:`_get_eps`. Currently unused.

        Returns
        -------
        None
            Always ``None`` in this release.
        """
        return None
        
    def _get_state_prep_lb(
            self, 
            epsilon_map: dict | None, 
            ) -> float | None:
        """Calculate the state preparation quality lower bound.

        Called once during ``__init__``. Requires an epsilon value at
        depth 0.

        Parameters
        ----------
        epsilon_map : dict | None
            Output of :meth:`_get_eps`.

        Returns
        -------
        float | None
            State preparation lower bound, or ``None`` (with a logged
            warning) if ``epsilon_map`` is ``None`` or missing depth 0.
        """
        if epsilon_map is None:
            logger.warning(
                "Cannot determine state preparation lower bound, no epsilon data %s", 
                epsilon_map, 
                )
            return None
        if not all(key in epsilon_map.keys() for key in [0]): 
            logger.warning(
                "Cannot determine state preparation lower bound without all [0, 2] depths."
                )
            return None
        else: return state_prep_lb(epsilon_map[0]["eps"])
        
    def _get_measurement_lb(
            self, 
            epsilon_map: dict | None, 
            ) -> float | None:
        """Calculate the measurement quality lower bound.

        Called once during ``__init__``. Requires an epsilon value at
        depth 0.

        Parameters
        ----------
        epsilon_map : dict | None
            Output of :meth:`_get_eps`.

        Returns
        -------
        float | None
            Measurement lower bound, or ``None`` (with a logged warning) if
            ``epsilon_map`` is ``None`` or missing depth 0.
        """
        if epsilon_map is None:
            logger.warning(
                "Cannot determine measurement lower bound, no epsilon data %s", 
                epsilon_map, 
                )
            return None
        if not all(key in epsilon_map.keys() for key in [0]): 
            logger.warning(
                "Cannot determine measurement lower bound without all [0, 2] depths."
                )
            return None
        else: return measurement_lb(epsilon_map[0]["eps"])
    
    def _get_long_seq_fidelity_lb(self) -> float | None:
        """Calculate long-sequence performance indicators, best-effort.

        Called once during ``__init__``. Delegates to
        :func:`~.qsq_long_sequences.long_seq_fidelity`.

        Returns
        -------
        dict[str, float] | None
            Long-sequence fidelity/amplitude/angle bounds (see
            :class:`QSQBenchmarkResult`'s ``long_seq_results`` field for the
            expected keys), or ``None`` if the analysis could not be
            performed.

        .. warning::
            **Known limitation.** This currently catches *any* exception
            from :func:`long_seq_fidelity` (bare ``except:``) and reports
            ``None`` in all cases — including insufficient depth data
            (expected/benign), a failing internal sanity check (also
            benign, logged inside ``long_seq_fidelity`` itself), and a
            genuine bug in that module (not benign). These three cases are
            currently indistinguishable to a caller of this class. A
            follow-up should narrow the exception type and/or propagate a
            reason string alongside the ``None``. See also the
            module-level warning regarding the reliability of
            ``long_seq_fidelity`` itself.
        """
        try:
            return long_seq_fidelity(self.points, self._depths, 1 - self._precision)
        except:
            return None
        
    def _plot(
            self, 
            *, 
            show = True, 
            save = False, 
            savefolder = '', 
            savename = '', 
            plot_style = None
            ) -> tuple[plt.Figure, plt.Axes]:
        """Plot pairs of epsilon values (ε_n) against circuit depth.

        .. note::
            This overrides (does not call) the parent class's ``_plot``,
            which plots raw survival probability instead. This method plots
            ``self._eps_map`` — computed during ``__init__`` — and therefore
            requires at least one valid epsilon pair to have been found;
            calling this directly with an empty/``None`` ``self._eps_map``
            will raise, since axis data is built directly from it without a
            guard.

        Parameters
        ----------
        show : bool, optional
            Whether to display the plot, by default ``True``.
        save : bool, optional
            Whether to save the plot in PDF and PNG formats, by default
            ``False``.
        savefolder : str, optional
            Folder path under which to save the plot, by default ``''``.
        savename : str, optional
            File name under which to save the plot, by default ``''``.
        plot_style : dict | None, optional
            Optional style dictionary overriding ``DEFAULT_PLOT_STYLE``.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The created figure and axes, for further customization.

        Notes
        -----
        As with the parent class's ``_plot``, figure-closing behavior when
        both ``show=True`` and ``save=True`` are set is not yet finalized —
        see the parent class's docstring.
        """
        style = DEFAULT_PLOT_STYLE | (plot_style or {})

        with plt.rc_context(style):
            fig, ax = plt.subplots()

            label = "Simulated data" if self._simulation else "Measured data"
            color = "green" if self._simulation else None

            ax.errorbar(
                self._eps_map.keys(),
                [y["eps"] for y in self._eps_map.values()],
                [y["NA"] for y in self._eps_map.values()],
                fmt="D",
                markersize=4,
                color=color,
                label=label,
            )

            ax.set(
                xlabel="Circuit Length $n$",
                ylabel=r"$\varepsilon_n$",
                ylim=(0, 2.1),
            )

            ax.legend(loc="upper right")
            ax.grid(True)

            ax.xaxis.label.set_fontsize(14)
            ax.yaxis.label.set_fontsize(22)

            if save: self._save_plot(fig, savefolder, savename)
            if show: plt.show()

            return fig, ax
        
    
    def __str__(self):
        return self.results.__str__()
    

# %%%%%%%% Formatting aids for QSQBenchmarkResult
def _fmt(val, precision=8):
            return f"{val:.{precision}f}" if val is not None else "None"

def _format_depths_list(depths, max_line_length=71, indent_spaces=0):
    """Format a list of depths to wrap before column 81."""
    depths_str = str(depths)
    indent = " " * indent_spaces
    
    # If the string fits, return it directly
    if len(depths_str) <= max_line_length - indent_spaces:
        return depths_str
    
    # Otherwise, format with line breaks
    result_parts = []
    current_line = "["
    indent_for_continuation = " " * (len("depths = ") + indent_spaces)
    
    for i, d in enumerate(depths):
        item = str(d)
        # Check if adding this item would exceed the line length
        if current_line != "[" and len(current_line + ",-" + item) > max_line_length - indent_spaces:
            result_parts.append(current_line + ",-")
            current_line = " " + item
        elif current_line == "[": 
            current_line += item
        else:
            current_line += ", " + item
    
    current_line += "]"
    result_parts.append(current_line)
    
    return ("\n" + indent_for_continuation).join(result_parts)