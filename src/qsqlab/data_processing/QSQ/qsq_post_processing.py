"""QSQ (Quantum System Quizzing) benchmark post-processing.
 
Single-qubit QSQ analysis of sequencer data. From the measured zero-state
probabilities ``P[0|n]`` at even circuit depths ``n``, the module derives
the quantities ``epsilon_n`` and, from them, certified (confidence-interval)
bounds on:
 
* the short-sequence gate fidelity lower-bound (needs depths 0, 2 and 4),
* the phase, amplitude and long-sequence fidelity lower-bounds (needs user-chosen
  parameters ``i``, ``j`` and optionally ``ell``).
 
All intervals are Clopper-Pearson intervals combined with a Bonferroni
correction, so each reported interval has joint coverage of at least
``1 - precision``. The numerical building blocks live in
:mod:`.qsq_helpers`.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import logging
logger = logging.getLogger(__name__)

from pathlib import Path

from .qsq_helpers import (
    Interval,
    NAN_INTERVAL,
    available_epsilon_indices, 
    bonferroni_box, 
    epsilon, 
    epsilon_interval, 
    epsilon_probability_indices, 
    short_fidelity, 
    phase_candidates,
    phase_result,
    amplitude_result,
    long_fidelity, 
    )
from ..post_processing import PostProcessing, BenchmarkResult

from ...config.plot_config import DEFAULT_PLOT_STYLE 
from ...exceptions import NoAvailableEpsilon
 

class QSQPostProcessing(PostProcessing):
    """QSQ (Quantum System Quizzing) post-processing of sequencer data.
 
    Child of :class:`~..post_processing.PostProcessing`. The parent reads and
    validates the sequencer data and builds the per-depth ``points`` table;
    this class then performs the QSQ analysis on it:
 
    1. ``epsilon_n = 1 - (-1)**(n/2) * (P[0|n] - P[0|n+2])`` for every even
       ``n`` where both ``n`` and ``n + 2`` were measured, with confidence
       intervals (:meth:`_get_eps`).
    2. The short-sequence high-fidelity condition and fidelity lower bound
       from ``epsilon_0`` and ``epsilon_2`` (:meth:`_get_short_seq_analysis`).
    3. Optionally, phase, amplitude and long-sequence fidelity bounds for a
       user-supplied ``(i, j, ell)`` (:meth:`_get_long_seq_analysis`).
 
    All results are computed in the constructor. Every interval is an
    :class:`~.qsq_helpers.Interval` ``(estimate, lower, upper)``;
    non-certifiable quantities are filled with ``NaN``.
 
    Attributes
    ----------
    precision : float
        Accepted precision in calculation of statistical errors. 
    confidence : float
        Confidence level of all reported intervals is ``1 - precision``.
    long_seq_params : tuple[int, int, int | None] | None
        User-supplied ``(i, j, ell)`` for the long-sequence analysis, or
        ``None`` if no long-sequence analysis was requested.
    epsilon_map : pandas.DataFrame
        Epsilon values and intervals indexed by depth ``n``.
    available_long_seq_params : dict[str, list]
            Candidate ``(i, j)`` pairs and ``ell`` depths found in the data.
    short_seq_results : dict[str, Interval] | None
        Short-sequence results, or ``None`` if depths 0, 2 and 4 are not all
        available.
    long_seq_results : dict[str, object] | None
        Long-sequence results, or ``None`` if no parameters were given.
 
    Notes
    -----
    Inherited from the parent: ``results``, ``points``, ``data_raw``,
    ``backend``, ``gates``, ``simulation`` and ``depths``. ``points`` must
    contain the columns ``"p0"``, ``"0"`` and ``"total"``, indexed by circuit
    depth. Because of the restrictions in :meth:`_process_points`, only
    even depths enter the analysis and ``depths`` still reports all depths
    present in ``data_raw``, including odd ones.
    """
    REQUIRED_KEYS = frozenset(
            {"ideal_probs", "backend", "gate", "counts",
                "amount_of_gates", "angles", "shots"}
        )
    
    def __init__(
            self,
            sequencer_data: Path | list[dict] | None = None,
            precision: float = 0.01,
            long_seq_params: tuple[int, int, int | None] | None = None,
            *,
            show_results: bool = True,
            save_plot: bool = False,
            savefolder: str = "",
            savename: str = '', 
            simulation: bool | None = None,
            plot_style: dict | None = None,
        ):
        """Run the QSQ analysis on sequencer data.
 
        On construction the epsilon table, short-sequence results and (if
        ``long_seq_params`` is given) long-sequence results are computed,
        and the result container and plot are built via the parent's
        ``_finalize``. The container is available as the :attr:`results`.
 
        Parameters
        ----------
        sequencer_data : Path | list[dict] | None, optional
            Sequencer data to analyze: a path to a jsonl file or a list of
            dicts. If ``None``, the latest data in the default saving folder
            is used. By default ``None``.
        precision : float, optional
            Significance level ``alpha`` in the open interval ``(0, 1)``;
            smaller means tighter confidence, not tighter intervals.
            Intervals are computed at confidence ``1 - precision``, with the
            Bonferroni correction applied across all probabilities that
            enter a given quantity. By default ``0.01``.
        long_seq_params : tuple[int, int, int | None] | None, optional
            ``(i, j, ell)`` for the long-sequence analysis, by default
            ``None`` (skip it).

            * ``i``: starting depth (even).
            * ``j``: depth step (odd, positive). Epsilon is needed at depths
              ``i``, ``i + 2j``, ``i + 4j`` and ``i + 6j``.
            * ``ell``: depth for the refined amplitude estimate (must be one
              of the ``"ell"`` candidates), or ``None`` to use the xi-based
              amplitude bound.

            Valid choices are listed in :attr:`available_long_seq_parameters`.
        show_results : bool, optional
            Whether to display the analysis result container with the markdown 
            renderer, by default ``True``.
        save_plot : bool, optional
            Whether to save the epsilon plot as PDF and PNG, by default ``False``.
        savefolder : str, optional
            Folder in which to save the plot, by default ``""``.
        savename : str, optional
            File name for the saved plot, by default ``""``.
        simulation : bool | None, optional
            Whether the data comes from a simulation. Forwarded to the
            parent, which infers it from the data when ``None``. Controls
            the plot label and color. By default ``None``.
        plot_style : dict | None, optional
            Style overrides applied on top of ``DEFAULT_PLOT_STYLE``, by
            default ``None``.
 
        Raises
        ------
        ValueError
            If ``precision`` is not in ``(0, 1)``; if ``long_seq_params`` does
            not have length 3; if the records are not QSQ data (see
            :meth:`_validate_records`); or if ``long_seq_params`` is not
            supported by the data (see :meth:`_get_long_seq_analysis`).
        NoAvailableEpsilon
            If there are no even-depth records, or no depth pair
            ``(n, n + 2)`` with even ``n`` was measured.
        Exception
            Errors from the parent constructor (reading, validating,
            mixed backends, undefined ideal state) propagate unchanged.

        Notes
        -----
        If depths 0, 2 and 4 are not all available, the short-sequence
        analysis logs a warning and :attr:`short_seq_results` is ``None``
        instead of raising.
        """
        if not 0 < precision < 1:
            raise ValueError(f"precision must lie in the open interval (0, 1), got {precision!r}")
        if long_seq_params is not None and len(long_seq_params) != 3:
            raise ValueError("long_seq_params must be a tuple (i, j, ell).")

        super().__init__(sequencer_data=sequencer_data, simulation=simulation)
        self._precision = precision
        self._long_seq_params = long_seq_params
        
        self._eps_map: pd.DataFrame = self._get_eps()
        self._eps_hat = self._eps_map["epsilon estimate"].to_dict() # helper epsilon in dict format
        self._short_seq_results: dict[str, Interval] | None = self._get_short_seq_analysis()

        self._available_long_seq_params = self._get_available_long_seq_params()
        self._long_seq_results = (
            self._get_long_seq_analysis(*self.long_seq_parameters) if self.long_seq_parameters is not None else None
        )

        self._finalize(
            show_results=show_results,
            save_plot=save_plot,
            savefolder=savefolder,
            savename=savename,
            plot_style=plot_style,
        )
        
    @property
    def precision(self) -> float:
        """float: Significance level ``alpha`` of the reported intervals."""
        return self._precision

    @property
    def confidence(self) -> float:
        """float: Confidence level of the reported intervals, ``1 - precision``."""
        return 1 - self._precision

    @property
    def epsilon_map(self) -> pd.DataFrame:
        """pandas.DataFrame: Epsilon values and intervals indexed by depth ``n``.
 
        Columns are ``epsilon estimate``, ``confidence lower``,
        ``confidence upper``, ``uncertainty lower`` and
        ``uncertainty upper``. See :meth:`_get_eps`.
        """
        return self._eps_map

    @property
    def short_seq_results(self) -> dict[str, Interval] | None:
        """dict[str, Interval] | None: Short-sequence results.
 
        Keys are ``"high_fidelity_condition"`` and ``"short_seq_fidelity"``.
        ``None`` if depths 0, 2 and 4 were not included in the measured data.
        """
        return self._short_seq_results

    @property
    def long_seq_parameters(self) -> tuple[int, int, int | None] | None:
        """tuple[int, int, int | None] | None: The ``(i, j, ell)`` given at
        construction, or ``None`` if no long-sequence analysis was requested."""
        return self._long_seq_params
    
    @property
    def available_long_seq_parameters(self) -> dict[str, list]:
        """dict[str, list]: Long-sequence parameter candidates found in the data.
 
        ``"ij"`` holds supported ``(i, j)`` pairs and ``"ell"`` holds
        candidate ``ell`` depths. See :meth:`_get_available_long_seq_params`.
        """
        return self._available_long_seq_params

    @property
    def long_seq_results(self) -> dict[str, object] | None:
        """dict[str, object] | None: Long-sequence results, or ``None`` if no
        ``long_seq_params`` were given. See :meth:`_get_long_seq_analysis`
        for the keys."""
        return self._long_seq_results
    
    def _validate_records(self, records: list[dict]):
        """Check the parent's key requirements, then that records are QSQ data.

        Parameters
        ----------
        records : list[dict]
            Sequencer records as read by :meth:`PostProcessing._read`.

        Raises
        ------
        TypeError, KeyError
            From the parent check (non-dict record, missing required keys).
        ValueError
            If the gate is not ``'rx'`` with ``theta = pi/2``, or if any
            record was measured in a basis other than ``'Z'``. A missing
            ``measurement_basis`` key is treated as ``'Z'``.
        """
        super()._validate_records(records)
        gates = {r["gate"] for r in records}
        angles = {r["angles"]["theta"] for r in records}
        bases = {r["measurement_basis"] for r in records}
        if gates != {"rx"} and angles != np.pi/2:
            raise ValueError(
                f"QSQ requires gate 'rx' with angle pi/2 only, found {sorted(gates), angles}.")
        if bases != {"Z"}:
            raise ValueError(
                f"QSQ requires Z-basis measurement, found {sorted(bases)}.")

    def _process_points(self, data: pd.DataFrame):
        """Drop odd-depth records, then aggregate as the parent does.

        QSQ uses only even depths, and odd depths of the repeated pi/2
        rotation have ideal probabilities of 0.5/0.5, which the parent
        rejects as an undefined ideal state.

        Parameters
        ----------
        data : pandas.DataFrame
            Sequencer records as returned by :meth:`_read`.

        Returns
        -------
        pandas.DataFrame
            Per-depth table as in :meth:`PostProcessing._process_points`,
            restricted to even depths.

        Raises
        ------
        NoAvailableEpsilon
            If no even-depth records remain.
        UndefinedIdealState
            If an even-depth record has no well-defined ideal state.
        """
        even = data[data["amount_of_gates"] % 2 == 0]
        if even.empty:
            raise NoAvailableEpsilon(
                "No even-depth records to analyse. Cannot perform QSQ analysis"
                )
        return super()._process_points(even)

    def _get_eps(
            self,
        ) -> pd.DataFrame:
        """Compute ``epsilon_n`` with Bonferroni-corrected Clopper-Pearson intervals.
 
        For every even ``n`` for which both ``n`` and ``n + 2`` were
        measured, ``epsilon_n = 1 - (-1)**(n/2) * (P[0|n] - P[0|n+2])``. Its
        interval is the exact range of ``epsilon_n`` over the rectangular
        Clopper-Pearson box on ``(P[0|n], P[0|n+2])``, with confidence split
        between the two probabilities.
 
        Each interval is therefore valid at ``1 - precision`` on its own
        (marginally), not jointly across different ``n``. Downstream methods
        build their own joint boxes where joint coverage is needed.
 
        Returns
        -------
        pandas.DataFrame
            Indexed by ``n`` with columns:
 
            * ``epsilon estimate``: point estimate.
            * ``confidence lower`` / ``confidence upper``: interval bounds.
            * ``uncertainty lower`` / ``uncertainty upper``: distances from
              the estimate to the bounds, clipped at 0 (suitable as
              asymmetric error bars).
 
        Raises
        ------
        NoAvailableEpsilon
            If no valid depth pair exists in the data.
        """
        p0_hat = self._points["p0"].astype(float).to_dict()  # {n: P[0|n]}
        confidence = 1 - self.precision

        rows = {}
        for n in available_epsilon_indices(self._points.index):
            probability_box, _ = bonferroni_box(
                self._points, [n, n + 2], confidence)
            estimate = epsilon(n, p0_hat)
            lower, upper = epsilon_interval(n, probability_box)
            rows[n] = {
                "epsilon estimate": estimate,
                "confidence lower": lower,
                "confidence upper": upper,
                "uncertainty lower": estimate - lower if estimate - lower > 0 else 0,
                "uncertainty upper": upper - estimate if upper - estimate > 0 else 0,
            }
        cols = ["epsilon estimate", "confidence lower", "confidence upper",
                "uncertainty lower", "uncertainty upper"]
        if rows == {}:
            raise NoAvailableEpsilon(
                "Analysed data has no available epsilon probability pairs"
            )
        eps = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
        eps.index.name = "n"

        return eps


    # =========================== Short seq analysis ===========================

    def _get_short_seq_analysis(self) -> dict[str, Interval] | None:
        """Short-sequence fidelity lower bound and its respective high-fidelity condition.
 
        Computes the quantity ``sqrt(eps_0) + sqrt(eps_2)`` (the
        *high-fidelity condition*) and from it the fidelity lower bound
        ``F = 1/3 + 2/3 * sqrt(1 - (sqrt(eps_0) + sqrt(eps_2))**2)``.
 
        Uses one joint Bonferroni box over ``P[0|0]``, ``P[0|2]`` and
        ``P[0|4]``, so both intervals hold jointly with coverage of at least
        ``1 - precision``. Negative epsilon values are clipped to 0 before
        taking square roots.
 
        The fidelity bound is only defined where the condition value is
        ``<= 0.5``:
 
        * If the point estimate exceeds 0.5, the whole fidelity interval is
          ``NaN``.
        * If only the upper end of the condition interval exceeds 0.5, the
          fidelity's lower bound is ``NaN`` (the estimate and upper bound are
          still reported).

        This check is not the high-fidelity regime requirement itself. The
        condition interval is returned so the caller can compare it with the
        threshold required for the bound to be valid; it is not enforced here.
 
        Returns
        -------
        dict[str, Interval] | None
            ``None`` (with a logged warning) if epsilon at depths 0 and 2
            is unavailable, i.e. depths 0, 2 and 4 were not all measured.
            Otherwise:
 
            * ``"high_fidelity_condition"``: Interval for
              ``sqrt(eps_0) + sqrt(eps_2)``.
            * ``"short_seq_fidelity"``: Interval for the fidelity lower
              bound, possibly containing ``NaN`` as described above.
        """
        if not {0, 2} <= self._eps_hat.keys():
            logger.warning(
                "Short-sequence analysis needs depths 0, 2 and 4; skipping.")
            return None

        box, _ = bonferroni_box(self._points, [0, 2, 4], self.confidence)
        e0_low, e0_high = epsilon_interval(0, box)
        e2_low, e2_high = epsilon_interval(2, box)

        sum_hat = math.sqrt(
            max(0.0, self._eps_hat[0])) + math.sqrt(max(0.0, self._eps_hat[2]))
        sum_low = math.sqrt(max(0.0, e0_low)) + math.sqrt(max(0.0, e2_low))
        sum_high = math.sqrt(max(0.0, e0_high)) + math.sqrt(max(0.0, e2_high))

        if sum_hat <= 1:
            # sum_high > 1 would make the sqrt in short_fidelity undefined
            fid_low = short_fidelity(sum_high) if sum_high <= 1 else math.nan
            fid_high = short_fidelity(sum_low)
            fidelity = Interval(
                short_fidelity(sum_hat),
                fid_low, 
                fid_high
                )
        else:
            fidelity = NAN_INTERVAL

        return {
            # high-fidelity regime sum condition must be < 0.5 (for 99% conf.)
            "high_fidelity_condition": Interval(sum_hat, sum_low, sum_high),
            "short_seq_fidelity": fidelity,
        }

    # =========================== Long seq analysis ===========================

    def _get_available_long_seq_params(self) -> dict[str, list]:
        """Find candidate parameters for :meth:`_get_long_seq_analysis`.
 
        Returns
        -------
        dict[str, list]
            * ``"ij"``: all ``(i, j)`` pairs (``i`` even, ``j`` odd) for
              which epsilon is available at ``i``, ``i + 2j``, ``i + 4j``
              and ``i + 6j``.
            * ``"ell"``: depths ``n >= 22`` with ``|1 - eps_n| >= 0.7`` at
              the point estimate. This is necessary but not sufficient for
              the certified condition checked in the analysis, which uses the
              lower end of the interval.

        Notes
        -----
        Logs a warning when no ``(i, j)`` pair or no ``ell`` candidate
        exists, even if no long-sequence analysis was requested. An empty
        ``"ell"`` list still allows the analysis with ``ell=None``.
        """
        indices = list(self._eps_map.index)
        ij_candidates = phase_candidates(indices)
        ell_candidates = [n for n in indices if n >= 22
                        and abs(1 - self._eps_hat[n]) >= 0.7]
        if not ij_candidates:
            logger.warning(
                "No (i, j) pair has epsilon at all of i, i+2j, i+4j, i+6j; "
                "long-sequence analysis is not possible with this data."
            )
        if not ell_candidates:
            logger.warning(
                "No depth qualifies as ell (n >= 22, |1 - eps_n| >= 0.7); "
                "only the xi-based amplitude bound (ell=None) can be used."
            )

        return {"ij": ij_candidates, "ell": ell_candidates}

    def _get_long_seq_analysis(
        self, i: int, j: int, ell: int | None
    ) -> dict[str, object]:
        """Phase, amplitude and long-sequence fidelity bounds (Lemmas 3-5, Eq. 19).
 
        The phase uses the four epsilon values at depths ``i``, ``i + 2j``,
        ``i + 4j``, ``i + 6j``, sorted ascending as ``e0 <= e1 <= e2 <= e3``:
        ``xi = sqrt((e3 + e2 - e1 - e0) / (1 - e1))``, phase
        ``theta = arcsin(xi) / (2j)`` and
        ``eta_max = (2*pi - arcsin(xi)) / (2j)``.
 
        The amplitude ``|lambda|`` is obtained in one of two ways:
 
        * ``ell is None``: the "xi-based" bound ``(1 - xi)**(1/(2j))``.
        * ``ell`` given: the "refined l-based" estimate
          ``|1 - eps_ell|**(1/(ell - 11))``, certified only if
          ``|1 - eps_ell| >= 0.7`` holds over the whole interval.
 
        The fidelity is ``1/3 + |lambda| * (1 + cos(phase)) / 3``.
 
        Coverage: the phase and amplitude intervals each come from their own
        Bonferroni box. The fidelity interval uses a single joint box over
        all probabilities involved, so it has joint coverage of at least
        ``1 - precision``. When the fidelity or a component cannot be
        certified, the affected interval entries are ``NaN`` and the
        corresponding status string says why.
 
        Parameters
        ----------
        i : int
            Starting depth (even).
        j : int
            Depth step (odd, positive). Parity and sign are not checked
            directly; only availability of the four phase depths is.
        ell : int | None
            Depth for the refined amplitude estimate; must be one of
            ``available_long_seq_parameters["ell"]``. ``None`` uses the
            xi-based bound.
 
        Returns
        -------
        dict[str, object]
            * ``"params"``: dict with ``i``, ``j``, ``ell``.
            * ``"phase"``: Interval for ``theta`` [rad].
            * ``"phase status"``: ``"ok"`` or the reason the phase interval
              is not certified.
            * ``"eta max"``: Interval for ``eta_max`` [rad], the maximum
              initial deviation ``|theta - pi/2|`` the analysis tolerates
              (use the lower end as the required trust).
            * ``"amplitude"``: Interval for ``|lambda|``.
            * ``"amplitude method"``: ``"xi-based"`` or ``"refined l-based"``.
            * ``"amplitude status"``: ``"ok"`` or the reason the amplitude
              interval is not certified.
            * ``"long seq fid"``: Interval for the long-sequence fidelity
              lower bound.
            * ``"long seq fidelity status"``: ``"ok"`` or the first reason
              the fidelity interval is not certified.
 
        Raises
        ------
        ValueError
            If the data has no ``(i, j)`` candidates at all; if epsilon is
            not available at all of ``i``, ``i + 2j``, ``i + 4j`` and
            ``i + 6j``; or if ``ell`` is not in the ``"ell"`` candidates.
        """
        available = set(self._eps_map.index)
        phase_eps_idx = (i, i + 2 * j, i + 4 * j, i + 6 * j)
        if not self.available_long_seq_parameters["ij"]:
            raise ValueError(
                "Provided data has no available 'ij' indices for long sequence analysis."
            )
        if not set(phase_eps_idx) <= available:
            raise ValueError(
                f"(i={i}, j={j}) needs epsilon at depths {phase_eps_idx}; "
                f"see available_long_seq_params."
                )
        if ell is not None and ell not in self.available_long_seq_parameters["ell"]:
            raise ValueError(
                f"ell={ell} is not an available epsilon depth." 
                f"Choose from {self.available_long_seq_parameters}"
                )

        conf, points, eps = self.confidence, self._points, self._eps_hat
        phase_keys = epsilon_probability_indices(phase_eps_idx)

        # --- phase
        phase_box, _ = bonferroni_box(points, phase_keys, conf)
        phase = phase_result(eps, phase_box, i, j)
        eta_max = Interval(
            phase["eta_max"], phase["eta_max lower"], phase["eta_max upper"])

        # --- amplitude
        if ell is None:
            amp_keys = []
            amplitude = {
                "|lambda| estimate": phase["|lambda| xi bound"],
                "|lambda| lower": phase["|lambda| xi bound lower"],
                "|lambda| upper": phase["|lambda| xi bound upper"],
                "status": phase["confidence status"],
            }
            method = "xi-based"
        else:
            amp_keys = [ell, ell + 2]
            amp_box, _ = bonferroni_box(points, amp_keys, conf)
            amplitude = amplitude_result(eps[ell], amp_box, ell)
            method = "refined l-based"

        # --- fidelity
        if ell is None:
            phase_fid, amp_fid = phase, amplitude
            point_ok = phase_fid["status"] == "ok" and math.isfinite(
                amp_fid["|lambda| estimate"])
            conf_ok = (phase_fid["confidence status"] == "ok"
                       and math.isfinite(amp_fid["|lambda| lower"]))
        else:
            fid_box, _ = bonferroni_box(points, sorted(
                set(phase_keys) | set(amp_keys)), conf)
            phase_fid = phase_result(eps, fid_box, i, j)
            amp_fid = amplitude_result(eps[ell], fid_box, ell)
            point_ok = phase_fid["status"] == "ok" and amp_fid["|1-epsilon_l| estimate"] >= 0.7
            conf_ok = phase_fid["confidence status"] == "ok" and amp_fid["status"] == "ok"

        est = (long_fidelity(amp_fid["|lambda| estimate"], phase_fid["phase"])
               if point_ok else math.nan)
        if conf_ok:
            low = long_fidelity(
                amp_fid["|lambda| lower"], phase_fid["phase upper"])
            high = long_fidelity(
                amp_fid["|lambda| upper"], phase_fid["phase lower"])
            status = "ok"
        else:
            low = high = math.nan
            status = (phase_fid["confidence status"]
                      if phase_fid["confidence status"] != "ok" else amp_fid["status"])

        return {
            "params": {"i": i, "j": j, "ell": ell},
            "phase": Interval(phase["phase"], phase["phase lower"], phase["phase upper"]),
            "phase status": phase["confidence status"],
            "eta max": eta_max,   # required initial trust: |theta - pi/2| <= eta max.lower
            "amplitude": Interval(amplitude["|lambda| estimate"],
                                  amplitude["|lambda| lower"], amplitude["|lambda| upper"]),
            "amplitude method": method,
            "amplitude status": amplitude["status"],
            "long seq fid": Interval(est, low, high),
            "long seq fidelity status": status,
        }

    # ================================= other ==================================

    def _plot(
            self,
            *,
            save=False,
            savefolder='',
            savename='',
            plot_style=None
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot epsilon values ``epsilon_n`` with error bars against circuit depth.
 
        Each point is ``epsilon_n`` with asymmetric error bars from
        ``uncertainty lower/upper`` (marginal intervals at ``1 - precision``,
        not jointly valid across depths). Points are blue for measured data
        and green for simulated data. If long-sequence parameters were given,
        the depths ``i``, ``i + 2j``, ``i + 4j`` and ``i + 6j`` are drawn on
        top in distinct colors, each with its own legend entry; ``ell`` is
        highlighted too, but only when it is not ``None``. The y-axis is
        fixed to ``(0, 2.1)``.
 
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot in PDF and PNG formats, by default
            ``False``.
        savefolder : str, optional
            Folder path under which to save the plot, by default ``''``.
        savename : str, optional
            File name under which to save the plot, by default ``''``.
        plot_style : dict | None, optional
            Style overrides applied on top of ``DEFAULT_PLOT_STYLE``.
 
        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            The created figure and axes, for further customization.
        """
        style = DEFAULT_PLOT_STYLE | (plot_style or {})

        with plt.rc_context(style):
            fig, ax = plt.subplots()

            label = "Simulated data" if self.simulation else "Measured data"
            color = "green" if self.simulation else "blue"

            if self.long_seq_results is not None:
                p = self.long_seq_results["params"]
                i, j, ell = p["i"], p["j"], p["ell"]


            elif self.long_seq_parameters is not None:
                i, j, ell = self.long_seq_parameters
            else:
                i = j = ell = None

            highlights: dict[int, tuple[str, str]] = {}
            if i is not None:
                points = [
                    (i,         r"$n = i$"),
                    (i + 2*j,   r"$n = i+2j$"),
                    (i + 4*j,   r"$n = i+4j$"),
                    (i + 6*j,   r"$n = i+6j$"),
                    (ell,       r"$n = \ell$"),
                ]
                # distinct colors that don't clash with the green/blue data points
                palette = ["tab:red", "tab:orange", "tab:purple", "tab:brown", "tab:pink"]
                highlights = {
                    n: (c, f"{lbl}$ = {n}$")
                    for (n, lbl), c in zip(points, palette)
                }

            index = self._eps_map.index
            is_highlighted = index.isin(highlights.keys())

            def plot_subset(mask, c, lbl, zorder):
                if not mask.any():
                    return
                sub = self._eps_map[mask]
                ax.errorbar(
                    x=sub.index,
                    y=sub["epsilon estimate"],
                    yerr=(sub["uncertainty lower"], sub["uncertainty upper"]),
                    fmt="none",
                    ecolor=c,
                    zorder=zorder,
                )
                ax.scatter(
                    x=sub.index,
                    y=sub["epsilon estimate"],
                    marker="D",
                    s=4**2,
                    c=c,
                    label=lbl,
                    zorder=zorder + 1,
                )

            # default points: bottom layer
            plot_subset(~is_highlighted, color, label, zorder=2)

            # highlighted points: top layer, each with its own legend entry
            for n, (hl_color, hl_label) in highlights.items():
                plot_subset(index == n, hl_color, hl_label, zorder=4)

            ax.set(
                xlabel="Circuit Length $n$",
                ylabel=r"$\varepsilon_n$",
                ylim=(0, 2.1),
            )

            ax.legend(loc="upper right")
            ax.grid(True)

            # ax.xaxis.label.set_fontsize(14)
            # ax.yaxis.label.set_fontsize(22)

            if save:
                self._save_plot(fig, savefolder, savename)

            return fig, ax

    def _create_results(self) -> BenchmarkResult:
        """Create the result container from the QSQ parameters and results.
 
        Called by the parent's ``_finalize``. The ``results`` mapping is
        organized in sections whose headers are keys with value ``None``
        (prefixed ``---``):
 
        * **Short sequence**: fidelity lower bound and high-fidelity
          condition as compact ``estimate(error)`` strings, or a notice if
          depths 0, 2 and 4 were not available.
        * **Long sequence**: phase, ``eta_max``, amplitude and fidelity
          lower bound in compact form, plus ``"long sequence details"``
          with every interval formatted as ``estimate [lower, upper]``. If
          no ``long_seq_params`` were given, a notice pointing to the
          available options is shown instead.
        * **General**: available ``(i, j)`` pairs, available ``ell`` depths
          and an epsilon table with lower statistical uncertainties and confidences.
 
        Uncertified intervals are rendered as ``"not certified"`` in
        compact form.
 
        Returns
        -------
        BenchmarkResult
            Container whose ``parameters`` hold the long-sequence
            parameters (or ``None``), backend, sorted list of gates,
            simulation flag, depths and ``"post_processing": "QSQ"``, whose
            ``results`` hold the sections above, and whose ``figure`` is the
            epsilon plot.
        """
        def format_intervals(obj, precision: int = 8):
            if isinstance(obj, Interval):
                return obj.full(precision)
            if isinstance(obj, dict):
                return {k: format_intervals(v, precision) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [format_intervals(v, precision) for v in obj]
            return obj

        results = {}

        results.update({
            "--- Short sequence results": None,
                        r"$F_{\text{short}}$ fidelity lower bound": self.short_seq_results[
                            "short_seq_fidelity"].shorthand(),
                        "short high-fid condition": self.short_seq_results[
                                "high_fidelity_condition"].shorthand(),
            } if self._short_seq_results is not None else {
                "--- Short sequence analysis unavailable. "
                "Depths ``{0, 2, 4}`` not included in analysis.": None
            })

        results.update({
                "---  Long sequence results": None,
                r"phase $\theta$ [rad]": self.long_seq_results["phase"].shorthand(),
                r"$\eta_\text{max}$[rad]": self.long_seq_results["eta max"].shorthand(),
                r"amplitude $|\lambda|$": self.long_seq_results["amplitude"].shorthand(),
                r"$F_{\text{long}}$ fidelity lower bound": 
                self.long_seq_results["long seq fid"].shorthand(),
                "long sequence details": format_intervals(self.long_seq_results), 
            } if self._long_seq_params is not None else {
                "--- No provided long sequence parameters, choose from available "
                "parameters. See available ij/ell ": None
            })

        results.update({
            "--- General results ": None,
            r"available $ij$": self.available_long_seq_parameters["ij"] if 
                self.available_long_seq_parameters["ij"] != [] else None,
            r"available $\ell$": self.available_long_seq_parameters["ell"] if 
                self.available_long_seq_parameters["ell"] != [] else None,
            r"$\varepsilon_n$ values": self.epsilon_map[["epsilon estimate",
                "uncertainty lower", "confidence lower"]].to_dict(orient="index"),
            })

        return BenchmarkResult(
            parameters = {
                "long sequence parameters": {
                    r"$i$": self._long_seq_params[0], 
                    r"$j$": self._long_seq_params[1],
                    r"$\ell$": self._long_seq_params[2]
                    } if self._long_seq_params else None,

                "backend": self.backend,
                "gates": sorted(self.gates),
                "simulation": self.simulation,
                "depths": self.depths,
                "post_processing": "QSQ"
            },
            results = results,
            figure = self._fig
        )
        

    def __repr__(self):
        """Concise single-line summary for logs, debuggers and notebooks.

        Shows precision, long-sequence parameters, backend, gates,
        simulation flag, depth range and counts of depths and circuits.
        Not ``eval``-able. Catches ``Exception`` (not ``BaseException``) and
        returns ``"<ClassName (data unavailable)>"`` if a summarized
        property cannot be computed, e.g. for empty or mixed-backend data.
        """
        cls = type(self).__name__
        try:
            depths = self.depths
            depth_range = f"{depths[0]}-{depths[-1]}" if depths else "none"
            return (
                f"<{cls} precision={self.precision!r} long_seq_params={self.long_seq_parameters!r} "
                f"backend={self.backend!r} gates={sorted(self.gates)!r} "
                f"simulation={self.simulation} depths=[{depth_range}] "
                f"n_depths={len(depths)} n_circuits={len(self.data_raw)}>"
            )
        except Exception:
            return f"<{cls} (data unavailable)>"
        
    