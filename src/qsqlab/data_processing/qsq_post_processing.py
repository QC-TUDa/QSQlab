import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import statistics
from pathlib import Path
from scipy.optimize import curve_fit

from dataclasses import dataclass

from .qsq_equations import *
from .post_processing import PostProcessing, BenchmarkResult


@dataclass(frozen=True)
class QSQBenchmarkResult(BenchmarkResult):
    """
    Result container for a QSQ (Quantum System Quizzing) benchmark run.

    This class stores the estimated error metrics, fidelity lower bounds,
    and performance indicators for state preparation and measurement, together
    with an overall confidence score. It extends ``BenchmarkResult`` with
    QSQ-specific quantities.

    Parameters
    ----------
    robust_soundness_lb_fidelity : float | None
        Analytically computed lower bound on the process fidelity, value in interval [0, 1].
        ``None`` if unavailable.
    robust_state_prep_quality : float | None
        Robust lower bound of state preparation quality, value in interval [0, 1].
        ``None`` if unavailable.
    robust_soundness_lb_fidelity : float | None
        Robust lower bound of measurement quality, value in interval [0, 1].
        ``None`` if unavailable.
    numerical_lb_fidelity : float | None
        Numerically computed lower bound on the process fidelity with the use of scaling constant, 
        value in interval [0, 1].
        ``None`` if unavailable.
    state_preparation_quality : float | None
        Performance indicator for state preparation quality, value in interval [0, 1].
        ``None`` if unavailable.
    measurement_quality : float | None
        Performance indicator for measurement quality, value in interval [0, 1].
        ``None`` if unavailable.
    epsilon_k_map : dict[int, float] | None
        Dictionary of εₖ values for available k's (based on sequence depths), values in interval 
        [0, 1].
        ``{}`` if unavailable.
    error_probability : tuple[float, float] | None
        Estimated error probability and it's binomial error (error_prob, binomial_error), 
        value in interval [0, 1].
        ``None`` if the error probability could not be estimated.
    confidences: tuple[float, dict[int, float]] | None
        General Statistical Confidence level based on Hoeffding's inequality for sequence depths 
        0, 2, 4, and confidence for the specific depths,
        value in interval (0, 1].

    Attributes
    ----------
    backend : Any
        Backend on which the benchmark was executed.
    gate : Any
        Quantum gate or operation in the circuit sequence.
    depths : list[int]
        Circuit depths used in the benchmark.

    Notes
    -----
    The numerical and analytical fidelity lower bounds may differ due to
    approximation methods and assumptions used in their computation.
    """
    robust_soundness_lb_fidelity: float | None
    robust_state_preparation_quality: float | None
    robust_measurement_quality: float | None

    numerical_lb_fidelity: float | None
    state_preparation_quality: float | None
    measurement_quality: float | None
    #
    epsilon_k_map: dict[int, float] | None
    error_probability: tuple[float] | None
    confidences: tuple[float, dict[int, float]] | None
    precision: float

    def __str__(self) -> str:
        return (
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% {self.__class__.__name__} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
            f"\n"
            f"============================== Short Sequnce Results =============================\n"
            f"robust_soundness_lb_fidelity = {_fmt(self.robust_soundness_lb_fidelity)}, \n"
            f"robust_state_prep_quality = {_fmt(self.robust_state_preparation_quality)}, \n"
            f"robust_measurement_quality = {_fmt(self.robust_measurement_quality)}, \n"
            f"--------------- Old numerical indicators:, \n"
            f"numerical_lb_fidelity = {_fmt(self.numerical_lb_fidelity)}, \n"
            f"state_preparation_quality = {_fmt(self.state_preparation_quality)}, \n"
            f"measurement_quality = {_fmt(self.measurement_quality)}, \n"
            f"============================= Additional Information =============================\n"
            f"averaged error probability for depths [0,2,4] = {"None" if self.error_probability is None else f"{_fmt(self.error_probability[0])} +/- {_fmt(self.error_probability[1])}"}, \n"
            f"εₖ map = {'None' if self.epsilon_k_map is None else ', '.join([f'{a}: {float(b):.8f}' for a, b in list(self.epsilon_k_map.items())[:4]] + (['...'] if len(self.epsilon_k_map) > 4 else []))}\n"
            f"summed confidence for depths [0,2,4] = {_fmt(self.confidences[0])}, \n" #TODO fix this
            f"individual confidences = {', '.join(f'{a}: {float(b):.8f}' for a, b in self.confidences[1].items())}\n"
            f"backend = {self.backend}, \n"
            f"gate = {self.gate}, \n"
            f"depths = {_format_depths_list(self.depths, indent_spaces=0)}, \n"
        )

 
class QSQPostProcessing(PostProcessing):
    """ QSQPostProcessing class, child of PostProcessing
    This class evaluates data from BenchmarkSequencer the same way as its parent class 
    and performs QSQ (Quantum System Quizzing) analysis with measurement data.
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
        """Constructor for QSQPostProcessing class. During initialization, Single qubit QSQ 
        (Quantum System Quizzing) performance indicators are calculated. Results like such as error 
        probability, numerical and analytical lower bound fidelities, state preparation and 
        measurement quality are stored in a data class under the attribute "results".

        Parameters
        ----------
        sequencer_data : Path | list[dict] | None, optional
            sequencer data to analyze. can be the path to a jsonl file or a list of dicts. 
            If None is specified, it reads the latest data from the default saving folder, 
            by default None
        gate_numerical_constant : float, optional
            linear scaling constant for numerical lb fidelity, by default 1
        precision : float, optional
            minimum acceptable statistical deviation, by default 0.01
        show_plot : bool, optional
            boolean to show plot, by default True
        save : bool, optional
            boolean to save plot both in pdf and png formats, by default False
        postprocessing_kwargs: dict, None
            optional set of custom postprocessing options
        """
        super().__init__(
            sequencer_data=sequencer_data, 
            show_plot=show_plot, 
            save=save,
            **postprocessing_kwargs,
            )
        
        self._precision = precision
        
        self._err_prob_map: dict[str, float | dict[int, float]] | None = self._get_error_probabilities(
            self.points
            )
        self._confidence_map: dict[str, float | dict[int, float]] | None = self._get_confidences(
            self.points
            )
        self._eps_map: dict[str, float | dict[int, float]] | None = self._get_eps(
            self.points
        )
        
        self._robust_soundness_lb_fidelity: float = self._get_robust_soundness_lb_fidelity(self._eps_map)
        self._robust_state_prep_quality: float = self._get_robust_state_prep_quality(self._eps_map) 
        self._robust_measurement_quality: float = self._get_robust_measurement_quality(self._eps_map) 

        self._numerical_lb_fidelity: float = self._get_numerical_lb_fidelity(self._err_prob_map)
        self._measurement_quality: float = self._get_measurement_quality(self._err_prob_map)
        self._state_prep_quality: float = self._get_state_preparation_quality(self._err_prob_map)

    @property
    def results(self) -> QSQBenchmarkResult:
        """ Result container for QSQ performance indicators """
        return QSQBenchmarkResult(
            error_probability=(
                self._err_prob_map.get("error_prob"), 
                self._err_prob_map.get("binomial_error")
                ) if self._err_prob_map is not None else None,
            confidences = (
                self._confidence_map.get("confidence"),
                self._confidence_map.get("depths")
                ) if self._confidence_map is not None else None,
            epsilon_k_map = self._eps_map,
            precision = self._precision,
            #
            robust_soundness_lb_fidelity = self._robust_soundness_lb_fidelity,
            robust_state_preparation_quality = self._robust_state_prep_quality,
            robust_measurement_quality = self._robust_measurement_quality,

            numerical_lb_fidelity = self._numerical_lb_fidelity, 
            state_preparation_quality = self._state_prep_quality,
            measurement_quality = self._measurement_quality,
            #
            backend = self._backend, 
            gate = self._gate, 
            simulation = self._simulation, 
            depths = self._depths,
        )

    def _get_error_probabilities(
            self, 
            points: dict[int: dict] | None,
            ) -> dict[str, float | dict[int: float]] | None:
        """(called when initializing) this calculates the general error probability for all 
        sequence depths as well as its binomial error, and the individual error probabilities for 
        each of the sequence depths.

        Returns
        -------
        dict[str, float | dict[int: float]] | None
            dictionary with general error probability as well as its general binomial error, and 
            two dictionaries for the error probabilities and binomial errors at each of the 
            sequence depths.
        """
        if points is None: return None

        epsilon = {}
        err_epsilon = {}
        total_epsilon = 0
        total_err_epsilon = 0

        for i in points.keys():
            if i <= 4:
                epsilon.update({i:1 - points[i]['survival_prob']})
                err_epsilon.update(
                    {i: error_fail_probability(points[i]["survival_counts"], 
                                               points[i]["total_counts"])},
                    )
            else: continue
        if epsilon.keys() == 0:
            return None
        else:
            total_epsilon = sum([epsilon[j] for j in epsilon.keys()])/len(epsilon.keys())
            total_err_epsilon = (
                1/len(err_epsilon.keys()))*np.sqrt(sum([err_epsilon[j]**2 for j in err_epsilon.keys()])
                )
        return {
            "error_prob": total_epsilon, 
            "binomial_error": total_err_epsilon, 
            "depths": epsilon, 
            "binomial_error_depths": err_epsilon
            }
    
    def _get_confidences(
            self, 
            points: dict[int: dict] | None,
            ) -> dict[str, float | dict[int: float]] | None:
        """(called when initializing) This determines the confidences which would be obtained for
          a provided statistical deviation according to Hoeffding's inequality given the amount of 
          measurement samples in sequence depths 0, 2, 4.

        Returns
        -------
        dict[str, float | dict[int: float]] | None
            dictionary with general confidence for depths [0, 2, 4] as well as a dictionary for 
            the confidence at each of the sequence depths.
        """
        if points is None: return None

        conf = {}
        total_counts = 0
        total_conf = 0

        for i in points.keys():
            if i <= 4:
                conf.update({i: confidence(points[i]['total_counts'], self._precision)})
                total_counts += points[i]['total_counts']
            else: continue

        total_conf = confidence(total_counts, self._precision)
        return {
            "confidence": total_conf, 
            "depths": conf, 
            }
    
    def _get_eps(
            self,
            points: dict[int: dict] | None,
            ) -> dict[int: float] | None:
        """(called when initializing) this calculates the εₖ for all sequence 
        depths. For more information εₖ is defined in the qsq_equations.py file on this project

        Returns
        -------
        dict[int: float] | None
            dictionary with εₖ each of the sequence depths.
        """
        if points is None: return None
        p0_map = {}
        ek_map = {}
        k = 0
        for i in points:
            if i == 0 or i % 4 == 0: p0_map.update({i:points[i]['survival_prob']})
            elif i % 2 == 0 and i % 4 != 0: p0_map.update({i:1 - points[i]['survival_prob']})
        for j in sorted(p0_map.keys()):
            if p0_map.get(j + 2) is None:
                break
            else:
                ek_map.update(
                    {k:epsilon_k(p_0_2k=p0_map[j], p_0_2k_plus_two=p0_map[j + 2], k_order=k)}
                    )
                k += 1
        return ek_map

    def _get_robust_soundness_lb_fidelity(
            self, 
            epsilon_map: dict | None, 
            ) -> float | None:
        """(called when initializing) Calculates the analytical fidelity lower bound based on the 
        error probabilities in the sequence depths 0, 2, 4

        Returns
        -------
        float | None
            Analytical lower bound fidelity
        """
        if epsilon_map is None:
            logger.warning(
                "Cannot determine Robust Soundness Fidelity lower bound, no epsilon data %s", 
                epsilon_map, 
                )
            return None
        if not all(key in epsilon_map.keys() for key in [0, 1]): 
            logger.warning(
                "Cannot determine analytical Fidelity lower bound without all [0, 2, 4] depths."
                )
            return None
        else: return robust_soundness_lb_fidelity(
            epsilon_map[0], 
            epsilon_map[1]
            )
        
    def _get_robust_state_prep_quality(
            self, 
            epsilon_map: dict | None, 
            ) -> float | None:
        """(called when initializing) Calculates the robust state preparation quality based on the 
        error probabilities in the sequence depths 0, 2

        Returns
        -------
        float | None
            robust state preparation quality
        """
        if epsilon_map is None:
            logger.warning(
                "Cannot determine robust state preparation quality, no epsilon data %s", 
                epsilon_map, 
                )
            return None
        if not all(key in epsilon_map.keys() for key in [0]): 
            logger.warning(
                "Cannot determine robust state preparation quality without all [0, 2] depths."
                )
            return None
        else: return robust_state_preparation_quality(epsilon_map[0])
        
    def _get_robust_measurement_quality(
            self, 
            epsilon_map: dict | None, 
            ) -> float | None:
        """(called when initializing) Calculates the robust measurement quality based on the 
        error probabilities in the sequence depths 0, 2

        Returns
        -------
        float | None
            Robust measurement quality
        """
        if epsilon_map is None:
            logger.warning(
                "Cannot determine robust measurement quality, no epsilon data %s", 
                epsilon_map, 
                )
            return None
        if not all(key in epsilon_map.keys() for key in [0]): 
            logger.warning(
                "Cannot determine robust measurement quality without all [0, 2] depths."
                )
            return None
        else: return robust_measurement_quality(epsilon_map[0])
        
    def _get_numerical_lb_fidelity(
            self, 
            epsilon_map: dict | None,
            ) -> float | None:
        """(called when initializing) calculates the fidelity lower bound with a numerical constant.
        By choosing specific sequences you determine the quality for those specific sequence depths.

        Returns
        -------
        float | None
            Numerical gate fidelity lower bound 
        """
        if epsilon_map is None or "error_prob" not in epsilon_map: return None
        return numerical_lb_fidelity(
            epsilon_map["error_prob"], 
            1
            )

    def _get_measurement_quality(
            self, 
            epsilon_map: dict | None, 
            ) -> float | None:
        """(called when initializing) this determines the measurement quality based on the error 
        probability of provided data. By choosing specific sequences you determine the quality for 
        those specific sequence depths.

        Returns
        -------
        float | None
            Measurement quality.
        """
        if epsilon_map is None or "error_prob" not in epsilon_map: return None
        return measurement_quality(epsilon_map["error_prob"])
        
    def _get_state_preparation_quality(
            self, 
            epsilon_map: dict | None,
            ) -> float | None:
        """(called when initializing) this determines the state preparation quality based on the 
        error probability of provided data. By choosing specific sequences you determine the quality 
        for those specific sequence depths.

        Returns
        -------
        float | None
            State preparation quality.
        """
        if epsilon_map is None or "error_prob" not in epsilon_map: return None
        return state_preparation_quality(epsilon_map["error_prob"])
    
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