"""QSQ analysis file.
"""
import math

from scipy.stats import beta
from collections.abc import Iterable, Mapping
from typing import NamedTuple

import numpy as np
import pandas as pd


class Interval(NamedTuple):
    """Point estimate with its confidence interval."""
    estimate: float
    lower: float
    upper: float

    @property
    def certified(self) -> bool:
        return all(math.isfinite(x) for x in self)

    def shorthand(self, scale: float = 1.0, suffix: str = "") -> str:
        return compact_parenthetical(self.estimate, self.lower, self.upper, scale, suffix)

    def full(self, precision: int = 8) -> str:
        """Full-length form: estimate [lower, upper]."""
        p = precision
        return f"{self.estimate:.{p}g} [{self.lower:.{p}g}, {self.upper:.{p}g}]"

NAN_INTERVAL = Interval(math.nan, math.nan, math.nan)

def epsilon(n: int, probabilities: Mapping[int, float]) -> float:
    """epsilon_n = 1 - (-1)^(n/2) * (P[0|n] - P[0|n+2])."""
    return float(1 - (-1) ** (n // 2) * (probabilities[n] - probabilities[n + 2]))

def available_epsilon_indices(indices: Iterable[int]) -> list[int]:
    """Even n for which both n and n+2 were measured."""
    available = {int(n) for n in indices}
    return sorted(n for n in available if n % 2 == 0 and n + 2 in available)

def clopper_pearson(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    """Exact equal-tailed interval for one binomial probability."""
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper

def bonferroni_box(
    points: pd.DataFrame,
    probability_indices: Iterable[int],
    confidence: float,
) -> tuple[dict[int, tuple[float, float]], float]:
    """Clopper-Pearson box on P[0|n] with joint coverage >= ``confidence``.

    ``points`` is the frame from ``_process_points``: column ``"0"`` holds the
    zero counts and ``"total"`` the shots, indexed by ``amount_of_gates``.
    """
    indices = sorted(set(probability_indices))
    per_probability_confidence = 1 - (1 - confidence) / len(indices)
    box = {
        n: clopper_pearson(
            int(points.loc[n, "0"]),
            int(points.loc[n, "total"]),
            per_probability_confidence,
        )
        for n in indices
    }
    return box, per_probability_confidence

def epsilon_interval(
    n: int, probability_box: Mapping[int, tuple[float, float]]
) -> tuple[float, float]:
    """Exact range of epsilon_n over a rectangular probability box."""
    sign = (-1) ** (n // 2)
    p_n_low, p_n_high = probability_box[n]
    p_next_low, p_next_high = probability_box[n + 2]
    if sign == 1:
        return 1 - p_n_high + p_next_low, 1 - p_n_low + p_next_high
    return 1 + p_n_low - p_next_high, 1 + p_n_high - p_next_low

def compact_parenthetical(
    estimate: float, lower: float, upper: float, scale: float = 1.0, suffix: str = ""
) -> str:
    """Symmetric shorthand notation that conservatively encloses an asymmetric interval."""
    if not all(math.isfinite(x) for x in (estimate, lower, upper)):
        return "not certified"
    center, low, high = scale * estimate, scale * lower, scale * upper
    error = max(center - low, high - center)
    if error == 0:
        return f"{center:.10g}{suffix}"
    exponent = min(math.floor(math.log10(error)), 0)  # assumes error < 1
    while True:
        unit = 10.0 ** exponent
        rounded_center = round(center / unit) * unit
        required_error = max(rounded_center - low, high - rounded_center)
        digit = math.ceil(required_error / unit - 1e-12)
        if digit <= 9 or exponent == 0:
            break
        exponent += 1
    decimals = max(0, -exponent)
    return f"{rounded_center:.{decimals}f}({digit}){suffix}"

def epsilon_probability_indices(epsilon_indices: Iterable[int]) -> list[int]:
    return sorted({k for n in epsilon_indices for k in (n, n + 2)})

def ordered_epsilon_ranges(
    epsilon_indices: Iterable[int],
    probability_box: Mapping[int, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Conservative ranges for the four epsilon order statistics."""
    raw = [epsilon_interval(n, probability_box) for n in epsilon_indices]
    return np.sort([lo for lo, _ in raw]), np.sort([hi for _, hi in raw])

# ============================= Short sequence fidelity ===============================

def epsilon_n(p_0_n: float, p_0_n_plus_two: float, n_order: int) -> float:
    return 1 - ((-1)**(n_order/2))*(p_0_n - p_0_n_plus_two)

def binomial_error(successes: int, total: int) -> float:
    return np.sqrt(successes/total**2 - successes**2/total**3)


def short_fidelity(value: float) -> float:
    return 1 / 3 + 2 / 3 * math.sqrt(1 - value ** 2)

# ============================== Long sequence fidelity ===============================

def long_fidelity(lambda_abs: float, phase_deviation: float) -> float:
    return 1 / 3 + lambda_abs * (1 + math.cos(phase_deviation)) / 3

def phase_candidates(epsilon_indices: Iterable[int]) -> list[tuple[int, int]]:
    """All supported (i, j) with i even and j odd."""
    available = set(epsilon_indices)
    if not available:
        return []
    largest = max(available)
    return [
        (i, j)
        for i in sorted(available)
        for j in range(1, (largest - i) // 6 + 1, 2)
        if {i, i + 2 * j, i + 4 * j, i + 6 * j} <= available
    ]

def ordered_phase_point(eps_hat: Mapping[int, float], i: int, j: int) -> dict:
    """Evaluate xi after sorting the four epsilon values.

    ``eps_hat`` maps n -> epsilon estimate (e.g. from ``_get_eps``).
    """
    indices = (i, i + 2 * j, i + 4 * j, i + 6 * j)
    e0, e1, e2, e3 = sorted(eps_hat[n] for n in indices)

    result = {
        "epsilon indices": indices,
        "ordered epsilon values": (e0, e1, e2, e3),
        "xi": math.nan, "phase": math.nan, "eta_max": math.nan,
        "|lambda| xi bound": math.nan, "status": "ok",
    }
    if e3 > 1:
        result["status"] = "largest ordered epsilon > 1"
        return result

    xi = math.sqrt(max(0.0, e3 + e2 - e1 - e0) / (1 - e1))
    result["xi"] = xi
    if xi > 1:
        result["status"] = "xi > 1, so arcsin(xi) is not defined"
        return result

    angle = math.asin(xi)
    result["phase"] = angle / (2 * j)
    result["eta_max"] = (2 * math.pi - angle) / (2 * j)
    result["|lambda| xi bound"] = (1 - xi) ** (1 / (2 * j))
    return result

def phase_result(
    eps_hat: Mapping[int, float],
    probability_box: Mapping[int, tuple[float, float]],
    i: int,
    j: int,
) -> dict:
    """Propagate one joint CP probability box through the sorted xi formula."""
    point = ordered_phase_point(eps_hat, i, j)
    lower, upper = ordered_epsilon_ranges(
        point["epsilon indices"], probability_box
    )
    a_low, b_low, c_low, d_low = lower
    a_high, b_high, c_high, d_high = upper

    result = dict(point)
    result.update({
        "ordered epsilon lower": tuple(lower), "ordered epsilon upper": tuple(upper),
        "xi lower": math.nan, "xi upper": math.nan,
        "phase lower": math.nan, "phase upper": math.nan,
        "eta_max lower": math.nan, "eta_max upper": math.nan,
        "|lambda| xi bound lower": math.nan, "|lambda| xi bound upper": math.nan,
        "confidence status": "ok",
    })

    if d_high > 1:
        result["confidence status"] = "largest ordered epsilon <= 1 is not certified"
        return result

    numerator_low = max(0.0, d_low + c_low - b_high - a_high)
    numerator_high = max(0.0, d_high + c_high - b_low - a_low)
    xi_low = math.sqrt(numerator_low / (1 - b_low))
    xi_high = math.sqrt(numerator_high / (1 - b_high))
    result["xi lower"], result["xi upper"] = xi_low, xi_high

    if xi_high > 1:
        result["confidence status"] = "arcsin domain is not certified because xi upper > 1"
        return result

    result["phase lower"] = math.asin(xi_low) / (2 * j)
    result["phase upper"] = math.asin(xi_high) / (2 * j)
    result["eta_max lower"] = (2 * math.pi - math.asin(xi_high)) / (2 * j)
    result["eta_max upper"] = (2 * math.pi - math.asin(xi_low)) / (2 * j)
    result["|lambda| xi bound lower"] = (1 - xi_high) ** (1 / (2 * j))
    result["|lambda| xi bound upper"] = (1 - xi_low) ** (1 / (2 * j))
    return result

def amplitude_result(
    eps_hat: float,
    probability_box: Mapping[int, tuple[float, float]],
    ell: int,
) -> dict:
    epsilon_low, epsilon_high = epsilon_interval(ell, probability_box)
    distance_hat = abs(1 - eps_hat)
    distance_low = (
        0.0 if epsilon_low <= 1 <= epsilon_high
        else min(abs(1 - epsilon_low), abs(1 - epsilon_high))
    )
    distance_high = max(abs(1 - epsilon_low), abs(1 - epsilon_high))
    exponent = 1 / (ell - 11)

    if distance_low >= 0.7:
        lower, upper, status = distance_low ** exponent, distance_high ** exponent, "ok"
    else:
        lower = upper = math.nan
        status = "|1 - epsilon_l| >= 0.7 is not certified"

    return {
        "l": ell,
        "|1-epsilon_l| estimate": distance_hat,
        "|lambda| estimate": distance_hat ** exponent,
        "|lambda| lower": lower,
        "|lambda| upper": upper,
        "status": status,
    }
