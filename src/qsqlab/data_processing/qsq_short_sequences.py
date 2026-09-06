"""short sequence fidelity analysis file.
"""
import numpy as np

# ============================== Short sequence fidelity Bounds ===============================

def epsilon_n(p_0_n: float, p_0_n_plus_two: float, n_order: int) -> float:
    return 1 - ((-1)**(n_order/2))*(p_0_n - p_0_n_plus_two)

def binomial_error(successes: int, total: int) -> float:
    return np.sqrt(successes/total**2 - successes**2/total**3)

# Analytical Lower bound
def short_seq_fidelity_lb(e_0: float, e_1: float) -> float:
    f_lb = (1/3) + (2/3)*np.sqrt(1 - (np.sqrt(e_0) + np.sqrt(e_1))**2)
    return float(f_lb)

def state_prep_lb(e_0: float) -> float:
    return 1 - e_0

def measurement_lb(e_0: float) -> float:
    return  e_0

# ========================== Classical certification of quantum gates ===========================

def numerical_lb_fidelity(epsilon: float, m: float = 1) -> float:
    return 1 - m*epsilon

def measurement_quality(epsilon: float) -> float:
    return 1 - (5/2)*epsilon

def state_preparation_quality(epsilon: float) -> float:
    return 1 - (15/2)*epsilon 

# ========================================== General ============================================

def confidence(total_meas_amount: int, epsilon: float, m: float = 1) -> float:
    """ Confidence from Hoeffding's Inequality """
    return 1 - np.exp(-(2*total_meas_amount*(epsilon)**2)/(m**2))

def decaying_oscillation(m, A, B, w, decay, phi):
    return A*((1 + np.cos(w*m + phi)*(decay)**(m)) + B)

def fit_gate_fidelity(decay: float) -> float:
    return ((1 + decay) / 2)
