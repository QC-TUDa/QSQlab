import numpy as np

# ============================== Sound and Robust fidelity Bounds ===============================

def epsilon_k(p_0_2k: float, p_0_2k_plus_two: float, k_order: int) -> float:
    return 1 - ((-1)**k_order)*(p_0_2k - p_0_2k_plus_two)

def error_fail_probability(successes: int, total: int) -> float:
    return np.sqrt(successes/total**2 - successes**2/total**3)

# Analytical Lower bound
def robust_soundness_lb_fidelity(e_0: float, e_1: float) -> float:
    f_lb = (1/3) + (2/3)*np.sqrt(1 - (np.sqrt(e_0) + np.sqrt(e_1))**2)
    return float(f_lb)

def robust_state_preparation_quality(e_0: float) -> float:
    return 1 - e_0

def robust_measurement_quality(e_0: float) -> float:
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