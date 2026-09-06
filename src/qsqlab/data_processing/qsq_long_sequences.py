"""Long sequence fidelity analysis file. Still under active development.
.. warning::
    **The analysis carried out in this file ships an incomplete state for the first release.**
    
    The performance indicators obtained here are known to disagree with reference implementation. 
    Treat this file as provisional until this module is revisited.
"""

import numpy as np
import itertools as it

from scipy import stats
from scipy.optimize import root_scalar

import logging
logger = logging.getLogger(__name__)

def long_seq_fidelity(qsq_points, qsq_depths, confidence=0.99, penalty_short_seq=0.98, penalty_long_seq=0.1):
    maxlength, depths = _complete_qsq_dataset(  
        qsq_points=qsq_points,
        qsq_depths=qsq_depths,
        penalty_short_seq=penalty_short_seq,
        penalty_long_seq=penalty_long_seq,
    )

    if not np.sqrt(qsq_points[0]['eps']) + np.sqrt(qsq_points[2]['eps']) < 0.5:
        logger.info("Unable to run long sequence analysis. Errors too large!")
        return None
    if max(depths) <= 20:                        
        logger.info("Circuit depth too low to conduct long-sequence QSQ")
        return None
    logger.info("Initial trust sufficiently high to conduct long-sequence QSQ")
    max_depth = max(depths)               
    alpha = 1 - confidence

    xi_values = get_xi_values(qsq_points, depths, alpha, max_depth) 

    #--------- Post process potential xi_values to sort out useless values and invalid bounds ------
    logger.debug("=============== DETERMINE XI_VALUES ==================")
    xi0 = xi_values[0]
    xi_constraints_phase = [xi0]
    for xi in xi_values:
        xi['amplitude_bound'] = (1-xi['upper_CP'])**(1/(2*xi['j']))

        # Checks condition needed to ensure that value can be used to improve the bound on angle, 
                # taking into account the CP-intervals
        if xi['j']*np.arcsin(xi0['upper_CP']) + np.arcsin(xi['upper_CP']) < 2*np.pi: # Checks condition needed to ensure that value can be used to improve the bound on angle, taking into account the CP-intervals
            xi_constraints_phase.append(xi)

    #---------- Derive bounds on phase -------------------------
    logger.debug("=============== DERIVE PHASE BOUNDS ==================")
    ind_phi = np.argmin([np.arcsin(xi['xi'])/(2*xi['j']) for xi in xi_constraints_phase])
    xi_min = xi_constraints_phase[ind_phi]
    phi_min = np.arcsin(xi_min['xi'])/(2*xi_min['j'])

    #---------- Amplitude ---------------------------
    logger.debug("=============== AMPLITUDE BOUNDS =====================")
    # Preliminary lower bound on amplitude to ensure validity of subsequent steps
    l1_lower = max([xi['amplitude_bound'] for xi in xi_values])          
    n = 0

    # Refine lambda_1 lower bound if established trust is sufficiently high
    if l1_lower >= 0.9 and abs(np.sin(phi_min)) <= np.sin(np.pi/10):         
        l1_lower, l1, n = lambda1_lower_bound(qsq_points, alpha/2,l1_lower,maxlength)  
    else:
        logger.debug("Conditions for long sequence amplitude estimate not satisfied")
        j_ind = np.argmax([xi['amplitude_bound'] for xi in xi_values])
        j = xi_values[j_ind]['j']
        xi = xi_values[j_ind]
        l1 = (1-xi['xi'])**(1/(2*xi['j']))
        l1_lower = dict({'NA':(1-xi['upper_NA'])**(1/(2*j)),'CP':(1-xi['upper_CP'])**(1/(2*j))})

    #------------- Fidelities with and without statistical noise -----------------------
    Fid =  _fidelity_gate(l1,phi_min)
    Fid_lower_NA = _fidelity_gate(l1_lower['NA'], np.arcsin(xi_min['upper_NA'])/(2*xi_min['j']))
    Fid_lower_CP = _fidelity_gate(l1_lower['CP'], np.arcsin(xi_min['upper_CP'])/(2*xi_min['j']))

    #------------- Print Results ---------------------------------------------------------
    logger.debug("=============== RESULTS ==============================")
    if n!= 0:
        logger.debug(
            f"Bound on amplitude:       |lambda_1| >= {l1:.8f}         obtained from epsilon_{n}"
            )
    else: logger.debug(
        f"Bound on amplitude:       |lambda_1| >= {l1:.8f}         obtained from Xi_{j}"
        )
    logger.debug(
        f"Bound on angle:             |phi|    <= {phi_min/np.pi:.8f} rad     obtained from Xi_{xi_min['j']}"
        )
    logger.debug(
        f"Combined fidelity bound:     Fid     >= {Fid:.8f} +- {Fid-Fid_lower_NA:.8f}/{Fid-Fid_lower_CP:.8f}"
        )
    logger.debug(
        f"with {100*(confidence)}%-confidence interval from normal approximation/Clopper-Pearson respectively"
        )

    return {
        'fid_estimate': Fid, 
        'fid_lower_NA': Fid_lower_NA, 
        'fid_lower_CP': Fid_lower_CP, 
        'l1_estimate': l1, 
        'l1_lower_NA': l1_lower['NA'], 
        'l1_lower_CP': l1_lower['CP'],
        'phi_min': phi_min/np.pi, 
        'xi_min': xi_min,
        'optimal_n': n,
        }

def lambda1_lower_bound( 
                    qsq_points: dict[int, dict], 
                    confidence: float,
                    l1: float, 
                    eps_lengths: list[int]
                    ) -> tuple[dict, float, int]:
    """Function which calculates a lower bound and different confidence bounds on the eigenvalue 
    amplitude from very long sequences (typically depth 100 onwards), assuming that a lower bound 
    l1 >= 0.9 is known from previous analysis"""
    candidates = []
    for n in eps_lengths:
        if n>= 50 and _test_lsc_condition(qsq_points[n]['eps'], l1) == True:
            candidates.append(qsq_points[n])
    ctr = 1
    for c in candidates:
        c['bound'] = _l1_solver(c['depth'], _eta(c['eps']), l1)
        ctr += 1

    ind = np.argmax([c['bound'] for c in candidates])
    optimal_point = candidates[ind]
    n = optimal_point['depth']

    warnings = []
    if qsq_points[8]['warning'] is not None:
        warnings.append(qsq_points[8]['warning'])
    if qsq_points[n]['warning'] is not None:
        warnings.append(qsq_points[n]['warning'])
        
    if warnings:
        logger.debug("Warnings: %s", " ".join(warnings))

    l1_bound = optimal_point['bound']
    eps = _eta(optimal_point['eps'])
    z = _standard_normal_quantile(1 - confidence)
    if optimal_point['eps']<= 1:
        l1_lower_CP = _l1_solver(
            n, _exact_confidence_upper([qsq_points[n], qsq_points[n+2]], confidence), l1
            )
    if optimal_point['eps']> 1:
        l1_lower_CP = _l1_solver(
            n, 2 - _exact_confidence_lower([qsq_points[n], qsq_points[n+2]], confidence), l1
            )
    l1_lower_NA = _l1_solver(n, eps + z* optimal_point['eps_error'], l1)
    return {'CP':l1_lower_CP,'NA': l1_lower_NA}, l1_bound, n

def _complete_qsq_dataset(
        qsq_points: dict[int, dict],
        qsq_depths: list[int],
        penalty_short_seq: float = 0.98,
        penalty_long_seq: float = 0.1,
        ) -> tuple[list[int], set[int]]:
    """... (docstring as before) ...

    Returns
    -------
    tuple[list[int], set[int]]
        Depths with a computable epsilon value, and the full set of depths
        now present in ``qsq_points`` (including any depths fabricated
        during this call — e.g. depth 8 or extrapolated 'virtual' points).
    """
    depths = set(qsq_depths) 

    if 8 not in depths:
        succ = int((2*qsq_points[4]['0'] - qsq_points[0]['0']))
        shots = int(qsq_points[6]['total_counts'])
        succ = int((1 + penalty_short_seq)/2 * succ)
        qsq_points[8] = {
            '0': succ,
            '1': shots - succ,
            'survival_prob': succ/shots,
            'survival_counts': succ,
            'total_counts': shots,
            'binomial_error': qsq_points[4]['binomial_error'],
            'warning': "Extrapolated missing data for depth n=8",
        }
        depths.add(8)

    available_eps_depths = []
    for n in sorted(depths):
        qsq_points[n]['fail_counts'] = qsq_points[n]['total_counts'] - qsq_points[n]['survival_counts']
        qsq_points[n]['warning'] = None
        if n + 2 in depths:
            available_eps_depths.append(n)
            qsq_points[n]['eps'] = 2 - qsq_points[n]['survival_prob'] - qsq_points[n + 2]['survival_prob']
            qsq_points[n]['eps_error'] = np.sqrt(
                qsq_points[n]['binomial_error']**2 + qsq_points[n + 2]['binomial_error']**2
            )
            qsq_points[n]['warning'] = None
        qsq_points[n]['depth'] = n

    if max(available_eps_depths) <= 50:
        for n in sorted(depths):
            if n >= 50 and n + 2 not in depths and qsq_points[n]['warning'] is None:
                n_s = qsq_points[n]['survival_counts']
                n_tot = qsq_points[n]['total_counts']
                n_s = n_s + int(penalty_long_seq * n_tot) * (
                    -_kr_delta(n_s / n_tot >= 0.5) + _kr_delta(n_s / n_tot < 0.5)
                )
                qsq_points[n + 2] = {
                    'depth': n + 2,
                    'survival_counts': n_s,
                    'total_counts': n_tot,
                    'fail_counts': n_tot - n_s,
                    'survival_prob': n_s / n_tot,
                    'binomial_error': np.sqrt((n_s/n_tot**2) - (n_s**2/n_tot**3) + qsq_points[n]['binomial_error']**2),
                }
                qsq_points[n]['eps'] = 1 - qsq_points[n]['survival_prob'] + (n_tot - n_s) / n_tot
                qsq_points[n]['eps_error'] = 2 * qsq_points[n]['binomial_error'] + qsq_points[n + 2]['binomial_error']
                qsq_points[n]['warning'] = f'Epsilon uses made-up data for depth {n+2}, recycled value of depth {n}'
                qsq_points[n + 2]['warning'] = 'Virtual point'
                available_eps_depths.append(n)
                depths.add(n + 2)  

    return available_eps_depths, depths

def get_xi_values(
        qsq_points: dict,
        qsq_depths: list[int], 
        confidence: float, 
        max_depth: int,
        ) -> list[dict]:
    """Creates a dictionary of Xi values and their respective statistical bounds for each of the coefficients i, j"""
    xi_values=[]
    for i in range(2):                                                                                         
        for j in range(1,int(max_depth/4 + 1)):
            
            #Tests for which i,j-combinations we have sufficient data to compute xi_ij
            if j % 2 == 1 and set([2*(i+k*j) + 2* l for l in [0,1] for k in [0,1,2,3]]) <= set(qsq_depths): 
                xi, upper_NA, upper_CP = xi_ij_statistical_bound(qsq_points=qsq_points, i=i, j=j, confidence = confidence/2)
                if max(xi,upper_CP) < 1:
                    xi_values.append(
                        dict({'i':i, 'j':j, 'xi':xi, 'upper_NA':upper_NA, 'upper_CP':upper_CP})
                        ) 
                    
    return xi_values

def xi_ij_statistical_bound(
        qsq_points: dict,
        i: int,
        j: int,
        confidence: float
        ) -> tuple[float, float, float]:
    """Determines the optimal xi_ij to obtain bounds on phase including the statistical bounds, 
    obtained through normal approximation and direct propagation via Clopper-Pearson intervals.
    In principle, one can improve the bound a little bit when j=1, since the expression 
    simplifies, but the improvement doesn't justify the overhead in coding"""

    epsilons = [qsq_points[2*(k*j + i)]['eps'] for k in range(4)]
    """Epsilon values >=1 are useless, as the corresponding bound is not necessarily correct.
       Return values >=1 will be sorted out in the post-processing of xi-values"""

    if max(epsilons)>= 1:  
        return 2,2,2  

    # Calculate exact value and SD-interval first
    epsilons1 = np.sort(epsilons)   #Espilons sorted from smallest to largest
    x = epsilons1[1]
    ind = epsilons.index(x)
    standard_dev = np.zeros(4)

    for k in range(4):              # Determine standard deviation for the epsilons.
        d1 = qsq_points[2*(i+j*k)]['binomial_error']
        d2 = qsq_points[2*(i+j*k)+2]['binomial_error']
        standard_dev[k] = np.sqrt(d1**2 + d2**2) 
 
    if j != 1:  # In this case we know that the different epsilon are statistically independent            
        penalty = 0
    if j == 1:  # Otherwise we get a penalty linked to covariances Cov(eps(i),eps(i+1)) = Var(p_(i+1))
        penalty = sum([(qsq_points[2*k]['binomial_error']**2) for k in range(1,4)]) 

    # Decrease alpha a bit, to take into account the 5-sigma deviation for the fraction below     
    z = _standard_normal_quantile(1-0.99*confidence) 
    sd = z * np.sqrt((sum([standard_dev[i]**2 for i in range(4)]) +  2 * penalty))
    aux = epsilons1[3]  + epsilons1[2] - epsilons1[1] - epsilons1[0] 
    ksi =  np.sqrt(aux/(1-x))  
    upper_NA = np.sqrt((aux + sd)/(1 - x - 5*standard_dev[ind])) 

    "Now calculate the upper confidence bound via Clopper-Pearson and direct propagation"
    eps_lower = [_exact_confidence_lower(
        [qsq_points[2*(i + k*j)],qsq_points[2*(i + k*j+1)]], confidence/5
        ) for k in range(4)]
    eps_upper = [_exact_confidence_upper(
        [qsq_points[2*(i + k*j)],qsq_points[2*(i + k*j+1)]], confidence/5
        ) for k in range(4)]
    values = []
    for sigma in it.permutations([0,1,2,3]):
        values.append(
            (eps_upper[sigma[0]] + eps_upper[sigma[1]] - eps_lower[sigma[2]] - eps_lower[sigma[3]])
            /(1 - eps_upper[2])
            )
        
    return ksi, upper_NA, np.sqrt(max(values))

def _exact_confidence_upper(
        datapoints: dict | list,
        confidence: float
        ) -> float:
    """Determines exact upper confidence interval according to Clopper-Pearson 
    method for (sum of) binomial distributions"""
    match datapoints:
        case dict():
            datapoints = [datapoints]
    m = len(datapoints)
    n_s = [dp['fail_counts'] for dp in datapoints]
    n = [dp['total_counts'] for dp in datapoints]
    return  sum([stats.beta.ppf(1 - confidence/m, n_s[i] + 1, n[i] - n_s[i]) for i in range(m)])

def _exact_confidence_lower(
        datapoints: dict | list,
        confidence: float
        ) -> float:
    """Determines exact lower confidence interval according to Clopper-Pearson 
    method for (sum of) binomial distributions"""
    match datapoints:
        case dict():
            datapoints = [datapoints]
    m = len(datapoints)
    n_s = [dp['fail_counts'] for dp in datapoints]
    n = [dp['total_counts'] for dp in datapoints]
    return sum([stats.beta.ppf(confidence/m, n_s[i], n[i] - n_s[i] + 1) for i in range(m)])

def _l1_solver(
        n: int,
        eta: float,
        l1: float
        ) -> float:
    """Subroutine to solve the polynomial expression to determine the
    lower bound on lambda_1 in lambda1_lower_bound"""
    if n <= 20:     # If depth is too low, we get nonsense
        return 0

    # Checks if the values at the boundary of the passed interval have opposite signs,
    # otherwise the method will fail
    if _fun(l1, n, eta) * _fun(1, n, eta) >= 0:
        return 0

    sol = root_scalar(
        _fun,
        args=(n, eta),
        bracket=[l1, 1],
        x0=l1,
        method='bisect',
        xtol=1e-16,
    )
    return sol.root

def _test_lsc_condition(
        eps: float, 
        l_min: float,
        ) -> bool:
    """Tests some necessary condition to ensure that datapoint can be used to bound l1 via 
    long-sequence analysis in lambda1_lower_bound subroutine"""
    return abs(1-eps) >= 7*(1-l_min)

def _fidelity_gate(amplitude:float, angle: float) -> float:
    """Calculates fidelity lower bound as function of amplitude and phase of complex eigenvalue 
    with gauge optimized for gate"""
    return 1/3 + amplitude*(1 + np.cos(angle))/3

def _standard_normal_quantile(confidence: float) -> float: 
    """Function which determines the necessary prefactor of the standard deviation to 
    obtain the confidence interval for different confidence levels (normal distribution)"""
    dist = stats.norm(loc = 0, scale=1)
    return dist.ppf(0.5 + 0.5*confidence)

def _kr_delta(cond: bool) -> int:
    """Kronecker Delta"""
    if cond == True:
        return 1
    return 0

def _fun(x: float, n: int, eta: float) -> float:
    """fun helper function"""
    return x**(2*n)*np.sqrt(2/(1-8*x*(1-x)) -1 ) + 2*(1-x)- abs(1-eta)

def _eta(x: float) -> float:
    """eta helper function"""
    return min(x, 2 - x)