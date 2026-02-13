import numpy as np
from scipy.optimize import brentq
from typing import Callable, Tuple, Dict


def bootstrap_hazard_rates(
    oas_spreads: Dict[float, float],  # {tenor_years: oas_bps}
    zero_curve: Callable,
    recovery_rate: float = 0.40
) -> Tuple[Callable, np.ndarray, np.ndarray]:
    """
    Bootstrap piecewise constant hazard rates from OAS spreads.
    
    Args:
        oas_spreads: Dictionary mapping tenors to OAS in basis points
        zero_curve: Zero rate curve function
        recovery_rate: Recovery rate (default 40%)
    
    Returns:
        Tuple of (hazard_function, tenor_array, hazard_array)
    """
    tenors = np.array(sorted(oas_spreads.keys()))
    hazards = []
    
    for i, T in enumerate(tenors):
        oas = oas_spreads[T] / 10000.0  # bps to decimal
        
        if i == 0:
            # First tenor: simple approximation s ≈ (1-R)*λ
            lam = oas / (1 - recovery_rate)
            hazards.append(lam)
        else:
            # Solve for hazard rate to match CDS premium
            def cds_equation(lam):
                # Calculate survival probabilities
                S_prev = compute_survival(tenors[:i], hazards, tenors[i-1] if i > 0 else 0)
                
                # Current bucket contribution
                t_prev = tenors[i-1] if i > 0 else 0
                dt = T - t_prev
                S_curr = S_prev * np.exp(-lam * dt)
                
                # Protection leg PV
                protection = 0.0
                for j in range(i):
                    t_start = tenors[j-1] if j > 0 else 0
                    t_end = tenors[j]
                    S_start = compute_survival(tenors[:j+1], hazards[:j+1], t_start)
                    S_end = compute_survival(tenors[:j+1], hazards[:j+1], t_end)
                    df = np.exp(-zero_curve(t_end) * t_end)
                    protection += (1 - recovery_rate) * (S_start - S_end) * df
                
                # Add current bucket
                df = np.exp(-zero_curve(T) * T)
                protection += (1 - recovery_rate) * (S_prev - S_curr) * df
                
                # Premium leg PV
                premium = 0.0
                for j in range(i+1):
                    t = tenors[j]
                    dt_prem = tenors[j] - (tenors[j-1] if j > 0 else 0)
                    
                    if j < i:
                        S_t = compute_survival(tenors[:j+1], hazards[:j+1], t)
                    else:
                        S_t = S_curr
                    
                    df = np.exp(-zero_curve(t) * t)
                    premium += oas * dt_prem * S_t * df
                
                return protection - premium
            
            # Solve for hazard rate
            try:
                lam = brentq(cds_equation, 0.0001, 0.5, xtol=1e-6, maxiter=100)
                hazards.append(lam)
            except Exception as e:
                # Fallback to approximation
                print(f"Warning: Hazard rate bootstrap failed at {T}Y, using approximation")
                lam = oas / (1 - recovery_rate)
                hazards.append(lam)
    
    hazards = np.array(hazards)
    
    # Create piecewise constant hazard function
    def hazard_func(t):
        """Return hazard rate at time t."""
        t = np.asarray(t)
        scalar_input = t.ndim == 0
        t = np.atleast_1d(t)
        
        result = np.zeros_like(t)
        for i in range(len(t)):
            for j, T in enumerate(tenors):
                if t[i] <= T:
                    result[i] = hazards[j]
                    break
            else:
                result[i] = hazards[-1]
        
        return result.item() if scalar_input else result
    
    return hazard_func, tenors, hazards


def compute_survival(tenors: np.ndarray, hazards: np.ndarray, t: float) -> float:
    """
    Compute survival probability S(t) = P(τ > t) for piecewise constant hazards.
    
    Args:
        tenors: Tenor buckets
        hazards: Hazard rates for each bucket
        t: Time point
    
    Returns:
        Survival probability
    """
    integral = 0.0
    for i, T in enumerate(tenors):
        t_start = tenors[i-1] if i > 0 else 0
        t_end = min(T, t)
        
        if t_end > t_start:
            integral += hazards[i] * (t_end - t_start)
        
        if t <= T:
            break
    
    return np.exp(-integral)


def compute_marginal_pd(hazard_func: Callable, time_grid: np.ndarray) -> np.ndarray:
    """
    Compute marginal default probabilities for each time bucket.
    
    Args:
        hazard_func: Hazard rate function
        time_grid: Array of time points
    
    Returns:
        Array of marginal PDs
    """
    survival = np.zeros_like(time_grid)
    for i, t in enumerate(time_grid):
        # Integrate hazard rate to get cumulative hazard
        # For piecewise constant, this is sum of hazard * dt
        if i == 0:
            survival[i] = 1.0
        else:
            integral = 0.0
            for j in range(i):
                dt = time_grid[j+1] - time_grid[j]
                lam = hazard_func(time_grid[j])
                integral += lam * dt
            survival[i] = np.exp(-integral)
    
    # Marginal PD = S(t-1) - S(t)
    marginal_pd = np.diff(1 - survival, prepend=0)
    return marginal_pd


if __name__ == "__main__":
    # Test credit curve
    print("Testing credit curve builder...")
    
    from curve_builder import bootstrap_zero_curve
    
    # Build zero curve
    tenors_yc = np.array([1, 2, 5, 10, 30])
    yields = np.array([0.04, 0.042, 0.045, 0.047, 0.048])
    zero_curve = bootstrap_zero_curve(tenors_yc, yields)
    
    # OAS spreads (in bps)
    oas_spreads = {
        1: 50,
        2: 60,
        5: 80,
        10: 100,
        30: 120
    }
    
    # Bootstrap hazards
    hazard_func, tenors, hazards = bootstrap_hazard_rates(oas_spreads, zero_curve)
    
    print("\nHazard Rates:")
    for t, lam in zip(tenors, hazards):
        print(f"  {t:6.1f}Y: {lam*10000:7.2f} bps")
    
    # Test survival probabilities
    test_times = np.array([1, 3, 5, 10, 20])
    print("\nSurvival Probabilities:")
    for t in test_times:
        S = compute_survival(tenors, hazards, t)
        print(f"  {t:6.1f}Y: {S*100:6.2f}%")
    
    # Test marginal PDs
    time_grid = np.linspace(0, 10, 11)
    marginal_pds = compute_marginal_pd(hazard_func, time_grid)
    print("\nMarginal Default Probabilities:")
    for t, pd in zip(time_grid[1:6], marginal_pds[1:6]):
        print(f"  {time_grid[np.where(time_grid==t)[0][0]-1]:.1f}Y - {t:.1f}Y: {pd*100:6.3f}%")