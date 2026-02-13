import numpy as np
from scipy.interpolate import interp1d
from typing import Callable


def bootstrap_zero_curve(tenor_years: np.ndarray, yields: np.ndarray) -> Callable:
    """
    Construct continuously compounded zero rate curve.
    
    Args:
        tenor_years: Array of tenor points in years
        yields: Array of yields (as decimals, e.g., 0.05 for 5%)
    
    Returns:
        Function r(t) that returns zero rate for time t
    """
    # Ensure inputs are numpy arrays
    tenor_years = np.asarray(tenor_years, dtype=float)
    yields = np.asarray(yields, dtype=float)
    
    # Sort by tenor
    sort_idx = np.argsort(tenor_years)
    tenor_years = tenor_years[sort_idx]
    yields = yields[sort_idx]
    
    # For short tenors, add a point at origin
    if tenor_years[0] > 0.1:
        tenor_years = np.insert(tenor_years, 0, 0.01)
        yields = np.insert(yields, 0, yields[0])
    
    # Log-linear interpolation for zero rates
    log_interp = interp1d(
        tenor_years,
        np.log(1 + yields),
        kind='linear',
        fill_value='extrapolate',
        bounds_error=False
    )
    
    def zero_rate(t):
        """Return zero rate for time t (or array of times)."""
        t = np.asarray(t)
        return np.exp(log_interp(t)) - 1
    
    return zero_rate


def compute_discount_factors(zero_curve: Callable, time_grid: np.ndarray) -> np.ndarray:
    """
    Compute discount factors: DF(t) = exp(-r(t) * t)
    
    Args:
        zero_curve: Zero rate function
        time_grid: Array of time points
    
    Returns:
        Array of discount factors
    """
    rates = zero_curve(time_grid)
    return np.exp(-rates * time_grid)


def compute_forward_curve(zero_curve: Callable, time_grid: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous forward rates: f(0,t) = r(t) + t * dr/dt
    Using numerical differentiation.
    
    Args:
        zero_curve: Zero rate function
        time_grid: Array of time points
    
    Returns:
        Array of forward rates
    """
    dt = 0.001  # Small increment for numerical derivative
    forwards = np.zeros_like(time_grid)
    
    for i, t in enumerate(time_grid):
        if t < dt:
            forwards[i] = zero_curve(dt)
        else:
            # Central difference approximation
            r_plus = zero_curve(t + dt)
            r_minus = zero_curve(t - dt)
            dr_dt = (r_plus - r_minus) / (2 * dt)
            forwards[i] = zero_curve(t) + t * dr_dt
    
    return forwards


if __name__ == "__main__":
    # Test the curve builder
    print("Testing curve builder...")
    
    # Sample yield curve
    tenors = np.array([1, 2, 5, 10, 30])
    yields = np.array([0.04, 0.042, 0.045, 0.047, 0.048])
    
    # Build zero curve
    zero_curve = bootstrap_zero_curve(tenors, yields)
    
    # Test interpolation
    test_times = np.linspace(0.5, 30, 20)
    zero_rates = zero_curve(test_times)
    
    print("\nZero Rates:")
    for t, r in zip(test_times, zero_rates):
        print(f"  {t:6.2f}Y: {r*100:6.3f}%")
    
    # Test discount factors
    dfs = compute_discount_factors(zero_curve, test_times)
    print("\nDiscount Factors:")
    for t, df in zip(test_times[:5], dfs[:5]):
        print(f"  {t:6.2f}Y: {df:8.6f}")
    
    # Test forward curve
    forwards = compute_forward_curve(zero_curve, test_times)
    print("\nForward Rates:")
    for t, f in zip(test_times[:5], forwards[:5]):
        print(f"  {t:6.2f}Y: {f*100:6.3f}%")