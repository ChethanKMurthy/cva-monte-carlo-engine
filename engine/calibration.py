import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple
import warnings
warnings.filterwarnings('ignore')


def calibrate_hull_white(
    historical_rates: np.ndarray,
    zero_curve: Callable,
    initial_guess: Tuple[float, float] = (0.1, 0.01)
) -> Tuple[float, float]:
    """
    Calibrate Hull-White (a, sigma) parameters to historical rate volatility.
    
    Args:
        historical_rates: Historical short rate time series
        zero_curve: Zero rate curve function
        initial_guess: Initial (a, sigma) guess
    
    Returns:
        Tuple of (a, sigma)
    """
    # Remove NaN values
    historical_rates = historical_rates[~np.isnan(historical_rates)]
    
    if len(historical_rates) < 100:
        print("Warning: Limited historical data, using default parameters")
        return 0.1, 0.01
    
    # Compute historical volatility (annualized)
    # Shift rates to avoid log of negative/zero
    shifted_rates = historical_rates + 0.02
    rate_returns = np.diff(np.log(shifted_rates))
    hist_vol = np.std(rate_returns) * np.sqrt(252)  # Annualize daily vol
    
    # Compute autocorrelation for mean reversion
    acf_1day = np.corrcoef(historical_rates[:-1], historical_rates[1:])[0, 1]
    
    print(f"Historical volatility: {hist_vol*100:.2f}%")
    print(f"1-day autocorrelation: {acf_1day:.4f}")
    
    def objective(params):
        a, sigma = params
        
        # Constraints
        if a <= 0 or a > 2.0 or sigma <= 0 or sigma > 0.1:
            return 1e10
        
        # Model-implied volatility: Var(r(T)) = σ²/(2a) * (1 - e^(-2aT))
        T = 1.0  # 1-year horizon
        model_var = (sigma**2 / (2 * a)) * (1 - np.exp(-2 * a * T))
        model_vol = np.sqrt(model_var)
        
        # Model-implied autocorrelation: ρ(Δt) = e^(-aΔt)
        dt = 1/252  # Daily
        model_acf = np.exp(-a * dt)
        
        # Combined error
        vol_error = (model_vol - hist_vol)**2
        acf_error = (model_acf - acf_1day)**2
        
        return vol_error + 0.5 * acf_error
    
    # Optimize
    result = minimize(
        objective,
        x0=initial_guess,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6}
    )
    
    a_opt, sigma_opt = result.x
    
    # Sanity checks
    if a_opt < 0.01 or a_opt > 1.5:
        print(f"Warning: Unusual mean reversion a={a_opt:.4f}, using default")
        a_opt = 0.1
    
    if sigma_opt < 0.001 or sigma_opt > 0.05:
        print(f"Warning: Unusual volatility σ={sigma_opt:.4f}, using default")
        sigma_opt = 0.01
    
    print(f"\nCalibrated parameters:")
    print(f"  Mean reversion (a): {a_opt:.4f}")
    print(f"  Volatility (σ): {sigma_opt:.4f}")
    
    return a_opt, sigma_opt


def compute_theta_array(
    time_grid: np.ndarray,
    zero_curve: Callable,
    a: float,
    sigma: float
) -> np.ndarray:
    """
    Compute time-dependent drift θ(t) for Hull-White model.
    
    θ(t) = ∂f(0,t)/∂t + a*f(0,t) + σ²/(2a) * (1 - e^(-2at))
    
    Args:
        time_grid: Time grid
        zero_curve: Zero rate curve function
        a: Mean reversion
        sigma: Volatility
    
    Returns:
        Array of θ values for each time step
    """
    n_steps = len(time_grid) - 1
    theta = np.zeros(n_steps)
    
    dt = 0.001  # For numerical derivative
    
    for i in range(n_steps):
        t = time_grid[i]
        
        # Compute instantaneous forward rate f(0,t)
        if t < dt:
            f_t = zero_curve(dt)
        else:
            # f(0,t) = -∂log(P(0,t))/∂t = r(t) + t * dr/dt
            r_t = zero_curve(t)
            r_plus = zero_curve(t + dt)
            r_minus = zero_curve(max(0, t - dt))
            dr_dt = (r_plus - r_minus) / (2 * dt)
            f_t = r_t + t * dr_dt
        
        # Compute ∂f(0,t)/∂t numerically
        if t < dt:
            f_plus = zero_curve(t + dt)
            df_dt = (f_plus - f_t) / dt
        else:
            t_minus = max(0, t - dt)
            # Forward rates at t-dt and t+dt
            r_minus = zero_curve(t_minus)
            r_plus = zero_curve(t + dt)
            r_minus_plus = zero_curve(t_minus + dt)
            r_plus_minus = zero_curve(t + dt - dt)
            
            f_minus = r_minus + t_minus * (r_minus_plus - r_minus) / dt
            f_plus = r_plus + (t + dt) * (r_plus - r_plus_minus) / dt
            
            df_dt = (f_plus - f_minus) / (2 * dt)
        
        # θ(t) = ∂f/∂t + a*f + σ²/(2a)*(1-e^(-2at))
        theta[i] = df_dt + a * f_t + (sigma**2 / (2*a)) * (1 - np.exp(-2*a*t))
    
    return theta


if __name__ == "__main__":
    # Test calibration
    print("Testing Hull-White calibration...")
    
    from curve_builder import bootstrap_zero_curve
    
    # Build zero curve
    tenors = np.array([1, 2, 5, 10, 30])
    yields = np.array([0.04, 0.042, 0.045, 0.047, 0.048])
    zero_curve = bootstrap_zero_curve(tenors, yields)
    
    # Generate synthetic historical rates
    np.random.seed(42)
    n_days = 252 * 2  # 2 years
    historical_rates = 0.05 + 0.01 * np.random.randn(n_days).cumsum() / np.sqrt(n_days)
    historical_rates = np.clip(historical_rates, 0.01, 0.10)
    
    # Calibrate
    a, sigma = calibrate_hull_white(historical_rates, zero_curve)
    
    # Test theta computation
    print("\nTesting theta computation...")
    time_grid = np.linspace(0, 10, 101)
    theta_array = compute_theta_array(time_grid, zero_curve, a, sigma)
    
    print(f"\nTheta values (first 5 time steps):")
    for i in range(5):
        print(f"  t={time_grid[i]:.2f}: θ={theta_array[i]:.6f}")