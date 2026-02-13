import numpy as np
from numba import njit, prange
from typing import List, Dict, Callable
from dataclasses import dataclass


@dataclass
class Swap:
    """Interest rate swap contract."""
    notional: float
    fixed_rate: float
    start_time: float  # Years from today
    maturity: float    # Years from today
    pay_fixed: bool    # True for payer swap, False for receiver


@njit(fastmath=True)
def compute_bond_price_hw(r_t: float, t: float, T: float, a: float, sigma: float, 
                          P_0_T: float, P_0_t: float) -> float:
    """
    Compute zero-coupon bond price P(t,T) in Hull-White model.
    
    P(t,T) = A(t,T) * exp(-B(t,T) * r(t))
    
    where:
    B(t,T) = (1 - exp(-a(T-t))) / a
    A(t,T) = P(0,T)/P(0,t) * exp(B(t,T)*f(0,t) - σ²/(4a)*B²(1-exp(-2at)))
    
    Simplified using initial discount factors.
    """
    tau = T - t
    if tau <= 0:
        return 1.0
    
    B = (1 - np.exp(-a * tau)) / a
    
    # Simplified A term using initial curve
    log_A = np.log(P_0_T / P_0_t) - sigma**2 / (4*a) * B**2 * (1 - np.exp(-2*a*t))
    
    return np.exp(log_A - B * r_t)


@njit(parallel=True, fastmath=True)
def compute_swap_values(
    rate_paths: np.ndarray,      # (n_paths, n_steps+1)
    time_grid: np.ndarray,        # (n_steps+1,)
    swap_notionals: np.ndarray,   # (n_swaps,)
    swap_fixed_rates: np.ndarray, # (n_swaps,)
    swap_start_idx: np.ndarray,   # (n_swaps,) - integer indices
    swap_end_idx: np.ndarray,     # (n_swaps,) - integer indices
    swap_directions: np.ndarray,  # (n_swaps,) - +1 for payer, -1 for receiver
    a: float,
    sigma: float,
    P_0_grid: np.ndarray          # (n_steps+1,) - initial discount factors
) -> np.ndarray:
    """
    Compute portfolio value for all paths and all time steps.
    
    Returns: portfolio_values of shape (n_paths, n_steps+1)
    """
    n_paths, n_steps_plus1 = rate_paths.shape
    n_swaps = len(swap_notionals)
    
    portfolio_values = np.zeros((n_paths, n_steps_plus1), dtype=np.float64)
    
    # For each path
    for path_idx in prange(n_paths):
        # For each valuation time
        for t_idx in range(n_steps_plus1):
            t = time_grid[t_idx]
            r_t = rate_paths[path_idx, t_idx]
            
            # For each swap
            for swap_idx in range(n_swaps):
                start_idx = swap_start_idx[swap_idx]
                end_idx = swap_end_idx[swap_idx]
                
                # Skip if swap hasn't started or has matured
                if t_idx < start_idx or t_idx >= end_idx:
                    continue
                
                notional = swap_notionals[swap_idx]
                fixed_rate = swap_fixed_rates[swap_idx]
                direction = swap_directions[swap_idx]
                
                # Compute swap value
                swap_value = 0.0
                
                # Sum over remaining cash flows
                for cf_idx in range(t_idx + 1, end_idx + 1):
                    if cf_idx >= n_steps_plus1:
                        break
                    
                    T_cf = time_grid[cf_idx]
                    dt = time_grid[cf_idx] - time_grid[cf_idx - 1]
                    
                    # Compute P(t, T_cf) using Hull-White formula
                    P_t_T = compute_bond_price_hw(
                        r_t, t, T_cf, a, sigma,
                        P_0_grid[cf_idx], P_0_grid[t_idx]
                    )
                    
                    # Forward rate approximation
                    if cf_idx > t_idx + 1:
                        T_prev = time_grid[cf_idx - 1]
                        P_t_Tprev = compute_bond_price_hw(
                            r_t, t, T_prev, a, sigma,
                            P_0_grid[cf_idx - 1], P_0_grid[t_idx]
                        )
                        forward_rate = (P_t_Tprev / P_t_T - 1.0) / dt
                    else:
                        # First period: use short rate as approximation
                        forward_rate = r_t
                    
                    # Cash flow PV
                    cf_pv = (forward_rate - fixed_rate) * dt * P_t_T
                    swap_value += cf_pv
                
                # Add to portfolio
                portfolio_values[path_idx, t_idx] += direction * notional * swap_value
    
    return portfolio_values


class CVAEngine:
    """Credit Valuation Adjustment calculation engine."""
    
    def __init__(
        self,
        zero_curve: Callable,
        hazard_curve: Callable,
        recovery_rate: float = 0.40
    ):
        self.zero_curve = zero_curve
        self.hazard_curve = hazard_curve
        self.recovery_rate = recovery_rate
    
    def calculate_cva(
        self,
        portfolio: List[Swap],
        rate_paths: np.ndarray,
        time_grid: np.ndarray,
        a: float,
        sigma: float
    ) -> Dict:
        """
        Calculate CVA with full exposure profile.
        
        Args:
            portfolio: List of Swap objects
            rate_paths: Simulated rate paths (n_paths, n_steps+1)
            time_grid: Time grid
            a: Hull-White mean reversion
            sigma: Hull-White volatility
        
        Returns:
            Dictionary with CVA and exposure metrics
        """
        n_paths, n_steps_plus1 = rate_paths.shape
        
        # Prepare swap parameters for Numba
        n_swaps = len(portfolio)
        swap_notionals = np.array([s.notional for s in portfolio])
        swap_fixed_rates = np.array([s.fixed_rate for s in portfolio])
        swap_directions = np.array([1.0 if s.pay_fixed else -1.0 for s in portfolio])
        
        # Convert times to indices
        swap_start_idx = np.zeros(n_swaps, dtype=np.int64)
        swap_end_idx = np.zeros(n_swaps, dtype=np.int64)
        
        for i, swap in enumerate(portfolio):
            # Find closest time indices
            swap_start_idx[i] = np.argmin(np.abs(time_grid - swap.start_time))
            swap_end_idx[i] = np.argmin(np.abs(time_grid - swap.maturity))
        
        # Compute initial discount factors
        P_0_grid = np.array([np.exp(-self.zero_curve(t) * t) for t in time_grid])
        
        # Compute portfolio values
        print("Computing portfolio values across all paths...")
        portfolio_values = compute_swap_values(
            rate_paths,
            time_grid,
            swap_notionals,
            swap_fixed_rates,
            swap_start_idx,
            swap_end_idx,
            swap_directions,
            a,
            sigma,
            P_0_grid
        )
        
        # Expected Exposure
        EE = np.mean(np.maximum(portfolio_values, 0.0), axis=0)
        
        # Potential Future Exposure (95th percentile)
        PFE_95 = np.percentile(np.maximum(portfolio_values, 0.0), 95, axis=0)
        
        # Compute marginal default probabilities
        marginal_PD = self._compute_marginal_pd(time_grid)
        
        # CVA calculation
        cva = (1 - self.recovery_rate) * np.sum(P_0_grid * EE * marginal_PD)
        
        return {
            'cva': cva,
            'expected_exposure': EE,
            'pfe_95': PFE_95,
            'time_grid': time_grid,
            'portfolio_values': portfolio_values  # For debugging
        }
    
    def _compute_marginal_pd(self, time_grid: np.ndarray) -> np.ndarray:
        """Compute marginal default probabilities for each time bucket."""
        survival = np.zeros_like(time_grid)
        
        for i, t in enumerate(time_grid):
            if i == 0:
                survival[i] = 1.0
            else:
                # Integrate hazard rate
                integral = 0.0
                for j in range(i):
                    dt = time_grid[j+1] - time_grid[j]
                    lam = self.hazard_curve(time_grid[j])
                    integral += lam * dt
                survival[i] = np.exp(-integral)
        
        # Marginal PD = S(t-1) - S(t)
        marginal_pd = -np.diff(survival, prepend=1.0)
        
        return marginal_pd


if __name__ == "__main__":
    # Test CVA engine
    print("Testing CVA Risk Engine...")
    
    from curve_builder import bootstrap_zero_curve
    from credit_curve import bootstrap_hazard_rates
    from simulation import generate_sobol_normals_bb, simulate_hull_white
    from calibration import compute_theta_array
    
    # Setup curves
    tenors = np.array([1, 2, 5, 10, 30])
    yields = np.array([0.04, 0.042, 0.045, 0.047, 0.048])
    zero_curve = bootstrap_zero_curve(tenors, yields)
    
    oas_spreads = {1: 50, 5: 80, 10: 100}
    hazard_curve, _, _ = bootstrap_hazard_rates(oas_spreads, zero_curve)
    
    # Create test portfolio
    portfolio = [
        Swap(notional=1_000_000, fixed_rate=0.045, start_time=0.0, 
             maturity=5.0, pay_fixed=True),
        Swap(notional=500_000, fixed_rate=0.047, start_time=0.0,
             maturity=10.0, pay_fixed=False),
    ]
    
    # Simulate paths
    n_paths = 5000
    n_steps = 50
    T = 10.0
    time_grid = np.linspace(0, T, n_steps + 1)
    
    a, sigma = 0.1, 0.01
    r0 = zero_curve(0.01)
    theta_array = compute_theta_array(time_grid, zero_curve, a, sigma)
    
    z_matrix = generate_sobol_normals_bb(n_paths, n_steps, seed=42)
    rate_paths = simulate_hull_white(n_paths, time_grid, r0, a, sigma, 
                                     theta_array, z_matrix)
    
    # Calculate CVA
    engine = CVAEngine(zero_curve, hazard_curve)
    results = engine.calculate_cva(portfolio, rate_paths, time_grid, a, sigma)
    
    print(f"\nResults:")
    print(f"  CVA: ${results['cva']:,.2f}")
    print(f"  Max EE: ${results['expected_exposure'].max():,.2f}")
    print(f"  Max PFE95: ${results['pfe_95'].max():,.2f}")
    
    # Show exposure profile
    print(f"\nExposure Profile (first 10 points):")
    for i in range(min(10, len(time_grid))):
        print(f"  {time_grid[i]:5.2f}Y: EE=${results['expected_exposure'][i]:>12,.0f}  "
              f"PFE95=${results['pfe_95'][i]:>12,.0f}")