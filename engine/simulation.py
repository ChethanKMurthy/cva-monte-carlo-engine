import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from numba import njit, prange
from typing import Tuple


def generate_sobol_normals(n_paths: int, n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Generate Sobol quasi-random normal variates.
    
    Args:
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps per path
        seed: Random seed for scrambling
    
    Returns:
        Array of shape (n_paths, n_steps) with standard normals
    """
    # Generate Sobol in [0,1]^n_steps
    sobol = Sobol(d=n_steps, scramble=True, seed=seed)
    sobol_uniform = sobol.random(n=n_paths)
    
    # Transform to standard normals via inverse CDF
    sobol_normals = norm.ppf(sobol_uniform)
    
    # Clip extreme values for numerical stability
    sobol_normals = np.clip(sobol_normals, -6, 6)
    
    return sobol_normals


def brownian_bridge_indices(n_steps: int) -> np.ndarray:
    """
    Generate Brownian bridge sampling order.
    Samples endpoint first, then recursively samples midpoints.
    
    Args:
        n_steps: Number of time steps
    
    Returns:
        Array of indices in Brownian bridge order
    """
    if n_steps <= 1:
        return np.array([0])
    
    indices = []
    
    def recurse(left, right):
        if left >= right:
            return
        mid = (left + right) // 2
        indices.append(mid)
        recurse(left, mid)
        recurse(mid + 1, right)
    
    # Start with endpoint
    indices.append(n_steps - 1)
    recurse(0, n_steps - 1)
    
    return np.array(indices)


def generate_sobol_normals_bb(n_paths: int, n_steps: int, seed: int = 42) -> np.ndarray:
    """
    Generate Sobol sequences with Brownian bridge ordering for variance reduction.
    
    Args:
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps per path
        seed: Random seed
    
    Returns:
        Array of shape (n_paths, n_steps) with standard normals in BB order
    """
    # Generate base Sobol normals
    sobol_normals = generate_sobol_normals(n_paths, n_steps, seed)
    
    # Reorder using Brownian bridge indices
    bb_indices = brownian_bridge_indices(n_steps)
    
    # Ensure we have the right number of indices
    if len(bb_indices) < n_steps:
        remaining = set(range(n_steps)) - set(bb_indices)
        bb_indices = np.concatenate([bb_indices, np.array(sorted(remaining))])
    
    z_bb = sobol_normals[:, bb_indices[:n_steps]]
    
    return z_bb


@njit(parallel=True, fastmath=True, cache=True)
def simulate_hull_white(
    n_paths: int,
    time_grid: np.ndarray,
    r0: float,
    a: float,
    sigma: float,
    theta_array: np.ndarray,
    z_matrix: np.ndarray
) -> np.ndarray:
    """
    Simulate Hull-White short rate paths using Euler-Maruyama discretization.
    Numba-accelerated with parallel processing.
    
    Args:
        n_paths: Number of paths
        time_grid: Time grid of shape (n_steps+1,)
        r0: Initial short rate
        a: Mean reversion speed
        sigma: Volatility
        theta_array: Time-dependent drift, shape (n_steps,)
        z_matrix: Standard normal matrix, shape (n_paths, n_steps)
    
    Returns:
        Rate paths of shape (n_paths, n_steps+1)
    """
    n_steps = len(time_grid) - 1
    rate_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    
    # Parallel across paths
    for i in prange(n_paths):
        rate_paths[i, 0] = r0
        
        for j in range(n_steps):
            dt = time_grid[j+1] - time_grid[j]
            
            # Euler-Maruyama step: dr = (θ - ar)dt + σdW
            drift = (theta_array[j] - a * rate_paths[i, j]) * dt
            diffusion = sigma * np.sqrt(dt) * z_matrix[i, j]
            
            rate_paths[i, j+1] = rate_paths[i, j] + drift + diffusion
            
            # Numerical stability: clip extreme rates
            rate_paths[i, j+1] = max(rate_paths[i, j+1], -0.10)
            rate_paths[i, j+1] = min(rate_paths[i, j+1], 0.30)
    
    return rate_paths


if __name__ == "__main__":
    import time
    
    print("Testing Monte Carlo simulation...")
    
    # Parameters
    n_paths = 10000
    n_steps = 100
    T = 10.0
    time_grid = np.linspace(0, T, n_steps + 1)
    
    # Hull-White parameters
    r0 = 0.05
    a = 0.1
    sigma = 0.01
    theta_array = np.full(n_steps, 0.05)  # Constant drift for simplicity
    
    # Test 1: Standard Sobol
    print("\n1. Standard Sobol sequence:")
    start = time.time()
    z_sobol = generate_sobol_normals(n_paths, n_steps, seed=42)
    paths_sobol = simulate_hull_white(n_paths, time_grid, r0, a, sigma, theta_array, z_sobol)
    time_sobol = time.time() - start
    print(f"   Time: {time_sobol*1000:.1f} ms")
    print(f"   Final rate mean: {paths_sobol[:, -1].mean():.4f}")
    print(f"   Final rate std: {paths_sobol[:, -1].std():.4f}")
    
    # Test 2: Sobol with Brownian Bridge
    print("\n2. Sobol with Brownian Bridge:")
    start = time.time()
    z_bb = generate_sobol_normals_bb(n_paths, n_steps, seed=42)
    paths_bb = simulate_hull_white(n_paths, time_grid, r0, a, sigma, theta_array, z_bb)
    time_bb = time.time() - start
    print(f"   Time: {time_bb*1000:.1f} ms")
    print(f"   Final rate mean: {paths_bb[:, -1].mean():.4f}")
    print(f"   Final rate std: {paths_bb[:, -1].std():.4f}")
    
    # Test 3: Pseudo-random (baseline)
    print("\n3. Pseudo-random (baseline):")
    start = time.time()
    z_random = np.random.randn(n_paths, n_steps)
    paths_random = simulate_hull_white(n_paths, time_grid, r0, a, sigma, theta_array, z_random)
    time_random = time.time() - start
    print(f"   Time: {time_random*1000:.1f} ms")
    print(f"   Final rate mean: {paths_random[:, -1].mean():.4f}")
    print(f"   Final rate std: {paths_random[:, -1].std():.4f}")
    
    print(f"\nSpeedup: {time_random/time_bb:.2f}x")