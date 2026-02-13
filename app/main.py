from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import numpy as np
from datetime import date
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import CVARequest, CVAResponse, ExposurePoint, HealthResponse
from engine.risk_engine import CVAEngine, Swap
from engine.curve_builder import bootstrap_zero_curve, compute_discount_factors
from engine.credit_curve import bootstrap_hazard_rates
from engine.calibration import calibrate_hull_white, compute_theta_array
from engine.simulation import generate_sobol_normals_bb, generate_sobol_normals, simulate_hull_white
from data.fetcher import FREDDataFetcher

app = FastAPI(
    title="CVA Monte Carlo Engine",
    description="Production-grade Credit Valuation Adjustment calculator for interest rate swaps",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for calibrated models
_model_cache = {}
_cache_date = None


def get_calibrated_model():
    """Get or build calibrated model (cached daily)."""
    global _model_cache, _cache_date
    
    today = date.today()
    
    if _cache_date != today or not _model_cache:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Calibrating model...")
        
        try:
            fetcher = FREDDataFetcher()
            
            # Build yield curve
            yield_curve_df = fetcher.build_yield_curve()
            print(f"Yield curve:\n{yield_curve_df}")
            
            zero_curve = bootstrap_zero_curve(
                yield_curve_df['tenor'].values,
                yield_curve_df['rate'].values
            )
            
            # Calibrate Hull-White
            sofr_history = fetcher.fetch_sofr_history(lookback_days=504)  # 2 years
            sofr_values = sofr_history.values.flatten()
            
            a, sigma = calibrate_hull_white(sofr_values, zero_curve)
            
            # Build credit curves for different ratings
            credit_curves = {}
            for rating in ['IG', 'AAA']:
                oas = fetcher.fetch_credit_spread(rating)
                
                # Create term structure of OAS (simplified - flat with slight upward slope)
                oas_spreads = {
                    1: oas * 10000 * 0.8,    # bps
                    2: oas * 10000 * 0.9,
                    5: oas * 10000 * 1.0,
                    10: oas * 10000 * 1.1,
                    30: oas * 10000 * 1.2
                }
                
                hazard_curve, tenors, hazards = bootstrap_hazard_rates(
                    oas_spreads,
                    zero_curve
                )
                
                credit_curves[rating] = hazard_curve
            
            _model_cache = {
                'zero_curve': zero_curve,
                'credit_curves': credit_curves,
                'hw_params': {'a': float(a), 'sigma': float(sigma)},
                'r0': float(zero_curve(0.01))
            }
            _cache_date = today
            
            print(f"Model calibrated successfully: a={a:.4f}, σ={sigma:.4f}")
            
        except Exception as e:
            print(f"Error during calibration: {e}")
            print("Using fallback parameters...")
            
            # Fallback model
            tenors = np.array([1, 2, 5, 10, 30])
            yields = np.array([0.04, 0.042, 0.045, 0.047, 0.048])
            zero_curve = bootstrap_zero_curve(tenors, yields)
            
            oas_spreads_ig = {1: 50, 5: 80, 10: 100, 30: 120}
            oas_spreads_aaa = {1: 30, 5: 50, 10: 60, 30: 70}
            
            hazard_ig, _, _ = bootstrap_hazard_rates(oas_spreads_ig, zero_curve)
            hazard_aaa, _, _ = bootstrap_hazard_rates(oas_spreads_aaa, zero_curve)
            
            _model_cache = {
                'zero_curve': zero_curve,
                'credit_curves': {'IG': hazard_ig, 'AAA': hazard_aaa},
                'hw_params': {'a': 0.1, 'sigma': 0.01},
                'r0': 0.04
            }
            _cache_date = today
    
    return _model_cache


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "CVA Monte Carlo Engine API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0"
    )


@app.post("/calculate_cva", response_model=CVAResponse)
async def calculate_cva(request: CVARequest):
    """
    Calculate CVA for a portfolio of interest rate swaps.
    """
    start_time = time.time()
    
    try:
        # Get calibrated model
        model = get_calibrated_model()
        
        # Convert portfolio to Swap objects
        portfolio = []
        total_notional = 0.0
        max_maturity = 0.0
        
        for swap_input in request.portfolio:
            # Convert dates to years from today
            days_to_start = (swap_input.start_date - date.today()).days
            days_to_maturity = (swap_input.maturity - date.today()).days
            
            start_time_years = max(0.0, days_to_start / 365.0)
            maturity_years = max(0.5, days_to_maturity / 365.0)
            
            if maturity_years > max_maturity:
                max_maturity = maturity_years
            
            total_notional += abs(swap_input.notional)
            
            portfolio.append(Swap(
                notional=swap_input.notional,
                fixed_rate=swap_input.fixed_rate,
                start_time=start_time_years,
                maturity=maturity_years,
                pay_fixed=swap_input.pay_fixed
            ))
        
        # Generate time grid
        time_grid = np.linspace(0, max_maturity, request.n_steps + 1)
        
        # Compute theta array
        theta_array = compute_theta_array(
            time_grid,
            model['zero_curve'],
            model['hw_params']['a'],
            model['hw_params']['sigma']
        )
        
        # Generate random numbers
        if request.use_brownian_bridge:
            z_matrix = generate_sobol_normals_bb(request.n_paths, request.n_steps)
        else:
            z_matrix = generate_sobol_normals(request.n_paths, request.n_steps)
        
        # Simulate rate paths
        rate_paths = simulate_hull_white(
            request.n_paths,
            time_grid,
            model['r0'],
            model['hw_params']['a'],
            model['hw_params']['sigma'],
            theta_array,
            z_matrix
        )
        
        # Get credit curve for rating
        rating = request.counterparty_rating if request.counterparty_rating in model['credit_curves'] else 'IG'
        hazard_curve = model['credit_curves'][rating]
        
        # Initialize CVA engine
        engine = CVAEngine(
            model['zero_curve'],
            hazard_curve,
            recovery_rate=0.40
        )
        
        # Calculate CVA
        results = engine.calculate_cva(
            portfolio,
            rate_paths,
            time_grid,
            model['hw_params']['a'],
            model['hw_params']['sigma']
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Format response
        exposure_profile = [
            ExposurePoint(
                time=float(t),
                expected_exposure=float(ee),
                pfe_95=float(pfe)
            )
            for t, ee, pfe in zip(
                time_grid,
                results['expected_exposure'],
                results['pfe_95']
            )
        ]
        
        cva_bps = (results['cva'] / total_notional * 10000) if total_notional > 0 else 0
        
        return CVAResponse(
            cva=float(results['cva']),
            cva_bps=float(cva_bps),
            exposure_profile=exposure_profile,
            computation_time_ms=elapsed_ms,
            n_paths_used=request.n_paths,
            model_parameters={
                'mean_reversion': model['hw_params']['a'],
                'volatility': model['hw_params']['sigma'],
                'recovery_rate': 0.40,
                'initial_rate': model['r0']
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)