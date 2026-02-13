from pydantic import BaseModel, Field
from typing import List
from datetime import date


class SwapInput(BaseModel):
    """Input schema for a single interest rate swap."""
    notional: float = Field(..., gt=0, description="Notional amount")
    fixed_rate: float = Field(..., ge=0, le=0.3, description="Fixed rate (decimal)")
    start_date: date = Field(..., description="Swap start date")
    maturity: date = Field(..., description="Swap maturity date")
    pay_fixed: bool = Field(..., description="True if paying fixed, False if receiving")
    
    class Config:
        json_schema_extra = {
            "example": {
                "notional": 1000000.0,
                "fixed_rate": 0.045,
                "start_date": "2024-01-01",
                "maturity": "2029-01-01",
                "pay_fixed": True
            }
        }


class CVARequest(BaseModel):
    """Request schema for CVA calculation."""
    portfolio: List[SwapInput] = Field(..., min_length=1, max_length=50)
    counterparty_rating: str = Field('IG', description="Credit rating: IG or AAA")
    n_paths: int = Field(10000, ge=1000, le=200000, description="Number of MC paths")
    n_steps: int = Field(100, ge=20, le=500, description="Number of time steps")
    use_brownian_bridge: bool = Field(True, description="Use Brownian Bridge variance reduction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "portfolio": [
                    {
                        "notional": 1000000,
                        "fixed_rate": 0.045,
                        "start_date": "2024-01-01",
                        "maturity": "2029-01-01",
                        "pay_fixed": True
                    }
                ],
                "counterparty_rating": "IG",
                "n_paths": 10000,
                "n_steps": 100,
                "use_brownian_bridge": True
            }
        }


class ExposurePoint(BaseModel):
    """Single point in exposure profile."""
    time: float = Field(..., description="Time in years")
    expected_exposure: float = Field(..., description="Expected Exposure")
    pfe_95: float = Field(..., description="95th percentile PFE")


class CVAResponse(BaseModel):
    """Response schema for CVA calculation."""
    cva: float = Field(..., description="Credit Valuation Adjustment in currency")
    cva_bps: float = Field(..., description="CVA in basis points of total notional")
    exposure_profile: List[ExposurePoint]
    computation_time_ms: float = Field(..., description="Computation time in milliseconds")
    n_paths_used: int
    model_parameters: dict = Field(..., description="Calibrated model parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "cva": 15432.50,
                "cva_bps": 15.43,
                "exposure_profile": [
                    {"time": 0.0, "expected_exposure": 0.0, "pfe_95": 0.0},
                    {"time": 1.0, "expected_exposure": 45000.0, "pfe_95": 120000.0}
                ],
                "computation_time_ms": 1850.5,
                "n_paths_used": 10000,
                "model_parameters": {
                    "mean_reversion": 0.1,
                    "volatility": 0.01,
                    "recovery_rate": 0.40
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    version: str = "1.0.0"