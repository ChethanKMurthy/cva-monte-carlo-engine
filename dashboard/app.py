import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import json

st.set_page_config(
    page_title="CVA Monte Carlo Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🏦 CVA Monte Carlo Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Grade XVA Calculator for Interest Rate Swaps</div>', unsafe_allow_html=True)

# API Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# Check API health
try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    if health_response.status_code == 200:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Error")
except:
    st.sidebar.error("❌ Cannot connect to API")
    st.error(f"Cannot connect to API at {API_URL}. Make sure the FastAPI server is running.")
    st.code("python app/main.py", language="bash")
    st.stop()

# Sidebar Configuration
st.sidebar.header("⚙️ Portfolio Configuration")

# Number of swaps
n_swaps = st.sidebar.number_input(
    "Number of Swaps",
    min_value=1,
    max_value=20,
    value=2,
    help="Number of swaps in the portfolio"
)

# Portfolio input
portfolio = []
for i in range(n_swaps):
    with st.sidebar.expander(f"💼 Swap {i+1}", expanded=(i < 2)):
        col1, col2 = st.columns(2)
        
        with col1:
            notional = st.number_input(
                "Notional",
                min_value=100000.0,
                max_value=100000000.0,
                value=1000000.0,
                step=100000.0,
                key=f"notional_{i}",
                format="%.0f"
            )
            
            start_date = st.date_input(
                "Start Date",
                value=date.today(),
                key=f"start_{i}"
            )
        
        with col2:
            fixed_rate = st.number_input(
                "Fixed Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=4.5,
                step=0.1,
                key=f"rate_{i}"
            ) / 100
            
            maturity = st.date_input(
                "Maturity",
                value=date.today() + timedelta(days=365*5),
                min_value=date.today() + timedelta(days=30),
                key=f"maturity_{i}"
            )
        
        direction = st.selectbox(
            "Direction",
            ["Pay Fixed", "Receive Fixed"],
            key=f"direction_{i}"
        )
        
        portfolio.append({
            "notional": notional,
            "fixed_rate": fixed_rate,
            "start_date": start_date.isoformat(),
            "maturity": maturity.isoformat(),
            "pay_fixed": (direction == "Pay Fixed")
        })

st.sidebar.markdown("---")
st.sidebar.header("🔬 Simulation Parameters")

col1, col2 = st.sidebar.columns(2)
with col1:
    n_paths = st.select_slider(
        "MC Paths",
        options=[1000, 5000, 10000, 25000, 50000, 100000],
        value=10000,
        help="More paths = higher accuracy but slower"
    )

with col2:
    n_steps = st.select_slider(
        "Time Steps",
        options=[20, 50, 100, 150, 200],
        value=100,
        help="More steps = finer time grid"
    )

use_bb = st.sidebar.checkbox(
    "Use Brownian Bridge",
    value=True,
    help="Variance reduction technique (10-50x speedup)"
)

rating = st.sidebar.selectbox(
    "Counterparty Rating",
    ["IG", "AAA"],
    help="Credit quality of counterparty"
)

st.sidebar.markdown("---")

# Calculate button
if st.sidebar.button("🚀 Calculate CVA", type="primary", use_container_width=True):
    with st.spinner("Running Monte Carlo simulation..."):
        # Prepare payload
        payload = {
            "portfolio": portfolio,
            "counterparty_rating": rating,
            "n_paths": n_paths,
            "n_steps": n_steps,
            "use_brownian_bridge": use_bb
        }
        
        try:
            # Make API call
            response = requests.post(
                f"{API_URL}/calculate_cva",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Store in session state
                st.session_state['result'] = result
                st.session_state['portfolio'] = portfolio
                
                st.sidebar.success("✅ Calculation complete!")
            else:
                st.error(f"API Error: {response.status_code}")
                st.code(response.text)
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Calculation timeout (>5 min). Try reducing paths or steps.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display results
if 'result' in st.session_state:
    result = st.session_state['result']
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 CVA (USD)",
            f"${result['cva']:,.2f}",
            help="Credit Valuation Adjustment in dollars"
        )
    
    with col2:
        st.metric(
            "📊 CVA (bps)",
            f"{result['cva_bps']:.2f}",
            help="CVA as basis points of total notional"
        )
    
    with col3:
        st.metric(
            "⏱️ Compute Time",
            f"{result['computation_time_ms']:.0f} ms",
            help="Total calculation time"
        )
    
    with col4:
        st.metric(
            "🎯 Paths Used",
            f"{result['n_paths_used']:,}",
            help="Number of Monte Carlo paths"
        )
    
    st.markdown("---")
    
    # Exposure Profile Chart
    st.subheader("📈 Expected Exposure Profile")
    
    df_exposure = pd.DataFrame(result['exposure_profile'])
    
    fig = go.Figure()
    
    # Expected Exposure
    fig.add_trace(go.Scatter(
        x=df_exposure['time'],
        y=df_exposure['expected_exposure'],
        mode='lines',
        name='Expected Exposure (EE)',
        line=dict(color='#1f77b4', width=3),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    # 95% PFE
    fig.add_trace(go.Scatter(
        x=df_exposure['time'],
        y=df_exposure['pfe_95'],
        mode='lines',
        name='95% PFE',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Time (years)",
        yaxis_title="Exposure (USD)",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Exposure Statistics")
        
        max_ee = df_exposure['expected_exposure'].max()
        max_pfe = df_exposure['pfe_95'].max()
        time_max_ee = df_exposure.loc[df_exposure['expected_exposure'].idxmax(), 'time']
        
        st.write(f"**Maximum EE:** ${max_ee:,.0f} at {time_max_ee:.1f} years")
        st.write(f"**Maximum PFE95:** ${max_pfe:,.0f}")
        st.write(f"**PFE/EE Ratio:** {max_pfe/max_ee:.2f}x" if max_ee > 0 else "N/A")
    
    with col2:
        st.subheader("🔧 Model Parameters")
        
        params = result['model_parameters']
        st.write(f"**Mean Reversion (a):** {params['mean_reversion']:.4f}")
        st.write(f"**Volatility (σ):** {params['volatility']:.4f}")
        st.write(f"**Recovery Rate:** {params['recovery_rate']:.0%}")
        st.write(f"**Initial Rate:** {params.get('initial_rate', 0):.2%}")
    
    # Download results
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Prepare CSV export
        csv_data = df_exposure.to_csv(index=False)
        st.download_button(
            label="📥 Download Exposure Profile",
            data=csv_data,
            file_name="cva_exposure_profile.csv",
            mime="text/csv"
        )
        
        # JSON export
        json_data = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download Full Results (JSON)",
            data=json_data,
            file_name="cva_results.json",
            mime="application/json"
        )

else:
    # Show instructions when no results
    st.info("👈 Configure your portfolio in the sidebar and click **Calculate CVA** to begin")
    
    # Show example portfolio
    with st.expander("📚 Example: 2-Swap Portfolio"):
        st.markdown("""
        **Swap 1:**
        - Notional: $1,000,000
        - Fixed Rate: 4.5%
        - Maturity: 5 years
        - Direction: Pay Fixed
        
        **Swap 2:**
        - Notional: $500,000
        - Fixed Rate: 4.7%
        - Maturity: 10 years
        - Direction: Receive Fixed
        
        **Simulation:**
        - MC Paths: 10,000
        - Time Steps: 100
        - Brownian Bridge: Enabled
        
        **Expected CVA:** ~$15,000-25,000 (depends on market conditions)
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>CVA Monte Carlo Engine v1.0 | Built with FastAPI, Streamlit, Numba</p>
        <p>⚡ Powered by Sobol sequences + Brownian Bridge variance reduction</p>
    </div>
""", unsafe_allow_html=True)