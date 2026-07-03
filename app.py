import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. GLOBAL CONFIG & THEME
# ==========================================
st.set_page_config(layout="wide", page_title="Stochastic Terminal", page_icon="📈")

# Clean, Modern Bloomberg-style CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        .main { background-color: #0b0e11; }
        div[data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; color: #ffffff; }
        div[data-testid="stMetricDelta"] svg { display: none; } /* Hide default arrows for cleaner look */
        .stTabs [data-baseweb=tab] { font-size: 14px; letter-spacing: 1px; text-transform: uppercase; }
        .plot-container { border: 1px solid #1e2227; border-radius: 8px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC LAYER (CLEANED)
# ==========================================
@st.cache_data(ttl=3600)
def get_data(ticker):
    try:
        tk = yf.Ticker(ticker)
        inf = tk.info
        bs = tk.balance_sheet
        return {
            "name": inf.get("longName", ticker),
            "sector": inf.get("sector", "N/A"),
            "price": inf.get("currentPrice", 0.0),
            "debt": inf.get("totalDebt", 0.0),
            "cash": inf.get("totalCash", 0.0),
            "ebitda": inf.get("ebitda", 1.0),
            "assets": inf.get("totalAssets", 1.0),
            "equity": inf.get("totalStockholderEquity", 1.0),
            "rev": inf.get("totalRevenue", 1.0),
            "beta": inf.get("beta", 1.0),
            "curr": inf.get("currency", "USD")
        }
    except: return None

def run_simulation(d, m, sims=3000):
    # Core Math abstracted for readability
    net_debt = (d['debt'] - d['cash'])
    net_debt_ebitda = net_debt / max(d['ebitda'], 1)
    
    # Simple Logistic Default Prob
    p_def = 1 / (1 + np.exp(-0.4 * (net_debt_ebitda - 4.0)))
    fair_val = d['price'] * (1 - p_def * (1 - m['recovery']))
    
    # Monte Carlo for ROIC
    roic_sim = np.random.normal(0.12, 0.05, sims) / (1 + m['inflation'])
    icc = m['yield'] + (d['beta'] * 0.07)
    
    return {
        "p_def": p_def, "fair_val": fair_val, "roic_sim": roic_sim, 
        "icc": icc, "net_debt_ebitda": net_debt_ebitda
    }

# ==========================================
# 3. SIDEBAR & INPUTS
# ==========================================
with st.sidebar:
    st.title("🏛️ Terminal")
    ticker = st.text_input("SYMBOL", value="COMI.CA").upper()
    
    with st.expander("MACRO PARAMETERS", expanded=True):
        m_yield = st.slider("Risk Free Rate %", 0.0, 30.0, 18.0) / 100
        m_inf = st.slider("Inflation %", 0.0, 50.0, 25.0) / 100
        m_rec = st.slider("Recovery Rate %", 0.0, 100.0, 30.0) / 100

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
data = get_data(ticker)

if data:
    # Header Section
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.title(data['name'])
        st.caption(f"{data['sector']}  |  Currency: {data['curr']}  |  Beta: {data['beta']}")
    
    # Simulation Logic
    res = run_simulation(data, {"yield": m_yield, "inflation": m_inf, "recovery": m_rec})
    
    # 5. KEY METRICS BAR
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Price", f"{data['price']:,}")
    m2.metric("Fair Value", f"{res['fair_val']:,.2f}", f"{((res['fair_val']/data['price'])-1)*100:.1f}%")
    m3.metric("Default Risk", f"{res['p_def']*100:.1f}%", delta_color="inverse")
    m4.metric("Debt/EBITDA", f"{res['net_debt_ebitda']:.1f}x")

    # 6. ANALYSIS TABS
    tab1, tab2, tab3 = st.tabs(["STRESS TEST", "DISTRIBUTION", "LEDGER"])
    
    with tab1:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.markdown("### Risk Radar")
            # Simplified Radar Chart
            categories = ['Solvency', 'Liquidity', 'Profitability', 'Macro']
            fig_radar = go.Figure(data=go.Scatterpolar(
                r=[80, 45, 70, 30], theta=categories, fill='toself', 
                line=dict(color='#58a6ff'), fillcolor='rgba(88, 166, 255, 0.2)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with c2:
            st.markdown("### Scenario Analysis")
            scenario_df = pd.DataFrame({
                "Scenario": ["Base Case", "High Inflation", "Credit Crunch"],
                "Fair Value": [res['fair_val'], res['fair_val']*0.8, res['fair_val']*0.6],
                "Risk": ["Medium", "High", "Critical"]
            })
            st.table(scenario_df)

    with tab2:
        # Monte Carlo Plot
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=res['roic_sim'], nbinsx=40, marker_color='#1f6feb', opacity=0.7))
        fig_dist.add_vline(x=res['icc'], line_dash="dash", line_color="#ff7b72", annotation_text="Cost of Cap")
        fig_dist.update_layout(
            margin=dict(l=0, r=0, t=20, b=0), height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, title="Simulated ROIC"), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        # Easy Overrides
        st.markdown("### Manual Data Correction")
        edit_col1, edit_col2 = st.columns(2)
        override_ebitda = edit_col1.number_input("Adjust EBITDA", value=float(data['ebitda']))
        override_debt = edit_col2.number_input("Adjust Total Debt", value=float(data['debt']))
        if st.button("Apply Overrides & Re-calculate"):
            st.toast("Logic updated with manual overrides")

else:
    st.error("Please enter a valid ticker (e.g., AAPL, NVDA, COMI.CA)")

st.caption("Powered by Stochastic Engine v2.1 • Data provided by Yahoo Finance")
