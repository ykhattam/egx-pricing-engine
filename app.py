import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime

# ==========================================
# 1. UI CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Stochastic Terminal", page_icon="🏛️")

# Clean, minimalist dark theme CSS
st.markdown("""
    <style>
        .main { background-color: #0d1117; }
        .stMetric { background: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
        h1, h2, h3 { color: #c9d1d9 !important; font-family: sans-serif; }
        .css-1r6slb0 { background-color: #0d1117; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATA ENGINE
# ==========================================
def to_float(val, default=0.0):
    """Ensures values are always numbers, never None."""
    try: return float(val) if val is not None else default
    except: return default

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "price": to_float(info.get("currentPrice") or info.get("previousClose"), 1.0),
            "sector": info.get("sector", "N/A"),
            "pe": to_float(info.get("trailingPE")),
            "pb": to_float(info.get("priceToBook")),
            "roe": to_float(info.get("returnOnEquity"), 0.1),
            "roa": to_float(info.get("returnOnAssets"), 0.05),
            "debt": to_float(info.get("totalDebt")),
            "cash": to_float(info.get("totalCash")),
            "ebitda": to_float(info.get("ebitda"), 1.0),
            "beta": to_float(info.get("beta"), 1.0)
        }
    except: return None

def get_metrics(d, m):
    # Fixed calculation to prevent NoneType errors
    net_debt = max(d['debt'] - d['cash'], 0.1)
    leverage = net_debt / max(d['ebitda'], 1.0)
    p_def = 1 / (1 + np.exp(-0.4 * (leverage - 4.0)))
    fair_val = d['price'] * (1 - p_def * (1 - m['recovery']))
    
    # Ensure ROA is a float for math
    sim_roic = np.random.normal(d['roa'] * 1.5, 0.05, 2000) / (1 + m['inflation'])
    icc = m['yield'] + (d['beta'] * 0.075)
    
    return {"p_def": p_def, "fair_val": fair_val, "sim_roic": sim_roic, "icc": icc, "z_score": 3.0 - (p_def * 2)}

# ==========================================
# 3. SIDEBAR & INTERFACE
# ==========================================
with st.sidebar:
    st.title("🏛️ ANALYST")
    ticker = st.text_input("Enter Ticker", value="COMI.CA").upper()
    st.markdown("---")
    m_yield = st.slider("Risk Free Rate %", 0.0, 40.0, 18.0) / 100
    m_infl = st.slider("Expected Inflation %", 0.0, 50.0, 25.0) / 100
    m_rec = st.slider("Recovery Rate %", 10, 90, 30) / 100

# ==========================================
# 4. DASHBOARD
# ==========================================
data = fetch_data(ticker)

if data:
    res = get_metrics(data, {"yield": m_yield, "inflation": m_infl, "recovery": m_rec})
    
    st.header(data['name'])
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Price", f"{data['price']:,}")
    c2.metric("Fair Value", f"{res['fair_val']:,.2f}")
    c3.metric("Default Risk", f"{res['p_def']*100:.1f}%")
    c4.metric("Z-Score", f"{res['z_score']:.2f}")

    # Tabs
    tab1, tab2 = st.tabs(["📊 PERFORMANCE", "📈 SIMULATION"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("P/E Ratio", f"{data['pe']:.1f}")
        col2.metric("P/B Ratio", f"{data['pb']:.1f}")
        col3.metric("ROE", f"{data['roe']*100:.1f}%")

    with tab2:
        fig = go.Figure(go.Histogram(x=res['sim_roic']*100, marker_color='#58a6ff'))
        fig.add_vline(x=res['icc']*100, line_dash="dash", line_color="#ff7b72")
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Distribution of projected ROIC vs Cost of Capital (Red Line)")

else:
    st.error("Ticker not found or data unavailable. Please check the symbol.")
