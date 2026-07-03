import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import re
import plotly.graph_objects as go

# ====================== CONFIG & STYLES ======================
st.set_page_config(layout="wide", page_title="Institutional Valuation Engine", page_icon="🏛️")
st.markdown("""
<style>
    .stMetric { background-color: rgba(240,242,246,0.5); padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb=tab] { font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

# ====================== DATA SCRAPER ======================
@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Aggressive fetching for Altman Z components
    return {
        "totalDebt": info.get("totalDebt", 0),
        "totalCash": info.get("totalCash", 0),
        "ebitda": info.get("ebitda", 1000),
        "totalAssets": info.get("totalAssets", 10000),
        "totalStockholderEquity": info.get("totalStockholderEquity", 5000),
        "totalRevenue": info.get("totalRevenue", 10000),
        "operatingIncome": info.get("operatingIncome", 1000),
        "beta": info.get("beta", 1.0),
        "interest_expense": info.get("interestExpense", 100),
        "current_liabilities": info.get("totalCurrentLiabilities", 1000),
        "retained_earnings": info.get("retainedEarnings", 500),
        "sector": info.get("sector", "Unknown"),
        "ticker": ticker,
        "state_owned": 0
    }

@st.cache_data(ttl=120)
def fetch_live_price(ticker):
    try:
        history = yf.Ticker(ticker).history(period="1d")
        return round(history['Close'].iloc[-1], 2) if not history.empty else 0.0
    except:
        return 10.0 # Standard fallback

# ====================== MATH ENGINE ======================
def calculate_results(fundamentals, price, macro, sims=5000):
    ebitda = max(fundamentals['ebitda'], 1e-5)
    debt = fundamentals['totalDebt']
    if fundamentals.get('debt_denomination') == "USD":
        debt *= (1 + macro['fx_friction'])

    # Solvency math
    icr = ebitda / max(fundamentals.get('interest_expense', 1e-5), 1e-5)
    icr_penalty = 1.5 if icr < 1.5 else 1.0
    qual_mult = {"Low": 0.8, "Medium": 1.0, "High": 1.4}[macro['qualitative_risk']]
    state_mult = 0.5 if fundamentals['state_owned'] else 1.0
    
    risk_score = ((debt - fundamentals['totalCash']) / ebitda) * icr_penalty * qual_mult * state_mult
    p_default = 1 / (1 + np.exp(-0.35 * (risk_score - 4.5)))

    # ROIC Monte Carlo
    sim_infl = np.random.triangular(macro['infl_low'], macro['infl_mode'], macro['infl_high'], sims)
    sim_roic = (fundamentals['operatingIncome'] * (1 - 0.20) / (1 + sim_infl)) / \
               max(debt + fundamentals['totalStockholderEquity'] - fundamentals['totalCash'], 1e-5)

    icc = macro['sovereign_yield'] + (fundamentals['beta'] * 0.08) * qual_mult

    # Altman Z
    z = (
        1.2 * (fundamentals['totalCash'] - fundamentals['current_liabilities']) / max(fundamentals['totalAssets'], 1e-5) +
        1.4 * (fundamentals['retained_earnings'] / max(fundamentals['totalAssets'], 1e-5)) +
        3.3 * (ebitda / max(fundamentals['totalAssets'], 1e-5)) +
        0.6 * (fundamentals['totalStockholderEquity'] / max(fundamentals['totalAssets'] - fundamentals['totalStockholderEquity'], 1e-5)) +
        1.0 * (fundamentals['totalRevenue'] / max(fundamentals['totalAssets'], 1e-5))
    ) * (0.7 if fundamentals['state_owned'] else 1.0)

    return {
        "p_default": p_default,
        "real_price": price * (1 - p_default * 0.75),
        "p_roic_gt_icc": float((sim_roic > icc).mean()),
        "icc": icc,
        "simulated_roic": sim_roic,
        "altman_z": z
    }

# ====================== UI ======================
if "audit_log" not in st.session_state: st.session_state.audit_log = []

st.title("🏛️ Quantitative Equity Stress-Tester")

# Ticker Selection
t_cols = st.columns(5)
tickers = ["COMI.CA", "TMGH.CA", "ABUK.CA", "AAPL", "MSFT"]
for i, t in enumerate(tickers):
    if t_cols[i].button(t): st.session_state.ticker = t

user_ticker = st.text_input("Search Ticker", value=st.session_state.get("ticker", "COMI.CA")).upper()

# Macro Sidebar
with st.sidebar:
    st.header("Macro Inputs")
    macro = {
        "sovereign_yield": st.slider("Sovereign Yield %", 5.0, 40.0, 22.0) / 100,
        "qualitative_risk": st.selectbox("Risk Overlay", ["Low", "Medium", "High"], index=1),
        "infl_low": 0.15, "infl_mode": st.slider("Inflation (Mode) %", 10.0, 60.0, 30.0) / 100, "infl_high": 0.70,
        "fx_friction": st.slider("FX Friction %", 0.0, 20.0, 2.0) / 100
    }

# Fetch Data
raw_data = fetch_fundamentals(user_ticker)
price = fetch_live_price(user_ticker)

# Fundamentals Overrides
with st.expander("📝 Corporate Financial Overrides"):
    c1, c2, c3 = st.columns(3)
    overrides = {}
    overrides["totalDebt"] = c1.number_input("Total Debt", value=float(raw_data['totalDebt']))
    overrides["totalCash"] = c1.number_input("Cash", value=float(raw_data['totalCash']))
    overrides["ebitda"] = c2.number_input("EBITDA", value=float(raw_data['ebitda']))
    overrides["operatingIncome"] = c2.number_input("Op. Income", value=float(raw_data['operatingIncome']))
    overrides["totalAssets"] = c3.number_input("Total Assets", value=float(raw_data['totalAssets']))
    overrides["totalStockholderEquity"] = c3.number_input("Equity", value=float(raw_data['totalStockholderEquity']))
    overrides["state_owned"] = st.checkbox("State Backed?")
    
    # Audit logging
    for k, v in overrides.items():
        if v != raw_data.get(k):
            st.session_state.audit_log.append(f"{k} changed to {v}")
    
    fund_final = {**raw_data, **overrides}

# Results
res = calculate_results(fund_final, price, macro)

# Dashboard Display
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Live Market Price", f"{price:.2f}")
k2.metric("Stress-Tested Fair Value", f"{res['real_price']:.2f}", f"{(1-res['real_price']/price)*-100:.1f}% Risk Adj")
k3.metric("Default Probability", f"{res['p_default']*100:.2f}%", "Watchlist" if res['p_default'] > 0.2 else "Healthy")
k4.metric("Altman Z-Score", f"{res['altman_z']:.2f}", "Distress" if res['altman_z'] < 1.8 else "Safe")

# Charts
st.markdown("### Visualization")
chart_tab1, chart_tab2 = st.tabs(["Risk Radar", "ROIC Distribution"])

with chart_tab1:
    risk_scores = [
        min(100, res['altman_z'] * 20),
        res['p_roic_gt_icc'] * 100,
        min(100, (fund_final['totalCash']/max(fund_final['totalAssets'],1e-5))*500),
        100 - (macro['fx_friction']*200)
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=risk_scores,
        theta=['Solvency', 'Efficiency', 'Liquidity', 'Macro FX'],
        fill='toself'
    ))
    st.plotly_chart(fig_radar, use_container_width=True)

with chart_tab2:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=res['simulated_roic']*100, marker_color='#3366ff'))
    fig_hist.add_vline(x=res['icc']*100, line_color="red", annotation_text="Cost of Capital")
    st.plotly_chart(fig_hist, use_container_width=True)

# Final Audit Log
with st.expander("Audit Log"):
    st.write(st.session_state.audit_log)
