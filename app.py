import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import re
import plotly.graph_objects as go

# ====================== CONFIG & STYLES ======================
st.set_page_config(layout="wide", page_title="Valuation Engine", page_icon="🏛️")
st.markdown("""
<style>
    .metric-card {background: rgba(240,242,246,0.5); border-radius: 8px; padding: 12px; margin-bottom: 1rem;}
    .stTabs [data-baseweb=tab] {font-size: 1rem; padding: 0 16px;}
    .stButton>button {background: #f8f9fa; border: 1px solid #dee2e6;}
</style>
""", unsafe_allow_html=True)

# ====================== HELPER FUNCTIONS ======================
def safe_fetch(ticker, func):
    try:
        return func(ticker)
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {str(e)}")
        return 0.0 if func.__name__ == "fetch_live_price" else {"ticker": ticker}

@st.cache_data(ttl=3600)
def fetch_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "totalDebt": info.get("totalDebt", 0),
        "totalCash": info.get("totalCash", 0),
        "ebitda": info.get("ebitda", 1000),
        "totalAssets": info.get("totalAssets", 10000),
        "totalStockholderEquity": info.get("totalStockholderEquity", 5000),
        "totalRevenue": info.get("totalRevenue", 10000),
        "operatingIncome": info.get("operatingIncome", 1000),
        "beta": info.get("beta", 1.0),
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
        return 0.0

def log_override(field, old, new):
    st.session_state.audit_log.append({
        "timestamp": datetime.datetime.now(),
        "field": field,
        "old_value": old,
        "new_value": new
    })

def calculate_results(fundamentals, price, macro, sims=5000):
    ebitda = max(fundamentals['ebitda'] + fundamentals.get('one_time_items', 0), 1e-5)
    debt = fundamentals['totalDebt'] + fundamentals.get('operating_leases', 0)
    if fundamentals.get('debt_denomination') == "USD":
        debt *= (1 + macro['fx_friction'])

    p_default = 1 / (1 + np.exp(-0.35 * (
        (debt - fundamentals['totalCash']) / ebitda *
        (1.5 if ebitda / max(fundamentals.get('interest_expense', 1e-5), 1e-5) < 1.5 else 1.0) *
        {"Low": 0.8, "Medium": 1.0, "High": 1.4}[macro['qualitative_risk']] *
        (0.5 if fundamentals['state_owned'] else 1.0) - 4.5
    )))

    simulated_inflation = np.random.triangular(
        macro['infl_low'], macro['infl_mode'], macro['infl_high'], sims
    )
    simulated_roic = (
        fundamentals['operatingIncome'] *
        (1 - np.random.uniform(0.1, 0.25, sims)) /
        (1 + simulated_inflation)
    ) / max(fundamentals['totalDebt'] + fundamentals['totalStockholderEquity'] - fundamentals['totalCash'], 1e-5)

    icc = macro['sovereign_yield'] + (fundamentals['beta'] * 0.08) * {"Low": 0.8, "Medium": 1.0, "High": 1.4}[macro['qualitative_risk']]

    return {
        "p_default": p_default,
        "real_price": price * (1 - p_default * 0.75),
        "p_roic_gt_icc": float((simulated_roic > icc).mean()),
        "icc": icc,
        "simulated_roic": simulated_roic,
        "altman_z": (
            1.2 * (fundamentals['totalCash'] - fundamentals.get('current_liabilities', 1000)) /
            max(fundamentals['totalAssets'], 1e-5) +
            1.4 * (fundamentals.get('retained_earnings', 0) / max(fundamentals['totalAssets'], 1e-5)) +
            3.3 * (ebitda / max(fundamentals['totalAssets'], 1e-5)) +
            0.6 * (fundamentals['totalStockholderEquity'] / max(fundamentals['totalAssets'] - fundamentals['totalStockholderEquity'], 1e-5)) +
            1.0 * (fundamentals['totalRevenue'] / max(fundamentals['totalAssets'], 1e-5))
        ) * (0.7 if fundamentals['state_owned'] else 1.0)
    }

# ====================== UI COMPONENTS ======================
def ticker_input():
    st.markdown("## ⚡ Quantitative Valuation Engine")
    cols = st.columns(4)
    for i, t in enumerate(["COMI.CA", "TMGH.CA", "EGX30.CA", "CIEB.CA"]):
        if cols[i].button(t):
            st.session_state.ticker = t

    ticker = st.text_input("Enter Ticker", value=st.session_state.get("ticker", "COMI.CA"))
    if ticker and not re.match(r"^[A-Z]{1,5}(\.[A-Z]{2})?$", ticker.upper()):
        st.error("Invalid format. Use e.g., COMI.CA or AAPL")
    return ticker.upper()

def macro_inputs():
    with st.sidebar.expander("Macro Parameters", expanded=True):
        return {
            "sovereign_yield": st.slider("Sovereign Yield %", 2.0, 40.0, 20.0) / 100,
            "qualitative_risk": st.selectbox("Risk Overlay", ["Low", "Medium", "High"]),
            "infl_low": st.slider("Inflation Low %", 5.0, 30.0) / 100,
            "infl_mode": st.slider("Inflation Mode %", 10.0, 50.0) / 100,
            "infl_high": st.slider("Inflation High %", 20.0, 80.0) / 100,
            "fx_friction": st.slider("FX Friction %", 0.0, 10.0, 0.65) / 100,
            "debt_denomination": st.radio("Debt Denomination", ["Local", "USD"])
        }

def fundamentals_override(data, ticker):
    with st.expander(f"📊 Override Fundamentals ({ticker})"):
        cols = st.columns(3)
        fields = {
            "totalDebt": "Total Debt",
            "totalCash": "Cash & Equiv",
            "ebitda": "EBITDA",
            "operatingIncome": "Operating Income",
            "totalAssets": "Total Assets",
            "totalStockholderEquity": "Total Equity"
        }
        overrides = {}
        for i, (field, label) in enumerate(fields.items()):
            with cols[i % 3]:
                overrides[field] = st.number_input(label, value=float(data.get(field, 0)))
                if overrides[field] != data.get(field, 0):
                    log_override(field, data.get(field, 0), overrides[field])
        overrides["state_owned"] = st.checkbox("State Owned?", value=data.get("state_owned", 0))
        return {**data, **overrides}

def render_results(results, price, ticker):
    st.divider()
    cols = st.columns(4)
    metrics = [
        ("Live Price", f"{price:.2f}", ""),
        ("Fair Value", f"{results['real_price']:.2f}", f"±{results['real_price']*0.1:.2f}"),
        ("Default Risk", f"{results['p_default']*100:.1f}%", f"±{results['p_default']*10:.1f}%"),
        ("Altman Z", f"{results['altman_z']:.2f}", "Distress" if results['altman_z'] < 1.81 else "Safe")
    ]
    for i, (title, value, delta) in enumerate(metrics):
        cols[i].metric(title, value, delta, delta_color="off")

    # Risk Radar
    risk_scores = {
        "Solvency": min(100, results['altman_z'] * 20),
        "Profitability": min(100, results['p_roic_gt_icc'] * 100),
        "Liquidity": min(100, (data['totalCash'] / data.get('current_liabilities', 1000)) * 50),
        "FX Risk": min(100, 100 - macro['fx_friction'] * 1000)
    }
    fig = go.Figure(go.Scatterpolar(
        r=list(risk_scores.values()),
        theta=list(risk_scores.keys()),
        fill="toself"
    ))
    st.plotly_chart(fig)

    # ROIC vs ICC
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=results['simulated_roic']*100, nbinsx=50))
    fig.add_vline(x=results['icc']*100, line_color="red",
                 annotation_text=f"Cost of Capital ({results['icc']*100:.1f}%)")
    st.plotly_chart(fig)

def what_if_scenarios(results, fundamentals, macro):
    with st.expander("🔮 What-If Scenarios"):
        scenario = st.selectbox("Select Scenario", ["Base", "Devaluation", "Recession", "Hyperinflation"])
        if scenario == "Devaluation":
            macro['fx_friction'] *= 1.2
        elif scenario == "Recession":
            fundamentals['operatingIncome'] *= 0.7
        elif scenario == "Hyperinflation":
            macro['infl_mode'], macro['infl_high'] = 0.75, 1.0

        if st.button("Run Scenario"):
            new_results = calculate_results(fundamentals, fetch_live_price(st.session_state.ticker), macro)
            st.experimental_rerun()
            return new_results
    return results

# ====================== MAIN APP ======================
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "previous_results" not in st.session_state:
    st.session_state.previous_results = None

ticker = ticker_input()
if not ticker:
    st.stop()

data = safe_fetch(ticker, fetch_fundamentals)
macro = macro_inputs()
fundamentals = fundamentals_override(data, ticker)
price = fetch_live_price(ticker)

if price == 0.0:
    st.warning("Could not fetch live price. Using placeholder value.")
    price = 10.0  # Fallback value

results = calculate_results(fundamentals, price, macro)
render_results(results, price, ticker)
results = what_if_scenarios(results, fundamentals, macro) or results

# Delta Tracking
if st.session_state.previous_results:
    delta = results['p_default'] - st.session_state.previous_results['p_default']
    st.warning(f"Model Updated: Default Risk {delta:+.1%}")

st.session_state.previous_results = results

# Audit Log
with st.expander("🔍 Audit Log"):
    if st.session_state.audit_log:
        st.dataframe(pd.DataFrame(st.session_state.audit_log))
    else:
        st.info("No overrides recorded")
