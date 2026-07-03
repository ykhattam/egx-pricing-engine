import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime

# ==========================================
# 1. THE TERMINAL DESIGN (CSS)
# ==========================================
st.set_page_config(layout="wide", page_title="Stochastic Valuation Terminal", page_icon="🏛️")

st.markdown("""
    <style>
        .stApp { background-color: #0e1117; color: #ffffff; }
        [data-testid="stMetric"] {
            background-color: #161b22;
            border: 1px solid #30363d;
            padding: 15px;
            border-radius: 10px;
        }
        .stTabs [data-baseweb=tab] { font-weight: 600; font-size: 1.1rem; color: #8b949e; }
        .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. THE DATA INTELLIGENCE LAYER
# ==========================================
@st.cache_data(ttl=3600)
def fetch_global_data(ticker):
    """Fetches any global ticker and scores the data quality."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        bs = stock.balance_sheet
        fin = stock.financials
        
        def safe_val(df, row, default=0.0):
            try: return float(df.loc[row].iloc[0]) if not pd.isna(df.loc[row].iloc[0]) else default
            except: return default

        data = {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "total_debt": info.get("totalDebt", safe_val(bs, "Total Debt", 0.0)),
            "cash": info.get("totalCash", safe_val(bs, "Cash And Cash Equivalents", 0.0)),
            "ebitda": info.get("ebitda", safe_val(fin, "EBITDA", 1.0)),
            "assets": info.get("totalAssets", safe_val(bs, "Total Assets", 1.0)),
            "equity": info.get("totalStockholderEquity", safe_val(bs, "Stockholders Equity", 1.0)),
            "revenue": info.get("totalRevenue", safe_val(fin, "Total Revenue", 1.0)),
            "operating_income": info.get("operatingIncome", safe_val(fin, "Operating Income", 1.0)),
            "beta": info.get("beta", 1.2 if ".CA" in ticker else 1.0),
            "interest_expense": safe_val(fin, "Interest Expense", 1.0),
            "retained_earnings": safe_get_retained_earnings(bs),
            "current_liabilities": safe_val(bs, "Total Current Liabilities", 1.0),
            "currency": info.get("currency", "USD")
        }
        
        # Data Quality Score
        zeros = sum(1 for v in data.values() if v == 0.0 or v == 1.0)
        data['quality_score'] = int((1 - (zeros / len(data))) * 100)
        return data
    except:
        return None

def safe_get_retained_earnings(bs):
    for label in ["Retained Earnings", "Cumulative Retained Earnings"]:
        try: return float(bs.loc[label].iloc[0])
        except: continue
    return 0.0

# ==========================================
# 3. THE STOCHASTIC ENGINE
# ==========================================
def run_stochastic_model(fund, macro, sims=5000):
    # A. Solvency & Default Risk
    ebitda = max(fund['ebitda'], 1e-5)
    debt = fund['total_debt']
    if macro['debt_denom'] == "USD" and macro['is_frontier']:
        debt *= (1 + macro['fx_friction'])
    
    # State-owned backing discount (V1 logic)
    risk_mult = 0.5 if fund['is_state_owned'] else 1.0
    qual_mult = {"Low": 0.8, "Medium": 1.0, "High": 1.4}[macro['qual_risk']]
    
    net_debt_ebitda = (debt - fund['cash']) / ebitda
    icr_penalty = 1.5 if (ebitda / max(fund['interest_expense'], 1e-5)) < 1.5 else 1.0
    
    p_default = 1 / (1 + np.exp(-0.35 * (net_debt_ebitda * icr_penalty * qual_mult * risk_mult - 4.5)))
    
    # B. Altman Z (Frontier Variant)
    z = (1.2 * (fund['cash'] - fund['current_liabilities']) / fund['assets'] +
         1.4 * fund['retained_earnings'] / fund['assets'] +
         3.3 * ebitda / fund['assets'] +
         0.6 * fund['equity'] / (fund['assets'] - fund['equity']) +
         1.0 * fund['revenue'] / fund['assets']) * (0.7 if fund['is_state_owned'] else 1.0)

    # C. Monte Carlo ROIC (V1 Stochastic paths)
    sim_infl = np.random.triangular(macro['inf_l'], macro['inf_m'], macro['inf_h'], sims)
    sim_tax = np.random.uniform(0.15, 0.25, sims)
    
    invested_cap = max(debt + fund['equity'] - fund['cash'], 1e-5)
    sim_roic = (fund['operating_income'] * (1 - sim_tax) / (1 + sim_infl)) / invested_cap
    icc = macro['yield'] + (fund['beta'] * 0.08) * qual_mult
    
    return {
        "p_def": p_default,
        "fair_value": fund['price'] * (1 - p_default * (1 - macro['recovery'])),
        "z_score": z,
        "roic_array": sim_roic,
        "icc": icc,
        "value_creation_prob": float((sim_roic > icc).mean())
    }

# ==========================================
# 4. THE INTERFACE (EASE OF USE)
# ==========================================
st.sidebar.title("🏛️ Terminal Settings")

# Global Ticker Logic
ticker = st.sidebar.text_input("Global Search (Ticker)", value="COMI.CA").upper()
is_frontier = ".CA" in ticker or st.sidebar.checkbox("Apply Frontier Market Logic", value=True)

# Macro Inputs
with st.sidebar.expander("Macro Environment", expanded=True):
    m_yield = st.slider("Sovereign Yield %", 5.0, 45.0, 22.0 if is_frontier else 4.5) / 100
    inf_m = st.slider("Shadow Inflation (Mode) %", 2.0, 60.0, 30.0 if is_frontier else 3.0) / 100
    qual_risk = st.selectbox("Systemic Risk Overlay", ["Low", "Medium", "High"], index=1 if is_frontier else 0)
    fx_fric = st.slider("FX Friction / Parallel Spread %", 0.0, 20.0, 1.5 if is_frontier else 0.0) / 100
    recovery = st.slider("Estimated Recovery Rate %", 10.0, 90.0, 25.0) / 100

# Fetch Data
raw = fetch_global_data(ticker)
if not raw:
    st.error(f"Ticker {ticker} not found. Ensure format is correct (e.g., AAPL or TMGH.CA)")
    st.stop()

# Verification Metric
st.title(f"{raw['name']} ({ticker})")
st.caption(f"Sector: {raw['sector']} | Data Confidence: {raw['quality_score']}% | Base Currency: {raw['currency']}")

# Data Overrides
with st.expander("📝 Financial Ledger (Overrides)"):
    c1, c2, c3 = st.columns(3)
    o_debt = c1.number_input("Total Debt", value=float(raw['total_debt']))
    o_cash = c1.number_input("Total Cash", value=float(raw['cash']))
    o_ebitda = c2.number_input("EBITDA", value=float(raw['ebitda']))
    o_opinc = c2.number_input("Operating Income", value=float(raw['operating_income']))
    o_assets = c3.number_input("Total Assets", value=float(raw['assets']))
    o_equity = c3.number_input("Total Equity", value=float(raw['equity']))
    is_state = st.checkbox("State-Backed Entity / Sovereign Support?", value=False)
    debt_denom = st.radio("Debt Denom", ["Local Currency", "USD"], horizontal=True)

# Run Engine
price = round(yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1], 2)
active_fund = {
    **raw, "total_debt": o_debt, "cash": o_cash, "ebitda": o_ebitda, 
    "operating_income": o_opinc, "assets": o_assets, "equity": o_equity, 
    "is_state_owned": is_state, "price": price
}
active_macro = {
    "yield": m_yield, "inf_l": inf_m*0.6, "inf_m": inf_m, "inf_h": inf_m*1.5,
    "qual_risk": qual_risk, "fx_friction": fx_fric, "debt_denom": debt_denom,
    "is_frontier": is_frontier, "recovery": recovery
}

res = run_stochastic_model(active_fund, active_macro)

# ==========================================
# 5. THE PROFESSIONAL DASHBOARD
# ==========================================
tab1, tab2 = st.tabs(["📊 Stress-Test Analysis", "📈 Monte Carlo Distribution"])

with tab1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Live Market Price", f"{price}")
    k2.metric("Stochastic Fair Value", f"{res['fair_value']:.2f}", f"{(1-res['fair_value']/price)*-100:.1f}% Risk Adj")
    k3.metric("Default Probability", f"{res['p_def']*100:.2f}%", "Watchlist" if res['p_def'] > 0.2 else "Stable")
    k4.metric("Altman Z-Score", f"{res['z_score']:.2f}", "Distress" if res['z_score'] < 1.8 else "Safe")

    # Radar Chart
    risk_radar = {
        "Solvency": min(100, res['z_score'] * 20),
        "Efficiency": res['value_creation_prob'] * 100,
        "Liquidity": min(100, (active_fund['cash']/active_fund['current_liabilities'])*50),
        "Macro Strength": 100 - (active_macro['fx_friction']*500)
    }
    
    fig_radar = go.Figure(go.Scatterpolar(
        r=list(risk_radar.values()), theta=list(risk_radar.keys()), fill='toself',
        marker=dict(color='#58a6ff')
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False), bgcolor="#161b22"), 
                            paper_bgcolor="rgba(0,0,0,0)", height=400, showlegend=False)
    st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.markdown("#### Profitability vs. Hurdle Rate (5,000 Path Simulation)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=res['roic_array']*100, marker_color='#238636', nbinsx=60))
    fig_hist.add_vline(x=res['icc']*100, line_dash="dash", line_color="#f85149", 
                       annotation_text=f"Cost of Capital ({res['icc']*100:.1f}%)")
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400,
                           xaxis=dict(title="Simulated ROIC (%)", gridcolor="#30363d"),
                           yaxis=dict(gridcolor="#30363d"))
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.info(f"**Insight:** This company creates positive economic value in **{res['value_creation_prob']*100:.1f}%** of modeled macroeconomic scenarios.")
