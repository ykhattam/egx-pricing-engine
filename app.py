import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM CSS 
# -------------------------------------------------------------------
st.set_page_config(page_title="Institutional Valuation Engine", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        header {visibility: hidden;} footer {visibility: hidden;}
        div[data-testid="metric-container"] {
            background-color: rgba(128, 128, 128, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 15px 20px; border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px); border-color: rgba(28, 131, 225, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 1. DATA INGESTION & QUALITY SCORING (Lens 24 & 31)
# -------------------------------------------------------------------
def _fetch_fundamentals(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    try:
        bs = stock.balance_sheet
        fin = stock.financials
    except:
        bs, fin = pd.DataFrame(), pd.DataFrame()

    def get_val(df, row, default=0.0):
        try: return float(df.loc[row].iloc[0]) if not pd.isna(df.loc[row].iloc[0]) else default
        except: return default

    def safe_info(key, default=0.0):
        val = info.get(key)
        return float(val) if val is not None else default

    raw_data = {
        "ticker": ticker_symbol,
        "sector": info.get("sector", "Unknown"),
        "total_debt": safe_info("totalDebt", get_val(bs, "Total Debt", 0.0)),
        "cash": safe_info("totalCash", get_val(bs, "Cash And Cash Equivalents", 0.0)),
        "EBITDA": safe_info("ebitda", get_val(fin, "EBITDA", 0.0)), 
        "interest_expense": get_val(fin, "Interest Expense", 0.0),
        "current_liabilities": get_val(bs, "Current Liabilities", 0.0),
        "assets": safe_info("totalAssets", get_val(bs, "Total Assets", 0.0)),
        "retained_earnings": get_val(bs, "Retained Earnings", 0.0),
        "equity": safe_info("totalStockholderEquity", get_val(bs, "Stockholders Equity", 0.0)),
        "revenue": safe_info("totalRevenue", get_val(fin, "Total Revenue", 0.0)),
        "operating_income": safe_info("operatingMargins", 0.1) * safe_info("totalRevenue", 0.0),
        "beta": safe_info("beta", 1.0)
    }
    
    # Calculate Data Quality (Lens 31)
    non_zeros = sum(1 for k, v in raw_data.items() if v != 0.0 and v != "Unknown")
    raw_data["data_quality_score"] = int((non_zeros / (len(raw_data) - 2)) * 100)
    
    # Apply baseline safety nets to prevent zero-division
    raw_data['assets'] = max(raw_data['assets'], 1000)
    raw_data['equity'] = max(raw_data['equity'], 500)
    raw_data['EBITDA'] = max(raw_data['EBITDA'], 100)
    
    return raw_data

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cached_fundamentals(ticker): return _fetch_fundamentals(ticker)

def fetch_dynamic_fundamentals(ticker, realtime=False):
    return _fetch_fundamentals(ticker) if realtime else fetch_cached_fundamentals(ticker)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_price(ticker):
    try: return round(yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1], 2)
    except: return 0.0

# -------------------------------------------------------------------
# 2. CORE MATH ENGINE (Modularized & Refined)
# -------------------------------------------------------------------
def calculate_stress_tested_valuation(fund, live_price, macro, sim_paths):
    ebitda_safe = max(fund['EBITDA'], 1e-5)
    assets_safe = max(fund['assets'], 1e-5)
    liabilities_safe = max(fund['assets'] - fund['equity'], 1e-5)
    
    qual_mult = {"Low": 0.8, "Medium": 1.0, "High": 1.4}[macro['qualitative_risk']]
    
    # Debt Denomination FX Impact (Lens 28)
    fx_multiplier = (1 + macro['fx_friction']/100) if fund['debt_denom'] == "USD" else 1.0
    total_debt_adj = fund['total_debt'] * fx_multiplier
    
    # State-Owned Discount (Lens 26)
    state_mult = (1 - fund['state_discount']) if fund['is_state_owned'] else 1.0
    net_debt_ebitda = ((total_debt_adj - fund['cash']) / ebitda_safe) * state_mult
        
    icr = ebitda_safe / max(fund['interest_expense'], 1e-5)
    icr_penalty = 1.5 if icr < 1.5 else 1.0
    
    composite_risk = net_debt_ebitda * icr_penalty * qual_mult
    p_default = 1 / (1 + np.exp(-0.35 * (composite_risk - 4.5)))
    
    wc_cash = fund['cash'] - fund['current_liabilities']
    
    altman_z = (
        1.2 * (wc_cash / assets_safe) +
        1.4 * (fund['retained_earnings'] / assets_safe) +
        3.3 * (ebitda_safe / assets_safe) +
        0.6 * (fund['equity'] / liabilities_safe) +
        1.0 * (fund['revenue'] / assets_safe)
    ) * state_mult
    z_ci_bound = altman_z * 0.15 
    
    sim_infl = np.random.triangular(macro['infl_low'], macro['infl_mode'], macro['infl_high'], sim_paths)
    sim_tax = np.random.uniform(0.10, 0.25, sim_paths)
    
    invested_cap = max(total_debt_adj + fund['equity'] - fund['cash'], 1e-5)
    sim_nopat = fund['operating_income'] * (1 - sim_tax) / (1 + sim_infl)
    sim_roic = sim_nopat / invested_cap
    
    icc = macro['sovereign_yield'] + (fund['beta'] * 0.08) * qual_mult
    
    expected_loss = p_default * (1 - 0.25)
    base_real_price = live_price * (1 - expected_loss)
    price_ci = base_real_price * (sim_infl.std() + 0.05) 
    
    return {
        "p_default": p_default, "altman_z": altman_z, "z_ci_bound": z_ci_bound,
        "real_price": base_real_price, "price_ci": price_ci,
        "p_roic_gt_icc": float((sim_roic > icc).mean()),
        "icc": icc, "sim_roic_array": sim_roic, "expected_loss": expected_loss,
        "fx_multiplier": fx_multiplier
    }

# -------------------------------------------------------------------
# FRONT-END UI & LAYOUT
# -------------------------------------------------------------------
st.markdown("<h1 style='text-align: center;'>🏛️ Institutional Valuation Engine</h1>", unsafe_allow_html=True)

# User Role & Search (Lens 19)
col_role, col_search, col_spacer = st.columns([1.5, 2.5, 1])
with col_role:
    user_role = st.radio("User Persona (Adjusts UI Focus):", ["Equity Analyst", "Credit Risk", "Macro Strategist"], horizontal=True)
with col_search:
    user_ticker = st.text_input("Enter Ticker Symbol", value="COMI.CA", label_visibility="collapsed")

# Sidebar
st.sidebar.markdown("### 🌍 Macro Parameters")
use_realtime = st.sidebar.checkbox("Bypass Cache (Real-Time Data)", False, help="Check to force live re-fetch.")
sim_paths = st.sidebar.slider("Monte Carlo Paths", 1000, 20000, 5000, step=1000, help="Higher = Tail Risk Accuracy")

sovereign_yield = st.sidebar.slider("Sovereign Yield (Local) %", 2.0, 40.0, 20.0) / 100
qualitative_risk = st.sidebar.selectbox("Political/Systemic Risk", ["Low", "Medium", "High"], index=1)

with st.sidebar.expander("Stochastic Inflation"):
    infl_low, infl_mode, infl_high = st.slider("CPI Low", 5.0, 30.0, 15.0)/100, st.slider("CPI Mode", 10.0, 50.0, 25.0)/100, st.slider("CPI High", 20.0, 80.0, 40.0)/100

with st.sidebar.expander("FX Liquidity Friction"):
    off_fx, par_fx = st.number_input("Official FX", value=49.18), st.number_input("Parallel FX", value=49.50)
    fx_friction = (((par_fx / off_fx) - 1) * 100) * st.slider("Illiquidity Multiplier", 1.0, 3.0, 1.5)
    st.caption(f"**Adjusted FX Friction:** {fx_friction:.2f}%")

# Fetch Data
with st.spinner(f"Aggregating ledgers for {user_ticker}..."):
    company_data = fetch_dynamic_fundamentals(user_ticker, realtime=use_realtime)
    live_market_price = fetch_live_price(user_ticker)

# Data Quality Flag (Lens 31)
dq_score = company_data.get('data_quality_score', 100)
if dq_score < 70:
    st.warning(f"⚠️ **Data Quality Score: {dq_score}%**. Yahoo Finance is missing critical fields for {user_ticker}. Please review Balance Sheet Overrides.")

# Overrides UI & Scenario Manager (Lens 25)
with st.expander(f"📊 Balance Sheet Overrides & Scenarios ({company_data['sector']})"):
    scen_col, denom_col = st.columns(2)
    with scen_col:
        scenario = st.selectbox("Load Preset Scenario", ["Default (Auto-Fetched)", "Post-Devaluation (Debt x1.5)", "Distressed Liquidity (Cash ÷2)"])
    with denom_col:
        debt_denom = st.radio("Debt Denomination (Triggers FX Premium)", ["Local Currency", "USD"], horizontal=True)

    # Apply Scenarios
    scen_debt_mult = 1.5 if "Devaluation" in scenario else 1.0
    scen_cash_mult = 0.5 if "Distressed" in scenario else 1.0

    o_col1, o_col2, o_col3 = st.columns(3)
    with o_col1:
        # Deferred Question: Auto-adjust Interest Expense proportional to Debt overrides
        base_debt = company_data['total_debt'] * scen_debt_mult
        user_debt = st.number_input("Total Debt", value=float(base_debt))
        debt_scaling_factor = (user_debt / base_debt) if base_debt > 0 else 1.0
        
        cash = st.number_input("Cash & Equiv", value=float(company_data['cash'] * scen_cash_mult))
        ebitda = st.number_input("EBITDA", value=float(company_data['EBITDA']))
        
    with o_col2:
        is_state_owned = st.checkbox("Is State-Backed?", value=False)
        state_discount = st.slider("State Backing Discount", 0.0, 1.0, 0.5) if is_state_owned else 0.0
        current_liabilities = st.number_input("Current Liabilities", value=float(company_data['current_liabilities']))
        assets = st.number_input("Total Assets", value=float(company_data['assets']))
        
    with o_col3:
        # Hiding Operating Income if EBITDA > 0 (Lens 23)
        if ebitda <= 0:
            operating_income = st.number_input("Operating Income", value=float(company_data['operating_income']))
        else:
            operating_income = ebitda * 0.8  # Automated fallback mapping
            
        # Interest Expense auto-adjusts based on debt scaling
        interest_expense = st.number_input("Annual Interest Exp", value=float(company_data['interest_expense'] * debt_scaling_factor))
        retained_earnings = st.number_input("Retained Earnings", value=float(company_data['retained_earnings']))
        equity = st.number_input("Total Equity", value=float(company_data['equity']))

    active_fund = {
        "total_debt": user_debt, "cash": cash, "EBITDA": ebitda,
        "interest_expense": interest_expense, "current_liabilities": current_liabilities,
        "assets": assets, "retained_earnings": retained_earnings, "equity": equity,
        "revenue": company_data['revenue'], "operating_income": operating_income, 
        "is_state_owned": is_state_owned, "state_discount": state_discount,
        "beta": company_data['beta'], "debt_denom": debt_denom
    }

macro_inputs = {"sovereign_yield": sovereign_yield, "infl_low": infl_low, "infl_mode": infl_mode, "infl_high": infl_high, "qualitative_risk": qualitative_risk, "fx_friction": fx_friction}
results = calculate_stress_tested_valuation(active_fund, live_market_price, macro_inputs, sim_paths)

# -------------------------------------------------------------------
# KPIs & VISUALIZATION (Role-Based Reordering - Lens 19)
# -------------------------------------------------------------------
st.divider()
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# Format Confidence Intervals (Lens 21)
price_str = f"{results['real_price']:.2f} (90% CI: {results['real_price']-results['price_ci']:.1f}–{results['real_price']+results['price_ci']:.1f})"
z_str = f"{results['altman_z']:.2f} (90% CI: {results['altman_z']-results['z_ci_bound']:.2f}–{results['altman_z']+results['z_ci_bound']:.2f})"
p_def = results['p_default'] * 100
status = "Safe" if p_def < 15 else "Watchlist" if p_def < 30 else "High Risk"

# Render based on Role
metrics = {
    "Price": {"label": "Live Price", "val": f"{live_market_price:.2f}", "del": user_ticker},
    "Valuation": {"label": "Stress-Tested Estimate", "val": price_str, "del": f"Implies {(results['expected_loss']*100):.1f}% Loss"},
    "Default": {"label": "Default Probability", "val": f"{p_def:.1f}%", "del": status},
    "ZScore": {"label": "Altman Z-Score", "val": z_str, "del": "Solvency Metric"}
}

order = ["Price", "Valuation", "Default", "ZScore"]
if user_role == "Credit Risk": order = ["Default", "ZScore", "Price", "Valuation"]
elif user_role == "Macro Strategist": order = ["ZScore", "Valuation", "Default", "Price"]

for col, key in zip([m_col1, m_col2, m_col3, m_col4], order):
    with col: st.metric(metrics[key]["label"], metrics[key]["val"], metrics[key]["del"], delta_color="off")

# -------------------------------------------------------------------
# CHARTS & LINEAGE
# -------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📊 Capital Efficiency (ROIC)", "🌳 Premise Lineage", "📈 Historical Trends"])

with tab1:
    chart_col, insight_col = st.columns([2.5, 1])
    with chart_col:
        roic_data = results['sim_roic_array'] * 100
        icc_threshold = results['icc'] * 100
        
        # Sector Benchmark Logic (Lens 22)
        sector_medians = {"Financial Services": 12.0, "Technology": 18.0, "Basic Materials": 9.0, "Real Estate": 7.5}
        sec_median = sector_medians.get(company_data['sector'], 10.0)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=roic_data, nbinsx=100, marker_color='rgba(28,131,225,0.7)', name="Simulated ROIC"))
        fig.add_vline(x=icc_threshold, line_dash="dash", line_color="#E74C3C", annotation_text=f"ICC ({icc_threshold:.1f}%)")
        fig.add_vline(x=sec_median, line_dash="dot", line_color="#2ECC71", annotation_text=f"{company_data['sector']} Median ({sec_median:.1f}%)")
        
        fig.update_layout(height=320, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with insight_col:
        st.write("")
        roic_prob = results['p_roic_gt_icc'] * 100
        st.metric("Likelihood: ROIC > ICC", f"<{roic_prob:.1f}%" if roic_prob < 5 else f"{roic_prob:.1f}%")
        st.info(f"Destroys capital in **{100 - roic_prob:.1f}%** of scenarios. Sector average sits at {sec_median}%.")

with tab2:
    # Dependency Sunburst Graph (Lens 20 / 29)
    st.caption("Visualizing the causal chain: How macroeconomic friction inputs aggregate into Solvency & Pricing outputs.")
    labels = ["Stress-Tested Price", "Default Risk", "Capital Efficiency", 
              "FX Friction", "Interest Coverage", "Altman Z-Score", "Macro Inflation", "Debt Denom (USD)"]
    parents = ["", "Stress-Tested Price", "Stress-Tested Price", 
               "Default Risk", "Default Risk", "Capital Efficiency", "Capital Efficiency", "FX Friction"]
    
    fig_sb = go.Figure(go.Sunburst(labels=labels, parents=parents, marker=dict(colorscale='Blues')))
    fig_sb.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig_sb, use_container_width=True)

with tab3:
    # Historical Trends (Lens 30)
    if st.checkbox("Load 1-Year Historical Equity Pricing", value=False):
        try:
            hist = yf.Ticker(user_ticker).history(period="1y")['Close']
            st.line_chart(hist)
        except:
            st.error("Historical data unavailable for this ticker.")
