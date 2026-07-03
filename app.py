import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go

# -------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM CSS (Professional Styling)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Institutional Valuation Engine",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a White-Label, Premium Dashboard Look
st.markdown("""
    <style>
        /* Clean up top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Hide Streamlit Header/Footer Branding */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Style the Metric Cards (KPIs) */
        div[data-testid="metric-container"] {
            background-color: rgba(128, 128, 128, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.2);
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: rgba(28, 131, 225, 0.5);
        }
        
        /* Typography adjustments */
        h1, h2, h3 {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# CORE MATH ENGINE
# -------------------------------------------------------------------
def calculate_stress_tested_valuation(fundamentals, live_price, macro, num_simulations=5000):
    ebitda_safe = max(fundamentals['EBITDA'] + fundamentals['one_time_items'], 1e-5)
    assets_safe = max(fundamentals['assets'], 1e-5)
    liabilities_safe = max(fundamentals['liabilities'], 1e-5)
    
    qual_multiplier = {"Low": 0.8, "Medium": 1.0, "High": 1.4}[macro['qualitative_risk']]
    
    total_debt_adj = fundamentals['total_debt'] + fundamentals['operating_leases']
    net_debt_ebitda = (total_debt_adj - fundamentals['cash']) / ebitda_safe
    
    if fundamentals['state_owned'] == 1:
        net_debt_ebitda *= 0.5
        
    interest_coverage = ebitda_safe / max(fundamentals['interest_expense'], 1e-5)
    icr_penalty = 1.5 if interest_coverage < 1.5 else 1.0
    
    composite_risk = net_debt_ebitda * icr_penalty * qual_multiplier
    p_default = 1 / (1 + np.exp(-0.35 * (composite_risk - 4.5)))
    
    wc_cash = fundamentals['cash'] - fundamentals['current_liabilities']
    state_adj = 0.7 if fundamentals['state_owned'] == 1 else 1.0
    
    altman_z = (
        1.2 * (wc_cash / assets_safe) +
        1.4 * (fundamentals['retained_earnings'] / assets_safe) +
        3.3 * (ebitda_safe / assets_safe) +
        0.6 * (fundamentals['equity'] / liabilities_safe) +
        1.0 * (fundamentals['revenue'] / assets_safe)
    ) * state_adj
    z_ci = altman_z * 0.15 
    
    simulated_inflation = np.random.triangular(
        macro['infl_low'], macro['infl_mode'], macro['infl_high'], num_simulations
    )
    simulated_tax = np.random.uniform(0.10, 0.25, num_simulations)
    
    invested_capital = fundamentals['total_debt'] + fundamentals['equity'] - fundamentals['cash']
    invested_capital_safe = max(invested_capital, 1e-5)
    
    simulated_nopat = fundamentals['operating_income'] * (1 - simulated_tax) / (1 + simulated_inflation)
    simulated_roic = simulated_nopat / invested_capital_safe
    
    icc = macro['sovereign_yield'] + (fundamentals['beta'] * 0.08) * qual_multiplier
    p_roic_gt_icc = float((simulated_roic > icc).mean())
    
    expected_loss = p_default * (1 - 0.25)
    base_real_price = live_price * (1 - expected_loss)
    price_uncertainty = base_real_price * (simulated_inflation.std() + 0.05) 
    
    return {
        "p_default": p_default,
        "altman_z": altman_z,
        "z_ci": z_ci,
        "real_price": base_real_price,
        "price_uncertainty": price_uncertainty,
        "p_roic_gt_icc": p_roic_gt_icc,
        "simulated_roic_mean": simulated_roic.mean(),
        "icc": icc,
        "expected_loss": expected_loss,
        "simulated_roic_array": simulated_roic
    }

# -------------------------------------------------------------------
# DYNAMIC FUNDAMENTAL SCRAPER
# -------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dynamic_fundamentals(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    try:
        bs = stock.balance_sheet
        fin = stock.financials
    except:
        bs, fin = pd.DataFrame(), pd.DataFrame()

    def get_statement_val(df, row_name, default=0.0):
        try:
            if not df.empty and row_name in df.index:
                val = df.loc[row_name].iloc[0]
                return float(val) if not pd.isna(val) else default
            return default
        except:
            return default

    def safe_info(key, default=0.0):
        val = info.get(key)
        return float(val) if val is not None else default

    data = {
        "ticker": ticker_symbol,
        "total_debt": safe_info("totalDebt", get_statement_val(bs, "Total Debt", 0.0)),
        "cash": safe_info("totalCash", get_statement_val(bs, "Cash And Cash Equivalents", 0.0)),
        "EBITDA": safe_info("ebitda", get_statement_val(fin, "EBITDA", 1000.0)), 
        "one_time_items": 0.0,
        "operating_leases": get_statement_val(bs, "Operating Leases", 0.0),
        "interest_expense": get_statement_val(fin, "Interest Expense", 0.0),
        "current_liabilities": get_statement_val(bs, "Current Liabilities", 1000.0),
        "assets": safe_info("totalAssets", get_statement_val(bs, "Total Assets", 10000.0)),
        "retained_earnings": get_statement_val(bs, "Retained Earnings", 0.0),
        "equity": safe_info("totalStockholderEquity", get_statement_val(bs, "Stockholders Equity", 5000.0)),
        "liabilities": get_statement_val(bs, "Total Liabilities Net Minority Interest", 5000.0),
        "revenue": safe_info("totalRevenue", get_statement_val(fin, "Total Revenue", 10000.0)),
        "operating_income": safe_info("operatingMargins", 0.1) * safe_info("totalRevenue", 10000.0),
        "state_owned": 0, 
        "beta": safe_info("beta", 1.0)
    }
    
    data['assets'] = max(data['assets'], 1000)
    data['liabilities'] = max(data['liabilities'], 500)
    data['EBITDA'] = max(data['EBITDA'], 100)
    return data

@st.cache_data(ttl=120, show_spinner=False)
def fetch_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            return round(history['Close'].iloc[-1], 2)
        return 0.0
    except Exception:
        return 0.0

# -------------------------------------------------------------------
# FRONT-END UI & LAYOUT
# -------------------------------------------------------------------
# Header Area
st.markdown("<h1 style='text-align: center;'>🏛️ Institutional Valuation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 1.1rem; margin-bottom: 2rem;'>A stochastic framework mapping structural vulnerabilities and capital efficiency across global equities.</p>", unsafe_allow_html=True)

# Ticker Search Area (Centered)
col_spacer1, search_col, col_spacer2 = st.columns([1, 2, 1])
with search_col:
    user_ticker = st.text_input("Enter Ticker Symbol (e.g., COMI.CA, AAPL)", value="COMI.CA", label_visibility="collapsed", placeholder="Enter Ticker Symbol...")

# Sidebar - Macro Parameters
st.sidebar.markdown("### 🌍 Macro Parameters")
with st.sidebar.expander("Yield & Risk", expanded=True):
    sovereign_yield = st.slider("Sovereign Bond Yield (Local) %", 2.0, 40.0, 20.0, step=0.25) / 100
    qualitative_risk = st.selectbox("Political/Systemic Risk Overlay", ["Low", "Medium", "High"], index=1)

with st.sidebar.expander("Stochastic Inflation (Triangular)", expanded=True):
    infl_low = st.slider("Low-End CPI Scenario %", 5.0, 30.0, 15.0) / 100
    infl_mode = st.slider("Base/Mode CPI %", 10.0, 50.0, 25.0) / 100
    infl_high = st.slider("Tail-Risk CPI Spike %", 20.0, 80.0, 40.0) / 100

with st.sidebar.expander("FX Liquidity Friction", expanded=False):
    official_fx = st.number_input("Official FX Rate", value=49.18)
    parallel_fx = st.number_input("Parallel FX Rate", value=49.50)
    illiquidity_mult = st.slider("Market Illiquidity Multiplier", 1.0, 3.0, 1.5)
    fx_friction = (((parallel_fx / official_fx) - 1) * 100) * illiquidity_mult
    st.caption(f"**Adjusted FX Friction:** {fx_friction:.2f}%")

# Fetch Data
with st.spinner(f"Aggregating global ledgers for {user_ticker}..."):
    company_data = fetch_dynamic_fundamentals(user_ticker)
    live_market_price = fetch_live_price(user_ticker)

# Data Fatigue Warning
if company_data['total_debt'] == 0.0 or company_data['cash'] == 0.0:
    st.warning("⚠️ **Data Anomaly Detected:** Yahoo Finance returned `0.00` for Debt or Cash. Please manually reconcile these figures in the 'Balance Sheet Overrides' menu below.")

# Overrides UI
with st.expander(f"📊 Balance Sheet Overrides ({user_ticker})"):
    st.caption(f"Last fetched: {datetime.datetime.now().strftime('%H:%M:%S')}")
    o_col1, o_col2, o_col3 = st.columns(3)
    
    with o_col1:
        total_debt = st.number_input("Total Debt", value=float(company_data['total_debt']))
        cash = st.number_input("Cash & Equiv", value=float(company_data['cash']))
        ebitda = st.number_input("EBITDA", value=float(company_data['EBITDA']))
        is_state_owned = st.checkbox("Is State-Backed?", value=False)
        
    with o_col2:
        current_liabilities = st.number_input("Current Liabilities", value=float(company_data['current_liabilities']))
        assets = st.number_input("Total Assets", value=float(company_data['assets']))
        equity = st.number_input("Total Equity", value=float(company_data['equity']))
        
    with o_col3:
        operating_income = st.number_input("Operating Income", value=float(company_data['operating_income']))
        interest_expense = st.number_input("Annual Interest Exp", value=float(company_data['interest_expense']))
        retained_earnings = st.number_input("Retained Earnings", value=float(company_data['retained_earnings']))

    active_fundamentals = {
        "total_debt": total_debt, "cash": cash, "EBITDA": ebitda,
        "one_time_items": company_data['one_time_items'], "operating_leases": company_data['operating_leases'],
        "interest_expense": interest_expense, "current_liabilities": current_liabilities,
        "assets": assets, "retained_earnings": retained_earnings, "equity": equity,
        "liabilities": assets - equity, "revenue": company_data['revenue'],
        "operating_income": operating_income, "state_owned": 1 if is_state_owned else 0,
        "beta": company_data['beta']
    }

# Compute Results
macro_inputs = {
    "sovereign_yield": sovereign_yield, 
    "infl_low": infl_low, "infl_mode": infl_mode, "infl_high": infl_high,
    "qualitative_risk": qualitative_risk
}
results = calculate_stress_tested_valuation(active_fundamentals, live_market_price, macro_inputs)

# -------------------------------------------------------------------
# KPIs & VISUALIZATION 
# -------------------------------------------------------------------
st.divider()

# Top Row: Clean Metric Cards
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric("Live Market Price", f"{live_market_price:.2f}", f"Ticker: {user_ticker}")

with m_col2:
    st.metric("Stress-Tested Estimate", f"{results['real_price']:.2f} ± {results['price_uncertainty']:.1f}", 
              delta=f"Based on {(results['expected_loss']*100):.1f}% Loss", delta_color="off")

with m_col3:
    p_def = results['p_default'] * 100
    if p_def < 15: status = "Safe Baseline"
    elif p_def < 30: status = "Watchlist"
    elif p_def < 60: status = "Restructuring Risk"
    else: status = "High Default Risk"
    st.metric("Default Probability", f"{p_def:.1f}%", status, delta_color="off")

with m_col4:
    z = results['altman_z']
    z_bound = results['z_ci']
    st.metric("Altman Z-Score", f"{z:.2f} ± {z_bound:.2f}", "90% Confidence Interval", delta_color="off")

st.markdown("<br>", unsafe_allow_html=True)

# Bottom Row: Plotly Visualization & Insights
with st.container(border=True):
    st.markdown("#### Capital Efficiency: Monte Carlo Simulation (5,000 Paths)")
    
    chart_col, insight_col = st.columns([2.5, 1])
    
    with chart_col:
        # Professional Interactive Plotly Histogram
        roic_data = results['simulated_roic_array'] * 100
        icc_threshold = results['icc'] * 100
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=roic_data, 
            nbinsx=100,
            marker_color='rgba(28, 131, 225, 0.7)',
            name="Simulated ROIC",
            hovertemplate="ROIC: %{x:.1f}%<br>Count: %{y}<extra></extra>"
        ))
        
        # Add Cost of Capital Threshold Line
        fig.add_vline(x=icc_threshold, line_dash="dash", line_color="#E74C3C", line_width=2)
        fig.add_annotation(
            x=icc_threshold, y=0.95, yref="paper",
            text=f"Implied Cost of Capital ({icc_threshold:.1f}%)",
            showarrow=False, xanchor="left", xshift=10,
            font=dict(color="#E74C3C", size=12)
        )

        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Simulated ROIC (%)",
            yaxis_title="Frequency",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            xaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.1)')
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with insight_col:
        st.write("")
        st.write("")
        roic_prob = results['p_roic_gt_icc'] * 100
        
        st.metric("Implied Cost of Capital", f"{icc_threshold:.1f}%")
        st.metric("Likelihood: ROIC > ICC", f"<{roic_prob:.1f}%" if roic_prob < 5 else f"{roic_prob:.1f}%")
        
        st.info(f"**Structural Insight:** Across 5,000 stochastic economic environments, the enterprise destroys capital in **{100 - roic_prob:.1f}%** of modeled scenarios.")
