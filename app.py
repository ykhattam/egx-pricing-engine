import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="EGX Survival & Real Pricing Engine",
    page_icon="🇪🇬",
    layout="wide"
)

# -------------------------------------------------------------------
# 1. CORE MATH ENGINE
# -------------------------------------------------------------------
def calculate_egx_real_price(fundamentals, live_price, macro, num_simulations=500):
    """
    Computes frontier-market adjusted solvency, Monte Carlo ROIC, and risk-adjusted valuation.
    """
    ebitda_safe = max(fundamentals['EBITDA'] + fundamentals['one_time_items'], 1e-5)
    assets_safe = max(fundamentals['assets'], 1e-5)
    liabilities_safe = max(fundamentals['liabilities'], 1e-5)
    
    # A. Solvency & Default Probability
    total_debt_adj = fundamentals['total_debt'] + fundamentals['operating_leases']
    net_debt_ebitda = (total_debt_adj - fundamentals['cash']) / ebitda_safe
    
    if fundamentals['state_owned'] == 1:
        net_debt_ebitda *= 0.5
        
    interest_coverage = ebitda_safe / max(fundamentals['interest_expense'], 1e-5)
    icr_penalty = 1.5 if interest_coverage < 1.5 else 1.0
    composite_risk = net_debt_ebitda * icr_penalty
    p_default = 1 / (1 + np.exp(-0.35 * (composite_risk - 4.5)))
    
    # B. Altman Z-Score
    wc_cash = fundamentals['cash'] - fundamentals['current_liabilities']
    state_adj = 0.7 if fundamentals['state_owned'] == 1 else 1.0
    
    altman_z = (
        1.2 * (wc_cash / assets_safe) +
        1.4 * (fundamentals['retained_earnings'] / assets_safe) +
        3.3 * (ebitda_safe / assets_safe) +
        0.6 * (fundamentals['equity'] / liabilities_safe) +
        1.0 * (fundamentals['revenue'] / assets_safe)
    ) * state_adj
    
    # C. Real Price Calculation
    expected_loss = p_default * (1 - 0.25)
    real_price = live_price * (1 - expected_loss)
    
    # D. Monte Carlo Simulation for ROIC Spread
    simulated_tax = np.random.uniform(0.10, 0.20, num_simulations)
    simulated_inflation = np.random.uniform(macro['shadow_inflation'] - 0.05, macro['shadow_inflation'] + 0.05, num_simulations)
    
    invested_capital = fundamentals['total_debt'] + fundamentals['equity'] - fundamentals['cash']
    invested_capital_safe = max(invested_capital, 1e-5)
    
    simulated_nopat = fundamentals['operating_income'] * (1 - simulated_tax) / (1 + simulated_inflation)
    simulated_roic = simulated_nopat / invested_capital_safe
    
    icc = macro['cbe_rate'] + (fundamentals['beta'] * 0.08)
    p_roic_gt_icc = float((simulated_roic > icc).mean())
    
    return {
        "p_default": p_default,
        "altman_z": altman_z,
        "real_price": real_price,
        "p_roic_gt_icc": p_roic_gt_icc,
        "simulated_roic_mean": simulated_roic.mean(),
        "icc": icc,
        "expected_loss": expected_loss
    }

# -------------------------------------------------------------------
# 2. DYNAMIC FUNDAMENTAL SCRAPER (YFINANCE)
# -------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Caches data for 1 hour to prevent API throttling
def fetch_dynamic_fundamentals(ticker_symbol):
    """
    Dynamically pulls accounting data for any global ticker. 
    Includes smart fallbacks for sparse frontier market data.
    """
    stock = yf.Ticker(ticker_symbol)
    info = stock.info
    
    try:
        bs = stock.balance_sheet
        fin = stock.financials
    except:
        bs, fin = pd.DataFrame(), pd.DataFrame()

    # Helper function to safely extract values from Yahoo DataFrames
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

    # Build automated corporate structure
    data = {
        "ticker": ticker_symbol,
        "total_debt": safe_info("totalDebt", get_statement_val(bs, "Total Debt", 0)),
        "cash": safe_info("totalCash", get_statement_val(bs, "Cash And Cash Equivalents", 0)),
        "EBITDA": safe_info("ebitda", get_statement_val(fin, "EBITDA", 1000)), 
        "one_time_items": 0.0,
        "operating_leases": get_statement_val(bs, "Operating Leases", 0.0),
        "interest_expense": get_statement_val(fin, "Interest Expense", 0.0),
        "current_liabilities": get_statement_val(bs, "Current Liabilities", 1000),
        "assets": safe_info("totalAssets", get_statement_val(bs, "Total Assets", 10000)),
        "retained_earnings": get_statement_val(bs, "Retained Earnings", 0.0),
        "equity": safe_info("totalStockholderEquity", get_statement_val(bs, "Stockholders Equity", 5000)),
        "liabilities": get_statement_val(bs, "Total Liabilities Net Minority Interest", 5000),
        "revenue": safe_info("totalRevenue", get_statement_val(fin, "Total Revenue", 10000)),
        "operating_income": safe_info("operatingMargins", 0.1) * safe_info("totalRevenue", 10000),
        "state_owned": 0, # Cannot be fetched via API, default to 0
        "beta": safe_info("beta", 1.0)
    }
    
    # Ensure no dividing by zero crashes
    data['assets'] = max(data['assets'], 1000)
    data['liabilities'] = max(data['liabilities'], 500)
    data['EBITDA'] = max(data['EBITDA'], 100)
    
    return data

@st.cache_data(ttl=120)
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
# 3. INTERACTIVE WEB INTERFACE
# -------------------------------------------------------------------
st.title("🌍 Universal Stock Solvency & Real Pricing Engine")
st.caption("Enter any ticker symbol globally (e.g., COMI.CA for Egypt, AAPL for US) to fetch and stress-test its financials.")

# Sidebar Configuration
st.sidebar.header("Global Macro Inputs")
cbe_rate = st.sidebar.slider("Risk-Free / Policy Rate (%)", 2.0, 35.0, 20.0, step=0.25) / 100
shadow_inflation = st.sidebar.slider("Estimated Inflation (%)", 2.0, 50.0, 22.0, step=1.0) / 100
official_fx = st.sidebar.number_input("Official FX Rate", value=49.18)
parallel_fx = st.sidebar.number_input("Parallel FX Rate", value=49.50)

st.sidebar.metric("Currency Friction Premium", f"{(((parallel_fx / official_fx) - 1) * 100):.2f}%")

# Main Dashboard - DYNAMIC TICKER INPUT
user_ticker = st.text_input("🔍 Search Ticker Symbol (Yahoo Finance Format)", value="COMI.CA")

with st.spinner(f"Fetching financial statements for {user_ticker}..."):
    company_data = fetch_dynamic_fundamentals(user_ticker)
    live_market_price = fetch_live_price(user_ticker)

# Editable Fundamentals (Crucial for Frontier Markets where YF data is often missing)
with st.expander(f"📝 View or Override {user_ticker} Fundamentals (Auto-Fetched)"):
    st.info("Yahoo Finance data for frontier markets can be incomplete. Override missing or 0 values below.")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_debt = st.number_input("Total Debt", value=float(company_data['total_debt']))
        cash = st.number_input("Cash & Equiv", value=float(company_data['cash']))
        ebitda = st.number_input("EBITDA", value=float(company_data['EBITDA']))
        is_state_owned = st.checkbox("Is State-Owned / Backed?", value=False)
        
    with col2:
        current_liabilities = st.number_input("Current Liabilities", value=float(company_data['current_liabilities']))
        assets = st.number_input("Total Assets", value=float(company_data['assets']))
        equity = st.number_input("Total Equity", value=float(company_data['equity']))
        
    with col3:
        operating_income = st.number_input("Operating Income", value=float(company_data['operating_income']))
        interest_expense = st.number_input("Annual Interest Exp", value=float(company_data['interest_expense']))
        retained_earnings = st.number_input("Retained Earnings", value=float(company_data['retained_earnings']))

    active_fundamentals = {
        "total_debt": total_debt, "cash": cash, "EBITDA": ebitda,
        "one_time_items": company_data['one_time_items'], "operating_leases": company_data['operating_leases'],
        "interest_expense": interest_expense, "current_liabilities": current_liabilities,
        "assets": assets, "retained_earnings": retained_earnings, "equity": equity,
        "liabilities": assets - equity, "revenue": company_data['revenue'],
        "operating_income": operating_income, 
        "state_owned": 1 if is_state_owned else 0,
        "beta": company_data['beta']
    }

# Compute Calculations
macro_inputs = {"cbe_rate": cbe_rate, "shadow_inflation": shadow_inflation}
results = calculate_egx_real_price(active_fundamentals, live_market_price, macro_inputs)

# -------------------------------------------------------------------
# DISPLAY METRICS
# -------------------------------------------------------------------
st.subheader("Live Solvency & Pricing Assessment")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.metric(label="Live Market Price", value=f"{live_market_price:.2f}", delta=user_ticker)
with m_col2:
    discount_pct = results['expected_loss'] * 100
    st.metric(label="Calculated Real Price", value=f"{results['real_price']:.2f}", delta=f"-{discount_pct:.1f}% Risk Adjustment", delta_color="inverse")
with m_col3:
    p_def = results['p_default'] * 100
    st.metric(label="Default Probability", value=f"{p_def:.2f}%", delta="Elevated Risk" if p_def > 15 else "Healthy Baseline", delta_color="inverse" if p_def > 15 else "normal")
with m_col4:
    z = results['altman_z']
    status = "Distress Zone 🚨" if z < 1.1 else "Grey Zone ⚠️" if z < 2.6 else "Safe Zone ✅"
    st.metric(label="Altman Z-Score", value=f"{z:.2f}", delta=status)

# Visual Charts Section
st.markdown("---")
st.subheader("Monte Carlo Capital Efficiency Analysis")
c_col1, c_col2 = st.columns([2, 1])

with c_col1:
    sim_data = np.random.normal(results['simulated_roic_mean'], 0.05, 500)
    df_chart = pd.DataFrame({
        "Simulated ROIC Distribution": sim_data * 100,
        "Implied Cost of Capital (ICC)": [results['icc'] * 100] * 500
    })
    st.line_chart(df_chart)

with c_col2:
    st.metric(label="Implied Cost of Capital", value=f"{(results['icc'] * 100):.2f}%")
    st.metric(label="Likelihood: ROIC > ICC", value=f"{(results['p_roic_gt_icc'] * 100):.1f}%")
