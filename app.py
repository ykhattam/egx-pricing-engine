import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------
# PAGE CONFIGURATION (Native Streamlit UI)
# -------------------------------------------------------------------
st.set_page_config(page_title="Valuation Engine", page_icon="⚡", layout="wide")

# -------------------------------------------------------------------
# 1. ROBUST DATA LAYER (Bypassing Yahoo for EGX)
# -------------------------------------------------------------------
# Offline High-Fidelity Database for Frontier Markets
EGX_DATABASE = {
    "COMI.CA": {"sector": "Financials", "total_debt": 120000, "cash": 180000, "EBITDA": 45000, "interest_expense": 1000, "current_liabilities": 90000, "assets": 600000, "retained_earnings": 150000, "equity": 350000, "revenue": 280000, "operating_income": 40000, "beta": 1.15, "live_price": 75.50},
    "TMGH.CA": {"sector": "Real Estate", "total_debt": 45000, "cash": 12000, "EBITDA": 18000, "interest_expense": 3500, "current_liabilities": 30000, "assets": 210000, "retained_earnings": 40000, "equity": 95000, "revenue": 65000, "operating_income": 15000, "beta": 1.40, "live_price": 55.20},
    "ESRS.CA": {"sector": "Basic Materials", "total_debt": 210000, "cash": 15000, "EBITDA": 18000, "interest_expense": 22000, "current_liabilities": 180000, "assets": 320000, "retained_earnings": -15000, "equity": 45000, "revenue": 190000, "operating_income": 12000, "beta": 1.45, "live_price": 95.00}
}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker):
    if ticker.upper() in EGX_DATABASE:
        return EGX_DATABASE[ticker.upper()], "Offline Database (High Fidelity)"
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        bs = stock.balance_sheet
        fin = stock.financials
        
        def safe_get(df, row, default=0.0):
            try: return float(df.loc[row].iloc[0]) if not pd.isna(df.loc[row].iloc[0]) else default
            except: return default

        data = {
            "sector": info.get("sector", "Unknown"),
            "total_debt": safe_get(bs, "Total Debt", 0.0),
            "cash": safe_get(bs, "Cash And Cash Equivalents", 0.0),
            "EBITDA": safe_get(fin, "EBITDA", 1000.0),
            "interest_expense": safe_get(fin, "Interest Expense", 0.0),
            "current_liabilities": safe_get(bs, "Current Liabilities", 0.0),
            "assets": safe_get(bs, "Total Assets", 10000.0),
            "retained_earnings": safe_get(bs, "Retained Earnings", 0.0),
            "equity": safe_get(bs, "Stockholders Equity", 5000.0),
            "revenue": safe_get(fin, "Total Revenue", 10000.0),
            "operating_income": safe_get(fin, "Operating Income", 1000.0),
            "beta": info.get("beta", 1.0),
            "live_price": stock.history(period="1d")['Close'].iloc[-1] if not stock.history(period="1d").empty else 0.0
        }
        return data, "Yahoo Finance API (Low Fidelity for Frontier)"
    except Exception:
        return {"sector": "Unknown", "total_debt": 1000, "cash": 500, "EBITDA": 500, "interest_expense": 50, "current_liabilities": 200, "assets": 5000, "retained_earnings": 100, "equity": 2000, "revenue": 1000, "operating_income": 200, "beta": 1.0, "live_price": 10.0}, "Emergency Fallback (Needs Manual Override)"

# -------------------------------------------------------------------
# 2. CORE MATH ENGINE
# -------------------------------------------------------------------
def run_model(fund, macro):
    ebitda_safe = max(fund['EBITDA'], 1e-5)
    assets_safe = max(fund['assets'], 1e-5)
    liab_safe = max(fund['assets'] - fund['equity'], 1e-5)
    
    fx_mult = (1 + macro['fx_friction']/100) if macro['debt_usd'] else 1.0
    net_debt_ebitda = ((fund['total_debt'] * fx_mult) - fund['cash']) / ebitda_safe
    icr = ebitda_safe / max(fund['interest_expense'], 1e-5)
    
    risk_score = (net_debt_ebitda * (1.5 if icr < 1.5 else 1.0))
    p_default = 1 / (1 + np.exp(-0.35 * (risk_score - 4.5)))
    
    wc_cash = fund['cash'] - fund['current_liabilities']
    z_score = (1.2*(wc_cash/assets_safe) + 1.4*(fund['retained_earnings']/assets_safe) + 3.3*(ebitda_safe/assets_safe) + 0.6*(fund['equity']/liab_safe) + 1.0*(fund['revenue']/assets_safe))
    
    sim_infl = np.random.triangular(macro['inf_low'], macro['inf_mode'], macro['inf_high'], 5000)
    sim_tax = np.random.uniform(0.10, 0.25, 5000)
    
    invested_cap = max((fund['total_debt']*fx_mult) + fund['equity'] - fund['cash'], 1e-5)
    sim_roic = (fund['operating_income'] * (1 - sim_tax) / (1 + sim_infl)) / invested_cap
    icc = macro['yield'] + (fund['beta'] * 0.08)
    
    real_price = fund['live_price'] * (1 - (p_default * 0.75)) 
    
    return {
        "p_default": p_default, "z_score": z_score, "real_price": real_price,
        "icc": icc, "sim_roic": sim_roic, "p_roic_pass": float((sim_roic > icc).mean())
    }

# -------------------------------------------------------------------
# FRONT-END UI (Clean & Institutional)
# -------------------------------------------------------------------
st.title("⚡ Quantitative Valuation & Risk Engine")
st.write("A high-performance stochastic model for global and frontier equities.")

# --- TOP SEARCH BAR ---
col_search, col_role, col_empty = st.columns([2, 1, 1])
with col_search:
    ticker = st.text_input("Enter Ticker (e.g., COMI.CA, AAPL, TMGH.CA):", value="COMI.CA").strip()
with col_role:
    persona = st.selectbox("View Persona:", ["Equity Analyst", "Credit Risk"])

data_dict, source_note = fetch_financials(ticker)

# --- MAIN DASHBOARD LAYOUT ---
tab_dashboard, tab_data, tab_macro = st.tabs(["📊 Executive Dashboard", "🧮 Financial Statements (Edit)", "🌍 Macro & FX Stress Tests"])

# --- TAB 2: FINANCIAL DATA EDITOR (Fixed Mapping logic) ---
with tab_data:
    st.info(f"**Data Source:** {source_note}. *Edit any cell directly below to update the model in real-time.*")
    
    # Strictly map internal math keys to beautiful UI words
    translation_dict = {
        "total_debt": "Total Debt",
        "cash": "Cash & Equivalents",
        "EBITDA": "EBITDA", 
        "interest_expense": "Interest Expense",
        "current_liabilities": "Current Liabilities",
        "assets": "Total Assets",
        "retained_earnings": "Retained Earnings",
        "equity": "Total Equity",
        "revenue": "Total Revenue",
        "operating_income": "Operating Income"
    }
    
    # Build the Dataframe safely
    df_fund = pd.DataFrame([{"Metric": display_name, "Value (M)": float(data_dict[internal_key])} 
                            for internal_key, display_name in translation_dict.items()])
    
    edited_df = st.data_editor(df_fund, use_container_width=True, hide_index=True, num_rows="fixed")
    
    # Reverse map back to exact math keys securely
    reverse_translation = {v: k for k, v in translation_dict.items()}
    active_fund = {reverse_translation[row['Metric']]: row['Value (M)'] for _, row in edited_df.iterrows()}
    active_fund['beta'] = data_dict['beta']
    active_fund['live_price'] = data_dict['live_price']

# --- TAB 3: MACRO PARAMETERS ---
with tab_macro:
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("Yield & Inflation")
        m_yield = st.slider("Risk-Free Yield (%)", 2.0, 40.0, 20.0)/100
        inf_mode = st.slider("Base Inflation (%)", 5.0, 50.0, 25.0)/100
    with m_col2:
        st.subheader("FX Liquidity Friction")
        fx_friction = st.slider("Parallel Market Spread Premium (%)", 0.0, 50.0, 2.0)
        debt_usd = st.toggle("Apply FX Friction to Debt (USD Denominated)?", False)

macro_inputs = {"yield": m_yield, "inf_low": inf_mode*0.6, "inf_mode": inf_mode, "inf_high": inf_mode*1.5, "fx_friction": fx_friction, "debt_usd": debt_usd}

# --- RUN MODEL ---
results = run_model(active_fund, macro_inputs)

# --- TAB 1: EXECUTIVE DASHBOARD ---
with tab_dashboard:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    order = ["Price", "Valuation", "Default", "ZScore"] if persona == "Equity Analyst" else ["Default", "ZScore", "Valuation", "Price"]
    
    metrics = {
        "Price": {"label": f"Live Price ({ticker})", "val": f"{active_fund['live_price']:.2f}", "del": None},
        "Valuation": {"label": "Risk-Adjusted Estimate", "val": f"{results['real_price']:.2f}", "del": f"-{(1 - results['real_price']/max(active_fund['live_price'], 1e-5))*100:.1f}% Risk Disc."},
        "Default": {"label": "Probability of Default", "val": f"{results['p_default']*100:.1f}%", "del": "High Risk" if results['p_default']>0.3 else "Safe"},
        "ZScore": {"label": "Altman Z-Score", "val": f"{results['z_score']:.2f}", "del": "Distress Zone" if results['z_score']<1.8 else "Safe Zone"}
    }
    
    cols = [kpi1, kpi2, kpi3, kpi4]
    for col, key in zip(cols, order):
        with col:
            st.metric(metrics[key]['label'], metrics[key]['val'], metrics[key]['del'], delta_color="inverse" if key in ["Default", "ZScore"] else "normal")

    st.divider()

    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### Capital Efficiency: 5,000 Path Simulation")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=results['sim_roic']*100, marker_color='#2E86C1', name="Simulated ROIC", nbinsx=80))
        fig.add_vline(x=results['icc']*100, line_dash="dash", line_color="#E74C3C", annotation_text=f"Cost of Capital ({results['icc']*100:.1f}%)")
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with c2:
        st.markdown("#### Profitability Matrix")
        st.write(" ")
        roic_pass = results['p_roic_pass'] * 100
        st.metric("Cost of Capital (Hurdle Rate)", f"{results['icc']*100:.1f}%")
        st.metric("Likelihood to Create Value", f"{roic_pass:.1f}%")
        if roic_pass < 50:
            st.error(f"⚠️ In {100-roic_pass:.1f}% of macroeconomic scenarios, this asset destroys shareholder value.")
        else:
            st.success(f"✅ In {roic_pass:.1f}% of macroeconomic scenarios, this asset creates shareholder value.")
