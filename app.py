import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime

# ==========================================
# 1. THEME & UI SETUP
# ==========================================
st.set_page_config(layout="wide", page_title="Stochastic Analyst Terminal", page_icon="🏛️")

# Dark/Light Theme Toggle
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

# CSS for Financial Terminal Aesthetics
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@300;400;600&display=swap');
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            background-color: {'#0d1117' if theme == 'Dark' else '#ffffff'};
            color: {'#c9d1d9' if theme == 'Dark' else '#1f2328'};
        }}
        .metric-value {{ font-family: 'Fira Code', monospace; font-size: 1.1rem; }}
        .metric-delta {{ font-size: 0.8rem; }}
        .stMetric {{
            background-color: {'#161b22' if theme == 'Dark' else '#f6f8fa'};
            border: 1px solid {'#30363d' if theme == 'Dark' else '#d0d7de'};
            border-radius: 8px;
            padding: 12px;
        }}
        .risk-high {{ color: #ff7b72; }}
        .risk-medium {{ color: #d29922; }}
        .risk-low {{ color: #3fb950; }}
        .override-diff {{ font-size: 0.8rem; opacity: 0.7; }}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATA LAYER
# ==========================================
def to_float(val, default=0.0):
    """Safely cast value to float, preventing NoneType crashes."""
    try:
        if val is None or pd.isna(val):
            return default
        return float(val)
    except:
        return default

@st.cache_data(ttl=3600)
def search_ticker(query):
    """Hits Yahoo Finance search API for instant ticker matching."""
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = res.json().get('quotes', [])
        return [f"{i['symbol']} | {i.get('shortname', '')} ({i.get('exchDisp', '')})" for i in data if 'symbol' in i]
    except:
        return []

@st.cache_data(ttl=3600)
def fetch_comprehensive_data(ticker):
    """Fetches high-fidelity core financial metrics with statement fallbacks."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        bs = stock.balance_sheet
        fin = stock.financials

        def get_df_val(df, row_labels, default=0.0):
            if df is None or df.empty:
                return default
            for label in row_labels:
                matched_rows = [r for r in df.index if label.lower() in str(r).lower()]
                if matched_rows:
                    try:
                        val = df.loc[matched_rows[0]].iloc[0]
                        return to_float(val, default)
                    except:
                        continue
            return default

        total_assets = to_float(info.get("totalAssets"), get_df_val(bs, ["Total Assets"], 1.0))
        total_debt = to_float(info.get("totalDebt"), get_df_val(bs, ["Total Debt", "Long Term Debt"], 0.0))
        total_cash = to_float(info.get("totalCash"), get_df_val(bs, ["Cash And Cash Equivalents", "Cash"], 0.0))
        ebitda = to_float(info.get("ebitda"), get_df_val(fin, ["EBITDA", "Operating Income"], 1.0))
        revenue = to_float(info.get("totalRevenue"), get_df_val(fin, ["Total Revenue", "Revenue"], 1.0))
        op_inc = to_float(info.get("operatingIncome"), get_df_val(fin, ["Operating Income"], 1.0))
        ret_earn = get_df_val(bs, ["Retained Earnings", "Cumulative Retained Earnings"], 0.0)
        curr_liab = get_df_val(bs, ["Total Current Liabilities"], 1.0)
        interest_exp = get_df_val(fin, ["Interest Expense"], 1.0)

        raw_metrics = {
            "name": info.get("longName", ticker),
            "price": to_float(info.get("currentPrice") or info.get("previousClose"), 1.0),
            "currency": info.get("currency", "USD"),
            "sector": info.get("sector", "N/A"),
            "country": info.get("country", "N/A"),
            "pe_ratio": info.get("trailingPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "interest_expense": interest_exp,
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "ebitda_margin": info.get("ebitdaMargins"),
            "beta": to_float(info.get("beta"), 1.0),
            "short_ratio": info.get("shortRatio"),
            "dividend_yield": info.get("dividendYield"),
            "ebitda": ebitda,
            "total_debt": total_debt,
            "total_cash": total_cash,
            "total_assets": total_assets,
            "revenue": revenue,
            "operating_income": op_inc,
            "retained_earnings": ret_earn,
            "current_liabilities": curr_liab
        }

        nulls = sum(1 for v in raw_metrics.values() if v in (None, 0.0, 1.0))
        raw_metrics['quality_score'] = int((1 - (nulls / len(raw_metrics))) * 100)
        return raw_metrics
    except Exception as e:
        st.error(f"Failed to fetch data: {str(e)}")
        return None

# ==========================================
# 3. VALUATION ENGINE
# ==========================================
def calculate_stochastic_metrics(d, m, sims=3000):
    """Calculates all key risk-adjusted parameters and simulated returns."""
    ebitda = max(d['ebitda'], 1e-4)
    total_assets = max(d['total_assets'], 1e-4)
    equity = max(total_assets - d['total_debt'], 1e-4)
    cash = d['total_cash']
    liabilities = max(d['current_liabilities'], 1e-4)

    debt = d['total_debt']
    if m.get('debt_denom') == "USD" and m.get('is_frontier'):
        debt *= (1 + m.get('fx_friction', 0.0))

    net_debt_ebitda = (debt - cash) / ebitda
    icr = ebitda / max(d['interest_expense'], 1e-4)
    icr_penalty = 1.5 if icr < 1.5 else 1.0
    qual_mult = {"Low": 0.8, "Medium": 1.0, "High": 1.4}[m['qual_risk']]

    p_def = 1 / (1 + np.exp(-0.35 * (net_debt_ebitda * icr_penalty * qual_mult - 4.5)))
    p_def = np.clip(p_def, 0.0, 0.99)

    fair_val = d['price'] * (1 - p_def * (1 - m['recovery']))

    z_score = (
        1.2 * ((cash - liabilities) / total_assets) +
        1.4 * (d['retained_earnings'] / total_assets) +
        3.3 * (ebitda / total_assets) +
        0.6 * (equity / max(debt, 1e-4)) +
        1.0 * (d['revenue'] / total_assets)
    )

    sim_infl = np.random.triangular(m['inf_l'], m['inf_m'], m['inf_h'], sims)
    sim_tax = np.random.uniform(0.15, 0.25, sims)
    invested_cap = max(debt + equity - cash, 1e-4)

    sim_roic = (d['operating_income'] * (1 - sim_tax) / (1 + sim_infl)) / invested_cap
    icc = m['yield'] + (d['beta'] * 0.08) * qual_mult

    return {
        "p_def": p_def,
        "fair_value": fair_val,
        "z_score": z_score,
        "roic_array": sim_roic,
        "icc": icc,
        "value_creation_prob": float((sim_roic > icc).mean()),
        "net_debt_ebitda": net_debt_ebitda,
        "interest_coverage": icr
    }

# ==========================================
# 4. CONTROLLER & UI FLOW
# ==========================================
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []
if 'previous_results' not in st.session_state:
    st.session_state.previous_results = None

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.title("🔍 INPUTS")
    st.markdown("---")

    query = st.text_input("Find Asset", value="COMI.CA")
    options = search_ticker(query)
    ticker_choice = st.selectbox("Confirm Asset", options) if options else None
    ticker = ticker_choice.split(" | ")[0] if ticker_choice else query.upper()

    st.markdown("---")
    st.subheader("Macro Settings")
    is_frontier = st.checkbox("Frontier Market Adjustments", value=".CA" in ticker)
    m_yield = st.number_input("Risk-Free Rate (%)", value=22.0 if is_frontier else 4.5) / 100
    m_inf_m = st.number_input("Expected Inflation (%)", value=30.0 if is_frontier else 3.0) / 100
    qual_risk = st.selectbox("Country Risk Level", ["Low", "Medium", "High"], index=1 if is_frontier else 0)
    m_rec = st.slider("Recovery Rate (%)", 10, 90, 25) / 100

# Fetch data
data = fetch_comprehensive_data(ticker)

if data:
    # --- MODEL ASSUMPTIONS (OVERRIDES) ---
    st.markdown("### 📝 MODEL ASSUMPTIONS")
    with st.expander("Adjust Financial Ledger", expanded=True):
        c1, c2, c3 = st.columns(3)

        def log_change(field, old, new):
            diff_pct = ((new - old) / old * 100) if old != 0 else 0
            st.session_state.audit_log.append({
                "timestamp": datetime.now(),
                "field": field,
                "old": old,
                "new": new,
                "diff_pct": diff_pct
            })
            return f"<span class='override-diff'>({diff_pct:+.1f}%)</span>"

        o_debt = c1.number_input("Total Debt", min_value=0.0, value=float(data['total_debt']))
        if o_debt != data['total_debt']:
            diff_html = log_change("Debt", data['total_debt'], o_debt)
            st.markdown(f"**Debt Adjusted:** {o_debt:,.0f} {diff_html}", unsafe_allow_html=True)

        o_cash = c2.number_input("Total Cash", min_value=0.0, value=float(data['total_cash']))
        if o_cash != data['total_cash']:
            diff_html = log_change("Cash", data['total_cash'], o_cash)
            st.markdown(f"**Cash Adjusted:** {o_cash:,.0f} {diff_html}", unsafe_allow_html=True)

        o_ebitda = c3.number_input("EBITDA", min_value=0.1, value=float(data['ebitda']))
        if o_ebitda != data['ebitda']:
            diff_html = log_change("EBITDA", data['ebitda'], o_ebitda)
            st.markdown(f"**EBITDA Adjusted:** {o_ebitda:,.0f} {diff_html}", unsafe_allow_html=True)

    # Populate active variables
    active_fund = {
        **data,
        "total_debt": o_debt,
        "total_cash": o_cash,
        "ebitda": o_ebitda
    }

    active_macro = {
        "yield": m_yield,
        "inf_l": m_inf_m * 0.6,
        "inf_m": m_inf_m,
        "inf_h": m_inf_m * 1.5,
        "qual_risk": qual_risk,
        "fx_friction": 0.02 if is_frontier else 0.0,
        "debt_denom": "USD" if is_frontier else "Local",
        "is_frontier": is_frontier,
        "recovery": m_rec
    }

    # Execute risk model
    res = calculate_stochastic_metrics(active_fund, active_macro)

    # --- ALERTS IN SIDEBAR ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("🚨 ALERTS")
        if st.session_state.previous_results:
            delta_fv = res['fair_value'] - st.session_state.previous_results['fair_val
