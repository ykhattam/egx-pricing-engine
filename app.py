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

# CSS for a dark financial terminal aesthetic
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #c9d1d9; }
        .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
        div[data-testid="stExpander"] { border: 1px solid #30363d; background-color: #0d1117; }
        .stTabs [data-baseweb=tab] { font-size: 0.85rem; letter-spacing: 0.5px; text-transform: uppercase; }
        .stButton>button { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; width: 100%; }
        .stButton>button:hover { border-color: #58a6ff; color: #58a6ff; }
        .stDataFrame { background-color: #161b22; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. ROBUST DATA INTELLIGENCE LAYER
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

        # Fallback dictionary parser for pandas financial statements
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

        # Aggregate raw metrics across multiple endpoints
        total_assets = to_float(info.get("totalAssets"), get_df_val(bs, ["Total Assets"], 1.0))
        total_debt = to_float(info.get("totalDebt"), get_df_val(bs, ["Total Debt", "Long Term Debt"], 0.0))
        total_cash = to_float(info.get("totalCash"), get_df_val(bs, ["Cash And Cash Equivalents", "Cash"], 0.0))
        ebitda = to_float(info.get("ebitda"), get_df_val(fin, ["EBITDA", "Operating Income"], 1.0))
        revenue = to_float(info.get("totalRevenue"), get_df_val(fin, ["Total Revenue", "Revenue"], 1.0))
        op_inc = to_float(info.get("operatingIncome"), get_df_val(fin, ["Operating Income"], 1.0))
        ret_earn = get_df_val(bs, ["Retained Earnings", "Cumulative Retained Earnings"], 0.0)
        curr_liab = get_df_val(bs, ["Total Current Liabilities"], 1.0)
        interest_exp = get_df_val(fin, ["Interest Expense"], 1.0)

        # Assemble full profile
        raw_metrics = {
            "name": info.get("longName", ticker),
            "price": to_float(info.get("currentPrice") or info.get("previousClose"), 1.0),
            "currency": info.get("currency", "USD"),
            "sector": info.get("sector", "Unknown Sector"),
            "country": info.get("country", "Unknown Country"),
            
            # Valuation Ratios
            "pe_ratio": info.get("trailingPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            
            # Liquidity & Solvency
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "interest_expense": interest_exp,

            # Profitability
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "ebitda_margin": info.get("ebitdaMargins"),
            
            # Market Risk
            "beta": to_float(info.get("beta"), 1.0),
            "short_ratio": info.get("shortRatio"),
            "dividend_yield": info.get("dividendYield"),
            
            # Core Accounting numbers
            "ebitda": ebitda,
            "total_debt": total_debt,
            "total_cash": total_cash,
            "total_assets": total_assets,
            "revenue": revenue,
            "operating_income": op_inc,
            "retained_earnings": ret_earn,
            "current_liabilities": curr_liab
        }

        # Quality scoring based on presence of key metrics
        nulls = sum(1 for v in raw_metrics.values() if v in (None, 0.0, 1.0))
        raw_metrics['quality_score'] = int((1 - (nulls / len(raw_metrics))) * 100)
        return raw_metrics
    except Exception as e:
        st.error(f"Failed to resolve statement logic: {e}")
        return None

# ==========================================
# 3. VALUATION & RISK ENGINE
# ==========================================
def calculate_stochastic_metrics(d, m, sims=3000):
    """Calculates all key risk-adjusted parameters and simulated returns."""
    # Safety values to prevent division-by-zero
    ebitda = max(d['ebitda'], 1e-4)
    total_assets = max(d['total_assets'], 1e-4)
    equity = max(total_assets - d['total_debt'], 1e-4)
    cash = d['total_cash']
    liabilities = max(d['current_liabilities'], 1e-4)

    # Risk Adjusted Debt calculation
    debt = d['total_debt']
    if m['debt_denom'] == "USD" and m['is_frontier']:
        debt *= (1 + m['fx_friction'])

    # Default Probability Logistic Equation
    net_debt_ebitda = (debt - cash) / ebitda
    icr = ebitda / max(d['interest_expense'], 1e-4)
    icr_penalty = 1.5 if icr < 1.5 else 1.0
    qual_mult = {"Low": 0.8, "Medium": 1.0, "High": 1.4}[m['qual_risk']]
    
    p_def = 1 / (1 + np.exp(-0.35 * (net_debt_ebitda * icr_penalty * qual_mult - 4.5)))
    p_def = np.clip(p_def, 0.0, 0.99)

    # Stochastic Valuation
    fair_val = d['price'] * (1 - p_def * (1 - m['recovery']))

    # Altman Z-Score Calculation (Frontier/Emerging Market Variant)
    z_score = (
        1.2 * ((cash - liabilities) / total_assets) +
        1.4 * (d['retained_earnings'] / total_assets) +
        3.3 * (ebitda / total_assets) +
        0.6 * (equity / max(debt, 1e-4)) +
        1.0 * (d['revenue'] / total_assets)
    )

    # Cost of Capital & Monte Carlo ROIC Simulation
    sim_infl = np.random.triangular(m['inf_l'], m['inf_m'], m['inf_h'], sims)
    sim_tax = np.random.uniform(0.15, 0.25, sims)
    invested_cap = max(debt + equity - cash, 1e-4)
    
    sim_roic = (d['operating_income'] * (1 - sim_tax) / (1 + sim_infl)) / invested_cap
    icc = m['yield'] + (d['beta'] * 0.08) * qual_mult # Cost of Capital

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
# 4. CONTROLLER & UI INTEGRATION
# ==========================================
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []
if 'previous_results' not in st.session_state:
    st.session_state.previous_results = None

# Sidebar Config
with st.sidebar:
    st.title("🏛️ STOCHASTIC TERMINAL")
    st.markdown("---")
    
    # Search Ticker logic
    query = st.text_input("Find Asset / Ticker", value="COMI.CA")
    options = search_ticker(query)
    ticker_choice = st.selectbox("Confirm Asset Selection", options) if options else None
    
    ticker = ticker_choice.split(" | ")[0] if ticker_choice else query.upper()
    is_frontier = ".CA" in ticker or st.checkbox("Frontier Market Adjustments", value=".CA" in ticker)

    st.markdown("---")
    st.subheader("🌍 Macro Environment Settings")
    m_yield = st.number_input("Sovereign Yield (Risk-Free %)", value=22.0 if is_frontier else 4.5) / 100
    m_inf_m = st.number_input("Expected Shadow Inflation %", value=30.0 if is_frontier else 3.0) / 100
    qual_risk = st.selectbox("Country Risk Premium Overlay", ["Low", "Medium", "High"], index=1 if is_frontier else 0)
    fx_fric = st.number_input("Currency Friction %", value=2.0 if is_frontier else 0.0) / 100
    m_rec = st.slider("Expected Asset Recovery %", 10, 90, 25) / 100
    debt_denom = st.radio("Debt Currency Type", ["Local Currency", "USD"], index=1 if is_frontier else 0)

# Main Screen Data Flow
data = fetch_comprehensive_data(ticker)

if data:
    # Overrides Panel
    with st.expander("📝 Financial Statement Adjustments (Overrides)"):
        c1, c2, c3 = st.columns(3)
        
        def log_change(field, old, new):
            st.session_state.audit_log.append({
                "timestamp": datetime.now(), "field": field, "old": old, "new": new
            })

        o_debt = c1.number_input("Total Debt Outstanding", value=float(data['total_debt']))
        if o_debt != data['total_debt']: log_change("Debt", data['total_debt'], o_debt)

        o_cash = c1.number_input("Total Cash Equivalents", value=float(data['total_cash']))
        if o_cash != data['total_cash']: log_change("Cash", data['total_cash'], o_cash)

        o_ebitda = c2.number_input("Reported EBITDA", value=float(data['ebitda']))
        if o_ebitda != data['ebitda']: log_change("EBITDA", data['ebitda'], o_ebitda)

        o_opinc = c2.number_input("Operating Income (EBIT)", value=float(data['operating_income']))
        if o_opinc != data['operating_income']: log_change("EBIT", data['operating_income'], o_opinc)

        o_assets = c3.number_input("Total Book Assets", value=float(data['total_assets']))
        if o_assets != data['total_assets']: log_change("Assets", data['total_assets'], o_assets)

        o_curr_liab = c3.number_input("Current Liabilities", value=float(data['current_liabilities']))
        if o_curr_liab != data['current_liabilities']: log_change("Liabilities", data['current_liabilities'], o_curr_liab)

    # Inject modified variables back into engine data profile
    active_fund = {
        **data,
        "total_debt": o_debt,
        "total_cash": o_cash,
        "ebitda": o_ebitda,
        "operating_income": o_opinc,
        "total_assets": o_assets,
        "current_liabilities": o_curr_liab
    }

    active_macro = {
        "yield": m_yield,
        "inf_l": m_inf_m * 0.6,
        "inf_m": m_inf_m,
        "inf_h": m_inf_m * 1.5,
        "qual_risk": qual_risk,
        "fx_friction": fx_fric,
        "debt_denom": debt_denom,
        "is_frontier": is_frontier,
        "recovery": m_rec
    }

    # Execute risk model
    res = calculate_stochastic_metrics(active_fund, active_macro)

    # Delta tracking warning alert
    if st.session_state.previous_results:
        delta_fv = res['fair_value'] - st.session_state.previous_results['fair_value']
        if abs(delta_fv) > 1e-4:
            st.warning(f"Re-Valuation triggered. Shift in Fair Value: {delta_fv:+.2f} {data['currency']}")
    st.session_state.previous_results = res

    # Page Header Info
    st.title(data['name'])
    st.caption(f"Sector: {data['sector']} | Jurisdiction: {data['country']} | Quality Score: {data['quality_score']}%")

    # --- TOP MAIN KPI BAR ---
    st.markdown("### 🔑 Valuation Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Price", f"{data['price']:,} {data['currency']}", help="Current price on the exchange.")
    k2.metric("Stochastic Fair Value", f"{res['fair_value']:,.2f} {data['currency']}",
              f"{((res['fair_value']/data['price'])-1)*100:+.1f}%", help="Valuation after accounting for default outcomes and sovereign risk.")
    k3.metric("Probability of Default", f"{res['p_def']*100:.1f}%", help="Statistical chance of technical insolvency.")
    k4.metric("Altman Z-Score", f"{res['z_score']:.2f}", 
              "Safe Zone" if res['z_score'] > 2.6 else "Distressed", help="Ratios indicating short-to-medium term bankruptcy risk.")

    # --- INDICATOR TAB MODULES ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 MULTIPLES & RETURN", "🛡️ LEVERAGE & SOLVENCY", "📈 MC ROIC SIMULATION", "📖 GLOSSARY & AUDIT"])

    with tab1:
        st.markdown("#### Valuation Ratios & Asset Return Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P/E Ratio", f"{data['pe_ratio']:.1f}" if data['pe_ratio'] else "N/A", help="Price-to-Earnings Ratio.")
        c2.metric("P/S Ratio", f"{data['ps_ratio']:.2f}" if data['ps_ratio'] else "N/A", help="Price-to-Sales Ratio.")
        c3.metric("P/B Ratio", f"{data['pb_ratio']:.2f}" if data['pb_ratio'] else "N/A", help="Price-to-Book Ratio.")
        c4.metric("EV/EBITDA", f"{data['ev_ebitda']:.1f}x" if data['ev_ebitda'] else "N/A", help="Enterprise Value / EBITDA.")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Return on Equity (ROE)", f"{(to_float(data['roe'])*100):.1f}%", help="Efficiency of shareholder's capital.")
        c6.metric("Return on Assets (ROA)", f"{(to_float(data['roa'])*100):.1f}%", help="Efficiency of operational assets.")
        c7.metric("EBITDA Margin", f"{(to_float(data['ebitda_margin'])*100):.1f}%" if data['ebitda_margin'] else "N/A", help="Operational cash flow profitability.")
        c8.metric("Dividend Yield", f"{(to_float(data['dividend_yield'])*100):.1f}%" if data['dividend_yield'] else "0.0%", help="Annual cash payout performance.")

    with tab2:
        st.markdown("#### Balance Sheet Leverage & Health Ratios")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Debt-to-Equity Ratio", f"{to_float(data['debt_equity']):.1f}%" if data['debt_equity'] else "N/A", help="Total debt over stockholder equity.")
        s2.metric("Current Ratio", f"{to_float(data['current_ratio']):.2f}x" if data['current_ratio'] else "N/A", help="Short term liquidity metric.")
        s3.metric("Quick Ratio", f"{to_float(data['quick_ratio']):.2f}x" if data['quick_ratio'] else "N/A", help="Liquidity metric excluding inventory risk.")
        s4.metric("Interest Coverage", f"{res['interest_coverage']:.2f}x" if res['interest_coverage'] else "N/A", help="How easily EBITDA covers debt service costs.")

    with tab3:
        st.markdown("#### Return on Capital vs. Investment Hurdle Rate")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res['roic_array']*100, marker_color='#238636', opacity=0.8, nbinsx=45))
        fig.add_vline(x=res['icc']*100, line_dash="dash", line_color="#f85149", annotation_text=f"ICC ({res['icc']*100:.1f}%)")
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0), height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", xaxis=dict(title="Simulated Real ROIC %", gridcolor="#30363d"), yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"**Long-Term Output:** This company is projected to generate economic profits (ROIC > Hurdle Rate) in **{res['value_creation_prob']*100:.1f}%** of simulated macro scenarios.")

    with tab4:
        st.subheader("📖 Analyst Glossary")
        st.write("""
        *   **Stochastic Fair Value:** Calculates the probability of credit events and default costs given current micro-leverage and macro friction constraints.
        *   **Default Risk:** Built via emerging-market calibrated logs referencing Net Debt/EBITDA, interest coverage thresholds, and USD currency penalties.
        *   **Altman Z-Score:** A classic metric tracking bankruptcy risks within a 2-year window. Scores below 1.8 signify critical risk profiles.
        """)
        
        st.subheader("📑 Ledger Override Trail")
        if st.session_state.audit_log:
            st.dataframe(pd.DataFrame(st.session_state.audit_log))
        else:
            st.caption("No overrides applied during this session.")

else:
    st.error("No active connection or unable to parse target ticker. Search again in the sidebar.")

st.divider()
st.caption(f"Status: Live Terminal Sync | Sync Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
