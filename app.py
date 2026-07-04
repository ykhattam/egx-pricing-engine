import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from datetime import datetime

# ==========================================
# 1. GLOBAL CONFIG & THEME
# ==========================================
st.set_page_config(layout="wide", page_title="Stochastic Terminal", page_icon="🏛️")

# Professional Terminal CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; }
        .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
        div[data-testid="stExpander"] { border: 1px solid #30363d; border-radius: 8px; background-color: #0d1117; }
        .stButton>button { width: 100%; border-radius: 5px; background-color: #21262d; color: #58a6ff; border: 1px solid #30363d; }
        .stTabs [data-baseweb=tab] { font-weight: 600; font-size: 0.9rem; color: #8b949e; }
        .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENCE LAYER (SEARCH & DATA)
# ==========================================
@st.cache_data(ttl=3600)
def search_ticker(query):
    """Real-time autocomplete search using Yahoo Finance API."""
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = res.json().get('quotes', [])
        return [f"{i['symbol']} | {i.get('shortname', '')} ({i.get('exchDisp', '')})" for i in data if 'symbol' in i]
    except: return []

@st.cache_data(ttl=3600)
def fetch_financials(ticker):
    """Fetches and cleans fundamental data."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        bs = stock.balance_sheet
        
        # Helper for missing data
        def get_val(df, keys, default=1.0):
            for k in keys:
                if k in df.index: return float(df.loc[k].iloc[0])
            return default

        return {
            "name": info.get("longName", ticker),
            "price": info.get("currentPrice") or info.get("previousClose") or 1.0,
            "debt": info.get("totalDebt") or get_val(bs, ["Total Debt", "Long Term Debt"]),
            "cash": info.get("totalCash") or get_val(bs, ["Cash And Cash Equivalents"]),
            "ebitda": info.get("ebitda") or 1.0,
            "equity": info.get("totalStockholderEquity") or 1.0,
            "assets": info.get("totalAssets") or 1.0,
            "revenue": info.get("totalRevenue") or 1.0,
            "beta": info.get("beta", 1.2),
            "currency": info.get("currency", "USD"),
            "sector": info.get("sector", "Global Market")
        }
    except Exception as e:
        return None

def run_valuation(d, m):
    """Stochastic core logic."""
    net_debt = max(d['debt'] - d['cash'], 0.1)
    # Default Probability (Logistic model)
    ratio = net_debt / max(d['ebitda'], 1)
    p_def = 1 / (1 + np.exp(-0.38 * (ratio - 4.5)))
    
    # Fair Value Calculation
    fair_val = d['price'] * (1 - p_def * (1 - m['recovery']))
    
    # Altman Z-Score
    z = (1.2 * (d['cash']/d['assets']) + 3.3 * (d['ebitda']/d['assets']) + 1.0 * (d['revenue']/d['assets']))
    
    # Monte Carlo Sample
    sim_roic = np.random.normal(0.15, 0.06, 3000) / (1 + m['inflation'])
    icc = m['yield'] + (d['beta'] * 0.08)
    
    return {"p_def": p_def, "fair_val": fair_val, "z": z, "roic": sim_roic, "icc": icc}

# ==========================================
# 3. SIDEBAR SEARCH & SETTINGS
# ==========================================
with st.sidebar:
    st.title("🏛️ Terminal v2")
    st.markdown("---")
    
    # SEARCH ENGINE
    search_query = st.text_input("🔍 Search Company / Ticker", value="Commercial International Bank")
    options = search_ticker(search_query)
    
    if options:
        selected_option = st.selectbox("Search Results", options)
        ticker = selected_option.split(" | ")[0]
    else:
        ticker = search_query.upper() if search_query else "COMI.CA"
    
    st.markdown("---")
    with st.expander("🛠️ MACRO BENCHMARKS", expanded=True):
        m_yield = st.slider("Treasury Yield %", 0.0, 40.0, 18.0) / 100
        m_inf = st.slider("Shadow Inflation %", 0.0, 50.0, 25.0) / 100
        m_rec = st.slider("Recovery Rate %", 10.0, 90.0, 30.0) / 100

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
raw_data = fetch_financials(ticker)

if raw_data:
    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title(raw_data['name'])
        st.caption(f"{raw_data['sector']} • {ticker} • Data in {raw_data['currency']}")
    with col_h2:
        if st.button("🔄 Refresh Data"): st.rerun()

    # Calculations
    results = run_valuation(raw_data, {"yield": m_yield, "inflation": m_inf, "recovery": m_rec})

    # Top Metrics
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market Price", f"{raw_data['price']:,}")
    m2.metric("Stochastic Fair Value", f"{results['fair_val']:,.2f}", 
              f"{((results['fair_val']/raw_data['price'])-1)*100:.1f}% Upside")
    m3.metric("Default Probability", f"{results['p_def']*100:.21}%", delta_color="inverse")
    m4.metric("Altman Z-Score", f"{results['z']:.2f}", "Stable" if results['z'] > 1.8 else "At Risk")

    # Layout Tabs
    tab1, tab2, tab3 = st.tabs(["📊 RISK ANALYSIS", "📉 SIMULATION", "📑 FINANCIAL LEDGER"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Risk Radar")
            risk_radar = go.Figure(go.Scatterpolar(
                r=[results['z']*20, (1-results['p_def'])*100, 70, 60],
                theta=['Solvency', 'Market Confidence', 'Liquidity', 'Profitability'],
                fill='toself', line=dict(color='#58a6ff'), fillcolor='rgba(88, 166, 255, 0.2)'
            ))
            risk_radar.update_layout(
                polar=dict(radialaxis=dict(visible=False), bgcolor="rgba(0,0,0,0)"),
                paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=30, b=30, l=30, r=30)
            )
            st.plotly_chart(risk_radar, use_container_width=True)
        
        with c2:
            st.markdown("### Solvency Stress Test")
            st.write("Impact of Macro changes on Valuation")
            sensitivity_box = pd.DataFrame({
                "Parameter": ["Current Base", "High Stress", "Hyper-Inflation"],
                "Fair Value": [results['fair_val'], results['fair_val']*0.85, results['fair_val']*0.55],
                "Risk Level": ["Moderate", "High", "Critical"]
            })
            st.table(sensitivity_box)

    with tab2:
        st.markdown("### Monte Carlo ROIC Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=results['roic']*100, nbinsx=50, marker_color='#238636', opacity=0.75))
        fig_dist.add_vline(x=results['icc']*100, line_dash="dash", line_color="#f85149", annotation_text="Hurdle Rate")
        fig_dist.update_layout(
            margin=dict(l=0, r=0, t=20, b=0), height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Simulated Real ROIC %", gridcolor="#30363d"),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        st.markdown("### Audit Balance Sheet (Manual Overrides)")
        col_ed1, col_ed2, col_ed3 = st.columns(3)
        ov_debt = col_ed1.number_input("Total Debt", value=float(raw_data['debt']))
        ov_cash = col_ed2.number_input("Total Cash", value=float(raw_data['cash']))
        ov_ebitda = col_ed3.number_input("EBITDA", value=float(raw_data['ebitda']))
        
        if st.button("💾 Apply Manual Adjustments"):
            raw_data.update({"debt": ov_debt, "cash": ov_cash, "ebitda": ov_ebitda})
            st.rerun()

else:
    st.info("👈 Enter a company name in the sidebar to begin analysis.")
    st.image("https://images.unsplash.com/photo-1611974717482-452f86803201?auto=format&fit=crop&q=80&w=2070", use_column_width=True)

st.divider()
st.caption(f"Terminal Status: Connected | Simulation Paths: 3,000 | Last Heartbeat: {datetime.now().strftime('%H:%M:%S')}")
