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
st.set_page_config(layout="wide", page_title="Stochastic Pro Terminal", page_icon="🏛️")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0d1117; color: #c9d1d9; }
        .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
        div[data-testid="stExpander"] { border: 1px solid #30363d; background-color: #0d1117; }
        .stTabs [data-baseweb=tab] { font-size: 0.85rem; letter-spacing: 0.5px; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA DEEP-DIVE LAYER
# ==========================================
@st.cache_data(ttl=3600)
def search_ticker(query):
    if not query or len(query) < 2: return []
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = res.json().get('quotes', [])
        return [f"{i['symbol']} | {i.get('shortname', '')} ({i.get('exchDisp', '')})" for i in data]
    except: return []

@st.cache_data(ttl=3600)
def fetch_comprehensive_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "name": info.get("longName", ticker),
            "price": info.get("currentPrice") or info.get("previousClose") or 1.0,
            "currency": info.get("currency", "USD"),
            "sector": info.get("sector", "N/A"),
            "pe_ratio": info.get("trailingPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "ebitda": info.get("ebitda", 1.0),
            "total_debt": info.get("totalDebt", 0.0),
            "total_cash": info.get("totalCash", 0.0),
            "beta": info.get("beta", 1.0)
        }
    except: return None

def calculate_stochastic_metrics(d, m):
    net_debt = max(d['total_debt'] - d['total_cash'], 0.1)
    leverage = net_debt / max(d['ebitda'], 1.0)
    p_def = 1 / (1 + np.exp(-0.4 * (leverage - 4.0)))
    fair_val = d['price'] * (1 - p_def * (1 - m['recovery']))
    sim_roic = np.random.normal(d.get('roa', 0.1) * 1.5, 0.05, 2000) / (1 + m['inflation'])
    icc = m['yield'] + (d['beta'] * 0.075)
    return {"p_def": p_def, "fair_val": fair_val, "sim_roic": sim_roic, "icc": icc, "z_score": 3.0 - (p_def * 2)}

# ==========================================
# 3. SIDEBAR & SEARCH
# ==========================================
with st.sidebar:
    st.title("🏛️ ANALYST TERMINAL")
    query = st.text_input("Find Asset (e.g. Apple, CIB, Tesla)", value="COMI.CA")
    options = search_ticker(query)
    ticker = options[0].split(" | ")[0] if options else query.upper()
    if options: st.selectbox("Confirm Selection", options, key="ticker_select")
    
    st.markdown("---")
    m_yield = st.number_input("Risk Free Rate %", value=18.0) / 100
    m_infl = st.number_input("Expected Inflation %", value=25.0) / 100
    m_rec = st.slider("Recovery in Default %", 10, 90, 30) / 100

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
data = fetch_comprehensive_data(ticker)

if data:
    res = calculate_stochastic_metrics(data, {"yield": m_yield, "inflation": m_infl, "recovery": m_rec})
    
    st.title(data['name'])
    st.caption(f"TICKER: {ticker} | SECTOR: {data['sector']}")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Price", f"{data['price']:,}")
    k2.metric("Stochastic Fair Value", f"{res['fair_val']:,.2f}")
    k3.metric("Default Risk", f"{res['p_def']*100:.1f}%")
    k4.metric("Altman Z-Score", f"{res['z_score']:.2f}")

    t1, t2, t3, t4 = st.tabs(["📊 VALUATION", "🛡️ SOLVENCY", "📈 SIMULATION", "📖 DICTIONARY"])

    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("P/E Ratio", f"{data['pe_ratio'] or 'N/A'}")
        c2.metric("P/B Ratio", f"{data['pb_ratio'] or 'N/A'}")
        c3.metric("ROE", f"{(data['roe']*100 if data['roe'] else 0):.1f}%")

    with t2:
        s1, s2, s3 = st.columns(3)
        s1.metric("Debt/Equity", f"{data['debt_equity'] or 0:.1f}")
        s2.metric("Current Ratio", f"{data['current_ratio'] or 0:.2f}x")
        s3.metric("Int. Coverage", "High" if res['p_def'] < 0.2 else "Low")

    with t3:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res['sim_roic']*100, marker_color='#58a6ff'))
        fig.add_vline(x=res['icc']*100, line_color="#ff7b72")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        st.markdown("| Indicator | Meaning |\n| :--- | :--- |\n| P/E | Valuation |\n| ROE | Profitability |\n| Z-Score | Bankruptcy Risk |")

else:
    st.error("Data could not be fetched. Check ticker/internet connection.")

st.divider()
st.caption(f"Terminal Active | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
