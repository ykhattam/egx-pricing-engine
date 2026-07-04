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
        
        # Core Indicators
        data = {
            "name": info.get("longName", ticker),
            "price": info.get("currentPrice") or info.get("previousClose") or 1.0,
            "currency": info.get("currency", "USD"),
            "sector": info.get("sector", "N/A"),
            
            # Valuation Multiples
            "pe_ratio": info.get("trailingPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            
            # Solvency & Liquidity
            "debt_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "interest_coverage": info.get("ebitda", 0) / max(info.get("totalDebt", 1), 1), # Simplified
            
            # Profitability
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "profit_margin": info.get("profitMargins"),
            "ebitda_margin": info.get("ebitdaMargins"),
            
            # Risk/Market
            "beta": info.get("beta", 1.0),
            "short_ratio": info.get("shortRatio"),
            "dividend_yield": info.get("dividendYield"),
            
            # Raw for Math
            "ebitda": info.get("ebitda", 1.0),
            "total_debt": info.get("totalDebt", 0.0),
            "total_cash": info.get("totalCash", 0.0),
            "total_assets": info.get("totalAssets", 1.0),
            "revenue": info.get("totalRevenue", 1.0)
        }
        return data
    except: return None

def calculate_stochastic_metrics(d, m):
    # Default Risk Calculation
    net_debt = max(d['total_debt'] - d['total_cash'], 0.1)
    leverage = net_debt / max(d['ebitda'], 1.0)
    p_def = 1 / (1 + np.exp(-0.4 * (leverage - 4.0))) # Logistic Default Curve
    
    # Fair Value based on default-adjusted recovery
    fair_val = d['price'] * (1 - p_def * (1 - m['recovery']))
    
    # ROIC Simulation (Monte Carlo)
    sim_roic = np.random.normal(d.get('roa', 0.1) * 1.5, 0.05, 2000) / (1 + m['inflation'])
    icc = m['yield'] + (d['beta'] * 0.075) # Cost of Capital
    
    return {"p_def": p_def, "fair_val": fair_val, "sim_roic": sim_roic, "icc": icc}

# ==========================================
# 3. SIDEBAR & SEARCH
# ==========================================
with st.sidebar:
    st.title("🏛️ ANALYST TERMINAL")
    query = st.text_input("Find Asset (e.g. Apple, CIB, Tesla)", value="Commercial International Bank")
    options = search_ticker(query)
    ticker = options[0].split(" | ")[0] if options else query.upper()
    if options: st.selectbox("Confirm Selection", options, key="ticker_select")
    
    st.markdown("---")
    st.write("🌍 **MACRO ENVIRONMENT**")
    m_yield = st.number_input("Risk Free Rate (Sovereign %)", value=18.0) / 100
    m_infl = st.number_input("Expected Inflation %", value=25.0) / 100
    m_rec = st.slider("Recovery in Default %", 10, 90, 30) / 100

# ==========================================
# 4. MAIN DASHBOARD CONTENT
# ==========================================
data = fetch_comprehensive_data(ticker)

if data:
    res = calculate_stochastic_metrics(data, {"yield": m_yield, "inflation": m_infl, "recovery": m_rec})
    
    # --- HEADER ---
    c_h1, c_h2 = st.columns([3, 1])
    with c_h1:
        st.title(data['name'])
        st.caption(f"SECTOR: {data['sector']} | CURRENCY: {data['currency']} | TICKER: {ticker}")
    
    # --- TOP KPI INDICATORS ---
    st.markdown("### 🔑 Critical Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Market Price", f"{data['price']:,}", help="Current trading price on the exchange.")
    k2.metric("Stochastic Fair Value", f"{res['fair_val']:,.2f}", f"{((res['fair_val']/data['price'])-1)*100:.1f}%", help="The theoretical value adjusted for local macro risks and default probability.")
    k3.metric("Default Risk", f"{res['p_def']*100:.1f}%", help="The mathematical probability of the company failing to meet debt obligations based on Net Debt/EBITDA.")
    k4.metric("Cost of Capital (ICC)", f"{res['icc']*100:.1f}%", help="The minimum return an investor expects for providing capital, adjusted for the stock's Beta.")

    # --- TABS FOR ORGANIZED ANALYSIS ---
    t1, t2, t3, t4 = st.tabs(["📊 VALUATION & PROFIT", "🛡️ SOLVENCY & RISK", "📈 SIMULATION", "📖 INDICATOR DICTIONARY"])

    with t1:
        st.markdown("#### Profitability & Growth Indicators")
        c1, c2, c3 = st.columns(3)
        c1.metric("P/E Ratio", f"{data['pe_ratio'] if data['pe_ratio'] else 'N/A'}", help="Price-to-Earnings: How much you pay for $1 of profit.")
        c2.metric("P/S Ratio", f"{data['ps_ratio']:.2f}" if data['ps_ratio'] else "N/A", help="Price-to-Sales: Useful for companies with low earnings but high revenue.")
        c3.metric("ROE", f"{(data['roe']*100):.1f}%" if data['roe'] else "N/A", help="Return on Equity: How efficiently the company uses shareholders' money.")
        
        c4, c5, c6 = st.columns(3)
        c4.metric("EBITDA Margin", f"{(data['ebitda_margin']*100):.1f}%" if data['ebitda_margin'] else "N/A", help="Operating profitability as a % of total revenue.")
        c5.metric("EV/EBITDA", f"{data['ev_ebitda']:.1f}x" if data['ev_ebitda'] else "N/A", help="Enterprise Value to EBITDA: The standard for 'takeover' valuation.")
        c6.metric("Div. Yield", f"{(data['dividend_yield']*100):.1f}%" if data['dividend_yield'] else "0.0%", help="Annual dividend payments divided by stock price.")

    with t2:
        st.markdown("#### Balance Sheet & Liquidity Indicators")
        s1, s2, s3 = st.columns(3)
        s1.metric("Debt/Equity", f"{data['debt_equity']:.1f}%" if data['debt_equity'] else "N/A", help="Total Debt divided by Shareholders Equity. Lower is usually safer.")
        s2.metric("Current Ratio", f"{data['current_ratio']:.2f}x" if data['current_ratio'] else "N/A", help="Current Assets / Current Liabilities. >1.0 means the company can pay short-term bills.")
        s3.metric("Quick Ratio", f"{data['quick_ratio']:.2f}x" if data['quick_ratio'] else "N/A", help="Like Current Ratio but excludes inventory. A tougher test of liquidity.")

    with t3:
        st.markdown("#### Monte Carlo Output")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res['sim_roic']*100, marker_color='#58a6ff', nbinsx=40))
        fig.add_vline(x=res['icc']*100, line_dash="dash", line_color="#ff7b72", annotation_text="Hurdle Rate")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#8b949e", height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Insight: In {(res['sim_roic'] > res['icc']).mean()*100:.1f}% of future scenarios, this company generates returns above its cost of capital.")

    with t4:
        st.markdown("### 📖 Terminal Indicator Dictionary")
        st.write("""
        | Indicator | Category | Meaning |
        | :--- | :--- | :--- |
        | **P/E Ratio** | Valuation | High P/E usually means market expects high growth; Low P/E can mean undervaluation or trouble. |
        | **Beta** | Risk | Measure of volatility. Beta > 1 is more volatile than the market; Beta < 1 is more stable. |
        | **Net Debt/EBITDA** | Solvency | How many years of profit it takes to pay off debt. > 4.0x is usually an alarm for risk. |
        | **ROIC** | Efficiency | Return on Invested Capital. If ROIC > Cost of Capital, the company is 'Creating Value'. |
        | **Altman Z-Score** | Bankruptcy | A score based on 5 ratios. < 1.8 means High Distress, > 3.0 means 'Safe Zone'. |
        | **Shadow Inflation** | Macro | Used in this app to 'de-rate' future earnings. High inflation eats into real shareholder returns. |
        | **Stochastic Fair Value**| Proprietary | A calculated price that factors in the probability of a 'worst-case' scenario (Default). |
        """)

else:
    st.error("Missing Data for the selected ticker. Please try a different symbol (e.g. NVDA or AAPL).")

st.divider()
st.caption(f"Last Terminal Sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Logic: Stochastic Value-at-Risk Engine")

k4.metric("Altman Z-Score", f"{res['z_score']:.2f}", "Safe" if res['z_score'] > 2.6 else "Distressed")

    # --- TABS FOR ORGANIZED ANALYSIS ---
    t1, t2, t3, t4 = st.tabs(["📊 VALUATION & PROFIT", "🛡️ SOLVENCY & RISK", "📈 SIMULATION", "📖 INDICATOR DICTIONARY"])

    with t1:
        st.markdown("#### Performance Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("P/E Ratio", f"{data['pe_ratio'] if data['pe_ratio'] else 'N/A'}")
        c2.metric("P/B Ratio", f"{data['pb_ratio']:.2f}" if data['pb_ratio'] else "N/A")
        c3.metric("ROE", f"{(data['roe']*100):.1f}%" if data['roe'] else "N/A")

    with t2:
        st.markdown("#### Balance Sheet Health")
        s1, s2, s3 = st.columns(3)
        s1.metric("Debt/Equity", f"{data['debt_equity']:.1f}" if data['debt_equity'] else "N/A")
        s2.metric("Current Ratio", f"{data['current_ratio']:.2f}x" if data['current_ratio'] else "N/A")
        s3.metric("Int. Coverage", f"{res['interest_coverage']:.2f}x")

    with t3:
        st.markdown("#### Monte Carlo ROIC Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res['roic_array']*100, marker_color='#58a6ff', nbinsx=40))
        fig.add_vline(x=res['icc']*100, line_dash="dash", line_color="#ff7b72", annotation_text="Cost of Cap")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Economic Value Creation Probability: **{res['value_creation_prob']*100:.1f}%**")

    with t4:
        st.markdown("### 📖 Terminal Indicator Dictionary")
        st.write("""
        | Indicator | Category | Meaning |
        | :--- | :--- | :--- |
        | **Default Prob.** | Risk | Probability of default based on leverage and interest coverage. |
        | **ICC** | Hurdle Rate | Implied Cost of Capital; the return required to justify the investment. |
        | **Altman Z** | Solvency | Score indicating likelihood of bankruptcy. < 1.81 is distressed. |
        """)

else:
    st.error("Data could not be fetched for this ticker. Please check your spelling or try another asset.")

st.divider()
st.caption(f"Terminal Active | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
