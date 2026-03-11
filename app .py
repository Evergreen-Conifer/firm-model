import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

# --- PAGE SETUP ---
st.set_page_config(page_title="Firm Power Optimizer", layout="wide", initial_sidebar_state="expanded")

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.title("⚙️ System Inputs")
    
    if "NREL_API_KEY" in st.secrets:
        api_key = st.secrets["NREL_API_KEY"]
        st.success("✅ NREL API Key loaded.")
    else:
        api_key = st.text_input("NREL API Key", type="password")
        
    st.header("1. Project Details")
    state = st.text_input("State", value="TX")
    iso = st.selectbox("ISO", ["ERCOT", "MISO", "PJM", "CAISO", "SPP"])
    load_mw = st.number_input("Firm Load (MW)", value=100.0)
    target_rel = st.slider("Target Reliability (%)", 90.0, 100.0, 100.0) / 100.0
    
    st.header("2. Technology Selection")
    config_mode = st.radio("Resource Mix", ["Solar + Wind", "Solar Only", "Wind Only"])
    rte = st.slider("Storage RTE (%)", 30, 95, 60) / 100.0
    
    st.header("3. Financial Engineering")
    use_leverage = st.toggle("Apply Project Finance (Leverage)", value=True)
    if use_leverage:
        debt_pct = st.slider("Debt Fraction (%)", 50, 90, 75) / 100.0
        interest_rate = st.slider("Interest Rate (%)", 3.0, 10.0, 6.5) / 100.0
        equity_irr = st.slider("Target Equity IRR (%)", 8.0, 25.0, 12.0) / 100.0
        # Calculate WACC
        wacc = (debt_pct * interest_rate * (1 - 0.21)) + ((1 - debt_pct) * equity_irr)
        st.caption(f"Blended WACC: {wacc*100:.2f}%")
    else:
        wacc = st.slider("Unlevered IRR (%)", 5.0, 20.0, 10.0) / 100.0
        
    itc_percent = st.slider("ITC Tax Credit (%)", 30, 70, 40) / 100.0

    st.header("4. CAPEX Settings")
    solar_capex = st.number_input("Solar $/kW", value=900)
    wind_capex = st.number_input("Wind $/kW", value=1400)
    st_pwr_capex = st.number_input("Storage Power $/kW", value=2000)
    st_en_capex = st.number_input("Storage Energy $/kWh", value=10)

    st.divider()
    run_opt = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# --- HELPER FUNCTIONS (Same as before but with config_mode logic) ---
@st.cache_data(show_spinner=False)
def get_weather(api_key, lat, lon):
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    params = {'api_key': api_key, 'lat': lat, 'lon': lon, 'system_capacity': 1000, 'azimuth': 180, 'tilt': lat, 'array_type': 1, 'module_type': 0, 'losses': 14, 'timeframe': 'hourly'}
    res = requests.get(url, params=params).json()
    solar = np.array(res['outputs']['ac']) / 1e6
    # Synthetic wind (Night-peaking for Texas/Midwest)
    h = np.arange(8760)
    wind = np.clip(0.4 + 0.15*np.cos((h%24-2)*np.pi/12) + np.random.normal(0,0.1,8760), 0, 1)
    return solar, wind

def dispatch(s_mw, w_mw, st_mw, st_mwh, sol_1, win_1, load_mw, rte):
    gen = (s_mw * sol_1) + (w_mw * win_1)
    soc, unmet = np.zeros(8760), np.zeros(8760)
    cur = st_mwh * 0.5
    for i in range(8760):
        net = load_mw - gen[i]
        if net < 0:
            chg = min(-net, st_mw, (st_mwh - cur)/rte)
            cur += chg * rte
        else:
            dis = min(net, st_mw, cur)
            cur -= dis
            unmet[i] = net - dis
        soc[i] = cur
    rel = 1 - (np.sum(unmet) / (load_mw * 8760))
    return rel, soc, unmet

# --- MAIN ENGINE ---
if run_opt:
    with st.spinner("Optimizing with WACC and Leverage..."):
        s1, w1 = get_weather(api_key, 31.12, -97.41)
        
        # Adjust grid based on mode
        s_grid = [0] if config_mode == "Wind Only" else np.linspace(load_mw, load_mw*8, 8)
        w_grid = [0] if config_mode == "Solar Only" else np.linspace(load_mw, load_mw*8, 8)
        st_grid = np.linspace(load_mw*50, load_mw*450, 10)
        
        best = {'ppa': float('inf')}
        for s in s_grid:
            for w in w_grid:
                for c in st_grid:
                    rel, soc, unm = dispatch(s, w, load_mw, c, s1, w1, load_mw, rte)
                    if rel >= target_rel:
                        capex = (s*1000*solar_capex) + (w*1000*wind_capex) + (load_mw*1000*st_pwr_capex) + (c*1000*st_en_capex)
                        net_capex = capex * (1 - itc_percent)
                        opex = (s*15 + w*40 + load_mw*10) * 1000
                        # PPA Calculation using WACC
                        rev = -npf.pmt(wacc, 20, net_capex) + opex
                        ppa = rev / (load_mw * 8760)
                        if ppa < best['ppa']:
                            best = {'ppa': ppa, 's':s, 'w':w, 'c':c, 'rel':rel, 'soc':soc, 'capex':net_capex}

        if best['ppa'] == float('inf'):
            st.error("No solution found. Try lower reliability or more resources.")
        else:
            st.success(f"Optimal PPA Found: ${best['ppa']:.2f}/MWh")
            # Summary Table
            df = pd.DataFrame([{
                "ISO": iso, "PPA": f"${best['ppa']:.2f}", "Solar": f"{best['s']:.0f}MW", 
                "Wind": f"{best['w']:.0f}MW", "LDES": f"{best['c']/load_mw:.0f} hrs", "Rel": f"{best['rel']*100:.1f}%"
            }])
            st.table(df)
            
            # Annual Chart
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(best['soc']/best['c']*100, color='blue', alpha=0.8)
            ax.set_title("Annual Battery Inventory (SOC %)")
            st.pyplot(fig)
