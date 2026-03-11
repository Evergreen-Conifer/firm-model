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
        
    st.header("1. Location & Load")
    state = st.text_input("State", value="TX")
    iso = st.selectbox("ISO", ["ERCOT", "MISO", "PJM", "CAISO", "SPP"])
    lat = st.number_input("Latitude", value=31.12)
    lon = st.number_input("Longitude", value=-97.41)
    load_mw = st.number_input("Firm Load (MW)", value=100.0)
    target_rel = st.slider("Target Reliability (%)", 90.0, 100.0, 100.0) / 100.0
    
    st.header("2. Technology & Fuel")
    config_mode = st.radio("Resource Mix", ["Solar + Wind", "Solar Only", "Wind Only"])
    rte = st.slider("Storage RTE (%)", 30, 95, 60) / 100.0
    
    fuel_type = st.selectbox("Firming Fuel", ["No Fuel (Battery Only)", "Biomethanol (LDES/RSOFC)", "Custom Green Fuel"])
    if "Fuel" in fuel_type and "No Fuel" not in fuel_type:
        use_fuel = True
        fuel_cost_tonne = st.number_input("Fuel Cost ($/tonne)", value=600.0)
        fuel_density = 5.53 if "Biomethanol" in fuel_type else st.number_input("MWh/tonne", value=15.0)
    else:
        use_fuel, fuel_cost_tonne, fuel_density = False, 0, 1
    
    st.header("3. Financial Engineering")
    use_leverage = st.toggle("Apply Project Finance (Leverage)", value=True)
    if use_leverage:
        debt_pct = st.slider("Debt Fraction (%)", 50, 90, 75) / 100.0
        int_rate = st.slider("Interest Rate (%)", 3.0, 10.0, 6.5) / 100.0
        eq_irr = st.slider("Target Equity IRR (%)", 8.0, 25.0, 12.0) / 100.0
        wacc = (debt_pct * int_rate * (1 - 0.21)) + ((1 - debt_pct) * eq_irr)
        st.caption(f"Blended WACC: {wacc*100:.2f}%")
    else:
        wacc = st.slider("Unlevered IRR (%)", 5.0, 20.0, 10.0) / 100.0
        
    itc_percent = st.slider("ITC Tax Credit (%)", 30, 70, 40) / 100.0

    st.header("4. CAPEX & OPEX")
    solar_capex = st.number_input("Solar $/kW", value=900)
    wind_capex = st.number_input("Wind $/kW", value=1400)
    st_p_capex = st.number_input("Storage Power $/kW", value=2000)
    st_e_capex = st.number_input("Storage Energy $/kWh", value=10)
    
    st.markdown("**Annual O&M ($/kW-yr)**")
    s_om = st.number_input("Solar O&M", value=15.0)
    w_om = st.number_input("Wind O&M", value=40.0)
    st_om = st.number_input("Storage O&M", value=10.0)

    st.divider()
    run_opt = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# --- HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_weather(api_key, lat, lon):
    url = f"https://developer.nrel.gov/api/pvwatts/v8.json?api_key={api_key}&lat={lat}&lon={lon}&system_capacity=1000&azimuth=180&tilt={lat}&array_type=1&module_type=0&losses=14&timeframe=hourly"
    res = requests.get(url).json()
    sol = np.array(res['outputs']['ac']) / 1e6
    h = np.arange(8760)
    win = np.clip(0.35 + (np.cos((h%24-2)*np.pi/12)*-0.15) + (np.cos((h-4380)*np.pi/4380)*0.2) + np.random.normal(0,0.1,8760), 0, 1)
    return sol, win

def simulate(s_mw, w_mw, st_p_mw, st_e_mwh, sol_1, win_1, load_mw, rte, use_fuel):
    gen = (s_mw * sol_1) + (w_mw * win_1)
    def run_pass(start_soc):
        soc, fuel, unmet = np.zeros(8760), np.zeros(8760), np.zeros(8760)
        cur = start_soc
        for i in range(8760):
            net = load_mw - gen[i]
            if net < 0:
                chg = min(-net, st_p_mw, (st_e_mwh - cur)/rte)
                cur += chg * rte
            else: dis = min(net, st_p_mw, cur); cur -= dis; short = net - dis
            if short > 0:
                if use_fuel: fuel[i] = short
                else: unmet[i] = short
            soc[i] = cur
        return soc, fuel, unmet, cur
    _, _, _, end_soc = run_pass(st_e_mwh * 0.5)
    return run_pass(end_soc)

# --- EXECUTION ---
if run_opt:
    with st.spinner("Finding Optimal Scenario..."):
        s1, w1 = get_weather(api_key, lat, lon)
        s_grid = [0] if config_mode == "Wind Only" else np.linspace(load_mw, load_mw*10, 8)
        w_grid = [0] if config_mode == "Solar Only" else np.linspace(load_mw, load_mw*10, 8)
        st_grid = np.linspace(load_mw*20, load_mw*450, 12)

        best = {'ppa': float('inf')}
        for s in s_grid:
            for w in w_grid:
                for c in st_grid:
                    soc, fuel, unm, _ = simulate(s, w, load_mw, c, s1, w1, load_mw, rte, use_fuel)
                    rel = 1 - (np.sum(unm)/(load_mw*8760))
                    if rel >= target_rel:
                        capex = (s*1000*solar_capex) + (w*1000*wind_capex) + (load_mw*1000*st_p_capex) + (c*1000*st_e_capex)
                        ann_fuel = (np.sum(fuel)/fuel_density) * fuel_cost_tonne
                        opex = (s*s_om + w*w_om + load_mw*st_om)*1000 + ann_fuel
                        rev = -npf.pmt(wacc, 20, capex*(1-itc_percent)) + opex
                        ppa = rev / (load_mw * 8760)
                        if ppa < best['ppa']:
                            best = {'ppa':ppa, 's':s, 'w':w, 'c':c, 'rel':rel, 'soc':soc, 'fuel':fuel, 'capex':capex, 'ann_fuel':ann_fuel}

        if best['ppa'] == float('inf'):
            st.error("No solution found.")
        else:
            st.success("Optimization Complete!")
            # 1. Summary Table
            st.dataframe(pd.DataFrame([{
                "ISO": iso, "PPA": f"${best['ppa']:.2f}", "Solar": f"{best['s']:.0f}MW", "Wind": f"{best['w']:.0f}MW", 
                "Storage": f"{best['c']/load_mw:.0f} hrs", "Fuel (t)": f"{np.sum(best['fuel'])/fuel_density:,.0f}"
            }]), hide_index=True)
            
            st.divider()
            # 2. Metric Tiles
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("PPA ($/MWh)", f"${best['ppa']:.2f}")
            m2.metric("Net CAPEX ($M)", f"${(best['capex']*(1-itc_percent))/1e6:.1f}M")
            m3.metric("Storage Duration", f"{best['c']/load_mw:.0f} hours")
            m4.metric("Achieved CFE", f"{best['rel']*100:.2f}%")
            
            st.divider()
            # 3. Drought Forensics
            dr_cols = st.columns([2, 1])
            with dr_cols[0]:
                st.subheader("Annual Operational Profile")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                dates = pd.date_range("2025-01-01", periods=8760, freq="H")
                ax1.plot(dates, best['soc']/best['c']*100, color='teal'); ax1.set_ylabel("SOC %")
                ax2.fill_between(dates, best['fuel'], color='red', alpha=0.5); ax2.set_ylabel("Fuel (MW)")
                st.pyplot(fig)
            
            with dr_cols[1]:
                st.subheader("⚠️ Top Energy Droughts")
                df_f = pd.DataFrame({'Date': dates, 'Fuel': best['fuel']}).groupby(pd.Grouper(key='Date', freq='D')).sum()
                st.table(df_f.sort_values(by='Fuel', ascending=False).head(5))
