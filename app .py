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
    st.markdown("Adjust parameters to find the optimal architecture.")
    
    st.header("1. Location & Load")
    # Check if key is in the secure vault; if not, show the input box
    if "NREL_API_KEY" in st.secrets:
        api_key = st.secrets["NREL_API_KEY"]
        st.success("✅ NREL API Key securely loaded.")
    else:
        api_key = st.text_input("NREL API Key", type="password")
    lat = st.number_input("Latitude", value=42.36)
    lon = st.number_input("Longitude", value=-71.05)
    load_mw = st.number_input("Firm Load target (MW)", value=10.0, step=1.0)
    target_reliability = st.slider("Target Reliability (%)", 80.0, 100.0, 100.0) / 100.0
    
    st.header("2. Firming & Storage")
    rte = st.slider("Storage Round Trip Efficiency (%)", 30, 95, 45) / 100.0
    
    fuel_type = st.selectbox("Firming Fuel Type", ["Biomethanol (Advanced LDES/RSOFC)", "Custom Green Fuel", "No Fuel (Battery Only)"])
    
    if fuel_type == "Biomethanol (Advanced LDES/RSOFC)":
        use_fuel = True
        fuel_cost_tonne = st.number_input("Biomethanol Cost ($/tonne)", value=600.0, step=50.0)
        fuel_mwh_per_tonne = 5.53
    elif fuel_type == "Custom Green Fuel":
        use_fuel = True
        fuel_cost_tonne = st.number_input("Fuel Cost ($/tonne)", value=500.0)
        fuel_mwh_per_tonne = st.number_input("Fuel Energy Density (MWh/tonne)", value=15.0)
    else:
        use_fuel = False
        fuel_cost_tonne = 0
        fuel_mwh_per_tonne = 1
        
    st.header("3. CAPEX ($/kW or $/kWh)")
    solar_capex_kw = st.number_input("Solar CAPEX ($/kW)", value=1000)
    wind_capex_kw = st.number_input("Wind CAPEX ($/kW)", value=1400)
    
    st.markdown("**Storage Cost Structure**")
    st_capex_kw = st.number_input("Power Block CAPEX ($/kW)", value=2000)
    st_capex_kwh = st.number_input("Energy Block CAPEX ($/kWh)", value=10)
    
    st.header("4. OPEX & Financials")
    solar_opex_kw = st.number_input("Solar O&M ($/kW-yr)", value=15)
    wind_opex_kw = st.number_input("Wind O&M ($/kW-yr)", value=40)
    st_opex_kw = st.number_input("Storage O&M ($/kW-yr)", value=10)
    
    target_irr = st.slider("Unlevered IRR (%)", 5.0, 20.0, 10.0) / 100.0
    itc_percent = st.slider("ITC Tax Credit (%)", 0.0, 60.0, 40.0) / 100.0
    
    st.divider()
    run_opt = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

# --- HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_weather_profiles(api_key, lat, lon):
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    params = {'api_key': api_key, 'lat': lat, 'lon': lon, 'system_capacity': 1000, 
              'azimuth': 180, 'tilt': lat, 'array_type': 1, 'module_type': 0, 'losses': 14, 'timeframe': 'hourly'}
    res = requests.get(url, params=params)
    if res.status_code == 200:
        solar_1mw = np.array(res.json()['outputs']['ac']) / 1_000_000
    else:
        return None, None
        
    hours = np.arange(8760)
    diurnal = np.cos((hours % 24 - 2) * np.pi / 12) * -0.15
    seasonal = np.cos((hours - 4380) * np.pi / 4380) * 0.20
    wind_1mw = np.clip(0.35 + diurnal + seasonal + np.random.normal(0, 0.1, 8760), 0, 1.0)
    
    return solar_1mw, wind_1mw

def simulate_dispatch(s_mw, w_mw, st_mw, st_mwh, solar_1mw, wind_1mw, load_mw, rte, use_fuel):
    solar_gen = s_mw * solar_1mw
    wind_gen = w_mw * wind_1mw
    total_gen = solar_gen + wind_gen
    load = np.full(8760, load_mw)
    
    soc = np.zeros(8760)
    fuel_used_mwh = np.zeros(8760)
    unmet_load = np.zeros(8760)
    current_soc = 0.0
    
    for i in range(8760):
        net_load = load[i] - total_gen[i]
        if net_load < 0: 
            charge = min(-net_load, st_mw, (st_mwh - current_soc) / rte)
            current_soc += charge * rte
        elif net_load > 0: 
            discharge = min(net_load, st_mw, current_soc)
            current_soc -= discharge
            shortfall = net_load - discharge
            if shortfall > 0:
                if use_fuel:
                    fuel_used_mwh[i] = shortfall
                else:
                    unmet_load[i] = shortfall
        soc[i] = current_soc
        
    energy_delivered = (load_mw * 8760) - np.sum(unmet_load)
    reliability = energy_delivered / (load_mw * 8760)
    
    return reliability, np.sum(fuel_used_mwh), soc, total_gen, unmet_load, fuel_used_mwh

# --- MAIN PAGE OUTPUTS ---
st.title("📊 24/7 Clean Firm Power Outputs")

if run_opt:
    if not api_key:
        st.error("Please enter an NREL API Key in the sidebar.")
        st.stop()
        
    with st.spinner("Simulating extreme boundaries to find optimal CAPEX/OPEX balance..."):
        solar_1mw, wind_1mw = get_weather_profiles(api_key, lat, lon)
        if solar_1mw is None:
            st.error("Invalid API Key or Location.")
            st.stop()
            
        # Extreme grid search constraints
        s_grid = np.linspace(load_mw, load_mw * 12, 8) 
        w_grid = np.linspace(load_mw, load_mw * 12, 8)
        st_pwr_grid = [load_mw] 
        st_cap_grid = np.linspace(0, load_mw * 350, 15) 
        
        best_ppa_net = float('inf')
        best_system = None
        total_load_annual = load_mw * 8760
        
        for s in s_grid:
            for w in w_grid:
                for st_p in st_pwr_grid:
                    for st_c in st_cap_grid:
                        rel, fuel_mwh, _, _, _, _ = simulate_dispatch(s, w, st_p, st_c, solar_1mw, wind_1mw, load_mw, rte, use_fuel)
                        
                        if rel >= target_reliability:
                            capex_gross = (s * 1000 * solar_capex_kw) + (w * 1000 * wind_capex_kw) + \
                                          (st_p * 1000 * st_capex_kw) + (st_c * 1000 * st_capex_kwh)
                            capex_net = capex_gross * (1 - itc_percent)
                            
                            annual_opex = (s * 1000 * solar_opex_kw) + (w * 1000 * wind_opex_kw) + (st_p * 1000 * st_opex_kw)
                            annual_fuel_cost = (fuel_mwh / fuel_mwh_per_tonne) * fuel_cost_tonne if use_fuel else 0
                            
                            # Calculate PPAs
                            rev_req_gross = -npf.pmt(target_irr, 20, capex_gross) + annual_opex + annual_fuel_cost
                            ppa_gross = rev_req_gross / (total_load_annual * rel)
                            
                            rev_req_net = -npf.pmt(target_irr, 20, capex_net) + annual_opex + annual_fuel_cost
                            ppa_net = rev_req_net / (total_load_annual * rel)
                            
                            if ppa_net < best_ppa_net:
                                best_ppa_net = ppa_net
                                best_system = {
                                    's_mw': s, 'w_mw': w, 'st_mw': st_p, 'st_mwh': st_c,
                                    'rel': rel, 'fuel_mwh': fuel_mwh, 'capex_gross': capex_gross, 
                                    'capex_net': capex_net, 'ppa_gross': ppa_gross, 'ppa_net': ppa_net,
                                    'annual_fuel_cost': annual_fuel_cost
                                }

        if best_system is None:
            st.error("Could not reach target reliability even with massive overbuilds. You must allow fuel or lower the target.")
            st.stop()
            
        rel, fuel_mwh, soc, total_gen, unmet, fuel_arr = simulate_dispatch(
            best_system['s_mw'], best_system['w_mw'], best_system['st_mw'], 
            best_system['st_mwh'], solar_1mw, wind_1mw, load_mw, rte, use_fuel
        )
        
        # --- FINANCIAL METRICS ---
        st.subheader("Financial Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PPA (No ITC)", f"${best_system['ppa_gross']:.2f} / MWh")
        c2.metric(f"PPA (With {itc_percent*100:.0f}% ITC)", f"${best_system['ppa_net']:.2f} / MWh")
        c3.metric("Gross CAPEX", f"${best_system['capex_gross']/1e6:.1f}M")
        c4.metric("Net CAPEX (Post-ITC)", f"${best_system['capex_net']/1e6:.1f}M")
        
        st.divider()
        
        # --- PHYSICAL ARCHITECTURE ---
        st.subheader("Optimized Physical Architecture")
        phys1, phys2, phys3, phys4 = st.columns(4)
        phys1.metric("Solar Required", f"{best_system['s_mw']:.1f} MW", f"~{best_system['s_mw']*6:.0f} acres", delta_color="off")
        phys2.metric("Wind Required", f"{best_system['w_mw']:.1f} MW", f"~{best_system['w_mw']*60:.0f} acres", delta_color="off")
        duration = best_system['st_mwh'] / best_system['st_mw'] if best_system['st_mw'] > 0 else 0
        phys3.metric("Storage Required", f"{best_system['st_mw']:.1f} MW", f"{duration:.0f} hours ({best_system['st_mwh']:.0f} MWh)", delta_color="off")
        
        tonnes = best_system['fuel_mwh']/fuel_mwh_per_tonne if use_fuel else 0
        phys4.metric("Fuel Needed", f"{tonnes:,.0f} tonnes/yr" if use_fuel else "N/A", f"${best_system['annual_fuel_cost']/1e6:.2f}M / yr OPEX" if use_fuel else "", delta_color="off")
        
        st.divider()

# --- DATE TRACKING & DROUGHTS ---
        date_rng = pd.date_range(start='2025-01-01', periods=8760, freq='h')
        
        df_dispatch = pd.DataFrame({
            'Date': date_rng,
            'SOC_MWh': soc,
            'Fuel_MWh': fuel_arr,
            'Unmet_MWh': unmet
        })
        
        if use_fuel and tonnes > 0:
            st.subheader("⚠️ Top 5 Energy Droughts (Fuel Usage)")
            df_daily_fuel = df_dispatch.groupby(df_dispatch['Date'].dt.date)['Fuel_MWh'].sum().reset_index()
            top_droughts = df_daily_fuel[df_daily_fuel['Fuel_MWh'] > 0].sort_values(by='Fuel_MWh', ascending=False).head(5)
            
            # Format the dataframe for display
            top_droughts.columns = ['Date', 'Fuel Burned (MWh)']
            top_droughts['Est. Tonnes Used'] = top_droughts['Fuel Burned (MWh)'] / fuel_mwh_per_tonne
            st.dataframe(top_droughts.style.format({'Fuel Burned (MWh)': '{:.1f}', 'Est. Tonnes Used': '{:.1f}'}), use_container_width=True)
            
        elif not use_fuel and np.sum(unmet) > 0:
            st.subheader("⚠️ Top 5 Grid Failures (Unmet Load)")
            df_daily_fail = df_dispatch.groupby(df_dispatch['Date'].dt.date)['Unmet_MWh'].sum().reset_index()
            top_fails = df_daily_fail[df_daily_fail['Unmet_MWh'] > 0].sort_values(by='Unmet_MWh', ascending=False).head(5)
            top_fails.columns = ['Date', 'Unmet Load (MWh)']
            st.dataframe(top_fails.style.format({'Unmet Load (MWh)': '{:.1f}'}), use_container_width=True)

        # --- ANNUAL GRAPHS ---
        st.subheader("Annual Operational Profile (8760 Hours)")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: State of Charge
        soc_percent = (soc / best_system['st_mwh'] * 100) if best_system['st_mwh'] > 0 else np.zeros(8760)
        ax1.plot(date_rng, soc_percent, color='blue', linewidth=0.5)
        ax1.fill_between(date_rng, 0, soc_percent, color='blue', alpha=0.2)
        ax1.set_ylabel("Storage SOC (%)")
        ax1.set_title("Storage Inventory Throughout the Year")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fuel Dispatch
        if use_fuel:
            ax2.bar(date_rng, fuel_arr, color='red', width=0.05)
            ax2.set_ylabel("Fuel Burned (MWh/hr)")
            ax2.set_title("Firming Fuel Dispatch Events")
        else:
            ax2.bar(date_rng, unmet, color='black', width=0.05)
            ax2.set_ylabel("Blackout (MWh/hr)")
            ax2.set_title("Unmet Load Events (No Fuel)")
            
        ax2.grid(True, alpha=0.3)
        
        # Format X-axis for months
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Enter your parameters in the sidebar and click **Run Optimization**.")
