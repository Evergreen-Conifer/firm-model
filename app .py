import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import requests

# --- PAGE SETUP ---
st.set_page_config(page_title="Firm Power Optimizer", layout="wide")
st.title("24/7 Clean Firm Power Optimizer")
st.markdown("Optimizes the trade-off between VRE overbuild (CAPEX) and LDES / Fuel usage (OPEX) to find the lowest PPA.")

# --- LAYOUT: Left (Outputs), Right (Inputs) ---
col_out, col_in = st.columns([3, 1])

with col_in:
    st.header("System Inputs")
    
    st.subheader("1. Load & Target")
    api_key = st.text_input("NREL API Key", type="password")
    lat = st.number_input("Latitude", value=42.36)
    lon = st.number_input("Longitude", value=-71.05)
    load_mw = st.number_input("Firm Load target (MW)", value=10.0, step=1.0)
    target_reliability = st.slider("Target Reliability (%)", 80.0, 100.0, 100.0) / 100.0
    
    st.subheader("2. Firming & Storage")
    # Low RTE mathematically forces larger solar/wind buildouts in the simulation
    rte = st.slider("Storage Round Trip Efficiency (%)", 30, 95, 45) / 100.0
    
    fuel_type = st.selectbox("Firming Fuel Type", ["No Fuel (Battery Only)", "Biomethanol (Advanced LDES/RSOFC)", "Custom Green Fuel"])
    
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
        
    st.subheader("3. CAPEX ($/kW or $/kWh)")
    solar_capex_kw = st.number_input("Solar CAPEX ($/kW)", value=1000)
    wind_capex_kw = st.number_input("Wind CAPEX ($/kW)", value=1400)
    
    # Decoupled LDES Pricing: $2000/kW + $10/kWh = $3000/kW at 100 hours ($30/kWh)
    st.markdown("**Storage Cost Structure**")
    st_capex_kw = st.number_input("Power Block CAPEX ($/kW)", value=2000, help="Cost of inverters, fuel cells, BOP.")
    st_capex_kwh = st.number_input("Energy Block CAPEX ($/kWh)", value=10, help="Marginal cost of additional hours (tanks, media).")
    
    st.subheader("4. OPEX ($/kW-year)")
    solar_opex_kw = st.number_input("Solar O&M", value=15)
    wind_opex_kw = st.number_input("Wind O&M", value=40)
    st_opex_kw = st.number_input("Storage O&M", value=10)
    
    st.subheader("5. Financials")
    target_irr = st.slider("Unlevered IRR (%)", 5.0, 20.0, 10.0) / 100.0
    run_opt = st.button("Run Optimization", type="primary", use_container_width=True)

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
            # RTE is applied here: Requires 1/RTE MWh from solar/wind to store 1 MWh in battery
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

# --- OPTIMIZATION ENGINE ---
with col_out:
    if run_opt:
        if not api_key:
            st.error("Please enter an NREL API Key to fetch weather data.")
            st.stop()
            
        with st.spinner("Simulating systems across all durations (up to 150 hours) to find optimal PPA..."):
            solar_1mw, wind_1mw = get_weather_profiles(api_key, lat, lon)
            if solar_1mw is None:
                st.error("Invalid API Key or Location.")
                st.stop()
                
            # Expanded grid to check 0 to 150 hours of storage (16 increments)
            s_grid = np.linspace(load_mw * 0.5, load_mw * 6, 8) 
            w_grid = np.linspace(load_mw * 0.5, load_mw * 6, 8)
            st_pwr_grid = [load_mw] 
            st_cap_grid = np.linspace(0, load_mw * 150, 16) 
            
            best_ppa = float('inf')
            best_system = None
            total_load_annual = load_mw * 8760
            
            for s in s_grid:
                for w in w_grid:
                    for st_p in st_pwr_grid:
                        for st_c in st_cap_grid:
                            rel, fuel_mwh, _, _, _, _ = simulate_dispatch(s, w, st_p, st_c, solar_1mw, wind_1mw, load_mw, rte, use_fuel)
                            
                            if rel >= target_reliability:
                                capex = (s * 1000 * solar_capex_kw) + (w * 1000 * wind_capex_kw) + \
                                        (st_p * 1000 * st_capex_kw) + (st_c * 1000 * st_capex_kwh)
                                
                                annual_opex = (s * 1000 * solar_opex_kw) + (w * 1000 * wind_opex_kw) + (st_p * 1000 * st_opex_kw)
                                annual_fuel_cost = (fuel_mwh / fuel_mwh_per_tonne) * fuel_cost_tonne if use_fuel else 0
                                
                                rev_req = -npf.pmt(target_irr, 20, capex) + annual_opex + annual_fuel_cost
                                ppa = rev_req / (total_load_annual * rel)
                                
                                if ppa < best_ppa:
                                    best_ppa = ppa
                                    best_system = {
                                        's_mw': s, 'w_mw': w, 'st_mw': st_p, 'st_mwh': st_c,
                                        'rel': rel, 'fuel_mwh': fuel_mwh, 'capex': capex, 'ppa': ppa,
                                        'annual_fuel_cost': annual_fuel_cost
                                    }

            if best_system is None:
                st.error("Could not reach target reliability. Try lowering the target or expanding boundaries.")
                st.stop()
                
            rel, fuel_mwh, soc, total_gen, unmet, fuel_arr = simulate_dispatch(
                best_system['s_mw'], best_system['w_mw'], best_system['st_mw'], 
                best_system['st_mwh'], solar_1mw, wind_1mw, load_mw, rte, use_fuel
            )
            
            # --- OUTPUT METRICS ---
            st.success("Optimization Complete! Found the lowest PPA across all durations.")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Optimized PPA", f"${best_system['ppa']:.2f} / MWh")
            c2.metric("Achieved Reliability", f"{best_system['rel']*100:.2f}%")
            c3.metric("Total CAPEX", f"${best_system['capex']/1e6:.1f}M")
            
            tonnes = best_system['fuel_mwh']/fuel_mwh_per_tonne if use_fuel else 0
            c4.metric("Fuel Needed", f"{tonnes:,.0f} tonnes/yr" if use_fuel else "N/A", 
                      f"${best_system['annual_fuel_cost']/1e6:.2f}M / yr OPEX" if use_fuel else None, 
                      delta_color="off")
            
            st.divider()
            
            st.subheader("Optimized Physical Architecture")
            phys1, phys2, phys3 = st.columns(3)
            phys1.write(f"**Solar Required:** {best_system['s_mw']:.1f} MW")
            phys1.write(f"*(~{best_system['s_mw']*6:.0f} acres)*")
            
            phys2.write(f"**Wind Required:** {best_system['w_mw']:.1f} MW")
            phys2.write(f"*(~{best_system['w_mw']*60:.0f} total acres)*")
            
            duration = best_system['st_mwh'] / best_system['st_mw'] if best_system['st_mw'] > 0 else 0
            phys3.write(f"**Storage Output:** {best_system['st_mw']:.1f} MW")
            phys3.write(f"**Storage Capacity:** {best_system['st_mwh']:.1f} MWh ({duration:.1f} hrs)")
            
            st.divider()
            
            st.subheader("System Dispatch Profile (First Week of January)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(total_gen[0:168], label="VRE Gen (Solar+Wind)", color='green')
            ax.plot(np.full(168, load_mw), label="Firm Load Target", color='black', linestyle='--')
            
            if best_system['st_mwh'] > 0:
                ax.fill_between(range(168), 0, soc[0:168] / best_system['st_mwh'] * load_mw, 
                                label="Storage SOC (%)", color='blue', alpha=0.3)
            if use_fuel:
                ax.bar(range(168), fuel_arr[0:168], label="Firming Fuel Dispatched", color='red', alpha=0.6)
            
            ax.set_ylabel("Megawatts (MW)")
            ax.set_xlabel("Hours")
            ax.legend(loc="upper right")
            st.pyplot(fig)
            
    else:
        st.info("👈 Enter your parameters on the right and click **Run Optimization**.")
