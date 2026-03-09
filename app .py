import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

# --- PAGE SETUP ---
st.set_page_config(page_title="24/7 Clean Firm Power Modeler", layout="wide")
st.title("24/7 Clean Firm Power Sizing & PPA Calculator")
st.markdown("Compare Li-Ion vs. Noon (RSOFC) architecture for 24/7 firm load matching.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. System Sizing (MW/MWh)")
load_mw = st.sidebar.number_input("Constant Load Target (MW)", value=10.0, step=1.0)
solar_mw = st.sidebar.slider("Solar Capacity (MW)", 0.0, 100.0, 40.0)
storage_mw = st.sidebar.slider("Storage Inverter/Power (MW)", 0.0, 50.0, 10.0)
storage_mwh = st.sidebar.slider("Storage Capacity (MWh)", 0.0, 1000.0, 100.0)

st.sidebar.header("2. Technology Profile")
tech_type = st.sidebar.selectbox("Storage Technology", ["Noon (RSOFC + Green Fuel)", "Lithium-Ion"])
rte = st.sidebar.slider("Round Trip Efficiency (%)", 40, 95, 70 if "Noon" in tech_type else 85) / 100.0

st.sidebar.header("3. Financials (CAPEX & OPEX)")
solar_capex_kw = st.sidebar.number_input("Solar $/kW", value=1000)
st_capex_kw = st.sidebar.number_input("Storage $/kW", value=300)
st_capex_kwh = st.sidebar.number_input("Storage $/kWh", value=20 if "Noon" in tech_type else 200)
green_fuel_cost = st.sidebar.number_input("Green Fuel ($/MWh equivalent)", value=150)

st.sidebar.header("4. Project Returns")
target_irr = st.sidebar.slider("Target Unlevered IRR (%)", 5.0, 20.0, 10.0) / 100.0
itc_percent = st.sidebar.slider("ITC Tax Credit (%)", 0.0, 50.0, 40.0) / 100.0

# --- SIMULATION ENGINE (Simplified 8760 Hourly Model) ---
# Note: In production, this will use an NREL API for 3-5 year weather data.
# For this prototype, we generate a synthetic solar curve (bell curve during day).
hours = np.arange(8760)
solar_cf = np.where((hours % 24 > 7) & (hours % 24 < 19), np.sin(np.pi * (hours % 24 - 7) / 12), 0) * 0.25 # Avg ~20% CF
solar_gen = solar_mw * solar_cf
load = np.full(8760, load_mw)

# Dispatch Variables
soc = np.zeros(8760)  # State of Charge
fuel_used_mwh = np.zeros(8760)
grid_shortfall = np.zeros(8760)
current_soc = 0.0

for i in range(8760):
    net_load = load[i] - solar_gen[i]
    
    if net_load < 0: # Excess Solar -> Charge Battery
        charge_amount = min(-net_load, storage_mw, (storage_mwh - current_soc) / rte)
        current_soc += charge_amount * rte
    elif net_load > 0: # Deficit -> Discharge Battery or Use Fuel
        discharge_amount = min(net_load, storage_mw, current_soc)
        current_soc -= discharge_amount
        unmet_load = net_load - discharge_amount
        
        if unmet_load > 0:
            if "Noon" in tech_type:
                # Noon acts as fuel cell, burns green fuel to meet unmet load
                fuel_used_mwh[i] = unmet_load
            else:
                # Li-ion fails to meet load (Blackout)
                grid_shortfall[i] = unmet_load
                
    soc[i] = current_soc

# --- FINANCIAL CALCULATOR ---
total_solar_capex = solar_mw * 1000 * solar_capex_kw
total_storage_capex = (storage_mw * 1000 * st_capex_kw) + (storage_mwh * 1000 * st_capex_kwh)
gross_capex = total_solar_capex + total_storage_capex
net_capex = gross_capex * (1 - itc_percent)

annual_fuel_cost = np.sum(fuel_used_mwh) * green_fuel_cost
annual_energy_delivered = load_mw * 8760 - np.sum(grid_shortfall)
reliability = annual_energy_delivered / (load_mw * 8760)

# Calculate Required Annual Revenue to hit Target IRR over 20 years
# PMT formula: Calculates the constant payment needed to achieve the IRR
annual_rev_req_net = -npf.pmt(target_irr, 20, net_capex) + annual_fuel_cost
annual_rev_req_gross = -npf.pmt(target_irr, 20, gross_capex) + annual_fuel_cost

ppa_net = annual_rev_req_net / annual_energy_delivered if annual_energy_delivered > 0 else 0
ppa_gross = annual_rev_req_gross / annual_energy_delivered if annual_energy_delivered > 0 else 0

# --- DASHBOARD OUTPUT ---
st.header("System Performance & Economics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("System Reliability", f"{reliability*100:.1f}%", "Must be 100% for Firm Power")
col2.metric("Gross CAPEX", f"${gross_capex/1e6:.1f}M")
col3.metric("Required PPA (No ITC)", f"${ppa_gross:.2f} / MWh")
col4.metric(f"Required PPA ({itc_percent*100:.0f}% ITC)", f"${ppa_net:.2f} / MWh")

st.divider()

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Cost Breakdown")
    st.write(f"**Solar CAPEX:** ${total_solar_capex/1e6:.1f}M")
    st.write(f"**Storage CAPEX:** ${total_storage_capex/1e6:.1f}M")
    st.write(f"**Annual Green Fuel Cost:** ${annual_fuel_cost/1e6:.2f}M / year")
    st.write(f"**Total Green Fuel Needed:** {np.sum(fuel_used_mwh):,.0f} MWh/yr")

with col_b:
    st.subheader("Summer Week Dispatch Profile")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(solar_gen[4000:4168], label="Solar Gen (MW)", color='orange')
    ax.plot(load[4000:4168], label="Load Target (MW)", color='black', linestyle='--')
    ax.fill_between(range(168), 0, soc[4000:4168] / storage_mwh * load_mw, label="Battery SOC (%)", color='blue', alpha=0.3)
    if "Noon" in tech_type:
        ax.bar(range(168), fuel_used_mwh[4000:4168], label="Green Fuel Used (MW)", color='green', alpha=0.5)
    ax.legend()
    st.pyplot(fig)
