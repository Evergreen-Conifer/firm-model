import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import requests

# --- PAGE SETUP ---
st.set_page_config(page_title="24/7 Clean Firm Power Modeler", layout="wide")
st.title("24/7 Clean Firm Power Sizing & PPA Calculator")
st.markdown("Compare Li-Ion vs. Advanced LDES architecture using real NREL solar data.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Location Data (NREL API)")
api_key = st.sidebar.text_input("NREL API Key", type="password", help="Get a free key at developer.nrel.gov/signup/")
lat = st.sidebar.number_input("Latitude", value=42.36)  # Default: Boston area
lon = st.sidebar.number_input("Longitude", value=-71.05)

st.sidebar.header("2. System Sizing (MW/MWh)")
load_mw = st.sidebar.number_input("Constant Load Target (MW)", value=10.0, step=1.0)
solar_mw = st.sidebar.slider("Solar Capacity (MW)", 0.0, 100.0, 40.0)
storage_mw = st.sidebar.slider("Storage Inverter/Power (MW)", 0.0, 50.0, 10.0)
storage_mwh = st.sidebar.slider("Storage Capacity (MWh)", 0.0, 1000.0, 100.0)

st.sidebar.header("3. Technology Profile")
# Anonymized the technology name
tech_type = st.sidebar.selectbox("Storage Technology", ["Advanced LDES (RFC + e-Fuel)", "Lithium-Ion"])
rte = st.sidebar.slider("Round Trip Efficiency (%)", 40, 95, 70 if "LDES" in tech_type else 85) / 100.0

st.sidebar.header("4. Financials (CAPEX & OPEX)")
solar_capex_kw = st.sidebar.number_input("Solar $/kW", value=1000)
st_capex_kw = st.sidebar.number_input("Storage $/kW", value=300)
# Default values are kept generic
st_capex_kwh = st.sidebar.number_input("Storage $/kWh", value=25 if "LDES" in tech_type else 200)
green_fuel_cost = st.sidebar.number_input("Green Fuel ($/MWh equivalent)", value=150)

st.sidebar.header("5. Project Returns")
target_irr = st.sidebar.slider("Target Unlevered IRR (%)", 5.0, 20.0, 10.0) / 100.0
itc_percent = st.sidebar.slider("ITC Tax Credit (%)", 0.0, 50.0, 40.0) / 100.0

# --- NREL API FETCHER (Cached for speed) ---
@st.cache_data(show_spinner="Fetching Weather Data from NREL...")
def get_nrel_solar_1mw(api_key, lat, lon):
    if not api_key:
        return None
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"
    params = {
        'api_key': api_key,
        'lat': lat,
        'lon': lon,
        'system_capacity': 1000, 
        'azimuth': 180,
        'tilt': lat,
        'array_type': 1,
        'module_type': 0,
        'losses': 14,
        'timeframe': 'hourly'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        ac_watts = np.array(data['outputs']['ac'])
        return ac_watts / 1_000_000
    else:
        st.sidebar.error(f"API Error: {response.status_code}")
        return None

# --- SIMULATION ENGINE ---
if not api_key:
    st.warning("Please enter your NREL API Key in the sidebar to run the simulation using real historical weather data.")
    st.stop()

solar_profile_1mw = get_nrel_solar_1mw(api_key, lat, lon)
if solar_profile_1mw is None:
    st.stop()

solar_gen = solar_profile_1mw * solar_mw
load = np.full(8760, load_mw)

# Dispatch Variables
soc = np.zeros(8760)
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
            if "LDES" in tech_type:
                # RFC acts as fuel cell, burns green fuel to meet unmet load
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
