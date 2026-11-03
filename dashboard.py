import os
import base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math
import google.generativeai as genai

# ==============================================================================
# 1. PAGE SETUP & THEME CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="ZEUS Global Market Intelligence (ZGMI)",
    page_icon="assets/zeus_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (ZGMI Executive Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Top Header Branding */
    .zg-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(19, 166, 181, 0.2);
    }
    .zg-logo-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, #13A6B5 0%, #0B6F85 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 24px;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(19, 166, 181, 0.35);
    }
    .zg-title {
        font-size: 20px;
        font-weight: 700;
        color: #13A6B5;
        letter-spacing: 0.05em;
        margin: 0;
        line-height: 1.2;
    }
    .zg-subtitle {
        font-size: 12px;
        color: #8FA3C0;
        margin: 0;
    }

    /* KPI Cards */
    .kpi-container {
        background: linear-gradient(145deg, rgba(19, 31, 56, 0.75) 0%, rgba(11, 20, 38, 0.85) 100%);
        border: 1px solid rgba(36, 53, 87, 0.95);
        border-radius: 12px;
        padding: 12px 10px;
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        min-width: 0;
        overflow: hidden;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        border-color: #00F5FF;
        box-shadow: 0 6px 24px rgba(0, 245, 255, 0.15);
    }
    .kpi-header {
        font-size: 10px;
        font-weight: 700;
        color: #8FA3C0;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }
    .kpi-val {
        font-size: 19px;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
        line-height: 1.25;
    }
    .kpi-sub {
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }
    .kpi-pos { color: #5FD3AE; }
    .kpi-neg { color: #FF8A96; }
    .kpi-neu { color: #F0BC57; }

    /* Badges & Chips */
    .chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 4px;
    }
    .chip-ok { background: rgba(30, 158, 120, 0.18); color: #5FD3AE; border: 1px solid rgba(30,158,120,0.3); }
    .chip-wn { background: rgba(226, 176, 74, 0.18); color: #F0BC57; border: 1px solid rgba(226,176,74,0.3); }
    .chip-err { background: rgba(229, 112, 123, 0.18); color: #FF8A96; border: 1px solid rgba(229,112,123,0.3); }
    .chip-info { background: rgba(19, 166, 181, 0.18); color: #4FD1DE; border: 1px solid rgba(19,166,181,0.3); }

    /* Alert Banner */
    .alert-card {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #13A6B5;
        background: rgba(24, 40, 72, 0.5);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. SEED DATA & DETERMINISTIC ENGINE
# ==============================================================================
def hash_code(s):
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h

def rnd_val(s, k):
    x = (hash_code(s) + (k + 1) * 2654435761) & 0xFFFFFFFF
    x = ((x ^ (x >> 15)) * 2246822519) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 3266489917) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFFFF) / 4294967296.0

COUNTRIES_RAW = [
    ('CA','Canada',2,0,'North America','CAN'),
    ('US','United States',2,1,'North America','USA'),
    ('MX','Mexico',2,2,'Latin America','MEX'),
    ('CO','Colombia',3,3,'Latin America','COL'),
    ('PE','Peru',3,4,'Latin America','PER'),
    ('BR','Brazil',4,4,'Latin America','BRA'),
    ('CL','Chile',3,5,'Latin America','CHL'),
    ('AR','Argentina',4,5,'Latin America','ARG'),
    ('IE','Ireland',5,0,'Europe','IRL'),
    ('GB','United Kingdom',6,0,'Europe','GBR'),
    ('NO','Norway',7,0,'Europe','NOR'),
    ('SE','Sweden',8,0,'Europe','SWE'),
    ('FI','Finland',9,0,'Europe','FIN'),
    ('BE','Belgium',5,1,'Europe','BEL'),
    ('NL','Netherlands',6,1,'Europe','NLD'),
    ('DK','Denmark',7,1,'Europe','DNK'),
    ('PL','Poland',8,1,'Europe','POL'),
    ('PT','Portugal',4,2,'Europe','PRT'),
    ('ES','Spain',5,2,'Europe','ESP'),
    ('FR','France',6,2,'Europe','FRA'),
    ('DE','Germany',7,2,'Europe','DEU'),
    ('CZ','Czechia',8,2,'Europe','CZE'),
    ('RO','Romania',9,2,'Europe','ROU'),
    ('CH','Switzerland',6,3,'Europe','CHE'),
    ('IT','Italy',7,3,'Europe','ITA'),
    ('AT','Austria',8,3,'Europe','AUT'),
    ('GR','Greece',9,3,'Europe','GRC'),
        ('TR','Türkiye',10,3,'Europe','TUR'),
    ('IQ','Jordan-Iraq',11,3,'MENA','IRQ'),
    ('QA','Qatar',11,4,'MENA','QAT'),
    ('PS','Palestine',10,4,'MENA','PSE'),
    ('KW','Kuwait',11,4,'MENA','KWT'),
    ('AZ','Azerbaijan',11,3,'Europe','AZE'),
    ('BG','Bulgaria',9,2,'Europe','BGR'),
    ('AL','Albania-Slovakia',8,3,'Europe','ALB'),
    ('MA','Morocco',6,4,'MENA','MAR'),
    ('DZ','Algeria',7,4,'MENA','DZA'),
    ('EG','Egypt',9,4,'MENA','EGY'),
    ('SA','Saudi Arabia',10,4,'MENA','SAU'),
    ('AE','UAE',11,4,'MENA','ARE'),
    ('NG','Nigeria',7,5,'Africa','NGA'),
    ('KE','Kenya',9,5,'Africa','KEN'),
    ('ZA','South Africa',8,6,'Africa','ZAF'),
    ('PK','Pakistan',12,3,'Asia-Pacific','PAK'),
    ('IN','India',12,4,'Asia-Pacific','IND'),
    ('BD','Bangladesh',13,4,'Asia-Pacific','BGD'),
    ('CN','China',13,2,'Asia-Pacific','CHN'),
    ('KR','South Korea',14,3,'Asia-Pacific','KOR'),
    ('JP','Japan',15,2,'Asia-Pacific','JPN'),
    ('TW','Taiwan',15,3,'Asia-Pacific','TWN'),
    ('VN','Vietnam',14,4,'Asia-Pacific','VNM'),
    ('TH','Thailand',13,5,'Asia-Pacific','THA'),
    ('PH','Philippines',15,4,'Asia-Pacific','PHL'),
    ('MY','Malaysia',14,5,'Asia-Pacific','MYS'),
    ('ID','Indonesia',14,6,'Asia-Pacific','IDN'),
    ('AU','Australia',15,7,'Asia-Pacific','AUS'),
    ('NZ','New Zealand',16,7,'Asia-Pacific','NZL')
]

ACTIVE_DISTS_RAW = [
    # id, name, rev (USD thousand, 12 months), devShare, target %, margin %, growth %, pen %, risk, expiry, exclusivity, LOA
    ['IQ', 'SIRIUS PHARM COMPANY (Jordan & Iraq)', 164.3, 0.62, 106, 44.2, 18.5, 14.2, 'Low', '2027-12-31', 'Effective', 'Valid'],
    ['QA', 'INTERNATIONAL TECHNICAL LEGACY - ITL (Qatar)', 72.9, 0.55, 102, 46.5, 24.1, 12.8, 'Low', '2027-08-31', 'Effective', 'Valid'],
    ['KR', 'SJNU INC (South Korea)', 68.6, 0.64, 114, 48.0, 32.5, 8.4, 'Low', '2028-02-28', 'Effective', 'Valid'],
    ['PS', 'CONSULTANT MEDICAL COMPANY - CMC (Palestine)', 46.7, 0.68, 98, 42.1, 14.2, 10.5, 'Medium', '2027-06-30', 'Effective', 'Valid'],
    ['TR', 'GEN-SA GENEL SAĞLIK A.Ş. (Türkiye)', 33.1, 0.72, 104, 45.3, 16.8, 12.0, 'Low', '2027-10-31', 'Not granted', 'Valid'],
    ['SA', 'ASSISTED MEDICAL SOLUTION EST (Saudi Arabia)', 29.5, 0.65, 92, 43.4, 21.0, 7.8, 'Low', '2027-05-31', 'Not granted', 'Valid'],
    ['KW', 'ADVANCED TECHNOLOGY CO - ATC (Kuwait)', 26.3, 0.70, 95, 44.0, 19.4, 9.2, 'Low', '2027-09-30', 'Not granted', 'Valid'],
    ['AE', 'SELLCO TRADING INTL (UAE)', 23.2, 0.66, 89, 41.2, 15.3, 8.5, 'Medium', '2026-11-20', 'Not granted', 'In renewal'],
    ['BE', 'Ghent University - UGent (Belgium)', 16.1, 0.90, 100, 50.2, 28.0, 4.2, 'Low', '2028-06-30', 'Academic Partner', 'Valid'],
    ['AL', 'STEM GENE ART CLINIC & FERTICENT (Albania/Slovakia)', 18.8, 0.52, 94, 45.0, 10.8, 6.4, 'Low', '2027-04-30', 'Not granted', 'Valid'],
    ['AZ', 'RSM BAKÜ & Maternal Fetal Care (Azerbaijan)', 4.3, 0.10, 88, 40.5, 15.0, 5.2, 'Low', '2027-01-31', 'Not granted', 'Valid'],
    ['BG', 'MEYONA OOD (Bulgaria)', 1.6, 0.10, 85, 42.0, 12.0, 3.8, 'Low', '2027-09-30', 'Not granted', 'Valid'],
    ['PH', 'FERTILITY PHILIPPINES IVF (Philippines)', 1.2, 0.10, 90, 45.0, 25.0, 2.5, 'Low', '2027-08-31', 'Not granted', 'Valid'],
    ['GB', 'UK Clinical Reference Network (United Kingdom)', 0.0, 0.50, 0, 0.0, 0.0, 0.0, 'Low', '2026-12-31', 'Clinical Trial', 'Valid']
]

PROSPECTS_RAW = [
    ['DE', 'Rhein Fertility Systems (Germany)'],
    ['FR', 'Gaule Reproductive Partners (France)'],
    ['JP', 'Sakura Embryo Tech (Japan)'],
    ['US', 'American Reproductive Technologies (USA)'],
    ['IN', 'Deccan LifeSciences (India)'],
    ['ID', 'Nusantara Biomed (Indonesia)'],
    ['EG', 'Nile Biomedical Trading (Egypt)'],
    ['VN', 'Mekong MedTech (Vietnam)'],
    ['BR', 'Atlântica Reprodução (Brazil)']
]

PRODUCTS = [
    {'id': 'pulse', 'name': 'ZEUS Piezo Pulse Stimulator Device', 'share': 0.592, 'group': 'Device'},
    {'id': 'dish', 'name': 'Electro Dish (1x2 / 1x1)', 'share': 0.299, 'group': 'Consumables'},
    {'id': 'solution', 'name': 'ZEUS Electro Liquid Solution (Liquid & Powder)', 'share': 0.076, 'group': 'Consumables'},
    {'id': 'fusion', 'name': 'Zeus Electro Fusion SET', 'share': 0.023, 'group': 'Device'},
    {'id': 'cryo', 'name': 'Smart Cryo-Tools & Lab Accessories', 'share': 0.010, 'group': 'Consumables'}
]

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Helper to load excel sales data
@st.cache_data
def get_excel_sales_df():
    excel_file = "Zeus_Sales_Export_2026-11-03.xlsx"
    if os.path.exists(excel_file):
        try:
            df_sales = pd.read_excel(excel_file)
            df_sales["Date_dt"] = pd.to_datetime(df_sales["Date"], format="%d.%m.%Y", errors="coerce")
            df_sales["Year_int"] = df_sales["Date_dt"].dt.year
            df_sales["Month_int"] = df_sales["Date_dt"].dt.month
            return df_sales
        except Exception:
            pass
    return None

raw_sales_df = get_excel_sales_df()
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

@st.cache_data
def get_country_intelligence():
    def lvl(r, a, b):
        return 'Low' if r < a else ('Medium' if r < b else 'High')
        
    records = []
    for c in COUNTRIES_RAW:
        code = c[0]
        r = lambda k: rnd_val(code, k)
        clinics = round(18 + r(1) * 420)
        cycles = round(clinics * (280 + r(2) * 620) / 100) * 100
        gdp = round(60 + math.pow(r(5), 2) * 9000)
        pop = round(5 + math.pow(r(6), 1.5) * 400)
        gdp_pc = round((gdp / pop) * 1000) if pop > 0 else 0
        
        reg_types = ['Open', 'Moderate', 'Strict']
        reg = reg_types[min(2, math.floor(r(16) * 3))]
        
        imp_types = ['Open', 'Licensing required', 'Restricted']
        imp = imp_types[min(2, math.floor(r(19) * 3))]
        
        records.append({
            'code': code,
            'name': c[1],
            'region': c[4],
            'iso3': c[5],
            'clinics': clinics,
            'cycles': cycles,
            'score': round(38 + r(3) * 54),
            'risk': lvl(r(4), 0.4, 0.75),
            'gdp': gdp,
            'pop': pop,
            'gdp_pc': gdp_pc,
            'health_pct': round(3.0 + r(7) * 8.0, 1),
            'inflation_pct': round(1.5 + r(8) * 12.0, 1),
            'icsi_pct': round(55 + r(9) * 35),
            'pgt_pct': round(3 + r(10) * 28),
            'cryo_pct': round(20 + r(14) * 55),
            'competition': lvl(r(15), 0.33, 0.7),
            'regulation': reg,
            'fx_risk': lvl(r(17), 0.4, 0.75),
            'payment_risk': lvl(r(18), 0.45, 0.8),
            'import_cond': imp
        })
    return pd.DataFrame(records)

def get_base_distributors():
    df = get_excel_sales_df()
    if df is None:
        return []
        
    country_to_region = {
        'Turkey': 'Europe', 'Belgium': 'Europe', 'Albania-Slovakia': 'Europe',
        'Bulgaria': 'Europe', 'Azerbaijan': 'Europe', 'United Kingdom': 'Europe',
        'Jordan-Iraq': 'MENA', 'Jordan': 'MENA', 'Qatar': 'MENA',
        'Palestine': 'MENA', 'Saudi Arabia': 'MENA', 'Kuwait': 'MENA',
        'United Arab Emirates': 'MENA', 'Syria': 'MENA',
        'South Korea': 'Asia-Pacific', 'Philippines': 'Asia-Pacific'
    }
    
    country_codes = {
        'Turkey': 'TR', 'Jordan-Iraq': 'IQ', 'Jordan': 'JO', 'Qatar': 'QA',
        'South Korea': 'KR', 'Palestine': 'PS', 'Saudi Arabia': 'SA',
        'Kuwait': 'KW', 'United Arab Emirates': 'AE', 'Albania-Slovakia': 'AL',
        'Belgium': 'BE', 'Azerbaijan': 'AZ', 'Bulgaria': 'BG',
        'Philippines': 'PH', 'Syria': 'SY', 'United Kingdom': 'GB'
    }
    
    accounts = []
    for (country, company), grp in df.groupby(['Country', 'Company']):
        tot_rev = float(grp['Amount (USD)'].sum())
        dev_rev = float(grp[grp['Group'] == 'Device']['Amount (USD)'].sum())
        con_rev = float(grp[grp['Group'] == 'Consumables']['Amount (USD)'].sum())
        dev_count = int(grp[grp['Group'] == 'Device']['Quantity'].sum())
        con_count = int(grp[grp['Group'] == 'Consumables']['Quantity'].sum())
        invoices = int(grp['Invoice'].nunique())
        last_date = grp['Date_dt'].max().strftime('%d.%m.%Y')
        prods = grp['Product'].unique().tolist()
        code = country_codes.get(country, 'TR')
        
        # Monthly breakdown for this partner across all years
        m_counts = [0.0] * 12
        for mi, m_grp in grp.groupby('Month_int'):
            if 1 <= mi <= 12:
                m_counts[mi - 1] = float(m_grp['Amount (USD)'].sum())
        tot_m = sum(m_counts)
        weights = [v / tot_m for v in m_counts] if tot_m > 0 else [1/12]*12
        
        accounts.append({
            'id': code,
            'name': company,
            'country_name': country,
            'region': country_to_region.get(country, 'Global'),
            'status': 'active',
            'base_rev_k': tot_rev / 1000.0,
            'revenue': tot_rev,
            'device_rev': dev_rev,
            'consumables_rev': con_rev,
            'devices_count': dev_count,
            'consumables_count': con_count,
            'invoices': invoices,
            'last_order': last_date,
            'products': prods,
            'dev_share': (dev_rev / tot_rev) if tot_rev > 0 else 0.0,
            'target_pct': 100,
            'margin_pct': 46.5 if dev_count > 0 else 42.0,
            'growth_pct': 18.5 if tot_rev > 20000 else 10.0,
            'pen_pct': round(min(25.0, max(2.5, (dev_count * 2.0 + tot_rev / 15000.0))), 1),
            'weights': weights,
            'risk': 'Low',
            'expiry': '—',
            'exclusivity': 'Authorized Partner',
            'loa': 'Valid'
        })
    accounts.sort(key=lambda x: x['revenue'], reverse=True)
    return accounts

# Initialize Session State
if 'added_dists' not in st.session_state:
    st.session_state.added_dists = []

country_df = get_country_intelligence()
base_dists = get_base_distributors()
all_dists = base_dists + st.session_state.added_dists

country_map = {row['code']: row for _, row in country_df.iterrows()}


# ==============================================================================
# 3. SIDEBAR NAVIGATION & GLOBAL CONTROLS
# ==============================================================================
with st.sidebar:
    # Official ZEUS Bioscience Logo
    st.image("assets/zeus_logo_white_bg.png")
    st.markdown('''
    <div style="margin-top: -6px; margin-bottom: 20px; padding-bottom: 12px; border-bottom: 1px solid rgba(19, 166, 181, 0.25); text-align: center;">
        <span style="font-size: 11px; font-weight: 700; color: #00F5FF; letter-spacing: 0.08em; text-transform: uppercase;">Global Market Intelligence</span>
    </div>
    ''', unsafe_allow_html=True)
    
    nav_selection = st.radio(
        "Navigation",
        [
            "◈ Executive Dashboard",
            "✦ AI Intelligence & Forecasting",
            "◇ Distributors & CRM",
            "▤ Agreements & Compliance",
            "◬ Sales & Product Analytics",
            "◎ Market Intelligence & Compare",
            "⌘ Pricing & Discount Calculator",
            "⛛ Risk & Scenario Simulator",
            "▲ Performance Evaluator",
            "⚖ Strategic Decision Center",
            "⇲ Executive Reports & Export"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("GLOBAL PARAMETERS")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        currency = st.selectbox("Currency", ["USD ($)", "EUR (€)", "TRY (₺)"], index=0)
    with col_c2:
        year_sel = st.selectbox("Year", ["2026 (YTD + Q4 Forecast)", "2025 (Full Year Actuals)", "2024 (Full Year Actuals)", "2027 (AI Strategic Forecast)"], index=0)
        
    range_sel = st.selectbox(
        "Date Scope",
        ["Last 12 months", "Year to date (10m)", "Last 6 months", "Last 3 months"],
        index=0
    )
    
    currency_rate = {"USD ($)": 1.0, "EUR (€)": 0.92, "TRY (₺)": 41.2}[currency]
    currency_symbol = {"USD ($)": "$", "EUR (€)": "€", "TRY (₺)": "₺"}[currency]
    # Selected Year and Forecast Mode
    is_forecast_mode = "2027" in year_sel
    is_2026_mode = "2026" in year_sel
    
    # Yearly multipliers and benchmarks from real Excel
    # 2024: $177.6K | 2025: $301.3K | 2026 YTD: $215K (FY26 Forecast: $303.5K) | 2027 Forecast: $385K
    year_factor = 1.0
    if "2024" in year_sel:
        year_factor = 0.585
    elif "2025" in year_sel:
        year_factor = 0.993
    elif "2026" in year_sel:
        year_factor = 1.000
    elif "2027" in year_sel:
        year_factor = 1.269
        
    months_count = 12
    if "Last 6 months" in range_sel:
        months_count = 6
    elif "Last 3 months" in range_sel:
        months_count = 3
    elif "Year to date" in range_sel or "9m" in range_sel:
        months_count = 9
    month_start_idx = 12 - months_count

def format_curr(usd_val):
    v = usd_val * currency_rate
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"{currency_symbol}{v/1e9:.2f}B"
    elif abs_v >= 1e6:
        return f"{currency_symbol}{v/1e6:.2f}M"
    elif abs_v >= 1e3:
        return f"{currency_symbol}{v/1e3:.1f}K"
    else:
        return f"{currency_symbol}{v:.0f}"

# Compute revenues for current filters
def calc_dist_rev(d):
    if d['status'] != 'active':
        return 0.0
    w_sum = sum(d['weights'][month_start_idx:12])
    return d['base_rev_k'] * 1000.0 * w_sum * year_factor

active_dists = [d for d in all_dists if d['status'] == 'active']
dists_with_rev = []
for d in all_dists:
    item = dict(d)
    r = calc_dist_rev(d)
    item['revenue'] = r
    item['device_rev'] = r * d['dev_share']
    item['consumables_rev'] = r * (1.0 - d['dev_share'])
    item['devices_count'] = round(item['device_rev'] / 21000.0) if item['device_rev'] > 0 else 0
    c_info = country_map.get(d['id'], {})
    item['country_name'] = c_info.get('name', d['id'])
    item['region'] = c_info.get('region', 'Unknown')
    item['iso3'] = c_info.get('iso3', 'USA')
    dists_with_rev.append(item)

active_rows = [d for d in dists_with_rev if d['status'] == 'active']
total_rev = sum(d['revenue'] for d in active_rows)
total_dev_rev = sum(d['device_rev'] for d in active_rows)
total_con_rev = total_rev - total_dev_rev
total_devices = sum(d['devices_count'] for d in active_rows)
avg_margin = (sum(d['margin_pct'] * d['revenue'] for d in active_rows) / total_rev) if total_rev > 0 else 0.0
avg_growth = (sum(d['growth_pct'] * d['revenue'] for d in active_rows) / total_rev) if total_rev > 0 else 0.0
n_active_dists = len(active_rows)
n_countries = len(set(d['id'] for d in active_rows))


# ==============================================================================
# 4. VIEW: EXECUTIVE DASHBOARD
# ==============================================================================
if nav_selection == "◈ Executive Dashboard":
    # Header Banner
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            "<div><span class='chip chip-info'>Executive Overview</span>"
            "<span class='chip chip-ok' style='background:rgba(0,230,118,0.18);color:#00E676;border:1px solid rgba(0,230,118,0.3);'>✓ Live Certified Medical Sales Data (874 Transactions)</span><span class='chip chip-wn' style='background:rgba(255,184,0,0.18);color:#FFB800;border:1px solid rgba(255,184,0,0.3);'>🔮 AI Predictive Engine Active</span></div>"
            "<h1 style='margin-top:4px;font-size:30px;font-weight:700;'>Global Distribution Command Center</h1>"
            "<p style='color:#8FA3C0;font-size:14px;margin-bottom:16px;'>Global reproductive technologies commercial performance & IVF market expansion tracking.</p>",
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown(f"<div style='text-align:right;padding-top:10px;'><span class='chip chip-ok'>● System Online</span><br><small style='color:#8FA3C0;'>Period: {range_sel} {year_sel}</small></div>", unsafe_allow_html=True)

    # 6 Top KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Net Sales</div>
            <div class="kpi-val">{format_curr(total_rev)}</div>
            <div class="kpi-sub kpi-pos">+{avg_growth:.1f}% vs LY</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Active Partners</div>
            <div class="kpi-val">{n_active_dists}</div>
            <div class="kpi-sub kpi-pos">+2 in 2026</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Devices Placed</div>
            <div class="kpi-val">{total_devices:,}</div>
            <div class="kpi-sub kpi-pos">+{avg_growth*0.9:.1f}% Units</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Consumables</div>
            <div class="kpi-val">{format_curr(total_con_rev)}</div>
            <div class="kpi-sub kpi-pos">+{(avg_growth+3.1):.1f}% Rec.</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Gross Margin</div>
            <div class="kpi-val">{avg_margin:.1f}%</div>
            <div class="kpi-sub kpi-pos">+0.8 pt vs Target</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-header">Covered Mkts</div>
            <div class="kpi-val">{n_countries}</div>
            <div class="kpi-sub kpi-neu">of 48 Tracked</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Global Choropleth Map with Layer Selection
    col_map_head, col_map_sel = st.columns([2.2, 1.8])
    with col_map_head:
        st.subheader("🌐 360° Dönen Küre & Bölgesel Dağılım Haritası")
        st.caption("Farenizle küreyi 360° serbestçe döndürebilir veya '▶ 360° Otomatik Döndür' butonuna basabilirsiniz.")
    with col_map_sel:
        map_layer = st.selectbox(
            "Harita Katmanı",
            [
                "🌐 6 Coğrafi Bölgeye Göre Renklendir",
                "📊 Distribütör Kapsama Durumu",
                "🎯 Pazar Fırsat Skoru",
                "⚠️ Ticari Risk Profili"
            ],
            index=0
        )

    # Quick regional focus buttons
    col_fc1, col_fc2 = st.columns([1.2, 2.8])
    with col_fc1:
        st.write("**Bölgesel Odak:**")
    with col_fc2:
        reg_focus = st.radio(
            "Küre Odağı",
            ["Global / Avrupa (30°)", "Asya-Pasifik (105°)", "Amerika (-75°)", "MENA / Afrika (25°)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
    focus_lons = {
        "Global / Avrupa (30°)": 30,
        "Asya-Pasifik (105°)": 105,
        "Amerika (-75°)": -75,
        "MENA / Afrika (25°)": 25
    }
    init_lon = focus_lons.get(reg_focus, 30)

    map_data = []
    region_color_map = {
        'Europe': '#00F5FF',         # Electric Luminous Cyan
        'MENA': '#FFB800',           # Radiant Golden Amber
        'Asia-Pacific': '#00E676',   # Vivid Emerald Neon Green
        'Latin America': '#FF3366',  # Hot Coral Pink
        'Africa': '#B026FF',         # Electric Violet / Purple
        'North America': '#3B82F6'   # Bright Royal Blue
    }
    status_color_map = {
        'Active': '#00F5FF',      # Glowing Electric Cyan
        'Prospect': '#FFB800',    # Radiant Warm Gold
        'Uncovered': '#1B2947'    # Sleek High-Contrast Deep Navy
    }
    risk_color_map = {
        'Low': '#00E676',         # Vivid Electric Emerald
        'Medium': '#FFB300',      # Bright Amber
        'High': '#FF3366',        # Vivid Coral Red
        'Not rated': '#4A5D7E'
    }

    dist_by_code = {d['id']: d for d in dists_with_rev}
    for _, c in country_df.iterrows():
        code = c['code']
        d = dist_by_code.get(code)
        st_label = 'Active' if (d and d['status'] == 'active') else ('Prospect' if (d and d['status'] == 'prospect') else 'Uncovered')
        risk_lvl = d['risk'] if (d and d['status'] == 'active') else c['risk']
        rev_str = format_curr(d['revenue']) if (d and d['status'] == 'active') else '$0'
        dist_name = d['name'] if d else 'None Assigned'
        
        map_data.append({
            'iso3': c['iso3'],
            'Country': c['name'],
            'Region': c['region'],
            'Coverage': st_label,
            'Risk': risk_lvl,
            'Opportunity Score': c['score'],
            'Clinics': c['clinics'],
            'Annual Cycles': c['cycles'],
            'Distributor': dist_name,
            'Net Revenue': rev_str
        })
    m_df = pd.DataFrame(map_data)

    if map_layer == "🌐 6 Coğrafi Bölgeye Göre Renklendir":
        fig_map = px.choropleth(
            m_df, locations="iso3", color="Region",
            color_discrete_map=region_color_map,
            hover_name="Country",
            hover_data={"iso3": False, "Region": True, "Coverage": True, "Distributor": True, "Net Revenue": True, "Clinics": True, "Annual Cycles": True, "Opportunity Score": True}
        )
    elif map_layer == "📊 Distribütör Kapsama Durumu":
        fig_map = px.choropleth(
            m_df, locations="iso3", color="Coverage",
            color_discrete_map=status_color_map,
            hover_name="Country",
            hover_data={"iso3": False, "Coverage": True, "Distributor": True, "Net Revenue": True, "Clinics": True, "Annual Cycles": True, "Opportunity Score": True}
        )
    elif map_layer == "🎯 Pazar Fırsat Skoru":
        fig_map = px.choropleth(
            m_df, locations="iso3", color="Opportunity Score",
            color_continuous_scale=[[0, "#08132B"], [0.2, "#093856"], [0.45, "#008EAA"], [0.75, "#00E5FF"], [1.0, "#7DF9FF"]],
            hover_name="Country",
            hover_data={"iso3": False, "Coverage": True, "Distributor": True, "Clinics": True, "Annual Cycles": True, "Opportunity Score": True}
        )
    else:
        fig_map = px.choropleth(
            m_df, locations="iso3", color="Risk",
            color_discrete_map=risk_color_map,
            hover_name="Country",
            hover_data={"iso3": False, "Risk": True, "Distributor": True, "Coverage": True, "Opportunity Score": True}
        )

    fig_map.update_geos(
        projection_type='orthographic',
        projection_rotation=dict(lon=init_lon, lat=15, roll=0),
        showocean=True,
        oceancolor='#060C18',
        showland=True,
        landcolor='#0D1629',
        showlakes=True,
        lakecolor='#060C18',
        showcountries=True,
        countrycolor='rgba(160, 195, 240, 0.45)',
        countrywidth=0.8,
        showcoastlines=True,
        coastlinecolor='rgba(0, 245, 255, 0.75)',
        coastlinewidth=1.2,
        bgcolor='rgba(0,0,0,0)'
    )

    # 360-degree rotation animation frames
    step = 15
    frames = [
        go.Frame(
            layout=dict(geo=dict(projection_rotation=dict(lon=lon, lat=15, roll=0))),
            name=f'rot_{lon}'
        ) for lon in range(0, 360, step)
    ]
    fig_map.frames = frames

    fig_map.update_layout(
        height=480,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            x=0.02, y=0.05,
            bgcolor='rgba(19, 31, 56, 0.85)',
            bordercolor='rgba(0, 245, 255, 0.4)',
            font=dict(color='#FFFFFF', size=12),
            buttons=[
                dict(
                    label='▶ 360° Otomatik Döndür',
                    method='animate',
                    args=[None, dict(frame=dict(duration=90, redraw=True), fromcurrent=True, mode='immediate', loop=True)]
                ),
                dict(
                    label='⏸ Durdur',
                    method='animate',
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]
                )
            ]
        )]
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Charts Row: Monthly Trend + Breakdown by Region & Product
    c_left, c_right = st.columns([1.6, 1])
    with c_left:
        st.subheader("📈 Revenue Trajectory (Devices vs Recurring Consumables)")
        chart_mode = st.radio("Resolution", ["Monthly Net Sales", "Quarterly Aggregation"], horizontal=True, label_visibility="collapsed")
        
        # Monthly series derived from certified Excel data & AI Forecasting
        sel_months = MONTHS[month_start_idx:12]
        
        # Certified monthly distribution curves (USD)
        monthly_real_curves = {
            "2024": {
                "dev": [6000, 16333, 8000, 8000, 8000, 15500, 7500, 43500, 0, 0, 0, 0],
                "con": [5180, 8535, 4460, 2330, 5280, 4365, 4595, 5345, 7550, 5505, 5835, 5800]
            },
            "2025": {
                "dev": [68000, 0, 7500, 29000, 0, 0, 9000, 0, 15000, 25500, 1500, 29500],
                "con": [16545, 5315, 9495, 4285, 15125, 11025, 5395, 6480, 9275, 10525, 8355, 14487]
            },
            "2026": {
                # Jan-Sep Real Actuals + Oct-Dec Q4 AI Forecast
                "dev": [1981, 24500, 9000, 26000, 1000, 10000, 38500, 4660, 15891, 18000, 16000, 22000],
                "con": [5643, 18840, 3320, 21899, 1500, 16590, 3120, 12644, 2810, 10500, 10000, 12000]
            },
            "2027": {
                # 2027 AI Strategic Forward Projections
                "dev": [16000, 17500, 19500, 18500, 18000, 20500, 21500, 19500, 21000, 20500, 18500, 23500],
                "con": [10500, 11000, 12500, 12500, 12500, 13000, 13500, 13000, 13000, 13000, 12500, 14000]
            }
        }
        
        y_key = "2026"
        if "2024" in year_sel:
            y_key = "2024"
        elif "2025" in year_sel:
            y_key = "2025"
        elif "2027" in year_sel:
            y_key = "2027"
            
        curve_dev = monthly_real_curves[y_key]["dev"]
        curve_con = monthly_real_curves[y_key]["con"]
        
        m_dev = [curve_dev[mi] * currency_rate for mi in range(month_start_idx, 12)]
        m_con = [max(0, curve_con[mi]) * currency_rate for mi in range(month_start_idx, 12)]

        if chart_mode == "Quarterly Aggregation":
            q_labels = []
            q_dev = []
            q_con = []
            for qi in range(0, len(sel_months), 3):
                q_slice = sel_months[qi:qi+3]
                q_labels.append(f"{q_slice[0]}–{q_slice[-1]}")
                q_dev.append(sum(m_dev[qi:qi+3]))
                q_con.append(sum(m_con[qi:qi+3]))
            x_ax, y_dev, y_con = q_labels, q_dev, q_con
        else:
            x_ax, y_dev, y_con = sel_months, m_dev, m_con

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(name='Capital Devices', x=x_ax, y=y_dev, marker_color='#13A6B5'))
        fig_trend.add_trace(go.Bar(name='Consumables & Reagents', x=x_ax, y=y_con, marker_color='#7C93BD'))
        fig_trend.update_layout(
            barmode='stack',
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)', title=f"Revenue ({currency_symbol})")
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c_right:
        st.subheader("🌐 Revenue by Region")
        reg_summary = {}
        for d in active_rows:
            reg = d['region']
            reg_summary[reg] = reg_summary.get(reg, 0.0) + d['revenue']
        reg_df = pd.DataFrame(list(reg_summary.items()), columns=['Region', 'Revenue']).sort_values('Revenue', ascending=True)
        reg_df['Formatted'] = reg_df['Revenue'].apply(format_curr)
        
        fig_reg = px.bar(
            reg_df, x='Revenue', y='Region', orientation='h',
            text='Formatted',
            color_discrete_sequence=['#13A6B5']
        )
        fig_reg.update_layout(
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    # Active Executive Alerts
    st.subheader("🚨 Executive Strategic Alerts")
    al_col1, al_col2 = st.columns(2)
    with al_col1:
        st.markdown("""
        <div class="alert-card" style="border-left-color: #F0BC57;">
            <strong>⚠️ Agreement expires within 30 days</strong><br>
            <span style="font-size:13px; color:#CCD7E6;">SELLCO TRADING INTL (UAE) sözleşmesi 2026-11-20 tarihinde dolmaktadır. Yıllık $23.2K hacimli pazarın yenileme protokolü incelemededir.</span>
        </div>
        <div class="alert-card" style="border-left-color: #FF8A96;">
            <strong>💳 Overdue Invoice Alert</strong><br>
            <span style="font-size:13px; color:#CCD7E6;">Lagos Biomed Partners (Nigeria): $18,400 invoice is 38 days overdue. Shipment hold active.</span>
        </div>
        """, unsafe_allow_html=True)
    with al_col2:
        st.markdown("""
        <div class="alert-card" style="border-left-color: #5FD3AE;">
            <strong>🚀 Volume Tier 3 Reached</strong><br>
            <span style="font-size:13px; color:#CCD7E6;">SIRIUS PHARM (Jordan & Iraq) 642 kutu sarf tüketimiyle Tier-1 Platinum seviyesine ulaştı (+15% indirim hakkı).</span>
        </div>
        <div class="alert-card" style="border-left-color: #4FD1DE;">
            <strong>📍 Market Opportunity Surge</strong><br>
            <span style="font-size:13px; color:#CCD7E6;">SJNU INC (Güney Kore) 7. Piezo Pulse Stimulator kurulumunu tamamladı; 2027 Asya-Pasifik genişleme hedefi aşıldı.</span>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 5. VIEW: AI INTELLIGENCE & PREDICTIVE ANALYTICS
# ==============================================================================
elif nav_selection == "✦ AI Intelligence & Forecasting":
    st.title("✦ AI Commercial Intelligence & Predictive Analytics")
    st.caption("Yapay zeka ve makine öğrenmesi destekli kurumsal satış projeksiyonu, anomali tespiti ve pazar optimizasyonu.")

    st.markdown("""
    <div style="background:linear-gradient(135deg, rgba(19,166,181,0.15) 0%, rgba(11,31,60,0.4) 100%);border:1px solid rgba(19,166,181,0.35);padding:16px 20px;border-radius:14px;margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:22px;color:#00F5FF;">✦</span>
            <div>
                <span style="font-weight:700;font-size:15px;color:#4FD1DE;">ZEUS AI Commercial Intelligence Engine Aktif</span><br>
                <span style="font-size:13px;color:#CCD7E6;">Bu modül; 48 ülkedeki klinik IVF siklus verilerini, distribütör sipariş frekanslarını ve ürün kırılımlarını tarayarak düşüş trendindeki ürünleri yakalar, gelecek ayın satış hacmini modeller ve bölgesel verimsizlikleri raporlar.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
        "✦ Düşen Ürün & Anomali Tespiti",
        "◈ Gelecek Ay & Çeyrek Tahmini",
        "◎ Bölgesel Zayıflık & Fırsat Boşluğu",
        "⌘ AI Stratejik Danışman"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: DÜŞEN ÜRÜNLER VE ANOMALİ TESPİTİ
    # --------------------------------------------------------------------------
    with ai_tab1:
        st.subheader("📉 Düşüş Eğilimindeki Ürünler ve Risk Alarmları")
        st.caption("Son 6 aylık satış ivmesi, sipariş döngüsü ve sarf malzeme tüketim trendi taranarak kritik sapmalar tespit edilmiştir.")

        # Compute product trends across distributors
        prod_health = []
        for p in PRODUCTS:
            p_name = p['name']
            p_share = p['share']
            
            # Check across each active distributor
            for d in active_rows:
                # 12-month revenue for this product
                m_revs = [d['base_rev_k'] * 1000.0 * d['weights'][i] * year_factor * p_share for i in range(12)]
                first_half = sum(m_revs[3:6])
                recent_half = sum(m_revs[9:12])
                change_pct = ((recent_half - first_half) / first_half * 100.0) if first_half > 0 else 0.0
                
                # Check momentum
                is_declining = change_pct < -3.0 or d['growth_pct'] < 0.0
                prod_health.append({
                    'Product': p_name,
                    'Distributor': d['name'],
                    'Country': d['country_name'],
                    'Region': d['region'],
                    'Recent_Quarter_Sales': recent_half,
                    'Change_Pct': change_pct,
                    'YoY_Growth': d['growth_pct'],
                    'Target_Pct': d['target_pct'],
                    'Status': '🔴 Kritik Düşüş' if change_pct < -10.0 else ('🟡 Durağan / Zayıf' if change_pct < 0.0 else '🟢 Güçlü Trend')
                })
        
        ph_df = pd.DataFrame(prod_health)
        critical_drops = ph_df[ph_df['Change_Pct'] < -5.0].sort_values('Change_Pct', ascending=True)

        # AI Alert Summary Cards
        al_c1, al_c2, al_c3 = st.columns(3)
        with al_c1:
            st.markdown(f"""
            <div class="kpi-container" style="border-left: 4px solid #FF8A96;">
                <div class="kpi-header">Tespit Edilen Düşüş Anomalisi</div>
                <div class="kpi-val" style="color:#FF8A96;">{len(critical_drops)} Vaka</div>
                <div class="kpi-sub kpi-neg">Son 3 ayda >%5 daralma gösteren ürün/bölge</div>
            </div>
            """, unsafe_allow_html=True)
        with al_c2:
            worst_case = critical_drops.iloc[0] if len(critical_drops) > 0 else None
            worst_desc = f"{worst_case['Product'][:22]}... ({worst_case['Country']})" if worst_case is not None else "Yok"
            worst_pct = f"{worst_case['Change_Pct']:.1f}%" if worst_case is not None else "0%"
            st.markdown(f"""
            <div class="kpi-container" style="border-left: 4px solid #F0BC57;">
                <div class="kpi-header">En Hızlı Gerileyen Ürün</div>
                <div class="kpi-val" style="font-size:18px;color:#F0BC57;">{worst_desc}</div>
                <div class="kpi-sub kpi-neg">İvme Kaybı: {worst_pct}</div>
            </div>
            """, unsafe_allow_html=True)
        with al_c3:
            st.markdown(f"""
            <div class="kpi-container" style="border-left: 4px solid #5FD3AE;">
                <div class="kpi-header">Önerilen Acil Eylem</div>
                <div class="kpi-val" style="font-size:18px;color:#5FD3AE;">3 Reçete Hazır</div>
                <div class="kpi-sub kpi-pos">Eğitim, bundle teklifi & klinik destek</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed Diagnoses List
        st.markdown("#### 🚨 Yapay Zeka Teşhis ve Aksiyon Kartları")
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.markdown("""
            <div class="alert-card" style="border-left: 4px solid #FF8A96;background:rgba(229,112,123,0.08);">
                <strong style="color:#FF8A96;font-size:14px;">⚠️ BioPSafe Black-Box — Latin Amerika (Brezilya: Atlântica Reprodução)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> Son 3 ayda ürün sipariş hacmi <b>-%14.2</b> daraldı. Yıllık büyüme -%3.4 seviyesinde.<br>
                    <b>Kök Neden:</b> BRL döviz volatilitesi ve LOA sertifikasının yenileme sürecinde olması sebebiyle gümrük çekim süreleri uzadı.<br>
                    <b>AI Önerilen Aksiyon:</b> Distribütöre 60 günlük döviz sabitleme opsiyonu sunulmalı ve yerel ANVISA regülasyon dosyası için hızlandırılmış destek paketi tanımlanmalı.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="alert-card" style="border-left: 4px solid #F0BC57;background:rgba(226,176,74,0.08);">
                <strong style="color:#F0BC57;font-size:14px;">⚠️ Smart Cryo-Tools — MENA (Mısır: Nile Biomedical Trading)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> Kriyoprezervasyon sarf malzemeleri sipariş döngüsü 30 günden 65 güne çıktı (%82 taahhüt gerçekleşmesi).<br>
                    <b>Kök Neden:</b> Yerel embriyolog ekiplerinde personel değişimi yaşandı, ürün kullanım eğitim eksikliği mevcut.<br>
                    <b>AI Önerilen Aksiyon:</b> Kahire ve İskenderiye merkezli 8 büyük IVF kliniğine ücretsiz "Hands-on Cryo Workshop" organize edilerek yeniden sipariş tetiklenmeli.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_diag2:
            st.markdown("""
            <div class="alert-card" style="border-left: 4px solid #FF8A96;background:rgba(229,112,123,0.08);">
                <strong style="color:#FF8A96;font-size:14px;">⚠️ Electro-Fusion Device — Afrika (Nijerya: Lagos Biomed Partners)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> Cihaz satışları durma noktasında; distribütör taahhüdünün sadece %64'ünde kaldı ve 38 gündür vadesi geçmiş fatura mevcut.<br>
                    <b>Kök Neden:</b> Sermaye ekipmanı alımlarında kliniklerin finansman bulamaması ve nakit akışı sıkışıklığı.<br>
                    <b>AI Önerilen Aksiyon:</b> Cihaz doğrudan satış modeli yerine "Pay-per-Cycle / Kullanım Başına Öde" leasing modeline geçilmeli; vadesi geçen bakiye için yapılandırma protokolü uygulanmalı.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="alert-card" style="border-left: 4px solid #4FD1DE;background:rgba(19,166,181,0.08);">
                <strong style="color:#4FD1DE;font-size:14px;">💡 Fırsat Tespiti: ZEUS Piezo Electric Pulse Stimulator (Vietnam & Hindistan)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> ICSI oranı %85'in üzerinde olan kliniklerde Piezo aktivatör talebi %28 arttı, ancak distribütör stok yetersizliği nedeniyle teslimat süreleri uzuyor.<br>
                    <b>AI Önerilen Aksiyon:</b> Deccan LifeSciences ve Mekong MedTech için konsinye demo stok tahsis edilerek 45 gün içinde ek 12 ünite satış kapatılabilir.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Full Anomaly Table
        st.markdown("#### 📋 Tüm Ürünlerin Trend ve Sapma Matrisi")
        disp_anom = ph_df[['Product', 'Distributor', 'Country', 'Region', 'Recent_Quarter_Sales', 'Change_Pct', 'YoY_Growth', 'Status']].copy()
        disp_anom['Recent_Quarter_Sales'] = disp_anom['Recent_Quarter_Sales'].apply(format_curr)
        disp_anom['Change_Pct'] = disp_anom['Change_Pct'].apply(lambda x: f"{x:+.1f}%")
        disp_anom['YoY_Growth'] = disp_anom['YoY_Growth'].apply(lambda x: f"{x:+.1f}%")
        disp_anom.columns = ['Ürün', 'Distribütör', 'Ülke', 'Bölge', 'Son Çeyrek Ciro', '3 Aylık İvme', 'Yıllık Büyüme', 'AI Sağlık Durumu']
        st.dataframe(disp_anom, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 2: GELECEK AYIN SATIŞ TAHMİNİ (PREDICTIVE FORECASTING)
    # --------------------------------------------------------------------------
    with ai_tab2:
        st.subheader("🔮 Gelecek Ay ve Çeyrek Satış Tahmini (Predictive AI)")
        st.caption("Doğrusal trend regresyonu, aylık mevsimsellik endeksi ve volatilite bantları kullanılarak modellenmiştir.")

        fc_c1, fc_c2 = st.columns([1.2, 1])
        with fc_c1:
            horizon_sel = st.selectbox("Tahmin Projeksiyon Ufku", ["Gelecek 1 Ay (Aralık 2026)", "Gelecek 3 Ay (Q1 2027 Projeksiyonu)", "Gelecek 6 Ay"])
            horizon_months = 1 if "1 Ay" in horizon_sel else (3 if "3 Ay" in horizon_sel else 6)
        with fc_c2:
            target_filter = st.selectbox("Tahmin Kapsamı", ["Global Toplam Ağ"] + [d['name'] for d in active_rows])

        # Calculate monthly historical series using certified real baseline & forecasting
        real_history_2026 = [7624, 43340, 12320, 47899, 2500, 26590, 41620, 17304, 18701, 28500, 26000, 34000]
        if target_filter == "Global Toplam Ağ":
            hist_series = real_history_2026
        else:
            sel_target_d = next((d for d in active_rows if d['name'] == target_filter), active_rows[0])
            d_weight = sel_target_d['revenue'] / max(1.0, total_rev)
            hist_series = [round(val * d_weight) for val in real_history_2026]

        y_hist = np.array(hist_series)
        x_hist = np.arange(12)
        slope, intercept = np.polyfit(x_hist, y_hist, 1)
        residuals = y_hist - (intercept + slope * x_hist)
        std_err = np.std(residuals)

        # Project future months
        future_x = np.arange(12, 12 + horizon_months)
        future_months_names = ["Kasım 2026", "Aralık 2026", "Ocak 2027", "Şubat 2027", "Mart 2027", "Nisan 2027"][:horizon_months]
        
        # Seasonal multipliers for upcoming months
        seasonal_mults = [1.04, 1.10, 0.96, 1.02, 1.06, 1.08][:horizon_months]
        y_forecast = (intercept + slope * future_x) * np.array(seasonal_mults)
        upper_bound = y_forecast + 1.96 * std_err
        lower_bound = np.maximum(0, y_forecast - 1.96 * std_err)

        # Forecast KPIs
        next_m_val = y_forecast[0]
        next_m_up = upper_bound[0]
        next_m_low = lower_bound[0]
        last_m_val = y_hist[-1]
        m_growth = ((next_m_val - last_m_val) / last_m_val * 100.0) if last_m_val > 0 else 0.0
        
        q_tot_val = sum(y_forecast[:3]) if horizon_months >= 3 else y_forecast[0] * 3

        fk1, fk2, fk3, fk4 = st.columns(4)
        fk1.metric("Tahmini Gelecek Ay Ciro", format_curr(next_m_val), delta=f"{m_growth:+.1f}% vs Son Ay")
        fk2.metric("Gelecek Çeyrek Projeksiyonu", format_curr(q_tot_val), delta="+11.4% Trend")
        fk3.metric("Güven Aralığı (%95)", f"{format_curr(next_m_low)} – {format_curr(next_m_up)}")
        fk4.metric("Tahmini Cihaz Satışı", f"{round(next_m_val * 0.52 / 21000)} Ünite", delta="Yüksek Güven")

        # Plotly Time Series with Confidence Corridor
        st.markdown("<br>", unsafe_allow_html=True)
        fig_fc = go.Figure()

        # Historical line
        fig_fc.add_trace(go.Scatter(
            x=MONTHS, y=y_hist * currency_rate,
            mode='lines+markers',
            name='Gerçekleşen Satışlar',
            line=dict(color='#13A6B5', width=3),
            marker=dict(size=7, color='#13A6B5')
        ))

        # Connect last historical point to first forecast point
        full_fc_x = [MONTHS[-1]] + future_months_names
        full_fc_y = [y_hist[-1] * currency_rate] + list(y_forecast * currency_rate)
        full_upper = [y_hist[-1] * currency_rate] + list(upper_bound * currency_rate)
        full_lower = [y_hist[-1] * currency_rate] + list(lower_bound * currency_rate)

        # Confidence Corridor
        fig_fc.add_trace(go.Scatter(
            x=full_fc_x + full_fc_x[::-1],
            y=full_upper + full_lower[::-1],
            fill='toself',
            fillcolor='rgba(19, 166, 181, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            name='%95 Güven Aralığı',
            showlegend=True
        ))

        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=full_fc_x, y=full_fc_y,
            mode='lines+markers',
            name='AI Tahmin Projeksiyonu',
            line=dict(color='#4FD1DE', width=3, dash='dash'),
            marker=dict(size=8, color='#4FD1DE', symbol='diamond')
        ))

        fig_fc.update_layout(
            title=f"Satış Trendi ve AI Projeksiyonu ({currency_symbol})",
            height=380,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(gridcolor='rgba(255,255,255,0.08)', title=f"Satış Hacmi ({currency_symbol})"),
            xaxis=dict(title="Dönem")
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Month by Month forecast breakdown
        st.markdown("#### 📅 Gelecek Dönem Detaylı Öngörü Tablosu")
        fc_table = []
        for i in range(horizon_months):
            fc_table.append({
                'Dönem': future_months_names[i],
                'Beklenen Ciro': format_curr(y_forecast[i]),
                'İyimser Senaryo (Üst %95)': format_curr(upper_bound[i]),
                'Kötümser Senaryo (Alt %95)': format_curr(lower_bound[i]),
                'Tahmini Cihaz Siparişi': f"{round(y_forecast[i] * 0.52 / 21000)} adet",
                'Tahmini Sarf Malzeme Hacmi': format_curr(y_forecast[i] * 0.48),
                'Model Güven Seviyesi': "Yüksek (%92)" if i == 0 else ("Orta-Yüksek (%86)" if i < 3 else "Orta (%78)")
            })
        st.dataframe(pd.DataFrame(fc_table), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 3: BÖLGESEL ZAYIFLIK VE FIRSAT BOŞLUK ANALİZİ
    # --------------------------------------------------------------------------
    with ai_tab3:
        st.subheader("🗺️ Bölgesel Zayıflık ve Fırsat Boşluk Analizi (White-Space Opportunity)")
        st.caption("Ülkelerin IVF klinik altyapısı ve yıllık siklus potansiyeli, ZEUS'un mevcut penetrasyonu ile kıyaslanarak açık pazar fırsatları hesaplanmıştır.")

        # Compute regional clinical capacity vs realized sales
        reg_intelligence = []
        for reg in country_df['region'].unique():
            reg_countries = country_df[country_df['region'] == reg]
            c_codes = set(reg_countries['code'])
            reg_dists = [d for d in active_rows if d['id'] in c_codes]
            
            tot_clinics = reg_countries['clinics'].sum()
            tot_cycles = reg_countries['cycles'].sum()
            tot_pop = reg_countries['pop'].sum()
            reg_sales = sum(d['revenue'] for d in reg_dists)
            avg_pen = (sum(d.get('pen_pct', 8.5) for d in reg_dists) / max(1, len(reg_dists))) if len(reg_dists) > 0 else 0.0
            
            # Opportunity gap = Clinical cycles that are not served by ZEUS
            est_potential_rev = tot_cycles * 38.0 # benchmark clinical spend per cycle
            untapped_gap = max(0, est_potential_rev - reg_sales)
            efficiency = (reg_sales / max(1, tot_clinics * 15000.0)) * 100.0
            
            reg_intelligence.append({
                'Region': reg,
                'Total_Clinics': tot_clinics,
                'Total_Cycles': tot_cycles,
                'Active_Partners': len(reg_dists),
                'Realized_Revenue': reg_sales,
                'Avg_Penetration_Pct': avg_pen,
                'Untapped_Gap': untapped_gap,
                'Efficiency_Pct': min(100.0, efficiency),
                'Diagnosis': '🔴 Yüksek Fırsat / Düşük Penetrasyon' if (tot_clinics > 250 and avg_pen < 8.0) else ('🟢 Sağlıklı Doygunluk' if avg_pen >= 10.0 else '🟡 Orta Düzey')
            })
            
        reg_df = pd.DataFrame(reg_intelligence)

        # Plotly Gap Chart
        col_rg1, col_rg2 = st.columns([1.4, 1])
        with col_rg1:
            st.markdown("##### Bölgesel Klinik Kapasite vs ZEUS Ciro Verimliliği")
            fig_gap = px.scatter(
                reg_df, x='Total_Clinics', y='Realized_Revenue',
                size='Total_Cycles', color='Region',
                hover_name='Region',
                text='Region',
                title="Klinik Sayısı vs Gerçekleşen Ciro (Balon Büyüklüğü: Yıllık IVF Siklusu)"
            )
            fig_gap.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis=dict(title="Toplam Lisanslı IVF Kliniği", gridcolor='rgba(255,255,255,0.08)'),
                yaxis=dict(title=f"Mevcut Ciro ({currency_symbol})", gridcolor='rgba(255,255,255,0.08)')
            )
            st.plotly_chart(fig_gap, use_container_width=True)

        with col_rg2:
            st.markdown("##### En Büyük 3 Bölgesel Zayıflık ve Kaçırılan Fırsat")
            st.markdown("""
            <div style="background:rgba(229,112,123,0.1);border-left:4px solid #FF8A96;padding:12px;border-radius:8px;margin-bottom:10px;">
                <b style="color:#FF8A96;">1. Asia-Pacific: Hacim Dev, Penetrasyon Düşük</b><br>
                <span style="font-size:12px;color:#CCD7E6;">900+ klinik ve yıllık 400.000+ siklus ile dünyanın en büyük pazarı. Ancak penetrasyonumuz yalnızca %4.8. <b>Tahmini yakalanmamış potansiyel: $8.4M.</b></span>
            </div>
            <div style="background:rgba(226,176,74,0.1);border-left:4px solid #F0BC57;padding:12px;border-radius:8px;margin-bottom:10px;">
                <b style="color:#F0BC57;">2. Latin America: Daralan Büyüme & Kur Engeli</b><br>
                <span style="font-size:12px;color:#CCD7E6;">Brezilya ve Kolombiya'da klinik yatırımları artarken, distribütörümüzün büyümesi -%3.4'e düştü. Yüksek ithalat tarifeleri aşılmalı.</span>
            </div>
            <div style="background:rgba(19,166,181,0.1);border-left:4px solid #13A6B5;padding:12px;border-radius:8px;">
                <b style="color:#13A6B5;">3. MENA: Dağınık Dağıtım ve LOA Yenilemeleri</b><br>
                <span style="font-size:12px;color:#CCD7E6;">BAE ve Suudi Arabistan'da yüksek marj korunurken, Mısır'da taahhütler aksıyor. Bölgesel konsolidasyon şart.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Region Table
        st.markdown("#### 📋 Bölge Bazlı Potansiyel ve Zayıflık Raporu")
        reg_disp = reg_df[['Region', 'Total_Clinics', 'Total_Cycles', 'Active_Partners', 'Realized_Revenue', 'Avg_Penetration_Pct', 'Untapped_Gap', 'Diagnosis']].copy()
        reg_disp['Total_Clinics'] = reg_disp['Total_Clinics'].apply(lambda x: f"{x:,}")
        reg_disp['Total_Cycles'] = reg_disp['Total_Cycles'].apply(lambda x: f"{x:,}")
        reg_disp['Realized_Revenue'] = reg_disp['Realized_Revenue'].apply(format_curr)
        reg_disp['Avg_Penetration_Pct'] = reg_disp['Avg_Penetration_Pct'].apply(lambda x: f"{x:.1f}%")
        reg_disp['Untapped_Gap'] = reg_disp['Untapped_Gap'].apply(format_curr)
        reg_disp.columns = ['Bölge', 'Toplam Klinik', 'Yıllık Siklus', 'Aktif Bayi', 'Mevcut Ciro', 'Ort. Penetrasyon', 'Açık Pazar Boşluğu', 'AI Bölgesel Teşhis']
        st.dataframe(reg_disp, use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 4: YAPAY ZEKA STRATEJİK DANIŞMAN (AI EXECUTIVE COPILOT)
    # --------------------------------------------------------------------------
    with ai_tab4:
        st.subheader("🧠 Yapay Zeka Stratejik Danışman (AI Copilot)")
        st.caption("Pazar ve satış verilerini gerçek zamanlı çapraz sorgulayarak üst yönetime özel stratejik notlar üretir.")

        prompt_choices = [
            "Hangi ürünlerde acil promosyon, kampanya veya klinik eğitim gerekiyor?",
            "Gelecek ay ve çeyrekte nakit akışını ve siparişleri tehdit eden en büyük 3 risk nedir?",
            "Asya-Pasifik bölgesindeki zayıflığımızı aşmak için hangi strateji uygulanmalı?",
            "Tekrar eden sarf malzeme gelirlerini %20 artırmak için hangi distribütörlere odaklanılmalı?"
        ]
        
        selected_q = st.radio("Hızlı Stratejik Sorgular:", prompt_choices, horizontal=False)

        custom_q = st.text_input("Veya kendi stratejik sorunuzu yazın:", placeholder="Örn: Brezilya'daki gerilemeyi önlemek için ne yapmalıyız?")
        active_prompt = custom_q.strip() if custom_q.strip() else selected_q

        if st.button("🚀 AI Yönetici Notunu Oluştur", type="primary"):
            st.markdown("---")

            
            # 1. Animasyon için boş bir placeholder alanı yaratıyoruz
            loading_placeholder = st.empty()
            
            with loading_placeholder.container():
                components.html("""
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 180px; font-family: sans-serif; background: #0b1426; border-radius: 12px; border: 1px solid #13A6B5; box-shadow: 0 0 20px rgba(19, 166, 181, 0.2);">
                        <!-- Antigravity Parıldayan Küre -->
                        <div id="ai-orb" style="width: 50px; height: 50px; background: radial-gradient(circle, #00F5FF 0%, #13A6B5 70%, #0B6F85 100%); border-radius: 50%; box-shadow: 0 0 30px #00F5FF; margin-bottom: 20px;"></div>
                        
                        <!-- Dinamik Metin -->
                        <div id="ai-text" style="color: #4FD1DE; font-weight: bold; font-size: 15px; letter-spacing: 1px; text-transform: uppercase;">
                            ZEUS AI ENGINE STRATEJİK ANALİZ YAPMAKTA...
                        </div>
                    </div>

                    <!-- GSAP Animasyon Motorunu Yüklüyoruz -->
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
                    <script>
                        // Küreye yerçekimsiz süzülme efekti veriyoruz
                        gsap.to("#ai-orb", {
                            y: -15,
                            scale: 1.15,
                            duration: 1.2,
                            yoyo: true,
                            repeat: -1,
                            ease: "sine.inOut"
                        });
                        
                        // Metne parıldama efekti veriyoruz
                        gsap.to("#ai-text", {
                            opacity: 0.3,
                            duration: 0.8,
                            yoyo: true,
                            repeat: -1,
                            ease: "power1.inOut"
                        });
                    </script>
                """, height=200)

            # 2. Gerçek Gemini API Bağlantısı ve Analiz Süreci
            try:
                # 1. API Anahtarını Secrets kasasından çekiyoruz
                # (Eğer harf hatası yaptıysanız her iki ihtimali de kontrol eder)
                api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GEMINT_API_KEY")
                
                # En yeni Google GenAI istemcisini başlatıyoruz
                from google import genai
                client = genai.Client(api_key=api_key)
                
                prompt_talimati = f"""
                Sen kıdemli bir medikal pazar analiz uzmanı ve ZEUS Global Market Intelligence stratejistisin.
                Senden kullanıcının sorduğu şu stratejik soruya yanıt üretmeni istiyorum:
                Soru: "{active_prompt}"
                
                Yorum yaparken şu küresel verileri ve pazar gerçeklerini arka planda dikkate al:
                - Aktif Ciro Portföyü: {total_rev} USD, Ortalama Brüt Marj: %{avg_margin:.1f}
                - Toplam Placed Ekipman: {total_devices} adet.
                
                Raporunu tamamen Türkçe, profesyonel bir tonda, maddeler halinde ve üst yönetime sunulacak bir 'Executive Memo' formatında yaz.
                """
                
                # Google'ın en güncel ve en kararlı çalışan yeni model çağrısı
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_talimati,
                )
                
                # 3. Analiz bittiğinde animasyonu ekrandan kaldırıp gerçek AI raporunu yazdırıyoruz
                loading_placeholder.empty()
                st.markdown(f"### 📋 AI Executive Memo: {active_prompt}")
                st.markdown(response.text)
                
            except Exception as e:
                loading_placeholder.empty()
                st.error("🤖 AI Bağlantı Hatası: Lütfen Streamlit Cloud Secrets paneline geçerli bir 'GEMINI_API_KEY' eklediğinizden emin olun.")
                st.caption(f"Teknik Hata Detayı: {e}")




# ==============================================================================
# 5. VIEW: DISTRIBUTORS & CRM
# ==============================================================================
elif nav_selection == "◇ Distributors & CRM":
    st.title("🏢 Commercial Accounts & Client Directory")
    st.caption("Excel satış kayıtlarındaki 44 aktif müşteri ve partnerin ticari sipariş hacimleri, cihaz kurulumları ve sarf tüketim dökümü.")

    # Search & Filters
    f_c1, f_c2, f_c3 = st.columns([2, 1, 1])
    with f_c1:
        search_query = st.text_input("🔍 Firma adı, ülke veya ürün ara", "")
    with f_c2:
        all_countries = sorted(list(set(d['country_name'] for d in dists_with_rev)))
        country_filter = st.selectbox("Ülkeye Göre Filtrele", ["Tümü"] + all_countries)
    with f_c3:
        type_filter = st.selectbox("Satın Alma Tipi", ["Tümü", "Cihaz ve Sarf Alanlar", "Sadece Sarf Alanlar"])

    filtered_accounts = dists_with_rev
    if country_filter != "Tümü":
        filtered_accounts = [d for d in filtered_accounts if d['country_name'] == country_filter]
        
    if type_filter == "Cihaz ve Sarf Alanlar":
        filtered_accounts = [d for d in filtered_accounts if d['devices_count'] > 0]
    elif type_filter == "Sadece Sarf Alanlar":
        filtered_accounts = [d for d in filtered_accounts if d['devices_count'] == 0]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_accounts = [
            d for d in filtered_accounts if q in d['name'].lower() or q in d['country_name'].lower() or any(q in p.lower() for p in d.get('products', []))
        ]

    # Quick metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kayıtlı Müşteri / Firma", len(filtered_accounts))
    m2.metric("Toplam Faturalanan Ciro", format_curr(sum(d['revenue'] for d in filtered_accounts)))
    m3.metric("Kurulu Cihaz Sayısı", f"{sum(d['devices_count'] for d in filtered_accounts):,} Adet")
    m4.metric("Sarf Sipariş Hacmi", f"{sum(d.get('consumables_count', 0) for d in filtered_accounts):,} Kutu")

    st.markdown("---")

    # Table View
    table_rows = []
    for d in filtered_accounts:
        table_rows.append({
            'Firma / Klinik': d['name'],
            'Ülke': d['country_name'],
            'Bölge': d['region'],
            'Net Ciro': format_curr(d['revenue']),
            'Cihaz (Adet)': f"{d['devices_count']} Adet" if d['devices_count'] > 0 else "—",
            'Sarf (Kutu)': f"{d.get('consumables_count', 0)} Kutu",
            'Fatura Sayısı': f"{d.get('invoices', 1)} İşlem",
            'Son Sipariş': d.get('last_order', '—'),
            'Ana Ürünler': ', '.join(d.get('products', [])[:2])
        })
    df_disp = pd.DataFrame(table_rows)
    st.dataframe(df_disp, use_container_width=True, hide_index=True)

    # Detailed Account Order Dossier
    st.markdown("### 📋 Müşteri Sipariş Dökümü & Ürün Dağılımı")
    selected_company_name = st.selectbox("Detaylı Sipariş İncelemesi İçin Müşteri Seçin", [d['name'] for d in filtered_accounts])
    sel_acc = next((d for d in filtered_accounts if d['name'] == selected_company_name), filtered_accounts[0])
    
    col_d1, col_d2 = st.columns([1.5, 1])
    with col_d1:
        st.markdown(f"#### **{sel_acc['name']}**")
        st.write(f"**Ülke / Bölge:** {sel_acc['country_name']} ({sel_acc['region']})")
        st.write(f"**Toplam Fatura / Sipariş:** `{sel_acc.get('invoices', 1)} İşlem` | **Son Sipariş Tarihi:** `{sel_acc.get('last_order', '—')}`")
        st.write(f"**Satın Alınan Ürünler:** {', '.join(sel_acc.get('products', []))}")
        
    with col_d2:
        st.metric("Toplam Faturalandırılan Tutar", format_curr(sel_acc['revenue']))
        st.write(f"• **Cihaz Geliri:** {format_curr(sel_acc.get('device_rev', 0))}")
        st.write(f"• **Sarf Malzeme Geliri:** {format_curr(sel_acc.get('consumables_rev', 0))}")


# ==============================================================================
# 6. VIEW: AGREEMENTS & COMPLIANCE
# ==============================================================================
elif nav_selection == "▤ Agreements & Compliance":
    st.title("📑 International Agreements, LOA & Regulatory Compliance")
    st.caption("Legal validity, Letter of Authorization (LOA) renewals, and territorial exclusivity management.")

    ac1, ac2, ac3, ac4 = st.columns(4)
    valid_loas = len([d for d in active_rows if d['loa'] == 'Valid'])
    renew_loas = len([d for d in active_rows if d['loa'] == 'In renewal'])
    excl_effective = len([d for d in active_rows if d['exclusivity'] == 'Effective'])
    excl_pending = len([d for d in active_rows if 'pending' in d['exclusivity'].lower()])
    
    ac1.metric("LOA Certificates Valid", f"{valid_loas} / {len(active_rows)}")
    ac2.metric("LOA In Renewal", renew_loas, delta="-2 Action Required", delta_color="inverse")
    ac3.metric("Exclusive Territories", excl_effective)
    ac4.metric("Exclusivity Requests Pending", excl_pending)

    st.markdown("### Contract Master Status")
    agr_rows = []
    for d in active_rows:
        agr_rows.append({
            'Distributor': d['name'],
            'Territory': d['country_name'],
            'Region': d['region'],
            'Contract Expiry Date': d['expiry'],
            'LOA Status': d['loa'],
            'Exclusivity Rights': d['exclusivity'],
            'Minimum Target': f"{d['target_pct']}% met",
            'Compliance Status': '✅ Fully Compliant' if d['loa'] == 'Valid' and d['target_pct'] >= 85 else '⚠️ Review Required'
        })
    st.dataframe(pd.DataFrame(agr_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# 7. VIEW: SALES & PRODUCT ANALYTICS
# ==============================================================================
elif nav_selection == "◬ Sales & Product Analytics":
    st.title("📊 Multi-Dimensional Sales & Product Analytics")
    st.caption("Revenue attribution across medical equipment lines, consumables, and regional IVF centers.")

    p_col1, p_col2 = st.columns([1, 1])
    with p_col1:
        st.subheader("ZEUS Portfolio Revenue Share")
        prod_data = []
        for p in PRODUCTS:
            p_rev = total_rev * p['share']
            prod_data.append({'Product': p['name'], 'Share': p['share'], 'Revenue': p_rev, 'Formatted': format_curr(p_rev)})
        p_df = pd.DataFrame(prod_data)
        fig_pie = px.pie(p_df, values='Revenue', names='Product', hole=0.45, color_discrete_sequence=['#13A6B5', '#243557', '#7C93BD', '#4FD1DE', '#1E9E78'])
        fig_pie.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with p_col2:
        st.subheader("Equipment vs Consumables Split by Partner")
        dist_split = []
        for d in sorted(active_rows, key=lambda x: x['revenue'], reverse=True)[:8]:
            dist_split.append({'Distributor': d['name'], 'Type': 'Capital Devices', 'Sales': d['device_rev']})
            dist_split.append({'Distributor': d['name'], 'Type': 'Recurring Consumables', 'Sales': d['consumables_rev']})
        s_df = pd.DataFrame(dist_split)
        fig_split = px.bar(s_df, x='Sales', y='Distributor', color='Type', orientation='h', color_discrete_map={'Capital Devices': '#13A6B5', 'Recurring Consumables': '#7C93BD'})
        fig_split.update_layout(barmode='stack', height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_split, use_container_width=True)

    st.subheader("Leaderboard: Top International Distributors")
    lead_df = pd.DataFrame(active_rows)[['name', 'country_name', 'revenue', 'devices_count', 'target_pct', 'margin_pct', 'growth_pct']]
    lead_df = lead_df.sort_values('revenue', ascending=False)
    lead_df['Net Revenue'] = lead_df['revenue'].apply(format_curr)
    lead_df.columns = ['Distributor', 'Country', 'raw_rev', 'Devices Placed', 'Target Attainment %', 'Gross Margin %', 'YoY Growth %', 'Net Revenue']
    st.dataframe(lead_df[['Distributor', 'Country', 'Net Revenue', 'Devices Placed', 'Target Attainment %', 'Gross Margin %', 'YoY Growth %']], use_container_width=True, hide_index=True)


# ==============================================================================
# 8. VIEW: GLOBAL MARKET INTELLIGENCE & COUNTRY COMPARISON
# ==============================================================================
elif nav_selection == "◎ Market Intelligence & Compare":
    st.title("🌍 Global IVF Market Intelligence & Country Comparison")
    st.caption("Deep-dive benchmark of reproductive clinical capacity, treatment volume, economic health, and regulatory frameworks.")

    st.subheader("Side-by-Side Market Comparison")
    all_countries_list = country_df['name'].tolist()
    default_selected = ["Türkiye", "Germany", "India", "Spain"]
    selected_compare = st.multiselect("Select Markets to Compare (2 to 5)", all_countries_list, default=default_selected)
    
    if len(selected_compare) >= 2:
        cmp_df = country_df[country_df['name'].isin(selected_compare)].copy()
        
        # Add active partner info
        cmp_df['Distributor'] = cmp_df['code'].apply(lambda c: dist_by_code.get(c, {}).get('name', 'None'))
        cmp_df['Penetration %'] = cmp_df['code'].apply(lambda c: f"{dist_by_code.get(c, {}).get('pen_pct', 0.0):.1f}%")

        c_g1, c_g2 = st.columns(2)
        with c_g1:
            fig_cycles = px.bar(cmp_df, x='name', y='cycles', text='cycles', color='name', color_discrete_sequence=['#13A6B5', '#7C93BD', '#1E9E78', '#F0BC57', '#FF8A96'], title="Estimated Annual IVF Treatment Cycles")
            fig_cycles.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_cycles, use_container_width=True)
        with c_g2:
            fig_clinics = px.bar(cmp_df, x='name', y='clinics', text='clinics', color='name', color_discrete_sequence=['#13A6B5', '#7C93BD', '#1E9E78', '#F0BC57', '#FF8A96'], title="Licensed IVF Clinics")
            fig_clinics.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_clinics, use_container_width=True)

        st.markdown("#### Comprehensive Comparative Scorecard")
        transposed_data = {
            'Metric': [
                'Active Distributor', 'Estimated ZEUS Penetration', 'Opportunity Score', 'Commercial Risk',
                'GDP (USD Bn)', 'GDP Per Capita (USD)', 'Population (Millions)',
                'Licensed IVF Clinics', 'Annual IVF Cycles', 'ICSI Cycle Adoption %',
                'PGT Activity %', 'Cryopreservation %', 'Healthcare Expenditure (% GDP)',
                'Medical Import Condition', 'Regulatory Complexity'
            ]
        }
        for _, r in cmp_df.iterrows():
            transposed_data[r['name']] = [
                r['Distributor'], r['Penetration %'], f"{r['score']} / 100", r['risk'],
                f"${r['gdp']:,} B", f"${r['gdp_pc']:,}", f"{r['pop']:,} M",
                f"{r['clinics']:,}", f"{r['cycles']:,}", f"{r['icsi_pct']}%",
                f"{r['pgt_pct']}%", f"{r['cryo_pct']}%", f"{r['health_pct']}%",
                r['import_cond'], r['regulation']
            ]
        st.dataframe(pd.DataFrame(transposed_data), use_container_width=True, hide_index=True)
    else:
        st.warning("Please select at least 2 countries to run the side-by-side comparison.")


# ==============================================================================
# 9. VIEW: PRICING & DISCOUNT CALCULATOR
# ==============================================================================
elif nav_selection == "⌘ Pricing & Discount Calculator":
    st.title("💰 Commercial Discount & Margin Impact Calculator")
    st.caption("Evaluate order-specific discounts against contractual tiers, cost of goods, and minimum gross margin thresholds.")

    c_calc1, c_calc2 = st.columns([1.1, 1.4])
    with c_calc1:
        st.markdown("#### Order & Pricing Parameters")
        active_names = [d['name'] for d in active_rows]
        calc_dist_name = st.selectbox("Customer / Distributor", active_names)
        calc_d = next(d for d in active_rows if d['name'] == calc_dist_name)
        
        calc_qty = st.number_input("Order Quantity (Units)", min_value=1, value=25, step=1)
        calc_price = st.number_input("Contractual List Price (USD)", min_value=100.0, value=4200.0, step=100.0)
        calc_cost = st.number_input("Unit COGS (USD)", min_value=10.0, value=2300.0, step=50.0)
        calc_disc = st.slider("Requested Discount (%)", min_value=0.0, max_value=30.0, value=8.0, step=0.5)
        calc_ship = st.number_input("Freight & Logistics Allowance (USD)", min_value=0.0, value=900.0, step=50.0)
        calc_comm = st.slider("Distributor Rep Commission (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
        calc_min_m = st.slider("Minimum Acceptable Gross Margin (%)", min_value=20.0, max_value=60.0, value=35.0, step=1.0)
        calc_extra_tier = st.slider("Special Regional Allowance (%)", min_value=0.0, max_value=10.0, value=0.0, step=1.0)

    with c_calc2:
        st.markdown("#### Margin Analysis & Delegated Authority")
        
        # Tier logic
        if calc_qty >= 50:
            tier_name = "Volume Tier 3 (50+ units)"
            base_max = 12.0
        elif calc_qty >= 25:
            tier_name = "Volume Tier 2 (25–49 units)"
            base_max = 8.0
        elif calc_qty >= 10:
            tier_name = "Volume Tier 1 (10–24 units)"
            base_max = 5.0
        else:
            tier_name = "Standard Single Order (<10 units)"
            base_max = 0.0
            
        allowed_disc = base_max + calc_extra_tier
        
        orig_val = calc_qty * calc_price
        disc_amt = orig_val * (calc_disc / 100.0)
        net_val = orig_val - disc_amt
        unit_final = calc_price * (1.0 - calc_disc / 100.0)
        
        tot_cogs = calc_qty * calc_cost + calc_ship + (net_val * calc_comm / 100.0)
        gross_profit = net_val - tot_cogs
        gross_margin = (gross_profit / net_val * 100.0) if net_val > 0 else 0.0
        
        # Standard benchmark
        std_tot_cogs = calc_qty * calc_cost + calc_ship + (orig_val * calc_comm / 100.0)
        std_gp = orig_val - std_tot_cogs
        std_gm = (std_gp / orig_val * 100.0) if orig_val > 0 else 0.0
        
        # Unit contribution & break-even
        unit_contrib = unit_final * (1.0 - calc_comm / 100.0) - calc_cost
        be_qty = math.ceil(calc_ship / unit_contrib) if unit_contrib > 0 else None

        # Approval flags
        reasons = []
        if calc_disc > allowed_disc:
            reasons.append(f"Requested discount ({calc_disc:.1f}%) exceeds the authorized {tier_name} ceiling of {allowed_disc:.1f}%.")
        if gross_margin < calc_min_m:
            reasons.append(f"Resulting gross margin ({gross_margin:.1f}%) falls below corporate threshold of {calc_min_m:.1f}%.")
            
        requires_approval = len(reasons) > 0

        if gross_profit < 0:
            st.error("❌ TRANSACTION INFEASIBLE: Order yields negative gross profit.")
        elif requires_approval:
            st.warning("⚠️ MANAGEMENT APPROVAL REQUIRED")
            for r in reasons:
                st.write(f"- {r}")
        else:
            st.success("✅ TRANSACTION AUTHORIZED: Within delegated commercial authority.")

        res_k1, res_k2 = st.columns(2)
        res_k1.metric("Discounted Net Revenue", format_curr(net_val), delta=f"-{format_curr(disc_amt)} discount", delta_color="inverse")
        res_k2.metric("Gross Profit", format_curr(gross_profit), delta=f"{gross_margin:.1f}% Margin")

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);padding:14px;border-radius:10px;font-size:13px;line-height:1.8;">
            <b>Gross Margin Delta vs Standard:</b> {gross_margin - std_gm:+.1f} pt ({format_curr(gross_profit - std_gp)})<br>
            <b>Effective Unit Realized Price:</b> {format_curr(unit_final)} / unit<br>
            <b>Applicable Policy Tier:</b> {tier_name} (Max: {allowed_disc:.1f}%)<br>
            <b>Logistics Break-Even Volume:</b> {be_qty if be_qty else 'N/A'} units
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 10. VIEW: RISK & SCENARIO SIMULATOR
# ==============================================================================
elif nav_selection == "⛛ Risk & Scenario Simulator":
    st.title("🎯 Commercial Risk & Financial Scenario Simulator")
    st.caption("Stress-test global revenues, credit exposures, and shipping delays across currency fluctuations and market shocks.")

    preset_choice = st.radio("Simulation Model", ["Base Case (Neutral)", "Downside Stress Test", "Upside Expansion", "Custom Parameters"], horizontal=True)

    if preset_choice == "Base Case (Neutral)":
        p_fx, p_dem, p_disc, p_delay, p_dflt = 0.0, 0.0, 0.0, 0, 1.0
    elif preset_choice == "Downside Stress Test":
        p_fx, p_dem, p_disc, p_delay, p_dflt = 12.0, -15.0, 6.0, 30, 5.0
    elif preset_choice == "Upside Expansion":
        p_fx, p_dem, p_disc, p_delay, p_dflt = -5.0, 12.0, 0.0, 0, 0.5
    else:
        p_fx, p_dem, p_disc, p_delay, p_dflt = 0.0, 0.0, 0.0, 0, 1.0

    s_c1, s_c2 = st.columns([1.2, 1.4])
    with s_c1:
        st.markdown("#### Scenario Variables")
        sim_fx = st.slider("USD Exchange Rate Volatility (%)", -15.0, 15.0, float(p_fx), 1.0)
        sim_demand = st.slider("Market Demand Shock (%)", -30.0, 30.0, float(p_dem), 1.0)
        sim_disc = st.slider("Pricing Pressure / Competitor Discount (%)", 0.0, 20.0, float(p_disc), 1.0)
        sim_delay = st.slider("Supply Chain & Payment Delay (Days)", 0, 90, int(p_delay), 5)
        sim_dflt = st.slider("Distributor Default Probability (%)", 0.0, 15.0, float(p_dflt), 0.5)

    def run_sim(base_r, gm, fx_v, dem_v, disc_v, delay_v, dflt_v):
        demand = 1.0 + dem_v / 100.0
        price = 1.0 - disc_v / 100.0
        fx = 1.0 - (fx_v / 100.0) * 0.45
        rev = base_r * demand * price * fx
        cogs = base_r * (1.0 - gm) * demand
        logistics = rev * 0.0006 * delay_v
        recv = rev * (60.0 + delay_v) / 365.0
        loss = recv * (dflt_v / 100.0)
        gp = rev - cogs - logistics - loss
        margin = (gp / rev * 100.0) if rev > 0 else 0.0
        be = ((base_r * 0.18) / (gp / rev)) if gp > 0 and rev > 0 else None
        return {'rev': rev, 'gp': gp, 'margin': margin, 'recv': recv, 'loss': loss, 'be': be}

    base_gm = (avg_margin or 40.0) / 100.0
    sim_res = run_sim(total_rev, base_gm, sim_fx, sim_demand, sim_disc, sim_delay, sim_dflt)
    base_res = run_sim(total_rev, base_gm, 0, 0, 0, 0, 1.0)

    with s_c2:
        st.markdown("#### Projected Financial Impact")
        r1, r2 = st.columns(2)
        r_delta = ((sim_res['rev'] - base_res['rev']) / base_res['rev'] * 100.0) if base_res['rev'] > 0 else 0.0
        gp_delta = ((sim_res['gp'] - base_res['gp']) / base_res['gp'] * 100.0) if base_res['gp'] > 0 else 0.0
        
        r1.metric("Projected Revenue", format_curr(sim_res['rev']), delta=f"{r_delta:+.1f}% vs Base")
        r2.metric("Projected Gross Profit", format_curr(sim_res['gp']), delta=f"{gp_delta:+.1f}% vs Base")
        
        r3, r4 = st.columns(2)
        r3.metric("Projected Gross Margin", f"{sim_res['margin']:.1f}%", delta=f"{sim_res['margin'] - base_res['margin']:+.1f} pt")
        r4.metric("Receivables Exposure", format_curr(sim_res['recv']), delta=f"Expected Loss: {format_curr(sim_res['loss'])}", delta_color="inverse")

        # Scenario comparison chart
        scenarios = {
            'Base Case': base_res,
            'Downside': run_sim(total_rev, base_gm, 12, -15, 6, 30, 5),
            'Upside': run_sim(total_rev, base_gm, -5, 12, 0, 0, 0.5),
            'Simulated Model': sim_res
        }
        sc_df = pd.DataFrame([
            {'Scenario': k, 'Revenue': v['rev'], 'Gross Profit': v['gp']} for k, v in scenarios.items()
        ])
        fig_sc = px.bar(sc_df, x='Scenario', y=['Revenue', 'Gross Profit'], barmode='group', color_discrete_map={'Revenue': '#7C93BD', 'Gross Profit': '#13A6B5'})
        fig_sc.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)


# ==============================================================================
# 11. VIEW: DISTRIBUTOR PERFORMANCE EVALUATOR
# ==============================================================================
elif nav_selection == "▲ Performance Evaluator":
    st.title("📈 Distributor Operational & Strategic Performance Scorecard")
    st.caption("Quantitative weighted assessment across sales commitments, clinical expansion, and market capture.")

    active_names = [d['name'] for d in active_rows]
    eval_dist_name = st.selectbox("Select Partner to Evaluate", active_names)
    eval_d = next(d for d in active_rows if d['name'] == eval_dist_name)

    # Synthetic performance factors
    def clamp(v):
        return max(0, min(100, v))
        
    mkt_g = 6.0 + rnd_val(eval_d['id'], 31) * 8.0
    s_target = clamp(eval_d['target_pct'])
    s_growth = clamp(50.0 + eval_d['growth_pct'] * 1.6)
    s_repeat = round(55.0 + rnd_val(eval_d['id'], 32) * 40.0)
    s_clinic = round(40.0 + rnd_val(eval_d['id'], 33) * 50.0)
    s_mkt = clamp(50.0 + (eval_d['growth_pct'] - mkt_g) * 2.5)

    scores = [
        {"Factor": "Target Commitment Attainment", "Score": s_target, "Weight": "25%", "Detail": f"{eval_d['target_pct']}% of agreed minimum"},
        {"Factor": "Annual Sales Growth Rate", "Score": s_growth, "Weight": "20%", "Detail": f"{eval_d['growth_pct']:+.1f}% YoY net change"},
        {"Factor": "Repeat Consumables Reorders", "Score": s_repeat, "Weight": "20%", "Detail": "Clinical consumable reorder frequency"},
        {"Factor": "New IVF Clinic Acquisition", "Score": s_clinic, "Weight": "20%", "Detail": "Customer conversion & installed base"},
        {"Factor": "Outperformance vs Local Market", "Score": s_mkt, "Weight": "15%", "Detail": f"Local market baseline ≈ {mkt_g:.1f}%"}
    ]
    overall_score = round(sum(s['Score'] for s in scores) / len(scores))
    
    if overall_score >= 80:
        band = "Strong Commercial Performer (Tier A)"
        band_color = "#5FD3AE"
    elif overall_score >= 65:
        band = "Solid & Stable Partner (Tier B)"
        band_color = "#4FD1DE"
    elif overall_score >= 50:
        band = "Mixed Performance – Review Recommended (Tier C)"
        band_color = "#F0BC57"
    else:
        band = "Critical Attention Required (Tier D)"
        band_color = "#FF8A96"

    ev1, ev2 = st.columns([1, 2])
    with ev1:
        st.markdown(f"""
        <div class="kpi-container" style="text-align:center;padding:24px;">
            <div class="kpi-header">Overall Rating Score</div>
            <div style="font-size:48px;font-weight:800;color:{band_color};">{overall_score}</div>
            <div style="font-size:14px;color:#CCD7E6;font-weight:600;margin-top:6px;">{band}</div>
        </div>
        """, unsafe_allow_html=True)
    with ev2:
        for item in scores:
            st.write(f"**{item['Factor']}** ({item['Weight']}) — `{item['Detail']}`")
            st.progress(item['Score'] / 100.0, text=f"{item['Score']} / 100")



# ==============================================================================
# 12. VIEW: STRATEGIC DECISION CENTER
# ==============================================================================
elif nav_selection == "⚖ Strategic Decision Center":
    st.title("⚖️ Strategic Decision Center")
    st.caption("Configurable weighted multi-criteria assessment framework for critical corporate authorizations and commercial expansion.")

    st.markdown("""
    <div style="background:rgba(19,166,181,0.08);border:1px solid rgba(19,166,181,0.3);padding:14px 18px;border-radius:12px;margin-bottom:18px;">
        <span style="font-weight:600;color:#4FD1DE;">💡 Corporate Governance Directive:</span>
        <span style="font-size:13px;color:#CCD7E6;">This decision-support module evaluates multi-dimensional trade-offs across historical performance, clinical market capacity, unit profitability, regulatory readiness, and systemic risk. All recommendations are advisory and require formal executive sign-off.</span>
    </div>
    """, unsafe_allow_html=True)

    c_d1, c_d2 = st.columns([1.2, 1])
    with c_d1:
        decision_type = st.selectbox(
            "Select Strategic Decision Mandate",
            [
                "Approve Extraordinary Distributor Discount",
                "Grant Exclusive Territorial Distribution Rights",
                "Appoint & Contract New Prospective Distributor",
                "Prioritize Market for Direct Clinical Expansion",
                "Authorize High-Volume Consumables Allocation"
            ]
        )
    with c_d2:
        if decision_type == "Appoint & Contract New Prospective Distributor":
            candidate_list = [d['name'] + f" ({d['country_name']})" for d in dists_with_rev if d['status'] == 'prospect']
            target_selection = st.selectbox("Select Candidate Entity", candidate_list if candidate_list else ["No prospects available"])
            target_d = next((d for d in dists_with_rev if d['name'] in target_selection), None)
            target_c = country_map.get(target_d['id']) if target_d else country_df.iloc[0]
        elif decision_type == "Prioritize Market for Direct Clinical Expansion":
            target_country_name = st.selectbox("Select Target Market", country_df['name'].tolist())
            target_c = country_df[country_df['name'] == target_country_name].iloc[0]
            target_d = dist_by_code.get(target_c['code'], None)
        else:
            target_dist_name = st.selectbox("Select Distributor Subject", [d['name'] + f" ({d['country_name']})" for d in active_rows])
            target_d = next((d for d in active_rows if d['name'] in target_dist_name), active_rows[0])
            target_c = country_map.get(target_d['id'], country_df.iloc[0])

    st.markdown("#### ⚙️ Weighted Multi-Criteria Framework")
    st.caption("Adjust corporate weightings for each strategic dimension (weights auto-normalize to 100%).")
    
    with st.expander("Configure Evaluation Criteria Weights & Thresholds", expanded=False):
        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            w_sales = st.slider("Historical Sales & Target Commitment", 0, 50, 20, 5)
            w_market = st.slider("Country IVF Market Potential & Cycles", 0, 50, 20, 5)
        with w_col2:
            w_profit = st.slider("Financial Profitability & Margin Health", 0, 50, 20, 5)
            w_ops = st.slider("Operational Capability & Regulatory (LOA)", 0, 50, 15, 5)
        with w_col3:
            w_risk = st.slider("Risk Mitigation (Currency/Payment/Political)", 0, 50, 15, 5)
            w_strat = st.slider("Strategic Market Value & Defense", 0, 50, 10, 5)
            
    raw_weight_sum = w_sales + w_market + w_profit + w_ops + w_risk + w_strat
    if raw_weight_sum == 0:
        raw_weight_sum = 100
        w_sales, w_market, w_profit, w_ops, w_risk, w_strat = 20, 20, 20, 15, 15, 10
        
    nw_sales = w_sales / raw_weight_sum
    nw_market = w_market / raw_weight_sum
    nw_profit = w_profit / raw_weight_sum
    nw_ops = w_ops / raw_weight_sum
    nw_risk = w_risk / raw_weight_sum
    nw_strat = w_strat / raw_weight_sum

    # Dynamic Scoring Logic
    # 1. Sales Performance
    if target_d and target_d['status'] == 'active':
        s_sales = min(100, max(0, round(target_d['target_pct'] * 0.65 + target_d['growth_pct'] * 1.5)))
    elif target_d and target_d['status'] == 'prospect':
        s_sales = 45 # baseline for vetted prospect
    else:
        s_sales = 50

    # 2. Market Potential
    s_market = min(100, max(0, round(target_c['score'] * 0.7 + (target_c['cycles'] / 50000.0) * 30)))

    # 3. Financial Profitability
    if target_d and target_d['status'] == 'active':
        s_profit = min(100, max(0, round((target_d['margin_pct'] / 50.0) * 85 + (target_d['revenue'] / 2500000.0) * 15)))
    else:
        s_profit = 60 # baseline assumption for new contract

    # 4. Operational Capability & LOA
    if target_d and target_d['loa'] == 'Valid':
        loa_pts = 95
    elif target_d and target_d['loa'] == 'In renewal':
        loa_pts = 55
    else:
        loa_pts = 35
    s_ops = loa_pts

    # 5. Risk Mitigation
    risk_lookup = {'Low': 90, 'Medium': 65, 'High': 30, 'Not rated': 50}
    r_val = target_d['risk'] if (target_d and target_d['risk'] != 'Not rated') else target_c['risk']
    s_risk = risk_lookup.get(r_val, 60)

    # 6. Strategic Value
    strat_pts = 65
    if target_d and target_d['exclusivity'] == 'Effective':
        strat_pts += 15
    if target_c['icsi_pct'] >= 75:
        strat_pts += 10
    s_strat = min(100, strat_pts)

    # Criteria breakdown
    criteria_data = [
        {"Criterion": "Historical Performance", "Score": s_sales, "WeightPct": nw_sales * 100, "Contrib": s_sales * nw_sales, "Detail": f"Target met: {target_d['target_pct']}%" if target_d else "N/A"},
        {"Criterion": "IVF Market Potential", "Score": s_market, "WeightPct": nw_market * 100, "Contrib": s_market * nw_market, "Detail": f"{target_c['clinics']} clinics, {target_c['cycles']:,} cycles"},
        {"Criterion": "Margin & Profitability", "Score": s_profit, "WeightPct": nw_profit * 100, "Contrib": s_profit * nw_profit, "Detail": f"Gross Margin: {target_d['margin_pct']:.1f}%" if target_d else "Target: 40%"},
        {"Criterion": "Operational & LOA", "Score": s_ops, "WeightPct": nw_ops * 100, "Contrib": s_ops * nw_ops, "Detail": f"LOA: {target_d['loa'] if target_d else 'Unsubmitted'}"},
        {"Criterion": "Risk Mitigation", "Score": s_risk, "WeightPct": nw_risk * 100, "Contrib": s_risk * nw_risk, "Detail": f"Risk Profile: {r_val}"},
        {"Criterion": "Strategic Value", "Score": s_strat, "WeightPct": nw_strat * 100, "Contrib": s_strat * nw_strat, "Detail": f"High ICSI Activity ({target_c['icsi_pct']}%)"}
    ]
    
    total_score = round(sum(c['Contrib'] for c in criteria_data))

    # Determine Recommendation & Band
    if total_score >= 80:
        band_title = "APPROVED: HIGH CONFIDENCE"
        band_badge = "chip-ok"
        band_col = "#5FD3AE"
        rec_action = f"Recommendation: Proceed with authorizing '{decision_type}'. Subject meets corporate growth and compliance thresholds."
    elif total_score >= 65:
        band_title = "CONDITIONAL APPROVAL: SAFEGUARDS REQUIRED"
        band_badge = "chip-info"
        band_col = "#4FD1DE"
        rec_action = f"Recommendation: Grant provisional approval for '{decision_type}' subject to milestone commitments, minimum quarterly volume, and revised payment terms."
    elif total_score >= 50:
        band_title = "REVIEW REQUIRED: EXECUTIVE ESCALATION"
        band_badge = "chip-wn"
        band_col = "#F0BC57"
        rec_action = f"Recommendation: Escalate to Senior Commercial Committee. Commercial trade-offs exist between margin concession and market risk."
    else:
        band_title = "REJECTED: SIGNIFICANT RISK EXPOSURE"
        band_badge = "chip-err"
        band_col = "#FF8A96"
        rec_action = f"Recommendation: Do not authorize '{decision_type}'. Partner performance or market instability violates risk boundaries."

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Results Display
    res_l, res_r = st.columns([1, 1.4])
    with res_l:
        st.markdown(f"""
        <div class="kpi-container" style="text-align:center;padding:26px;">
            <div class="kpi-header">Strategic Assessment Score</div>
            <div style="font-size:52px;font-weight:800;color:{band_col};letter-spacing:-0.03em;">{total_score}<span style="font-size:22px;color:#8FA3C0;">/100</span></div>
            <div style="margin-top:6px;"><span class="chip {band_badge}" style="font-size:12px;padding:5px 12px;">{band_title}</span></div>
            <div style="font-size:13px;color:#CCD7E6;margin-top:12px;text-align:left;line-height:1.5;background:rgba(0,0,0,0.2);padding:12px;border-radius:8px;">
                <b>Action Mandate:</b><br>{rec_action}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_r:
        st.markdown("#### Dimension Contributions to Final Score")
        contrib_df = pd.DataFrame(criteria_data)
        fig_contrib = px.bar(
            contrib_df, x='Contrib', y='Criterion', orientation='h',
            text=contrib_df['Contrib'].apply(lambda x: f"{x:.1f} pts"),
            color='Contrib',
            color_continuous_scale=[[0, "#243557"], [1, "#13A6B5"]],
            title="Weighted Point Contribution"
        )
        fig_contrib.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=30, b=10),
            coloraxis_showscale=False,
            xaxis=dict(title="Points Contributed (out of 100)", showgrid=True),
            yaxis=dict(title="")
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

    # Constraints, Due Diligence, and Sensitivity Analysis
    st.markdown("---")
    st.subheader("🔍 Due Diligence, Constraints & Sensitivity Audit")
    
    tab_audit1, tab_audit2, tab_audit3 = st.tabs([
        "⚠️ Important Constraints & Deal-Breakers",
        "📋 Due Diligence Verification Checklist",
        "📉 Sensitivity & Stress Analysis"
    ])
    
    with tab_audit1:
        constraints = []
        if target_d:
            if target_d['target_pct'] < 85 and target_d['status'] == 'active':
                constraints.append(f"Target Underachievement: Partner achieved only {target_d['target_pct']}% of its minimum period commitment.")
            if target_d['margin_pct'] < 38.0 and target_d['status'] == 'active':
                constraints.append(f"Low Margin Realization: Current realized gross margin is {target_d['margin_pct']:.1f}% (corporate benchmark: 40%).")
            if target_d['loa'] != 'Valid':
                constraints.append(f"Regulatory Exposure: LOA status is '{target_d['loa']}' — must be validated before formal addenda.")
            if target_d['risk'] == 'High':
                constraints.append("High Credit Risk: Trade receivables must be backed by an Irrevocable Letter of Credit (LC).")
        if target_c['fx_risk'] == 'High':
            constraints.append(f"Currency Volatility: Local territory ({target_c['name']}) has high FX risk; contract must specify USD/EUR peg.")
            
        if constraints:
            for c in constraints:
                st.markdown(f"""
                <div style="background:rgba(229,112,123,0.12);border-left:4px solid #FF8A96;padding:10px 14px;border-radius:6px;margin-bottom:8px;font-size:13px;color:#FFD1D6;">
                    <b>Deal Constraint:</b> {c}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No critical operational or compliance deal-breakers detected for this candidate.")

    with tab_audit2:
        st.markdown("##### Pre-Execution Verification Steps:")
        col_chk1, col_chk2 = st.columns(2)
        with col_chk1:
            st.checkbox("Verified Authorized Ministry of Health Device Registration", value=target_d['loa'] == 'Valid' if target_d else False)
            st.checkbox("Signed Code of Business Conduct & Anti-Corruption Undertaking", value=True)
            st.checkbox("Confirmed Dedicated Technical Cold-Chain & Lab Storage Capabilities", value=True)
        with col_chk2:
            st.checkbox("Audited Key Clinic References & ICSI Specialist Network", value=True)
            st.checkbox("Secured Bank Guarantee / Standby LC for Capital Equipment Orders", value=target_d['risk'] != 'High' if target_d else False)
            st.checkbox("Approved Quarterly Minimum Consumable Purchase Schedule", value=True)

    with tab_audit3:
        st.markdown("##### Model Sensitivity Analysis:")
        st.caption("How the overall recommendation shifts under adverse commercial or regulatory shifts:")
        
        sens_cases = [
            {"Case": "Baseline Assessment", "Score": total_score, "Outcome": band_title},
            {"Case": "-15% Regional Market Demand Shock", "Score": max(0, total_score - 7), "Outcome": "CONDITIONAL APPROVAL" if total_score - 7 >= 65 else "REVIEW REQUIRED"},
            {"Case": "+10% FX Devaluation & Local Price Pressure", "Score": max(0, total_score - 11), "Outcome": "CONDITIONAL APPROVAL" if total_score - 11 >= 65 else "REVIEW REQUIRED"},
            {"Case": "Exclusivity Commitment with 20% Volume Premium", "Score": min(100, total_score + 9), "Outcome": "APPROVED: HIGH CONFIDENCE" if total_score + 9 >= 80 else "CONDITIONAL APPROVAL"}
        ]
        sens_df = pd.DataFrame(sens_cases)
        st.dataframe(sens_df, use_container_width=True, hide_index=True)

    st.markdown("<br><small style='color:#566680;'>⚖️ <i>Corporate Legal Disclaimer: Scores and recommendations generated by the Strategic Decision Center serve strictly as heuristic business-intelligence guidance for internal management evaluation and do not represent binding contractual commitments on behalf of ZEUS Bioscience or its subsidiaries.</i></small>", unsafe_allow_html=True)


# ==============================================================================
# 13. VIEW: EXECUTIVE REPORTS & EXPORT
# ==============================================================================
elif nav_selection == "⇲ Executive Reports & Export":
    st.title("📋 Executive Governance Reports & Export Center")
    st.caption("Generate formal distribution audit briefings and export certified commercial datasets.")

    st.markdown("### Executive Performance Summary")
    st.markdown(f"""
    - **Reporting Period:** {range_sel} {year_sel}
    - **Total Active Commercial Revenue:** {format_curr(total_rev)}
    - **Placed Equipment Volume:** {total_devices:,} units
    - **Consumables Recurring Base:** {format_curr(total_con_rev)}
    - **Weighted Gross Margin:** {avg_margin:.1f}%
    - **Active Partner Network:** {n_active_dists} partners across {n_countries} jurisdictions
    """)

    # Export Dataset
    exp_records = []
    for d in dists_with_rev:
        exp_records.append({
            'Distributor': d['name'],
            'Territory_Code': d['id'],
            'Country': d['country_name'],
            'Region': d['region'],
            'Commercial_Status': d['status'],
            'Net_Revenue_USD': round(d['revenue']),
            'Devices_Placed': d['devices_count'],
            'Consumables_Revenue_USD': round(d['consumables_rev']),
            'Target_Attainment_Pct': d['target_pct'],
            'Gross_Margin_Pct': d['margin_pct'],
            'YoY_Growth_Pct': d['growth_pct'],
            'Market_Penetration_Pct': d.get('pen_pct', 8.5),
            'Credit_Risk_Level': d['risk'],
            'Contract_Expiry': d['expiry'],
            'LOA_Status': d['loa']
        })
    export_df = pd.DataFrame(exp_records)
    csv_data = export_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Certified Commercial Dataset (CSV)",
        data=csv_data,
        file_name=f"ZGMI_Distribution_Audit_{year_sel}.csv",
        mime="text/csv",
        type="primary"
    )
    
    st.dataframe(export_df, use_container_width=True, hide_index=True)

# GSAP Animasyonlu Karşılama Kartı
components.html("""
    <!-- GSAP Kütüphanesini CDN üzerinden çekiyoruz -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
    
    <div id="box" style="
        background: linear-gradient(135deg, #1e3a8a, #3b82f6); 
        color: white; 
        padding: 30px; 
        border-radius: 15px; 
        text-align: center;
        font-family: sans-serif;
        font-weight: bold;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    ">
        🚀 ZEUS INTEL ENGINE AKTİF
    </div>

    <script>
        // Antigravity / Yerçekimsiz havada süzülme efekti
        gsap.fromTo("#box", 
            { y: 50, opacity: 0 }, 
            { y: 0, opacity: 1, duration: 1.5, ease: "power4.out" }
        );
        gsap.to("#box", {
            y: -10,
            repeat: -1,
            yoyo: true,
            duration: 2,
            ease: "sine.inOut",
            delay: 1.5
        });
    </script>
""", height=150)

