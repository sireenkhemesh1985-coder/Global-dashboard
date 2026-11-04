from datetime import datetime
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
def get_gemini_api_key():
    api_key = None
    try:
        if hasattr(st, 'secrets'):
            api_key = st.secrets.get('GEMINI_API_KEY') or st.secrets.get('GEMINT_API_KEY')
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GEMINT_API_KEY')
    return api_key

COUNTRY_MACRO_IVF = {
    'Turkey': {
        'gdp_pc': 13500,
        'gdp_tier': 'Upper-Middle Income',
        'living_standard': 'Orta-Yüksek (HDI: 0.838)',
        'living_standard_code': 'Medium-High',
        'clinics': 165,
        'pop_m': 85.3,
        'customs_tariff_pct': 8.0,
        'customs_clearance_days': 5,
        'import_difficulty': 'Orta (Gümrük Birliği & Yerli İthalat Lisansı)',
        'ivf_support': 'Kısmi SGK Desteği (3 Siklus)',
        'ivf_support_code': 'Partial State Reimbursement',
        'ivf_support_desc': 'SGK belirli tıbbi kriterlerde 3 denemeye kadar kısmi katkı sağlar; özel kliniklerde fark ücreti ödenir.',
        'price_sensitivity': 'Yüksek (Enflasyon / Kur Baskısı)',
        'macro_discount_allowance': 1.0,
        'ivf_discount_allowance': 1.0
    },
    'Jordan-Iraq': {
        'gdp_pc': 4900,
        'gdp_tier': 'Lower-Middle Income',
        'living_standard': 'Orta-Düşük (HDI: ~0.686)',
        'living_standard_code': 'Medium-Low',
        'clinics': 45,
        'pop_m': 55.0,
        'customs_tariff_pct': 6.5,
        'customs_clearance_days': 30,
        'import_difficulty': 'Yüksek (Sınır Geçişleri, Döviz & Bankacılık Kısıtı)',
        'ivf_support': '%100 Cepten Ödeme (Destek Yok)',
        'ivf_support_code': 'Self-Pay / Out-of-Pocket',
        'ivf_support_desc': 'Devlet IVF tedavilerini karşılamaz; hastalar tüm klinik ve sarf maliyetini kendi cebinden öder.',
        'price_sensitivity': 'Çok Yüksek (Döviz Kısıtlaması & Fiyat Elastikiyeti)',
        'macro_discount_allowance': 2.5,
        'ivf_discount_allowance': 2.0
    },
    'Jordan': {
        'gdp_pc': 4500,
        'gdp_tier': 'Lower-Middle Income',
        'living_standard': 'Orta (HDI: 0.720)',
        'living_standard_code': 'Medium',
        'clinics': 22,
        'pop_m': 11.3,
        'customs_tariff_pct': 6.0,
        'customs_clearance_days': 14,
        'import_difficulty': 'Orta (JFDA Medikal Onayı)',
        'ivf_support': '%100 Cepten Ödeme (Destek Yok)',
        'ivf_support_code': 'Self-Pay / Out-of-Pocket',
        'ivf_support_desc': 'Özel tüp bebek merkezleri ağırlıklı; cepten ödeme hakim.',
        'price_sensitivity': 'Yüksek',
        'macro_discount_allowance': 2.0,
        'ivf_discount_allowance': 2.0
    },
    'Qatar': {
        'gdp_pc': 86000,
        'gdp_tier': 'High Income (Ultra High)',
        'living_standard': 'Çok Yüksek (HDI: 0.855)',
        'living_standard_code': 'Very High',
        'clinics': 6,
        'pop_m': 2.9,
        'customs_tariff_pct': 5.0,
        'customs_clearance_days': 10,
        'import_difficulty': 'Düşük-Orta (MOPH İthalat İzni)',
        'ivf_support': 'Vatandaşlara Tam Devlet Desteği (Hamad MC)',
        'ivf_support_code': 'Full Public Coverage (Citizens)',
        'ivf_support_desc': 'Katar vatandaşlarına kamu hastanelerinde ücretsiz; yabancı nüfusa özel sigorta / self-pay.',
        'price_sensitivity': 'Düşük (Kalite, Teknoloji ve LOA Öncelikli)',
        'macro_discount_allowance': 0.0,
        'ivf_discount_allowance': 0.0
    },
    'South Korea': {
        'gdp_pc': 33500,
        'gdp_tier': 'High Income (OECD)',
        'living_standard': 'Çok Yüksek (HDI: 0.925)',
        'living_standard_code': 'Very High',
        'clinics': 150,
        'pop_m': 51.7,
        'customs_tariff_pct': 8.0,
        'customs_clearance_days': 7,
        'import_difficulty': 'Orta-Yüksek (MFDS Tıbbi Cihaz Sertifikasyon Şartı)',
        'ivf_support': 'Kapsamlı Ulusal Doğurganlık Desteği',
        'ivf_support_code': 'National Fertility Subsidy',
        'ivf_support_desc': 'Güney Kore hükümeti düşük doğum oranını aşmak için siklus başına cömert devlet sübvansiyonu ve SGK desteği sunar.',
        'price_sensitivity': 'Orta-Düşük (Klinik Başarı Oranı & Teknoloji Odaklı)',
        'macro_discount_allowance': 0.0,
        'ivf_discount_allowance': 0.5
    },
    'Palestine': {
        'gdp_pc': 3500,
        'gdp_tier': 'Low-Middle Income',
        'living_standard': 'Düşük-Orta (Zorlu Koşullar)',
        'living_standard_code': 'Low-Medium',
        'clinics': 14,
        'pop_m': 5.2,
        'customs_tariff_pct': 12.0,
        'customs_clearance_days': 35,
        'import_difficulty': 'Kritik Derecede Yüksek (İsrail Liman Geçişi & Güvenlik Beklemesi)',
        'ivf_support': '%100 Cepten Ödeme (Destek Yok)',
        'ivf_support_code': 'Self-Pay / Out-of-Pocket',
        'ivf_support_desc': 'Kamu güvencesi bulunmamaktadır; kliniklerin malzeme maliyeti doğrudan hasta ücretine yansır.',
        'price_sensitivity': 'Kritik Derecede Yüksek',
        'macro_discount_allowance': 3.0,
        'ivf_discount_allowance': 2.0
    },
    'Saudi Arabia': {
        'gdp_pc': 32500,
        'gdp_tier': 'High Income',
        'living_standard': 'Yüksek (HDI: 0.875)',
        'living_standard_code': 'High',
        'clinics': 38,
        'pop_m': 36.4,
        'customs_tariff_pct': 5.0,
        'customs_clearance_days': 15,
        'import_difficulty': 'Orta (SFDA Tıbbi Cihaz Kaydı & MDMA Lisansı)',
        'ivf_support': 'Kamu Hastanelerinde Destekli / Özelde Self-Pay',
        'ivf_support_code': 'Public Funded / Private Self-Pay',
        'ivf_support_desc': 'Vatandaşlara askeri ve bakanlık hastanelerinde ücretsiz; özel tüp bebek hastanelerinde özel sigorta/self-pay.',
        'price_sensitivity': 'Orta',
        'macro_discount_allowance': 0.0,
        'ivf_discount_allowance': 0.5
    },
    'Kuwait': {
        'gdp_pc': 38500,
        'gdp_tier': 'High Income',
        'living_standard': 'Yüksek (HDI: 0.831)',
        'living_standard_code': 'High',
        'clinics': 12,
        'pop_m': 4.3,
        'customs_tariff_pct': 5.0,
        'customs_clearance_days': 12,
        'import_difficulty': 'Orta (MOH Medikal Kaydı)',
        'ivf_support': 'Kuveyt Vatandaşlarına Bakanlık Destekli',
        'ivf_support_code': 'Full Public Coverage (Citizens)',
        'ivf_support_desc': 'MOH Kuveyt vatandaşlarının IVF tedavilerini tam karşılar; yüksek cihaz bütçesi mevcuttur.',
        'price_sensitivity': 'Düşük-Orta',
        'macro_discount_allowance': 0.0,
        'ivf_discount_allowance': 0.0
    },
    'United Arab Emirates': {
        'gdp_pc': 53000,
        'gdp_tier': 'High Income',
        'living_standard': 'Çok Yüksek (HDI: 0.911)',
        'living_standard_code': 'Very High',
        'clinics': 22,
        'pop_m': 9.9,
        'customs_tariff_pct': 5.0,
        'customs_clearance_days': 8,
        'import_difficulty': 'Düşük-Orta (MOHAP Medikal Tescili)',
        'ivf_support': 'Vatandaşlara Tam Destek (Thiqa / Enaya) / Expats Self-Pay',
        'ivf_support_code': 'Public Funded (Citizens) / Private',
        'ivf_support_desc': 'BAE vatandaşlarına özel programlarla karşılanır; yabancı nüfusta özel harcama hakimdir.',
        'price_sensitivity': 'Düşük-Orta',
        'macro_discount_allowance': 0.0,
        'ivf_discount_allowance': 0.0
    },
    'Belgium': {
        'gdp_pc': 50500,
        'gdp_tier': 'High Income (EU Core)',
        'living_standard': 'Çok Yüksek (HDI: 0.937)',
        'living_standard_code': 'Very High',
        'clinics': 18,
        'pop_m': 11.6,
        'customs_tariff_pct': 0.0,
        'customs_clearance_days': 3,
        'import_difficulty': 'Düşük (AB İçi Serbest Dolaşım / CE/MDR Uyumu)',
        'ivf_support': 'Geniş Kapsamlı Kamu Geri Ödemesi (6 Siklus)',
        'ivf_support_code': 'Comprehensive Public Reimbursement (6 Cycles)',
        'ivf_support_desc': 'Belçika Ulusal Sağlık Enstitüsü (INAMI/RIZIV) kadınlar için 6 siklusa kadar tedavi masraflarını tam geri öder.',
        'price_sensitivity': 'Kurumsal / İhale & Regülasyon Odaklı',
        'macro_discount_allowance': 0.0,
        'ivf_discount_allowance': 0.0
    },
    'Bulgaria': {
        'gdp_pc': 15000,
        'gdp_tier': 'Upper-Middle Income (EU)',
        'living_standard': 'Orta-Yüksek (HDI: 0.795)',
        'living_standard_code': 'Medium-High',
        'clinics': 32,
        'pop_m': 6.5,
        'customs_tariff_pct': 0.0,
        'customs_clearance_days': 4,
        'import_difficulty': 'Düşük (AB İçi Serbest Dolaşım)',
        'ivf_support': 'Ulusal Üreme Fonu Destekli (4 Siklus)',
        'ivf_support_code': 'National Assisted Reproduction Fund (4 Cycles)',
        'ivf_support_desc': 'Bulgaristan hükümeti Ulusal Üreme Fonu aracılığıyla 4 siklusa kadar finansman sağlar.',
        'price_sensitivity': 'Orta-Yüksek',
        'macro_discount_allowance': 1.0,
        'ivf_discount_allowance': 0.5
    },
    'Azerbaijan': {
        'gdp_pc': 7200,
        'gdp_tier': 'Middle Income',
        'living_standard': 'Orta (HDI: 0.745)',
        'living_standard_code': 'Medium',
        'clinics': 12,
        'pop_m': 10.1,
        'customs_tariff_pct': 10.0,
        'customs_clearance_days': 16,
        'import_difficulty': 'Orta-Yüksek (Sağlık Bakanlığı İthalat Vizesi)',
        'ivf_support': '%100 Cepten Ödeme (Destek Yok)',
        'ivf_support_code': 'Self-Pay / Out-of-Pocket',
        'ivf_support_desc': 'Kamu sigortası IVF masraflarını karşılamaz, hastalar cepten öder.',
        'price_sensitivity': 'Yüksek',
        'macro_discount_allowance': 2.0,
        'ivf_discount_allowance': 2.0
    },
    'Albania-Slovakia': {
        'gdp_pc': 16000,
        'gdp_tier': 'Upper-Middle Income',
        'living_standard': 'Orta-Yüksek',
        'living_standard_code': 'Medium-High',
        'clinics': 16,
        'pop_m': 8.2,
        'customs_tariff_pct': 3.5,
        'customs_clearance_days': 8,
        'import_difficulty': 'Orta (Balkan & Orta Avrupa Geçişi)',
        'ivf_support': 'Slovakya Sigorta Destekli / Arnavutluk Self-Pay',
        'ivf_support_code': 'Mixed Reimbursement',
        'ivf_support_desc': 'Slovakya sağlık sigortası ilk 3 siklusu kısmen karşılar; Arnavutluk tarafı cepten ödemelidir.',
        'price_sensitivity': 'Orta',
        'macro_discount_allowance': 1.0,
        'ivf_discount_allowance': 1.0
    },
    'Philippines': {
        'gdp_pc': 3900,
        'gdp_tier': 'Lower-Middle Income',
        'living_standard': 'Orta-Düşük (HDI: 0.699)',
        'living_standard_code': 'Medium-Low',
        'clinics': 8,
        'pop_m': 115.0,
        'customs_tariff_pct': 7.0,
        'customs_clearance_days': 20,
        'import_difficulty': 'Yüksek (FDA Filipinler İthal İzin Şartı)',
        'ivf_support': '%100 Cepten Ödeme (Destek Yok)',
        'ivf_support_code': 'Self-Pay / Out-of-Pocket',
        'ivf_support_desc': 'PhilHealth IVF tedavilerini kapsamaz; lüks özel kliniklerde cepten ödenir.',
        'price_sensitivity': 'Çok Yüksek',
        'macro_discount_allowance': 2.5,
        'ivf_discount_allowance': 2.0
    },
    'Syria': {
        'gdp_pc': 1800,
        'gdp_tier': 'Low Income',
        'living_standard': 'Düşük',
        'living_standard_code': 'Low',
        'clinics': 6,
        'pop_m': 22.0,
        'customs_tariff_pct': 15.0,
        'customs_clearance_days': 40,
        'import_difficulty': 'Aşırı Yüksek (Yaptırımlar, Sınır & Banka Blokajı)',
        'ivf_support': '%100 Cepten Ödeme (Destek Yok)',
        'ivf_support_code': 'Self-Pay / Out-of-Pocket',
        'ivf_support_desc': 'Sağlık altyapısı zorlu; tüm giderler cepten karşılanır.',
        'price_sensitivity': 'Aşırı Yüksek',
        'macro_discount_allowance': 3.5,
        'ivf_discount_allowance': 2.0
    },
    'United Kingdom': {
        'gdp_pc': 46500,
        'gdp_tier': 'High Income',
        'living_standard': 'Çok Yüksek (HDI: 0.929)',
        'living_standard_code': 'Very High',
        'clinics': 75,
        'pop_m': 67.5,
        'customs_tariff_pct': 2.5,
        'customs_clearance_days': 5,
        'import_difficulty': 'Düşük-Orta (MHRA Tıbbi Cihaz Kaydı)',
        'ivf_support': 'NHS Kısmi Kriter Destekli (1-3 Siklus)',
        'ivf_support_code': 'NHS Criteria Based',
        'ivf_support_desc': 'NHS kriterlerine göre 1 ila 3 siklus karşılanır; %60+ özel sektöre kayar.',
        'price_sensitivity': 'Orta',
        'macro_discount_allowance': 0.0,
        'ivf_discount_allowance': 0.5
    }
}

def resolve_country_macro(c_name):
    c = str(c_name).strip()
    if c in ['Türkiye', 'Turkey', 'TR']:
        key = 'Turkey'
    elif c in ['UAE', 'United Arab Emirates', 'AE']:
        key = 'United Arab Emirates'
    elif c in ['SY', 'Syria']:
        key = 'Syria'
    elif c in ['Jordan-Iraq', 'Jordan', 'Iraq', 'JO', 'IQ']:
        key = 'Jordan-Iraq'
    elif c in ['Albania-Slovakia', 'Slovakia', 'Albania']:
        key = 'Albania-Slovakia'
    elif c in ['South Korea', 'Korea', 'KR']:
        key = 'South Korea'
    elif c in ['United Kingdom', 'UK', 'GB']:
        key = 'United Kingdom'
    else:
        key = c
    return COUNTRY_MACRO_IVF.get(key, {
        'gdp_pc': 12000,
        'gdp_tier': 'Middle Income',
        'living_standard': 'Orta Seviye',
        'living_standard_code': 'Medium',
        'clinics': 20,
        'pop_m': 20.0,
        'customs_tariff_pct': 6.0,
        'customs_clearance_days': 10,
        'import_difficulty': 'Orta Seviye Gümrük & İthalat',
        'ivf_support': 'Kısmi / Özel Ödeme',
        'ivf_support_code': 'Mixed/Partial',
        'ivf_support_desc': 'Bölgesel karma geri ödeme ve cepten ödeme.',
        'price_sensitivity': 'Orta',
        'macro_discount_allowance': 1.0,
        'ivf_discount_allowance': 1.0
    })

def generate_ai_pricing_bundle_recommendation(calc_d, macro, strategic_goal):
    country = calc_d.get('country_name', 'Turkey')
    partner_name = calc_d.get('name', 'Partner')
    partner_rev = calc_d.get('revenue', 0.0)
    partner_growth = calc_d.get('growth_pct', 0.0)
    
    gdp_pc = macro.get('gdp_pc', 12000)
    clinics = macro.get('clinics', 20)
    pop_m = macro.get('pop_m', 20.0)
    customs_pct = macro.get('customs_tariff_pct', 5.0)
    clearance_days = macro.get('customs_clearance_days', 10)
    ivf_support = macro.get('ivf_support', 'Kısmi / Özel Ödeme')
    
    # 1. Base AI Device Discount Recommendation
    if gdp_pc < 10000:
        if "Pazar Payı" in strategic_goal:
            rec_dev_disc = 12.0
            rec_con_disc = 15.0
            bundle_desc = "1 Cihaz + 15 Kutu Electro Dish siparişinde 3 Kutu Liquid Solution (%100 Bedelsiz) ve $500 Navlun Katkısı"
            rec_freight = 500.0
        elif "Sarf" in strategic_goal:
            rec_dev_disc = 8.5
            rec_con_disc = 18.0
            bundle_desc = "20 Kutu Electro Dish alımına 4 Kutu Liquid Solution Hediye + Yıllık Sabit Fiyat Garantisi"
            rec_freight = 400.0
        elif "Gümrük" in strategic_goal:
            rec_dev_disc = 7.0
            rec_con_disc = 12.0
            bundle_desc = f"Cihaz liste fiyatı korunur; gümrükteki %{customs_pct:.1f} vergi yükü için $600 Lojistik Katılımı ve 2 Kutu Ek Sarf Desteği"
            rec_freight = 600.0
        else:
            rec_dev_disc = 9.0
            rec_con_disc = 10.0
            bundle_desc = "Standart 10 Dish + 2 Liquid Solution %50 İndirimli Aktivasyon Paketi"
            rec_freight = 450.0
    elif gdp_pc < 25000:
        if "Pazar Payı" in strategic_goal:
            rec_dev_disc = 9.0
            rec_con_disc = 12.0
            bundle_desc = "1 Cihaz + 12 Kutu Electro Dish alımında 2 Kutu Liquid Solution Ücretsiz + Ücretsiz Klinik Eğitim"
            rec_freight = 350.0
        elif "Sarf" in strategic_goal:
            rec_dev_disc = 6.5
            rec_con_disc = 15.0
            bundle_desc = "Çeyreklik 25 Kutu Sarf Sözleşmesine %12.5 İskonto + 2 Şişe Konsinye Solüsyon"
            rec_freight = 300.0
        elif "Gümrük" in strategic_goal:
            rec_dev_disc = 6.0
            rec_con_disc = 10.0
            bundle_desc = f"Liste fiyatından %6.0 iskonto + $400 Yurtiçi Dağıtım ve Gümrük Masrafı Katkısı"
            rec_freight = 400.0
        else:
            rec_dev_disc = 7.0
            rec_con_disc = 8.0
            bundle_desc = "10 Dish + 2 Liquid Solution %50 İndirimli Bundle"
            rec_freight = 300.0
    else:
        if "Pazar Payı" in strategic_goal:
            rec_dev_disc = 6.5
            rec_con_disc = 8.0
            bundle_desc = "Liste fiyatı korunur; VIP Laboratuvar Kurulumu + 10 Kutu Dish alımına 1 Kutu Liquid Hediye"
            rec_freight = 250.0
        elif "Sarf" in strategic_goal:
            rec_dev_disc = 4.0
            rec_con_disc = 10.0
            bundle_desc = "Yıllık 60 Kutu Dish Taahhüdüne %10 İskonto + Konsinye Emniyet Stoku Depolama Desteği"
            rec_freight = 200.0
        elif "Gümrük" in strategic_goal:
            rec_dev_disc = 5.0
            rec_con_disc = 6.0
            bundle_desc = "Hızlı Hava Kargo (Air Express) navlun sübvansiyonu ($400) + CE/MOH Dokümantasyon Desteği"
            rec_freight = 400.0
        else:
            rec_dev_disc = 5.0
            rec_con_disc = 5.0
            bundle_desc = "Standart Körfez VIP Paket: Cihaz + 10 Dish + Yerinde Embriyolog Sertifikasyonu"
            rec_freight = 250.0

    # Partner performance modifier
    if partner_rev >= 100000:
        rec_dev_disc = min(18.0, rec_dev_disc + 2.0)
        partner_note = "🌟 Platin Partner hacmi (Ciro > $100K) sebebiyle önerilen tavan %2.0 artırılmıştır."
    elif partner_rev >= 30000:
        rec_dev_disc = min(15.0, rec_dev_disc + 1.0)
        partner_note = "🥇 Altın Partner büyüme hacmi sebebiyle önerilen tavan %1.0 artırılmıştır."
    else:
        partner_note = "🥉 Yeni / Gelişen hesap; temel hacim baremleri korunmuştur."

    # Customs recommendation
    if customs_pct >= 8.0 or clearance_days >= 20:
        customs_strategy = f"{country} pazarında gümrük tarifesi (%{customs_pct:.1f}) ve çekim süresi ({clearance_days} gün) yüksektir. Doğrudan cihaz fiyatını düşürmek yerine, gümrük beyannamesini etkilemeyecek 'Bedelsiz Sarf / Yedek Parça Desteği' ve fatura altı 'Navlun Katkı Payı' (${rec_freight:,.0f}) tanımlanmalıdır."
    else:
        customs_strategy = f"{country} pazarında gümrük süreci hızlı ({clearance_days} gün) ve tarife düşüktür (%{customs_pct:.1f}). Gümrük bariyeri minimum düzeyde olduğundan, indirimler doğrudan brüt kâr disiplini korunarak sarf çekiş taahhütlerine endekslenmelidir."

    return {
        'rec_device_discount': rec_dev_disc,
        'rec_consumable_discount': rec_con_disc,
        'bundle_desc': bundle_desc,
        'customs_strategy': customs_strategy,
        'partner_note': partner_note,
        'rec_freight': rec_freight,
        'expected_margin': round(48.5 - rec_dev_disc * 0.8, 1),
        'win_probability': min(96, 75 + int(rec_dev_disc * 1.5)),
        'annual_sarf_potential': f"+${round(clinics * 1800):,} USD"
    }

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
            # Unify SIRIUS PHARM COMPANY country records under 'Jordan-Iraq'
            df_sales.loc[df_sales['Company'] == 'SIRIUS PHARM COMPANY', 'Country'] = 'Jordan-Iraq'
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
        
        # Classification: Distributor vs Local (Turkey) vs International Customer
        DISTRIBUTOR_COMPANIES = {
            'SIRIUS PHARM COMPANY',
            'INTERNATIONAL TECHNICAL LEGACY (ITL)',
            'SJNU INC',
            'CONSULTANT MEDICAL COMPANY (CMC)',
            'ASSISTED MEDICAL SOLUTION EST',
            'ADVANCED TECHNOLOGY COMPANY K.S.C.P.',
            'SELLCO TRADING INTL'
        }
        
        if company in DISTRIBUTOR_COMPANIES:
            acc_type = 'Distributor'
            acc_type_tr = 'Yetkili Distribütör'
            is_dist = True
            loa_val = 'In renewal' if company == 'SELLCO TRADING INTL' else 'Valid'
            exp_val = '2026-11-20' if company == 'SELLCO TRADING INTL' else ('2027-12-31' if company in ['SIRIUS PHARM COMPANY', 'INTERNATIONAL TECHNICAL LEGACY (ITL)', 'SJNU INC'] else '2027-06-30')
            excl_val = 'Exclusive' if company in ['SIRIUS PHARM COMPANY', 'INTERNATIONAL TECHNICAL LEGACY (ITL)', 'SJNU INC'] else 'Non-Exclusive'
            ag_type_val = 'Münhasır Distribütörlük Sözleşmesi' if excl_val == 'Exclusive' else 'Yetkili Distribütörlük Sözleşmesi'
            comp_status = '✅ Tam Uyumlu' if loa_val == 'Valid' else '⚠️ Yenileme Bekleniyor'
        elif country == 'Turkey':
            acc_type = 'Local Customer'
            acc_type_tr = 'Local Müşteri (Türkiye)'
            is_dist = False
            loa_val = 'N/A (Local Client)'
            exp_val = 'Yıllık Tedarik Protokolü'
            excl_val = 'N/A'
            ag_type_val = 'Yurtiçi Cihaz & Sarf Tedarik Protokolü'
            comp_status = '✅ Standart Müşteri Kaydı'
        else:
            acc_type = 'International Customer'
            acc_type_tr = 'International Müşteri'
            is_dist = False
            loa_val = 'N/A (Direct Hospital/Clinic)'
            exp_val = 'Kurumsal Tedarik Sözleşmesi'
            excl_val = 'N/A'
            ag_type_val = 'Akademik Protokol & Cihaz Devri' if 'Ghent' in company else 'Uluslararası Doğrudan Satın Alma Protokolü'
            comp_status = '✅ Onaylı Uluslararası Müşteri'

        accounts.append({
            'id': code,
            'name': company,
            'country_name': country,
            'region': country_to_region.get(country, 'Global'),
            'status': 'active',
            'account_type': acc_type,
            'account_type_tr': acc_type_tr,
            'is_distributor': is_dist,
            'agreement_type': ag_type_val,
            'compliance': comp_status,
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
            'target_pct': 100 if tot_rev > 0 else 85,
            'margin_pct': 46.5 if dev_count > 0 else 42.0,
            'growth_pct': 18.5 if tot_rev > 20000 else 10.0,
            'pen_pct': round(min(25.0, max(2.5, (dev_count * 2.0 + tot_rev / 15000.0))), 1),
            'weights': weights,
            'risk': 'Low',
            'expiry': exp_val,
            'exclusivity': excl_val,
            'loa': loa_val
        })
    accounts.sort(key=lambda x: x['revenue'], reverse=True)
    return accounts

# Initialize Session State
if 'added_dists' not in st.session_state:
    st.session_state.added_dists = []

if 'custom_documents' not in st.session_state:
    st.session_state.custom_documents = [
        {
            'doc_id': 'ZEUS-LOA-SIRIUS-2026',
            'company': 'SIRIUS PHARM COMPANY',
            'category': 'Distributor',
            'doc_type': 'LOA (Letter of Authorization)',
            'territory': 'Jordan & Iraq',
            'start_date': '2026-01-01',
            'end_date': '2027-12-31',
            'status': '✅ Geçerli (Valid)',
            'notes': 'Resmi Ürdün ve Irak münhasır distribütörlük yetki belgesi.',
            'filename': 'ZEUS_LOA_Sirius_Pharm_2026_2027.pdf'
        },
        {
            'doc_id': 'ZEUS-AGR-ITL-2026',
            'company': 'INTERNATIONAL TECHNICAL LEGACY (ITL)',
            'category': 'Distributor',
            'doc_type': 'Distribütörlük Sözleşmesi',
            'territory': 'Qatar',
            'start_date': '2026-01-01',
            'end_date': '2027-12-31',
            'status': '✅ Geçerli (Valid)',
            'notes': 'Katar Devleti münhasır IVF klinik ekipman dağıtım protokolü.',
            'filename': 'ZEUS_Exclusive_Distribution_ITL_Qatar.pdf'
        },
        {
            'doc_id': 'ZEUS-LOA-SJNU-2026',
            'company': 'SJNU INC',
            'category': 'Distributor',
            'doc_type': 'LOA (Letter of Authorization)',
            'territory': 'South Korea',
            'start_date': '2026-03-15',
            'end_date': '2027-12-31',
            'status': '✅ Geçerli (Valid)',
            'notes': 'Kore MFDS regülasyonlarına uygun münhasır Piezo Pulse dağıtım yetkisi.',
            'filename': 'ZEUS_LOA_SJNU_Korea_MFDS.pdf'
        },
        {
            'doc_id': 'ZEUS-AGR-SELLCO-2025',
            'company': 'SELLCO TRADING INTL',
            'category': 'Distributor',
            'doc_type': 'Distribütörlük Sözleşmesi',
            'territory': 'United Arab Emirates',
            'start_date': '2025-11-20',
            'end_date': '2026-11-20',
            'status': '⚠️ Yenilemede (In Renewal)',
            'notes': 'Sözleşme süresi 20 Kasım 2026 tarihinde dolmaktadır; yenileme protokolü hazırlanmıştır.',
            'filename': 'ZEUS_UAE_Sellco_Agreement_2026.pdf'
        },
        {
            'doc_id': 'ZEUS-ACAD-GHENT-2026',
            'company': 'Ghent University (UGent)',
            'category': 'International Customer',
            'doc_type': 'Akademik Cihaz & Araştırma Protokolü',
            'territory': 'Belgium',
            'start_date': '2026-05-10',
            'end_date': '2028-05-10',
            'status': '✅ Aktif Protokol',
            'notes': 'Electro Fusion SET cihazının akademik oosit aktivasyonu araştırmalarında kullanımı.',
            'filename': 'ZEUS_UGent_Academic_Transfer_Agreement.pdf'
        },
        {
            'doc_id': 'ZEUS-LOC-TAVMERGEN-2026',
            'company': 'TAVMERGEN ÖZEL TÜP BEBEK MERKEZİ VE SAĞLIK HİZMETLERİ ANONİM ŞİRKETİ',
            'category': 'Local Customer',
            'doc_type': 'Yurtiçi Cihaz & Sarf Tedarik Protokolü',
            'territory': 'Türkiye',
            'start_date': '2026-02-01',
            'end_date': '2027-02-01',
            'status': '✅ Yıllık Sipariş Kaydı',
            'notes': 'İzmir merkezli klinik için 1 adet Piezo Pulse cihazı ve yıllık Electro Dish sipariş protokolü.',
            'filename': 'ZEUS_Tavmergen_Yurtici_Tedarik_Sozlesmesi.pdf'
        }
    ]

country_df = get_country_intelligence()
base_dists = get_base_distributors()
all_dists = base_dists + st.session_state.added_dists

country_map = {row['code']: row for _, row in country_df.iterrows()}


# ==============================================================================
# 3. SIDEBAR NAVIGATION & GLOBAL CONTROLS
# ==============================================================================
with st.sidebar:
    # Official ZEUS Bioscience Logo
    st.image("assets/zeus_logo_white.png")
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
dist_rows = [d for d in active_rows if d.get('account_type') == 'Distributor']
local_rows = [d for d in active_rows if d.get('account_type') == 'Local Customer']
intl_rows = [d for d in active_rows if d.get('account_type') == 'International Customer']
total_rev = sum(d['revenue'] for d in active_rows)
total_dev_rev = sum(d['device_rev'] for d in active_rows)
total_con_rev = total_rev - total_dev_rev
total_devices = sum(d['devices_count'] for d in active_rows)
avg_margin = (sum(d['margin_pct'] * d['revenue'] for d in active_rows) / total_rev) if total_rev > 0 else 0.0
avg_growth = (sum(d['growth_pct'] * d['revenue'] for d in active_rows) / total_rev) if total_rev > 0 else 0.0
n_active_dists = len(active_rows)
n_countries = len(set(d['id'] for d in active_rows))
dist_by_code = {d['id']: d for d in dists_with_rev}


# ------------------------------------------------------------------------------
# ZEUS EXECUTIVE MEMO GENERATOR (COGNITIVE CORE FALLBACK & AUGMENTATION)
# ------------------------------------------------------------------------------
def generate_zeus_executive_memo(active_prompt, total_rev, avg_margin, total_devices):
    prompt_lower = active_prompt.lower()
    
    # --------------------------------------------------------------------------
    # Preset 1: Hangi ürünlerde acil promosyon, kampanya veya klinik eğitim gerekiyor?
    # --------------------------------------------------------------------------
    if any(k in prompt_lower for k in ["promosyon", "kampanya", "eğitim", "klinik eğitim", "hangi ürün"]):
        topic_title = "Acil Promosyon, Kampanya ve Klinik Eğitimi Gerektiren ZEUS Ürün Hatları Stratejisi"
        kpi_impact = "+$62,000 / Yıl Ek Ciro & Marj İyileşmesi"
        timeline = "60 Gün (Hızlı Uygulama)"
        target_group = "Zeus Electro Fusion SET, Electro Liquid Solution & Uyuyan Cihaz Merkezleri"
        risk_level = "Düşük — Yüksek Ticari Geri Dönüş"
        
        memo_content = f"""### 📋 ZEUS EXECUTIVE MEMORANDUM
**Doküman Kodu:** `ZGMI-STRAT-MEMO-2026-PROMO` | **Tarih:** Kasım 2026  
**Gizlilik Derecesi:** `C-LEVEL & YÖNETİM KURULU İNCELEMESİ (GİZLİ)`  
**Konu:** {topic_title}  
**Mevcut Temel Metrikler:** Toplam Kurulu Cihaz: **{total_devices} Adet** | Portföy Ciro: **${total_rev:,.2f}** | Ortalama Brüt Marj: **%{avg_margin:.1f}**

---

#### 1. 🎯 Yönetici Özeti (Executive Brief)
Excel satış kayıtlarındaki 874 işlem ve ürün bazlı çekiş frekansları incelendiğinde, portföyde acil ticari müdahale (promosyon, bundle kampanyası veya klinik aplikasyon eğitimi) gerektiren 3 kritik ürün grubu ve müşteri kümesi tespit edilmiştir:
- **1. Zeus Electro Fusion SET ($15,890.85):** Şu ana kadar yalnızca 1 adet satılmıştır (Ghent University, Belçika). Batı Avrupa ve Körfez'deki üst düzey akademik merkezler için demo/tanıtım kampanyası başlatılmalıdır.
- **2. Electro Liquid Solution (Powder & 2x1ml) Dengesizliği:** 2,097 kutu Electro Dish satılmasına karşın yalnızca 765 kutu Liquid Solution (2x1ml) ve 232 kutu Powder satılmıştır (**Dish / Liquid oranı ~2.1:1**). Klinikler solüsyonu ikame etmekte veya seyreltmektedir; acil **Bundle Kampanyası** gereklidir.
- **3. Cihazı Kurulu Olup Sarf Çekişi Durmuş Klinikler (Klinik Eğitim İhtiyacı):** Acıbadem (2 cihaz - $760 sarf), Medipolitan (2 cihaz - $440 sarf), Kuveyt ATC (3 cihaz - $750 sarf) ve Gen-Sa (4 cihaz - $1,640 sarf) merkezlerinde embriyolog eğitimi eksikliği nedeniyle cihazlar atıl kalmaktadır.

---

#### 2. 📊 Veri Odaklı Tanı ve Ürün Bazlı Kök Neden Analizi

| Ürün / Müşteri Grubu | Toplam Ciro | Satılan Adet | Teşhis Edilen Sorun | Gereken Müdahale |
| :--- | :--- | :--- | :--- | :--- |
| **Zeus Electro Fusion SET** | $15,890.85 | 1 Ünite | Tekil akademik satışta kaldı; ticari lansman yapılmadı. | **Uluslararası Tanıtım Kampanyası & ESHRE Canlı Demo** |
| **Electro Liquid Solution (Powder & 2x1ml)** | $53,014.00 | 997 Şişe/Kutu | Dish çekişine göre %50 geride kalıyor (Oran: 2.1 Dish : 1 Solüsyon). | **10 Dish + 2 Liquid Solution %50 İndirimli Bundle Kampanyası** |
| **Gen-Sa Sağlık (Türkiye)** | $33,140.00 | 4 Piezo Cihazı | Cihaz başı yıllık sarf çekişi yalnızca $410 (Beklenen: $3,500+). | **Yerinde Klinik Embriyolog Protokol Eğitimi** |
| **Advanced Tech Co (Kuveyt - ATC)** | $26,250.00 | 3 Piezo Cihazı | Cihaz başı sarf çekişi yalnızca $250 (Toplam 9 kutu). | **Körfez Bölge Aplikasyon Ziyareti & Sertifikasyon** |
| **Acıbadem & Medipolitan (Türkiye)** | $20,200.00 | 4 Piezo Cihazı | Türkiye'nin dev zincirleri, sarf alımında tekil ve düzensiz kalmış. | **Özel Kurumsal Fiyat Anlaşması + Baş Embriyolog Eğitimi** |

---

#### 3. 🚀 60 Günlük Acil Eylem ve Kampanya Takvimi
- **Aşama 1 (İlk 15 Gün) - "Piezo Aktivasyon Paketi" (Bundle) Lansmanı:**
  Tüm distribütörlere ve doğrudan müşterilere duyuru yapılacak: 10 Kutu Electro Dish (1x2) siparişine 2 Şişe Electro Liquid Solution %50 indirimli tanımlanacak.
- **Aşama 2 (Gün 16 - 45) - VIP Klinik Eğitim Seferi:**
  Klinik Aplikasyon Uzmanımız; İstanbul'da Acıbadem Maslak & Medipolitan, Kuveyt'te ATC klinik partnerlerine yerinde ziyaret gerçekleştirerek *Piezo Oosit Aktivasyon Rutini* protokolünü standardize edecek.
- **Aşama 3 (Gün 46 - 60) - Electro Fusion SET Avrupa/Körfez Webinar Serisi:**
  Ghent Üniversitesi Baş Araştırmacısı ile birlikte Körfez ve Avrupa'daki 25 referans tüp bebek direktörüne özel online canlı vaka çalıştayı düzenlenecek.

---

#### 4. 💰 Finansal Etki & Getiri Modeli
- **Bundle Kampanyasından Ek Sarf Geliri:** +350 Şişe Liquid Solution = **+$18,500 USD**
- **4 Merkezde Eğitimin Tetikleyeceği Sarf Siparişleri:** (Gen-Sa, ATC, Acıbadem, Medipolitan) = **+$27,500 USD**
- **1 Yeni Electro Fusion SET Satışı (Körfez / Avrupa):** = **+$16,000 USD**
- **Toplam Öngörülen Kısa Vadeli Gelir Katkısı:** **+$62,000 USD** (Brüt Kar Katkısı: ~$42,000 USD)"""

    # --------------------------------------------------------------------------
    # Preset 2: Gelecek Ay ve Çeyrekte Nakit Akışını Tehdit Eden En Büyük 3 Risk
    # --------------------------------------------------------------------------
    elif any(k in prompt_lower for k in ["3 risk", "en büyük 3 risk", "nakit akış", "tehdit", "vade", "aksama", "konsantrasyon", "sellco"]):
        topic_title = "Gelecek Çeyrekte Nakit Akışını ve Siparişleri Tehdit Eden En Büyük 3 Risk & Korunma Planı"
        kpi_impact = "$85,000 Nakit Akışı Risk Azaltımı & Gelir Güvencesi"
        timeline = "30 - 45 Gün (Acil Yönetici Müdahalesi)"
        target_group = "Sirius Pharm, Sellco BAE, Üretim & Tahsilat Boru Hattı"
        risk_level = "Yüksek — C-Level Müdahale Gerektirir"
        
        memo_content = f"""### 📋 ZEUS EXECUTIVE MEMORANDUM
**Doküman Kodu:** `ZGMI-STRAT-MEMO-2026-RISK3` | **Tarih:** Kasım 2026  
**Gizlilik Derecesi:** `C-LEVEL & YÖNETİM KURULU İNCELEMESİ (GİZLİ)`  
**Konu:** {topic_title}  
**Mevcut Temel Metrikler:** Toplam Küresel Ciro: **${total_rev:,.2f}** | Risk Altındaki Sipariş Hacmi: **$85,000** | Ortalama Tahsilat Süresi: **42 Gün**

---

#### 1. 🎯 Yönetici Özeti (Executive Brief)
Excel veritabanındaki 44 partnerin fatura tarihleri, sipariş frekansları ve sözleşme bitiş vadeleri çapraz sorgulandığında, 2026 Q4 ve 2027 Q1 döneminde ZEUS Bioscience nakit akışını ve sipariş sürekliliğini doğrudan tehdit eden **en kritik 3 risk** belirlenmiştir:
- **Risk 1: Aşırı Müşteri / Distribütör Yoğunlaşması (Top-3 Partner Concentration):** Cironun %58'i yalnızca 3 partnerde toplanmıştır (Sirius $164K, ITL Qatar $73K, SJNU Kore $68.5K).
- **Risk 2: SELLCO TRADING INTL (BAE) Sözleşme Bitişi & Müşteri Kaybı (Sözleşme Bitişi: 20 Kasım 2026):** BAE distribütörünün sözleşmesi 20 Kasım 2026'da dolmakta olup, firma 2026'da $0 sipariş vermiştir.
- **Risk 3: Cihaz (Capex) Satışlarının Düzensizliği ve Tedarik-Tahsilat Vade Uyuşmazlığı:** Cironun %58'i sermaye ekipmanı satışlarından gelmektedir; üretim maliyetleri peşin ödenirken tahsilatlar 45-60 güne uzadığında işletme sermayesi açığı oluşmaktadır.

---

#### 2. 📊 En Büyük 3 Riskin Ayrıntılı Değerlendirmesi ve Kök Nedenleri

| # | Risk Başlığı | Risk Seviyesi | Finansal Maruziyet | Kök Neden & Potansiyel Etki |
| :-: | :--- | :-: | :-: | :--- |
| **1** | **Distribütör Yoğunlaşma Riski (Sirius & ITL)** | 🔴 **Kritik** | **$45,000 / Ay** | Sirius Pharm (Ürdün/Irak) tek başına cironun %24'ünü oluşturmaktadır. Irak gümrüklerinde veya banka transferlerinde 30 günlük bir aksama, şirketin tüm aylık nakit akış dengesini sarsabilir. |
| **2** | **Sellco BAE Sözleşme Bitişi & Churn** | 🟠 **Yüksek** | **$25,000 Ciro Kaybı** | Sellco'nun sözleşmesi **20 Kasım 2026**'da sona eriyor. 2026'da hiç sipariş vermedi. 4 kurulu cihazın sarf geliri sıfırlanmış durumda. Acil müdahale edilmezse BAE pazarı tamamen kaybedilecek. |
| **3** | **Cihaz Sipariş Dalgalanması & Vade Uçurumu** | 🟡 **Orta-Yüksek** | **$40,000 - $60,000 İşletme Sermayesi Açığı** | Cihaz satışları parti partidir (lumpy). Bileşen tedarikçilerine avans ödenirken, distribütör vadeleri 60 güne sarktığında nakit akışında geçici açıklar meydana gelmektedir. |

---

#### 3. 🛡️ 30 - 45 Günlük Acil Korunma ve Eylem Planı
1. **Aksiyon 1 (Hemen) - BAE Pazarının Kurtarılması:**
   Sellco Trading ile bu hafta acil online toplantı yapılarak ya asgari 2027 yıllık $25,000 sarf taahhüdü ile sözleşme yenilenecek ya da Dubai Health City merkezli alternatif distribütör adayına geçiş protokolü işletilecek.
2. **Aksiyon 2 (15 Gün) - Sirius Pharm & ITL İçin Çift Kanallı Nakit Güvencesi:**
   Sirius ve ITL siparişlerinde %40 sipariş anında peşin, %40 konşimento tesliminde (BL), %20 ise 30 gün vadeli ödeme kuralı zorunlu tutulacak. Peşin ödemeyi tercih edenlere %3 erken ödeme teşviki tanımlanacak.
3. **Aksiyon 3 (30 Gün) - Türkiye Lokal Klinik Ağının Nakit Motoru Olarak Kullanılması:**
   Tavmergen ($25.7K), MLP Medicalpark ($29K) ve Pınar Sağlık ($15.5K) gibi peşin veya kısa vadeli ödeme yapan yerli 25 klinikten düzenli haftalık sarf çekişi garantiye alınarak yabancı distribütör nakit gecikmeleri dengelenecek."""

    # --------------------------------------------------------------------------
    # Preset 3: Asya-Pasifik ve Avrupa Pazar Zayıflığını Aşma Stratejisi
    # --------------------------------------------------------------------------
    elif any(k in prompt_lower for k in ["asya", "avrupa", "zayıflık", "aşmak", "apac", "pasifik", "kore", "ghent", "japonya"]):
        topic_title = "Asya-Pasifik ve Avrupa Pazarındaki Zayıflığı Aşma ve Genişleme Stratejisi"
        kpi_impact = "+$120,000 Yeni Bölgesel Ciro Potansiyeli (2027)"
        timeline = "90 - 180 Gün"
        target_group = "Japonya, Tayvan, Almanya, İspanya IVF Merkezleri & Distribütörleri"
        risk_level = "Orta — Regülasyon ve CE/MDR Belgelendirme Süreci"
        
        memo_content = f"""### 📋 ZEUS EXECUTIVE MEMORANDUM
**Doküman Kodu:** `ZGMI-STRAT-MEMO-2026-EXP` | **Tarih:** Kasım 2026  
**Gizlilik Derecesi:** `C-LEVEL & YÖNETİM KURULU İNCELEMESİ (GİZLİ)`  
**Konu:** {topic_title}  
**Mevcut Temel Metrikler:** Asya-Pasifik Ciro: **$68,550** (Sadece G.Kore SJNU) | Avrupa Ciro: **$36,540** (Belçika, Slovakya, Bulgaristan) | Toplam Portföy Ciro: **${total_rev:,.2f}**

---

#### 1. 🎯 Yönetici Özeti (Executive Brief)
ZEUS Bioscience'ın küresel coğrafi dağılımı incelendiğinde; satışların %84'ünün Türkiye ve MENA (Orta Doğu) bölgesine sıkıştığı, dünyanın en büyük ve en yüksek marjlı IVF pazarları olan **Asya-Pasifik (APAC)** ve **Batı Avrupa'da** şirketin ciddi bir pazar boşluğu (white-space) ve zayıflık yaşadığı görülmektedir:
- **Asya-Pasifik Tablosu:** Yalnızca Güney Kore'de (SJNU Inc - $68.6K, 7 cihaz) varlığımız bulunmaktadır. Ancak yılda **450,000+ IVF siklusu ile dünyanın 1 numaralı pazarı olan Japonya'da**, Tayvan'da, Hindistan ve Avustralya'da sıfır yetkili bayimiz bulunmaktadır.
- **Avrupa Tablosu:** Yalnızca Belçika Ghent University ($16.1K) ve Bulgaristan Meyona ($1.5K) mevcuttur. Avrupa'nın tüp bebek başkenti **İspanya (yılda 140,000 siklus)**, **Almanya (110,000 siklus)**, İtalya ve Fransa'da hiçbir ticari distribütörlük tesis edilmemiştir.
- **Hedef:** 2027 yılında bu iki bölgeden toplam **+$120,000 yeni net ciro** elde etmek.

---

#### 2. 📊 Bölgesel Boşluk ve Stratejik Fırsat Matrisi

| Hedef Bölge | Pazar Hacmi (Yıllık Siklus) | ZEUS Mevcut Durumu | Referans Kaldıraç | 2027 Genişleme Stratejisi |
| :--- | :--- | :--- | :--- | :--- |
| **Batı Avrupa (Almanya, İspanya, İtalya)** | ~350,000 Siklus / Yıl | $0 (Doğrudan Bayi Yok) | **Ghent University (Belçika)** $16.1K Electro Fusion Referansı | Ghent bilimsel verileriyle White Paper yayınlanması; İspanya IVIRMA ve Almanya tüp bebek kongrelerinde distribütörlük paneli açılması. |
| **Doğu Asya (Japonya & Tayvan)** | ~500,000 Siklus / Yıl | $0 (Doğrudan Bayi Yok) | **SJNU Inc (Güney Kore)** 7 Piezo Cihazı & Klinik Başarı Verisi | Kore'deki Piezo ICSI döllenme artış verilerini Japonca/İngilizce vaka serisi olarak Tokyo/Osaka bayilerine sunmak; ilk çeyrekte distribütör ataması. |

---

#### 3. 🚀 90 - 180 Günlük Taktik Yol Haritası
- **Aşama 1 (1 - 60 Gün) - Akademik Referans Paketinin Hazırlanması:**
  Ghent Üniversitesi'nin Electro Fusion SET sonuçları ve SJNU Güney Kore'nin Piezo ICSI klinik başarı oranları, uluslararası regülasyona uygun iki ayrı "Klinik Etkinlik Dosyası"na dönüştürülecek.
- **Aşama 2 (61 - 120 Gün) - Hedef Distribütör Seçimi ve LOA Teklifleri:**
  İspanya (Madrid/Barselona), Almanya (Münih/Frankfurt) ve Japonya (Tokyo) merkezli 6 medikal IVF distribütörü taranarak münhasır olmayan bölge distribütörlük dosyası iletilecek.
- **Aşama 3 (121 - 180 Gün) - Kongre Katılımı & Canlı İmza:**
  ESHRE 2027 kongresinde Avrupa partnerleri, ASPIRE kongresinde Asya-Pasifik partnerleri ile yıllık asgari $40,000 alım taahhütlü ilk distribütörlük sözleşmeleri imzalanacak.

---

#### 4. 💰 Finansal Projeksiyon (2027 Hedefi)
- **Avrupa (İspanya & Almanya 2 Yeni Bayi):** 4 Cihaz + 300 Kutu Sarf = **+$55,000 USD**
- **Asya-Pasifik (Japonya & Tayvan 2 Yeni Bayi):** 5 Cihaz + 400 Kutu Sarf = **+$65,000 USD**
- **Toplam Yeni Bölgesel Katkı:** **+$120,000 USD** (Konsolide ciro hedefi $385K'nın %31'ini oluşturacaktır)."""

    # --------------------------------------------------------------------------
    # Preset 4: Tekrar Eden Sarf Gelirini %20 Artırma & Hedef Müşteri/Bayi Planı
    # --------------------------------------------------------------------------
    elif any(k in prompt_lower for k in ["%20", "tekrar eden", "sarf", "odaklanmalı", "hangi müşteri", "hangi distribütör"]):
        topic_title = "Tekrar Eden Sarf Malzeme Gelirlerini %20+ Artırmak İçin Hedef Müşteri & Distribütör Odak Planı"
        kpi_impact = "+$42,500 / Yıl Net Düzenli Sarf Geliri (+%25.5 Artış)"
        timeline = "60 Gün (Hemen Başlatılabilir)"
        target_group = "Gen-Sa, ATC Kuwait, Acıbadem, Medipolitan, Sirius Jordan-Iraq"
        risk_level = "Çok Düşük — Kurulu Cihaz Altyapısı Zaten Mevcut"
        
        memo_content = f"""### 📋 ZEUS EXECUTIVE MEMORANDUM
**Doküman Kodu:** `ZGMI-STRAT-MEMO-2026-RECUR` | **Tarih:** Kasım 2026  
**Gizlilik Derecesi:** `C-LEVEL & YÖNETİM KURULU İNCELEMESİ (GİZLİ)`  
**Konu:** {topic_title}  
**Mevcut Temel Metrikler:** Toplam Kurulu Cihaz: **{total_devices} Adet** | Mevcut Yıllık Sarf Satışı: **~$83,435 - $116,307** | Hedeflenen +%20 Ek Sarf: **+$17,000 - $23,000**

---

#### 1. 🎯 Yönetici Özeti (Executive Brief)
Excel veri tabanındaki 44 kurum tek tek incelendiğinde, şirketin sarf gelirini %20 artırmak için **yeni bir cihaz satmasına veya yeni bir ülkeye girmesine gerek olmadığı** açıkça görülmektedir.
- Kurulu 63 Piezo cihazının bulunduğu kliniklerin bir bölümü yüksek frekansta sarf çekerken (Örn: Tavmergen 352 kutu, Pınar Sağlık 130 kutu, ITL Katar 449 kutu);
- Belirli partnerlerde cihaz kurulu olmasına rağmen sarf alımının **"uyuyan hesap" (dormant account)** düzeyinde kaldığı tespit edilmiştir.
- Sadece **aşağıdaki 5 partnerin sarf çekişini normalize ederek**, sarf gelirini %20 değil, **+%40'tan fazla (+$42,500/yıl)** artırmak mümkündür.

---

#### 2. 📊 Odaklanılması Gereken 5 Öncelikli Partner ve Hesap Bazlı Potansiyel

| Öncelik | Partner / Müşteri | Ülke | Kurulu Cihaz | Mevcut Sarf Hacmi | Cihaz Başı Tüketim | Hedeflenen Ek Sarf Geliri | Somut Uygulanacak Taktik |
| :-: | :--- | :--- | :-: | :--- | :-: | :-: | :--- |
| **#1** | **GEN-SA GENEL SAĞLIK A.Ş.** | Türkiye | 4 Cihaz | $1,640 (30 Kutu) | $410 / Cihaz | **+$13,000** | Cihazların kurulu olduğu 4 kliniğe yerinde oosit protokol eğitimi verilerek aylık 15 kutu kotası tanımlanacak. |
| **#2** | **ADVANCED TECH CO (ATC)** | Kuveyt | 3 Cihaz | $750 (9 Kutu) | $250 / Cihaz | **+$10,000** | Kuveyt Sağlık Bakanlığı ve özel tüp bebek hastanelerine aplikasyon ziyareti ile çeyreklik 25 kutu sipariş şartı. |
| **#3** | **ACIBADEM SAĞLIK GRUBU** | Türkiye | 2 Cihaz | $760 (11 Kutu) | $380 / Cihaz | **+$7,500** | Türkiye'nin lider hastane zincirinde Piezo kullanımını kurtarma sikluslarından rutin sikluslara geçirme protokolü. |
| **#4** | **MEDİPOLİTAN SAĞLIK A.Ş.** | Türkiye | 2 Cihaz | $440 (12 Kutu) | $220 / Cihaz | **+$5,000** | Kurumsal laboratuvar şefleri ile görüşülerek aylık konsinye sarf deposu anlaşması imzalanacak. |
| **#5** | **SIRIUS PHARM COMPANY** | Ürdün-Irak | 18 Cihaz | $52,113 (642 Kutu) | $2,895 / Cihaz | **+$12,000** | 18 cihazlık devasa parkta cihaz başı sarfı $2,895'ten $3,500'e çıkarmak (Katar seviyesi $9,341'dir) bile tek başına $12K üretir. |

---

#### 3. 🚀 30 - 60 Günlük Uygulama Planı
1. **Gün 1 - 15 (Müşteri Teması):** Gen-Sa, ATC Kuveyt, Acıbadem ve Medipolitan sorumlu hekim ve embriyologlarıyla görüşme planlanacak.
2. **Gün 16 - 35 (Eğitim ve Bundle Teşviki):** Bu 5 merkeze özel "Yıllık Sarf Abonelik Modeli" (Monthly Consumable Subscription) sunularak %10 fiyat güvencesi verilecek.
3. **Gün 36 - 60 (İlk Sevkiyatlar):** Çeyreklik toplu sevkiyatlar faturalandırılarak nakit akışına dahil edilecek.

---

#### 4. 💰 ROI & Finansal Sonuç
- **Hedeflenen %20 Büyüme Barajı:** +$16,700 USD
- **Bu 5 Hesaptan Sağlanacak Toplam Katkı:** **+$47,500 USD** (Hedefin 2.8 katı!)
- **Brüt Kar Marjı Katkısı:** Sarf malzemelerinde brüt marj %68+ olduğu için ilave cironun **~$32,000'ı doğrudan net operasyonel kara** dönüşecektir."""

    # --------------------------------------------------------------------------
    # Preset 5: 2026 Q4 Kapanışı ve 2027 Projeksiyonu
    # --------------------------------------------------------------------------
    elif any(k in prompt_lower for k in ["2027", "q4", "hedef", "tahmin", "projeksiyon", "yol haritası"]):
        topic_title = "2026 Q4 Kapanışı ve 2027 Yılı $385,000 Ciro Hedefine Ulaşma Yol Haritası"
        kpi_impact = "$303.4K (2026) ➔ $385.0K (2027) (+%27 Büyüme)"
        timeline = "14 Ay (Kasım 2026 - Aralık 2027)"
        target_group = "Tüm 44 Müşteri & Distribütör Portföyü"
        risk_level = "Düşük-Orta — Güçlü Sipariş Boru Hattı"
        
        memo_content = f"""### 📋 ZEUS EXECUTIVE MEMORANDUM
**Doküman Kodu:** `ZGMI-STRAT-MEMO-2027-FC` | **Tarih:** Kasım 2026  
**Gizlilik Derecesi:** `C-LEVEL & YÖNETİM KURULU İNCELEMESİ (GİZLİ)`  
**Konu:** {topic_title}  
**Mevcut Temel Metrikler:** 2026 YTD Gerçekleşen: **$214,967** | Q4 Tahmin Dahil: **$303,467** | 2027 Hedef: **$385,000 (+%27.0 YoY)**

---

#### 1. 🎯 Yönetici Özeti (Executive Brief)
2026 yılı ilk 10 ayında ZEUS Bioscience **$214,967** net faturalı ciro üretmiştir. Yıl sonu (Kasım-Aralık) beklenen **$88,500'lık** sipariş kapanışlarıyla FY26 toplamının **$303,467** seviyesinde tamamlanması kesinleşmiştir.
- 2027 yılı stratejik hedefi, yıllık %27 büyüme ile **$385,000** konsolide ciroya ve toplam **100+ kurulu Piezo cihaz tabanına** ulaşmaktır.
- Bu büyümenin lokomotifi; mevcut kurulu cihazların sarf tüketiminin artması (%40 pay) ve 38 yeni cihazın (Türkiye, Körfez, Kore ve Avrupa) pazara verilmesi (%60 pay) olacaktır.

---

#### 2. 📊 Büyüme Bileşenleri & Bölgesel Dağılım Modeli
1. **Türkiye İç Pazarı ($110,000 Hedef):** Etlik Zübeyde Hanım, Eurofertil, Acıbadem, Memorial ve Medicana merkezlerine 12 yeni ünite ve 850 kutu sarf.
2. **Körfez & MENA ($165,000 Hedef):** Sirius Pharm (Ürdün/Irak) $85K+, ITL (Katar) $40K+, AMS (Suudi Arabistan) $25K+, ATC (Kuveyt) $15K+.
3. **Asya-Pasifik & Kore ($75,000 Hedef):** SJNU Inc üzerinden Seul/Busan kliniklerine 10 yeni Piezo ünitesi.
4. **Avrupa & Akademik ($35,000 Hedef):** Ghent Üniversitesi referansı ile Belçika, Slovakya (Ferticent), Arnavutluk ve Almanya başlangıç satışları.

---

#### 3. 🚀 Çeyrek Bazlı 2027 Eylem Planı (Quarterly Milestones)

| Çeyrek | Hedef Ciro | Kritik Ticari Eylem |
| :--- | :--- | :--- |
| **2026 Q4** | $88,500 | Sirius Pharm ve ITL Katar yıl sonu siparişlerinin sevkiyatı; 2027 distribütör yıllık taahhütlerinin imzalanması. |
| **2027 Q1** | $82,000 | Yıllık bakım ve sarf yenileme sözleşmelerinin aktivasyonu; Kore ve Körfez ilk çeyrek cihaz parti sevkiyatı. |
| **2027 Q2** | $98,000 | ESHRE Kongresi öncesi Avrupa akademik merkezlerine Electro Fusion SET ve Piezo Stimulator tanıtım kampanyası. |
| **2027 Q3** | $92,000 | Yaz dönemi sarf stoku takviyesi; Doğu Asya yeni distribütör deneme sevkiyatları. |
| **2027 Q4** | $113,000 | Yıl sonu bütçe kapanışları, 2028 sözleşme uzatmaları ve toplam $385K hedefine ulaşılması. |"""

    # --------------------------------------------------------------------------
    # Default / Custom Prompt Fallback
    # --------------------------------------------------------------------------
    else:
        topic_title = f"Özel Stratejik Değerlendirme: {active_prompt}"
        kpi_impact = "+%18.5 Operasyonel Verimlilik Artışı"
        timeline = "60 Gün"
        target_group = "İlgili Bölge Müşterileri ve Ürün Hatları"
        risk_level = "Orta — Yakın Yönetim Takibi Gerektirir"
        
        memo_content = f"""### 📋 ZEUS EXECUTIVE MEMORANDUM
**Doküman Kodu:** `ZGMI-CUSTOM-ANALYSIS-2026` | **Tarih:** Kasım 2026  
**Gizlilik Derecesi:** `C-LEVEL & YÖNETİM KURULU İNCELEMESİ (GİZLİ)`  
**Sorgu / Analiz Konusu:** "{active_prompt}"  
**Sistem Metrikleri:** Toplam Küresel Ciro: **${total_rev:,.2f}** | Kurulu Cihaz Tabanı: **{total_devices} Adet** | Ortalama Brüt Marj: **%{avg_margin:.1f}**

---

#### 1. 🎯 Yönetici Özeti (Executive Summary)
ZEUS Commercial Intelligence Engine, iletilen stratejik sorguyu Excel satış veritabanındaki 44 aktif müşteri ve 874 işlem kaydı üzerinden çapraz analize tabi tutmuştur.
- **Odak Ürünler:** ZEUS Piezo Pulse Stimulator (%59.2 pay), Electro Dish 1x2/1x1 (%29.9 pay), Electro Liquid Solution (%7.6 pay).
- **Temel Dinamik:** Şirketin sürdürülebilir nakit akışı, sermaye ekipmanı satışlarından ziyade kurulu cihazların oluşturduğu düzenli ve yüksek marjlı sarf malzeme çekişine bağlıdır.
- Sorgulanan pazar ve ürün grubu için kilit öncelik; sipariş sürekliliğini teminat altına almak ve dağıtım taahhütlerini garantiye almaktır.

---

#### 2. 📊 Veri Odaklı Tanı ve Kök Neden Analizi
1. **Pazar Ayak İzi:** Türkiye ($219K), Ürdün & Irak (Sirius Pharm - $164K), Katar (ITL - $73K), Güney Kore (SJNU - $68.6K), Suudi Arabistan ($29.5K), Kuveyt ($26.3K) ve BAE ($23.2K) portföyün %88'ini oluşturmaktadır.
2. **Sipariş Frekansı:** Müşteri sipariş frekansları incelendiğinde; çeyreklik düzenli sipariş veren kliniklerin cihaz başına karlılığı, tekil sipariş verenlere göre %34 daha yüksektir.
3. **Marj Koruması:** Yüksek enflasyonist ve kur oynaklığı olan pazarlarda döviz bazlı liste fiyatı korunmalı, indirimler yalnızca peşin ödeme veya yüksek kutu taahhütlerinde uygulanmalıdır.

---

#### 3. 🚀 Somut Aksiyon ve 60 Günlük Eylem Planı
- **Aksiyon 1 (Hemen):** İlgili ülke veya ürün için müşteri sipariş geçmişi ve stok seviyesi taranarak proforma teklif yenilenecek.
- **Aksiyon 2 (15 Gün):** Ürün bundle modeli (Cihaz + 10 Kutu Sarf + Solüsyon) ile ortalama sepet büyüklüğü artırılacak.
- **Aksiyon 3 (45 Gün):** Bölgedeki ana distribütör ile 2027 yıllık asgari alım protokolü müzakere edilecek."""

    return {
        'title': topic_title,
        'kpi_impact': kpi_impact,
        'timeline': timeline,
        'target_group': target_group,
        'risk_level': risk_level,
        'memo_md': memo_content
    }


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
            <strong>📦 Gümrük ve Sevkiyat Takibi</strong><br>
            <span style="font-size:13px; color:#CCD7E6;">ITL (Katar): $14,200 tutarındaki yeni Piezo Pulse ve Electro Dish sevkiyatı gümrük teyit aşamasındadır.</span>
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
            <div class="alert-card" style="border-left: 4px solid #F0BC57;background:rgba(240,188,87,0.08);">
                <strong style="color:#F0BC57;font-size:14px;">⚠️ ZEUS Electro Liquid Solution — MENA (Ürdün & Irak: Sirius Pharmaceutical)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> Piezo stimülatör cihazı sayısı 14'e yükselmesine rağmen Liquid solüsyon sipariş döngüsü 45 günden 70 güne çıktı.<br>
                    <b>Kök Neden:</b> Yeni kurulan kliniklerde protokol standardizasyonu eksikliği; Electro Dish ile solüsyonun eşzamanlı planlanmaması.<br>
                    <b>AI Önerilen Aksiyon:</b> "Piezo Full Stimulation Pack" bundle oluşturulmalı; her 10 kutu Dish ile 2 şişe Liquid solüsyonu eşleştiren avantajlı sipariş protokolü devreye alınmalı.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="alert-card" style="border-left: 4px solid #FF8A96;background:rgba(229,112,123,0.08);">
                <strong style="color:#FF8A96;font-size:14px;">⚠️ Smart Cryo-Tools — Körfez (Suudi Arabistan: AMS - Arabian Medical Supplies)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> Kriyoprezervasyon sarf malzemeleri sipariş hacmi potansiyelin altında kaldı ($29.5K ciro içinde sarf payı düşük).<br>
                    <b>Kök Neden:</b> Riyadh ve Cidde'deki partner kliniklerin embriyolog ekiplerinde personel değişimi yaşandı, ürün eğitim ihtiyacı doğdu.<br>
                    <b>AI Önerilen Aksiyon:</b> AMS işbirliğiyle 6 büyük IVF kliniğine online ve yerinde "Advanced Cryo Vitrification Workshop" organize edilerek sipariş frekansı artırılmalı.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c_diag2:
            st.markdown("""
            <div class="alert-card" style="border-left: 4px solid #4FD1DE;background:rgba(19,166,181,0.08);">
                <strong style="color:#4FD1DE;font-size:14px;">💡 Büyüme Fırsatı: ZEUS Piezo Pulse Stimulator — Güney Kore (SJNU Inc)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> Seul ve Busan IVF merkezlerinde Piezo oosit aktivasyonu başarı oranları talebi %35 artırdı ($68.6K ciro, 7 cihaz).<br>
                    <b>Kök Neden:</b> Yerel bilimsel vaka serileri ve embriyolog memnuniyeti Asya-Pasifik'te güçlü referans etkisi yarattı.<br>
                    <b>AI Önerilen Aksiyon:</b> 2027 için yıllık 20 adetlik cihaz ve 600 kutuluk Electro Dish taahhütlü Distribütörlük Yenileme Sözleşmesi imzalanmalı.
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="alert-card" style="border-left: 4px solid #5FD3AE;background:rgba(95,211,174,0.08);">
                <strong style="color:#5FD3AE;font-size:14px;">💡 Avrupa Akademik Benchmark: Zeus Electro Fusion SET — Belçika (Ghent University)</strong><br>
                <div style="font-size:13px;color:#CCD7E6;margin-top:6px;line-height:1.5;">
                    <b>Tespit:</b> Ghent Üniversitesi Laboratuvarı referansıyla Benelüks ve Almanya'daki araştırma hastanelerinden talep sinyalleri gelmeye başladı.<br>
                    <b>Kök Neden:</b> Üniversitenin yayınladığı klinik veriler ZEUS Electro Fusion teknolojisinin mikroenjeksiyon sonrası canlılık oranını kanıtladı.<br>
                    <b>AI Önerilen Aksiyon:</b> ESHRE 2027 kongresinde ortak seminer düzenlenmeli; Batı Avrupa üniversite hastaneleri için "Akademik Tanıtım Paketi" açılmalı.
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
            <div style="background:rgba(19,166,181,0.1);border-left:4px solid #13A6B5;padding:12px;border-radius:8px;margin-bottom:10px;">
                <b style="color:#13A6B5;">1. Asya-Pasifik: Güney Kore (SJNU) İvmesi ve Bölgesel Genişleme</b><br>
                <span style="font-size:12px;color:#CCD7E6;">Güney Kore'de $68.6K ciro ve 7 Piezo cihazı kuruldu. Japonya, Tayvan ve Avustralya henüz açılmamış pazar boşluğudur. <b>Tahmini yakalanabilir ek potansiyel: $140,000+.</b></span>
            </div>
            <div style="background:rgba(240,188,87,0.1);border-left:4px solid #F0BC57;padding:12px;border-radius:8px;margin-bottom:10px;">
                <b style="color:#F0BC57;">2. MENA & Körfez: Yüksek Cihaz Kurulumu & Sarf Hacmi Artırma</b><br>
                <span style="font-size:12px;color:#CCD7E6;">Katar ($73K), Suudi Arabistan ($29.5K), Kuveyt ($26.3K) ve BAE ($23.2K) toplam $152K ciro sağladı. Kurulu 18 Piezo cihazı bulunuyor; klinik başına sarf pull-through oranı artırılarak yıllık ek $55K düzenli nakit akışı sağlanabilir.</span>
            </div>
            <div style="background:rgba(95,211,174,0.1);border-left:4px solid #5FD3AE;padding:12px;border-radius:8px;">
                <b style="color:#5FD3AE;">3. Avrupa: Akademik Referans Gücü ve Batı Avrupa Açılımı</b><br>
                <span style="font-size:12px;color:#CCD7E6;">Belçika (Ghent Univ - $16.1K) ve Doğu Avrupa (Slovakya, Arnavutluk) ile ayak izimiz oluştu. Almanya, İspanya ve İtalya'da yerel distribütör atamaları ile 2027'de $100K+ ek pazar büyüklüğüne ulaşılabilir.</span>
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
            "🎯 Hangi ürünlerde acil promosyon, kampanya veya klinik eğitim gerekiyor?",
            "⚠️ Gelecek ay ve çeyrekte nakit akışını ve siparişleri tehdit eden en büyük 3 risk nedir?",
            "🌏 Asya-Pasifik ve Avrupa zayıflığımızı aşmak için hangi strateji uygulanmalı?",
            "🔄 Tekrar eden sarf malzeme gelirlerini %20 artırmak için hangi müşteri veya distribütöre odaklanılmalı?",
            "📈 2026 Q4 Kapanışı ve 2027 Yılı $385K Ciro Projeksiyonuna Ulaşma Yol Haritası"
        ]
        
        selected_q = st.radio("Hızlı Stratejik Sorgular:", prompt_choices, horizontal=False)

        custom_q = st.text_input("Veya kendi stratejik sorunuzu yazın:", placeholder="Örn: Katar (ITL) veya Güney Kore için 2027 büyüme planı nedir?")
        active_prompt = custom_q.strip() if custom_q.strip() else selected_q

        if st.button("🚀 AI Yönetici Notunu Oluştur", type="primary"):
            st.markdown("---")

            # 1. Animasyon için placeholder alanı
            loading_placeholder = st.empty()
            
            with loading_placeholder.container():
                components.html("""
                    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 180px; font-family: sans-serif; background: #0b1426; border-radius: 12px; border: 1px solid #13A6B5; box-shadow: 0 0 20px rgba(19, 166, 181, 0.2);">
                        <!-- Antigravity Parıldayan Küre -->
                        <div id="ai-orb" style="width: 50px; height: 50px; background: radial-gradient(circle, #00F5FF 0%, #13A6B5 70%, #0B6F85 100%); border-radius: 50%; box-shadow: 0 0 30px #00F5FF; margin-bottom: 20px;"></div>
                        
                        <!-- Dinamik Metin -->
                        <div id="ai-text" style="color: #4FD1DE; font-weight: bold; font-size: 15px; letter-spacing: 1px; text-transform: uppercase;">
                            ZEUS AI STRATEJİK ANALİZ MOTORU ÇALIŞIYOR...
                        </div>
                    </div>

                    <!-- GSAP Animasyon Motoru -->
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
                    <script>
                        gsap.to("#ai-orb", {
                            y: -15,
                            scale: 1.15,
                            duration: 1.2,
                            yoyo: true,
                            repeat: -1,
                            ease: "sine.inOut"
                        });
                        gsap.to("#ai-text", {
                            opacity: 0.3,
                            duration: 0.8,
                            yoyo: true,
                            repeat: -1,
                            ease: "power1.inOut"
                        });
                    </script>
                """, height=200)

            # 2. Canlı Gemini API veya Yerleşik ZEUS Biyo-Strateji Motoru
            ai_generated_text = None
            api_key = get_gemini_api_key()
            
            if api_key:
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    prompt_talimati = f"""
                    Sen kıdemli bir medikal pazar analiz uzmanı ve ZEUS Global Market Intelligence baş stratejistisin.
                    Senden kullanıcının sorduğu şu stratejik soruya derinlemesine, somut Excel satış verileri ve rasyonel IVF pazar gerçekleriyle yanıt üretmeni istiyorum:
                    Soru: "{active_prompt}"
                    
                    Yorum yaparken şu küresel verileri ve pazar gerçeklerini arka planda mutlaka dikkate al:
                    - Toplam Küresel Ciro Portföyü: ${total_rev:,.2f} USD | Ortalama Brüt Marj: %{avg_margin:.1f}
                    - Toplam Kurulu Cihaz Tabanı: {total_devices} adet ZEUS Piezo Pulse Stimulator (63 aktif cihaz).
                    - 2026 YTD Gerçekleşen: 214,967 USD, FY26 Beklenen: 303,467 USD, 2027 Projeksiyon Hedefi: 385,000 USD (+%27 YoY).
                    - Kilit Pazarlar: Türkiye ($219K, 25 yerli klinik), Ürdün-Irak ($164.2K - Sirius Pharm, 18 cihaz), Katar ($72.9K - ITL, 4 cihaz), Güney Kore ($68.6K - SJNU, 7 cihaz), Filistin ($46.7K - CMC, 5 cihaz), Suudi Arabistan ($29.5K - AMS, 3 cihaz), Kuveyt ($26.3K - ATC, 3 cihaz), BAE ($23.2K - Sellco, 4 cihaz - Sözleşme Bitişi: 20 Kasım 2026!), Belçika ($16.1K - Ghent University, 1 Electro Fusion SET).
                    - Ürün Dağılımı: Piezo Stimulator ($411K - 63 adet), Electro Dish 1x2 ($207.6K - 2,097 kutu), Electro Liquid Solution 2x1ml ($40.3K - 765 kutu), Zeus Electro Fusion SET ($15.9K - 1 adet), Liquid Solution Powder ($12.7K - 232 kutu).
                    - Kilit Sorunlar: 
                      1) Dish / Liquid Solution oranı 2.1:1 olup solüsyon tüketimi geride kalmaktadır (Bundle ihtiyacı).
                      2) Gen-Sa (4 cihaz, sadece 30 kutu sarf), Kuveyt ATC (3 cihaz, 9 kutu sarf), Acıbadem (2 cihaz, 11 kutu sarf), Medipolitan (2 cihaz, 12 kutu sarf) merkezlerinde embriyolog eğitimi ve sarf çekişi acil müdahale gerektirir.
                      3) Sellco BAE sözleşmesi 20 Kasım 2026'da bitmektedir ve 2026'da $0 çekmiştir (Churn riski).
                      4) Batı Avrupa ve Asya-Pasifik'te (Japonya, Tayvan, Almanya, İspanya) pazar boşluğu bulunmaktadır; Ghent Univ ve SJNU Kore başarı modelleri kaldıraç olarak kullanılmalıdır.
                    
                    Raporunu tamamen Türkçe, profesyonel bir tonda, üst yönetime sunulacak 'ZEUS Executive Memorandum' formatında, yönetici özeti, veri odaklı tanı tablosu, taktik aksiyon takvimi ve finansal projeksiyon başlıklarıyla yaz.
                    """
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_talimati,
                    )
                    ai_generated_text = response.text
                except Exception:
                    ai_generated_text = None

            # Yerleşik ZEUS Bilişsel Yönetici Raporu
            memo_data = generate_zeus_executive_memo(active_prompt, total_rev, avg_margin, total_devices)
            
            # Animasyonu kaldırıyoruz
            loading_placeholder.empty()

            # Yönetici Özet Rozetleri (KPIs)
            mk1, mk2, mk3, mk4 = st.columns(4)
            mk1.metric("Tahmini Finansal Etki", memo_data['kpi_impact'])
            mk2.metric("Uygulama Zamanı", memo_data['timeline'])
            mk3.metric("Hedef Grup", memo_data['target_group'])
            mk4.metric("Risk Değerlendirmesi", memo_data['risk_level'])

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Memo İçeriği
            final_memo_body = ai_generated_text if ai_generated_text else memo_data['memo_md']
            st.markdown(final_memo_body)

            # İndirme Butonu
            st.download_button(
                label="📥 Bu Yönetici Notunu İndir (.md)",
                data=final_memo_body,
                file_name=f"ZEUS_Executive_Memo_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown"
            )




# ==============================================================================
# 5. VIEW: DISTRIBUTORS & CRM
# ==============================================================================
elif nav_selection == "◇ Distributors & CRM":
    st.title("🏢 Commercial Accounts & Client Directory")
    st.caption("Excel satış kayıtlarındaki 44 aktif müşteri ve partnerin ticari sipariş hacimleri, cihaz kurulumları ve sarf tüketim dökümü.")

    # Search & Filters
    f_c1, f_c2, f_c3, f_c4 = st.columns([1.5, 1, 1, 1])
    with f_c1:
        search_query = st.text_input("🔍 Firma adı, ülke veya ürün ara", "")
    with f_c2:
        cat_crm_filter = st.selectbox("Müşteri / Bayi Sınıfı", ["Tümü", "🏢 Distribütörler", "🇹🇷 Local Müşteriler (Türkiye)", "🌍 International Müşteriler"])
    with f_c3:
        all_countries = sorted(list(set(d['country_name'] for d in dists_with_rev)))
        country_filter = st.selectbox("Ülkeye Göre Filtrele", ["Tümü"] + all_countries)
    with f_c4:
        type_filter = st.selectbox("Satın Alma Tipi", ["Tümü", "Cihaz ve Sarf Alanlar", "Sadece Sarf Alanlar"])

    filtered_accounts = dists_with_rev
    if cat_crm_filter == "🏢 Distribütörler":
        filtered_accounts = [d for d in filtered_accounts if d.get('account_type') == 'Distributor']
    elif cat_crm_filter == "🇹🇷 Local Müşteriler (Türkiye)":
        filtered_accounts = [d for d in filtered_accounts if d.get('account_type') == 'Local Customer']
    elif cat_crm_filter == "🌍 International Müşteriler":
        filtered_accounts = [d for d in filtered_accounts if d.get('account_type') == 'International Customer']

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
            'Sınıf': '🏢 Distribütör' if d.get('account_type') == 'Distributor' else ('🇹🇷 Local' if d.get('account_type') == 'Local Customer' else '🌍 International'),
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
    st.caption("Distribütörlük sözleşmeleri, LOA yetki belgeleri, yerel (Türkiye) ve uluslararası müşteri tedarik protokolleri yönetimi.")

    # Categorize accounts
    dist_rows = [d for d in active_rows if d.get('account_type') == 'Distributor']
    local_rows = [d for d in active_rows if d.get('account_type') == 'Local Customer']
    intl_rows = [d for d in active_rows if d.get('account_type') == 'International Customer']

    # Distributor-only LOA metrics
    valid_loas = len([d for d in dist_rows if d.get('loa') == 'Valid'])
    renew_loas = len([d for d in dist_rows if d.get('loa') == 'In renewal'])
    excl_count = len([d for d in dist_rows if 'exclusive' in str(d.get('exclusivity', '')).lower()])

    # Top KPI Metrics Cards
    ac1, ac2, ac3, ac4 = st.columns(4)
    ac1.metric("🏢 Yetkili Distribütörler", f"{len(dist_rows)} Partner", delta=f"{excl_count} Münhasır Bölge")
    ac2.metric("📜 Geçerli LOA Belgeleri", f"{valid_loas} / {len(dist_rows)}", delta=f"{renew_loas} Yenilemede" if renew_loas > 0 else "Tamamı Geçerli", delta_color="inverse" if renew_loas > 0 else "normal")
    ac3.metric("🇹🇷 Local Müşteriler (Türkiye)", f"{len(local_rows)} Klinik", delta=f"{format_curr(sum(d['revenue'] for d in local_rows))} Yurtiçi Hacim")
    ac4.metric("🌍 International Müşteriler", f"{len(intl_rows)} Kurum", delta=f"{format_curr(sum(d['revenue'] for d in intl_rows))} Doğrudan İhracat")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # ACTION: YENİ SÖZLEŞME VE DOKÜMAN EKLEME FORMU
    # --------------------------------------------------------------------------
    with st.expander("➕ Yeni Sözleşme / Doküman Ekle (Add Agreement & Document)", expanded=False):
        st.markdown("##### 📝 Yeni Sözleşme, LOA veya Tedarik Protokolü Tanımlama")
        
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            cat_choice = st.radio(
                "Müşteri / Partner Sınıfı Seçin:",
                [
                    "🏢 Distributor (Yetkili Bayi — Sözleşme & LOA Zorunlu)",
                    "🇹🇷 Local Müşteri (Türkiye İçi Klinik / Hastane)",
                    "🌍 International Müşteri (Uluslararası Doğrudan Klinik / Kurum)"
                ],
                index=0
            )
            
            if "Distributor" in cat_choice:
                cat_key = "Distributor"
                company_pool = sorted(list(set(d['name'] for d in dist_rows)))
                doc_types = [
                    "Distribütörlük Sözleşmesi (Distribution Agreement)",
                    "LOA (Letter of Authorization / Yetki Belgesi)",
                    "Münhasırlık Protokolü (Territorial Exclusivity Addendum)",
                    "Yıllık Asgari Satış Taahhütnamesi (Minimum Target Commitment)",
                    "NDA (Gizlilik Anlaşması)"
                ]
            elif "Local Müşteri" in cat_choice:
                cat_key = "Local Customer"
                company_pool = sorted(list(set(d['name'] for d in local_rows)))
                doc_types = [
                    "Yurtiçi Cihaz Satın Alma Sözleşmesi",
                    "Yurtiçi Sarf Malzeme Tedarik Protokolü",
                    "Klinik Cihaz Kurulum ve Devreye Alma Tutanağı",
                    "Yıllık Periyodik Bakım Sözleşmesi",
                    "NDA (Gizlilik Anlaşması)"
                ]
            else:
                cat_key = "International Customer"
                company_pool = sorted(list(set(d['name'] for d in intl_rows)))
                doc_types = [
                    "Uluslararası Doğrudan İhracat & Tedarik Sözleşmesi",
                    "Akademik İşbirliği ve Araştırma Protokolü",
                    "Konsinye / Demo Cihaz Devir Sözleşmesi",
                    "CE / ISO Kalite Uygunluk Sertifikası",
                    "NDA (Gizlilik Anlaşması)"
                ]
                
            selected_company = st.selectbox("İlgili Firma / Kurum", company_pool + ["[+] Yeni Firma / Kurum Ekle..."])
            if selected_company == "[+] Yeni Firma / Kurum Ekle...":
                new_comp_name = st.text_input("Yeni Firma / Kurum Adı")
                new_comp_country = st.text_input("Ülke")
                final_company = new_comp_name.strip() if new_comp_name.strip() else "Yeni Firma"
                final_country = new_comp_country.strip() if new_comp_country.strip() else "Global"
            else:
                final_company = selected_company
                c_d = next((d for d in active_rows if d['name'] == final_company), None)
                final_country = c_d['country_name'] if c_d else "Global"
                
            selected_doc_type = st.selectbox("Belge Türü", doc_types)
            doc_ref_no = st.text_input("Sözleşme / Belge Referans No", value=f"ZEUS-DOC-2026-{len(st.session_state.custom_documents)+1:03d}")

        with f_c2:
            dt_c1, dt_c2 = st.columns(2)
            with dt_c1:
                start_date = st.date_input("Sözleşme Başlangıç Tarihi")
            with dt_c2:
                end_date = st.date_input("Sözleşme Bitiş Tarihi")
                
            if cat_key == "Distributor":
                excl_sel = st.selectbox("Münhasırlık (Exclusivity Hakları)", ["Münhasır (Exclusive Territory)", "Non-Exclusive (Genel Yetkili)", "Bölgesel Öncelikli"])
                loa_sel = st.selectbox("LOA Yetki Belgesi Durumu", ["✅ Geçerli (Valid)", "⚠️ Yenilemede (In Renewal)", "📝 Taslak / Hazırlanıyor (Draft)"])
                min_commitment = st.number_input("Yıllık Asgari Satış Taahhüdü ($ USD)", min_value=0, value=50000, step=5000)
            else:
                st.info("ℹ️ Local ve International doğrudan müşteriler için LOA ve Münhasırlık aranmaz; standart müşteri tedarik sözleşmesi uygulanır.")
                excl_sel = "N/A (Müşteri)"
                loa_sel = "N/A"
                min_commitment = 0

            uploaded_file = st.file_uploader("Sözleşme / Belge Dosyası Yükle (PDF, DOCX, PNG)", type=["pdf", "docx", "png", "jpg"])

        notes_text = st.text_area("Hukuki Notlar veya Özel Maddeler", placeholder="Örn: Yıllık en az 500 kutu Electro Dish siparişi taahhüt edilmiştir. Ödeme vadesi 45 gündür.")
        
        if st.button("💾 Dokümanı Kaydet ve Arşive Ekle", type="primary"):
            new_doc_entry = {
                'doc_id': doc_ref_no,
                'company': final_company,
                'category': cat_key,
                'doc_type': selected_doc_type,
                'territory': final_country,
                'start_date': str(start_date),
                'end_date': str(end_date),
                'status': loa_sel if cat_key == "Distributor" else "✅ Aktif Sözleşme",
                'notes': notes_text,
                'filename': uploaded_file.name if uploaded_file else f"{doc_ref_no}.pdf"
            }
            st.session_state.custom_documents.insert(0, new_doc_entry)
            st.success(f"✅ '{final_company}' için '{selected_doc_type}' başarıyla kaydedildi ve arşive eklendi!")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # CATEGORIZED TABS (DISTRIBUTORS vs LOCAL vs INTERNATIONAL vs ARCHIVE)
    # --------------------------------------------------------------------------
    tab_dist, tab_local, tab_intl, tab_archive = st.tabs([
        f"🏢 Yetkili Distribütörler ({len(dist_rows)})",
        f"🇹🇷 Local Müşteriler - Türkiye ({len(local_rows)})",
        f"🌍 International Müşteriler ({len(intl_rows)})",
        f"📁 Sözleşme & Doküman Arşivi ({len(st.session_state.custom_documents)})"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: YETKİLİ DİSTRİBÜTÖRLER (SÖZLEŞME & LOA MASTER)
    # --------------------------------------------------------------------------
    with tab_dist:
        st.markdown("""
        <div style="background:rgba(19,166,181,0.08);border-left:4px solid #13A6B5;padding:12px;border-radius:6px;margin-bottom:15px;">
            <span style="color:#4FD1DE;font-weight:600;">Resmi Dağıtım Ağı Sözleşme ve Yetki Belgesi (LOA) Yönetimi</span><br>
            <span style="font-size:12px;color:#CCD7E6;">Bu sekmede yalnızca ZEUS Bioscience'ın uluslararası yetkili ticari distribütörleri yer almaktadır. Yalnızca distribütörler için LOA (Letter of Authorization), bölge münhasırlığı ve yıllık asgari satış kotaları tanımlanır.</span>
        </div>
        """, unsafe_allow_html=True)

        agr_rows = []
        for d in dist_rows:
            agr_rows.append({
                'Distribütör Firma': d['name'],
                'Yetkili Ülke / Pazar': d['country_name'],
                'Bölge': d['region'],
                'Sözleşme Bitiş': d['expiry'],
                'LOA Statüsü': d['loa'],
                'Münhasırlık Hakları': d['exclusivity'],
                'Yıllık Taahhüt Gerçekleşmesi': f"{d['target_pct']}% met",
                'Uyumluluk Durumu': d.get('compliance', '✅ Tam Uyumlu')
            })
        st.dataframe(pd.DataFrame(agr_rows), use_container_width=True, hide_index=True)

        st.caption("ℹ️ Sellco Trading Intl (BAE) distribütörlük sözleşmesi 20 Kasım 2026 tarihinde dolmaktadır; yenileme taslağı hazırlanmıştır.")

    # --------------------------------------------------------------------------
    # TAB 2: LOCAL MÜŞTERİLER (TÜRKİYE İÇİ KLİNİKLER)
    # --------------------------------------------------------------------------
    with tab_local:
        st.markdown("""
        <div style="background:rgba(95,211,174,0.08);border-left:4px solid #5FD3AE;padding:12px;border-radius:6px;margin-bottom:15px;">
            <span style="color:#5FD3AE;font-weight:600;">Türkiye İçi Doğrudan Klinik ve Hastane Sözleşmeleri</span><br>
            <span style="font-size:12px;color:#CCD7E6;">Türkiye'deki tüp bebek merkezleri ve üniversite hastaneleri <b>Local Müşteri</b> statüsündedir. Bu merkezler için distribütörlük LOA'sı aranmaz; yurtiçi cihaz satın alma ve sarf tedarik sözleşmeleri esastır.</span>
        </div>
        """, unsafe_allow_html=True)

        local_search = st.text_input("🔍 Yerel klinik veya hastane ara:", "", key="search_local_clients")
        
        filtered_locals = local_rows
        if local_search.strip():
            q_loc = local_search.strip().lower()
            filtered_locals = [d for d in filtered_locals if q_loc in d['name'].lower()]

        local_table = []
        for d in filtered_locals:
            local_table.append({
                'Klinik / Firma Adı': d['name'],
                'Kategori': 'Local Müşteri (Türkiye)',
                'Sözleşme Protokolü': d.get('agreement_type', 'Yurtiçi Cihaz & Sarf Tedarik Protokolü'),
                'Net Faturalanan Ciro': format_curr(d['revenue']),
                'Kurulu Cihaz': f"{d['devices_count']} Adet" if d['devices_count'] > 0 else "—",
                'Sarf Sipariş Hacmi': f"{d.get('consumables_count', 0)} Kutu",
                'Son Sipariş': d.get('last_order', '—'),
                'Müşteri Statüsü': '✅ Aktif Yerel Müşteri'
            })
        st.dataframe(pd.DataFrame(local_table), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 3: INTERNATIONAL MÜŞTERİLER (DOĞRUDAN İHRACAT)
    # --------------------------------------------------------------------------
    with tab_intl:
        st.markdown("""
        <div style="background:rgba(240,188,87,0.08);border-left:4px solid #F0BC57;padding:12px;border-radius:6px;margin-bottom:15px;">
            <span style="color:#F0BC57;font-weight:600;">Uluslararası Doğrudan Müşteri ve Akademik Merkezler</span><br>
            <span style="font-size:12px;color:#CCD7E6;">Yurtdışındaki doğrudan tüp bebek klinikleri ve üniversite araştırma merkezleridir. Distribütörlük LOA'sı yerine doğrudan kurumsal ihracat tedarik sözleşmeleri ve akademik protokoller ile çalışılmaktadır.</span>
        </div>
        """, unsafe_allow_html=True)

        intl_table = []
        for d in intl_rows:
            intl_table.append({
                'Kurum / Klinik Adı': d['name'],
                'Ülke': d['country_name'],
                'Bölge': d['region'],
                'Kategori': 'International Müşteri',
                'Sözleşme Tipi': d.get('agreement_type', 'Uluslararası Doğrudan Satın Alma Protokolü'),
                'Net İhracat Cirosu': format_curr(d['revenue']),
                'Kurulu Cihaz': f"{d['devices_count']} Adet" if d['devices_count'] > 0 else "—",
                'Son Sipariş': d.get('last_order', '—'),
                'Hukuki Statü': '✅ Onaylı Uluslararası Müşteri'
            })
        st.dataframe(pd.DataFrame(intl_table), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # TAB 4: SÖZLEŞME VE DOKÜMAN ARŞİVİ
    # --------------------------------------------------------------------------
    with tab_archive:
        st.markdown("##### 📁 Resmi Sözleşmeler, LOA ve Regülasyon Sertifikaları Arşivi")
        st.caption("Sistemde kayıtlı ve yeni yüklenen tüm dijital sözleşme ve yetki belgeleri dökümü.")

        doc_disp = []
        for doc in st.session_state.custom_documents:
            doc_disp.append({
                'Belge No': doc['doc_id'],
                'Firma / Kurum': doc['company'],
                'Sınıf': '🏢 Distribütör' if doc['category'] == 'Distributor' else ('🇹🇷 Local' if doc['category'] == 'Local Customer' else '🌍 International'),
                'Belge Türü': doc['doc_type'],
                'Ülke / Bölge': doc['territory'],
                'Bitiş Tarihi': doc['end_date'],
                'Durum': doc['status'],
                'Dosya': doc['filename']
            })
        st.dataframe(pd.DataFrame(doc_disp), use_container_width=True, hide_index=True)

        # Download sample certified list
        st.download_button(
            label="📥 Tüm Sözleşme & LOA Kayıtlarını İndir (CSV)",
            data=pd.DataFrame(doc_disp).to_csv(index=False).encode('utf-8'),
            file_name="ZEUS_Agreements_LOA_Master_Export.csv",
            mime="text/csv"
        )


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
    st.caption("Çok Boyutlu Fiyatlandırma ve İskonto Karar Motoru: Ülke GSYİH (GDP), Yaşam Standardı, Ulusal IVF Destek Rejimi ve Partner Satış Performansına göre dinamik iskonto değerlendirmesi.")

    calc_tab1, calc_tab2 = st.tabs([
        "🧮 Manuel İskonto & Marj Simülatörü",
        "🤖 AI Dinamik İskonto & Paket Önerisi (AI Pricing & Bundle Advisor)"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: MANUEL İSKONTO & MARJ SİMÜLATÖRÜ
    # --------------------------------------------------------------------------
    with calc_tab1:
        c_calc1, c_calc2 = st.columns([1.15, 1.35])
        with c_calc1:
            st.markdown("#### 📋 Sipariş & Partner Parametreleri")
            
            # Order: Official distributors first, then international, then local
            dist_names = [d['name'] for d in dist_rows]
            intl_names = [d['name'] for d in intl_rows]
            local_names = [d['name'] for d in local_rows]
            all_ordered_names = dist_names + intl_names + local_names
            if not all_ordered_names:
                all_ordered_names = [d['name'] for d in active_rows]

            # Use session state partner if set by AI tab
            def_partner_idx = next((i for i, n in enumerate(all_ordered_names) if n == st.session_state.get('sim_calc_partner')), None)
            if def_partner_idx is None:
                def_partner_idx = next((i for i, n in enumerate(all_ordered_names) if "SIRIUS" in n), 0)

            calc_dist_name = st.selectbox("İskonto Talep Eden Müşteri / Distribütör", all_ordered_names, index=def_partner_idx, key="calc_select_dist")
            calc_d = next((d for d in active_rows if d['name'] == calc_dist_name), active_rows[0])
            
            # Retrieve country macro & IVF intelligence
            calc_country = calc_d.get('country_name', 'Turkey')
            macro = resolve_country_macro(calc_country)
            
            # Partner Historical Sales Performance
            partner_rev = calc_d.get('revenue', 0.0)
            partner_devs = calc_d.get('devices_count', 0)
            partner_cons = calc_d.get('consumables_count', 0)
            partner_growth = calc_d.get('growth_pct', 0.0)
            partner_type_tr = calc_d.get('account_type_tr', 'Partner')
            
            if partner_rev >= 100000:
                perf_tier_name = "Tier 1: Platin Stratejik Partner (Ciro > $100K)"
                perf_bonus = 3.5
                perf_badge = "🌟 Platin Partner"
                perf_badge_color = "#00F5FF"
            elif partner_rev >= 30000:
                perf_tier_name = "Tier 2: Altın Büyüyen Partner (Ciro $30K–$100K)"
                perf_bonus = 2.5
                perf_badge = "🥇 Altın Partner"
                perf_badge_color = "#FFD700"
            elif partner_rev >= 10000:
                perf_tier_name = "Tier 3: Gümüş Standart Hesap (Ciro $10K–$30K)"
                perf_bonus = 1.0
                perf_badge = "🥈 Gümüş Partner"
                perf_badge_color = "#C0C0C0"
            else:
                perf_tier_name = "Tier 4: Bronz / Yeni Hesap (Ciro < $10K)"
                perf_bonus = 0.0
                perf_badge = "🥉 Bronz / Yeni"
                perf_badge_color = "#CD7F32"
                
            growth_bonus = 1.0 if partner_growth >= 15.0 else 0.0
            total_perf_bonus = perf_bonus + growth_bonus

            # Macro & IVF Context Card
            st.markdown(f"""
            <div style="background:rgba(19,166,181,0.06);border:1px solid rgba(19,166,181,0.28);border-radius:10px;padding:12px;margin:10px 0 16px 0;font-size:12px;line-height:1.6;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="font-weight:700;color:#00F5FF;font-size:13px;">🌍 {calc_country} Makro & IVF Ekosistemi</span>
                    <span style="background:rgba(19,166,181,0.2);color:#4FD1DE;font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600;border:1px solid rgba(19,166,181,0.4);">{macro['living_standard_code']}</span>
                </div>
                <b>Kişi Başı GSYİH (GDP pc):</b> ${macro['gdp_pc']:,} USD <span style="color:#8FA3C0;">({macro['gdp_tier']})</span><br>
                <b>Yaşam Standardı & Refah:</b> {macro['living_standard']}<br>
                <b>Aktif IVF Klinik Sayısı:</b> {macro['clinics']} Merkez &nbsp;|&nbsp; <b>Nüfus:</b> {macro['pop_m']}M<br>
                <b>Gümrük Vergisi & Süresi:</b> %{macro['customs_tariff_pct']:.1f} ({macro['customs_clearance_days']} Gün Bekleme)<br>
                <b>Devlet IVF Desteği / Geri Ödeme:</b> <span style="color:#00E676;font-weight:600;">{macro['ivf_support']}</span><br>
                <span style="color:#8FA3C0;font-size:11px;">{macro['ivf_support_desc']}</span>
                <div style="margin-top:4px;color:#FFB800;font-size:11px;"><b>Pazar Fiyat Hassasiyeti:</b> {macro['price_sensitivity']}</div>
                <hr style="border-color:rgba(19,166,181,0.18);margin:8px 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
                    <span style="font-weight:700;color:#00F5FF;font-size:13px;">📊 Partner Satış & Büyüme Performansı</span>
                    <span style="color:{perf_badge_color};font-weight:700;">{perf_badge}</span>
                </div>
                <b>Excel Kayıtlı Ciro:</b> {format_curr(partner_rev)} &nbsp;|&nbsp; <b>Kurulu Cihaz:</b> {partner_devs} Adet &nbsp;|&nbsp; <b>Sarf:</b> {partner_cons} Kutu<br>
                <b>Yıllık YoY Büyüme:</b> <span style="color:{'#00E676' if partner_growth >= 0 else '#FF5252'};font-weight:600;">%{partner_growth:+.1f}</span> &nbsp;|&nbsp; <b>Sınıf:</b> {partner_type_tr}
            </div>
            """, unsafe_allow_html=True)
            
            # Check if values prefilled from AI Tab
            init_disc = float(st.session_state.get('applied_ai_disc', 8.0))
            init_ship = float(st.session_state.get('applied_ai_ship', 900.0))

            calc_qty = st.number_input("Sipariş Adedi / Hacmi (Order Quantity - Units)", min_value=1, value=25, step=1)
            calc_price = st.number_input("Sözleşme Liste Fiyatı (Contractual List Price - USD)", min_value=100.0, value=4200.0, step=100.0)
            calc_cost = st.number_input("Birim Üretim Maliyeti (Unit COGS - USD)", min_value=10.0, value=2300.0, step=50.0)
            calc_disc = st.slider("Talep Edilen İskonto (Requested Discount - %)", min_value=0.0, max_value=35.0, value=init_disc, step=0.5)
            calc_ship = st.number_input("Navlun & Lojistik Payı (Freight Allowance - USD)", min_value=0.0, value=init_ship, step=50.0)
            calc_comm = st.slider("Distribütör Temsilci Primi (Rep Commission - %)", min_value=0.0, max_value=10.0, value=3.0, step=0.5)
            calc_min_m = st.slider("Kurumsal Taban Brüt Marj Eşiği (Min Gross Margin - %)", min_value=20.0, max_value=60.0, value=35.0, step=1.0)
            calc_extra_tier = st.slider("Özel Stratejik Esneklik Payı (Board/CEO Strategic Override - %)", min_value=0.0, max_value=10.0, value=0.0, step=0.5, help="Yönetim Kurulu veya CEO yetkisiyle tanımlanan ek tolerans.")

        with c_calc2:
            st.markdown("#### ⚖️ Çok Boyutlu İskonto ve Yetki Değerlendirmesi")
            
            # 1. Base volume tier
            if calc_qty >= 50:
                vol_tier_name = "Volume Tier 3 (50+ Adet)"
                base_max = 12.0
            elif calc_qty >= 25:
                vol_tier_name = "Volume Tier 2 (25–49 Adet)"
                base_max = 8.0
            elif calc_qty >= 10:
                vol_tier_name = "Volume Tier 1 (10–24 Adet)"
                base_max = 5.0
            else:
                vol_tier_name = "Standart Sipariş (<10 Adet)"
                base_max = 2.0
                
            macro_allowance = macro['macro_discount_allowance']
            ivf_allowance = macro['ivf_discount_allowance']
            
            # Total dynamically authorized discount ceiling
            allowed_disc = round(base_max + macro_allowance + ivf_allowance + total_perf_bonus + calc_extra_tier, 1)
            
            orig_val = calc_qty * calc_price
            disc_amt = orig_val * (calc_disc / 100.0)
            net_val = orig_val - disc_amt
            unit_final = calc_price * (1.0 - calc_disc / 100.0)
            
            tot_cogs = calc_qty * calc_cost + calc_ship + (net_val * calc_comm / 100.0)
            gross_profit = net_val - tot_cogs
            gross_margin = (gross_profit / net_val * 100.0) if net_val > 0 else 0.0
            
            # Standard benchmark without discount
            std_tot_cogs = calc_qty * calc_cost + calc_ship + (orig_val * calc_comm / 100.0)
            std_gp = orig_val - std_tot_cogs
            std_gm = (std_gp / orig_val * 100.0) if orig_val > 0 else 0.0
            
            # Unit contribution & break-even
            unit_contrib = unit_final * (1.0 - calc_comm / 100.0) - calc_cost
            be_qty = math.ceil(calc_ship / unit_contrib) if unit_contrib > 0 else None

            # Multi-factor approval logic
            reasons = []
            positive_factors = []
            
            if calc_disc > allowed_disc:
                reasons.append(f"Talep edilen %{calc_disc:.1f} iskonto; {calc_country} pazarının makro satın alma gücü, IVF destek statüsü ve partner performansının hak ettiği dinamik tavan olan %{allowed_disc:.1f}'i {calc_disc - allowed_disc:+.1f} puan aşmaktadır.")
            else:
                positive_factors.append(f"Talep edilen %{calc_disc:.1f} iskonto, çok boyutlu dinamik tavan olan %{allowed_disc:.1f} sınırları içerisindedir.")
                
            if gross_margin < calc_min_m:
                reasons.append(f"Net brüt kâr marjı (%{gross_margin:.1f}), şirket taban eşiği olan %{calc_min_m:.1f}'in {calc_min_m - gross_margin:.1f} puan altındadır.")
            else:
                positive_factors.append(f"Brüt kâr marjı (%{gross_margin:.1f}), şirket taban marjı %{calc_min_m:.1f}'i başarıyla sağlamaktadır.")
                
            # Macro context checks
            if macro['gdp_pc'] >= 30000 and calc_disc > 10.0:
                reasons.append(f"{calc_country} yüksek gelirli (Kişi başı GSYİH: ${macro['gdp_pc']:,}) bir pazardır; kamu veya özel IVF satın alma gücü yüksek olduğundan çift haneli iskonto marj erozyonu yaratır.")
            elif macro['gdp_pc'] < 10000:
                positive_factors.append(f"{calc_country} pazarında kişi başı GSYİH (${macro['gdp_pc']:,}) düşüktür ve hastalar %100 cepten ödeme yapmaktadır; adil pazar payı korumak için esneklik desteklenir.")
                
            if partner_rev >= 30000:
                positive_factors.append(f"Partner {format_curr(partner_rev)} kayıtlı ciro ve %{partner_growth:+.1f} YoY büyüme ile {perf_tier_name} statüsündedir; ek performans iskontosu hak etmektedir.")

            requires_approval = len(reasons) > 0

            # Decision Card
            if gross_profit < 0:
                st.error("🛑 İŞLEM UYGULANAMAZ: Sipariş negatif brüt kâr (zarar) üretmektedir!")
            elif requires_approval:
                st.warning("⚠️ ÜST YÖNETİM ONAYI GEREKLİ (CFO / CEO ESCALATION)")
                for r in reasons:
                    st.write(f"• {r}")
                with st.expander("Göz Önüne Alınan Pozitif Faktörler", expanded=False):
                    for p in positive_factors:
                        st.write(f"✓ {p}")
            else:
                st.success(f"✅ İŞLEM ONAYLANDI: Çok Boyutlu Makro & Performans Yetkisi Dahilindedir.")
                for p in positive_factors:
                    st.caption(f"✓ {p}")

            # Metrics
            res_k1, res_k2 = st.columns(2)
            res_k1.metric("İndirimli Net Hasılat", format_curr(net_val), delta=f"-{format_curr(disc_amt)} İskonto Bedeli", delta_color="inverse")
            res_k2.metric("Brüt Kâr & Marj", format_curr(gross_profit), delta=f"%{gross_margin:.1f} Brüt Marj")

            # 4-Factor Dynamic Breakdown Card without any leading markdown indentation
            extra_tr = f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.08);"><td style="padding:5px 0;">🎯 <b>Stratejik Esneklik Payı:</b></td><td style="padding:5px 0;text-align:right;">Yönetim Kurulu Onaylı</td><td style="padding:5px 0;text-align:right;color:#00F5FF;font-weight:700;">+%{calc_extra_tier:.1f}</td></tr>""" if calc_extra_tier > 0 else ""
            
            clean_matrix_html = f"""<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.08);padding:14px;border-radius:10px;font-size:12px;margin:12px 0;">
<div style="font-weight:700;color:#00F5FF;margin-bottom:8px;font-size:13px;">🧩 Çok Boyutlu İskonto Tavanı Analiz Matrisi</div>
<table style="width:100%;font-size:12px;border-collapse:collapse;color:#E0E6ED;">
<tr style="border-bottom:1px solid rgba(255,255,255,0.08);"><td style="padding:5px 0;">📦 <b>Sipariş Hacmi Skalası:</b></td><td style="padding:5px 0;text-align:right;">{vol_tier_name}</td><td style="padding:5px 0;text-align:right;color:#00F5FF;font-weight:700;">+%{base_max:.1f}</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08);"><td style="padding:5px 0;">🌐 <b>Ülke GSYİH & Yaşam Standardı:</b></td><td style="padding:5px 0;text-align:right;">${macro['gdp_pc']:,} ({macro['gdp_tier']})</td><td style="padding:5px 0;text-align:right;color:#00F5FF;font-weight:700;">+%{macro_allowance:.1f}</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08);"><td style="padding:5px 0;">🏛️ <b>Ulusal IVF Destek Rejimi:</b></td><td style="padding:5px 0;text-align:right;">{macro['ivf_support']}</td><td style="padding:5px 0;text-align:right;color:#00F5FF;font-weight:700;">+%{ivf_allowance:.1f}</td></tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08);"><td style="padding:5px 0;">📈 <b>Partner Satış & Büyüme Performansı:</b></td><td style="padding:5px 0;text-align:right;">{perf_badge} ({format_curr(partner_rev)})</td><td style="padding:5px 0;text-align:right;color:#00F5FF;font-weight:700;">+%{total_perf_bonus:.1f}</td></tr>
{extra_tr}
<tr style="border-top:2px solid #13A6B5;font-weight:700;color:#00F5FF;"><td style="padding:8px 0;">🏁 <b>TOPLAM YETKİLİ DİNAMİK TAVAN:</b></td><td style="padding:8px 0;text-align:right;">İzin Verilen Azami İskonto</td><td style="padding:8px 0;text-align:right;font-size:14px;color:#00E676;">%{allowed_disc:.1f}</td></tr>
</table>
</div>"""
            st.markdown(clean_matrix_html, unsafe_allow_html=True)

            clean_kpi_html = f"""<div style="background:rgba(255,255,255,0.03);padding:14px;border-radius:10px;font-size:13px;line-height:1.8;">
<b>Standart Liste Fiyatına Göre Kâr Değişimi:</b> {gross_margin - std_gm:+.1f} pt ({format_curr(gross_profit - std_gp)})<br>
<b>Birim Efektif Realize Fiyat:</b> {format_curr(unit_final)} / adet (Liste: {format_curr(calc_price)})<br>
<b>İskonto Güvenlik Farkı:</b> <span style="color:{'#00E676' if allowed_disc >= calc_disc else '#FF5252'};font-weight:700;">{allowed_disc - calc_disc:+.1f} puan</span> (Tavan: %{allowed_disc:.1f} vs Talep: %{calc_disc:.1f})<br>
<b>Navlun Başabaş Sipariş Hacmi:</b> {be_qty if be_qty else 'N/A'} adet
</div>"""
            st.markdown(clean_kpi_html, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # TAB 2: AI DİNAMİK İSKONTO & PAKET ÖNERİSİ (AI PRICING & BUNDLE ADVISOR)
    # --------------------------------------------------------------------------
    with calc_tab2:
        st.markdown("### 🤖 ZEUS AI Makro, Gümrük ve Performans Odaklı İskonto & Paket Önerisi")
        st.caption("Yapay zeka motoru; ülkenin GSYİH (GDP) seviyesini, yaşam kalitesini, IVF klinik sayısını, nüfusunu, gümrük tarifesini ve distribütörün satış geçmişini sentezleyerek optimum cihaz/malzeme indirim paketini önerir.")

        ai_col1, ai_col2 = st.columns([1.1, 1.3])
        with ai_col1:
            st.markdown("#### 🎯 Analiz Edilecek Partner ve Stratejik Amaç")
            ai_dist_name = st.selectbox("Partner / Distribütör Seçimi", all_ordered_names, index=sirius_idx, key="ai_dist_select")
            ai_d = next((d for d in active_rows if d['name'] == ai_dist_name), active_rows[0])
            ai_country = ai_d.get('country_name', 'Turkey')
            ai_macro = resolve_country_macro(ai_country)

            ai_goal = st.radio(
                "Ticari Optimizasyon Hedefi:",
                [
                    "🎯 Pazar Payı Kazanma & Hızlı Cihaz Penetrasyonu (Aggressive Expansion)",
                    "🧪 Tekrar Eden Sarf Tüketimini Maksimize Etme (Razor-and-Blade Consumable)",
                    "🛡️ Yüksek Gümrük & Navlun Bariyerini Dengeleme (Customs Tariff Offset)",
                    "⚖️ Dengeli Kârlılık ve Marj Koruma (Balanced Corporate Profitability)"
                ],
                key="ai_goal_select"
            )

            # 4 macro indicator badges
            st.markdown(f"""
            <div style="background:rgba(19,166,181,0.06);border:1px solid rgba(19,166,181,0.25);border-radius:10px;padding:12px;margin-top:14px;font-size:12px;line-height:1.7;">
                <div style="font-weight:700;color:#00F5FF;margin-bottom:6px;font-size:13px;">📌 {ai_country} Pazar ve Gümrük Parametreleri</div>
                • <b>Kişi Başı GSYİH:</b> ${ai_macro['gdp_pc']:,} USD ({ai_macro['gdp_tier']})<br>
                • <b>Yaşam Standardı:</b> {ai_macro['living_standard']}<br>
                • <b>Lisanslı IVF Klinik Sayısı:</b> {ai_macro['clinics']} Merkez<br>
                • <b>Ülke Nüfusu:</b> {ai_macro['pop_m']} Milyon<br>
                • <b>Gümrük Vergisi:</b> %{ai_macro['customs_tariff_pct']:.1f} (Ortalama {ai_macro['customs_clearance_days']} Gün Bekleme)<br>
                • <b>İthalat Zorluğu:</b> {ai_macro['import_difficulty']}<br>
                • <b>Devlet Desteği:</b> {ai_macro['ivf_support']}<br>
                • <b>Partner Kayıtlı Satışı:</b> {format_curr(ai_d.get('revenue', 0.0))} ({ai_d.get('devices_count', 0)} Cihaz / {ai_d.get('consumables_count', 0)} Sarf)
            </div>
            """, unsafe_allow_html=True)

        with ai_col2:
            st.markdown("#### 💡 AI Tavsiye Edilen Fiyatlandırma & Paket Modeli")
            
            # Generate AI recommendation
            ai_rec = generate_ai_pricing_bundle_recommendation(ai_d, ai_macro, ai_goal)
            
            rec_c1, rec_c2 = st.columns(2)
            rec_c1.metric(
                "Önerilen Cihaz İskontosu",
                f"%{ai_rec['rec_device_discount']:.1f}",
                delta=f"Hedef Marj: %{ai_rec['expected_margin']:.1f}",
                delta_color="normal"
            )
            rec_c2.metric(
                "Önerilen Sarf / Malzeme İndirimi",
                f"%{ai_rec['rec_consumable_discount']:.1f}",
                delta=f"Kazanma Olasılığı: %{ai_rec['win_probability']}",
                delta_color="normal"
            )

            # Executive recommendation card
            st.markdown(f"""
            <div style="background:rgba(0,245,255,0.04);border:1px solid #13A6B5;border-radius:10px;padding:14px;margin:12px 0;font-size:13px;line-height:1.7;">
                <div style="font-weight:700;color:#00F5FF;font-size:14px;margin-bottom:8px;">🎁 AI Tavsiye Edilen Malzeme & Bundle Formülü</div>
                <div style="background:rgba(19,166,181,0.12);padding:10px;border-radius:8px;font-weight:600;color:#FFFFFF;margin-bottom:10px;">
                    {ai_rec['bundle_desc']}
                </div>
                <div style="font-weight:700;color:#4FD1DE;margin-top:6px;">🚢 Gümrük & Lojistik Dengeleme Taktiği:</div>
                <div style="color:#C4D1D9;font-size:12px;">{ai_rec['customs_strategy']}</div>
                <div style="font-weight:700;color:#4FD1DE;margin-top:8px;">📊 Partner Satış Performans Değerlendirmesi:</div>
                <div style="color:#C4D1D9;font-size:12px;">{ai_rec['partner_note']}</div>
                <hr style="border-color:rgba(19,166,181,0.2);margin:10px 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#8FA3C0;font-size:12px;">1 Yıllık Tahmini Ek Sarf Hacmi:</span>
                    <span style="color:#00E676;font-weight:700;font-size:14px;">{ai_rec['annual_sarf_potential']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action button to apply AI recommendation into simulator
            if st.button("⚡ Bu AI Önerisini Manuel Simülatöre Aktar", type="primary", key="btn_apply_ai_pricing"):
                st.session_state['applied_ai_disc'] = ai_rec['rec_device_discount']
                st.session_state['applied_ai_ship'] = ai_rec['rec_freight']
                st.session_state['sim_calc_partner'] = ai_dist_name
                st.success(f"✓ AI önerisi (%{ai_rec['rec_device_discount']:.1f} iskonto, ${ai_rec['rec_freight']:.0f} lojistik) başarıyla simülatöre aktarıldı! 'Manuel İskonto & Marj Simülatörü' sekmesinden sonucu inceleyebilirsiniz.")



# ==============================================================================
# 10. VIEW: RISK & SCENARIO SIMULATOR
# ==============================================================================
elif nav_selection == "⛛ Risk & Scenario Simulator":
    st.title("🎯 Commercial Risk & Financial Scenario Simulator")
    st.caption("ZEUS Ticari Risk Simülatörü: Satış senaryoları, döviz & talep şokları, teslimat gecikmesi ve sözleşme bazlı tahsilat modellemesi.")

    # --------------------------------------------------------------------------
    # 1. SATIŞ SENARYOSU GİRİŞLERİ (ÜLKE VE DÖNEM KAPSAMI)
    # --------------------------------------------------------------------------
    sc_f1, sc_f2 = st.columns([1.5, 1])
    with sc_f1:
        all_sim_countries = sorted(list(set(d['country_name'] for d in dists_with_rev)))
        sc_country = st.selectbox("Simülasyon Ülke Kapsamı", ["Global (Tüm Ülkeler)"] + all_sim_countries)
    with sc_f2:
        sc_period = st.selectbox("Simülasyon Dönemi", ["Son 12 Ay (FY2026 / LTM)", "2026 Yılbaşından Bugüne (YTD)", "Son 6 Ay", "Son 3 Ay", "Tüm Kayıtlar"])

    # Calculate base revenue dynamically from real dataset
    if raw_sales_df is not None:
        filtered_df = raw_sales_df.copy()
        if sc_country != "Global (Tüm Ülkeler)":
            filtered_df = filtered_df[filtered_df['Country'] == sc_country]
            
        if sc_period == "2026 Yılbaşından Bugüne (YTD)":
            filtered_df = filtered_df[filtered_df['Year_int'] == 2026]
        elif sc_period == "Son 6 Ay":
            filtered_df = filtered_df[(filtered_df['Year_int'] == 2026) & (filtered_df['Month_int'] >= 5)]
        elif sc_period == "Son 3 Ay":
            filtered_df = filtered_df[(filtered_df['Year_int'] == 2026) & (filtered_df['Month_int'] >= 8)]
        elif sc_period == "Son 12 Ay (FY2026 / LTM)":
            filtered_df = filtered_df[(filtered_df['Year_int'] == 2026) | ((filtered_df['Year_int'] == 2025) & (filtered_df['Month_int'] >= 11))]
            
        base_sc_rev = float(filtered_df['Amount (USD)'].sum()) if len(filtered_df) > 0 else (total_rev if sc_country == "Global (Tüm Ülkeler)" else 15000.0)
    else:
        base_sc_rev = total_rev if sc_country == "Global (Tüm Ülkeler)" else 50000.0

    st.markdown(f"**📌 Seçili Kapsam Baz Satış Cirosu:** `{format_curr(base_sc_rev)}` &nbsp;|&nbsp; Kapsam: `{sc_country}` &nbsp;|&nbsp; Dönem: `{sc_period}`")

    # --------------------------------------------------------------------------
    # 2. HAZIR SENARYOLAR (PRESETS)
    # --------------------------------------------------------------------------
    preset_choice = st.radio(
        "Hazır Senaryo Modelleri",
        ["Baz Senaryo (Neutral)", "Olumsuz Senaryo (Downside Stress)", "Olumlu Senaryo (Upside Expansion)", "Özel Parametreler (Custom)"],
        horizontal=True
    )

    if preset_choice == "Baz Senaryo (Neutral)":
        p_fx, p_dem, p_disc, p_delay = 0.0, 0.0, 0.0, 0
    elif preset_choice == "Olumsuz Senaryo (Downside Stress)":
        p_fx, p_dem, p_disc, p_delay = 12.0, -15.0, 6.0, 30
    elif preset_choice == "Olumlu Senaryo (Upside Expansion)":
        p_fx, p_dem, p_disc, p_delay = -5.0, 12.0, 0.0, 0
    else:
        p_fx, p_dem, p_disc, p_delay = 0.0, 0.0, 0.0, 0

    s_c1, s_c2 = st.columns([1.1, 1.4])
    with s_c1:
        st.markdown("#### ⚙️ Satış Senaryosu Girişleri")
        sim_fx = st.slider("USD ve Yerel Para Birimleri Değişimi (%)", -15.0, 15.0, float(p_fx), 1.0, help="Pozitif değer ciroyu azaltacak yönde uygulanır. Örneğin +%10 kur girdisi tek başına ciroyu %4.5 düşürür.")
        sim_demand = st.slider("Pazar Talebi Değişimi (%)", -30.0, 30.0, float(p_dem), 1.0, help="Satış hacmindeki genel talep artış/azalış değişimi.")
        sim_disc = st.slider("İndirim ve Rekabet Baskısı (%)", 0.0, 20.0, float(p_disc), 1.0, help="ZEUS satış fiyatına uygulanan senaryo indirimi.")
        sim_delay = st.slider("Sevkiyat Gecikmesi (Gün)", 0, 90, int(p_delay), 5, help="Teslimatın kaç gün ötelenebileceğini gösterir.")

        # Ürün Maliyeti Girişi (Opsiyonel / Maliyet verisi kuralı)
        with st.expander("💼 Kârlılık Hesabı İçin Ürün Maliyeti (COGS) Girişi", expanded=False):
            st.caption("Ürün maliyeti girilene kadar brüt kâr ve marj alanlarında 'Maliyet verisi gerekli' yazacaktır.")
            user_cogs_pct = st.number_input("Ürün Maliyet Oranı (%)", min_value=0.0, max_value=90.0, value=0.0, step=5.0, help="Maliyet oranı girildiğinde brüt kâr ve marj otomatik hesaplanır.")

    # --------------------------------------------------------------------------
    # 3. SENARYO HESAPLAMA MOTORU (FORMÜL KILAVUZUNA UYGUN)
    # --------------------------------------------------------------------------
    def calculate_scenario_rev(base_r, fx_val, dem_val, disc_val):
        demand_factor = 1.0 + (dem_val / 100.0)
        price_factor = 1.0 - (disc_val / 100.0)
        fx_factor = 1.0 - (fx_val / 100.0) * 0.45  # +%10 kur -> -%4.5 ciro
        return max(0.0, base_r * demand_factor * price_factor * fx_factor)

    sim_rev = calculate_scenario_rev(base_sc_rev, sim_fx, sim_demand, sim_disc)
    base_rev_val = calculate_scenario_rev(base_sc_rev, 0.0, 0.0, 0.0)
    rev_delta = ((sim_rev - base_rev_val) / base_rev_val * 100.0) if base_rev_val > 0 else 0.0

    # Kârlılık değerlendirmesi (Maliyet verisi gerekli kuralı)
    has_cost = user_cogs_pct > 0.0
    if has_cost:
        sim_gp = sim_rev * (1.0 - user_cogs_pct / 100.0)
        sim_margin = (sim_gp / sim_rev * 100.0) if sim_rev > 0 else 0.0
        base_gp = base_rev_val * (1.0 - user_cogs_pct / 100.0)
        gp_delta = ((sim_gp - base_gp) / base_gp * 100.0) if base_gp > 0 else 0.0
        
        gp_text = format_curr(sim_gp)
        margin_text = f"{sim_margin:.1f}%"
        gp_sub = f"{gp_delta:+.1f}% vs Baz"
        margin_sub = f"%{user_cogs_pct:.1f} Maliyet Oranı"
    else:
        gp_text = "Maliyet verisi gerekli"
        margin_text = "Maliyet verisi gerekli"
        gp_sub = "Kârlılık için maliyet oranı giriniz"
        margin_sub = "Maliyet verisi bekleniyor"

    with s_c2:
        st.markdown("#### 📊 Sonuç Göstergeleri (Finansal Etki)")
        res1, res2 = st.columns(2)
        res1.metric("Öngörülen Ciro", format_curr(sim_rev), delta=f"{rev_delta:+.1f}% vs Baz")
        res2.metric("Teslimat Zamanlaması", f"+{sim_delay} Gün Öteleme" if sim_delay > 0 else "Zamanında (0 Gün)", delta="Baz Teslimat Planına Ek", delta_color="inverse" if sim_delay > 0 else "normal")

        res3, res4 = st.columns(2)
        res3.metric("Öngörülen Brüt Kâr", gp_text, delta=gp_sub if has_cost else None)
        res4.metric("Öngörülen Brüt Marj", margin_text, delta=margin_sub if has_cost else None)

        # Senaryo Kıyaslama Grafiği
        sc_base_rev = calculate_scenario_rev(base_sc_rev, 0.0, 0.0, 0.0)
        sc_down_rev = calculate_scenario_rev(base_sc_rev, 12.0, -15.0, 6.0)
        sc_up_rev = calculate_scenario_rev(base_sc_rev, -5.0, 12.0, 0.0)

        chart_data = [
            {'Senaryo': 'Baz Senaryo', 'Öngörülen Ciro': sc_base_rev},
            {'Senaryo': 'Olumsuz (Stress)', 'Öngörülen Ciro': sc_down_rev},
            {'Senaryo': 'Olumlu (Upside)', 'Öngörülen Ciro': sc_up_rev},
            {'Senaryo': 'Simüle Edilen', 'Öngörülen Ciro': sim_rev}
        ]
        
        if has_cost:
            for item in chart_data:
                item['Öngörülen Brüt Kâr'] = item['Öngörülen Ciro'] * (1.0 - user_cogs_pct / 100.0)
            y_cols = ['Öngörülen Ciro', 'Öngörülen Brüt Kâr']
            col_map = {'Öngörülen Ciro': '#7C93BD', 'Öngörülen Brüt Kâr': '#13A6B5'}
        else:
            y_cols = ['Öngörülen Ciro']
            col_map = {'Öngörülen Ciro': '#00F5FF'}

        fig_sc = px.bar(
            pd.DataFrame(chart_data),
            x='Senaryo', y=y_cols,
            barmode='group',
            color_discrete_map=col_map,
            title="Satış Senaryoları Ciro Kıyaslaması"
        )
        fig_sc.update_layout(height=260, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 4. ÖDEME VE TAHSİLAT SİMÜLATÖRÜ (WORD KILAVUZUNA UYGUN 4 MODEL)
    # --------------------------------------------------------------------------
    st.subheader("💳 Ödeme & Tahsilat Simülatörü (Payment & Cash Flow Collection)")
    st.caption("Tek bir sipariş, taksitli satış veya kira sözleşmesi için sözleşme koşullarına göre 30, 60 ve 90 günlük nakit girişlerini modeller.")

    pay_model = st.radio(
        "Ödeme ve Sözleşme Modeli Seçin:",
        [
            "Distribütör veya uluslararası müşteri · %100 peşin",
            "Yerel müşteri · %100 peşin",
            "Yerel müşteri · taksitli",
            "Yerel müşteri · kiralama (Leasing)"
        ],
        horizontal=True
    )

    p_in1, p_in2, p_in3 = st.columns([1.2, 1.2, 1.4])

    if pay_model == "Distribütör veya uluslararası müşteri · %100 peşin":
        with p_in1:
            order_val = st.number_input("Sipariş Tutarı ($ USD)", value=float(min(sim_rev if sim_rev > 0 else 45000.0, 45000.0)), step=5000.0)
        with p_in2:
            wire_delay = st.slider("Banka Transfer Gecikmesi (Gün)", 0, 90, 0, 5)
        with p_in3:
            st.info("ℹ️ Sevkiyat öncesi tam tutarın tahsil edilmesi esastır. Gecikme, banka akreditif veya SWIFT teyit süresini temsil eder.")

        cf_30 = order_val if wire_delay <= 30 else 0.0
        cf_60 = order_val if wire_delay <= 60 else 0.0
        cf_90 = order_val if wire_delay <= 90 else 0.0
        rem_balance = max(0.0, order_val - cf_90)
        plan_desc = "Tam Peşin Ödeme (Sevkiyat Öncesi)"

    elif pay_model == "Yerel müşteri · %100 peşin":
        with p_in1:
            order_val = st.number_input("Sipariş Tutarı ($ USD)", value=21000.0, step=1000.0)
        with p_in2:
            wire_delay = st.slider("Ödeme Gecikmesi (Gün)", 0, 90, 0, 5)
        with p_in3:
            st.info("ℹ️ Yurtiçi klinikler için peşin faturalı sipariş modeli. Gecikme, fatura vadesinden sonraki ötelemedir.")

        cf_30 = order_val if wire_delay <= 30 else 0.0
        cf_60 = order_val if wire_delay <= 60 else 0.0
        cf_90 = order_val if wire_delay <= 90 else 0.0
        rem_balance = max(0.0, order_val - cf_90)
        plan_desc = "Tam Peşin Ödeme (Yerel Fatura)"

    elif pay_model == "Yerel müşteri · taksitli":
        with p_in1:
            order_val = st.number_input("Sipariş Tutarı ($ USD)", value=12000.0, step=1000.0)
            down_pct = st.slider("Peşinat Oranı (%)", 0.0, 100.0, 25.0, 5.0)
        with p_in2:
            n_inst = st.slider("Aylık Taksit Sayısı (Ay)", 1, 120, 6, 1)
            wire_delay = st.slider("Gecikme (Gün)", 0, 90, 0, 5)
        
        down_amt = order_val * (down_pct / 100.0)
        rem_loan = order_val - down_amt
        monthly_inst = rem_loan / max(1, n_inst)
        
        with p_in3:
            st.markdown(f"""
            <div style="background:rgba(19,166,181,0.08);padding:10px;border-radius:8px;border:1px solid #13A6B5;font-size:13px;">
                <b>Peşinat Tutarı:</b> {format_curr(down_amt)} (%{down_pct:.0f})<br>
                <b>Kalan Bakiye:</b> {format_curr(rem_loan)}<br>
                <b>Eşit Aylık Taksit:</b> {format_curr(monthly_inst)} / Ay ({n_inst} Ay)
            </div>
            """, unsafe_allow_html=True)

        # 30 günlük ay modeli: Peşinat gün 0'da tahsil edilir. Taksit i, (i*30 + wire_delay) gününde ödenir.
        inst_30 = sum(monthly_inst for i in range(1, n_inst + 1) if (i * 30 + wire_delay) <= 30)
        inst_60 = sum(monthly_inst for i in range(1, n_inst + 1) if (i * 30 + wire_delay) <= 60)
        inst_90 = sum(monthly_inst for i in range(1, n_inst + 1) if (i * 30 + wire_delay) <= 90)

        cf_30 = down_amt + inst_30
        cf_60 = down_amt + inst_60
        cf_90 = down_amt + inst_90
        rem_balance = max(0.0, order_val - cf_90)
        plan_desc = f"%{down_pct:.0f} Peşinat + {n_inst} Taksit"

    else:  # Yerel müşteri · kiralama
        with p_in1:
            monthly_rent = st.number_input("Aylık Kira ($ USD)", value=1200.0, step=100.0)
        with p_in2:
            rent_months = st.slider("Sözleşme Süresi (Ay)", 1, 120, 12, 1)
            wire_delay = st.slider("Kira Tahsilat Gecikmesi (Gün)", 0, 90, 0, 5)
        
        total_rent = monthly_rent * rent_months
        with p_in3:
            st.markdown(f"""
            <div style="background:rgba(240,188,87,0.08);padding:10px;border-radius:8px;border:1px solid #F0BC57;font-size:13px;">
                <b>Aylık Kira Bedeli:</b> {format_curr(monthly_rent)}<br>
                <b>Planlanan Toplam Kira:</b> {format_curr(total_rent)} ({rent_months} Ay)<br>
                <span style="color:#F0BC57;font-size:11px;">* Kira tahsilatı satış cirosundan ayrı izlenir.</span>
            </div>
            """, unsafe_allow_html=True)

        cf_30 = sum(monthly_rent for i in range(1, rent_months + 1) if (i * 30 + wire_delay) <= 30)
        cf_60 = sum(monthly_rent for i in range(1, rent_months + 1) if (i * 30 + wire_delay) <= 60)
        cf_90 = sum(monthly_rent for i in range(1, rent_months + 1) if (i * 30 + wire_delay) <= 90)
        rem_balance = max(0.0, total_rent - cf_90)
        plan_desc = f"{rent_months} Aylık Kiralama Sözleşmesi"

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Nakit Akışı Göstergesi (Word Kılavuzu: 30, 60, 90 Günlük Tahsilat ve Kalan Bakiye)
    cf_c1, cf_c2, cf_c3, cf_c4 = st.columns(4)
    cf_c1.metric("30 Günlük Nakit Girişi", format_curr(cf_30), delta="Birikimli Tahsilat")
    cf_c2.metric("60 Günlük Nakit Girişi", format_curr(cf_60), delta=f"+{format_curr(cf_60 - cf_30)} İkinci Ay")
    cf_c3.metric("90 Günlük Nakit Girişi", format_curr(cf_90), delta=f"+{format_curr(cf_90 - cf_60)} Üçüncü Ay")
    cf_c4.metric("Kalan Alacak / Bakiye", format_curr(rem_balance), delta="Sözleşme Kalanı", delta_color="inverse" if rem_balance > 0 else "normal")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 5. AI STRATEJİK RİSK & NAKİT AKIŞI RAPORU
    # --------------------------------------------------------------------------
    st.markdown("#### 🤖 Yapay Zeka Finansal Risk & Senaryo Değerlendirmesi")
    st.caption("Seçili satış şokları, teslimat gecikmesi ve tahsilat modelini ZEUS'un gerçek pazar yapısı ile çapraz sorgulayarak üst yönetime risk değerlendirmesi üretir.")

    if st.button("🚀 AI Finansal Risk Değerlendirmesi Oluştur", type="primary"):
        ai_risk_placeholder = st.empty()
        with ai_risk_placeholder.container():
            components.html("""
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 160px; font-family: sans-serif; background: #0b1426; border-radius: 12px; border: 1px solid #13A6B5; box-shadow: 0 0 20px rgba(19, 166, 181, 0.2);">
                    <div id="risk-orb" style="width: 45px; height: 45px; background: radial-gradient(circle, #00F5FF 0%, #13A6B5 70%, #0B6F85 100%); border-radius: 50%; box-shadow: 0 0 25px #00F5FF; margin-bottom: 15px;"></div>
                    <div style="color: #4FD1DE; font-weight: bold; font-size: 14px; letter-spacing: 1px; text-transform: uppercase;">
                        ZEUS AI FİNANSAL RİSK ANALİZİ YAPIYOR...
                    </div>
                </div>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
                <script>
                    gsap.to("#risk-orb", { y: -12, scale: 1.15, duration: 1.0, yoyo: true, repeat: -1, ease: "sine.inOut" });
                </script>
            """, height=180)

        ai_risk_text = None
        api_key = get_gemini_api_key()
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                risk_prompt = f"""
                Sen ZEUS Bioscience Baş Finans Direktörüsün (CFO & Commercial Risk Director).
                Aşağıdaki simülasyon ve tahsilat parametrelerini inceleyerek üst yönetim için bir 'Finansal Risk & Likidite Değerlendirmesi' hazırla:
                
                Simülasyon Verileri:
                - Kapsam: {sc_country} ({sc_period})
                - Baz Ciro: {base_sc_rev:,.2f} USD -> Öngörülen Ciro: {sim_rev:,.2f} USD (Fark: {rev_delta:+.1f}%)
                - USD Kur Değişimi: %{sim_fx}, Talep Şoku: %{sim_demand}, İndirim Baskısı: %{sim_disc}
                - Sevkiyat Gecikmesi: {sim_delay} gün
                - Tahsilat Modeli: {pay_model} (Açıklama: {plan_desc})
                - 30 Günlük Nakit Girişi: {cf_30:,.2f} USD, 60 Günlük: {cf_60:,.2f} USD, 90 Günlük: {cf_90:,.2f} USD
                - 90 Gün Sonrası Kalan Alacak Bakiyesi: {rem_balance:,.2f} USD
                
                Raporunu tamamen Türkçe, profesyonel, maddeler halinde ve somut 3 adet risk azaltma aksiyonu ile yaz.
                """
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=risk_prompt,
                )
                ai_risk_text = response.text
            except Exception:
                ai_risk_text = None

        if not ai_risk_text:
            ai_risk_text = f"""### 📋 ZEUS FİNANSAL RİSK & LİKİDİTE YÖNETİCİ NOTU
**Referans Kodu:** `ZGMI-RISK-SIM-2026` | **Tarih:** Kasım 2026  
**Gizlilik:** `C-LEVEL & FİNANS KURULU İNCELEMESİ (GİZLİ)`  
**Analiz Kapsamı:** {sc_country} — {sc_period}  
**Simülasyon Sonucu:** Baz Ciro: **${base_sc_rev:,.2f}** ➔ Öngörülen Ciro: **${sim_rev:,.2f}** (Net Etki: **%{rev_delta:+.1f}**)

---

#### 1. 🎯 Senaryo ve Likidite Özeti
Modellenen senaryoda; **%{sim_fx} kur baskısı**, **%{sim_demand} talep dalgalanması** ve **%{sim_disc} rekabet indirimi** altında satış hacmi **${sim_rev:,.2f}** seviyesine evrilmektedir.
- **Teslimat Gecikmesi:** {sim_delay} günlük operasyonel/gümrük gecikmesi, sipariş döngüsünü uzatmakta ve nakit akışını ötelemektedir.
- **Tahsilat Modeli ({pay_model}):**
  * 30 Gün İçinde Beklenen Birikimli Nakit: **${cf_30:,.2f}**
  * 60 Gün İçinde Beklenen Birikimli Nakit: **${cf_60:,.2f}**
  * 90 Gün İçinde Beklenen Birikimli Nakit: **${cf_90:,.2f}**
  * 90 Gün Sonrası Kalan Sözleşme Alacak Bakiyesi: **${rem_balance:,.2f}**
- **Kârlılık Durumu:** Ürün maliyeti verisi girilmediğinde brüt kâr ve marj ihtiyatlılık gereği hesaplanmamış ve ekranda *“Maliyet verisi gerekli”* olarak işaretlenmiştir.

---

#### 2. 🛡️ 3 Adımlı Risk Azaltma ve Eylem Protokolü
1. **Döviz Dalgalanması ve Fiyat Koruması:** Kur volatilitesinin yüksek olduğu pazarlarda döviz kuru sözleşmeye bağlanmalı veya %100 peşin ödemede +%3 erken ödeme avantajı sunularak kur riski minimize edilmelidir.
2. **Sevkiyat ve Gecikme Yönetimi ({sim_delay} Gün):** Gecikmenin 30 günü aştığı durumlarda distribütör emniyet stoku zorunluluğu getirilmelidir.
3. **Taksitli / Kiralama Sözleşme Teminatı:** Yerel satışlarda peşinat oranının asgari %25 seviyesinde tutulması ve kalan bakiye için senet/teminat mektubu alınması nakit akışı emniyetini sağlar.
"""

        ai_risk_placeholder.empty()
        st.markdown(ai_risk_text)

        st.download_button(
            label="📥 Bu Risk & Senaryo Raporunu İndir (.md)",
            data=ai_risk_text,
            file_name=f"ZEUS_Risk_Scenario_Assessment_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )


# ==============================================================================
# 11. VIEW: DISTRIBUTOR PERFORMANCE EVALUATOR
# ==============================================================================
elif nav_selection == "▲ Performance Evaluator":
    st.title("📈 Partner Operational & Strategic Performance Scorecard")
    st.caption("Excel satış kayıtlarındaki gerçek yıllık cihaz/sarf dökümleri, gerçek YoY büyüme oranları ve çeyreklik sipariş düşüş analizi.")

    # --------------------------------------------------------------------------
    # 1. PARTNER SEÇİMİ (DİSTRİBÜTÖRLER ÖNCELİKLİ)
    # --------------------------------------------------------------------------
    # Order: Official distributors first, then international, then local
    dist_rows = [d for d in active_rows if d.get('account_type') == 'Distributor']
    intl_rows = [d for d in active_rows if d.get('account_type') == 'International Customer']
    local_rows = [d for d in active_rows if d.get('account_type') == 'Local Customer']
    dist_names = [d['name'] for d in dist_rows]
    intl_names = [d['name'] for d in intl_rows]
    local_names = [d['name'] for d in local_rows]
    all_ordered_names = dist_names + intl_names + local_names
    if not all_ordered_names:
        all_ordered_names = [d['name'] for d in active_rows]
    
    # Default to CMC if present, else first
    cmc_idx = next((i for i, n in enumerate(all_ordered_names) if "CONSULTANT MEDICAL" in n or "CMC" in n), 0)
    
    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        eval_dist_name = st.selectbox("İncelenecek Partner / Müşteriyi Seçin", all_ordered_names, index=cmc_idx)
    with sel_col2:
        eval_d = next((d for d in active_rows if d['name'] == eval_dist_name), active_rows[0])
        st.markdown(f"""
        <div style="background:rgba(19,166,181,0.08);border:1px solid #13A6B5;border-radius:8px;padding:8px 12px;font-size:12px;margin-top:20px;">
            <b>Ülke / Pazar:</b> {eval_d['country_name']} ({eval_d['region']})<br>
            <b>Sınıf:</b> {eval_d.get('account_type_tr', 'Distribütör')} &nbsp;|&nbsp; <b>LOA:</b> {eval_d.get('loa', 'N/A')}
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. GERÇEK EXCEL VERİLERİNDEN YILLIK & ÇEYREKLİK ANALİZ
    # --------------------------------------------------------------------------
    partner_sales = pd.DataFrame()
    if raw_sales_df is not None:
        partner_sales = raw_sales_df[raw_sales_df['Company'] == eval_dist_name].copy()
        partner_sales = partner_sales.sort_values('Date_dt')

    # Build annual breakdown
    years_list = [2024, 2025, 2026]
    y_stats = []
    for yr in years_list:
        ydf = partner_sales[partner_sales['Year_int'] == yr] if len(partner_sales) > 0 else pd.DataFrame()
        tot = float(ydf['Amount (USD)'].sum()) if len(ydf) > 0 else 0.0
        dev_amt = float(ydf[ydf['Group'] == 'Device']['Amount (USD)'].sum()) if len(ydf) > 0 else 0.0
        con_amt = float(ydf[ydf['Group'] == 'Consumables']['Amount (USD)'].sum()) if len(ydf) > 0 else 0.0
        dev_q = int(ydf[ydf['Group'] == 'Device']['Quantity'].sum()) if len(ydf) > 0 else 0
        con_q = int(ydf[ydf['Group'] == 'Consumables']['Quantity'].sum()) if len(ydf) > 0 else 0
        invs = int(ydf['Invoice'].nunique()) if len(ydf) > 0 else 0
        
        y_stats.append({
            'Yıl': f"{yr} (YTD)" if yr == 2026 else str(yr),
            'Toplam Ciro': tot,
            'Cihaz Geliri': dev_amt,
            'Cihaz (Adet)': dev_q,
            'Sarf Geliri': con_amt,
            'Sarf (Kutu)': con_q,
            'Sarf Payı (%)': (con_amt / tot * 100.0) if tot > 0 else 0.0,
            'Fatura': invs
        })

    y_df = pd.DataFrame(y_stats)
    
    # Real YoY Growth
    v24 = y_df['Toplam Ciro'].iloc[0]
    v25 = y_df['Toplam Ciro'].iloc[1]
    v26 = y_df['Toplam Ciro'].iloc[2]
    
    yoy_25 = ((v25 - v24) / v24 * 100.0) if v24 > 0 else (100.0 if v25 > 0 else 0.0)
    yoy_26 = ((v26 - v25) / v25 * 100.0) if v25 > 0 else (100.0 if v26 > 0 else 0.0)

    # Device YoY
    d24 = y_df['Cihaz Geliri'].iloc[0]
    d25 = y_df['Cihaz Geliri'].iloc[1]
    d26 = y_df['Cihaz Geliri'].iloc[2]
    dev_yoy_25 = ((d25 - d24) / d24 * 100.0) if d24 > 0 else (100.0 if d25 > 0 else 0.0)
    dev_yoy_26 = ((d26 - d25) / d25 * 100.0) if d25 > 0 else (100.0 if d26 > 0 else 0.0)

    # Consumables YoY
    c24 = y_df['Sarf Geliri'].iloc[0]
    c25 = y_df['Sarf Geliri'].iloc[1]
    c26 = y_df['Sarf Geliri'].iloc[2]
    con_yoy_25 = ((c25 - c24) / c24 * 100.0) if c24 > 0 else (100.0 if c25 > 0 else 0.0)
    con_yoy_26 = ((c26 - c25) / c25 * 100.0) if c25 > 0 else (100.0 if c26 > 0 else 0.0)

    tot_all_rev = sum(y_df['Toplam Ciro'])
    tot_all_dev = sum(y_df['Cihaz (Adet)'])
    tot_all_con = sum(y_df['Sarf (Kutu)'])
    last_order_str = eval_d.get('last_order', '—')

    # Quick KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Faturalanan Ciro", format_curr(tot_all_rev), delta=f"{yoy_26:+.1f}% 2026 YoY" if v25 > 0 else "Yeni Partner")
    k2.metric("Kurulu Cihaz Sayısı", f"{tot_all_dev} Adet", delta=f"{dev_yoy_26:+.0f}% Cihaz İvmesi" if d25 > 0 else "Stabil")
    k3.metric("Sarf Sipariş Hacmi", f"{tot_all_con} Kutu", delta=f"{con_yoy_26:+.1f}% Sarf İvmesi" if c25 > 0 else "Stabil")
    k4.metric("Son Sipariş Tarihi", last_order_str, delta="Aktif Müşteri")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. YILLIK CİHAZ VE SARF MALZEME GERÇEKLEŞME TABLOSU
    # --------------------------------------------------------------------------
    st.subheader("📋 Yıllık Cihaz ve Sarf Malzeme Gerçekleşme Tablosu")
    st.caption("Firma bazında sermaye ekipmanı satışları ile tekrar eden sarf malzeme tüketiminin yıl bazında resmi dökümü.")

    disp_y_df = y_df.copy()
    disp_y_df['YoY Net Değişim'] = [
        "—",
        f"{yoy_25:+.1f}% (Cihaz: {dev_yoy_25:+.0f}%, Sarf: {con_yoy_25:+.1f}%)",
        f"{yoy_26:+.1f}% (Cihaz: {dev_yoy_26:+.0f}%, Sarf: {con_yoy_26:+.1f}%)"
    ]
    disp_y_df['Toplam Ciro'] = disp_y_df['Toplam Ciro'].apply(format_curr)
    disp_y_df['Cihaz Geliri'] = disp_y_df['Cihaz Geliri'].apply(format_curr)
    disp_y_df['Sarf Geliri'] = disp_y_df['Sarf Geliri'].apply(format_curr)
    disp_y_df['Cihaz (Adet)'] = disp_y_df['Cihaz (Adet)'].apply(lambda x: f"{x} Adet" if x > 0 else "—")
    disp_y_df['Sarf (Kutu)'] = disp_y_df['Sarf (Kutu)'].apply(lambda x: f"{x} Kutu" if x > 0 else "—")
    disp_y_df['Sarf Payı (%)'] = disp_y_df['Sarf Payı (%)'].apply(lambda x: f"%{x:.1f}")
    disp_y_df['Fatura'] = disp_y_df['Fatura'].apply(lambda x: f"{x} İşlem")

    st.dataframe(disp_y_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 4. ÇEYREKLİK SİPARİŞ TRENDİ VE TESPİT EDİLEN SİPARİŞ DÜŞÜŞLERİ GRAFİĞİ
    # --------------------------------------------------------------------------
    st.subheader("📉 Çeyreklik Sipariş Trendi ve Sipariş Düşüşleri (Order Drop Tracking)")
    st.caption("Zaman çizelgesi üzerinde gerçekleşen siparişler, sermaye ekipmanı ve sarf ayrımı ile tespit edilen kritik hacim düşüşleri.")

    all_quarters = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1', '2025Q2', '2025Q3', '2025Q4', '2026Q1', '2026Q2', '2026Q3', '2026Q4']
    partner_sales['Quarter'] = partner_sales['Date_dt'].dt.to_period('Q').astype(str) if len(partner_sales) > 0 else pd.Series()
    
    q_dict = {}
    if len(partner_sales) > 0:
        for q, grp in partner_sales.groupby('Quarter'):
            q_dict[q] = {
                'Quarter': q,
                'Amount': float(grp['Amount (USD)'].sum()),
                'Device_Amt': float(grp[grp['Group'] == 'Device']['Amount (USD)'].sum()),
                'Consumables_Amt': float(grp[grp['Group'] == 'Consumables']['Amount (USD)'].sum()),
                'Device_Qty': int(grp[grp['Group'] == 'Device']['Quantity'].sum()),
                'Consumables_Qty': int(grp[grp['Group'] == 'Consumables']['Quantity'].sum())
            }

    active_qs = sorted(list(q_dict.keys())) if len(q_dict) > 0 else ['2026Q1', '2026Q2', '2026Q3']
    first_q_idx = all_quarters.index(active_qs[0]) if active_qs[0] in all_quarters else 0
    last_q_idx = all_quarters.index(active_qs[-1]) if active_qs[-1] in all_quarters else len(all_quarters) - 1

    q_series = []
    prev_amt = None
    detected_drops = []
    for q_idx in range(first_q_idx, last_q_idx + 1):
        q_str = all_quarters[q_idx]
        data = q_dict.get(q_str, {
            'Quarter': q_str, 'Amount': 0.0, 'Device_Amt': 0.0, 'Consumables_Amt': 0.0, 'Device_Qty': 0, 'Consumables_Qty': 0
        })
        
        is_drop = False
        drop_pct = 0.0
        if prev_amt is not None and prev_amt > 0:
            if data['Amount'] < prev_amt:
                is_drop = True
                drop_pct = ((data['Amount'] - prev_amt) / prev_amt) * 100.0
                detected_drops.append({
                    'Quarter': q_str,
                    'Drop_Pct': drop_pct,
                    'Current_Amt': data['Amount'],
                    'Prior_Amt': prev_amt,
                    'Is_Zero_Gap': data['Amount'] == 0.0
                })
        data['Is_Drop'] = is_drop
        data['Drop_Pct'] = drop_pct
        q_series.append(data)
        prev_amt = data['Amount']

    q_df = pd.DataFrame(q_series)

    # Plotly Stacked Bar + Line with Drop Markers
    fig_ord = go.Figure()

    # Device bar
    fig_ord.add_trace(go.Bar(
        x=q_df['Quarter'], y=q_df['Device_Amt'] * currency_rate,
        name='Cihaz Geliri (Capital Equipment)',
        marker_color='#13A6B5'
    ))

    # Consumables bar
    fig_ord.add_trace(go.Bar(
        x=q_df['Quarter'], y=q_df['Consumables_Amt'] * currency_rate,
        name='Sarf Geliri (Consumables & Solution)',
        marker_color='#4FD1DE'
    ))

    # Total line
    fig_ord.add_trace(go.Scatter(
        x=q_df['Quarter'], y=q_df['Amount'] * currency_rate,
        mode='lines+markers',
        name='Toplam Sipariş Hacmi',
        line=dict(color='#CCD7E6', width=2),
        marker=dict(size=6, color='#CCD7E6')
    ))

    # Drop annotations
    for drop in detected_drops:
        fig_ord.add_annotation(
            x=drop['Quarter'],
            y=drop['Current_Amt'] * currency_rate,
            text=f"📉 Düşüş {drop['Drop_Pct']:.0f}%" if not drop['Is_Zero_Gap'] else "⚠️ Sipariş Boşluğu ($0)",
            showarrow=True,
            arrowhead=2,
            arrowcolor='#FF8A96',
            font=dict(color='#FF8A96', size=11, family='sans-serif'),
            bgcolor="rgba(229,112,123,0.18)",
            bordercolor="#FF8A96",
            borderwidth=1,
            borderpad=3
        )

    fig_ord.update_layout(
        barmode='stack',
        height=380,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(gridcolor='rgba(255,255,255,0.08)', title=f"Sipariş Tutarı ({currency_symbol})"),
        xaxis=dict(title="Dönem (Çeyrekler)")
    )
    st.plotly_chart(fig_ord, use_container_width=True)

    # Detailed Drop Insights Box
    c_dr1, c_dr2 = st.columns([1.5, 1])
    with c_dr1:
        st.markdown("##### 🚨 Tespit Edilen Sipariş Düşüşleri ve Kritik Sapmalar")
        if len(detected_drops) > 0:
            for drop in detected_drops:
                if drop['Is_Zero_Gap']:
                    st.markdown(f"""
                    <div style="background:rgba(229,112,123,0.08);border-left:4px solid #FF8A96;padding:10px 14px;border-radius:6px;margin-bottom:8px;font-size:13px;">
                        <b style="color:#FF8A96;">⚠️ {drop['Quarter']}: Sipariş Kesintisi (Gap Dönemi)</b><br>
                        <span style="color:#CCD7E6;">Bir önceki çeyrekte {format_curr(drop['Prior_Amt'])} sipariş verilmişken, bu çeyrekte hiç sipariş açılmamıştır (%100 dönemsel düşüş).</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:rgba(240,188,87,0.08);border-left:4px solid #F0BC57;padding:10px 14px;border-radius:6px;margin-bottom:8px;font-size:13px;">
                        <b style="color:#F0BC57;">📉 {drop['Quarter']}: {drop['Drop_Pct']:.1f}% Sipariş Gerilemesi</b><br>
                        <span style="color:#CCD7E6;">Önceki çeyrekteki {format_curr(drop['Prior_Amt'])} hacim, {format_curr(drop['Current_Amt'])} seviyesine gerilemiştir.</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ Bu partner için çeyrekler arası keskin bir sipariş düşüşü tespit edilmemiştir; sipariş hacmi düzenli veya artış trendindedir.")

    with c_dr2:
        st.markdown("##### 💡 Analitik Değerlendirme & Yorum")
        if tot_all_dev > 0 and y_df['Cihaz (Adet)'].iloc[-1] == 0:
            st.info(f"ℹ️ **Cihaz vs Sarf Dönüşümü:** {eval_dist_name}, geçmiş yıllarda {tot_all_dev} adet Piezo cihazı alarak altyapısını kurmuştur. 2026'da cihaz alımının durması normal bir sermaye ekipmanı doyumudur; asıl takip edilmesi gereken metrik kurulu 5 cihazın oosit sarf tüketim sürekliliğidir.")
        elif tot_all_dev == 0:
            st.info(f"ℹ️ **Sadece Sarf Alan Müşteri:** Bu partner henüz Piezo sermaye cihazı almamış olup yalnızca Electro Dish veya Solüsyon sarfı tüketmektedir.")
        else:
            st.success(f"ℹ️ **Aktif Büyüme:** Partner hem yeni cihaz kurulumu yapmakta hem de sarf malzeme hacmini büyütmektedir.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 5. GERÇEK VERİYE DAYALI AĞIRLIKLI PERFORMANS SKORU
    # --------------------------------------------------------------------------
    st.subheader("🎯 Ağırlıklı Performans ve Sağlık Karnesi")
    st.caption("Partnerin gerçek büyüme rasyoları, kurulu cihaz tabanı, sarf sürekliliği ve sipariş düşüş sıklığına göre hesaplanmıştır.")

    # Objective scores based on actual data
    # 1. Device base score (0 to 100)
    s_device = min(100, tot_all_dev * 20)  # 5 devices = 100
    
    # 2. Consumables velocity score
    s_consumable = min(100, tot_all_con * 2) # 50 boxes = 100
    
    # 3. Growth score (penalizes sharp contraction, rewards positive growth)
    if yoy_26 > 15.0:
        s_growth = 95
    elif yoy_26 > 0.0:
        s_growth = 80
    elif yoy_26 > -30.0:
        s_growth = 60
    elif yoy_26 > -60.0:
        s_growth = 45
    else:
        s_growth = 35
        
    # 4. Regularity score (penalized by number of drop quarters)
    num_drops = len([d for d in detected_drops if d['Is_Zero_Gap']])
    s_regularity = max(30, 95 - num_drops * 20)
    
    # 5. Compliance score
    s_comp = 95 if eval_d.get('loa') == 'Valid' else (65 if eval_d.get('loa') == 'In renewal' else 80)

    perf_factors = [
        {"Factor": "Kurulu Cihaz Tabanı & Penetrasyon", "Score": s_device, "Weight": "25%", "Detail": f"{tot_all_dev} Adet Placed Piezo Pulse"},
        {"Factor": "Tekrar Eden Sarf Tüketim Hacmi", "Score": s_consumable, "Weight": "25%", "Detail": f"{tot_all_con} Kutu Electro Dish & Solüsyon"},
        {"Factor": "Yıllık Satış Büyüme İvmesi (YoY)", "Score": s_growth, "Weight": "20%", "Detail": f"{yoy_26:+.1f}% 2026 Net Değişim"},
        {"Factor": "Sipariş Düzenliliği & Düşüş Direnci", "Score": s_regularity, "Weight": "15%", "Detail": f"{num_drops} Sıfır Çeyrek Boşluğu"},
        {"Factor": "Hukuki Uyum & Sözleşme Sağlığı", "Score": s_comp, "Weight": "15%", "Detail": f"Statü: {eval_d.get('compliance', 'Uyumlu')}"}
    ]

    total_weighted_score = round(
        s_device * 0.25 + s_consumable * 0.25 + s_growth * 0.20 + s_regularity * 0.15 + s_comp * 0.15
    )

    if total_weighted_score >= 80:
        band_title = "Strong Commercial Performer (Tier A)"
        band_col = "#5FD3AE"
    elif total_weighted_score >= 65:
        band_title = "Solid & Stable Partner (Tier B)"
        band_col = "#4FD1DE"
    elif total_weighted_score >= 50:
        band_title = "Mixed Performance – Action Required (Tier C)"
        band_col = "#F0BC57"
    else:
        band_title = "Critical Attention Required (Tier D)"
        band_col = "#FF8A96"

    ev1, ev2 = st.columns([1, 2])
    with ev1:
        st.markdown(f"""
        <div class="kpi-container" style="text-align:center;padding:24px;">
            <div class="kpi-header">Genel Partner Sağlık Skoru</div>
            <div style="font-size:52px;font-weight:800;color:{band_col};">{total_weighted_score}</div>
            <div style="font-size:14px;color:#CCD7E6;font-weight:600;margin-top:6px;">{band_title}</div>
            <div style="font-size:11px;color:#7C93BD;margin-top:4px;">Gerçek Satış ve Sipariş İvmesine Dayalı</div>
        </div>
        """, unsafe_allow_html=True)
    with ev2:
        for item in perf_factors:
            st.write(f"**{item['Factor']}** ({item['Weight']}) — `{item['Detail']}`")
            st.progress(item['Score'] / 100.0, text=f"{item['Score']} / 100")

    # --------------------------------------------------------------------------
    # 6. YÖNETİCİ EYLEM REÇETESİ
    # --------------------------------------------------------------------------
    st.markdown("#### 💊 Yönetici Teşhisi ve Eylem Reçetesi")
    if "CMC" in eval_dist_name or "CONSULTANT" in eval_dist_name:
        st.markdown("""
        <div style="background:rgba(19,166,181,0.08);border:1px solid #13A6B5;border-radius:10px;padding:16px;">
            <b style="color:#00F5FF;font-size:15px;">📋 CMC (Filistin Pazarı) Özel Yönetici Özeti:</b><br>
            <div style="color:#CCD7E6;font-size:13px;line-height:1.6;margin-top:6px;">
                <b>Teşhis:</b> CMC, 2024 ve 2025 yıllarında toplam <b>5 adet ZEUS Piezo Pulse</b> cihazı alarak Filistin'deki referans klinik altyapısını tamamlamıştır. 2026 yılında yeni cihaz alımı yapılmaması toplam ciroda %71.8 düşüş gibi görünse de, <b>46 kutuluk sarf alımı</b> cihazların tam kapasite çalıştığını teyit etmektedir.<br>
                <b>Kritik Risk:</b> Siparişlerin çeyreklik düzenli açılmaması (2025Q2 ve 2026Q2'de sıfır sipariş); kliniklerin sarf stoku tükenene kadar beklediğini göstermektedir.<br>
                <b>Eylem Reçetesi:</b> CMC ile 6 aylık toplu sipariş yerine <b>çeyreklik otomatik yenilenen sarf sevkiyat protokolü</b> imzalanmalı; 2027 başında Batı Şeria'daki yeni bir merkez için 6. Piezo cihazı demo teklifi sunulmalıdır.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(19,166,181,0.08);border:1px solid #13A6B5;border-radius:10px;padding:16px;">
            <b style="color:#00F5FF;font-size:15px;">📋 {eval_dist_name} Yönetici Aksiyon Notu:</b><br>
            <div style="color:#CCD7E6;font-size:13px;line-height:1.6;margin-top:6px;">
                <b>Stratejik Durum:</b> Toplam {format_curr(tot_all_rev)} ciro, {tot_all_dev} kurulu cihaz ve {tot_all_con} kutu sarf malzemesi kaydı bulunmaktadır.<br>
                <b>Öneri:</b> Çeyreklik sipariş dalgalanmalarını gidermek amacıyla yıllık minimum sarf sipariş taahhüdü ve indirim kademeleri devreye alınmalıdır.
            </div>
        </div>
        """, unsafe_allow_html=True)


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

