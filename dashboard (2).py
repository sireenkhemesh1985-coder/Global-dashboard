import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# ##############################################################################
# 1. Sayfa Başlığı ve Ayarları
# ##############################################################################
st.set_page_config(page_title="Şirket Performans Panosu", layout="wide")
st.title("📈 Satış ve Performans Dashboard")

# ##############################################################################
# 2. Örnek Veri Seti Oluşturma (Normalde buraya kendi dosyanızı eklersiniz)
# ##############################################################################
data = {
    'Ay': ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran'],
    'Satış Tutarı':,
    'Kategori': ['Elektronik', 'Elektronik', 'Giyim', 'Giyim', 'Ev-Yaşam', 'Ev-Yaşam']
}
df = pd.DataFrame(data)

# ##############################################################################
# 3. Sol Panel (Sidebar) - Filtreleme Alanı
# ##############################################################################
st.sidebar.header("Filtreleme Seçenekleri")
secilen_kategori = st.sidebar.multiselect(
    "Kategori Seçin:",
    options=df['Kategori'].unique(),
    default=df['Kategori'].unique()
)

# Veriyi filtrele
filtreli_df = df[df['Kategori'].isin(secilen_kategori)]

# ##############################################################################
# 4. Özet Kartları (Metrics)
# ##############################################################################
sk1, sk2 = st.columns(2)
sk1.metric(label="Toplam Satış", value=f"{filtreli_df['Satış Tutarı'].sum()} TL", delta="12%")
sk2.metric(label="Aktif Kategori Sayısı", value=len(secilen_kategori))

# ##############################################################################
# 5. Grafik Alanı
# ##############################################################################
st.subheader("Aylık Satış Trendi")
fig = px.line(filtreli_df, x='Ay', y='Satış Tutarı', color='Kategori', title="Aylara Göre Satış Gelişimi")
st.plotly_chart(fig, use_container_width=True)

# ##############################################################################
# 6. Tablo Görünümü
# ##############################################################################
st.subheader("Ham Veri Tablosu")
st.dataframe(filtreli_df)

# ##############################################################################
# 🤖 7. YAPAY ZEKA (AI) ANALİZ ALANI (YENİ EKLENEN KISIM)
# ##############################################################################
st.markdown("---")
st.subheader("🔮 Yapay Zeka Stratejik Analiz İstasyonu")

# Streamlit Cloud üzerindeki "Secrets" kasasından anahtarı güvenle okur
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    # Güncel ve kararlı 1.5-flash modelini tanımlıyoruz
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    
    if st.button("🚀 Raporu Yapay Zekaya Yorumlat (AI Analizi)"):
        with st.spinner("Yapay zeka verilerinizi inceliyor, strateji hazırlıyor..."):
            
            # Verideki özet bilgileri AI'ın anlaması için metne çeviriyoruz
            toplam_ciro = filtreli_df['Satış Tutarı'].sum()
            veri_metni = filtreli_df.to_string(index=False)
            
            # İstediğiniz 3 analizi yaptıracak akıllı prompt
            prompt = f"""
            Sen profesyonel bir veri analisti ve satış danışmanısın. 
            Sana şirketimizin güncel satış verilerini veriyorum:
            
            [Filtrelenmiş Satış Tablosu]:
            {veri_metni}
            
            Toplam Ciro: {toplam_ciro} TL
            
            Senden bu verilere bakarak profesyonel bir rapor hazırlamanı istiyorum:
            1. 📉 DÜŞEN ÜRÜNLERİN VEYA DURAKLAYAN DÖNEMLERİN TESPİTİ: Verilerdeki zayıf ayları veya kategorileri alarm veren noktalarıyla listele.
            2. 🗺️ BÖLGESEL/KATEGORİSEL ZAYIFLIKLAR: Hangi kategoriler geride kalmış, canlandırmak için ne yapılmalı?
            3. 🔮 GELECEK DÖNEM TAHMİNİ VE STRATEJİSİ: Trend çizgisine göre önümüzdeki süreçte bizi ne bekliyor? Satışları artırmak için 3 somut öneri sun.
            
            Lütfen yanıtını tamamen Türkçe, maddeler halinde ve yönetici özeti tonda yaz.
            """
            
            response = model.generate_content(prompt)
            st.markdown("### 📋 Yapay Zeka Derin Analiz Raporu")
            st.markdown(response.text)

except Exception as e:
    # Eğer henüz canlıya almadıysanız ve localde test ediyorsanız bu uyarı görünür
    st.sidebar.warning("🤖 AI Aktif Değil: API anahtarınızı Streamlit Cloud Secrets paneline eklediğinizde bu alan otomatik çalışacaktır.")
