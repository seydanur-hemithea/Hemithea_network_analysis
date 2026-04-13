import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import requests
from io import StringIO

# --- 1. SAYFA AYARLARI VE STİL ---
st.set_page_config(layout="wide", page_title="Hemithea Network Analysis | Portfolio", page_icon="🌐")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_stdio=True)

@st.cache_data
def load_data(url_or_file, is_url=True):
    try:
        if is_url:
            response = requests.get(url_or_file)
            return pd.read_csv(StringIO(response.text))
        return pd.read_csv(url_or_file)
    except:
        return None

# --- 2. SIDEBAR: KONTROL MERKEZİ ---
st.sidebar.title("🛠️ Analiz Ayarları")
data_mode = st.sidebar.radio("Veri Kaynağı Seçin:", ["Efendi (Demo)", "Kendi Verini Yükle"])

# Efendi GitHub Raw URL'ini buraya kendi linkinle güncelle
EFENDI_URL = "https://raw.githubusercontent.com/seydanur/efendi/main/data.csv"

if data_mode == "Efendi (Demo)":
    data = load_data(EFENDI_URL, is_url=True)
    st.sidebar.success("✅ Efendi veri seti yüklendi.")
else:
    uploaded_file = st.sidebar.file_uploader("Analiz için CSV dosyası seçin", type=["csv"])
    data = load_data(uploaded_file, is_url=False) if uploaded_file else None

# --- 3. ANA MOTOR ---
st.title("🌐 Hemithea: Advanced Graph Analytics & AI")
st.info("Bu platform, karmaşık veri setlerindeki gizli ilişkileri matematiksel metrikler ve yapay zeka ile gün yüzüne çıkarır.")

if data is not None:
    # --- KOLON SEÇİCİ ---
    st.sidebar.subheader("🔗 İlişki Parametreleri")
    all_cols = data.columns.tolist()
    source_col = st.sidebar.selectbox("Kaynak (Source):", all_cols, index=0)
    target_col = st.sidebar.selectbox("Hedef (Target):", all_cols, index=min(1, len(all_cols)-1))

    # Temizlik ve Graph Oluşturma
    clean_df = data[[source_col, target_col]].dropna()
    G = nx.from_pandas_edgelist(clean_df, source=source_col, target=target_col)

    # --- AKADEMİK METRİKLER (İSTEDİĞİN KISIM) ---
    with st.spinner("Metrikler hesaplanıyor..."):
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G)
        clo_cent = nx.closeness_centrality(G)
        try:
            eig_cent = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eig_cent = deg_cent # Yakınsama hatası olursa dereceyi kullan

    # --- AI KÜMELEME ---
    nodes = list(G.nodes())
    features = np.array([[deg_cent[n], bet_cent[n], clo_cent[n]] for n in nodes])
    kmeans = KMeans(n_clusters=min(4, len(nodes)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
    cluster_map = dict(zip(nodes, clusters))

    # --- KPI DASHBOARD ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Düğüm", len(G.nodes))
    c2.metric("Toplam Bağlantı", len(G.edges))
    c3.metric("Ağ Yoğunluğu", f"{nx.density(G):.4f}")
    c4.metric("Küme Sayısı", len(set(clusters)))

    st.divider()

    # --- GÖRSELLEŞTİRME VE TABLO ---
    tab1, tab2 = st.tabs(["🕸️ İnteraktif Ağ Haritası", "📈 Analitik Metrik Tablosu"])

    with tab1:
        st.subheader("Dinamik İlişki Ağı")
        size_option = st.selectbox("Düğüm Boyutu Neye Göre Olsun?", ["Degree Centrality", "Betweenness Centrality", "Eigenvector"])
        
        net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        # Renk Paleti (Şık ve Modern)
        palette = ["#FF4B4B", "#1C83E1", "#00C781", "#FFBD45", "#7D3C98"]
        
        for node in net.nodes:
            n_id = node["id"]
            # Boyut Belirleme
            if size_option == "Degree Centrality": s = deg_cent[n_id]
            elif size_option == "Betweenness Centrality": s = bet_cent[n_id]
            else: s = eig_cent[n_id]
            
            node["size"] = 15 + (s * 100)
            node["color"] = palette[cluster_map[n_id] % len(palette)]
            node["title"] = f"Degree: {deg_cent[n_id]:.3f}\nBetweenness: {bet_cent[n_id]:.3f}\nCluster: {cluster_map[n_id]}"
        
        net.toggle_physics(True)
        components.html(net.generate_html(), height=700)

    with tab2:
        st.subheader("Detaylı Ağ İstatistikleri")
        results = pd.DataFrame({
            'Aktör/Birim': nodes,
            'Degree': [deg_cent[n] for n in nodes],
            'Betweenness': [bet_cent[n] for n in nodes],
            'Closeness': [clo_cent[n] for n in nodes],
            'Eigenvector': [eig_cent[n] for n in nodes],
            'AI_Cluster': [cluster_map[n] for n in nodes]
        }).sort_values(by='Betweenness', ascending=False)
        
        st.dataframe(results.style.background_gradient(cmap='Blues'), use_container_width=True)

else:
    st.warning("👈 Analize başlamak için sol panelden bir veri kaynağı seçin.")
