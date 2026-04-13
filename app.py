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

# --- 1. SAYFA YAPILANDIRMASI ---
st.set_page_config(layout="wide", page_title="Hemithea Network Analysis", page_icon="🌐")

@st.cache_data
def load_data(url_or_file, is_url=True):
    try:
        if is_url:
            response = requests.get(url_or_file)
            return pd.read_csv(StringIO(response.text))
        return pd.read_csv(url_or_file)
    except:
        return None

# --- 2. SIDEBAR: VERİ YÜKLEME VE SEÇİM ---
st.sidebar.title("🛠️ Veri Yönetimi")
data_mode = st.sidebar.radio("Analiz Modu:", ["Efendi (Hazır Demo)", "Kendi Verini Yükle"])

# Efendi GitHub Raw URL'i (Burayı kendi GitHub linkinle güncellemelisin)
EFENDI_URL = "https://github.com/seydanur-hemithea/Hemithea_network_analysis/blob/main/efendi_veri.csv"

if data_mode == "Efendi (Hazır Demo)":
    data = load_data(EFENDI_URL, is_url=True)
    st.sidebar.success("✅ Efendi verisi başarıyla çekildi.")
else:
    uploaded_file = st.sidebar.file_uploader("CSV dosyanızı yükleyin", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file, is_url=False)
    else:
        data = None

# --- 3. KOLON SEÇİMİ (Dinamik ve Esnek) ---
if data is not None:
    st.sidebar.divider()
    st.sidebar.subheader("🔗 İlişki Tanımları")
    all_cols = data.columns.tolist()
    
    # Kullanıcı kolonları kendisi seçer
    source_col = st.sidebar.selectbox("Kaynak (Source):", all_cols, index=0)
    target_col = st.sidebar.selectbox("Hedef (Target):", all_cols, index=min(1, len(all_cols)-1))
    
    if source_col == target_col:
        st.sidebar.error("Kaynak ve Hedef aynı olamaz!")
        st.stop()

# --- 4. ANA PANEL: ANALİZ VE GÖRSELLEŞTİRME ---
st.title("🌐 Hemithea: Advanced Network Analytics")

if data is not None:
    # Graf Teorisi Hesaplamaları
    clean_df = data[[source_col, target_col]].dropna()
    G = nx.from_pandas_edgelist(clean_df, source=source_col, target=target_col)

    # KRİTİK METRİKLER
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    clo_cent = nx.closeness_centrality(G)

    # AI KÜMELEME (K-Means)
    nodes = list(G.nodes())
    features = np.array([[deg_cent[n], bet_cent[n]] for n in nodes])
    kmeans = KMeans(n_clusters=min(4, len(nodes)), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
    cluster_map = dict(zip(nodes, clusters))

    # KPI Özet Ekranı
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Düğüm", len(G.nodes))
    c2.metric("Toplam Bağlantı", len(G.edges))
    c3.metric("Ağ Yoğunluğu", f"{nx.density(G):.4f}")
    c4.metric("En Köprü Aktör", max(bet_cent, key=bet_cent.get))

    st.divider()

    # Sekmeli Görünüm (Tablar)
    tab_graph, tab_stats = st.tabs(["🕸️ İnteraktif Ağ Haritası", "📊 Analitik Detaylar"])

    with tab_graph:
        st.subheader("İlişki Dinamiği ve Topluluk Tespiti")
        size_metric = st.selectbox("Düğüm Boyutlandırma Ölçütü:", ["Degree Centrality", "Betweenness Centrality"])
        
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        # Renk ve Boyut Özelleştirme
        palette = ["#FF4B4B", "#1C83E1", "#00C781", "#FFBD45"]
        for node in net.nodes:
            n_id = node["id"]
            val = deg_cent[n_id] if size_metric == "Degree Centrality" else bet_cent[n_id]
            node["size"] = 15 + (val * 100)
            node["color"] = palette[cluster_map[n_id] % len(palette)]
            node["title"] = f"Degree: {deg_cent[n_id]:.2f}\nBetweenness: {bet_cent[n_id]:.2f}"

        net.toggle_physics(True)
        components.html(net.generate_html(), height=650)

    with tab_stats:
        st.subheader("Aktör Bazlı İstatistik Tablosu")
        res_df = pd.DataFrame({
            'Birim': nodes,
            'Degree Centrality': [deg_cent[n] for n in nodes],
            'Betweenness': [bet_cent[n] for n in nodes],
            'Closeness': [clo_cent[n] for n in nodes],
            'AI_Cluster': [cluster_map[n] for n in nodes]
        }).sort_values(by='Betweenness', ascending=False)
        
        st.dataframe(res_df.style.background_gradient(cmap='YlGnBu'), use_container_width=True)

else:
    st.info("👈 Analize başlamak için sol taraftaki panelden 'Efendi' demosunu seçebilir veya kendi verinizi yükleyebilirsiniz.")
