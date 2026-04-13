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

# --- 1. SAYFA AYARLARI ---
st.set_page_config(layout="wide", page_title="Hemithea Network | Efendi Analysis")

# --- 2. VERİ YÜKLERKEN "EFENDİ"Yİ VARSAYILAN YAPMA ---
# Efendi verinin GitHub Raw URL'ini buraya koy (Örnek URL ekledim)
EFENDI_RAW_URL = "https://raw.githubusercontent.com/seydanur/efendi-project/main/data/efendi_nodes_edges.csv"

@st.cache_data
def load_data(url_or_file, is_url=True):
    try:
        if is_url:
            response = requests.get(url_or_file)
            if response.status_code == 200:
                return pd.read_csv(StringIO(response.text))
        else:
            return pd.read_csv(url_or_file)
    except:
        return None

# --- SIDEBAR: KONTROL PANELİ ---
st.sidebar.header("📂 Veri Kontrol Merkezi")
data_source = st.sidebar.radio("Analiz Edilecek Veriyi Seçin:", 
                                ["Efendi Projesi (Örnek Veri)", "Kendi Verimi Yükle"])

if data_source == "Efendi Projesi (Örnek Veri)":
    st.sidebar.info("Şu an Şeyda Nur Aydın'ın 'Efendi' projesine ait tarihsel ağ verileri analiz ediliyor.")
    data = load_data(EFENDI_RAW_URL, is_url=True)
else:
    uploaded_file = st.sidebar.file_uploader("CSV dosyanızı sürükleyin", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file, is_url=False)
    else:
        st.sidebar.warning("Lütfen bir CSV dosyası yükleyin.")
        data = None

# --- 3. ANA PANEL VE AI MOTORU ---
st.title("🌐 Hemithea Network: Çok Boyutlu Ağ Analizi")

if data is not None:
    # Sütun isimlerini standartlaştır (Source ve Target bekliyoruz)
    # Eğer Efendi verisinde sütunlar farklıysa burayı 'Kaynak', 'Hedef' yapabilirsin
    src_col = data.columns[0]
    tgt_col = data.columns[1]
    
    G = nx.from_pandas_edgelist(data, source=src_col, target=tgt_col)
    
    # AI Metrikleri
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    # K-Means Kümeleme
    nodes_list = list(G.nodes())
    if len(nodes_list) >= 2:
        features = np.array([[degree_cent[n], betweenness[n]] for n in nodes_list])
        n_clusters = min(4, len(nodes_list))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
        cluster_map = dict(zip(nodes_list, clusters))
    else:
        cluster_map = {n: 0 for n in nodes_list}

    # KPI Kartları
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Toplam Düğüm", len(G.nodes))
    col2.metric("Toplam Bağlantı", len(G.edges))
    col3.metric("En Merkezi Aktör", max(degree_cent, key=degree_cent.get))
    col4.metric("Ağ Yoğunluğu", f"{nx.density(G):.2f}")

    # Grafik ve Analiz
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("İnteraktif İlişki Haritası")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        cluster_colors = {0: "#FF4B4B", 1: "#1C83E1", 2: "#00C781", 3: "#FFBD45"}
        for node in net.nodes:
            n_id = node["id"]
            node["size"] = 15 + (degree_cent[n_id] * 70)
            node["color"] = cluster_colors.get(cluster_map[n_id], "#999999")
            node["title"] = f"Etki Skoru: {degree_cent[n_id]:.2f}\nKüme: {cluster_map[n_id]}"
        
        net.toggle_physics(True)
        html_data = net.generate_html()
        components.html(html_data, height=650)

    with right_col:
        st.subheader("Birim Analiz Detayları")
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Popülerlik': [f"{v:.2f}" for v in degree_cent.values()],
            'Küme': [cluster_map[n] for n in nodes_list]
        }).sort_values(by='Popülerlik', ascending=False)
        st.dataframe(metrics_df, use_container_width=True, height=550)
