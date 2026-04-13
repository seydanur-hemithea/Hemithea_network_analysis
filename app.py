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
st.set_page_config(layout="wide", page_title="Hemithea Network Analysis")

# --- 2. YÜKLEME FONKSİYONU ---
@st.cache_data
def load_data(url_or_file, is_url=True):
    try:
        if is_url:
            response = requests.get(url_or_file)
            return pd.read_csv(StringIO(response.text))
        return pd.read_csv(url_or_file)
    except:
        return None

# --- 3. SIDEBAR (DOSYA YÜKLEME ALANI) ---
st.sidebar.header("📊 Veri Yönetimi")

# Kullanıcıya seçenek sunalım ama varsayılanı 'Efendi' yapalım
data_choice = st.sidebar.selectbox(
    "Analiz Modu Seçin:",
    ["Efendi Projesi (Hazır Demo)", "Kendi CSV Dosyamı Analiz Et"]
)

# Efendi URL (Senin GitHub linkin buraya gelecek)
EFENDI_URL = "https://raw.githubusercontent.com/seydanur/efendi/main/data.csv"

if data_choice == "Efendi Projesi (Hazır Demo)":
    st.sidebar.info("Efendi verisi GitHub üzerinden çekiliyor...")
    data = load_data(EFENDI_URL, is_url=True)
else:
    # BU KISIM ARTIK GÖRÜNÜR OLACAK
    uploaded_file = st.sidebar.file_uploader("Kendi CSV dosyanızı buraya sürükleyin", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file, is_url=False)
    else:
        st.sidebar.warning("Lütfen bir dosya yükleyin.")
        data = None

# --- 4. ANALİZ VE GÖRSELLEŞTİRME ---
st.title("🌐 Hemithea Network Analysis")

if data is not None:
    # Verinin ilk iki sütununu Kaynak ve Hedef kabul et
    source_col = data.columns[0]
    target_col = data.columns[1]
    
    # NetworkX Graff Oluşturma
    G = nx.from_pandas_edgelist(data, source=source_col, target=target_col)
    
    # AI - Kümeleme (Clustering)
    degree_cent = nx.degree_centrality(G)
    nodes_list = list(G.nodes())
    
    if len(nodes_list) >= 2:
        # Metrikleri ölçeklendir ve K-Means uygula
        features = np.array([[degree_cent[n]] for n in nodes_list])
        kmeans = KMeans(n_clusters=min(3, len(nodes_list)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
        cluster_map = dict(zip(nodes_list, clusters))
    else:
        cluster_map = {n: 0 for n in nodes_list}

    # Üst Bilgi Kartları
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Düğüm", len(G.nodes))
    c2.metric("Toplam İlişki", len(G.edges))
    c3.metric("En Önemli Aktör", max(degree_cent, key=degree_cent.get))

    # Görselleştirme Paneli
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
    net.from_nx(G)
    
    # Düğüm Özelleştirme
    colors = {0: "#74ebd5", 1: "#acb6e5", 2: "#ff9a9e"}
    for node in net.nodes:
        n_id = node["id"]
        node["size"] = 20 + (degree_cent[n_id] * 50)
        node["color"] = colors.get(cluster_map[n_id], "#eeeeee")
        node["title"] = f"Skor: {degree_cent[n_id]:.2f}"

    html_data = net.generate_html()
    components.html(html_data, height=650)
    
    # Veri Tablosu
    st.subheader("📋 Veri Detayları")
    st.dataframe(data, use_container_width=True)

else:
    st.info("👈 Lütfen soldaki panelden 'Efendi' verisini seçin veya kendi verinizi yükleyerek başlayın.")
