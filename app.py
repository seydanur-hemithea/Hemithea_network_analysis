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
st.set_page_config(layout="wide", page_title="Hemithea Network Analysis Pro")

@st.cache_data
def load_data(url_or_file, is_url=True):
    try:
        if is_url:
            response = requests.get(url_or_file)
            return pd.read_csv(StringIO(response.text))
        return pd.read_csv(url_or_file)
    except:
        return None

# --- 2. SIDEBAR ---
st.sidebar.header("📂 Veri & Parametreler")
data_choice = st.sidebar.selectbox("Veri Kaynağı:", ["Efendi Projesi (Demo)", "Kendi Verimi Yükle"])

EFENDI_URL = "https://raw.githubusercontent.com/seydanur/efendi/main/data.csv"

if data_choice == "Efendi Projesi (Demo)":
    data = load_data(EFENDI_URL, is_url=True)
else:
    uploaded_file = st.sidebar.file_uploader("CSV Yükle", type=["csv"])
    data = load_data(uploaded_file, is_url=False) if uploaded_file else None

# --- 3. ANA ANALİZ MOTORU ---
st.title("🌐 Hemithea Network: Gelişmiş Analitik Dashboard")

if data is not None:
    # Kolon Seçimi
    all_columns = data.columns.tolist()
    col_source = st.sidebar.selectbox("Kaynak (Source):", all_columns, index=0)
    col_target = st.sidebar.selectbox("Hedef (Target):", all_columns, index=min(1, len(all_columns)-1))

    # --- HESAPLAMALAR ---
    working_df = data[[col_source, col_target]].dropna()
    G = nx.from_pandas_edgelist(working_df, source=col_source, target=col_target)

    # 1. Degree Centrality (Popülerlik/Bağlantı Sayısı)
    degree_cent = nx.degree_centrality(G)
    # 2. Betweenness Centrality (Köprü Olma/Bilgi Akışı Kontrolü)
    betweenness_cent = nx.betweenness_centrality(G)
    # 3. Closeness Centrality (Erişilebilirlik/Hız)
    closeness_cent = nx.closeness_centrality(G)

    # AI Kümeleme (K-Means)
    nodes_list = list(G.nodes())
    if len(nodes_list) >= 3:
        # Üç metriği de kullanarak kümeleme yapalım
        features = np.array([[degree_cent[n], betweenness_cent[n], closeness_cent[n]] for n in nodes_list])
        kmeans = KMeans(n_clusters=min(4, len(nodes_list)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
        cluster_map = dict(zip(nodes_list, clusters))
    else:
        cluster_map = {n: 0 for n in nodes_list}

    # KPI Kartları
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Düğüm", len(G.nodes))
    c2.metric("Toplam İlişki", len(G.edges))
    c3.metric("Ağ Yoğunluğu", f"{nx.density(G):.3f}")
    c4.metric("En Köprü Aktör", max(betweenness_cent, key=betweenness_cent.get))

    # GÖRSELLEŞTİRME
    tab_graph, tab_stats = st.tabs(["🕸️ İnteraktif Ağ Haritası", "📈 Metrik Detayları"])

    with tab_graph:
        size_metric = st.selectbox("Düğüm Boyutu Ne Olsun?", ["Degree Centrality", "Betweenness Centrality"])
        
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        colors = {0: "#FF4B4B", 1: "#1C83E1", 2: "#00C781", 3: "#FFBD45"}
        
        for node in net.nodes:
            n_id = node["id"]
            # Boyutu seçilen metriğe göre ayarla
            val = degree_cent[n_id] if size_metric == "Degree Centrality" else betweenness_cent[n_id]
            node["size"] = 15 + (val * 100)
            node["color"] = colors.get(cluster_map[n_id], "#999999")
            node["title"] = f"Degree: {degree_cent[n_id]:.2f}\nBetweenness: {betweenness_cent[n_id]:.2f}"

        net.toggle_physics(True)
        components.html(net.generate_html(), height=650)

    with tab_stats:
        st.subheader("Birim Bazlı Analitik Tablo")
        res_df = pd.DataFrame({
            'Aktör': nodes_list,
            'Degree Centrality': [f"{degree_cent[n]:.4f}" for n in nodes_list],
            'Betweenness': [f"{betweenness_cent[n]:.4f}" for n in nodes_list],
            'Closeness': [f"{closeness_cent[n]:.4f}" for n in nodes_list],
            'Küme (Cluster)': [cluster_map[n] for n in nodes_list]
        }).sort_values(by='Betweenness', ascending=False)
        
        st.dataframe(res_df, use_container_width=True)

else:
    st.info("👈 Analiz için veri seçin veya yükleyin.")
