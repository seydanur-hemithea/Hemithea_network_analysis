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
            # GitHub linklerini RAW formatına otomatik dönüştürür
            raw_url = url_or_file.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                return pd.read_csv(StringIO(response.text), sep=None, engine='python')
        else:
            return pd.read_csv(url_or_file, sep=None, engine='python')
    except Exception as e:
        return None

# --- 2. ÜST PANEL: VERİ SEÇİMİ ---
st.title("🌐 Hemithea: Advanced Network Analytics")
st.markdown("---")

col_top1, col_top2 = st.columns([1, 2])

with col_top1:
    dataset_choice = st.selectbox(
        "📂 Analiz Edilecek Veri Setini Seçin:",
        ["Efendi Projesi (Tarihsel)", "Game of Thrones (Popüler Kültür)", "Kendi CSV Dosyamı Yükle"]
    )

with col_top2:
    data = None
    if dataset_choice == "Efendi Projesi (Tarihsel)":
        url = "https://github.com/seydanur-hemithea/Hemithea_network_analysis/blob/main/efendi_veri.csv"
        data = load_data(url, is_url=True)
        st.success("✅ Efendi veri seti aktif.")
    
    elif dataset_choice == "Game of Thrones (Popüler Kültür)":
        url = "https://github.com/seydanur-hemithea/Hemithea_network_analysis/blob/main/got-edges.csv"
        data = load_data(url, is_url=True)
        st.success("✅ Westeros ilişki ağı yüklendi.")
        
    else:
        uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"])
        if uploaded_file:
            data = load_data(uploaded_file, is_url=False)

# --- 3. ANALİZ VE GÖRSELLEŞTİRME ---
if data is not None:
    st.markdown("### ⚙️ Yapılandırma ve Dinamik Metrikler")
    p1, p2, p3 = st.columns(3)
    
    all_cols = data.columns.tolist()
    with p1:
        source_col = st.selectbox("Kaynak (Source):", all_cols, index=0)
    with p2:
        target_col = st.selectbox("Hedef (Target):", all_cols, index=min(1, len(all_cols)-1))
    with p3:
        itme_kuvveti = st.slider("Düğüm Mesafesi (Ferahlık):", 50, 400, 200)

    # Veri İşleme
    df_clean = data[[source_col, target_col]].dropna()
    G = nx.from_pandas_edgelist(df_clean, source=source_col, target=target_col)

    # Metrik Hesaplamaları
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    
    # AI Kümeleme
    nodes = list(G.nodes())
    if len(nodes) >= 3:
        features = np.array([[deg_cent[n], bet_cent[n]] for n in nodes])
        kmeans = KMeans(n_clusters=min(5, len(nodes)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
        cluster_map = dict(zip(nodes, clusters))
    else:
        cluster_map = {n: 0 for n in nodes}

    # Özet Kartlar (KPI)
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Aktör", len(G.nodes))
    k2.metric("İlişki Sayısı", len(G.edges))
    k3.metric("Ağ Yoğunluğu", f"{nx.density(G):.4f}")
    k4.metric("Kilit Karakter/Birim", max(bet_cent, key=bet_cent.get))

    # Görselleştirme ve Tablo Sekmeleri
    tab_net, tab_stats = st.tabs(["🕸️ İnteraktif Harita", "📊 Analitik Veriler"])

    with tab_net:
        st.subheader(f"{dataset_choice} İlişki Dinamikleri")
        size_metric = st.selectbox("Düğüm Boyutlandırma Ölçütü:", ["Popülerlik (Degree)", "Stratejik Konum (Betweenness)"])
        
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        # Renk Paleti
        palette = ["#FF4B4B", "#1C83E1", "#00C781", "#FFBD45", "#7D3C98"]
        
        for node in net.nodes:
            n_id = node["id"]
            val = deg_cent[n_id] if size_metric == "Popülerlik (Degree)" else bet_cent[n_id]
            node["size"] = 20 + (val * 200) # GoT verisi için katsayıyı biraz artırdık
            node["color"] = palette[cluster_map[n_id] % len(palette)]
            node["title"] = f"Degree: {deg_cent[n_id]:.3f}\nBetweenness: {bet_cent[n_id]:.3f}"

        net.repulsion(node_distance=itme_kuvveti, central_gravity=0.5, spring_length=itme_kuvveti)
        components.html(net.generate_html(), height=750)

    with tab_stats:
        st.subheader("Birim Bazlı Detaylı İstatistikler")
        res_df = pd.DataFrame({
            'Aktör': nodes,
            'Degree Centrality': [deg_cent[n] for n in nodes],
            'Betweenness Centrality': [bet_cent[n] for n in nodes],
            'Küme (AI Cluster)': [cluster_map[n] for n in nodes]
        }).sort_values(by='Betweenness Centrality', ascending=False)
        st.dataframe(res_df, use_container_width=True)

else:
    st.info("👆 Analize başlamak için yukarıdaki menüden bir veri seti seçin.")
