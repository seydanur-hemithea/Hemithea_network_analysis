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
st.set_page_config(layout="wide", page_title="Hemithea Network Analysis", page_icon="🌐")

@st.cache_data
def load_data(url_or_file, is_url=True):
    try:
        if is_url:
            response = requests.get(url_or_file, timeout=10)
            if response.status_code == 200:
                return pd.read_csv(StringIO(response.text), sep=None, engine='python')
        else:
            return pd.read_csv(url_or_file, sep=None, engine='python')
    except Exception as e:
        return None

# --- 2. ANA SAYFA GİRİŞ VE VERİ SEÇİMİ ---
st.title("🌐 Hemithea: Advanced Network Analytics")
st.markdown("---")

# Veri Seçim Bölümü (Sayfanın Üstünde)
col_setup1, col_setup2 = st.columns([1, 2])

with col_setup1:
    data_mode = st.radio("📂 Veri Kaynağı Seçin:", ["Efendi Projesi (Hazır Demo)", "Kendi Verini Yükle"])

with col_setup2:
    EFENDI_URL = "https://github.com/seydanur-hemithea/Hemithea_network_analysis/blob/main/efendi_veri.csv"
    if data_mode == "Efendi Projesi (Hazır Demo)":
        data = load_data(EFENDI_URL, is_url=True)
        if data is not None:
            st.success("✅ Efendi veri seti başarıyla yüklendi.")
    else:
        uploaded_file = st.file_uploader("Analiz edilecek CSV dosyasını buraya bırakın", type=["csv"])
        data = load_data(uploaded_file, is_url=False) if uploaded_file else None

# --- 3. KOLON SEÇİMİ VE ANALİZ ---
if data is not None:
    st.markdown("### 🔗 İlişki Parametreleri")
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    
    all_cols = data.columns.tolist()
    with col_sel1:
        source_col = st.selectbox("Kaynak (Source) Kolonu:", all_cols, index=0)
    with col_sel2:
        target_col = st.selectbox("Hedef (Target) Kolonu:", all_cols, index=min(1, len(all_cols)-1))
    with col_sel3:
        size_metric = st.selectbox("Düğüm Boyutlandırma:", ["Degree Centrality", "Betweenness Centrality"])

    if source_col == target_col:
        st.warning("⚠️ Lütfen farklı Kaynak ve Hedef kolonları seçin.")
        st.stop()

    # --- HESAPLAMA MOTORU ---
    clean_df = data[[source_col, target_col]].dropna()
    G = nx.from_pandas_edgelist(clean_df, source=source_col, target=target_col)

    # Metrikler
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    
    # AI Kümeleme
    nodes = list(G.nodes())
    if len(nodes) >= 3:
        features = np.array([[deg_cent[n], bet_cent[n]] for n in nodes])
        kmeans = KMeans(n_clusters=min(4, len(nodes)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
        cluster_map = dict(zip(nodes, clusters))
    else:
        cluster_map = {n: 0 for n in nodes}

    # --- ÖZET KARTLARI (KPI) ---
    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Toplam Düğüm", len(G.nodes))
    kpi2.metric("Toplam Bağlantı", len(G.edges))
    kpi3.metric("Ağ Yoğunluğu", f"{nx.density(G):.4f}")
    kpi4.metric("En Stratejik Aktör", max(bet_cent, key=bet_cent.get))

    # --- GÖRSELLEŞTİRME ---
    tab_graph, tab_data = st.tabs(["🕸️ İnteraktif Ağ Haritası", "📊 Analitik Detaylar"])

    with tab_graph:
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        palette = ["#FF4B4B", "#1C83E1", "#00C781", "#FFBD45"]
        for node in net.nodes:
            n_id = node["id"]
            val = deg_cent[n_id] if size_metric == "Degree Centrality" else bet_cent[n_id]
            node["size"] = 20 + (val * 120)
            node["color"] = palette[cluster_map[n_id] % len(palette)]
            node["title"] = f"Degree: {deg_cent[n_id]:.3f}\nBetweenness: {bet_cent[n_id]:.3f}"

        # Grafiği toparlayan fizik ayarları
        net.repulsion(node_distance=150, central_gravity=0.6, spring_length=150)
        components.html(net.generate_html(), height=650)

    with tab_data:
        res_df = pd.DataFrame({
            'Birim/Aktör': nodes,
            'Popülerlik (Degree)': [deg_cent[n] for n in nodes],
            'Köprü Gücü (Betweenness)': [bet_cent[n] for n in nodes],
            'AI Kümesi': [cluster_map[n] for n in nodes]
        }).sort_values(by='Köprü Gücü (Betweenness)', ascending=False)
        st.dataframe(res_df, use_container_width=True)

else:
    st.info("👆 Analize başlamak için yukarıdan 'Efendi' demosunu seçin veya kendi CSV dosyanızı yükleyin.")
