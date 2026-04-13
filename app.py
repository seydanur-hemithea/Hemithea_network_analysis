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
            # GitHub linkini RAW formata dönüştür
            raw_url = url_or_file.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                return pd.read_csv(StringIO(response.text), sep=None, engine='python')
        else:
            return pd.read_csv(url_or_file, sep=None, engine='python')
    except Exception as e:
        st.error(f"Veri yüklenemedi: {e}")
        return None

# --- 2. ANA PANEL GİRİŞ ---
st.title("🌐 Hemithea: Advanced Network Analytics")
st.markdown("---")

# --- 3. VERİ SEÇİM ALANI (SAYFA BAŞINDA) ---
col_setup1, col_setup2 = st.columns([1, 2])

with col_setup1:
    data_mode = st.radio("📂 Analiz İçin Veri Seçin:", 
                         ["Efendi Projesi (Hazır Veri)", "Kendi CSV Dosyamı Yükle"])

with col_setup2:
    if data_mode == "Efendi Projesi (Hazır Veri)":
        # Gönderdiğin güncel link
        EFENDI_URL = "https://github.com/seydanur-hemithea/Hemithea_network_analysis/blob/main/efendi_veri.csv"
        data = load_data(EFENDI_URL, is_url=True)
        if data is not None:
            st.success("✅ Efendi veri seti başarıyla bağlandı!")
    else:
        uploaded_file = st.file_uploader("Kendi CSV dosyanızı yükleyin", type=["csv"])
        data = load_data(uploaded_file, is_url=False) if uploaded_file else None

# --- 4. ANALİZ VE GÖRSELLEŞTİRME ---
if data is not None:
    st.markdown("### ⚙️ Yapılandırma ve Filtreler")
    p1, p2, p3 = st.columns(3)
    
    all_cols = data.columns.tolist()
    with p1:
        source_col = st.selectbox("Kaynak (Source):", all_cols, index=0)
    with p2:
        target_col = st.selectbox("Hedef (Target):", all_cols, index=min(1, len(all_cols)-1))
    with p3:
        # Senin sabahki kodunda olan o meşhur itme kuvveti
        itme_kuvveti = st.slider("Düğüm Mesafesi (Ferahlık):", 50, 300, 150)

    # Veri Hazırlığı
    df_clean = data[[source_col, target_col]].dropna()
    G = nx.from_pandas_edgelist(df_clean, source=source_col, target=target_col)

    # Matematiksel Metrikler
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G)
    
    # AI Kümeleme (K-Means)
    nodes = list(G.nodes())
    if len(nodes) >= 3:
        features = np.array([[deg_cent[n], bet_cent[n]] for n in nodes])
        kmeans = KMeans(n_clusters=min(4, len(nodes)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
        cluster_map = dict(zip(nodes, clusters))
    else:
        cluster_map = {n: 0 for n in nodes}

    # Özet Kartlar
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam Aktör", len(G.nodes))
    k1.caption("Ağdaki benzersiz kişi/birim sayısı")
    k2.metric("İlişki Sayısı", len(G.edges))
    k3.metric("Ağ Yoğunluğu", f"{nx.density(G):.4f}")
    k4.metric("En Stratejik Kişi", max(bet_cent, key=bet_cent.get))

    # Görselleştirme ve Tablo Sekmeleri
    tab_net, tab_table = st.tabs(["🕸️ İnteraktif Harita", "📊 Detaylı İstatistikler"])

    with tab_net:
        st.subheader("İlişki Dinamikleri")
        size_metric = st.selectbox("Düğümler neye göre büyüsün?", ["Popülerlik (Degree)", "Stratejik Önem (Betweenness)"])
        
        net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        # Modern Renk Paleti
        palette = ["#FF4B4B", "#1C83E1", "#00C781", "#FFBD45"]
        
        for node in net.nodes:
            n_id = node["id"]
            # Boyutlandırma seçimi
            val = deg_cent[n_id] if size_metric == "Popülerlik (Degree)" else bet_cent[n_id]
            node["size"] = 20 + (val * 150)
            node["color"] = palette[cluster_map[n_id] % len(palette)]
            node["title"] = f"Degree: {deg_cent[n_id]:.3f}\nBetweenness: {bet_cent[n_id]:.3f}"

        # Fizik motoru ayarları
        net.repulsion(node_distance=itme_kuvveti, central_gravity=0.5, spring_length=itme_kuvveti)
        components.html(net.generate_html(), height=700)

    with tab_table:
        st.subheader("Aktör Bazlı Analitik Veriler")
        res_df = pd.DataFrame({
            'Aktör': nodes,
            'Degree Centrality': [deg_cent[n] for n in nodes],
            'Betweenness Centrality': [bet_cent[n] for n in nodes],
            'AI Kümesi': [cluster_map[n] for n in nodes]
        }).sort_values(by='Betweenness Centrality', ascending=False)
        st.dataframe(res_df, use_container_width=True)

else:
    st.info("👆 Analize başlamak için yukarıdaki alandan bir veri seti seçin.")
