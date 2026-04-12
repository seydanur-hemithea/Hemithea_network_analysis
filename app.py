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

# --- 2. VERİ YÜKLEME (RENDER) ---
RENDER_URL = "https://apphemitheanetwork.onrender.com/uploads/data.csv"

@st.cache_data(ttl=10) # 10 saniyede bir veriyi tazeler
def load_data():
    try:
        response = requests.get(RENDER_URL)
        if response.status_code == 200:
            return pd.read_csv(StringIO(response.text))
        return None
    except:
        return None

data = load_data()

# --- 3. ANA BAŞLIK ---
st.title("🌐 Hemithea: Yapay Zeka Destekli Sosyal Ağ Analizi")

if data is not None:
    # --- 4. ANALİZ VE HESAPLAMA MOTORU ---
    G = nx.from_pandas_edgelist(data, source='Source', target='Target')
    
    # Metrikleri hesapla
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    
    # Yapay Zeka (K-Means) ile Kümeleme (Topluluk Tespiti)
    nodes_list = list(G.nodes())
    if len(nodes_list) >= 2:
        # Derece ve Arasındalık metriklerini kullanarak kümeleme yapalım
        features = np.array([[degree_cent[n], betweenness[n]] for n in nodes_list])
        n_clusters = min(4, len(nodes_list))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(features))
        cluster_map = dict(zip(nodes_list, clusters))
    else:
        cluster_map = {n: 0 for n in nodes_list}

    # --- 5. ÜST ÖZET KARTLARI (KPIs) ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Düğüm", len(G.nodes))
    with col2:
        st.metric("Toplam Bağlantı", len(G.edges))
    with col3:
        top_node = max(degree_cent, key=degree_cent.get)
        st.metric("En Merkezi Aktör", top_node)
    with col4:
        st.metric("Ağ Yoğunluğu", f"{nx.density(G):.2f}")

    st.divider()

    # --- 6. ANA PANEL (Sol: Grafik, Sağ: Analiz Tablosu) ---
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("İnteraktif Ağ Haritası")
        # Sidebar'dan itme kuvvetini alalım (Ekstra özellik)
        itme = st.sidebar.slider("Düğüm Uzaklığı (Ferahlık)", 1.0, 5.0, 2.0)
        
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        # Küme renkleri
        cluster_colors = {0: "#FF4B4B", 1: "#1C83E1", 2: "#00C781", 3: "#FFBD45"}
        
        for node in net.nodes:
            n_id = node["id"]
            node["label"] = str(n_id)
            node["title"] = f"Bağlantı Sayısı: {G.degree(n_id)}\nKüme: {cluster_map[n_id]}"
            
            # Boyut: Bağlantı sayısına göre
            node["size"] = 15 + (degree_cent[n_id] * 80)
            
            # Renk: K-Means kümesine göre
            node["color"] = cluster_colors.get(cluster_map[n_id], "#999999")
            
            # En merkezi aktörü vurgula (Kenarlıkla)
            if n_id == top_node:
                node["borderWidth"] = 4
                node["color"] = "#FFD700" # Altın rengi

        net.toggle_physics(True)
        # Fizik ayarları ile düğüm mesafesini dinamik yap
        net.set_options(f"""
        var options = {{ "physics": {{ "barnesHut": {{ "gravitationalConstant": {-10000 * itme} }} }} }}
        """)
        
        html_data = net.generate_html()
        components.html(html_data, height=650)

    with right_col:
        st.subheader("Analiz Detayları")
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Popülerlik': [f"{v:.2f}" for v in degree_cent.values()],
            'Köprü Gücü': [f"{v:.2f}" for v in betweenness.values()],
            'Küme': [cluster_map[n] for n in nodes_list]
        }).sort_values(by='Popülerlik', ascending=False)

        st.dataframe(metrics_df, use_container_width=True, height=550)

else:
    st.warning("Henüz analiz edilecek bir veri yüklenmedi. Lütfen Android uygulamasından dosya gönderin.")
    st.info(f"Beklenen veri adresi: {RENDER_URL}")
