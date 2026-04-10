import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. ANALİZ MOTORU (Senin Orijinal Fonksiyonun) ---
def create_network_graph(df, source, target, itme_kuvveti=2.5):
    # Sütun isimlerini temizle
    df.columns = df.columns.str.strip().str.lower()
    source = source.strip().lower()
    target = target.strip().lower()
    
    G = nx.from_pandas_edgelist(df, source=source, target=target)
    nodes_list = list(G.nodes())
    
    # AI Kümeleme
    degrees = np.array([G.degree(n) for n in nodes_list]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=min(4, len(nodes_list)), random_state=42)
    clusters = kmeans.fit_predict(StandardScaler().fit_transform(degrees))
    
    # Yerleşim
    pos = nx.spring_layout(G, k=itme_kuvveti, iterations=500, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    colors = ['#FF4B4B', '#1C83E1', '#00C781', '#FFBD45']
    
    node_x, node_y, node_text, node_size, node_color_list = [], [], [], [], []
    for i, node in enumerate(nodes_list):
        x, y = pos[node]
        node_x.append(x); node_y.append(y)
        deg = G.degree(node)
        node_size.append((deg * 5) + 15)
        node_color_list.append(colors[clusters[i]])
        node_text.append(f"<b>{node}</b><br>Bağlantı: {deg}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=False, color=node_color_list, size=node_size, line_width=2, line_color='white')
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(showlegend=False, hovermode='closest', plot_bgcolor='white',
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  margin=dict(b=0, l=0, r=0, t=0)))
    return fig

# --- 2. HİBRİT YAPILANDIRMA ---
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

# URL Parametrelerini Oku
query_params = st.query_params
is_app_mode = query_params.get("mode") == "app"
android_action = query_params.get("action")
android_source = query_params.get("source")
android_target = query_params.get("target")

# Eğer Android'den gelindiyse Arayüzü Gizle (CSS)
if is_app_mode:
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stSidebar"] {display: none;}
        .stDeployButton {display:none;}
        </style>
        """, unsafe_allow_html=True)

# Session State Yönetimi
if 'mod' not in st.session_state:
    st.session_state.mod = None

# Android Tetikleme Kontrolü
if android_action == "analyze":
    st.session_state.mod = "efendi"

# --- 3. ARAYÜZ (WEB VE UYGULAMA AYRIMI) ---

# Sadece Web'de Sidebar'ı göster
if not is_app_mode:
    st.sidebar.title("⚙️ Grafik Ayarları")
    itme = st.sidebar.slider("Düğüm Uzaklığı (Ferahlık)", 1.0, 6.0, 2.5)
    st.title("🌐 Hemithea: Yapay Zeka Destekli Sosyal Ağ Analizi")
else:
    itme = 2.5 # Uygulamada sabit veya parametreyle gelen ferahlık

# Mod seçimi (Sadece Web'de ve mod seçilmemişse)
if not is_app_mode and st.session_state.mod is None:
    st.write("Hoş geldiniz. Analize başlamak için lütfen bir seçenek belirleyin:")
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📘 Efendi Örnek Proje")
        if st.button("Örneği Görüntüle", use_container_width=True):
            st.session_state.mod = "efendi"
    with col2:
        st.success("### 📁 Kendi Analizini Oluştur")
        if st.button("Kendi Verimi Yükle", use_container_width=True):
            st.session_state.mod = "kendi"

st.markdown("---")

# --- 4. MODÜL MANTIĞI ---

if st.session_state.mod == "efendi":
    if not is_app_mode:
        st.subheader("Efendi Kitabı: Derin Sosyal Ağ Analizi")
    
    try:
        efendi_df = pd.read_csv("efendi_veri.csv") 
        
        # Eğer Android'den sütun isimleri geldiyse onları kullan, yoksa varsayılanı kullan
        src = android_source if android_source else "source"
        trg = android_target if android_target else "target"
        
        fig = create_network_graph(efendi_df, src, trg, itme_kuvveti=itme)
        st.plotly_chart(fig, use_container_width=True)
        
        if not is_app_mode:
            st.success("Analiz başarıyla yüklendi!")
    except Exception as e:
        st.error(f"Veri işlenirken hata oluştu: {e}")
    
    if not is_app_mode and st.button("← Ana Menüye Dön"): 
        st.session_state.mod = None

elif st.session_state.mod == "kendi":
    st.subheader("Veri Yükleme Alanı")
    file = st.file_uploader("Dosyanızı seçin (CSV/TXT)", type=["csv", "txt"])
    
    if file:
        df = pd.read_csv(file, sep=None, engine='python')
        st.dataframe(df.head(3))
        
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        c1, c2 = st.columns(2)
        with c1: src = st.selectbox("Kaynak (Source)", cols)
        with c2: trg = st.selectbox("Hedef (Target)", cols)
        
        if st.button("Analizi Başlat"):
            fig = create_network_graph(df, src, trg, itme_kuvveti=itme)
            st.plotly_chart(fig, use_container_width=True)
    
    if not is_app_mode and st.button("←
