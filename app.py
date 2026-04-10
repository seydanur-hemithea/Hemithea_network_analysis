import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. ANALİZ MOTORU ---
def create_network_graph(df, source, target, itme_kuvveti=2.5):
    try:
        df.columns = df.columns.str.strip().str.lower()
        source = source.strip().lower()
        target = target.strip().lower()
        
        G = nx.from_pandas_edgelist(df, source=source, target=target)
        nodes_list = list(G.nodes())
        
        degrees = np.array([G.degree(n) for n in nodes_list]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=min(4, len(nodes_list)), random_state=42)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(degrees))
        
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
    except Exception as e:
        st.error(f"Grafik hatası: {e}")
        return None

# --- 2. AYARLAR ---
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

query_params = st.query_params
is_app_mode = query_params.get("mode") == "app"

# Sidebar ve Gizleme Kuralları
if is_app_mode:
    itme = 2.5
    st.markdown("""<style>#MainMenu, footer, header, [data-testid="stSidebar"] {visibility: hidden; display: none;}</style>""", unsafe_allow_html=True)
else:
    st.sidebar.title("⚙️ Grafik Ayarları")
    itme = st.sidebar.slider("Düğüm Uzaklığı (Ferahlık)", 1.0, 6.0, 2.5)

# Session State
if 'mod' not in st.session_state:
    st.session_state.mod = None

# Android Tetikleme
if query_params.get("action") == "analyze":
    st.session_state.mod = "efendi"

# --- 3. ANA SAYFA VE AÇIKLAMALAR ---
if not is_app_mode:
    st.title("🌐 Hemithea: Yapay Zeka Destekli Sosyal Ağ Analizi")
    
    if st.session_state.mod is None:
        st.write("### Hoş geldiniz! 👋")
        st.write("Bu platform, karmaşık sosyal ve ticari ağları görselleştirerek kilit aktörleri ve toplulukları saniyeler içinde tespit etmenizi sağlar.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("### 📘 Efendi Örnek Proje")
            st.write("Kitaptaki tarihi ve siyasi ağları hazır veriyle inceleyin.")
            if st.button("Örneği Görüntüle", use_container_width=True):
                st.session_state.mod = "efendi"
                st.rerun()
        with col2:
            st.success("### 📁 Kendi Analizini Oluştur")
            st.write("Kendi CSV dosyanızı yükleyerek ağ haritanızı çıkarın.")
            if st.button("Kendi Verimi Yükle", use_container_width=True):
                st.session_state.mod = "kendi"
                st.rerun()

# --- 4. MODÜL MANTIĞI ---

if st.session_state.mod == "efendi":
    if not is_app_mode:
        st.subheader("Efendi Kitabı: Derin Sosyal Ağ Analizi")
        st.write("Aşağıdaki grafikte düğümlerin büyüklüğü bağlantı sayısını, renkleri ise yapay zeka tarafından tespit edilen toplulukları temsil eder.")
    
    try:
        efendi_df = pd.read_csv("efendi_veri.csv")
        src = query_params.get("source", "source")
        trg = query_params.get("target", "target")
        
        fig = create_network_graph(efendi_df, src, trg, itme_kuvveti=itme)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
        # ANA MENÜYE DÖN BUTONU (Sadece Web'de Görünür)
        if not is_app_mode:
            if st.button("← Ana Menüye Dön", type="primary"):
                st.session_state.mod = None
                st.rerun()
                
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")

elif st.session_state.mod == "kendi":
    if not is_app_mode:
        st.subheader("Veri Yükleme ve Analiz")
        file = st.file_uploader("CSV Dosyanızı Seçin", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.write("Veri önizlemesi:")
            st.dataframe(df.head(3))
            
            # Dinamik sütun seçimi... (Burayı senin ihtiyacına göre geliştirebiliriz)
            
        if st.button("← Ana Menüye Dön"):
            st.session_state.mod = None
            st.rerun()
