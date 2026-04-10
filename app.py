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
        # Sütun temizliği
        df.columns = df.columns.str.strip().str.lower()
        source = str(source).strip().lower()
        target = str(target).strip().lower()
        
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
    except Exception as e:
        st.error(f"Grafik Hatası: {e}")
        return None

# --- 2. AYARLAR VE PARAMETRELER ---
st.set_page_config(page_title="Hemithea Analiz", layout="wide")

query_params = st.query_params
is_app_mode = query_params.get("mode") == "app"

if is_app_mode:
    itme = 2.5 # Android'de sabit
    st.markdown("""<style>#MainMenu, footer, header, [data-testid="stSidebar"] {visibility: hidden; display: none;}</style>""", unsafe_allow_html=True)
else:
    st.sidebar.title("⚙️ Grafik Ayarları")
    itme = st.sidebar.slider("Düğüm Uzaklığı (Ferahlık)", 1.0, 6.0, 2.5)

if 'mod' not in st.session_state:
    st.session_state.mod = None

if query_params.get("action") == "analyze":
    st.session_state.mod = "efendi"

# --- 3. ANA EKRAN ---
if not is_app_mode:
    st.title("🌐 Hemithea Network Analysis")
    if st.session_state.mod is None:
        st.write("Hoş geldiniz. Analiz türünü seçin:")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📘 Efendi Örneği"): st.session_state.mod = "efendi"; st.rerun()
        with c2:
            if st.button("📁 Kendi Verim"): st.session_state.mod = "kendi"; st.rerun()

# --- 4. MODÜLLER ---

# --- EFENDİ MODU ---
if st.session_state.mod == "efendi":
    try:
        df = pd.read_csv("efendi_veri.csv")
        s = query_params.get("source", "source")
        t = query_params.get("target", "target")
        fig = create_network_graph(df, s, t, itme_kuvveti=itme)
        if fig: st.plotly_chart(fig, use_container_width=True)
        
        if not is_app_mode and st.button("← Geri"):
            st.session_state.mod = None; st.rerun()
    except Exception as e:
        st.error(f"Hata: {e}")

# --- KENDİ VERİM MODU ---
elif st.session_state.mod == "kendi":
    file = st.file_uploader("CSV Yükle", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if not is_app_mode:
            st.write("Veri yüklendi.")
            cols = df.columns.tolist()
            c1, c2 = st.columns(2)
            with c1: s_sel = st.selectbox("Kaynak", cols)
            with c2: t_sel = st.selectbox("Hedef", cols)
            if st.button("Analiz Et"):
                fig = create_network_graph(df, s_sel, t_sel, itme_kuvveti=itme)
                if fig: st.plotly_chart(fig, use_container_width=True)
        else:
            # Android Otomatik
            s_app = query_params.get("source", df.columns[0])
            t_app = query_params.get("target", df.columns[1])
            fig = create_network_graph(df, s_app, t_app, itme_kuvveti=itme)
            if fig: st.plotly_chart(fig, use_container_width=True)

    if not is_app_mode and st.button("← Geri"):
        st.session_state.mod = None; st.rerun()
