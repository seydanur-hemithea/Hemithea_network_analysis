import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. SENİN ÇALIŞAN MOTORUN (FONKSİYONLAŞTIRILMIŞ) ---
def create_network_graph(df, source, target, itme_kuvveti=2.5):
    # Sütun isimlerini temizleme (Senin kodundaki mantık)
    df.columns = df.columns.str.strip().str.lower()
    
    G = nx.from_pandas_edgelist(df, source=source, target=target)
    nodes_list = list(G.nodes())
    degrees = np.array([G.degree(n) for n in nodes_list]).reshape(-1, 1)
    
    scaler = StandardScaler()
    scaled_degrees = scaler.fit_transform(degrees)
    
    n_clusters = min(4, len(nodes_list)) 
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_degrees)
    
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
        node_size.append((deg * 4) + 12)
        node_color_list.append(colors[clusters[i]])
        node_text.append(f"<b>Düğüm:</b> {node}<br><b>Bağlantı:</b> {deg}<br><b>Grup:</b> {clusters[i]+1}")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=False, color=node_color_list, size=node_size, 
                    line=dict(width=2, color='white'), opacity=1)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(showlegend=False, hovermode='closest', plot_bgcolor='white', paper_bgcolor='white',
                                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                  margin=dict(b=0, l=0, r=0, t=0)))
    return fig

# --- 2. STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="Heithea AI Analiz", layout="wide")

st.sidebar.title("Heithea Panel")
mod = st.sidebar.radio("Modül:", ["Efendi Projesi (Örnek)", "Kendi Verini Yükle", "NLP Asistanı"])

# Ferahlık ayarını kullanıcıya bırakalım
itme = st.sidebar.slider("Düğüm Uzaklığı (Ferahlık)", 1.0, 5.0, 2.5)

if mod == "Efendi Projesi (Örnek)":
    st.title("Efendi Kitabı Sosyal Ağ Analizi")
    # Örnek veri seti (Daha önce konuştuğumuz yapı)
    example_df = pd.DataFrame([["ittihat", "maliye"], ["maliye", "ticaret"], ["ticaret", "medya"]], columns=["source", "target"])
    fig = create_network_graph(example_df, "source", "target", itme_kuvveti=itme)
    st.plotly_chart(fig, use_container_width=True)

elif mod == "Kendi Verini Yükle":
    st.title("Kendi Veri Setini Analiz Et")
    uploaded_file = st.file_uploader("CSV veya TXT dosyanızı yükleyin", type=["csv", "txt"])
    
    if uploaded_file is not None:
        # Dosya okuma (txt veya csv ayrımı)
        df = pd.read_csv(uploaded_file, sep=None, engine='python') 
        st.write("Veri Önizleme:")
        st.dataframe(df.head())
        
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        
        col1, col2 = st.columns(2)
        with col1: src = st.selectbox("Kaynak Sütun", cols)
        with col2: trg = st.selectbox("Hedef Sütun", cols)
        
        if st.button("Grafiği Oluştur"):
            fig = create_network_graph(df, src, trg, itme_kuvveti=itme)
            st.plotly_chart(fig, use_container_width=True)
            st.success("Başarıyla oluşturuldu!")

elif mod == "NLP Asistanı":
    st.title("Heithea NLP Asistanı")
    st.info("Bu modül yakında daha aktif olacak!")
    # Mevcut NLP kodlarını buraya ekleyebilirsin
