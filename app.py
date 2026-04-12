import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from pyvis.network import Network
import streamlit.components.v1 as components

# --- 1. ANALİZ MOTORU ---
def create_network_graph(df, source, target, itme_kuvveti=2.5):
    try:
        # Sütun isimlerini temizle
        df.columns = df.columns.str.strip().str.lower()
        source = str(source).strip().lower()
        target = str(target).strip().lower()
        
        G = nx.from_pandas_edgelist(df, source=source, target=target)
        nodes_list = list(G.nodes())
        
        # Yapay Zeka ile Kümeleme (K-Means)
        degrees = np.array([G.degree(n) for n in nodes_list]).reshape(-1, 1)
        # En fazla 4 küme, veri azsa veri sayısı kadar
        n_clusters = min(4, len(nodes_list)) if len(nodes_list) > 0 else 1
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(StandardScaler().fit_transform(degrees))
        
        # Düğüm Yerleşimi (Spring Layout)
        pos = nx.spring_layout(G, k=itme_kuvveti, iterations=500, seed=42)
        
        # Kenarları Hazırla
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        
        # Düğüm Renkleri ve Boyutları
        colors = ['#FF4B4B', '#1C83E1', '#00C781', '#FFBD45']
        node_x, node_y, node_text, node_size, node_color_list = [], [], [], [], []
        
        for i, node in enumerate(nodes_list):
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            deg = G.degree(node)
            node_size.append((deg * 5) + 15)
            node_color_list.append(colors[clusters[i] % len(colors)])
            node_text.append(f"<b>Düğüm:</b> {node}<br><b>Bağlantı Sayısı:</b> {deg}")

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
            marker=dict(showscale=False, color=node_color_list, size=node_size, line_width=2, line_color='white')
        )

        # Plotly Grafiğini Oluştur
        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         showlegend=False, hovermode='closest', plot_bgcolor='white',
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         margin=dict(b=0, l=0, r=0, t=0)
                     ))
        return fig
    except Exception as e:
        st.error(f"Grafik oluşturulurken bir hata oluştu: {e}")
        return None

# --- 2. SAYFA AYARLARI ---
st.set_page_config(page_title="Hemithea Network Analysis", layout="wide")

# Sidebar - Ayarlar
st.sidebar.title("⚙️ Analiz Ayarları")
st.sidebar.markdown("Grafik üzerindeki düğümlerin birbirini itme kuvvetini ayarlayın.")
itme = st.sidebar.slider("Düğüm Uzaklığı (Ferahlık)", 1.0, 6.0, 2.5)

# Session State Yönetimi (Menüler arası geçiş için)
if 'mod' not in st.session_state:
    st.session_state.mod = None

# --- 3. ANA SAYFA VE MENÜ ---
st.title("🌐 Hemithea: Yapay Zeka Destekli Sosyal Ağ Analizi")

if st.session_state.mod is None:
    st.markdown("""
    ### Hoş Geldiniz! 
    Hemithea, karmaşık veri setlerindeki ilişkileri görselleştirerek kilit aktörleri ve toplulukları saniyeler içinde tespit etmenizi sağlar. 
    Lütfen bir işlem seçin:
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📘 Efendi Örnek Proje")
        st.write("Kitaptaki ağları hazır veriyle incelemek için tıklayın.")
        if st.button("Örneği Görüntüle", use_container_width=True):
            st.session_state.mod = "efendi"
            st.rerun()
            
    with col2:
        st.success("### 📁 Kendi Analizini Oluştur")
        st.write("Kendi CSV dosyanızı yükleyerek analiz yapın.")
        if st.button("Veri Yükleme Ekranına Git", use_container_width=True):
            st.session_state.mod = "kendi"
            st.rerun()

# --- 4. MODÜLLER ---

# --- MOD: EFENDİ ---
if st.session_state.mod == "efendi":
    st.subheader("Efendi Kitabı: Derin Sosyal Ağ Analizi")
    st.write("Aşağıdaki grafikte düğümlerin büyüklüğü bağlantı sayısını, renkleri ise yapay zeka tarafından tespit edilen toplulukları temsil eder.")
    
    try:
        # Efendi verisini oku
        df_efendi = pd.read_csv("efendi_veri.csv")
        fig = create_network_graph(df_efendi, "source", "target", itme_kuvveti=itme)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Örnek veri yüklenemedi: {e}")
        
    if st.button("← Ana Menüye Dön", type="primary"):
        st.session_state.mod = None
        st.rerun()

# --- MOD: KENDİ VERİN ---
elif st.session_state.mod == "kendi":
    st.subheader("📁 Kendi Verini Yükle ve Analiz Et")
    uploaded_file = st.file_uploader("Lütfen bir CSV dosyası seçin", type=["csv"])
    
    if uploaded_file:
        df_user = pd.read_csv(uploaded_file)
        st.write("### Veri Önizlemesi (İlk 5 Satır)")
        st.dataframe(df_user.head())
        
        # Sütun seçimi
        columns = df_user.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            source_col = st.selectbox("Kaynak Sütun (Kimden?)", columns)
        with c2:
            target_col = st.selectbox("Hedef Sütun (Kime?)", columns)
            
        if st.button("🚀 Analizi Başlat", use_container_width=True):
            fig = create_network_graph(df_user, source_col, target_col, itme_kuvveti=itme)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.success("Analiz başarıyla tamamlandı!")
                
    if st.button("← Ana Menüye Dön"):
        st.session_state.mod = None
        st.rerun()


if data is not None:
    # 1. HESAPLAMA MOTORU
    G = nx.from_pandas_edgelist(data, source='Source', target='Target')
    
    # Metrikleri hesapla
    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G) # Popülerlik puanı (Google algoritması)

    # 2. ÜST ÖZET KARTLARI (KPIs)
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

    # 3. ANA PANEL (Sol: Grafik, Sağ: Analiz Tablosu)
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("🌐 İnteraktif Ağ Haritası")
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333")
        net.from_nx(G)
        
        # Etiket ve Tooltip Ayarları
        for node in net.nodes:
            node["label"] = str(node["id"])
            node["title"] = f"Bağlantı Sayısı: {G.degree(node['id'])}"
            # Derecesine göre boyutlandır
            node["size"] = 15 + (degree_cent[node["id"]] * 100)
            node["color"] = "#E91E63" if node["id"] != top_node else "#FFD700"

        net.toggle_physics(True)
        html_data = net.generate_html()
        components.html(html_data, height=650)

    with right_col:
        st.subheader("📊 Analiz Detayları")
        
        # Metrik DataFrame'i
        metrics_df = pd.DataFrame({
            'Aktör': list(degree_cent.keys()),
            'Popülerlik': [f"{v:.2f}" for v in degree_cent.values()],
            'Köprü Gücü': [f"{v:.2f}" for v in betweenness.values()]
        }).sort_values(by='Popülerlik', ascending=False)

        st.dataframe(metrics_df, use_container_width=True, height=550)

    # 4. ALT PANEL: CSV İÇERİĞİ
    with st.expander("📄 Ham Veri Kaynağını Görüntüle"):
        st.write(data)
