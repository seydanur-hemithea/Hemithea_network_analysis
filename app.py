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

# API AYARLARI
BASE_API_URL = "https://apphemitheanetwork.onrender.com"

# --- LOGIN FONKSİYONU ---
def login_request(username, password):
    try:
        response = requests.post(f"{BASE_API_URL}/login", data={"username": username, "password": password})
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Sunucu hatası: {e}")
        return None

# --- GÜVENLİK DUVARI VE MİSAFİR MODU ---
if "token" not in st.session_state and "guest_mode" not in st.session_state:
    # URL'den gelen token kontrolü (Android için)
    q_params = st.query_params
    url_token = q_params.get("token")
    url_user = q_params.get("username")

    if url_token and url_user:
        st.session_state["token"] = url_token
        st.session_state["username"] = url_user
        st.rerun()
    else:
        # Karşılama Ekranı
        with st.columns(3)[1]:
            st.title("🌐 Hemithea Network")
            st.write("Gelişmiş ağ analitiği ve ilişki haritalama platformu.")
            
            # PORTFOLYO BUTONU (İş verenler için)
            if st.button("✨ Örnek Analizleri İncele (Demo/Portfolio)", use_container_width=True):
                st.session_state["guest_mode"] = True
                st.session_state["username"] = "misafir_user"
                st.rerun()
            
            st.divider()
            
            with st.form("login_form"):
                st.subheader("🔑 Kullanıcı Girişi")
                u = st.text_input("Kullanıcı Adı")
                p = st.text_input("Şifre", type="password")
                if st.form_submit_button("Giriş Yap", use_container_width=True):
                    res = login_request(u, p)
                    if res:
                        st.session_state["token"] = res["access_token"]
                        st.session_state["username"] = res["username"]
                        st.rerun()
                    else:
                        st.error("Hatalı giriş bilgileri!")
            st.info("Kendi veri setlerinizi yüklemek ve kaydetmek için giriş yapın.")
        st.stop()

# --- VERİ YÜKLEME FONKSİYONU ---
@st.cache_data
def load_data(url_or_file, is_url=True, token=None):
    try:
        if is_url:
            # Token varsa URL'ye ekle (Güvenli veri çekme)
            final_url = url_or_file
            if token:
                final_url = f"{url_or_file}?token={token}"
            
            raw_url = final_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            response = requests.get(raw_url, timeout=10)
            if response.status_code == 200:
                return pd.read_csv(StringIO(response.text), sep=None, engine='python')
        else:
            return pd.read_csv(url_or_file, sep=None, engine='python')
    except Exception as e:
        return None

# --- ANA PANEL ---
st.title("🌐 Hemithea: Advanced Network Analytics")
if st.session_state.get("guest_mode"):
    st.sidebar.warning("📢 Demo Modu: Sadece örnek veriler açıktır.")
    if st.sidebar.button("🚪 Çıkış ve Giriş Yap"):
        del st.session_state["guest_mode"]
        st.rerun()

# Menü Seçimi
menu_options = ["Efendi Projesi (Tarihsel)", "Game of Thrones (Popüler Kültür)"]
if not st.session_state.get("guest_mode"):
    menu_options.append("Kendi Verilerim (Android'den Gelen)")
    menu_options.append("Yeni CSV Yükle")

dataset_choice = st.sidebar.selectbox("📂 Veri Seti Seçin:", menu_options)

# Veri Belirleme Mantığı
data = None
if dataset_choice == "Efendi Projesi (Tarihsel)":
    url = "https://github.com/seydanur-hemithea/Hemithea_network_analysis/blob/main/efendi_veri.csv"
    data = load_data(url, is_url=True)

elif dataset_choice == "Game of Thrones (Popüler Kültür)":
    url = "https://github.com/seydanur-hemithea/Hemithea_network_analysis/blob/main/got-edges.csv"
    data = load_data(url, is_url=True)

elif dataset_choice == "Kendi Verilerim (Android'den Gelen)":
    user = st.session_state["username"]
    token = st.session_state["token"]
    url = f"{BASE_API_URL}/uploads/{user}/network_data.csv"
    data = load_data(url, is_url=True, token=token)

elif dataset_choice == "Yeni CSV Yükle":
    uploaded_file = st.sidebar.file_uploader("CSV yükle", type=["csv"])
    if uploaded_file:
        data = load_data(uploaded_file, is_url=False)

# --- ANALİZ BLOĞU (Buradan sonrası senin orijinal kodunla devam ediyor) ---
if data is not None:
    # ... (nx metrikleri, KMeans kümeleme ve tablar burada aynen çalışacak) ...
    st.success(f"✅ {dataset_choice} başarıyla analiz edildi.")
    # Senin mevcut G = nx... kodların buraya gelecek.
