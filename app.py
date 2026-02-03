import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import feedparser
import google.generativeai as genai
import requests
import re
import json
import io
import zipfile
import os
import glob
from deep_translator import GoogleTranslator
from textblob import TextBlob
from datetime import datetime, timedelta
from tradingview_ta import TA_Handler, Interval, Exchange
import time

# --- 1. AYARLAR & TASARIM ---
st.set_page_config(page_title="Cotton Geni's", page_icon="☁️", layout="wide")

st.markdown("""
<style>
    .stApp {background-color: #FAFAFA;}
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    h1, h2, h3 {color: #0F172A; font-family: 'Helvetica Neue', sans-serif; font-weight: 700;}
    section[data-testid="stSidebar"] {background-color: #F8F9FA; border-right: 1px solid #E5E7EB;}
    .success-box {padding: 10px; background-color: #D1FAE5; color: #065F46; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;}
    .warning-box {padding: 10px; background-color: #FEF3C7; color: #92400E; border-radius: 5px; margin-bottom: 10px; font-size: 0.9em;}
</style>
""", unsafe_allow_html=True)

# --- 2. VERİ MOTORLARI ---
@st.cache_data(ttl=10)
def get_live_price():
    try:
        cotton = TA_Handler(symbol="CT1!", screener="america", exchange="ICEUS", interval=Interval.INTERVAL_1_MINUTE)
        analysis = cotton.get_analysis()
        return analysis.indicators["close"], analysis.indicators["close"] - analysis.indicators["open"]
    except: return 0.0, 0.0

@st.cache_data(ttl=60)
def get_futures_table():
    url = "https://futures.tradingcharts.com/futures/quotes/ct.html?cbase=ct"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.google.com/"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            dfs = pd.read_html(r.text, match="Last")
            if len(dfs) > 0:
                raw_df = dfs[0]
                def clean(val):
                    txt = re.sub(r'[^\d.-]', '', str(val))
                    try: return float(txt)
                    except: return None
                raw_df['Last_Clean'] = raw_df['Last'].apply(clean)
                raw_df['Chg_Clean'] = raw_df['Chg'].apply(clean)
                final_df = raw_df.dropna(subset=['Last_Clean']).rename(columns={'Last_Clean': 'Son', 'Chg_Clean': 'Değişim', 'Month': 'Vade', 'Vol': 'Hacim'})
                cash_row = final_df[final_df['Vade'].astype(str).str.contains("Cash", case=False, na=False)]
                futures_df = final_df[~final_df['Vade'].astype(str).str.contains("Cash", case=False, na=False)]
                if not cash_row.empty:
                    return pd.concat([cash_row, futures_df])[['Vade', 'Son', 'Değişim', 'Hacim']].reset_index(drop=True), "Canlı Veri (Spot)"
                elif not futures_df.empty:
                    return futures_df[['Vade', 'Son', 'Değişim', 'Hacim']].reset_index(drop=True), "Canlı Veri (Vadeli)"
    except: pass
    
    month_map = {3: 'H', 5: 'K', 7: 'N', 12: 'Z'} 
    curr_date = datetime.now(); rows = []
    for i in range(24): 
        future_date = curr_date + timedelta(days=30*i)
        m, y = future_date.month, future_date.year
        if m in month_map:
            sym = f"CT{month_map[m]}{str(y)[-2:]}.NYB"; name = f"{future_date.strftime('%b')}'{str(y)[-2:]}"
            if y > curr_date.year or (y == curr_date.year and m >= curr_date.month):
                try:
                    import yfinance as yf
                    t = yf.Ticker(sym); hist = t.history(period="5d")
                    if not hist.empty: rows.append({"Vade": name, "Son": float(hist['Close'].iloc[-1]), "Değişim": float(hist['Close'].iloc[-1] - (hist['Close'].iloc[-2] if len(hist)>1 else hist['Close'].iloc[-1])), "Hacim": int(hist['Volume'].iloc[-1])})
                except: continue
                if len(rows) >= 8: break
    return pd.DataFrame(rows) if rows else pd.DataFrame(), "Yedek Veri" if rows else "Veri Yok"

@st.cache_data(ttl=60)
def get_market_history(period_str):
    import yfinance as yf
    mapping = {"3 Ay": "3mo", "6 Ay": "6mo", "1 Yıl": "1y", "3 Yıl": "3y"}
    
    # Israrcı Mod: Hata verirse 3 kez dene
    for attempt in range(3):
        try:
            data = yf.download("CT=F BZ=F DX-Y.NYB CNY=X", period=mapping[period_str], group_by='ticker', progress=False, threads=False)
            df = pd.DataFrame()
            if not data.empty:
                if 'CT=F' in data: df['Pamuk'] = data['CT=F']['Close']
                if 'BZ=F' in data: df['Petrol'] = data['BZ=F']['Close']
                if 'DX-Y.NYB' in data: df['DXY'] = data['DX-Y.NYB']['Close']
                if 'CNY=X' in data: df['USDCNY'] = data['CNY=X']['Close']
                result = df.dropna()
                if not result.empty: return result
        except:
            time.sleep(1) # Hata olursa 1 saniye bekle tekrar dene
            
    return pd.DataFrame()

@st.cache_data(ttl=600)
def get_comparison_data(ticker, period_str):
    import yfinance as yf
    mapping = {"3 Ay": "3mo", "6 Ay": "6mo", "1 Yıl": "1y", "3 Yıl": "3y"}
    try:
        data = yf.download(ticker, period=mapping[period_str], progress=False)
        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data['Close'].squeeze()
        return pd.Series()
    except: return pd.Series()

# --- GELİŞMİŞ COT ANALİZİ (AKILLI DOSYA BULUCU) ---
def parse_cftc_file(file_obj):
    """CFTC Dosyasını (CSV/TXT) Okur ve Temizler"""
    try:
        df = pd.read_csv(file_obj, header=None, low_memory=False)
        cotton_df = df[df[0].astype(str).str.contains("COTTON NO. 2", case=False, na=False)].copy()
        
        if not cotton_df.empty:
            clean_df = pd.DataFrame()
            clean_df['Date'] = pd.to_datetime(cotton_df[2])
            clean_df['Fon_Long'] = pd.to_numeric(cotton_df[8], errors='coerce')
            clean_df['Fon_Short'] = pd.to_numeric(cotton_df[9], errors='coerce')
            clean_df['Ticari_Long'] = pd.to_numeric(cotton_df[11], errors='coerce')
            clean_df['Ticari_Short'] = pd.to_numeric(cotton_df[12], errors='coerce')
            
            clean_df = clean_df.sort_values('Date').reset_index(drop=True)
            clean_df['Net_Fon'] = clean_df['Fon_Long'] - clean_df['Fon_Short']
            clean_df['Net_Ticari'] = clean_df['Ticari_Long'] - clean_df['Ticari_Short']
            return clean_df
    except Exception as e:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_cot_data(uploaded_file=None):
    source_msg = ""
    df = pd.DataFrame()

    # 1. YÖNTEM: KULLANICI ANLIK YÜKLEDİ Mİ?
    if uploaded_file is not None:
        df = parse_cftc_file(uploaded_file)
        if not df.empty: return df, "✅ Anlık Yüklenen Dosya"

    # 2. YÖNTEM: GITHUB/YEREL DOSYALARI BUL (Otomatik Birleştirme)
    # cot_history_*.csv formatındaki tüm dosyaları bul (2025, 2026 vs.)
    local_files = glob.glob("cot_history*.csv")
    if local_files:
        combined_dfs = []
        for f_path in local_files:
            with open(f_path, "r") as f:
                parsed = parse_cftc_file(f)
                if not parsed.empty: combined_dfs.append(parsed)
        
        if combined_dfs:
            full_df = pd.concat(combined_dfs).drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
            return full_df, f"✅ Sistem Veritabanı ({len(local_files)} Dosya)"

    # 3. YÖNTEM: WEB'DEN ÇEKMEYİ DENE (Yedek)
    try:
        url = "https://www.cftc.gov/dea/newcot/ice_lf.txt"
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200:
            df = parse_cftc_file(io.StringIO(r.text))
            if not df.empty: return df, "⚠️ Yedek Sunucu (Sadece Güncel)"
    except: pass
    
    return pd.DataFrame(), "❌ Veri Bulunamadı"

def calculate_cot_trends(df):
    if df.empty: return None
    last_row = df.iloc[-1]
    
    if len(df) >= 2:
        prev_row = df.iloc[-2]
        fund_chg_w = last_row['Net_Fon'] - prev_row['Net_Fon']
        month_ago = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
        fund_trend = "🟢 Artıyor" if (last_row['Net_Fon'] - month_ago['Net_Fon']) > 0 else "🔴 Azalıyor"
        comm_trend = "🟢 Artıyor" if (last_row['Net_Ticari'] - month_ago['Net_Ticari']) > 0 else "🔴 Azalıyor"
        graph_df = df.tail(52) # Son 1 Yıl
    else:
        fund_chg_w = 0; fund_trend = "Nötr"; comm_trend = "Nötr"; graph_df = df
    
    return {"current": last_row, "fund_chg_w": fund_chg_w, "fund_trend": fund_trend, "comm_trend": comm_trend, "df_graph": graph_df}

def calculate_indicators(df):
    delta = df['Pamuk'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss; df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA20'] = df['Pamuk'].rolling(20).mean(); df['Upper'] = df['SMA20'] + (df['Pamuk'].rolling(20).std() * 2); df['Lower'] = df['SMA20'] - (df['Pamuk'].rolling(20).std() * 2)
    return df

@st.cache_data(ttl=900)
def get_intel_news():
    queries = ['"cotton futures"', '"USDA" cotton export sales', '"H&M" sales', "polyester fiber price"]
    whitelist = ['cotton', 'pamuk', 'textile', 'h&m', 'inditex', 'polyester', 'price', 'usda', 'export']
    news_data = []; translator = GoogleTranslator(source='auto', target='tr'); current_year = datetime.now().year
    try:
        for q in queries:
            feed = feedparser.parse(f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en")
            for entry in feed.entries[:3]:
                if not any(w in entry.title.lower() for w in whitelist): continue
                try: 
                    if entry.published_parsed.tm_year < current_year - 1: continue 
                except: pass
                blob = TextBlob(entry.title); sent = "🟢" if blob.sentiment.polarity > 0.1 else "🔴" if blob.sentiment.polarity < -0.1 else "⚪"
                try: tr = translator.translate(entry.title)
                except: tr = entry.title
                news_data.append({"Duygu": sent, "Başlık": tr, "Link": entry.link, "Orjinal": entry.title})
                if len(news_data) >= 10: break
            if len(news_data) >= 10: break
    except: pass
    return pd.DataFrame(news_data)

def ask_gemini_with_chart(api_key, spot_val, news_df, table_df, poly_cent, cot_summary, scenario, weights):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model = genai.GenerativeModel(next((m for m in models if 'flash' in m), models[0] if models else None))
        table_txt = table_df.to_string(index=False) if not table_df.empty else "Veri Yok"
        priority_instruction = "AĞIRLIKLAR (0-10):\n" + "\n".join([f"- {k}: {v}" for k,v in weights.items()])
        prompt = f"""Sen Profesyonel Pamuk Tüccarısın. {priority_instruction}
        BUGÜN: {datetime.now().strftime("%Y-%m-%d")} | FİYAT: {spot_val:.2f}c | RAKİP: Polyester {poly_cent:.2f}c | COT: {cot_summary} 
        TABLO: {table_txt} | HABERLER: {news_df['Orjinal'].to_string() if not news_df.empty else "Yok"} 
        SENARYO: {scenario}
        GÖREV: Fiyat yönünü belirle ve 1 yıllık 3 tahmin noktası (JSON) oluştur.
        ÇIKTI FORMATI: ## 🧭 Stratejik Analiz \n* [Yorum...] \n## 🦊 Pozisyonlar \n* [Yorum...]
        ```json
        {{ "forecast": [ {{"label": "Bugün", "date": "{datetime.now().strftime("%Y-%m-%d")}", "price": {spot_val}}}, {{"label": "Kısa Vade", "date": "YYYY-MM-DD", "price": 00.00}}, {{"label": "Orta Vade", "date": "YYYY-MM-DD", "price": 00.00}}, {{"label": "Uzun Vade", "date": "YYYY-MM-DD", "price": 00.00}} ] }}
        ```"""
        return model.generate_content(prompt).text
    except Exception as e: return f"Hata: {str(e)}"

def parse_ai_chart_data(text):
    try:
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if match: return pd.DataFrame(json.loads(match.group(1))['forecast'])
    except: pass
    return pd.DataFrame()

# --- 3. ANA UYGULAMA ---
def check_login():
    if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
    if not st.session_state['logged_in']:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<br><br><h2 style='text-align: center;'>☁️ Cotton Geni's Giriş</h2>", unsafe_allow_html=True)
            user = st.text_input("Kullanıcı Adı")
            passwd = st.text_input("Şifre", type="password")
            if st.button("Giriş Yap", type="primary", use_container_width=True):
                if user == "admin" and passwd == "pamuk2026":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else: st.error("Hatalı bilgiler.")
        return False
    return True

if check_login():
    if "GEMINI_API_KEY" in st.secrets: api_key = st.secrets["GEMINI_API_KEY"]
    else: api_key = st.sidebar.text_input("API Anahtarı", type="password")

    if 'w_news' not in st.session_state: st.session_state.update({'w_news': 9, 'w_cot': 7, 'w_tech': 5, 'w_poly': 8, 'w_basis': 6})
    def set_auto_weights(): st.session_state.update({'w_news': 9, 'w_cot': 7, 'w_tech': 5, 'w_poly': 8, 'w_basis': 6})

    # SIDEBAR
    with st.sidebar:
        st.markdown("## ☁️ Menü")
        menu = st.radio("", ["📊 Ana Ekran", "⚖️ Karşılaştırma", "🦊 Pozisyonlar (COT)", "🤖 AI Strateji", "📉 Teknik"], label_visibility="collapsed")
        st.divider()
        
        # --- VERİ YÖNETİM PANELİ ---
        with st.expander("💾 Veri Yönetimi (COT)", expanded=False):
            st.info("Eğer otomatik veri gelmiyorsa, CFTC raporunu (Text) buradan yükleyebilirsiniz.")
            uploaded_cot = st.file_uploader("CFTC Raporu Yükle", type=['txt', 'csv'])
        
        with st.expander("⚙️ Veri & Grafik Ayarları"):
            poly_rmb = st.number_input("Polyester (RMB)", value=6587)
            period = st.selectbox("Grafik Süresi", ["3 Ay", "6 Ay", "1 Yıl", "3 Yıl"], index=1)
        
        with st.expander("🧠 AI Karar Mekanizması", expanded=False):
            st.caption("AI Analiz Ağırlıkları (0-10)")
            st.session_state['w_news'] = st.slider("Haberler", 0, 10, st.session_state['w_news'])
            st.session_state['w_poly'] = st.slider("Polyester", 0, 10, st.session_state['w_poly'])
            st.session_state['w_cot'] = st.slider("COT", 0, 10, st.session_state['w_cot'])
            st.session_state['w_basis'] = st.slider("Basis", 0, 10, st.session_state['w_basis'])
            st.session_state['w_tech'] = st.slider("Teknik", 0, 10, st.session_state['w_tech'])
            if st.button("✨ Otomatik", use_container_width=True): set_auto_weights(); st.rerun()
        if st.button("Çıkış", type="secondary"): st.session_state['logged_in'] = False; st.rerun()

    # --- VERİ HAZIRLIĞI ---
    # COT Verisi (Hibrit Sistem)
    cot_df, cot_source = get_cot_data(uploaded_cot)
    cot_analysis = calculate_cot_trends(cot_df)

    live_price, live_change = get_live_price()
    table_data, table_source = get_futures_table()
    df_hist = get_market_history(period)
    news_df = get_intel_news()

    if df_hist.empty: st.error("Fiyat verisi hatası. Lütfen sayfayı yenileyin (İnternet veya Yahoo Finance geçici olarak yanıt vermiyor olabilir)."); st.stop()
    else: df_hist = calculate_indicators(df_hist)

    usdcny = float(df_hist['USDCNY'].iloc[-1]) if 'USDCNY' in df_hist else 7.2
    poly_cent = (poly_rmb / usdcny / 2204.62) * 100

    # COT Özet Metni
    if cot_analysis and cot_analysis['current'] is not None:
        curr = cot_analysis['current']
        cot_summary = f"Tarih: {curr['Date'].strftime('%Y-%m-%d')} | Fon Net: {curr['Net_Fon']} ({cot_analysis['fund_trend']}), Ticari Net: {curr['Net_Ticari']} ({cot_analysis['comm_trend']})"
    else:
        cot_summary = "COT Verisi Yok"

    if live_price > 0: display_price = live_price; display_change = live_change
    else:
        if not table_data.empty: display_price = float(table_data.iloc[0]['Son']); display_change = float(table_data.iloc[0]['Değişim'])
        else: display_price = df_hist['Pamuk'].iloc[-1]; display_change = 0.0

    # --- EKRANLAR ---
    if menu == "📊 Ana Ekran":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pamuk (Canlı)", f"{display_price:.2f}c", f"{display_change:.2f}")
        c2.metric("Petrol", f"${df_hist['Petrol'].iloc[-1]:.2f}", f"{df_hist['Petrol'].iloc[-1]-df_hist['Petrol'].iloc[-2]:.2f}")
        c3.metric("Sentetik", f"{poly_cent:.2f}c", "Piyasa")
        c4.metric("DXY", f"{df_hist['DXY'].iloc[-1]:.2f}", f"{df_hist['DXY'].iloc[-1]-df_hist['DXY'].iloc[-2]:.2f}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Fiyat Grafiği")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Pamuk'], name='Pamuk', line=dict(color='#1E3A8A', width=3)))
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Petrol'], name='Petrol', line=dict(color='#DC2626', width=2, dash='dot'), yaxis='y2'))
        fig.update_layout(height=500, template="plotly_white", margin=dict(l=20,r=20,t=40,b=20), yaxis2=dict(overlaying='y', side='right', showgrid=False), legend=dict(orientation="h", y=1.1, x=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.subheader(f"📋 Derinlik ({table_source})")
            if not table_data.empty: st.dataframe(table_data, column_config={"Vade": st.column_config.TextColumn("Kontrat", width="small"), "Son": st.column_config.NumberColumn("Son Fiyat", format="%.2f"), "Değişim": st.column_config.NumberColumn("Değişim", format="%.2f"), "Hacim": st.column_config.ProgressColumn("Hacim", format="%d", min_value=0, max_value=int(table_data['Hacim'].max()))}, hide_index=True, use_container_width=True)
            else: st.warning("Veri bekleniyor...")
        with col_r:
            st.subheader("📡 İstihbarat")
            if not news_df.empty: st.dataframe(news_df[['Duygu', 'Başlık', 'Link']], column_config={"Link": st.column_config.LinkColumn("Oku", display_text="🔗"), "Duygu": st.column_config.TextColumn("Yön", width="small"), "Başlık": st.column_config.TextColumn("Haber Başlığı", width="medium")}, hide_index=True, use_container_width=True)

    elif menu == "⚖️ Karşılaştırma":
        st.subheader("⚖️ Emtia & Varlık Karşılaştırma")
        comp_assets = {"Mısır (Corn)": "ZC=F", "Soya (Soybean)": "ZS=F", "Buğday (Wheat)": "ZW=F", "Şeker (Sugar)": "SB=F", "Kahve (Coffee)": "KC=F", "Petrol (Brent)": "BZ=F", "Altın (Gold)": "GC=F"}
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1: selected_asset_name = st.selectbox("Kıyaslanacak Varlık:", list(comp_assets.keys()))
        with col_sel2: view_mode = st.radio("Grafik Modu:", ["Yüzdesel Getiri (%)", "Fiyat (Dolar)"], horizontal=True)
        selected_ticker = comp_assets[selected_asset_name]
        cotton_series = df_hist['Pamuk']
        target_series = get_comparison_data(selected_ticker, period)
        if not target_series.empty:
            df_comp = pd.DataFrame({'Pamuk': cotton_series, 'Rakip': target_series}).dropna()
            correlation = df_comp['Pamuk'].corr(df_comp['Rakip'])
            col_k1, col_k2 = st.columns([1, 3])
            with col_k1:
                st.metric("Korelasyon Katsayısı", f"{correlation:.2f}", delta_color="off")
                st.caption(f"1.00: Birebir aynı\n0.00: İlgisiz\n-1.00: Ters hareket")
            with col_k2:
                fig_comp = go.Figure()
                if view_mode == "Yüzdesel Getiri (%)":
                    norm_cotton = ((df_comp['Pamuk'] - df_comp['Pamuk'].iloc[0]) / df_comp['Pamuk'].iloc[0]) * 100
                    norm_target = ((df_comp['Rakip'] - df_comp['Rakip'].iloc[0]) / df_comp['Rakip'].iloc[0]) * 100
                    fig_comp.add_trace(go.Scatter(x=df_comp.index, y=norm_cotton, name='Pamuk (%)', line=dict(color='#1E3A8A', width=3)))
                    fig_comp.add_trace(go.Scatter(x=df_comp.index, y=norm_target, name=f'{selected_asset_name.split()[0]} (%)', line=dict(color='#EA580C', width=2)))
                    yaxis_title = "Getiri (%)"
                else:
                    fig_comp.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Pamuk'], name='Pamuk (Cent)', line=dict(color='#1E3A8A', width=3)))
                    fig_comp.add_trace(go.Scatter(x=df_comp.index, y=df_comp['Rakip'], name=f'{selected_asset_name.split()[0]}', line=dict(color='#EA580C', width=2, dash='dot'), yaxis='y2'))
                    yaxis_title = "Pamuk Fiyatı"
                    fig_comp.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title=selected_asset_name))
                fig_comp.update_layout(height=450, template="plotly_white", margin=dict(l=20,r=20,t=20,b=20), hovermode="x unified", yaxis_title=yaxis_title, legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_comp, use_container_width=True)
        else: st.warning("Seçilen varlık için veri çekilemedi.")

    elif menu == "🦊 Pozisyonlar (COT)":
        if cot_analysis and cot_analysis['current'] is not None:
            curr = cot_analysis['current']
            st.subheader(f"🦊 COT Analizi ({curr['Date'].strftime('%Y-%m-%d')})")
            
            # BİLGİ KUTUSU (KAYNAK)
            if "GitHub" in cot_source: st.markdown(f'<div class="success-box">{cot_source} kullanılıyor.</div>', unsafe_allow_html=True)
            elif "Anlık" in cot_source: st.markdown(f'<div class="success-box">{cot_source} (Geçici) kullanılıyor.</div>', unsafe_allow_html=True)
            else: st.markdown(f'<div class="warning-box">{cot_source}</div>', unsafe_allow_html=True)

            k1, k2, k3, k4 = st.columns(4)
            fon_delta = cot_analysis['fund_chg_w']
            k1.metric("Fon Net (Long-Short)", f"{curr['Net_Fon']:,}", f"{fon_delta:,}", delta_color="normal")
            k2.metric("Fon İştahı (Aylık)", cot_analysis['fund_trend'], delta_color="off")
            k3.metric("Ticari Net (Hedger)", f"{curr['Net_Ticari']:,}", delta_color="off") 
            k4.metric("Ticari Davranış", cot_analysis['comm_trend'], delta_color="off")
            
            st.divider()
            
            if len(cot_analysis['df_graph']) > 1:
                st.markdown("### 📊 Fon vs. Ticari Savaşı (Son 1 Yıl)")
                df6 = cot_analysis['df_graph']
                fig_cot = go.Figure()
                fig_cot.add_trace(go.Scatter(x=df6['Date'], y=df6['Net_Fon'], name='Fon Net Pozisyonu', line=dict(color='#10B981', width=3), fill='tozeroy', fillcolor='rgba(16, 185, 129, 0.1)'))
                fig_cot.add_trace(go.Scatter(x=df6['Date'], y=df6['Net_Ticari'], name='Ticari Net Pozisyonu', line=dict(color='#6B7280', width=2, dash='dot')))
                fig_cot.add_hline(y=0, line_width=1, line_color="black")
                fig_cot.update_layout(height=400, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_cot, use_container_width=True)
            else:
                st.warning("⚠️ Grafik için yeterli tarihsel veri yok.")
            
        else:
            st.error("⚠️ Veri Bulunamadı. Lütfen sol menüden CFTC dosyasını yükleyin veya GitHub'a 'cot_history.csv' ekleyin.")

    elif menu == "🤖 AI Strateji":
        st.subheader("Yapay Zeka Analisti")
        current_weights = {"Haberler": st.session_state['w_news'], "Polyester": st.session_state['w_poly'], "COT": st.session_state['w_cot'], "Basis": st.session_state['w_basis'], "Teknik": st.session_state['w_tech']}
        st.info(f"💡 AI Ağırlıkları: Haber={current_weights['Haberler']}, Teknik={current_weights['Teknik']}")
        scen = st.text_area("Senaryo / Soru:", placeholder="Örn: Faiz kararı sonrası pamuk ne olur?")
        if st.button("Analizi Başlat", type="primary") and api_key:
            with st.spinner("AI piyasayı tarıyor..."):
                report = ask_gemini_with_chart(api_key, display_price, news_df, table_data, poly_cent, cot_summary, scen, current_weights)
                st.markdown(report.split("```json")[0])
                forecast_df = parse_ai_chart_data(report)
                if not forecast_df.empty:
                    st.divider(); st.subheader("🤖 AI Gelecek Projeksiyonu")
                    fig_ai = go.Figure()
                    short_hist = df_hist.tail(45)
                    fig_ai.add_trace(go.Scatter(x=short_hist.index, y=short_hist['Pamuk'], name='Gerçekleşen', line=dict(color='black', width=2)))
                    fig_ai.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['price'], name='AI Tahmini', mode='lines+markers+text', text=forecast_df['price'], line=dict(color='#10B981', width=3, dash='dot')))
                    fig_ai.update_layout(template="plotly_white", height=450)
                    st.plotly_chart(fig_ai, use_container_width=True)

    elif menu == "📉 Teknik":
        st.subheader("Teknik Göstergeler")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        show_fib = col_opt1.checkbox("Fibonacci", value=True); show_bb = col_opt2.checkbox("Bollinger", value=True); show_rsi = col_opt3.checkbox("RSI", value=True)
        fig_tech = go.Figure()
        if show_bb:
            fig_tech.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Upper'], name='BB Üst', line=dict(color='gray', width=1, dash='dot')))
            fig_tech.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Lower'], name='BB Alt', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'))
        fig_tech.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Pamuk'], name='Fiyat', line=dict(color='black', width=2)))
        if show_fib:
            max_p, min_p = df_hist['Pamuk'].max(), df_hist['Pamuk'].min(); diff = max_p - min_p
            for r, c in [(0.0, "red"), (0.5, "blue"), (1.0, "black")]: fig_tech.add_hline(y=max_p - r * diff, line_dash="dash", line_color=c, annotation_text=f"Fib {r}")
        fig_tech.update_layout(template="plotly_white", height=500); st.plotly_chart(fig_tech, use_container_width=True)
        if show_rsi:
            fig_rsi = go.Figure(); fig_rsi.add_trace(go.Scatter(x=df_hist.index, y=df_hist['RSI'], name='RSI', line=dict(color='#8B5CF6')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=200, template="plotly_white", title="RSI Momentum", margin=dict(t=30,b=10)); st.plotly_chart(fig_rsi, use_container_width=True)
