import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import feedparser
import google.generativeai as genai
import requests
import re
import json
from deep_translator import GoogleTranslator
from textblob import TextBlob
from datetime import datetime, timedelta

# --- 1. AYARLAR (YENİ MARKA) ---
st.set_page_config(page_title="Cotton Geni's", page_icon="☁️", layout="wide")

# --- 2. TEK VE ANA VERİ MOTORU (TABLO + FİYAT) ---
@st.cache_data(ttl=60)
def get_futures_table_hybrid():
    # PLAN A: TradingCharts Sitesini Kazı
    url = "https://futures.tradingcharts.com/futures/quotes/ct.html?cbase=ct"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.google.com/"
    }
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
                final_df = raw_df.dropna(subset=['Last_Clean'])
                final_df = final_df.rename(columns={'Last_Clean': 'Son', 'Chg_Clean': 'Değişim', 'Month': 'Vade', 'Vol': 'Hacim'})
                
                cash_row = final_df[final_df['Vade'].astype(str).str.contains("Cash", case=False, na=False)]
                futures_df = final_df[~final_df['Vade'].astype(str).str.contains("Cash", case=False, na=False)]
                
                if not cash_row.empty:
                    combined = pd.concat([cash_row, futures_df])
                    return combined[['Vade', 'Son', 'Değişim', 'Hacim']].reset_index(drop=True), "TradingCharts (Spot)"
                elif not futures_df.empty:
                    return futures_df[['Vade', 'Son', 'Değişim', 'Hacim']].reset_index(drop=True), "TradingCharts (Vadeli)"
    except Exception as e: print(f"Plan A Hatası: {e}")

    # PLAN B: Yahoo Finance (Elle Oluşturma)
    month_map = {3: 'H', 5: 'K', 7: 'N', 12: 'Z'} 
    curr_date = datetime.now()
    rows = []
    for i in range(24): 
        future_date = curr_date + timedelta(days=30*i)
        m, y = future_date.month, future_date.year
        if m in month_map:
            sym = f"CT{month_map[m]}{str(y)[-2:]}.NYB"
            name = f"{future_date.strftime('%b')}'{str(y)[-2:]}"
            if y > curr_date.year or (y == curr_date.year and m >= curr_date.month):
                try:
                    t = yf.Ticker(sym)
                    hist = t.history(period="5d")
                    if not hist.empty:
                        last = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else last
                        vol = hist['Volume'].iloc[-1]
                        rows.append({"Vade": name, "Son": float(last), "Değişim": float(last - prev), "Hacim": int(vol)})
                except: continue
                if len(rows) >= 8: break
    if rows: return pd.DataFrame(rows), "Borsa (Yedek)"
    return pd.DataFrame(), "Veri Yok"

@st.cache_data(ttl=60)
def get_market_history(period_str):
    mapping = {"3 Ay": "3mo", "6 Ay": "6mo", "1 Yıl": "1y", "3 Yıl": "3y"}
    try:
        data = yf.download("CT=F BZ=F DX-Y.NYB CNY=X", period=mapping[period_str], group_by='ticker', progress=False)
        df = pd.DataFrame()
        if 'CT=F' in data: df['Pamuk'] = data['CT=F']['Close']
        if 'BZ=F' in data: df['Petrol'] = data['BZ=F']['Close']
        if 'DX-Y.NYB' in data: df['DXY'] = data['DX-Y.NYB']['Close']
        if 'CNY=X' in data: df['USDCNY'] = data['CNY=X']['Close']
        return df.dropna()
    except: return pd.DataFrame()

# --- 3. İNDİKATÖRLER ---
def calculate_indicators(df):
    delta = df['Pamuk'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['SMA20'] = df['Pamuk'].rolling(20).mean()
    df['Upper'] = df['SMA20'] + (df['Pamuk'].rolling(20).std() * 2)
    df['Lower'] = df['SMA20'] - (df['Pamuk'].rolling(20).std() * 2)
    return df

# --- 4. HABERLER ---
@st.cache_data(ttl=900)
def get_intel_news():
    queries = ['"cotton futures"', '"USDA" cotton export sales', '"H&M" sales', "polyester fiber price"]
    whitelist = ['cotton', 'pamuk', 'textile', 'h&m', 'inditex', 'polyester', 'price', 'usda', 'export']
    news_data = []
    translator = GoogleTranslator(source='auto', target='tr')
    current_year = datetime.now().year
    try:
        for q in queries:
            url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                if not any(w in entry.title.lower() for w in whitelist): continue
                try: 
                    if entry.published_parsed.tm_year < current_year - 1: continue 
                except: pass
                blob = TextBlob(entry.title)
                sent = "🟢" if blob.sentiment.polarity > 0.1 else "🔴" if blob.sentiment.polarity < -0.1 else "⚪"
                try: tr = translator.translate(entry.title)
                except: tr = entry.title
                news_data.append({"Duygu": sent, "Başlık": tr, "Link": entry.link, "Orjinal": entry.title})
                if len(news_data) >= 10: break
            if len(news_data) >= 10: break
    except: pass
    return pd.DataFrame(news_data)

# --- 5. AI ANALİST ---
def ask_gemini_with_chart(api_key, spot_val, news_df, table_df, poly_cent, cot_summary, scenario):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        model_name = next((m for m in models if 'flash' in m), models[0] if models else None)
        model = genai.GenerativeModel(model_name)
        today_str = datetime.now().strftime("%Y-%m-%d")
        table_txt = table_df.to_string(index=False) if not table_df.empty else "Veri Yok"
        prompt = f"""
        Sen Profesyonel Pamuk Tüccarısın. BUGÜN: {today_str} | EN YAKIN FİYAT: {spot_val:.2f}c
        RAKİP: Polyester {poly_cent:.2f}c | COT: {cot_summary}
        TABLO: {table_txt} | HABERLER: {news_df['Orjinal'].to_string() if not news_df.empty else "Yok"}
        SENARYO: {scenario}
        GÖREV: Tablo verilerine dayanarak fiyat yönünü belirle ve 1 yıllık 3 tahmin noktası (JSON) oluştur.
        ÇIKTI FORMATI:
        ## 🧭 Stratejik Analiz
        * [Yorum...]
        ## 🦊 Pozisyonlar
        * [Yorum...]
        ```json
        {{
            "forecast": [
                {{"label": "Bugün", "date": "{today_str}", "price": {spot_val}}},
                {{"label": "Kısa Vade", "date": "YYYY-MM-DD", "price": 00.00}},
                {{"label": "Orta Vade", "date": "YYYY-MM-DD", "price": 00.00}},
                {{"label": "Uzun Vade", "date": "YYYY-MM-DD", "price": 00.00}}
            ]
        }}
        ```
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e: return f"Hata: {str(e)}"

def parse_ai_chart_data(text):
    try:
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if match: return pd.DataFrame(json.loads(match.group(1))['forecast'])
    except: pass
    return pd.DataFrame()

# --- ARAYÜZ (YENİ MARKA) ---
if "GEMINI_API_KEY" in st.secrets: api_key = st.secrets["GEMINI_API_KEY"]
else: api_key = st.sidebar.text_input("API Anahtarı", type="password")

with st.sidebar:
    st.title("☁️ Cotton Geni's")
    menu = st.radio("MENÜ", ["📊 Ana Ekran", "🦊 Pozisyonlar (COT)", "🤖 AI Strateji", "📉 Teknik"])
    st.divider()
    poly_rmb = st.number_input("Polyester (RMB/Ton)", value=6587)
    st.divider()
    period = st.selectbox("Periyot", ["3 Ay", "6 Ay", "1 Yıl", "3 Yıl"], index=1)

# VERİ HAZIRLIĞI
table_data, table_source = get_futures_table_hybrid()
if not table_data.empty:
    front_price = float(table_data.iloc[0]['Son'])
    front_change = float(table_data.iloc[0]['Değişim'])
    front_name = str(table_data.iloc[0]['Vade'])
else:
    front_price, front_change, front_name = 0.0, 0.0, "Veri Yok"

df_hist = get_market_history(period)
news_df = get_intel_news()

if df_hist.empty: st.error("Veri bağlantısı kurulamadı."); st.stop()
else: df_hist = calculate_indicators(df_hist)

usdcny = float(df_hist['USDCNY'].iloc[-1]) if 'USDCNY' in df_hist else 7.2
poly_cent = (poly_rmb / usdcny / 2204.62) * 100

if 'cot_data' not in st.session_state: st.session_state['cot_data'] = {"cl": 15000, "cs": 65000, "fl": 40000, "fs": 20000}
cot = st.session_state['cot_data']
fix_ratio = (cot['cs'] / (cot['cl']+cot['cs']+cot['fl']+cot['fs'])) * 100
cot_summary = f"Ticari Short: {cot['cs']}, Fon Long: {cot['fl']}, Fix: {fix_ratio:.1f}%"
if front_price == 0: front_price = df_hist['Pamuk'].iloc[-1]; front_change = front_price - df_hist['Pamuk'].iloc[-2]; front_name = "Yedek Veri"

# --- EKRANLAR ---
if menu == "📊 Ana Ekran":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Pamuk ({front_name})", f"{front_price:.2f}c", f"{front_change:.2f}")
    c2.metric("Petrol", f"${df_hist['Petrol'].iloc[-1]:.2f}", f"{df_hist['Petrol'].iloc[-1]-df_hist['Petrol'].iloc[-2]:.2f}")
    c3.metric("Sentetik", f"{poly_cent:.2f}c", "Piyasa")
    c4.metric("DXY", f"{df_hist['DXY'].iloc[-1]:.2f}", f"{df_hist['DXY'].iloc[-1]-df_hist['DXY'].iloc[-2]:.2f}")
    st.divider()
    st.subheader("📈 Fiyat Grafiği")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Pamuk'], name='Pamuk', line=dict(color='#0052cc', width=3)))
    fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Petrol'], name='Petrol', line=dict(color='#e74c3c', width=2, dash='dot'), yaxis='y2'))
    fig.update_layout(height=450, margin=dict(l=20,r=20,t=20,b=20), yaxis2=dict(overlaying='y', side='right', showgrid=False), legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)
    st.divider()
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader(f"📋 Piyasa Derinliği ({table_source})")
        if not table_data.empty: st.dataframe(table_data, column_config={"Vade": st.column_config.TextColumn("Vade", width="small"), "Son": st.column_config.NumberColumn("Son", format="%.2f"), "Değişim": st.column_config.NumberColumn("Değişim", format="%.2f"), "Hacim": st.column_config.NumberColumn("Hacim", format="%d")}, hide_index=True, use_container_width=True)
        else: st.warning("Veri bekleniyor...")
    with col_r:
        st.subheader("📡 İstihbarat Masası")
        if not news_df.empty: st.dataframe(news_df[['Duygu', 'Başlık', 'Link']], column_config={"Link": st.column_config.LinkColumn("Kaynak", display_text="🔗"), "Duygu": st.column_config.TextColumn("Yön", width="small"), "Başlık": st.column_config.TextColumn("Haber", width="medium")}, hide_index=True, use_container_width=True)

elif menu == "🦊 Pozisyonlar (COT)":
    st.subheader("Piyasa Oyuncu Dağılımı")
    c1, c2 = st.columns([1, 2])
    with c1:
        new_cl = st.number_input("Ticari Long", value=cot['cl']); new_cs = st.number_input("Ticari Short (Fix)", value=cot['cs'])
        new_fl = st.number_input("Fon Long", value=cot['fl']); new_fs = st.number_input("Fon Short", value=cot['fs'])
        st.session_state['cot_data'] = {"cl": new_cl, "cs": new_cs, "fl": new_fl, "fs": new_fs}
    with c2:
        fig_pie = go.Figure(data=[go.Pie(labels=['Fix', 'Tic. Long', 'Fon Long', 'Fon Short'], values=[new_cs, new_cl, new_fl, new_fs], hole=.4)])
        st.plotly_chart(fig_pie, use_container_width=True)

elif menu == "🤖 AI Strateji":
    st.subheader("Yapay Zeka Projeksiyonu")
    scen = st.text_area("Senaryo:", placeholder="Hedge oranını artırmalı mıyım?")
    if st.button("Analiz Et") and api_key:
        with st.spinner("AI Fiyat Grafiği Oluşturuyor..."):
            report = ask_gemini_with_chart(api_key, front_price, news_df, table_data, poly_cent, cot_summary, scen)
            st.markdown(report.split("```json")[0])
            forecast_df = parse_ai_chart_data(report)
            if not forecast_df.empty:
                st.divider(); st.subheader("🤖 AI Gelecek Tahmini")
                fig_ai = go.Figure()
                short_hist = df_hist.tail(30)
                fig_ai.add_trace(go.Scatter(x=short_hist.index, y=short_hist['Pamuk'], name='Gerçekleşen', line=dict(color='black', width=2)))
                fig_ai.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['price'], name='Tahmin', mode='lines+markers+text', text=forecast_df['price'], line=dict(color='green', width=3, dash='dot')))
                st.plotly_chart(fig_ai, use_container_width=True)

elif menu == "📉 Teknik":
    st.subheader("Teknik Analiz")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    show_fib = col_opt1.checkbox("Fibonacci", value=True); show_bb = col_opt2.checkbox("Bollinger", value=True); show_rsi = col_opt3.checkbox("RSI", value=True)
    fig_tech = go.Figure()
    if show_bb:
        fig_tech.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Upper'], name='BB Üst', line=dict(color='gray', width=1, dash='dot')))
        fig_tech.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Lower'], name='BB Alt', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.2)'))
    fig_tech.add_trace(go.Scatter(x=df_hist.index, y=df_hist['Pamuk'], name='Fiyat', line=dict(color='black', width=2)))
    if show_fib:
        max_p, min_p = df_hist['Pamuk'].max(), df_hist['Pamuk'].min(); diff = max_p - min_p
        for r, c in [(0.0, "red"), (0.5, "blue"), (1.0, "black")]: fig_tech.add_hline(y=max_p - r * diff, line_dash="dash", line_color=c)
    st.plotly_chart(fig_tech, use_container_width=True)
    if show_rsi:
        fig_rsi = go.Figure(); fig_rsi.add_trace(go.Scatter(x=df_hist.index, y=df_hist['RSI'], name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(height=250, title="RSI", yaxis=dict(range=[0, 100])); st.plotly_chart(fig_rsi, use_container_width=True)
