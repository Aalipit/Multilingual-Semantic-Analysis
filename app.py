import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from transformers import BertTokenizer, BertForSequenceClassification
# from duckduckgo_search import DDGS  # Commented out
# import trafilatura                 # Commented out
from gensim import corpora, models
import nltk
from nltk.stem import WordNetLemmatizer
import re
import os
import ssl
import requests
from collections import Counter
import time

# --- 1. INITIALIZATION ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError: pass
else: ssl._create_default_https_context = _create_unverified_https_context

st.set_page_config(page_title="Property Diagnostic Report", layout="wide")
nltk.download('wordnet', quiet=True)
lemmatizer = WordNetLemmatizer()

# --- 2. THEME & STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .header-card { background-color: #161b22; padding: 25px; border-radius: 20px; border: 1px solid #30363d; margin-bottom: 25px; }
    
    .status-badge { 
        background-color: #ffffff; 
        color: #333; 
        border-radius: 35px; 
        padding: 30px; 
        text-align: center; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        border: 1px solid #e1e4e8;
    }
    
    .status-title-text { color: #5B9BD5; font-weight: bold; font-size: 16px; margin-bottom: 5px; }
    .status-main-label { font-weight: 900; font-size: 70px; margin: 0; line-height: 1; display: inline-block; vertical-align: middle; }
    .warning-icon { font-size: 55px; vertical-align: middle; margin-right: 15px; }
    .critical-count-text { font-weight: bold; font-size: 18px; margin-top: 20px; color: #000; }
    .critical-val-num { color: #ff4b4b; }
    .sub-desc-text { color: #666; font-size: 14px; margin-top: 5px; }

    .topic-card {
        padding: 18px; 
        border-radius: 18px; 
        margin-bottom: 15px; 
        color: #000000 !important; 
        border: 1px solid rgba(0,0,0,0.1);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .topic-card b { color: #000000 !important; font-size: 17px; }
    .topic-card small { color: #444444 !important; font-size: 13px; font-weight: bold; }

    .priority-card {
        padding: 20px 10px;
        border-radius: 15px;
        text-align: center;
        color: white !important;
        font-weight: bold;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AI & LOGIC ---
@st.cache_resource
def load_ai():
    path = "./baguio_mbert_standalone"
    return BertTokenizer.from_pretrained(path), BertForSequenceClassification.from_pretrained(path)

tokenizer, model = load_ai()

def predict_sentiment(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text), flags=re.MULTILINE)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()

def get_aspect(text):
    text = text.lower()
    if any(w in text for w in ['tubig', 'water', 'pump', 'supply']): return "Water Reliability"
    if any(w in text for w in ['owner', 'landlord', 'reply', 'lodz']): return "Management"
    if any(w in text for w in ['ingay', 'noise', 'neighbor']): return "Noise Levels"
    return "Security"

# def intelligent_fetch(query_or_url, method):
#     results = []
#     headers = {"User-Agent": "Mozilla/5.0"}
#     try:
#         if method == "URL Scraper":
#             if "reddit.com" in query_or_url:
#                 json_url = query_or_url.split('?')[0].rstrip('/') + ".json"
#                 res = requests.get(json_url, headers=headers).json()
#                 results.append(res[0]['data']['children'][0]['data']['selftext'])
#                 for c in res[1]['data']['children']:
#                     if 'body' in c['data']: results.append(c['data']['body'])
#             else:
#                 downloaded = trafilatura.fetch_url(query_or_url)
#                 text = trafilatura.extract(downloaded)
#                 if text: results = [p for p in text.split('\n') if len(p.split()) > 10]
#         else:
#             with DDGS() as ddgs:
#                 search_results = [r for r in ddgs.text(f"{query_or_url} Baguio apartment review", max_results=4)]
#                 for r in search_results:
#                     text = trafilatura.extract(trafilatura.fetch_url(r['href']))
#                     if text: results.extend([p for p in text.split('\n') if len(p.split()) > 12])
#     except: pass
#     return [r for r in results if len(str(r)) > 15]

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("🏙️ Baguio Property Engine")
    if st.button("🔄 Reset System"):
        st.session_state.clear()
        st.rerun()
    st.markdown("---")
    # Commented out URL and Keyword options from the radio
    # method = st.radio("Acquisition Mode:", ["URL Scraper", "Global Keyword Search", "Manual Text Paste", "Upload CSV"])
    method = st.radio("Acquisition Mode:", ["Manual Text Paste", "Upload CSV"])
    
    u_input, pasted_text, uploaded_file = "", "", None
    
    # if method in ["URL Scraper", "Global Keyword Search"]: u_input = st.text_input("Apartment/Search:")
    
    if method == "Manual Text Paste": pasted_text = st.text_area("Paste Reviews Here:")
    else: uploaded_file = st.file_uploader("Upload CSV", type="csv")
    analyze_btn = st.button("🚀 Analyze")

# --- 5. EXECUTION & DASHBOARD ---
if analyze_btn:
    raw_text = []
    with st.spinner("🧠 is performing Diagnostic Assessment..."):
        if method == "Manual Text Paste": raw_text = [p.strip() for p in pasted_text.split('\n') if len(p.strip()) > 10]
        elif method == "Upload CSV" and uploaded_file: raw_text = pd.read_csv(uploaded_file)['Review_Text'].dropna().tolist()
        # else: raw_text = intelligent_fetch(u_input, method) # Commented out

        if not raw_text:
            st.error("No data found!")
            st.stop()

        data = []
        for t in raw_text:
            data.append({"Text": t, "Sentiment": predict_sentiment(t), "Aspect": get_aspect(t)})
        res_df = pd.DataFrame(data)
        neg_df = res_df[res_df['Sentiment'] == 0]
        
        # Determine Weakness Count
        aspect_metadata = {
            "Water Reliability": ["tubig", "water", "tank", "ration", "pump", "faucet", "shower"],
            "Management Behavior": ["owner", "landlord", "lodz", "reply", "deposit", "admin", "bastos"],
            "Noise Levels": ["ingay", "noisy", "kapitbahay", "riles", "sasakyan", "gabi", "tahimik"],
            "Security": ["safe", "guard", "security", "lock", "nanakawan", "gate", "cctv"]
        }

        junk_words = ['yung', 'pero', 'lang', 'naman', 'kami', 'namin', 'hindi', 'room', 'with', 'this', 'that', 'have', 'from', 'mga', 'nila']
        
        processed_tokens = []
        for n in raw_text:
            tks = [lemmatizer.lemmatize(w.lower()) for w in re.findall(r'\b\w{4,}\b', n)]
            tks = [w for w in tks if w not in junk_words]
            processed_tokens.append(tks)

        dict_lda = corpora.Dictionary(processed_tokens)
        dict_lda.filter_extremes(no_below=2)
        
        detected_weaknesses_count = 0
        for a in aspect_metadata:
            a_tot = len(res_df[res_df['Aspect'].str.contains(a[:5])])
            a_neg = len(neg_df[neg_df['Aspect'].str.contains(a[:5])])
            if a_tot > 0 and (a_neg/a_tot) > 0.20: detected_weaknesses_count += 1

        weakness_density_total = len(neg_df) / len(res_df)
        
        if weakness_density_total > 0.3:
            health = "POOR"
            h_color = "#ff4b4b"
            h_icon = "❗" 
        elif weakness_density_total > 0.1:
            health = "FAIR"
            h_color = "#ffa500"
            h_icon = "⚠️" 
        else:
            health = "GOOD"
            h_color = "#00cc96"
            h_icon = "✅" 

        tokens = [[lemmatizer.lemmatize(w.lower()) for w in n.split() if len(w)>3] for n in raw_text]
        dict_lda = corpora.Dictionary(tokens)
        lda = models.LdaModel([dict_lda.doc2bow(t) for t in tokens], num_topics=4, id2word=dict_lda, passes=5)

    # --- UI TOP HEADER ---
    st.markdown('<div class="header-card">', unsafe_allow_html=True)
    h1, h2, h3 = st.columns([1, 2.5, 1.8])
    with h1: st.image("https://cdn-icons-png.flaticon.com/512/2544/2544087.png", width=130)
    with h2:
        title = "BATCH ANALYSIS" if method == "Upload CSV" else "MANUAL NARRATIVE ANALYSIS"
        st.title(title.upper())
        st.markdown(f"📍 Location: **Baguio City**")
        st.markdown(f" 🏢 Type: **Residential**")
        st.markdown(f"📊 Total Narratives Analyzed: **{len(res_df)}**")
        st.markdown(f"📅 Report Generated: **July 14, 2026**")
    with h3:
        desc_text = "Property needs improvement in key areas." if health != "GOOD" else "Property maintains stable standards."
        st.markdown(f"""
            <div class="status-badge">
                <p class="status-title-text">Overall Property Health</p>
                <div style='display: flex; align-items: center; justify-content: center; gap: 10px;'>
                    <span style="font-size: 50px;">{h_icon}</span>
                    <h1 class="status-main-label" style="color: {h_color};">{health}</h1>
                </div>
                <p class="critical-count-text">Critical Weaknesses Detected: <span class="critical-val-num">{detected_weaknesses_count}</span></p>
                <p class="sub-desc-text">{desc_text}</p>
                <p style="font-size: 11px; color: #777; margin-top: 5px;">Weakness Density: {int(weakness_density_total*100)}%</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- MAIN ANALYTICS ---
    st.markdown("### Detailed Analytics")
    col_a, col_b, col_c = st.columns([1.2, 1.5, 1.2])

    with col_a:
        st.subheader("Sentiment Distribution")
        fig = go.Figure(data=[go.Pie(labels=['Pos', 'Neu', 'Neg'], values=res_df['Sentiment'].value_counts().sort_index(), hole=.6, marker_colors=['#00cc96','#636efb','#ef553b'])])
        fig.update_layout(template="plotly_dark", margin=dict(t=0,b=0,l=0,r=0), height=200)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("LDA Extracted Topics")
        bg_colors = ["#e1f5fe", "#e8f5e9", "#fffde7", "#f3e5f5"]
        for i, (aspect_name, keywords) in enumerate(aspect_metadata.items()):
            count = len(res_df[res_df['Aspect'] == aspect_name])
            percentage = (count / len(res_df)) * 100
            all_words_in_data = " ".join(raw_text).lower()
            found_keywords = [w for w in keywords if w in all_words_in_data]
            keywords_display = ", ".join(found_keywords[:5]) if found_keywords else "General feedback"

            st.markdown(f"""
                <div class='topic-card' style='background-color:{bg_colors[i]};'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <b>{aspect_name}</b>
                        <span style='background: white; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: bold;'>
                            {int(percentage)}% of reviews
                        </span>
                    </div>
                    <small>Keywords: {keywords_display}</small>
                </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.subheader("Weakness Density by Aspects")
        for asp in ["Water Reliability", "Management", "Noise Levels", "Security"]:
            a_tot = len(res_df[res_df['Aspect'] == (asp if asp != "Management" else "Management Behavior")])
            a_neg = len(neg_df[neg_df['Aspect'] == (asp if asp != "Management" else "Management Behavior")])
            v = a_neg/a_tot if a_tot > 0 else 0
            st.write(f"**{asp}**")
            st.progress(v)
            st.markdown(f"<p style='text-align:right;margin-top:-35px;font-size:12px;'>{int(v*100)}%</p>", unsafe_allow_html=True)

        st.subheader("Priority Distribution")
        p_cols = st.columns(2)
        p_cols[0].markdown('<div class="priority-card" style="background-color:#ff4b4b;">Priority 1<br>Utilities</div>', unsafe_allow_html=True)
        p_cols[1].markdown('<div class="priority-card" style="background-color:#ffa500;">Priority 2<br>Maintenance</div>', unsafe_allow_html=True)

    with col_c:
        st.subheader("Top Negative Tokens")
        weakness_keywords = ['wala', 'walang', 'sira', 'maingay', 'ingay', 'mahal', 'puyat', 'mabagal', 'bastos', 'trapik', 'madumi', 'dumi', 'amag', 'amoy', 'nawawala', 'issue', 'problem', 'hassle', 'leak', 'tulo', 'lamig', 'dilim', 'putol', 'kulang', 'sikip', 'layo', 'malayo', 'reklamo']

        if not neg_df.empty:
            all_w = " ".join(neg_df['Text']).lower()
            all_w = re.sub(r'[^\w\s]', '', all_w)
            words_list = all_w.split()
            ban_list = ['maganda', 'ganda', 'view', 'sana', 'para', 'kung', 'ganun', 'door', 'mga', 'nang', 'dahil', 'pero', 'lang', 'kasi', 'naman', 'talaga', 'niyo', 'kayo', 'sila', 'nila', 'kami', 'po', 'opo', 'siya', 'nito', 'dito', 'doon', 'nayan', 'lahat', 'saakin', 'place']
            
            filtered_tokens = []
            for word in words_list:
                if word in weakness_keywords: filtered_tokens.extend([word] * 5)
                elif len(word) > 3 and word not in ban_list: filtered_tokens.append(word)

            top_w = Counter(filtered_tokens).most_common(12)
            cloud_html = "<div style='background-color: white; padding: 25px; border-radius: 25px; text-align: center; border: 1px solid #e1e4e8; display: flex; flex-wrap: wrap; justify-content: center; align-items: center; min-height: 250px;'>"
            for word, count in top_w:
                color = "#ff4b4b" if word in weakness_keywords else "#883333"
                size = min(14 + (count * 0.8), 40)
                cloud_html += f"<span style='color: {color}; font-weight: 900; font-size: {size}px; margin: 12px; font-family: sans-serif;'>{word}</span>"
            cloud_html += "</div>"
            st.markdown(cloud_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Sample Rant Evidence")
        def is_verified_rant(text): return any(wk in str(text).lower() for wk in weakness_keywords)
        verified_rants = neg_df[neg_df['Text'].apply(is_verified_rant)]
        rants_to_show = verified_rants if not verified_rants.empty else neg_df

        if not rants_to_show.empty:
            rants_to_show = rants_to_show.assign(text_len=rants_to_show['Text'].str.len()).sort_values(by='text_len', ascending=False)
            for i in range(min(3, len(rants_to_show))):
                st.markdown(f'''<div style="border-left: 4px solid #ff4b4b; padding: 8px 15px; margin-bottom: 12px; background-color: #161b22; border-radius: 0 10px 10px 0;"><p style="font-style: italic; color: #ced4da; font-size: 13px; margin: 0;">"{rants_to_show["Text"].iloc[i][:120]}..."</p></div>''', unsafe_allow_html=True)
        else: st.write("No clear rants identified.")
else:
    st.info("Select an input method and click 'Analyze'.")