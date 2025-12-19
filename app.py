import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V3 Ultra", page_icon="🎧", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #1A1A1A; }
    .reliability-bar-bg { background-color: #EEE; border-radius: 10px; height: 15px; width: 100%; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT (F#m = 11A) ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- MOTEUR D'ANALYSE ---
def analyze_segment(y, sr):
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    profile_minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    best_score, res_key = -1, ""
    for i in range(12):
        score = np.corrcoef(chroma_avg, np.roll(profile_minor, i))[0, 1]
        if score > best_score:
            best_score, res_key = score, f"{NOTES[i]} minor"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse ultra-précise en cours...")
def get_full_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    duration = librosa.get_duration(y=y, sr=sr)
    timeline_data, votes, all_chromas = [], [], []
    
    for start_t in range(0, int(duration) - 15, 10):
        y_seg = y[int(start_t*sr):int((start_t+15)*sr)]
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr)
        votes.append(key_seg)
        all_chromas.append(chroma_vec)
        timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": score_seg})
    
    dominante_vote = Counter(votes).most_common(1)[0][0]
    avg_chroma_global = np.mean(all_chromas, axis=0)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    profile_minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    best_synth_score, tonique_synth = -1, ""
    for i in range(12):
        score = np.corrcoef(avg_chroma_global, np.roll(profile_minor, i))[0, 1]
        if score > best_synth_score:
            best_synth_score, tonique_synth = score, f"{NOTES[i]} minor"

    stability = Counter(votes).most_common(1)[0][1] / len(votes)
    # Harmonisation du nom de la clé en 'confidence'
    confiance_globale = int(((stability * 0.6) + (best_synth_score * 0.4)) * 100)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = int(np.clip(np.mean(librosa.feature.rms(y=y))*35 + (float(tempo)/160), 1, 10))

    return {
        "vote": dominante_vote,
        "synthese": tonique_synth,
        "confidence": confiance_globale, # Correction ici
        "tempo": int(float(tempo)),
        "energy": energy,
        "timeline": timeline_data
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>RICARDO_DJ228 | ANALYSEUR V3 ULTRA</h1>", unsafe_allow_html=True)

file = st.file_uploader("Importer une track audio", type=['mp3', 'wav', 'flac'])

if file:
    res = get_full_analysis(file)
    
    # Récupération sécurisée de la confiance
    conf = res.get("confidence", 0)
    color = "#28A745" if conf > 80 else "#FFA500" if conf > 60 else "#FF4B4B"
    
    st.markdown(f"**Fiabilité de l'analyse : {conf}%**")
    st.markdown(f"""<div class="reliability-bar-bg"><div style="background-color: {color}; width: {conf}%; height: 15px; border-radius: 10px;"></div></div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_v, col_s = st.columns(2)
    with col_v:
        st.markdown(f"""<div class="metric-container">
            <div class="label-custom">VOTE (Dominante)</div>
            <div class="value-custom">{res['vote']}</div>
            <div style="color: {color}; font-weight: bold; font-size: 1.5em;">{get_camelot_pro(res['vote'])}</div>
        </div>""", unsafe_allow_html=True)
        
    with col_s:
        st.markdown(f"""<div class="metric-container" style="border-left: 5px solid #6366F1;">
            <div class="label-custom">SYNTHÈSE (Globale)</div>
            <div class="value-custom">{res['synthese']}</div>
            <div style="color: #6366F1; font-weight: bold; font-size: 1.5em;">{get_camelot_pro(res['synthese'])}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Détails Techniques")
    c1, c2, c3 = st.columns(3)
    c1.metric("TEMPO", f"{res['tempo']} BPM")
    c2.metric("ÉNERGIE", f"{res['energy']}/10")
    c3.metric("ÉCOSYSTÈME", "Stable" if res['vote'] == res['synthese'] else "Complexe")

    df = pd.DataFrame(res["timeline"])
    fig = px.scatter(df, x="Temps", y="Note", size="Confiance", color="Note", title="Stabilité Harmonique Temporelle")
    st.plotly_chart(fig, use_container_width=True)

    # Sauvegarde dans l'historique
    if not st.session_state.history or st.session_state.history[-1]["Fichier"] != file.name:
        st.session_state.history.append({
            "Fichier": file.name,
            "Vote": get_camelot_pro(res['vote']),
            "Synth": get_camelot_pro(res['synthese']),
            "BPM": res['tempo'],
            "Fiabilité": f"{conf}%"
        })

if st.session_state.history:
    with st.expander("🕒 Historique de la session"):
        st.table(pd.DataFrame(st.session_state.history))
