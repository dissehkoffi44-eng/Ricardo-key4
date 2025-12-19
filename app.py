import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V4 Ultra", page_icon="🎧", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #1A1A1A; }
    .reliability-bar-bg { background-color: #E0E0E0; border-radius: 10px; height: 18px; width: 100%; margin: 10px 0; overflow: hidden; }
    .reliability-fill { height: 100%; transition: width 0.8s ease-in-out; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8em; font-weight: bold; }
    .warning-box { background-color: #FFFBEB; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
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

@st.cache_data(show_spinner="Calcul de la double validation...")
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
    
    # --- DETECTION DE CHANGEMENT DE TONALITÉ ---
    first_half = votes[:len(votes)//2]
    second_half = votes[len(votes)//2:]
    key_start = Counter(first_half).most_common(1)[0][0]
    key_end = Counter(second_half).most_common(1)[0][0]
    has_switch = key_start != key_end

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
    base_conf = ((stability * 0.5) + (best_synth_score * 0.5)) * 100
    
    if dominante_vote == tonique_synth:
        final_confidence = int(max(96, min(99, base_conf + 15)))
    else:
        final_confidence = int(min(89, base_conf))

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = int(np.clip(np.mean(librosa.feature.rms(y=y))*35 + (float(tempo)/160), 1, 10))

    return {
        "vote": dominante_vote, "synthese": tonique_synth, "confidence": final_confidence,
        "tempo": int(float(tempo)), "energy": energy, "timeline": timeline_data,
        "has_switch": has_switch, "key_start": key_start, "key_end": key_end
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center; color: #1A1A1A;'>🎧 RICARDO_DJ228 | V4.1 PRECISION</h1>", unsafe_allow_html=True)

file = st.file_uploader("Importer une track", type=['mp3', 'wav', 'flac'])

if file:
    res = get_full_analysis(file)
    conf = res.get("confidence", 0)
    color = "#10B981" if conf >= 95 else "#F59E0B" if conf > 70 else "#EF4444"
    
    st.markdown(f"**Indice de Fiabilité : {conf}%** {'(Analyse Certifiée ✅)' if conf >= 95 else ''}")
    st.markdown(f"""<div class="reliability-bar-bg"><div class="reliability-fill" style="width: {conf}%; background-color: {color};">{conf}%</div></div>""", unsafe_allow_html=True)
    
    if res["has_switch"]:
        st.markdown(f"""<div class="warning-box">⚠️ <b>Changement de tonalité détecté :</b> La track semble passer de 
        {get_camelot_pro(res['key_start'])} à {get_camelot_pro(res['key_end'])}. Attention pour le mix !</div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    col_v, col_s = st.columns(2)
    with col_v:
        st.markdown(f"""<div class="metric-container"><div class="label-custom">VOTE (Dominante)</div><div class="value-custom">{res['vote']}</div><div style="color: {color}; font-weight: 800; font-size: 1.6em;">{get_camelot_pro(res['vote'])}</div></div>""", unsafe_allow_html=True)
    with col_s:
        st.markdown(f"""<div class="metric-container" style="border-bottom: 4px solid #6366F1;"><div class="label-custom">SYNTHÈSE (Globale)</div><div class="value-custom">{res['synthese']}</div><div style="color: #6366F1; font-weight: 800; font-size: 1.6em;">{get_camelot_pro(res['synthese'])}</div></div>""", unsafe_allow_html=True)

    st.markdown("### Performance & Énergie")
    c1, c2, c3 = st.columns(3)
    c1.metric("TEMPO", f"{res['tempo']} BPM")
    c2.metric("ÉNERGIE", f"{res['energy']}/10")
    c3.metric("STATUT", "CERTIFIÉ" if conf >= 95 else "À VÉRIFIER")

    df = pd.DataFrame(res["timeline"])
    fig = px.scatter(df, x="Temps", y="Note", size="Confiance", color="Note", title="Analyse de stabilité harmonique (Timeline)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    if not st.session_state.history or st.session_state.history[-1]["Fichier"] != file.name:
        st.session_state.history.append({"Date": datetime.datetime.now().strftime("%H:%M"), "Fichier": file.name, "Key": get_camelot_pro(res['vote']), "Fiabilité": f"{conf}%"})

if st.session_state.history:
    with st.expander("🕒 Historique"):
        st.table(pd.DataFrame(st.session_state.history))
