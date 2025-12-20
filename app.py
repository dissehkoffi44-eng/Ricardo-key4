import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io
import streamlit.components.v1 as components

# --- NOUVEL IMPORT POUR LES TAGS ---
try:
    from mutagen.id3 import ID3, TKEY
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V4.7 Pro Hybrid", page_icon="🎧", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN CSS & JS ENGINE ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    .value-secondary { font-size: 1.1em; font-weight: 600; color: #E67E22; margin-top: 5px; border-top: 1px dashed #DDD; padding-top: 5px; }
    .btn-sine { 
        background-color: #6366F1; color: white !important; border: none; 
        padding: 6px 12px; border-radius: 20px; cursor: pointer; 
        font-size: 11px; font-weight: bold; margin-top: 5px; width: 100%;
    }
    .btn-sine:active { transform: scale(0.98); background-color: #4F46E5; }
    audio { height: 30px; width: 100%; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Moteur Audio JavaScript (Sinus Pur)
JS_AUDIO = """
<script>
const freqMap = {'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88};
let audioCtx = null;

function playPureSine(noteName) {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freqMap[noteName.split(' ')[0]], audioCtx.currentTime);
    
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 2.0);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 2.0);
}
</script>
"""
components.html(JS_AUDIO, height=0)

# --- MAPPING CAMELOT (F# MINOR = 11A) ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- MOTEUR ANALYSE ORIGINAL ---
def check_drum_alignment(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return flatness < 0.045 or np.mean(np.max(chroma, axis=0)) > 0.75

def analyze_segment(y, sr):
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma_avg = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning), axis=1)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score: best_score, res_key = score, f"{NOTES[i]} {mode}"
    return res_key, best_score

@st.cache_data
def get_full_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    is_aligned = check_drum_alignment(y, sr)
    y_final = y if is_aligned else librosa.effects.hpss(y)[0]
    duration = librosa.get_duration(y=y_final, sr=sr)
    votes, timeline = [], []
    for start_t in range(0, int(duration) - 10, 10):
        key_seg, score_seg = analyze_segment(y_final[int(start_t*sr):int((start_t+10)*sr)], sr)
        votes.append(key_seg)
        timeline.append({"Temps": start_t, "Note": key_seg, "Confiance": round(score_seg * 100, 1)})
    
    dom = Counter(votes).most_common(1)[0][0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {"vote": dom, "confidence": 98, "tempo": int(float(tempo)), "timeline": timeline, "mode": "DIRECT" if is_aligned else "SÉPARÉ"}

# --- INTERFACE ---
st.title("🎧 RICARDO_DJ228 | V4.7 COMPARATEUR PRO")
tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    uploaded_files = st.file_uploader("Tracks", type=['mp3', 'wav'], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            with st.expander(f"🎵 {file.name}", expanded=True):
                res = get_full_analysis(file)
                cam = get_camelot_pro(res['vote'])
                
                # Sauvegarde historique
                if not any(h['Fichier'] == file.name for h in st.session_state.history):
                    st.session_state.history.append({"Date": datetime.now().strftime("%H:%M"), "Fichier": file.name, "Note": res['vote'], "Camelot": cam})

                # --- COMPARATEUR AUDIO ---
                st.write("**Comparaison en temps réel :**")
                st.audio(file) # Lecteur du son original
                
                c1, c2, c3 = st.columns(3)
                
                def sine_btn(note, label):
                    return f'<button class="btn-sine" onclick="playPureSine(\'{note}\')">🔊 {label}</button>'

                with c1:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">ANALYSE</div><div class="value-custom">{res["vote"]}</div><div>{cam}</div>{sine_btn(res["vote"], "JOUER SINUS")}</div>', unsafe_allow_html=True)
                
                with c2:
                    df_t = pd.DataFrame(res['timeline'])
                    best_n = df_t.sort_values(by="Confiance", ascending=False).iloc[0]['Note']
                    st.markdown(f'<div class="metric-container"><div class="label-custom">STABILITÉ MAX</div><div class="value-custom">{best_n}</div><div>{get_camelot_pro(best_n)}</div>{sine_btn(best_n, "JOUER SINUS")}</div>', unsafe_allow_html=True)
                
                with c3:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">BPM</div><div class="value-custom">{res["tempo"]}</div></div>', unsafe_allow_html=True)

                st.plotly_chart(px.scatter(pd.DataFrame(res['timeline']), x="Temps", y="Note", color="Confiance", template="plotly_white"), use_container_width=True)

with tabs[1]:
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
        if st.button("Effacer"): st.session_state.history = []; st.rerun()
