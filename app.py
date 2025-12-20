import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io
import streamlit.components.v1 as components

# --- IMPORT POUR LES TAGS MP3 ---
try:
    from mutagen.id3 import ID3, TKEY
    from mutagen.mp3 import MP3
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V4.7 Double Témoin", page_icon="🎧", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    .value-secondary { font-size: 0.9em; font-weight: 600; color: #E67E22; margin-top: 2px; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR AUDIO JS ---
def get_sine_witness(note_str, key_suffix=""):
    note = note_str.split(' ')[0]
    unique_id = f"playBtn_{note}_{key_suffix}"
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 26px; height: 26px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 10px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #666;">{note}</span>
    </div>
    <script>
    const freqs = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    let audioCtx = null; let oscillator = null; let gainNode = null;
    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (this.innerText === '▶') {{
            oscillator = audioCtx.createOscillator(); gainNode = audioCtx.createGain();
            oscillator.type = 'sine'; oscillator.frequency.setValueAtTime(freqs['{note}'], audioCtx.currentTime);
            gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
            oscillator.connect(gainNode); gainNode.connect(audioCtx.destination);
            oscillator.start(); this.innerText = '◼'; this.style.background = '#E74C3C';
        }} else {{
            oscillator.stop(); this.innerText = '▶'; this.style.background = '#6366F1';
        }}
    }};
    </script>
    """, height=35)

# --- MAPPING CAMELOT ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']: return BASE_CAMELOT_MINOR.get(key, "??")
        else: return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- FONCTION EXPORT TAGS ---
def get_tagged_file(file_buffer, camelot_key):
    if not MUTAGEN_AVAILABLE: return file_buffer
    try:
        file_buffer.seek(0)
        audio_data = io.BytesIO(file_buffer.read())
        audio = MP3(audio_data)
        if audio.tags is None: audio.add_tags()
        audio.tags.add(TKEY(encoding=3, text=camelot_key))
        output = io.BytesIO()
        audio.save(output)
        output.seek(0)
        return output
    except: return file_buffer

# --- MOTEUR ANALYSE ---
def check_drum_alignment(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return flatness < 0.045 or np.mean(np.max(chroma, axis=0)) > 0.75

def analyze_segment(y, sr):
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score: best_score, res_key = score, f"{NOTES[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse DJ Pro...")
def get_full_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    is_aligned = check_drum_alignment(y, sr)
    y_final = y if is_aligned else librosa.effects.hpss(y)[0]
    duration = librosa.get_duration(y=y_final, sr=sr)
    timeline_data, votes, all_chromas = [], [], []
    for start_t in range(0, int(duration) - 10, 10):
        y_seg = y_final[int(start_t*sr):int((start_t+10)*sr)]
        key_seg, score_seg, chroma_vec = analyze_segment(y_seg, sr)
        votes.append(key_seg)
        all_chromas.append(chroma_vec)
        timeline_data.append({"Temps": start_t, "Note": key_seg, "Confiance": round(score_seg * 100, 1)})
    
    dom = Counter(votes).most_common(1)[0][0]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return {"vote": dom, "timeline": timeline_data, "tempo": int(float(tempo)), "energy": int(np.mean(librosa.feature.rms(y=y))*100)}

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V4.7 ULTRA TAGGER</h1>", unsafe_allow_html=True)
tabs = st.tabs(["📁 ANALYSEUR PRO", "🕒 HISTORIQUE & EXPORT"])

with tabs[0]:
    files = st.file_uploader("Importer des tracks", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)
    if files:
        for file in files:
            with st.expander(f"🎵 {file.name}", expanded=True):
                res = get_full_analysis(file)
                df_t = pd.DataFrame(res['timeline'])
                df_s = df_t.sort_values(by="Confiance", ascending=False).reset_index()
                best_n = df_s.loc[0, 'Note']
                sec_n = df_s[df_s['Note'] != best_n].iloc[0]['Note'] if not df_s[df_s['Note'] != best_n].empty else best_n
                
                cam_best = get_camelot_pro(best_n)
                cam_sec = get_camelot_pro(sec_n)
                
                # --- HISTORIQUE ---
                if not any(h['Fichier'] == file.name for h in st.session_state.history):
                    st.session_state.history.insert(0, {"Date": datetime.now().strftime("%H:%M"), "Fichier": file.name, "Note": best_n, "Camelot": cam_best, "BPM": res['tempo']})

                st.audio(file)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">DOMINANTE</div><div class="value-custom">{res["vote"]}</div><div>{get_camelot_pro(res["vote"])}</div></div>', unsafe_allow_html=True)
                    get_sine_witness(res["vote"], "dom")
                with c2:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">BPM</div><div class="value-custom">{res["tempo"]}</div></div>', unsafe_allow_html=True)
                    # Bouton Export Tag
                    tagged_audio = get_tagged_file(file, cam_best)
                    st.download_button(label="💾 TAG & DOWNLOAD", data=tagged_audio, file_name=f"[{cam_best}] {file.name}", mime="audio/mpeg")
                
                with c3:
                    # AFFICHAGE CAMELOT DANS CONFIANCE
                    st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #F1C40F;"><div class="label-custom">TOP CONFIANCE</div><div style="font-size:0.8em;">🥇 {best_n} <b>({cam_best})</b></div><div style="font-size:0.8em;">🥈 {sec_n} <b>({cam_sec})</b></div></div>', unsafe_allow_html=True)
                    ct1, ct2 = st.columns(2)
                    with ct1: get_sine_witness(best_n, "b")
                    with ct2: get_sine_witness(sec_n, "s")
                
                with c4:
                    st.markdown(f'<div class="metric-container"><div class="label-custom">ENERGY</div><div class="value-custom">{res["energy"]}</div></div>', unsafe_allow_html=True)

                st.plotly_chart(px.scatter(df_t, x="Temps", y="Note", color="Confiance", size="Confiance", template="plotly_white"), use_container_width=True)

with tabs[1]:
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        
        c_down, c_clear = st.columns(2)
        with c_down:
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("📥 TÉLÉCHARGER HISTORIQUE (CSV)", csv, "historique_ricardo_dj.csv", "text/csv")
        with c_clear:
            if st.button("🗑 EFFACER TOUT"):
                st.session_state.history = []
                st.rerun()
