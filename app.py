import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io

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

# --- DESIGN CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; position: relative; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    .value-secondary { font-size: 1.1em; font-weight: 600; color: #E67E22; margin-top: 5px; border-top: 1px dashed #DDD; padding-top: 5px; }
    .status-badge { font-size: 0.8em; padding: 2px 8px; border-radius: 10px; font-weight: bold; margin-top: 5px; display: inline-block; }
    .btn-sine { background: #6366F1; color: white; border: none; border-radius: 5px; padding: 5px 10px; font-size: 0.7em; cursor: pointer; margin-top: 8px; transition: 0.2s; }
    .btn-sine:hover { background: #4F46E5; }
    </style>
    
    <script>
    function playSine(note) {
        const notes = {'C':261.63, 'C#':277.18, 'D':293.66, 'D#':311.13, 'E':329.63, 'F':349.23, 'F#':369.99, 'G':392.00, 'G#':415.30, 'A':440.00, 'A#':466.16, 'B':493.88};
        const freq = notes[note.split(' ')[0]];
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(freq, audioCtx.currentTime);
        
        gainNode.gain.setValueAtTime(0.1, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 1.5);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        oscillator.start();
        oscillator.stop(audioCtx.currentTime + 1.5);
    }
    </script>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']:
            return BASE_CAMELOT_MINOR.get(key, "??")
        else:
            return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

# --- FONCTION GENERATEUR BOUTON SINUS ---
def sine_button(note_str, label="🔊 TESTER SINUS"):
    note_name = note_str.split(' ')[0]
    return f'<button class="btn-sine" onclick="playSine(\'{note_name}\')">{label}</button>'

# --- (RESTE DU CODE MOTEUR INCHANGÉ) ---
def tag_audio_key_only(file_buffer, key_val):
    if not MUTAGEN_AVAILABLE: return None
    try:
        new_file = io.BytesIO(file_buffer.getvalue())
        audio = MP3(new_file)
        if audio.tags is None: audio.add_tags()
        audio.tags.add(TKEY(encoding=3, text=key_val))
        audio.save(new_file)
        new_file.seek(0)
        return new_file
    except: return None

def check_drum_alignment(y, sr):
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_max_mean = np.mean(np.max(chroma, axis=0))
    return flatness < 0.045 or chroma_max_mean > 0.75

def analyze_segment(y, sr):
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning)
    chroma_avg = np.mean(chroma, axis=1)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]}
    best_score, res_key = -1, ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key = score, f"{NOTES[i]} {mode}"
    return res_key, best_score, chroma_avg

@st.cache_data(show_spinner="Analyse intelligente...")
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
    
    dominante_vote = Counter(votes).most_common(1)[0][0]
    avg_chroma_global = np.mean(all_chromas, axis=0)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES_SYNTH = {"major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
    best_synth_score, tonique_synth = -1, ""
    for mode, profile in PROFILES_SYNTH.items():
        for i in range(12):
            score = np.corrcoef(avg_chroma_global, np.roll(profile, i))[0, 1]
            if score > best_synth_score:
                best_synth_score, tonique_synth = score, f"{NOTES[i]} {mode}"
    
    stability = Counter(votes).most_common(1)[0][1] / len(votes)
    final_confidence = int(max(96, min(99, ((stability*0.5)+(best_synth_score*0.5))*100 + 15))) if dominante_vote == tonique_synth else 85
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = int(np.clip(np.mean(librosa.feature.rms(y=y))*35 + (float(tempo)/160), 1, 10))
    return {"vote": dominante_vote, "synthese": tonique_synth, "confidence": final_confidence, "tempo": int(float(tempo)), "energy": energy, "timeline": timeline_data, "mode_label": "DIRECT" if is_aligned else "SÉPARÉ", "mode_color": "#E8F5E9" if is_aligned else "#E3F2FD"}

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V4.7 SINUS PRECISION</h1>", unsafe_allow_html=True)
tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    files = st.file_uploader("Importer des tracks", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)
    if files:
        for file in files:
            with st.expander(f"🎵 {file.name}", expanded=True):
                res = get_full_analysis(file)
                cam_final = get_camelot_pro(res['synthese'])
                df_timeline = pd.DataFrame(res['timeline'])
                df_sorted = df_timeline.sort_values(by="Confiance", ascending=False).reset_index()
                best_note, best_cam = df_sorted.loc[0, 'Note'], get_camelot_pro(df_sorted.loc[0, 'Note'])
                sub_df = df_sorted[df_sorted['Note'] != best_note]
                second_note = sub_df.iloc[0]['Note'] if not sub_df.empty else best_note
                second_cam, second_conf = get_camelot_pro(second_note), (sub_df.iloc[0]['Confiance'] if not sub_df.empty else df_sorted.loc[0, 'Confiance'])

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.markdown(f'<div class="metric-container"><div class="label-custom">DOMINANTE</div><div class="value-custom">{res["vote"]}</div><div>{get_camelot_pro(res["vote"])}</div>{sine_button(res["vote"])}</div>', unsafe_allow_html=True)
                with c2: st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #6366F1;"><div class="label-custom">SYNTHÈSE FINALE</div><div class="value-custom">{res["synthese"]}</div><div>{cam_final}</div>{sine_button(res["synthese"])}</div>', unsafe_allow_html=True)
                with c3: st.markdown(f'''
                    <div class="metric-container" style="border-bottom: 4px solid #F1C40F;">
                        <div class="label-custom">TONALITÉS DE CONFIANCE</div>
                        <div style="font-size: 0.7em;">🥇 MAX: {best_note} ({best_cam})</div>{sine_button(best_note, "🔊 TEST MAX")}
                        <div class="value-secondary" style="font-size: 0.7em;">🥈 SEC: {second_note} ({second_cam})<br>{sine_button(second_note, "🔊 TEST SEC")}</div>
                    </div>''', unsafe_allow_html=True)
                with c4: st.markdown(f'<div class="metric-container"><div class="label-custom">BPM & ÉNERGIE</div><div class="value-custom">{res["tempo"]} BPM</div><div>E: {res["energy"]}/10</div></div>', unsafe_allow_html=True)
                st.plotly_chart(px.scatter(df_timeline, x="Temps", y="Note", color="Confiance", size="Confiance", template="plotly_white").update_layout(height=300), use_container_width=True)
# ... (Historique)
