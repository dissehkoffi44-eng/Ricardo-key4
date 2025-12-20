import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io
import streamlit.components.v1 as components

# --- IMPORT POUR LES TAGS MP3 (MUTAGEN) ---
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

# --- DESIGN CSS ORIGINAL ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.9em; font-weight: bold; margin-bottom: 5px; }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #1A1A1A; }
    .value-secondary { font-size: 1.1em; font-weight: 600; color: #E67E22; margin-top: 5px; border-top: 1px dashed #DDD; padding-top: 5px; }
    .status-badge { font-size: 0.8em; padding: 2px 8px; border-radius: 10px; font-weight: bold; margin-top: 5px; display: inline-block; }
    .sine-witness { margin-top: 10px; border-top: 1px solid #EEE; padding-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- CORRECTION : MOTEUR AUDIO JS (JOUE MAINTENANT L'ACCORD) ---
def get_sine_witness(note_mode_str, key_suffix=""):
    parts = note_mode_str.split(' ')
    note = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    
    unique_id = f"playBtn_{note}_{mode}_{key_suffix}".replace("#", "sharp")
    
    return components.html(f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px; font-family: sans-serif;">
        <button id="{unique_id}" style="background: #6366F1; color: white; border: none; border-radius: 50%; width: 28px; height: 28px; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 12px;">▶</button>
        <span style="font-size: 9px; font-weight: bold; color: #666;">{note} {mode[:3].upper()}</span>
    </div>
    <script>
    const notesFreq = {{'C':261.63,'C#':277.18,'D':293.66,'D#':311.13,'E':329.63,'F':349.23,'F#':369.99,'G':392.00,'G#':415.30,'A':440.00,'A#':466.16,'B':493.88}};
    const semitones = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
    
    let audioCtx = null; 
    let oscillators = []; 
    let gainNode = null;

    document.getElementById('{unique_id}').onclick = function() {{
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        if (this.innerText === '▶') {{
            this.innerText = '◼'; this.style.background = '#E74C3C';
            gainNode = audioCtx.createGain();
            gainNode.gain.setValueAtTime(0.05, audioCtx.currentTime);
            gainNode.connect(audioCtx.destination);

            const rootIdx = semitones.indexOf('{note}');
            // Intervalles : Majeur (0, 4, 7) | Mineur (0, 3, 7)
            const intervals = ('{mode}' === 'minor' || '{mode}' === 'dorian') ? [0, 3, 7] : [0, 4, 7];
            
            intervals.forEach(interval => {{
                let osc = audioCtx.createOscillator();
                osc.type = 'sine';
                // Calcul de la fréquence pour les notes de l'accord
                let freq = notesFreq['{note}'] * Math.pow(2, interval / 12);
                osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
                osc.connect(gainNode);
                osc.start();
                oscillators.push(osc);
            }});
        }} else {{
            oscillators.forEach(o => o.stop());
            oscillators = [];
            this.innerText = '▶'; this.style.background = '#6366F1';
        }}
    }};
    </script>
    """, height=40)

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

# --- FONCTION DE TAGGING ---
def get_tagged_audio(file_buffer, key_val):
    if not MUTAGEN_AVAILABLE: return file_buffer
    try:
        file_buffer.seek(0)
        audio_data = io.BytesIO(file_buffer.read())
        audio = MP3(audio_data)
        if audio.tags is None: audio.add_tags()
        audio.tags.add(TKEY(encoding=3, text=key_val))
        output = io.BytesIO()
        audio.save(output)
        output.seek(0)
        return output
    except: return file_buffer

# --- MOTEUR ANALYSE ORIGINAL ---
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
            if score > best_score: best_score, res_key = score, f"{NOTES[i]} {mode}"
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
            if score > best_synth_score: best_synth_score, tonique_synth = score, f"{NOTES[i]} {mode}"
    
    stability = Counter(votes).most_common(1)[0][1] / len(votes)
    final_conf = int(max(96, min(99, ((stability*0.5)+(best_synth_score*0.5))*100 + 15))) if dominante_vote == tonique_synth else 89
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = int(np.clip(np.mean(librosa.feature.rms(y=y))*35 + (float(tempo)/160), 1, 10))
    return {"vote": dominante_vote, "synthese": tonique_synth, "confidence": final_conf, "tempo": int(float(tempo)), "energy": energy, "timeline": timeline_data, "mode_label": "DIRECT" if is_aligned else "SÉPARÉ", "mode_color": "#E8F5E9" if is_aligned else "#E3F2FD"}

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V4.7 DOUBLE TÉMOIN</h1>", unsafe_allow_html=True)
tabs = st.tabs(["📁 ANALYSEUR", "🕒 HISTORIQUE"])

with tabs[0]:
    files = st.file_uploader("Importer des tracks", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)
    if files:
        for file in files:
            with st.expander(f"🎵 {file.name}", expanded=True):
                res = get_full_analysis(file)
                cam_final = get_camelot_pro(res['synthese'])
                
                entry = {"Date": datetime.now().strftime("%d/%m %H:%M"), "Fichier": file.name, "Note": res['synthese'], "Camelot": cam_final, "BPM": res['tempo']}
                if not any(h['Fichier'] == file.name for h in st.session_state.history): st.session_state.history.insert(0, entry)

                st.audio(file) 

                c1, c2, c3, c4 = st.columns(4)
                with c1: 
                    st.markdown(f'<div class="metric-container"><div class="label-custom">DOMINANTE</div><div class="value-custom">{res["vote"]}</div><div>{get_camelot_pro(res["vote"])}</div></div>', unsafe_allow_html=True)
                    get_sine_witness(res["vote"], "dom")
                
                with c2: 
                    st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #6366F1;"><div class="label-custom">SYNTHÈSE</div><div class="value-custom">{res["synthese"]}</div><div>{cam_final}</div></div>', unsafe_allow_html=True)
                    get_sine_witness(res["synthese"], "synth")
                    tagged_audio = get_tagged_audio(file, cam_final)
                    st.download_button(label="💾 EXPORT TAGGED MP3", data=tagged_audio, file_name=f"[{cam_final}] {file.name}", mime="audio/mpeg")
                
                df_timeline = pd.DataFrame(res['timeline'])
                df_s = df_timeline.sort_values(by="Confiance", ascending=False).reset_index()
                best_n = df_s.loc[0, 'Note']
                sec_n = df_s[df_s['Note'] != best_n].iloc[0]['Note'] if not df_s[df_s['Note'] != best_n].empty else best_n
                
                cam_best = get_camelot_pro(best_n)
                cam_sec = get_camelot_pro(sec_n)

                with c3: 
                    st.markdown(f'<div class="metric-container" style="border-bottom: 4px solid #F1C40F;"><div class="label-custom">TOP CONFIANCE</div><div style="font-size:0.8em;">🥇 {best_n} <b>({cam_best})</b></div><div style="font-size:0.8em;">🥈 {sec_n} <b>({cam_sec})</b></div></div>', unsafe_allow_html=True)
                    col_t1, col_t2 = st.columns(2)
                    with col_t1: get_sine_witness(best_n, "best")
                    with col_t2: get_sine_witness(sec_n, "sec")
                
                with c4: 
                    st.markdown(f'<div class="metric-container"><div class="label-custom">BPM</div><div class="value-custom">{res["tempo"]}</div><div>E: {res["energy"]}</div></div>', unsafe_allow_html=True)

                st.plotly_chart(px.scatter(df_timeline, x="Temps", y="Note", color="Confiance", size="Confiance", template="plotly_white"), use_container_width=True)

with tabs[1]:
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        csv_data = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("📥 TÉLÉCHARGER HISTORIQUE (CSV)", csv_data, "historique_ricardo.csv", "text/csv")
    else: 
        st.info("Historique vide.")
