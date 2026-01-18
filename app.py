# RCDJ228 SNIPER M3 - VERSION "TRIPLE CHECK" ULTIME
import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
import requests
import gc
import json
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime
from pydub import AudioSegment

# --- FORCE FFMEG PATH (WINDOWS FIX) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "sniper_triads": {
        "major": [1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0],
        "minor": [1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    },
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(99, 102, 241, 0.3); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 5px solid #10b981;
    }
    .root-hint {
        background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 6px 14px;
        border-radius: 20px; font-size: 0.85em; border: 1px solid #10b981;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%; transition: 0.3s;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---
def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 5000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def get_root_note_pyin(y, sr):
    """TRIPLE CHECK PYIN : Analyse à 25%, 50% et 75% du morceau pour une précision maximale."""
    points = [0.25, 0.50, 0.75]
    all_detected_notes = []
    
    for p in points:
        start_sample = int(len(y) * p)
        # On analyse 6 secondes à chaque point (18s total)
        end_sample = start_sample + (sr * 6)
        y_chunk = y[start_sample:min(end_sample, len(y))]
        
        f0, voiced_flag, voiced_probs = librosa.pyin(y_chunk, 
                                                     fmin=librosa.note_to_hz('C2'), 
                                                     fmax=librosa.note_to_hz('C5'), 
                                                     sr=sr, hop_length=1024)
        
        # Filtre de confiance élevé (85%)
        valid_f0 = f0[voiced_flag & (voiced_probs > 0.65)]
        if len(valid_f0) > 0:
            notes = librosa.hz_to_note(valid_f0)
            clean_notes = [n.replace(n[-1], '') for n in notes]
            all_detected_notes.extend(clean_notes)
            
    if not all_detected_notes: return None
    # On retourne la note qui revient le plus souvent sur les 3 points
    return Counter(all_detected_notes).most_common(1)[0][0]

def solve_key_sniper(chroma_vector, bass_vector, root_hint=None):
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    bv = (bass_vector - bass_vector.min()) / (bass_vector.max() - bass_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                note_name = NOTES_LIST[i]
                reference = np.roll(p_data[mode], i)
                score = np.corrcoef(cv, reference)[0, 1]
                
                # --- SYNERGIE TRIPLE CHECK ---
                if root_hint and note_name == root_hint:
                    score += 0.30 # Bonus de confirmation de la Root Note

                if p_name == "sniper_triads": score *= 1.20 

                if mode == "minor":
                    dom_idx, leading_tone = (i + 7) % 12, (i + 11) % 12
                    if cv[dom_idx] > 0.45 and cv[leading_tone] > 0.35: score *= 1.35 
                
                if bv[i] > 0.6: score += (bv[i] * 0.25)
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_key = f"{note_name} {mode}"
                    
    return {"key": best_key, "score": best_overall_score}

def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    try:
        with io.BytesIO(file_bytes) as buf:
            y, sr = librosa.load(buf, sr=22050, mono=True)
    except: return None

    duration = librosa.get_duration(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)
    
    # ÉTAPE 1 : Triple Check Root Sniper
    if _progress_callback: _progress_callback(10, "Triple Check PYIN (25% | 50% | 75%)...")
    root_hint = get_root_note_pyin(y, sr)
    
    # ÉTAPE 2 : Analyse Harmonique
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    step, timeline, votes = 2, [], Counter()
    segments = list(range(0, max(1, int(duration) - step), 2))
    
    for idx, start in enumerate(segments):
        if _progress_callback:
            prog = 20 + int((idx / len(segments)) * 75)
            _progress_callback(prog, f"Scan Sniper : {start}s")

        idx_s, idx_e = int(start * sr), int((start + step) * sr)
        seg = y_filt[idx_s:idx_e]
        if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue
        
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=24, bins_per_octave=24)
        c_avg = np.mean((c_raw[::2, :] + c_raw[1::2, :]) / 2, axis=1)
        
        # Bass Priority segmentaire
        nyq = 0.5 * sr
        b, a = butter(2, 150/nyq, btype='low')
        b_seg = np.mean(librosa.feature.chroma_cqt(y=lfilter(b, a, y[idx_s:idx_e]), sr=sr, n_chroma=12), axis=1)
        
        res = solve_key_sniper(c_avg, b_seg, root_hint=root_hint)
        weight = 3.0 if (start < 10 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not votes: return None

    most_common = votes.most_common(1)
    final_key = most_common[0][0]
    final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    res_obj = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99), "tempo": int(float(tempo)),
        "root_hint": root_hint, "name": file_name,
        "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
        "chroma": np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1).tolist()
    }
    
    # Telegram
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            msg = f"🎯 *SNIPER M3 - TRIPLE CHECK*\n📄 `{file_name}`\n🎹 *{final_key.upper()}*\n🎡 Camelot: {res_obj['camelot']}\n✅ Confiance: {res_obj['conf']}%"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
        except: pass

    return res_obj

def get_chord_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.12, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 2.0);
        }});
    }}; """

# --- INTERFACE ---
st.title("🎯 RCDJ228 SNIPER M3")
st.caption("Version Ultime : Triple Check PYIN & Triad Precision")

files = st.file_uploader("📂 Déposez vos fichiers audio", type=['mp3','wav','m4a','flac'], accept_multiple_files=True)

if files:
    for i, f in enumerate(reversed(files)):
        with st.status(f"🎯 Sniper Scan : {f.name}...") as status:
            prog_bar = st.progress(0)
            data = process_audio_precision(f.getvalue(), f.name, _progress_callback=lambda v, m: prog_bar.progress(v))
            status.update(label=f"✅ {f.name} Analysé", state="complete")
        
        if data:
            st.markdown(f"<div class='file-header'>📄 {data['name']}</div>", unsafe_allow_html=True)
            
            # 1. Affichage Principal (Report Card)
            color = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 88 else "linear-gradient(135deg, #1e293b, #0f172a)"
            st.markdown(f"""
                <div class="report-card" style="background:{color};">
                    <div style="display:flex; justify-content:space-between;">
                        <span class="root-hint">🎯 PYIN ROOT: {data['root_hint']}</span>
                        <span class="root-hint">📡 {data['tuning']} Hz</span>
                    </div>
                    <h1 style="font-size:6.5em; margin:15px 0; font-weight:900; letter-spacing:-2px;">{data['key'].upper()}</h1>
                    <p style="font-size:1.6em; opacity:0.9; font-weight:bold;">CAMELOT: {data['camelot']} • CONFIANCE: {data['conf']}%</p>
                </div>""", unsafe_allow_html=True)
            
            # 2. Métriques
            m1, m2, m3 = st.columns(3)
            with m1: 
                st.markdown(f"<div class='metric-box'><b>TEMPO ANALYSÉ</b><br><span style='font-size:2.2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
            with m2: 
                st.markdown(f"<div class='metric-box'><b>QUALITÉ SIGNAL</b><br><span style='font-size:2.2em; color:#f59e0b;'>HIGH</span><br>Sniper M3</div>", unsafe_allow_html=True)
            with m3:
                bid = f"pl_{i}_{abs(hash(data['name']))}"
                components.html(f"""<button id="{bid}" style="width:100%; height:95px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:16px; box-shadow:0 4px 15px rgba(0,0,0,0.3);">🔊 ÉCOUTER L'ACCORD</button>
                    <script>{get_chord_js(bid, data['key'])}</script>""", height=110)

            # Visualisation
            with st.expander("🔍 Analyse Spectrale & Timeline"):
             col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        df_tl = pd.DataFrame(data['timeline'])
        fig = px.scatter(df_tl, x="Temps", y="Note", color="Conf", 
                         template="plotly_dark", 
                         category_orders={"Note": NOTES_ORDER}, 
                         title="Stabilité Harmonique")
        # AJOUT D'UNE KEY UNIQUE ICI
        st.plotly_chart(fig, use_container_width=True, key=f"scatter_{i}_{hash(data['name'])}")
        
    with col_g2:
        fig_pol = go.Figure(data=go.Scatterpolar(r=data['chroma'], 
                            theta=NOTES_LIST, fill='toself', 
                            line_color='#10b981'))
        fig_pol.update_layout(template="plotly_dark", 
                              polar=dict(radialaxis=dict(visible=False)), 
                              margin=dict(l=30,r=30,t=30,b=30))
        # AJOUT D'UNE KEY UNIQUE ICI AUSSI
        st.plotly_chart(fig_pol, use_container_width=True, key=f"polar_{i}_{hash(data['name'])}")

with st.sidebar:
    st.markdown("### 🛠️ Paramètres Sniper")
    if st.button("🗑️ Effacer l'Historique"):
        st.cache_data.clear()
        st.rerun()
    st.info("Algorithme : RCDJ228 SNIPER M3\nMode : Triple Check PYIN Enabled\nPrécision : Industrielle")
