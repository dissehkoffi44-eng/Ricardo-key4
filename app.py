import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V3 Ultra", page_icon="🎧", layout="wide")

# --- INITIALISATION DE L'HISTORIQUE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    h1 { font-family: 'Segoe UI', sans-serif; color: #1A1A1A; text-align: center; font-weight: 800; border-bottom: 2px solid #FF4B4B; padding-bottom: 10px; }
    .stMetric { background-color: #FFFFFF !important; border: 1px solid #E0E0E0 !important; border-radius: 12px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .recommendation-box { padding: 15px; border-radius: 10px; border-left: 5px solid #6366F1; background-color: #EEF2FF; color: #4338CA; margin: 10px 0; }
    .alert-box { padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; background-color: #FFEBEB; color: #B30000; font-weight: bold; }
    .success-box { padding: 15px; border-radius: 10px; border-left: 5px solid #28A745; background-color: #E8F5E9; color: #1B5E20; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT (F#m=11A réglé selon vos instructions) ---
BASE_CAMELOT_MINOR = {
    'Ab': '1A', 'G#': '1A', 'Eb': '2A', 'D#': '2A', 'Bb': '3A', 'A#': '3A',
    'F': '4A', 'C': '5A', 'G': '6A', 'D': '7A', 'A': '8A', 'E': '9A',
    'B': '10A', 'Cb': '10A', 'F#': '11A', 'Gb': '11A', 'Db': '12A', 'C#': '12A'
}
BASE_CAMELOT_MAJOR = {
    'B': '1B', 'Cb': '1B', 'F#': '2B', 'Gb': '2B', 'Db': '3B', 'C#': '3B',
    'Ab': '4B', 'G#': '4B', 'Eb': '5B', 'D#': '5B', 'Bb': '6B', 'A#': '6B',
    'F': '7B', 'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B'
}

# --- FONCTIONS TECHNIQUES ---

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']:
            return BASE_CAMELOT_MINOR.get(key, "??")
        return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def analyze_segment(y, sr):
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_harm, _ = librosa.effects.hpss(y, margin=(3.0, 1.0))
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, fmin=librosa.note_to_hz('C2'))
    chroma_avg = np.mean(chroma, axis=1)
    
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]
    }
    
    best_s, res_k, res_m = -1, "", ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_s:
                best_s, res_k, res_m = score, NOTES[i], mode
    return f"{res_k} {res_m}", best_s

@st.cache_data(show_spinner="Analyse multidimensionnelle en cours...")
def get_full_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 1. Tempo & Rythme (Plus précis)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    pulse_clarity = np.mean(librosa.beat.plp(y=y, sr=sr)) # Danceability
    
    # 2. Énergie & Structure
    rms = librosa.feature.rms(y=y)[0]
    energy = int(np.clip((np.mean(rms) * 35) + (float(tempo) / 150), 1, 10))
    
    # Détection des pics d'énergie (Segments / Drops)
    segments = librosa.onset.onset_detect(onset_envelope=rms, sr=sr, units='time', hop_length=512, backtrack=True)
    
    # 3. Analyse Harmonique temporelle
    timeline_data = []
    votes = []
    for start_t in range(0, int(duration) - 15, 10):
        seg, score = analyze_segment(y[int(start_t*sr):int((start_t+15)*sr)], sr)
        if score > 0.45:
            votes.append(seg)
            timeline_data.append({"Temps": start_t, "Note_Mode": seg, "Confiance": score})
            
    dominante = Counter(votes).most_common(1)[0][0] if votes else "Inconnue"
    
    return {
        "dominante": dominante,
        "timeline": timeline_data,
        "tempo": int(float(tempo)),
        "energy": energy,
        "danceability": int(pulse_clarity * 100),
        "duration": duration,
        "rms_profile": rms[::100].tolist() # Pour affichage graphique simplifié
    }

# --- LOGIQUE DE RECOMMANDATION (MIXING) ---
def get_mix_suggestions(current_camelot):
    if not st.session_state.history: return None
    
    val = int(current_camelot[:-1])
    letter = current_camelot[-1]
    
    compatibles = [
        f"{val}{letter}", # Même clé
        f"{(val % 12) + 1}{letter}", # +1
        f"{(val - 2) % 12 + 1}{letter}", # -1
        f"{val}{'B' if letter == 'A' else 'A'}" # Changement Major/Minor
    ]
    
    suggestions = [item for item in st.session_state.history[:-1] if item['Camelot'] in compatibles]
    return suggestions

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 | PRECISION V3 ULTRA</h1>", unsafe_allow_html=True)

file = st.file_uploader("Importer un fichier audio (MP3, WAV, FLAC)", type=['mp3', 'wav', 'flac'])

if file:
    res = get_full_analysis(file)
    
    # Synthèse de la clé
    note_weights = {}
    for d in res["timeline"]:
        n = d["Note_Mode"]
        note_weights[n] = note_weights.get(n, 0) + d["Confiance"]
    
    tonique_synth = max(note_weights, key=note_weights.get) if note_weights else res["dominante"]
    camelot = get_camelot_pro(tonique_synth)
    
    # Mise à jour historique
    history_entry = {
        "Heure": datetime.datetime.now().strftime("%H:%M:%S"),
        "Fichier": file.name,
        "Key": tonique_synth,
        "Camelot": camelot,
        "BPM": res['tempo'],
        "Energie": res['energy']
    }
    if not st.session_state.history or st.session_state.history[-1]["Fichier"] != file.name:
        st.session_state.history.append(history_entry)

    # --- AFFICHAGE MÉTRIQUES PRINCIPALES ---
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("KEY (Camelot)", camelot)
    m_col2.metric("TEMPO (BPM)", res['tempo'])
    m_col3.metric("ÉNERGIE", f"{res['energy']}/10")
    m_col4.metric("DANCEABILITY", f"{res['danceability']}%")

    # --- ALERTES ET CONSEILS DE MIX ---
    c_alert, c_mix = st.columns([1, 1])
    
    with c_alert:
        if res["dominante"] != tonique_synth:
            st.markdown(f'<div class="alert-box">⚠️ COMPLEXITÉ : Changements de clés détectés. Dominante: {res["dominante"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ STABILITÉ : Key confirmée sur toute la durée.</div>', unsafe_allow_html=True)
            
    with c_mix:
        mixes = get_mix_suggestions(camelot)
        if mixes:
            st.markdown(f'<div class="recommendation-box">💡 **Suggestion Mix :** Enchaînez avec "{mixes[-1]["Fichier"]}" ({mixes[-1]["Camelot"]})</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="recommendation-box">💡 Analysez d\'autres morceaux pour voir les suggestions de mix.</div>', unsafe_allow_html=True)

    # --- GRAPHIQUES AVANCÉS ---
    tab1, tab2 = st.tabs(["📊 Nuage Harmonique", "📈 Profil d'Énergie"])
    
    with tab1:
        df = pd.DataFrame(res["timeline"])
        fig = px.scatter(df, x="Temps", y="Note_Mode", size="Confiance", color="Note_Mode", 
                         title="Stabilité Harmonique Temporelle", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # Visualisation de la forme d'onde d'énergie
        st.line_chart(res["rms_profile"])
        st.caption("Évolution de l'amplitude (Volume RMS) sur la durée du morceau.")

# --- SECTION HISTORIQUE ---
if st.session_state.history:
    with st.expander("🕒 Consulter l'Historique de Session", expanded=False):
        df_hist = pd.DataFrame(st.session_state.history)
        st.table(df_hist)
        
        c1, c2 = st.columns(2)
        csv = df_hist.to_csv(index=False).encode('utf-8')
        c1.download_button("📂 Exporter l'historique (CSV)", data=csv, file_name="historique_ricardo.csv", mime="text/csv")
        if c2.button("🗑️ Effacer la session"):
            st.session_state.history = []
            st.rerun()

st.sidebar.markdown("### Ricardo_DJ228 Settings")
st.sidebar.info(f"Standard de Clé : Camelot System\nRéférence : F#m = 11A")
