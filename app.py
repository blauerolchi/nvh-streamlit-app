import streamlit as stimport streamlitimport numpy as np
import io
import scipy.io.wavfile as wav

st.set_page_config(page_title="NVH Signal Generator", layout="centered")
st.title("🎛️ NVH Hardware-Validierung")
st.write("Kontrollierte Signalquelle für isolierte Hardware-Tests.")

# ✅ Einheitliche Szenarionamen (Problem gelöst)
SCENARIOS = [
    "V2a: Frequenz-Sweep",
    "V2b: Pegel-Stufentest",
    "V3: Clipping-Test (Impuls)",
    "V4: Weißes Rauschen"
]

test_scenario = st.sidebar.radio("Wähle ein Messszenario:", SCENARIOS)

# Konstanten für die Audio-Generierung
SAMPLE_RATE = 44100

def generate_wav_bytes(signal):
    """Konvertiert ein NumPy-Array in WAV"""
    # Schutz gegen Division durch 0
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        signal_normalized = signal
    else:
        signal_normalized = np.int16(signal / max_val * 32767 * 0.8)

    byte_io = io.BytesIO()
    wav.write(byte_io, SAMPLE_RATE, signal_normalized)
    return byte_io.getvalue()

st.subheader(test_scenario)

# --------------------------------------------------
# V2a – Sweep
# --------------------------------------------------
if test_scenario == "V2a: Frequenz-Sweep":
    st.write("Extremer logarithmischer Sweep von 0.5 Hz bis ~22 kHz")

    duration = st.slider("Dauer (Sekunden)", 10, 120, 60)

    t = np.linspace(0, duration, SAMPLE_RATE * duration)

    f_start = 0.5
    f_end = SAMPLE_RATE / 2 * 0.95

    L = duration / np.log(f_end / f_start)
    phase = 2 * np.pi * f_start * L * (np.exp(t / L) - 1)
    signal = np.sin(phase)

    st.audio(generate_wav_bytes(signal), format="audio/wav")

# --------------------------------------------------
# ✅ V2b – FIXED VERSION
# --------------------------------------------------
elif test_scenario == "V2b: Pegel-Stufentest":

    st.write("Sinuston mit frei wählbarer Frequenz und Pegelabfall")

    # ✅ Frequenzsteuerung
    freq = st.slider(
        "Frequenz (Hz)",
        min_value=10,
        max_value=20000,
        value=550,
        step=10
    )

    # ✅ Dauer einstellbar (optional)
    duration = st.slider("Dauer (Sekunden)", 5, 60, 45)

    t = np.linspace(0, duration, SAMPLE_RATE * duration)

    # ✅ Pegelverlauf
    amplitude_envelope = np.linspace(1.0, 0.0, len(t))

    signal = np.sin(2 * np.pi * freq * t) * amplitude_envelope

    # ✅ Audio IMMER anzeigen
    audio_bytes = generate_wav_bytes(signal)
    st.audio(audio_bytes, format="audio/wav")

    # ✅ Anzeige
    st.success(f"Aktuelle Frequenz: {freq} Hz")

# --------------------------------------------------
# V3 – Impuls
# --------------------------------------------------
elif test_scenario == "V3: Clipping-Test (Impuls)":
    st.write("Kurzimpuls zur Clipping-Analyse")

    duration = 0.05
    signal = np.random.uniform(-1.0, 1.0, int(SAMPLE_RATE * duration))

    st.audio(generate_wav_bytes(signal), format="audio/wav")

# --------------------------------------------------
# V4 – Rauschen
# --------------------------------------------------
elif test_scenario == "V4: Weißes Rauschen":
    st.write("Breitbandiges weißes Rauschen")

    duration = st.slider("Dauer (Sekunden)", 5, 60, 15)

    signal = np.random.uniform(-1.0, 1.0, SAMPLE_RATE * duration)

    st.audio(generate_wav_bytes(signal), format="audio/wav")

# --------------------------------------------------
# Messprotokoll
# --------------------------------------------------
st.markdown("---")
st.subheader("📝 Digitales Messprotokoll")

col1, col2 = st.columns(2)

with col1:
    system = st.selectbox(
        "Getestetes System:",
        ["Bitte wählen...", "Pico-Tool", "SQuadriga II"]
    )

with col2:
    timestamp = st.number_input(
        "Signalverlust bei Sekunde (V2b):",
        min_value=0.0,
        max_value=60.0,
        step=0.1
    )

bemerkung = st.text_area("Besondere Auffälligkeiten / Notizen:")

if st.button("Protokolleintrag speichern"):
    if system != "Bitte wählen...":
        st.success(f"Daten für {system} gespeichert ✅")
    else:
        st.error("Bitte System auswählen")
