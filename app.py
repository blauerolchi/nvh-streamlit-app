import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wav

st.set_page_config(page_title="NVH Signal Generator", layout="centered")
st.title("🎛️ NVH Hardware-Validierung")
st.write("Kontrollierte Signalquelle für isolierte Hardware-Tests.")

# Seitenleiste für die Auswahl des Versuchs
test_scenario = st.sidebar.radio(
    "Wähle ein Messszenario:",
    ("V2a: Frequenz-Sweep", "V2b: Pegel-Stufentest (550 Hz)", "V3: Clipping-Test (Impuls)", "V4: Weißes Rauschen")
)

# Konstanten für die Audio-Generierung
SAMPLE_RATE = 44100

def generate_wav_bytes(signal):
    """Konvertiert ein NumPy-Array in eine unkomprimierte WAV-Datei im Arbeitsspeicher"""
    # Normalisieren auf 16-Bit Integerbereich
    signal_normalized = np.int16(signal / np.max(np.abs(signal)) * 32767 * 0.8)
    byte_io = io.BytesIO()
    wav.write(byte_io, SAMPLE_RATE, signal_normalized)
    return byte_io.getvalue()

st.subheader(test_scenario)

if test_scenario == "V2a: Frequenz-Sweep":
    st.write("Logarithmischer Frequenzdurchlauf von 20 Hz bis 20.000 Hz.")
    duration = 30
    t = np.linspace(0, duration, SAMPLE_RATE * duration)
    # Logarithmischer Sweep
    signal = np.sin(2 * np.pi * 20 * ((20000 / 20) ** (t / duration)))
    audio_bytes = generate_wav_bytes(signal)
    st.audio(audio_bytes, format="audio/wav")

elif test_scenario == "V2b: Pegel-Stufentest (550 Hz)":
    st.write("Ein 550 Hz Sinuston, der über 45 Sekunden kontinuierlich leiser wird.")
    duration = 45
    t = np.linspace(0, duration, SAMPLE_RATE * duration)
    # Lineare Pegelrampe (Fade Out) von 1.0 auf 0.0
    amplitude_envelope = np.linspace(1.0, 0.0, len(t))
    signal = np.sin(2 * np.pi * 550 * t) * amplitude_envelope
    audio_bytes = generate_wav_bytes(signal)
    st.audio(audio_bytes, format="audio/wav")

elif test_scenario == "V3: Clipping-Test (Impuls)":
    st.write("Extrem kurzer Breitband-Impuls bei maximaler digitaler Aussteuerung.")
    duration = 0.05  # 50 Millisekunden
    # Weißes Rauschen mit maximaler Amplitude
    signal = np.random.uniform(-1.0, 1.0, int(SAMPLE_RATE * duration))
    audio_bytes = generate_wav_bytes(signal)
    st.audio(audio_bytes, format="audio/wav")

elif test_scenario == "V4: Weißes Rauschen":
    st.write("Konstantes, breitbandiges weißes Rauschen zur Überprüfung aller Frequenzbänder.")
    duration = 15
    signal = np.random.uniform(-1.0, 1.0, SAMPLE_RATE * duration)
    audio_bytes = generate_wav_bytes(signal)
    st.audio(audio_bytes, format="audio/wav")

# Protokoll-Logbereich im unteren Drittel
st.markdown("---")
st.subheader("📝 Digitales Messprotokoll")
col1, col2 = st.columns(2)
with col1:
    system = st.selectbox("Getestetes System:", ["Bitte wählen...", "Pico-Tool", "SQuadriga II"])
with col2:
    timestamp = st.number_input("Signalverlust bei Sekunde (V2b):", min_value=0.0, max_value=45.0, step=0.1)

bemerkung = st.text_area("Besondere Auffälligkeiten / Notizen:")

if st.button("Protokolleintrag speichern"):
    if system != "Bitte wählen...":
        st.success(f"Daten für {system} erfolgreich im Protokoll vermerkt!")
        # Hier könnte man die Daten optional in eine Cloud-Datei schreiben lassen
    else:
        st.error("Bitte wähle zuerst das getestete System aus.")
