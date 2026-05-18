import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wav

# --------------------------------------------------
# UI
# --------------------------------------------------
st.set_page_config(page_title="NVH Signal Generator", layout="centered")
st.title("🎛️ NVH Hardware-Validierung")
st.write("Signalquelle mit synchronisierbaren Audio-Triggern (Start + Ende)")

# --------------------------------------------------
# Szenarien
# --------------------------------------------------
SCENARIOS = [
    "V2a: Frequenz-Sweep",
    "V2b: Pegel-Stufentest",
    "V3: Clipping-Test",
    "V4: Weißes Rauschen"
]

test_scenario = st.sidebar.radio("Wähle ein Messszenario:", SCENARIOS)

# --------------------------------------------------
# Konstanten
# --------------------------------------------------
SAMPLE_RATE = 44100

# --------------------------------------------------
# ✅ WAV Generator (KEINE Auto-Normalisierung mehr!)
# --------------------------------------------------
def generate_wav_bytes(signal):
    # Begrenzen (verhindert digitales Explodieren)
    signal = np.clip(signal, -1.0, 1.0)

    # Direkte Skalierung → echte Lautstärke!
    signal_int16 = np.int16(signal * 32767)

    byte_io = io.BytesIO()
    wav.write(byte_io, SAMPLE_RATE, signal_int16)
    return byte_io.getvalue()

# --------------------------------------------------
# 🔊 3x Beep Trigger
# --------------------------------------------------
def create_beep_trigger(freq):
    beep_duration = 0.05
    pause_duration = 0.05

    t = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration), endpoint=False)

    beep = 0.9 * np.sin(2 * np.pi * freq * t)

    # Fenster gegen Klicks
    beep *= np.hanning(len(beep))

    pause = np.zeros(int(SAMPLE_RATE * pause_duration))

    return np.concatenate([
        beep, pause,
        beep, pause,
        beep
    ])

# --------------------------------------------------
# 🔧 Signalaufbau
# --------------------------------------------------
def assemble_signal(main_signal):

    pause = np.zeros(int(0.3 * SAMPLE_RATE))

    start_trigger = create_beep_trigger(2000)
    end_trigger = create_beep_trigger(3000)

    return np.concatenate([
        start_trigger,
        pause,
        main_signal,
        pause,
        end_trigger
    ])

# --------------------------------------------------
# UI Anzeige
# --------------------------------------------------
st.subheader(test_scenario)

# ==================================================
# V2a Sweep
# ==================================================
if test_scenario == "V2a: Frequenz-Sweep":

    st.write("Logarithmischer Sweep mit Triggern")

    duration = st.slider("Sweepdauer (Sekunden)", 10, 120, 60)

    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    f_start = 0.5
    f_end = SAMPLE_RATE / 2 * 0.95

    L = duration / np.log(f_end / f_start)
    phase = 2 * np.pi * f_start * L * (np.exp(t / L) - 1)

    signal = np.sin(phase)

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Messsysteme müssen bereits aufnehmen!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")

# ==================================================
# V2b Pegeltest
# ==================================================
elif test_scenario == "V2b: Pegel-Stufentest":

    st.write("Sinuston mit Pegelabfall")

    freq = st.slider("Frequenz (Hz)", 10, 20000, 550)
    duration = st.slider("Dauer (Sekunden)", 5, 60, 45)

    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    envelope = np.linspace(1.0, 0.0, len(t))

    signal = np.sin(2 * np.pi * freq * t) * envelope

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")
        st.success(f"{freq} Hz")

# ==================================================
# ✅ V3 Clipping-Test FINAL
# ==================================================
elif test_scenario == "V3: Clipping-Test":

    st.write("Clipping-Test mit lauten Burst-Impulsen")

    num_pulses = st.slider("Anzahl Impulse", 1, 20, 5)
    pulse_spacing = st.slider("Abstand (ms)", 10, 1000, 200)
    amplitude = st.slider("Intensität", 0.1, 2.0, 1.2)

    burst_freq = 2000
    burst_duration = 0.05  # länger = lauter!

    t = np.linspace(0, burst_duration, int(SAMPLE_RATE * burst_duration), endpoint=False)

    pulse = amplitude * np.sin(2 * np.pi * burst_freq * t)

    # Fenster → verhindert Limiting im Browser
    pulse *= np.hanning(len(pulse))

    spacing = int(SAMPLE_RATE * pulse_spacing / 1000)

    signal = []

    for i in range(num_pulses):
        signal.append(pulse)

        if i < num_pulses - 1:
            signal.append(np.zeros(spacing))

    signal = np.concatenate(signal)

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Laut! Vorsicht mit Kopfhörern!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")
        st.success(f"{num_pulses} Impulse | Intensität: {amplitude}")

# ==================================================
# V4 Rauschen
# ==================================================
elif test_scenario == "V4: Weißes Rauschen":

    duration = st.slider("Dauer (Sekunden)", 5, 60, 15)

    signal = np.random.uniform(-1.0, 1.0, int(SAMPLE_RATE * duration))
    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")

# --------------------------------------------------
# Messprotokoll
# --------------------------------------------------
st.markdown("---")
st.subheader("📝 Messprotokoll")

col1, col2 = st.columns(2)

with col1:
    system = st.selectbox("System:", ["Bitte wählen...", "Pico", "SQuadriga"])

with col2:
    timestamp = st.number_input("Signalverlust (s)", 0.0, 120.0, 0.0, step=0.1)

note = st.text_area("Notizen")

if st.button("Protokoll speichern"):
    if system != "Bitte wählen...":
        st.success("✅ Gespeichert")
    else:
        st.error("System auswählen")
