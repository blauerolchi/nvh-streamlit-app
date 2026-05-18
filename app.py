import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wav

# --------------------------------------------------
# UI Setup
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
# WAV Generator
# --------------------------------------------------
def generate_wav_bytes(signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        signal_normalized = signal
    else:
        signal_normalized = np.int16(signal / max_val * 32767 * 0.8)

    byte_io = io.BytesIO()
    wav.write(byte_io, SAMPLE_RATE, signal_normalized)
    return byte_io.getvalue()

# --------------------------------------------------
# 🔊 3x BEEP TRIGGER
# --------------------------------------------------
def create_beep_trigger(freq=2000):
    beep_duration = 0.05
    pause_duration = 0.05

    t = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration), endpoint=False)

    beep = 0.9 * np.sin(2 * np.pi * freq * t)

    # Fenster gegen Klicks
    window = np.hanning(len(beep))
    beep = beep * window

    pause = np.zeros(int(SAMPLE_RATE * pause_duration))

    trigger = np.concatenate([
        beep, pause,
        beep, pause,
        beep
    ])

    return trigger

# --------------------------------------------------
# Signalzusammenbau
# --------------------------------------------------
def assemble_signal(main_signal):

    pause = np.zeros(int(0.3 * SAMPLE_RATE))

    start_trigger = create_beep_trigger(2000)
    end_trigger = create_beep_trigger(3000)

    full_signal = np.concatenate([
        start_trigger,
        pause,
        main_signal,
        pause,
        end_trigger
    ])

    return full_signal

# --------------------------------------------------
# UI Anzeige
# --------------------------------------------------
st.subheader(test_scenario)

# ==================================================
# V2a Sweep
# ==================================================
if test_scenario == "V2a: Frequenz-Sweep":

    st.write("Logarithmischer Sweep mit Start-/End-Trigger (3x Beep)")

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

    st.write("Sinuston mit variabler Frequenz + Trigger")

    freq = st.slider("Frequenz (Hz)", 10, 20000, 550)
    duration = st.slider("Dauer (Sekunden)", 5, 60, 45)

    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    envelope = np.linspace(1.0, 0.0, len(t))

    signal = np.sin(2 * np.pi * freq * t) * envelope

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Messsysteme vorher starten!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")
        st.success(f"Frequenz: {freq} Hz")

# ==================================================
# V3 Clipping-Test (FIXED VERSION)
# ==================================================
elif test_scenario == "V3: Clipping-Test":

    st.write("Impuls-Serie mit einstellbarer Intensität (Burst-basiert)")

    num_pulses = st.slider("Anzahl Impulse", 1, 20, 5)
    pulse_spacing = st.slider("Abstand zwischen Impulsen (ms)", 10, 1000, 200)
    amplitude = st.slider("Impulsstärke", 0.1, 2.0, 1.0)

    burst_freq = 2000
    burst_duration = 0.02

    t_burst = np.linspace(0, burst_duration, int(SAMPLE_RATE * burst_duration), endpoint=False)

    single_pulse = amplitude * np.sin(2 * np.pi * burst_freq * t_burst)

    window = np.hanning(len(single_pulse))
    single_pulse = single_pulse * window

    samples_spacing = int(SAMPLE_RATE * (pulse_spacing / 1000))

    signal = []

    for i in range(num_pulses):
        signal.append(single_pulse)

        if i < num_pulses - 1:
            signal.append(np.zeros(samples_spacing))

    signal = np.concatenate(signal)

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Messsysteme vorher starten!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")
        st.success(f"{num_pulses} Impulse | Intensität: {amplitude}")

# ==================================================
# V4 Rauschen
# ==================================================
elif test_scenario == "V4: Weißes Rauschen":

    st.write("Weißes Rauschen mit Triggern")

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
    timestamp = st.number_input("Signalverlust (Sekunde)", 0.0, 120.0, 0.0, step=0.1)

bemerkung = st.text_area("Notizen:")

if st.button("Protokoll speichern"):
    if system != "Bitte wählen...":
        st.success("✅ Eintrag gespeichert")
    else:
        st.error("Bitte System wählen")

