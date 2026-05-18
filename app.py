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
    """
    3 klare Sinus-Pieptöne für Synchronisation
    """

    beep_duration = 0.05   # 50 ms
    pause_duration = 0.05  # 50 ms

    t = np.linspace(0, beep_duration, int(SAMPLE_RATE * beep_duration), endpoint=False)

    # Sinus-Beep
    beep = 0.9 * np.sin(2 * np.pi * freq * t)

    # weiches Einschwingen (verhindert Klicks)
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

    pause = np.zeros(int(0.3 * SAMPLE_RATE))  # 300 ms Abstand

    start_trigger = create_beep_trigger(2000)   # Start
    end_trigger   = create_beep_trigger(3000)   # Ende (andere Frequenz!)

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

    t = np.linspace(0, duration, SAMPLE_RATE * duration)

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

    t = np.linspace(0, duration, SAMPLE_RATE * duration)
    envelope = np.linspace(1.0, 0.0, len(t))

    signal = np.sin(2 * np.pi * freq * t) * envelope

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Messsysteme vorher starten!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")
        st.success(f"Frequenz: {freq} Hz")

# ==================================================
# V3 Clipping / Impuls-Serie
# ==================================================
elif test_scenario == "V3: Clipping-Test":

    st.write("Serie von Impulsen zur Analyse von Clipping und Systemreaktion")

    # ✅ Anzahl Impulse
    num_pulses = st.slider("Anzahl Impulse", 1, 20, 5)

    # ✅ Abstand zwischen Impulsen
    pulse_spacing = st.slider("Abstand zwischen Impulsen (ms)", 10, 1000, 200)

    pulse_duration = 0.005  # 5 ms Impulsdauer

    samples_per_pulse = int(SAMPLE_RATE * pulse_duration)
    samples_spacing = int(SAMPLE_RATE * (pulse_spacing / 1000))

    # 👉 einzelner "Peitschenschlag" (breitbandig)
    single_pulse = np.random.uniform(-1.0, 1.0, samples_per_pulse)

    # 👉 Sequenz bauen
    signal = []

    for i in range(num_pulses):
        signal.append(single_pulse)

        # Abstand danach (außer beim letzten)
        if i < num_pulses - 1:
            signal.append(np.zeros(samples_spacing))

    signal = np.concatenate(signal)

    # 👉 Trigger hinzufügen (deine neue SynchronLogik!)
    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Messsysteme vorher starten!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")

        st.success(f"{num_pulses} Impulse erzeugt")

# ==================================================
# V4 Rauschen
# ==================================================
elif test_scenario == "V4: Weißes Rauschen":

    st.write("Weißes Rauschen mit Triggern")

    duration = st.slider("Dauer (Sekunden)", 5, 60, 15)

    signal = np.random.uniform(-1.0, 1.0, SAMPLE_RATE * duration)

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
    system = st.selectbox(
        "System:",
        ["Bitte wählen...", "Pico", "SQuadriga"]
    )

with col2:
    timestamp = st.number_input(
        "Signalverlust (Sekunde)",
        0.0, 120.0, 0.0, step=0.1
    )

bemerkung = st.text_area("Notizen:")

if st.button("Protokoll speichern"):
    if system != "Bitte wählen...":
        st.success("✅ Eintrag gespeichert")
    else:
        st.error("Bitte System wählen")
