import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wav

st.set_page_config(page_title="NVH Signal Generator", layout="centered")
st.title("🎛️ NVH Hardware-Validierung")
st.write("Signalgenerator mit synchronisierbaren Triggern für NVH-Messsysteme")

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
# 🔥 TRIGGER GENERATOR
# --------------------------------------------------
def create_trigger():
    """
    Erzeugt einen klaren, breitbandigen Impuls:
    - eindeutig detektierbar
    - robust gegenüber Filtern
    """
    trigger_length = int(0.01 * SAMPLE_RATE)  # 10 ms

    # kurzer Breitband-Impuls
    trigger = np.zeros(trigger_length)
    trigger[:int(trigger_length/4)] = 1.0
    trigger[int(trigger_length/4):int(trigger_length/2)] = -1.0

    return trigger


def assemble_signal(main_signal):
    """
    Kombiniert:
    [START TRIGGER] + [PAUSE] + [SIGNAL] + [PAUSE] + [END TRIGGER]
    """
    pause = np.zeros(int(0.2 * SAMPLE_RATE))  # 200 ms Pause

    start_trigger = create_trigger()
    end_trigger = create_trigger()

    full_signal = np.concatenate([
        start_trigger,
        pause,
        main_signal,
        pause,
        end_trigger
    ])

    return full_signal


st.subheader(test_scenario)

# ==================================================
# V2a Sweep
# ==================================================
if test_scenario == "V2a: Frequenz-Sweep":

    st.write("Logarithmischer Sweep mit Start-/End-Trigger")

    duration = st.slider("Sweepdauer (Sekunden)", 10, 120, 60)

    t = np.linspace(0, duration, SAMPLE_RATE * duration)

    f_start = 0.5
    f_end = SAMPLE_RATE / 2 * 0.95

    L = duration / np.log(f_end / f_start)
    phase = 2 * np.pi * f_start * L * (np.exp(t / L) - 1)

    signal = np.sin(phase)

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Beide Messsysteme müssen bereits aufnehmen!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")


# ==================================================
# V2b Pegeltest
# ==================================================
elif test_scenario == "V2b: Pegel-Stufentest":

    st.write("Sinuston mit Triggern zur Synchronisierung")

    freq = st.slider("Frequenz (Hz)", 10, 20000, 550)
    duration = st.slider("Dauer (Sekunden)", 5, 60, 45)

    t = np.linspace(0, duration, SAMPLE_RATE * duration)
    amplitude_envelope = np.linspace(1.0, 0.0, len(t))

    signal = np.sin(2 * np.pi * freq * t) * amplitude_envelope

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.warning("⚠️ Messsysteme müssen vorab scharf geschaltet sein!")
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")
        st.success(f"Frequenz: {freq} Hz")


# ==================================================
# V3 Clipping
# ==================================================
elif test_scenario == "V3: Clipping-Test":

    st.write("Clipping-Impuls mit Triggern")

    duration = 0.05
    signal = np.random.uniform(-1.0, 1.0, int(SAMPLE_RATE * duration))

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")


# ==================================================
# V4 Rauschen
# ==================================================
elif test_scenario == "V4: Weißes Rauschen":

    st.write("Weißes Rauschen mit Synchron-Triggern")

    duration = st.slider("Dauer (Sekunden)", 5, 60, 15)

    signal = np.random.uniform(-1.0, 1.0, SAMPLE_RATE * duration)

    full_signal = assemble_signal(signal)

    if st.button("▶️ Test starten"):
        st.audio(generate_wav_bytes(full_signal), format="audio/wav")


# ==================================================
# Messprotokoll
# ==================================================
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
