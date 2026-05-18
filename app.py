import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="NVH Signal Lab", layout="wide")

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------
SAMPLE_RATE = 44100

SCENARIOS = {
    "V2a": "Frequenz-Sweep",
    "V2b": "Pegel-Stufentest",
    "V3":  "Clipping / Noise Burst",
    "V4":  "Weißes Rauschen"
}

# --------------------------------------------------
# AUDIO
# --------------------------------------------------
def generate_wav_bytes(signal):
    signal = np.clip(signal, -1.0, 1.0)
    signal_int16 = np.int16(signal * 32767)
    byte_io = io.BytesIO()
    wav.write(byte_io, SAMPLE_RATE, signal_int16)
    return byte_io.getvalue()

def create_beep_trigger(freq):
    t = np.linspace(0, 0.05, int(SAMPLE_RATE * 0.05), endpoint=False)
    beep = 0.9 * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    pause = np.zeros(int(SAMPLE_RATE * 0.05))
    return np.concatenate([beep, pause, beep, pause, beep])

def assemble_signal(main_signal):
    pause = np.zeros(int(0.3 * SAMPLE_RATE))
    return np.concatenate([
        create_beep_trigger(2000),
        pause,
        main_signal,
        pause,
        create_beep_trigger(3000)
    ])

# --------------------------------------------------
# METRICS
# --------------------------------------------------
def compute_metrics(signal):
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    crest = peak / rms if rms > 0 else 0

    return {
        "duration": len(signal) / SAMPLE_RATE,
        "rms": rms,
        "peak": peak,
        "crest_db": 20 * np.log10(crest) if crest > 0 else -120,
        "db_rms": 20 * np.log10(rms) if rms > 0 else -120,
        "db_peak": 20 * np.log10(peak) if peak > 0 else -120
    }

# --------------------------------------------------
# PLOT
# --------------------------------------------------
def make_plot(signal):
    fig = plt.figure(figsize=(10, 6))

    t = np.linspace(0, len(signal) / SAMPLE_RATE, len(signal))

    plt.subplot(2,1,1)
    plt.plot(t, signal)
    plt.title("Zeitverlauf")

    # FFT
    fft = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1/SAMPLE_RATE)

    plt.subplot(2,1,2)
    if len(freqs) > 1:
        plt.semilogx(freqs[1:], 20*np.log10(np.maximum(fft[1:], 1e-12)))
    plt.title("Spektrum")

    plt.tight_layout()
    return fig

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
scenario = st.sidebar.radio("Szenario", list(SCENARIOS.keys()))

st.title("🔬 NVH Signal Lab")
st.subheader(f"{scenario} – {SCENARIOS[scenario]}")

# --------------------------------------------------
# PARAMETER / SIGNAL
# --------------------------------------------------
signal = None

# V2a
if scenario == "V2a":
    duration = st.slider("Dauer (s)", 10, 120, 60)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))

    f_start = 0.5
    f_end = SAMPLE_RATE/2 * 0.95

    L = duration / np.log(f_end / f_start)
    phase = 2*np.pi*f_start*L*(np.exp(t/L)-1)

    signal = np.sin(phase)

# V2b
elif scenario == "V2b":
    freq = st.slider("Frequenz (Hz)", 10, 20000, 550)
    duration = st.slider("Dauer (s)", 5, 60, 45)

    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    env = np.linspace(1, 0, len(t))

    signal = np.sin(2*np.pi*freq*t)*env

# V3
elif scenario == "V3":
    pulses = st.slider("Impulse", 1, 20, 5)
    spacing = st.slider("Abstand (ms)", 10, 1000, 200)
    amp = st.slider("Amplitude", 0.2, 2.0, 1.2)
    dur = st.slider("Burst (ms)", 5, 100, 50)

    s_burst = int(SAMPLE_RATE*dur/1000)
    s_space = int(SAMPLE_RATE*spacing/1000)

    base = np.random.uniform(-1,1,s_burst)
    pulse = amp * base * np.hanning(s_burst)

    sig = []
    for i in range(pulses):
        sig.append(pulse)
        if i < pulses-1:
            sig.append(np.zeros(s_space))
    signal = np.concatenate(sig)

# V4
elif scenario == "V4":
    duration = st.slider("Dauer (s)", 5, 60, 15)
    signal = np.random.uniform(-1,1,int(SAMPLE_RATE*duration))

# --------------------------------------------------
# OUTPUT
# --------------------------------------------------
if signal is not None:

    full = assemble_signal(signal)

    metrics = compute_metrics(full)

    st.write("### Kennwerte")
    st.write(metrics)

    fig = make_plot(full)
    st.pyplot(fig)

    if st.button("▶ Abspielen"):
        wav_bytes = generate_wav_bytes(full)
        st.audio(wav_bytes, format="audio/wav")

        st.download_button(
            "WAV Download",
            data=wav_bytes,
            file_name="signal.wav",
            mime="audio/wav"
        )
