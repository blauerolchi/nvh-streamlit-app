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
# Page Config & Theme
# --------------------------------------------------
st.set_page_config(
    page_title="NVH Signal Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS – Dark Scientific Aesthetic
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0c10;
    color: #c9d1d9;
}
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }

.nvh-header {
    display: flex; align-items: flex-start; justify-content: space-between;
    border-bottom: 1px solid #21262d; padding-bottom: 1.5rem; margin-bottom: 2rem;
}
.nvh-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem;
    font-weight: 600; color: #e6edf3; letter-spacing: -0.5px;
}
.nvh-subtitle {
    font-size: 0.78rem; color: #6e7681; font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.3rem; letter-spacing: 0.05em; text-transform: uppercase;
}
.nvh-badge {
    background: #161b22; border: 1px solid #30363d; border-radius: 6px;
    padding: 0.4rem 0.9rem; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem; color: #58a6ff; letter-spacing: 0.08em;
    display: inline-block; margin-bottom: 0.3rem;
}
.scenario-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1.4rem 1.6rem; margin-bottom: 1.5rem;
}
.scenario-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #6e7681;
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.2rem;
}
.scenario-name { font-size: 1.25rem; font-weight: 600; color: #e6edf3; }

.metric-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.8rem; margin: 1rem 0 1.5rem 0;
}
.metric-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 1rem 1.2rem;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #6e7681;
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem;
    font-weight: 600; color: #58a6ff; line-height: 1.1;
}
.metric-unit {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    color: #8b949e; margin-top: 0.2rem;
}
.section-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #6e7681;
    text-transform: uppercase; letter-spacing: 0.14em;
    padding-bottom: 0.6rem; border-bottom: 1px solid #21262d; margin-bottom: 1.2rem;
}
.warn-banner {
    background: rgba(210,153,34,0.1); border: 1px solid rgba(210,153,34,0.35);
    border-radius: 8px; padding: 0.7rem 1rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: #e3b341; margin-bottom: 1rem;
}
div[data-testid="stSidebar"] { background: #0d1117 !important; border-right: 1px solid #21262d; }
.stButton > button {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 500;
    border-radius: 6px; border: 1px solid #30363d; background: #21262d;
    color: #c9d1d9; letter-spacing: 0.04em;
}
.stButton > button:hover { background: #30363d; border-color: #58a6ff; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Constants
# --------------------------------------------------
SAMPLE_RATE = 44100
SCENARIOS = {
    "V2a": "Frequenz-Sweep",
    "V2b": "Pegel-Stufentest",
    "V3":  "Clipping / Noise Burst",
    "V4":  "Weißes Rauschen"
}

PLOT_STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#0d1117",
    "axes.edgecolor":    "#21262d",
    "axes.labelcolor":   "#8b949e",
    "axes.titlecolor":   "#c9d1d9",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.7,
    "xtick.color":       "#6e7681",
    "ytick.color":       "#6e7681",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "axes.labelsize":    9,
    "axes.titlesize":    10,
    "font.family":       "monospace",
    "text.color":        "#8b949e",
    "lines.linewidth":   1.2,
    "figure.dpi":        150,
}

# --------------------------------------------------
# Audio helpers
# --------------------------------------------------
def generate_wav_bytes(signal):
    signal = np.clip(signal, -1.0, 1.0)
    signal_int16 = np.int16(signal * 32767)
    byte_io = io.BytesIO()
    wav.write(byte_io, SAMPLE_RATE, signal_int16)
    return byte_io.getvalue()

def create_beep_trigger(freq, amplitude=0.9):
    beep_dur  = 0.05
    pause_dur = 0.05
    t    = np.linspace(0, beep_dur, int(SAMPLE_RATE * beep_dur), endpoint=False)
    beep = amplitude * np.sin(2 * np.pi * freq * t) * np.hanning(len(t))
    pause = np.zeros(int(SAMPLE_RATE * pause_dur))
    return np.concatenate([beep, pause, beep, pause, beep])

def assemble_signal(main_signal):
    pause         = np.zeros(int(0.3 * SAMPLE_RATE))
    start_trigger = create_beep_trigger(2000)
    end_trigger   = create_beep_trigger(3000)
    return np.concatenate([start_trigger, pause, main_signal, pause, end_trigger])

# --------------------------------------------------
# Scientific metrics
# --------------------------------------------------
def compute_metrics(signal, sr=SAMPLE_RATE):
    rms   = float(np.sqrt(np.mean(signal**2)))
    peak  = float(np.max(np.abs(signal)))
    crest = peak / rms if rms > 0 else float('inf')
    return {
        "duration_s": len(signal) / sr,
        "rms":        rms,
        "peak":       peak,
        "crest_db":   20 * np.log10(crest) if crest > 0 and np.isfinite(crest) else 0,
        "db_rms":     20 * np.log10(rms)  if rms  > 0 else -120.0,
        "db_peak":    20 * np.log10(peak) if peak > 0 else -120.0,
        "n_samples":  len(signal),
        "sr":         sr,
    }

# --------------------------------------------------
# Scientific figure (3-panel + info)
# --------------------------------------------------
def make_science_figure(signal, title, extra_info, scenario_code):
    plt.rcParams.update(PLOT_STYLE)
    n   = len(signal)
    sr  = SAMPLE_RATE
    t   = np.linspace(0, n / sr, n)

    # FFT
    fft_vals = np.abs(np.fft.rfft(signal)) / n
    freqs    = np.fft.rfftfreq(n, 1 / sr)
    fft_db   = 20 * np.log10(np.maximum(fft_vals, 1e-12))

    # Spectrogram segment size
    seg     = min(2048, max(64, n // 8))
    overlap = seg // 2

    metrics = compute_metrics(signal, sr)

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.55, wspace=0.40,
        left=0.07, right=0.97, top=0.87, bottom=0.09
    )

    ax_time = fig.add_subplot(gs[0, :2])
    ax_fft  = fig.add_subplot(gs[1, :2])
    ax_spec = fig.add_subplot(gs[2, :2])
    ax_info = fig.add_subplot(gs[:, 2])

    ACCENT = "#58a6ff"
    GREEN  = "#3fb950"
    ORANGE = "#d29922"

    # ── Zeitbereich ──────────────────────────────
    ax_time.set_title("Zeitbereich / Time Domain", pad=6, color="#c9d1d9")
    ax_time.plot(t, signal, color=ACCENT, lw=0.6, alpha=0.85)
    ax_time.axhline(0, color="#30363d", lw=0.8, zorder=0)
    ax_time.set_xlim(0, t[-1])
    ax_time.set_ylim(-1.1, 1.1)
    ax_time.set_xlabel("Zeit [s]")
    ax_time.set_ylabel("Amplitude (norm.)")
    ax_time.fill_between(t, signal, 0, color=ACCENT, alpha=0.06)
    rms = metrics["rms"]
    ax_time.axhline( rms, color=GREEN, lw=0.8, ls="--", alpha=0.7, label=f"RMS ±{rms:.3f}")
    ax_time.axhline(-rms, color=GREEN, lw=0.8, ls="--", alpha=0.7)
    ax_time.legend(fontsize=7, loc="upper right", framealpha=0.15)

    # ── Frequenzspektrum ─────────────────────────
    ax_fft.set_title("Frequenzspektrum / Magnitude Spectrum", pad=6, color="#c9d1d9")
    valid = freqs > 0
    ax_fft.plot(freqs[valid], fft_db[valid], color=ORANGE, lw=0.7)
    ax_fft.set_xscale("log")
    ax_fft.set_xlim(max(freqs[valid][0], 1), sr / 2)
    db_floor = max(float(np.min(fft_db[valid])) - 5, -120)
    ax_fft.set_ylim(bottom=db_floor)
    ax_fft.set_xlabel("Frequenz [Hz]")
    ax_fft.set_ylabel("Magnitude [dBFS]")
    ax_fft.fill_between(freqs[valid], fft_db[valid], db_floor, color=ORANGE, alpha=0.06)

    # ── Spektrogramm ─────────────────────────────
    ax_spec.set_title("Spektrogramm / Short-Time Fourier Transform", pad=6, color="#c9d1d9")
    try:
        ax_spec.specgram(
            signal, NFFT=seg, Fs=sr, noverlap=overlap,
            cmap="inferno", scale="dB", vmin=-120, vmax=0
        )
        ax_spec.set_xlabel("Zeit [s]")
        ax_spec.set_ylabel("Frequenz [Hz]")
        ax_spec.set_yscale("log")
        ax_spec.set_ylim(20, sr / 2)
    except Exception:
        ax_spec.text(0.5, 0.5, "Signal zu kurz für Spektrogramm",
                     ha="center", va="center", color="#6e7681",
                     transform=ax_spec.transAxes)

    # ── Info Panel ───────────────────────────────
    ax_info.set_axis_off()
    ax_info.set_facecolor("#0d1117")

    info_lines = [
        ("SZENARIO", f"{scenario_code}"),
        ("",         title),
        ("SEP",      ""),
        ("Abtastrate",   f"{sr:,} Hz"),
        ("Samples",      f"{metrics['n_samples']:,}"),
        ("Dauer",        f"{metrics['duration_s']:.2f} s"),
        ("",             ""),
        ("Peak",         f"{metrics['peak']:.4f}"),
        ("Peak dBFS",    f"{metrics['db_peak']:.1f} dBFS"),
        ("RMS",          f"{metrics['rms']:.4f}"),
        ("RMS dBFS",     f"{metrics['db_rms']:.1f} dBFS"),
        ("Crest Factor", f"{metrics['crest_db']:.1f} dB"),
        ("",             ""),
    ]
    for k, v in extra_info.items():
        info_lines.append((k, str(v)))
    info_lines += [
        ("",        ""),
        ("SEP",     ""),
        ("Erstellt", datetime.datetime.now().strftime("%Y-%m-%d")),
        ("Version",  "NVH Signal Lab v2.1"),
    ]

    y = 0.97
    for label, value in info_lines:
        if label == "SEP":
            # Draw separator line using axes coordinates via plot
            ax_info.plot(
                [0.02, 0.98], [y, y],
                color="#21262d", lw=0.7,
                transform=ax_info.transAxes,
                clip_on=False
            )
            y -= 0.025
            continue
        if label == "":
            y -= 0.022
            continue
        ax_info.text(0.04, y, label,
                     transform=ax_info.transAxes,
                     fontsize=7.5, color="#6e7681", fontfamily="monospace")
        ax_info.text(0.04, y - 0.028, value,
                     transform=ax_info.transAxes,
                     fontsize=8.5, color="#e6edf3",
                     fontfamily="monospace", fontweight="bold")
        y -= 0.062

    # ── Haupttitel ───────────────────────────────
    fig.text(0.04, 0.95, f"NVH Signal Lab  ·  {scenario_code}: {title}",
             fontsize=12, color="#e6edf3", fontfamily="monospace", fontweight="bold")
    fig.text(0.04, 0.922,
             f"fs = {sr:,} Hz  |  N = {metrics['n_samples']:,}  |  "
             f"Δf = {sr / metrics['n_samples']:.3f} Hz  |  "
             f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
             fontsize=8, color="#6e7681", fontfamily="monospace")

    return fig

# --------------------------------------------------
# PNG export
# --------------------------------------------------
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    return buf.getvalue()

# --------------------------------------------------
# Metric HTML
# --------------------------------------------------
def metric_card_html(label, value, unit=""):
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-unit">{unit}</div>'
            f'</div>')

def show_metrics(metrics):
    html = '<div class="metric-grid">'
    html += metric_card_html("Dauer",        f"{metrics['duration_s']:.2f}", "Sekunden")
    html += metric_card_html("Peak",         f"{metrics['db_peak']:.1f}",   "dBFS")
    html += metric_card_html("RMS",          f"{metrics['db_rms']:.1f}",    "dBFS")
    html += metric_card_html("Crest Factor", f"{metrics['crest_db']:.1f}",  "dB")
    html += metric_card_html("Samples",      f"{metrics['n_samples']:,}",   f"@ {metrics['sr']:,} Hz")
    html += metric_card_html("Peak Amp",     f"{metrics['peak']:.4f}",      "normiert")
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.65rem;
                color:#6e7681;text-transform:uppercase;letter-spacing:0.12em;
                margin-bottom:0.3rem;'>NVH Signal Lab</div>
    <div style='font-family:"IBM Plex Mono",monospace;font-size:1rem;
                color:#e6edf3;font-weight:600;margin-bottom:1.5rem;'>Messszenarien</div>
    """, unsafe_allow_html=True)

    selected_code = st.radio(
        "Szenario",
        list(SCENARIOS.keys()),
        format_func=lambda k: f"{k} · {SCENARIOS[k]}",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.65rem;
                color:#6e7681;text-transform:uppercase;letter-spacing:0.1em;
                margin-bottom:0.5rem;'>Triggerfrequenzen</div>
    <div style='font-family:"IBM Plex Mono",monospace;font-size:0.78rem;color:#8b949e;'>
    ▶ Start: 2000 Hz (3×)<br>■ Ende:  3000 Hz (3×)<br>
    <span style='font-size:0.65rem;color:#6e7681;'>Pause 300 ms vor/nach Signal</span>
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown(f"""
<div class="nvh-header">
  <div>
    <div class="nvh-title">NVH Signal Lab</div>
    <div class="nvh-subtitle">Noise · Vibration · Harshness  ·  Hardware-Validierung</div>
  </div>
  <div style="text-align:right">
    <div class="nvh-badge">fs = {SAMPLE_RATE:,} Hz</div>
    <div class="nvh-badge">16-bit PCM · Mono</div>
  </div>
</div>
""", unsafe_allow_html=True)

scenario_title = SCENARIOS[selected_code]

st.markdown(f"""
<div class="scenario-card">
  <div class="scenario-label">Aktives Szenario</div>
  <div class="scenario-name">{selected_code} · {scenario_title}</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# PARAMETERS + DATA COLUMNS
# ==================================================
col_params, col_data = st.columns([1, 2], gap="large")

signal      = None
full_signal = None
extra_info  = {}

with col_params:
    st.markdown('<div class="section-title">Parameter</div>', unsafe_allow_html=True)

    # V2a ─────────────────────────────────────────
    if selected_code == "V2a":
        duration  = st.slider("Sweepdauer (s)", 10, 120, 60)
        f_start   = st.number_input("Startfrequenz (Hz)", 0.1, 1000.0, 0.5, step=0.5, format="%.1f")
        f_end_pct = st.slider("Endfrequenz (% von Nyquist)", 50, 99, 95)
        amplitude = st.slider("Amplitude", 0.1, 1.0, 1.0)

        t     = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        f_end = SAMPLE_RATE / 2 * (f_end_pct / 100)
        L     = duration / np.log(f_end / f_start)
        phase = 2 * np.pi * f_start * L * (np.exp(t / L) - 1)
        signal      = amplitude * np.sin(phase)
        full_signal = assemble_signal(signal)
        extra_info  = {
            "Typ":       "Log. Sweep",
            "f_start":   f"{f_start:.1f} Hz",
            "f_end":     f"{f_end:.0f} Hz",
            "Amplitude": f"{amplitude:.2f}",
        }

    # V2b ─────────────────────────────────────────
    elif selected_code == "V2b":
        freq     = st.slider("Frequenz (Hz)", 10, 20000, 550)
        duration = st.slider("Dauer (s)", 5, 60, 45)
        env_type = st.selectbox("Hüllkurve", [
            "Linear fallend", "Linear steigend", "Konstant", "Sinus-moduliert"
        ])

        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        if env_type == "Linear fallend":
            env = np.linspace(1.0, 0.0, len(t))
        elif env_type == "Linear steigend":
            env = np.linspace(0.0, 1.0, len(t))
        elif env_type == "Konstant":
            env = np.ones(len(t))
        else:
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)

        signal      = np.sin(2 * np.pi * freq * t) * env
        full_signal = assemble_signal(signal)
        extra_info  = {
            "Frequenz":  f"{freq} Hz",
            "Hüllkurve": env_type,
            "Dauer":     f"{duration} s",
        }

    # V3 ──────────────────────────────────────────
    elif selected_code == "V3":
        num_pulses    = st.slider("Anzahl Impulse", 1, 20, 5)
        pulse_spacing = st.slider("Abstand (ms)", 10, 1000, 200)
        amplitude     = st.slider("Intensität (Amplitude)", 0.1, 2.0, 1.0)
        burst_ms      = st.slider("Burst-Dauer (ms)", 5, 100, 50)

        samples_burst   = int(SAMPLE_RATE * burst_ms / 1000)
        samples_spacing = int(SAMPLE_RATE * pulse_spacing / 1000)
        base_noise      = np.random.uniform(-1.0, 1.0, samples_burst)
        single_pulse    = amplitude * base_noise * np.hanning(samples_burst)

        parts = []
        for i in range(num_pulses):
            parts.append(single_pulse)
            if i < num_pulses - 1:
                parts.append(np.zeros(samples_spacing))
        signal      = np.concatenate(parts)
        full_signal = assemble_signal(signal)
        extra_info  = {
            "Impulse":     f"{num_pulses}",
            "Abstand":     f"{pulse_spacing} ms",
            "Burst-Dauer": f"{burst_ms} ms",
            "Amplitude":   f"{amplitude:.2f}",
        }

    # V4 ──────────────────────────────────────────
    elif selected_code == "V4":
        duration  = st.slider("Dauer (s)", 5, 60, 15)
        amplitude = st.slider("Amplitude", 0.1, 1.0, 1.0)
        signal      = amplitude * np.random.uniform(-1.0, 1.0, int(SAMPLE_RATE * duration))
        full_signal = assemble_signal(signal)
        extra_info  = {
            "Typ":       "White Noise",
            "Amplitude": f"{amplitude:.2f}",
        }

    do_run = st.button("▶  Signal abspielen", width="stretch", type="primary")

# ==================================================
# DATA COLUMN
# ==================================================
with col_data:
    if full_signal is not None:
        metrics = compute_metrics(full_signal)

        st.markdown('<div class="section-title">Wissenschaftliche Kennwerte</div>',
                    unsafe_allow_html=True)
        show_metrics(metrics)

        st.markdown('<div class="section-title">Signalanalyse</div>',
                    unsafe_allow_html=True)

        fig = make_science_figure(full_signal, scenario_title, extra_info, selected_code)
        st.pyplot(fig)
        plt.close(fig)

        # PNG Export
        fig_export = make_science_figure(full_signal, scenario_title, extra_info, selected_code)
        png_bytes  = fig_to_png_bytes(fig_export)
        plt.close(fig_export)

        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"NVH_{selected_code}_{ts}.png"
        st.download_button(
            label="⬇  Abbildung als PNG exportieren",
            data=png_bytes,
            file_name=fname,
            mime="image/png",
            width="stretch",
        )

        # Audio
        if do_run:
            st.markdown('<div class="section-title" style="margin-top:1rem">Audioausgabe</div>',
                        unsafe_allow_html=True)
            if selected_code == "V3":
                st.markdown('<div class="warn-banner">⚠  Sehr laut — Kopfhörer-Lautstärke reduzieren!</div>',
                            unsafe_allow_html=True)
            wav_bytes = generate_wav_bytes(full_signal)
            st.audio(wav_bytes, format="audio/wav")
            st.download_button(
                label="⬇  WAV-Datei herunterladen",
                data=wav_bytes,
                file_name=f"NVH_{selected_code}_{ts}.wav",
                mime="audio/wav",
                width="stretch",
            )

# ==================================================
# PROTOCOL
# ==================================================
st.markdown("---")
st.markdown('<div class="section-title">Messprotokoll</div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns([1.2, 1, 2])
with p1:
    system = st.selectbox("Messsystem", [
        "— auswählen —", "Pico Technology", "SQuadriga II",
        "HEAD Artemis", "Brüel & Kjær", "NI DAQ", "Sonstiges"
    ])
with p2:
    signal_loss = st.number_input("Signalverlust (s)", 0.0, 120.0, 0.0, step=0.1)
with p3:
    note = st.text_area("Notizen / Beobachtungen", height=80,
                         placeholder="z.B. Kanal 2 Rauschen bei >10 kHz…")

if st.button("Protokoll speichern"):
    if system != "— auswählen —":
        st.success(
            f"✓ Protokolleintrag gespeichert  ·  "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  "
            f"System: {system}  ·  Szenario: {selected_code}"
        )
    else:
        st.error("Bitte zuerst ein Messsystem auswählen.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(f"""
<div style='text-align:center;margin-top:3rem;padding-top:1.5rem;
            border-top:1px solid #21262d;
            font-family:"IBM Plex Mono",monospace;font-size:0.68rem;color:#6e7681;'>
    NVH Signal Lab v2.1  ·  fs = {SAMPLE_RATE:,} Hz  ·  16-bit PCM
    ·  {datetime.datetime.now().strftime('%Y')}
</div>
""", unsafe_allow_html=True)
