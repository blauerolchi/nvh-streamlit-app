import streamlit as st
import numpy as np
import io
import scipy.io.wavfile as wav
import scipy.signal as sg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import gc

st.set_page_config(page_title="NVH Signal Lab", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background-color:#0a0c10;color:#c9d1d9}
.main .block-container{padding:2rem 2.5rem;max-width:1400px}
.nvh-header{display:flex;align-items:flex-start;justify-content:space-between;border-bottom:1px solid #21262d;padding-bottom:1.5rem;margin-bottom:2rem}
.nvh-title{font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:600;color:#e6edf3;letter-spacing:-0.5px}
.nvh-subtitle{font-size:0.78rem;color:#6e7681;font-family:'IBM Plex Mono',monospace;margin-top:0.3rem;letter-spacing:0.05em;text-transform:uppercase}
.nvh-badge{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:0.4rem 0.9rem;font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#58a6ff;display:inline-block;margin-bottom:0.3rem}
.scenario-card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:1.4rem 1.6rem;margin-bottom:1.5rem}
.scenario-label{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.2rem}
.scenario-name{font-size:1.25rem;font-weight:600;color:#e6edf3}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:0.8rem;margin:1rem 0 1.5rem 0}
.metric-card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:1rem 1.2rem}
.metric-label{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem}
.metric-value{font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:600;color:#58a6ff;line-height:1.1}
.metric-unit{font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#8b949e;margin-top:0.2rem}
.section-title{font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.14em;padding-bottom:0.6rem;border-bottom:1px solid #21262d;margin-bottom:1.2rem}
.warn-banner{background:rgba(210,153,34,0.1);border:1px solid rgba(210,153,34,0.35);border-radius:8px;padding:0.7rem 1rem;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#e3b341;margin-bottom:1rem}
div[data-testid="stSidebar"]{background:#0d1117 !important;border-right:1px solid #21262d}
.stButton>button{font-family:'IBM Plex Mono',monospace;font-size:0.8rem;font-weight:500;border-radius:6px;border:1px solid #30363d;background:#21262d;color:#c9d1d9}
.stButton>button:hover{background:#30363d;border-color:#58a6ff;color:#58a6ff}
</style>
""", unsafe_allow_html=True)

SAMPLE_RATE = 44100
SCENARIOS = {"V2a": "Frequenz-Sweep", "V2b": "Pegel-Stufentest", "V3": "Clipping / Noise Burst", "V4": "Weißes Rauschen"}
PLOT_STYLE = {"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117", "axes.edgecolor": "#21262d",
              "axes.labelcolor": "#8b949e", "axes.titlecolor": "#c9d1d9", "axes.grid": True,
              "grid.color": "#21262d", "grid.linewidth": 0.7, "xtick.color": "#6e7681", "ytick.color": "#6e7681",
              "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 9, "axes.titlesize": 10,
              "font.family": "monospace", "text.color": "#8b949e", "lines.linewidth": 1.2, "figure.dpi": 100} # DPI reduziert für Cloud-Stabilität

def get_hann(N):
    if N <= 1: return np.ones(N)
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

def generate_wav_bytes(signal):
    signal = np.clip(signal, -1.0, 1.0)
    buf = io.BytesIO()
    wav.write(buf, SAMPLE_RATE, np.int16(signal * 32767))
    return buf.getvalue()

def create_beep_trigger(freq):
    t = np.linspace(0, 0.05, int(SAMPLE_RATE * 0.05), endpoint=False)
    beep = 0.9 * np.sin(2 * np.pi * freq * t) * get_hann(len(t))
    pause = np.zeros(int(SAMPLE_RATE * 0.05))
    return np.concatenate([beep, pause, beep, pause, beep])

def assemble_signal(main_signal):
    pause = np.zeros(int(0.3 * SAMPLE_RATE))
    return np.concatenate([create_beep_trigger(2000), pause, main_signal, pause, create_beep_trigger(3000)])

def compute_metrics(signal):
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    crest = peak / rms if rms > 0 else float('inf')
    return {"duration_s": len(signal)/SAMPLE_RATE, "rms": rms, "peak": peak,
            "crest_db": 20*np.log10(crest) if np.isfinite(crest) and crest>0 else 0,
            "db_rms": 20*np.log10(rms) if rms>0 else -120.0,
            "db_peak": 20*np.log10(peak) if peak>0 else -120.0, "n_samples": len(signal)}

def make_science_figure(signal, title, extra_info, scenario_code):
    plt.rcParams.update(PLOT_STYLE)
    n, sr = len(signal), SAMPLE_RATE
    t = np.linspace(0, n/sr, n)
    
    # FFT Berechnung
    fft_vals = np.abs(np.fft.rfft(signal)) / n
    freqs = np.fft.rfftfreq(n, 1/sr)
    fft_db = 20 * np.log10(np.maximum(fft_vals, 1e-12))
    m = compute_metrics(signal)
    
    # Aggressives Subsampling für Matplotlib Plotting (Cloud-OOM-Schutz)
    MAX_PLOT_PTS = 20000 
    
    if n > MAX_PLOT_PTS:
        step = n // MAX_PLOT_PTS
        t_plot = t[::step]
        sig_plot = signal[::step]
    else:
        t_plot, sig_plot = t, signal
        
    valid = freqs > 0
    f_val, fft_val = freqs[valid], fft_db[valid]
    
    if len(f_val) > MAX_PLOT_PTS:
        step_fft = len(f_val) // MAX_PLOT_PTS
        f_plot = f_val[::step_fft]
        fft_plot = fft_val[::step_fft]
    else:
        f_plot, fft_plot = f_val, fft_val
        
    # Spektrogramm Konfiguration
    target_time_bins = 400
    seg = max(512, n // target_time_bins)
    seg = min(4096, seg)
    
    fig = plt.figure(figsize=(13, 8))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40, left=0.07, right=0.97, top=0.87, bottom=0.09)
    
    ax_time = fig.add_subplot(gs[0, :2])
    ax_fft = fig.add_subplot(gs[1, :2])
    ax_spec = fig.add_subplot(gs[2, :2])
    ax_info = fig.add_subplot(gs[:, 2])
    
    ACCENT, GREEN, ORANGE = "#58a6ff", "#3fb950", "#d29922"
    
    # Plot 1: Zeit
    ax_time.set_title("Zeitbereich / Time Domain", pad=6, color="#c9d1d9")
    ax_time.plot(t_plot, sig_plot, color=ACCENT, lw=0.6, alpha=0.85)
    ax_time.axhline(0, color="#30363d", lw=0.8, zorder=0)
    ax_time.set_xlim(0, t[-1])
    ax_time.set_ylim(-1.1, 1.1)
    ax_time.set_xlabel("Zeit [s]")
    ax_time.set_ylabel("Amplitude")
    ax_time.fill_between(t_plot, sig_plot, 0, color=ACCENT, alpha=0.06)
    
    rms = m["rms"]
    ax_time.axhline(rms, color=GREEN, lw=0.8, ls="--", alpha=0.7, label=f"RMS ±{rms:.3f}")
    ax_time.axhline(-rms, color=GREEN, lw=0.8, ls="--", alpha=0.7)
    ax_time.legend(fontsize=7, loc="upper right", framealpha=0.15)
    
    # Plot 2: FFT
    ax_fft.set_title("Frequenzspektrum / Magnitude Spectrum", pad=6, color="#c9d1d9")
    ax_fft.plot(f_plot, fft_plot, color=ORANGE, lw=0.7)
    ax_fft.set_xscale("log")
    ax_fft.set_xlim(max(f_plot[0], 1), sr/2)
    db_floor = max(float(np.min(fft_plot))-5, -120)
    ax_fft.set_ylim(bottom=db_floor)
    ax_fft.set_xlabel("Frequenz [Hz]")
    ax_fft.set_ylabel("Magnitude [dBFS]")
    ax_fft.fill_between(f_plot, fft_plot, db_floor, color=ORANGE, alpha=0.06)
    
    # Plot 3: Speichersicheres Spektrogramm via SciPy
    ax_spec.set_title("Spektrogramm / STFT", pad=6, color="#c9d1d9")
    try:
        f_spec, t_spec, Sxx = sg.spectrogram(signal, fs=sr, nperseg=seg, noverlap=seg//2)
        Sxx_db = 10 * np.log10(np.maximum(Sxx, 1e-12))
        
        # rasterized=True ist hier essenziell um riesige SVG/PDF Puffer zu vermeiden
        ax_spec.pcolormesh(t_spec, f_spec, Sxx_db, cmap="inferno", vmin=-120, vmax=0, rasterized=True, shading='nearest')
        ax_spec.set_xlabel("Zeit [s]")
        ax_spec.set_ylabel("Frequenz [Hz]")
        ax_spec.set_yscale("log")
        ax_spec.set_ylim(20, sr/2)
    except Exception as e:
        ax_spec.text(0.5, 0.5, "Signal zu kurz für Spektrogramm", ha="center", va="center", color="#6e7681", transform=ax_spec.transAxes)
        
    # Info Panel
    ax_info.set_axis_off()
    info_lines = [("SZENARIO", scenario_code), ("", title), ("SEP", ""),
                  ("Abtastrate", f"{sr:,} Hz"), ("Samples", f"{m['n_samples']:,}"),
                  ("Dauer", f"{m['duration_s']:.2f} s"), ("", ""),
                  ("Peak", f"{m['peak']:.4f}"), ("Peak dBFS", f"{m['db_peak']:.1f}"),
                  ("RMS", f"{m['rms']:.4f}"), ("RMS dBFS", f"{m['db_rms']:.1f}"),
                  ("Crest Factor", f"{m['crest_db']:.1f} dB"), ("", "")]
                  
    for k, v in extra_info.items():
        info_lines.append((k, str(v)))
        
    info_lines += [("", ""), ("SEP", ""), ("Erstellt", datetime.datetime.now().strftime("%Y-%m-%d")), ("Version", "NVH Signal Lab v2.2")]
    y = 0.97
    
    for label, value in info_lines:
        if label == "SEP":
            ax_info.plot([0.02, 0.98], [y, y], color="#21262d", lw=0.7, transform=ax_info.transAxes, clip_on=False)
            y -= 0.025
            continue
        if label == "":
            y -= 0.022
            continue
        ax_info.text(0.04, y, label, transform=ax_info.transAxes, fontsize=7.5, color="#6e7681", fontfamily="monospace")
        ax_info.text(0.04, y-0.028, value, transform=ax_info.transAxes, fontsize=8.5, color="#e6edf3", fontfamily="monospace", fontweight="bold")
        y -= 0.062
        
    fig.text(0.04, 0.95, f"NVH Signal Lab  ·  {scenario_code}: {title}", fontsize=11, color="#e6edf3", fontfamily="monospace", fontweight="bold")
    fig.text(0.04, 0.922, f"fs={sr:,} Hz  |  N={m['n_samples']:,}  |  Δf={sr/m['n_samples']:.3f} Hz  |  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=7.5, color="#6e7681", fontfamily="monospace")
    
    return fig

def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0d1117", edgecolor="none")
    buf.seek(0)
    return buf.getvalue()

def metric_html(label, value, unit=""):
    return (f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div><div class="metric-unit">{unit}</div></div>')

def show_metrics(m):
    html = '<div class="metric-grid">'
    html += metric_html("Dauer", f"{m['duration_s']:.2f}", "Sekunden")
    html += metric_html("Peak", f"{m['db_peak']:.1f}", "dBFS")
    html += metric_html("RMS", f"{m['db_rms']:.1f}", "dBFS")
    html += metric_html("Crest Factor", f"{m['crest_db']:.1f}", "dB")
    html += metric_html("Samples", f"{m['n_samples']:,}", f"@ {SAMPLE_RATE:,} Hz")
    html += metric_html("Peak Amp", f"{m['peak']:.4f}", "normiert")
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem;'>NVH Signal Lab</div><div style='font-family:\"IBM Plex Mono\",monospace;font-size:1rem;color:#e6edf3;font-weight:600;margin-bottom:1.5rem;'>Messszenarien</div>", unsafe_allow_html=True)
    selected_code = st.radio("Szenario", list(SCENARIOS.keys()), format_func=lambda k: f"{k} · {SCENARIOS[k]}", label_visibility="collapsed")
    st.markdown("<div style='margin-top:1.5rem;font-family:\"IBM Plex Mono\",monospace;font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>Triggerfrequenzen</div><div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.78rem;color:#8b949e;'>▶ Start: 2000 Hz (3×)<br>■ Ende: 3000 Hz (3×)</div>", unsafe_allow_html=True)

# HEADER
scenario_title = SCENARIOS[selected_code]
st.markdown(f"""<div class="nvh-header"><div><div class="nvh-title">NVH Signal Lab</div><div class="nvh-subtitle">Noise · Vibration · Harshness  ·  Hardware-Validierung</div></div><div style="text-align:right"><div class="nvh-badge">fs = {SAMPLE_RATE:,} Hz</div><div class="nvh-badge">16-bit PCM · Mono</div></div></div>""", unsafe_allow_html=True)
st.markdown(f"""<div class="scenario-card"><div class="scenario-label">Aktives Szenario</div><div class="scenario-name">{selected_code} · {scenario_title}</div></div>""", unsafe_allow_html=True)

col_params, col_data = st.columns([1, 2], gap="large")

# PARAMETERS
with col_params:
    st.markdown('<div class="section-title">Parameter</div>', unsafe_allow_html=True)
    if selected_code == "V2a":
        duration = st.slider("Sweepdauer (s)", 10, 120, 30)
        f_start = st.number_input("Startfrequenz (Hz)", 0.1, 1000.0, 0.5, step=0.5, format="%.1f")
        f_end_pct = st.slider("Endfrequenz (% von Nyquist)", 50, 99, 95)
        amplitude = st.slider("Amplitude", 0.1, 1.0, 1.0)
        params = dict(duration=duration, f_start=f_start, f_end_pct=f_end_pct, amplitude=amplitude)
    elif selected_code == "V2b":
        freq = st.slider("Frequenz (Hz)", 10, 20000, 550)
        duration = st.slider("Dauer (s)", 5, 60, 15)
        env_type = st.selectbox("Hüllkurve", ["Linear fallend", "Linear steigend", "Konstant", "Sinus-moduliert"])
        params = dict(freq=freq, duration=duration, env_type=env_type)
    elif selected_code == "V3":
        num_pulses = st.slider("Anzahl Impulse", 1, 20, 5)
        pulse_spacing = st.slider("Abstand (ms)", 10, 1000, 200)
        amplitude = st.slider("Intensität", 0.1, 2.0, 1.0)
        burst_ms = st.slider("Burst-Dauer (ms)", 5, 100, 50)
        params = dict(num_pulses=num_pulses, pulse_spacing=pulse_spacing, amplitude=amplitude, burst_ms=burst_ms)
    elif selected_code == "V4":
        duration = st.slider("Dauer (s)", 5, 60, 10)
        amplitude = st.slider("Amplitude", 0.1, 1.0, 1.0)
        params = dict(duration=duration, amplitude=amplitude)
    
    do_run = st.button("▶  Signal generieren & abspielen", type="primary")

# SIGNAL GENERATION
if do_run:
    sr = SAMPLE_RATE
    if selected_code == "V2a":
        d, fs_start, fep, amp = params["duration"], params["f_start"], params["f_end_pct"], params["amplitude"]
        t = np.linspace(0, d, int(sr*d))
        f_end = sr/2 * (fep/100)
        L = d / np.log(f_end/fs_start)
        phase = 2 * np.pi * fs_start * L * (np.exp(t/L) - 1)
        sig = amp * np.sin(phase)
        extra = {"Typ": "Log. Sweep", "f_start": f"{fs_start:.1f} Hz", "f_end": f"{f_end:.0f} Hz", "Amplitude": f"{amp:.2f}"}
    elif selected_code == "V2b":
        freq, d, env_type = params["freq"], params["duration"], params["env_type"]
        t = np.linspace(0, d, int(sr*d))
        if env_type == "Linear fallend":
            env = np.linspace(1.0, 0.0, len(t))
        elif env_type == "Linear steigend":
            env = np.linspace(0.0, 1.0, len(t))
        elif env_type == "Konstant":
            env = np.ones(len(t))
        else:
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        sig = np.sin(2 * np.pi * freq * t) * env
        extra = {"Frequenz": f"{freq} Hz", "Hüllkurve": env_type, "Dauer": f"{d} s"}
    elif selected_code == "V3":
        np_, ps, amp, bm = params["num_pulses"], params["pulse_spacing"], params["amplitude"], params["burst_ms"]
        sb = int(sr * bm / 1000)
        sp = int(sr * ps / 1000)
        pulse = amp * np.random.uniform(-1.0, 1.0, sb) * get_hann(sb)
        parts = []
        for i in range(np_):
            parts.append(pulse)
            if i < np_ - 1:
                parts.append(np.zeros(sp))
        sig = np.concatenate(parts)
        extra = {"Impulse": f"{np_}", "Abstand": f"{ps} ms", "Burst-Dauer": f"{bm} ms", "Amplitude": f"{amp:.2f}"}
    elif selected_code == "V4":
        d, amp = params["duration"], params["amplitude"]
        sig = amp * np.random.uniform(-1.0, 1.0, int(sr * d))
        extra = {"Typ": "White Noise", "Amplitude": f"{amp:.2f}"}
        
    st.session_state["full_signal"] = assemble_signal(sig)
    st.session_state["extra_info"] = extra
    st.session_state["scenario_title"] = scenario_title
    st.session_state["selected_code"] = selected_code
    st.session_state["ts"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# DATA COLUMN
with col_data:
    if "full_signal" not in st.session_state:
        st.markdown("<div style='display:flex;align-items:center;justify-content:center;height:280px;border:1px dashed #21262d;border-radius:10px;flex-direction:column;gap:0.8rem;'><div style='font-size:2.5rem;color:#21262d;'>⚡</div><div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.8rem;color:#6e7681;'>Parameter wählen → Signal generieren</div></div>", unsafe_allow_html=True)
    else:
        fs_signal = st.session_state["full_signal"]
        ei = st.session_state["extra_info"]
        st_title = st.session_state["scenario_title"]
        sc = st.session_state["selected_code"]
        ts = st.session_state["ts"]
        m = compute_metrics(fs_signal)
        
        st.markdown('<div class="section-title">Wissenschaftliche Kennwerte</div>', unsafe_allow_html=True)
        show_metrics(m)
        st.markdown('<div class="section-title">Signalanalyse</div>', unsafe_allow_html=True)
        
        # 1. Einmal rendern, RAM sparen!
        fig = make_science_figure(fs_signal, st_title, ei, sc)
        png_bytes = fig_to_png(fig)
        
        # 2. Bild als echtes Bild anzeigen statt st.pyplot() zu nutzen
        st.image(png_bytes)
        
        # 3. Speichern bereinigen (Verhindert das Volllaufen des Servers)
        fig.clear()
        plt.close('all')
        gc.collect() 
        
        st.download_button("⬇  Abbildung als PNG exportieren", data=png_bytes, file_name=f"NVH_{sc}_{ts}.png", mime="image/png")
        
        st.markdown('<div class="section-title" style="margin-top:1rem">Audioausgabe</div>', unsafe_allow_html=True)
        if sc == "V3":
            st.markdown('<div class="warn-banner">⚠  Sehr laut — Lautstärke reduzieren!</div>', unsafe_allow_html=True)
            
        wav_b = generate_wav_bytes(fs_signal)
        st.audio(wav_b, format="audio/wav")
        st.download_button("⬇  WAV-Datei herunterladen", data=wav_b, file_name=f"NVH_{sc}_{ts}.wav", mime="audio/wav")

# PROTOCOL
st.markdown("---")
st.markdown('<div class="section-title">Messprotokoll</div>', unsafe_allow_html=True)
p1, p2, p3 = st.columns([1.2, 1, 2])
with p1:
    system = st.selectbox("Messsystem", ["— auswählen —", "Pico Technology", "SQuadriga II", "HEAD Artemis", "Brüel & Kjær", "NI DAQ", "Sonstiges"])
with p2:
    st.number_input("Signalverlust (s)", 0.0, 120.0, 0.0, step=0.1)
with p3:
    st.text_area("Notizen", height=80, placeholder="z.B. Kanal 2 Rauschen bei >10 kHz…")

if st.button("Protokoll speichern"):
    if system != "— auswählen —":
        st.success(f"✓ Gespeichert  ·  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ·  {system}  ·  {selected_code}")
    else:
        st.error("Bitte zuerst ein Messsystem auswählen.")
        
st.markdown(f"<div style='text-align:center;margin-top:3rem;padding-top:1.5rem;border-top:1px solid #21262d;font-family:\"IBM Plex Mono\",monospace;font-size:0.68rem;color:#6e7681;'>NVH Signal Lab v2.2  ·  fs = {SAMPLE_RATE:,} Hz  ·  16-bit PCM  ·  {datetime.datetime.now().strftime('%Y')}</div>", unsafe_allow_html=True)
