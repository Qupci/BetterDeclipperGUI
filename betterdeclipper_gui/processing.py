"""Running the declipper and writing results, shared by the single-file view and batch mode.

Mirrors the command line (betterdeclipper.cli): forced clip levels and knees switch the mode to hard
and soft, results are named '<input> [<restoration> <preset>].<ext>', and the analysis is reported
with the same lines the CLI prints.
"""
import os
import threading
from dataclasses import dataclass

import numpy as np
import soundfile as sf

PRESETS = ("fast", "normal", "high", "best")

# key: (label, container, subtype, extension)
FORMATS = {
    "wav32f": ("WAV 32-bit float", "WAV", "FLOAT", ".wav"),
    "wav24": ("WAV 24-bit", "WAV", "PCM_24", ".wav"),
    "wav16": ("WAV 16-bit", "WAV", "PCM_16", ".wav"),
    "flac24": ("FLAC 24-bit", "FLAC", "PCM_24", ".flac"),
    "flac16": ("FLAC 16-bit", "FLAC", "PCM_16", ".flac"),
}

AUDIO_EXTS = (".wav", ".flac", ".aif", ".aiff", ".aifc", ".mp3", ".ogg", ".oga", ".opus", ".w64", ".rf64",
              ".caf")

_gpu_lock = threading.Lock()  # one restoration at a time (single file and batch share the device)


class Cancelled(Exception):
    pass


@dataclass
class Settings:
    preset: str = "normal"
    mode: str = "auto"
    clip_level_db: float | None = None   # force a hard clip level (dBFS), skips the analysis
    knee_db: float | None = None         # force a soft-clip knee (dBFS), skips the analysis
    max_gain_db: float | None = None
    device: str = "auto"


@dataclass
class Output:
    fmt: str = "wav32f"
    target_db: float = -0.1
    folder: str = ""
    steps: int | None = None  # None: PCM / FLAC turned down to target_db if louder, never up (32-bit float kept
                              # as it is); 0 - 4: a fixed gain of -steps x 3.01 dB, PCM / FLAC clip what still
                              # exceeds 0 dBFS


GAIN_STEPS = (0, 1, 2, 3, 4)


def step_db(n):
    """The fixed gain of n steps in dB, n x 10 log10(2) down: 0, -3.01, -6.02, -9.03, -12.04 for n = 0 ... 4."""
    return -n * 10 * np.log10(2)


def db(v):
    return 20 * np.log10(max(abs(float(v)), 1e-12))


def device_description():
    """('cuda' or 'cpu', readable name) of what 'auto' picks."""
    import torch
    if torch.cuda.is_available():
        return "cuda", f"GPU: {torch.cuda.get_device_name(0)}"
    return "cpu", "CPU (no CUDA GPU found: processing takes minutes per song)"


ANALYSIS_MODES = ("auto", "hard", "soft", "limiter")  # the modes a saved analysis serves


def supports_analysis():
    """Does the installed declipper take a saved analysis (betterdeclipper 0.3.0 on)?"""
    import inspect
    from betterdeclipper import declip
    return "analysis" in inspect.signature(declip).parameters


def reuses_analysis(s):
    """Can a run with settings s use the analysis of an earlier run of the same input? (Forced clip levels
    and knees skip the analysis, legacy mode has its own.)"""
    return s.mode in ANALYSIS_MODES and s.clip_level_db is None and s.knee_db is None and supports_analysis()


def declip_array(y, sr, s, progress=None, cancel=None, analysis=None):
    """y (T, C) float64 -> (x, info, restoration label). progress(step, n, seconds) is called after each
    processing step; setting the `cancel` event stops at the next step (raises Cancelled). analysis: the
    analysis of an earlier run of y (info['analysis']['data']), used instead of analyzing it again where it
    applies (reuses_analysis); info['reused_analysis'] tells whether it was."""
    import torch
    from betterdeclipper import declip
    from betterdeclipper.cli import restoration_label

    C = y.shape[1]
    levels = knees = forced = None
    mode = s.mode
    tag = lambda v: f"{round(20 * np.log10(v), 1):g}dB"
    if s.clip_level_db is not None:
        lv = 10 ** (s.clip_level_db / 20)
        levels = [(lv, -lv)] * C
        mode, forced = "hard", tag(lv)
    if s.knee_db is not None:
        kv = 10 ** (s.knee_db / 20)
        knees = [(kv, -kv)] * C
        mode, forced = "soft", "knee " + tag(kv)
    extra = {"analysis": analysis} if analysis is not None and reuses_analysis(s) else {}

    def step(i, n, el):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        if progress:
            progress(i, n, el)

    if cancel is not None and cancel.is_set():
        raise Cancelled()
    with _gpu_lock:
        # flush-to-zero is per thread: the core sets it for the thread that imports it, but Gradio
        # runs this in a worker thread (without it the NMF updates run ~10x slower on older CPUs)
        torch.set_flush_denormal(True)
        try:
            x, info = declip(y, sr, preset=s.preset, levels=levels, knees=knees, mode=mode,
                             max_gain_db=s.max_gain_db, device=s.device, progress=step, **extra)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    info["reused_analysis"] = bool(extra)
    return x, info, restoration_label(mode, info, forced)


def report_lines(info, C, preset):
    """The analysis and summary the command line prints."""
    from betterdeclipper.cli import analysis_lines
    if info.get("analysis") is not None:
        lines = analysis_lines(info, C)
    else:
        lv = ", ".join(f"ch{c}: " + "/".join("-" if v is None else f"{db(v):.2f} dBFS" for v in lvl)
                       for c, lvl in enumerate(info["levels"] or []))
        lines = [f"mode: {info['mode']}   {'knees' if info['mode'] == 'soft' else 'clip levels'}: {lv}"]
    lines.append(f"flagged samples: {info['clipped_frac'] * 100:.2f}%   preset: {preset}   "
                 f"device: {info.get('device', 'cpu')}   time: {info['time']:.1f}s")
    if info["clipped_frac"] == 0:
        lines.append("no clipping or limiting detected; output equals input (use mode soft, a forced knee "
                     "or a forced clip level to restore anyway)")
    return [ln.replace("--mode ", "mode ") for ln in lines]  # the GUI's Mode setting, not the CLI flag


def output_gain(peak, out):
    """(gain, note, clips). PCM WAV and FLAC are turned down to the target peak if louder, never up (they cannot
    hold restored peaks above 0 dBFS), or get a fixed gain (out.steps) and clip what still exceeds 0 dBFS (clips
    True); 32-bit float is kept as it is, or gets the fixed gain."""
    if peak <= 0:
        return 1.0, "", False
    integer = FORMATS[out.fmt][2] != "FLOAT"
    if out.steps is None:
        if not integer:
            return 1.0, "", False
        g = 10 ** (out.target_db / 20) / peak
        if g >= 1.0:  # under the peak level already: left as it is
            return 1.0, "", False
        return g, f"turned down {-db(g):.2f} dB to {out.target_db:+.1f} dBFS", False
    g = 10 ** (step_db(out.steps) / 20)
    over = peak * g > 1.0
    note = f"{step_db(out.steps):+.2f} dB" if out.steps else "0 dB"
    if over:
        note += (f", peak {db(peak * g):+.1f} dBFS: clipped at 0 dBFS" if integer else
                 f", peak {db(peak * g):+.1f} dBFS kept (32-bit float)")
    return g, note, integer and over


def write_audio(x, sr, path, fmt, gain=1.0):
    """Write x (times gain) in a FORMATS format; integer formats clip at 0 dBFS. Returns the number of
    clipped samples."""
    _, container, subtype, _ = FORMATS[fmt]
    data = np.asarray(x) if gain == 1.0 else np.asarray(x) * gain
    n_clip = 0
    if subtype == "FLOAT":
        data = data.astype(np.float32)
    else:
        n_clip = int(np.count_nonzero(np.abs(data) > 1.0))
        if n_clip:
            data = np.clip(data, -1.0, 1.0)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, data, sr, format=container, subtype=subtype)
    return n_clip


def clip_note(n_clip, n_total):
    return f" ({n_clip} samples, {n_clip / max(n_total, 1) * 100:.3g} %)" if n_clip else ""


def output_name(stem, label, preset, fmt):
    """'<stem> [<restoration> <preset>].<ext>', as the command line names its default output."""
    return f"{stem} [{label} {preset}]{FORMATS[fmt][3]}"


def list_audio(folder):
    """Audio files directly inside a folder, sorted by name."""
    names = sorted(os.listdir(folder), key=str.lower)
    return [os.path.join(folder, n) for n in names
            if n.lower().endswith(AUDIO_EXTS) and os.path.isfile(os.path.join(folder, n))]
