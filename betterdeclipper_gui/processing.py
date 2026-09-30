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
    normalize: bool = False
    target_db: float = -0.1
    folder: str = ""


def db(v):
    return 20 * np.log10(max(abs(float(v)), 1e-12))


def device_description():
    """('cuda' or 'cpu', readable name) of what 'auto' picks."""
    import torch
    if torch.cuda.is_available():
        return "cuda", f"GPU: {torch.cuda.get_device_name(0)}"
    return "cpu", "CPU (no CUDA GPU found: processing takes minutes per song)"


def declip_array(y, sr, s, progress=None, cancel=None):
    """y (T, C) float64 -> (x, info, restoration label). progress(step, n, seconds) is called after each
    processing step; setting the `cancel` event stops at the next step (raises Cancelled)."""
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
                             max_gain_db=s.max_gain_db, device=s.device, progress=step)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
    """(gain, note): normalize to the target peak if asked; integer formats (PCM WAV, FLAC) cannot hold
    restored peaks above 0 dBFS, so they are turned down to the target instead of clipping again."""
    if peak <= 0:
        return 1.0, ""
    target = 10 ** (out.target_db / 20)
    if out.normalize:
        g = target / peak
        return g, f"normalized to {out.target_db:+.1f} dBFS ({db(g):+.1f} dB)"
    if FORMATS[out.fmt][2] != "FLOAT" and peak > 1.0:
        g = target / peak
        return g, (f"turned down {-db(g):.1f} dB to {out.target_db:+.1f} dBFS: {FORMATS[out.fmt][0]} "
                   f"cannot hold the restored peaks above 0 dBFS")
    return 1.0, ""


def write_audio(x, sr, path, fmt, gain=1.0):
    _, container, subtype, _ = FORMATS[fmt]
    data = np.asarray(x) if gain == 1.0 else np.asarray(x) * gain
    if subtype == "FLOAT":
        data = data.astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, data, sr, format=container, subtype=subtype)


def output_name(stem, label, preset, fmt):
    """'<stem> [<restoration> <preset>].<ext>', as the command line names its default output."""
    return f"{stem} [{label} {preset}]{FORMATS[fmt][3]}"


def list_audio(folder):
    """Audio files directly inside a folder, sorted by name."""
    names = sorted(os.listdir(folder), key=str.lower)
    return [os.path.join(folder, n) for n in names
            if n.lower().endswith(AUDIO_EXTS) and os.path.isfile(os.path.join(folder, n))]
