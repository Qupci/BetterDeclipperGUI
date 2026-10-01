"""Per-browser-tab state of the single-file view: the input, its results (history), the snippets and the
current view. Gradio-free, so it can be scripted and tested on its own.

Results are kept on disk as 32-bit float WAV (the raw restoration, before any output gain), next to a
float copy of the input when the input format cannot be read at arbitrary positions (MP3, Ogg); views
read only the samples they show. The snippet list is ranked from the first result that restored
anything and then stays put, so later results with other settings are compared at the same places (a
moment picked before that result stays on view).
"""
import os
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import render as R
from .processing import Cancelled, db, declip_array, output_gain, output_name, report_lines, write_audio
from .snippets import VIEW_S, RestorationMap, Snippet, rank

SEEKABLE = {"WAV", "WAVEX", "AIFF", "FLAC", "W64", "RF64", "CAF"}  # sample-exact random access
SPEC_CACHE = 24      # snippet spectrograms kept per session (~1 MB each)
OV_CACHE = 8         # full-length spectrograms kept per session (~1 MB each)
FULL_AUDIO_KEEP = 3  # full-length playback files kept per session (tens of MB each)
FADE_S = 0.005       # fade in/out of the snippet audio (no clicks at the cut)
SNAP_PX = 6          # a click on the overview this close to a snippet's flag selects that snippet

_created_dirs = set()  # work folders of this process (sessions end with it)


def remove_work_dirs():
    """Delete the work folders of all sessions of this process (at exit)."""
    for d in list(_created_dirs):
        shutil.rmtree(d, ignore_errors=True)


def channel_names(C):
    return ["L", "R"] if C == 2 else ["Mono"] if C == 1 else [f"Ch {i + 1}" for i in range(C)]


def parse_time(text):
    """'83.5', '1:23.5' or '0:01:23.5' -> seconds."""
    s = str(text).strip().replace(",", ".")
    parts = s.split(":")
    if not s or len(parts) > 3:
        raise ValueError(f"not a time: {text!r}")
    t = 0.0
    for p in parts:
        t = t * 60 + float(p)
    if t < 0:
        raise ValueError(f"not a time: {text!r}")
    return t


def same_file(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def read_segment(path, a, b, frames, C):
    """Samples [a, b) of all channels as float32 (b - a, C), zeros outside the file."""
    out = np.zeros((b - a, C), np.float32)
    lo, hi = max(a, 0), min(b, frames)
    if hi > lo:
        data, _ = sf.read(path, start=lo, stop=hi, dtype="float32", always_2d=True)
        out[lo - a:lo - a + len(data)] = data
    return out


@dataclass
class InputFile:
    path: str           # the uploaded file (processing reads it exactly as the CLI would)
    name: str
    sr: int
    channels: int
    frames: int
    format: str
    subtype: str
    view_path: str      # random-access copy for the views (the file itself when seekable)
    peak: float
    peak_pos: int       # center of the passage that reaches the peak most often (the most clipped one)
    peak_ch: int

    @property
    def stem(self):
        return os.path.splitext(self.name)[0]

    def describe(self):
        ch = {1: "mono", 2: "stereo"}.get(self.channels, f"{self.channels} channels")
        sub = self.subtype.replace("PCM_", "").replace("FLOAT", "32f").replace("DOUBLE", "64f")
        sub = f"{sub}-bit" if sub.isdigit() else sub.lower()
        return (f"**{self.name}**  \n{self.sr} Hz · {ch} · {R.fmt_time(self.frames / self.sr)} · "
                f"{self.format.lower()} {sub} · peak {db(self.peak):+.2f} dBFS")


def _busiest_peak(bmax, peak, view_blocks):
    """(block, channel) at the middle of the view-long passage with the most 10 ms blocks at the peak: the
    most clipped passage of a clipped file, or centered on the peak of an unclipped one."""
    near = (bmax >= 0.995 * peak).astype(np.float32)
    score = np.stack([np.convolve(near[:, c], np.ones(view_blocks), "same") for c in range(near.shape[1])], 1)
    b0, c = np.unravel_index(int(score.argmax()), score.shape)  # first maximum, in time
    b1 = b0
    while b1 + 1 < len(score) and score[b1 + 1, c] == score[b0, c]:
        b1 += 1
    return (b0 + b1) // 2, int(c)


def open_input(path, work):
    """Scan an audio file (peak and the most clipped passage); non-seekable formats are decoded to a
    float WAV for the views."""
    info = sf.info(path)
    sr, C = info.samplerate, info.channels
    seekable = info.format in SEEKABLE
    view_path = path
    writer = None
    if not seekable:
        os.makedirs(work, exist_ok=True)
        view_path = os.path.join(work, "input.wav")
        writer = sf.SoundFile(view_path, "w", sr, C, "FLOAT", format="WAV")
    B = max(1, int(round(0.01 * sr)))
    maxes, n = [], 0
    try:
        with sf.SoundFile(path) as f:
            for blk in f.blocks(blocksize=B * 512, dtype="float32", always_2d=True):
                a = np.abs(blk)
                nb = -(-len(a) // B)
                if len(a) < nb * B:
                    a = np.concatenate([a, np.zeros((nb * B - len(a), C), np.float32)])
                maxes.append(a.reshape(nb, B, C).max(1))
                if writer is not None:
                    writer.write(blk)
                n += len(blk)
    finally:
        if writer is not None:
            writer.close()
    if n == 0:
        raise ValueError("the file contains no audio")
    bmax = np.concatenate(maxes)
    peak = float(bmax.max())
    b, ch = _busiest_peak(bmax, peak, max(1, int(round(VIEW_S * sr / B))))
    pos = min(b * B + B // 2, n - 1)
    return InputFile(path, os.path.basename(path), sr, C, n, info.format, info.subtype, view_path, peak, pos, ch)


@dataclass
class Run:
    id: int
    preset: str
    mode: str
    label: str          # what was restored, e.g. 'auto clip+soft' (as in the CLI's output names)
    raw_path: str       # float32 WAV of the restoration
    peak: float
    seconds: float
    device: str
    flagged: float      # fraction of samples restored
    report: str
    rmap: RestorationMap
    exports: dict = field(default_factory=dict)

    @property
    def name(self):
        return f"#{self.id} {self.preset} · {self.label}"

    def choice(self):
        return f"#{self.id}  {self.preset} · {self.label} · peak {db(self.peak):+.1f} dBFS"


class Session:
    def __init__(self, root):
        self.root = root
        self.id = uuid.uuid4().hex[:16]
        self.lock = threading.RLock()
        self.cancel = threading.Event()        # Stop in the single-file view
        self.cancel_batch = threading.Event()  # Stop in batch mode
        self._dir = None
        self._clips = []
        self.tab = "full"         # sub-tab on view: 'full' (full length) or 'snip' (snippets)
        self.loads = 0            # inputs loaded so far (tells the players a new file from a new version)
        self.had = set()          # the files loaded since the input was last cleared
        self._reset()

    def __deepcopy__(self, memo):  # gr.State deep-copies its value; a session is one live object
        return self

    def _reset(self):
        self.input = None
        self.runs = []
        self.next_id = 1
        self.snippets = []
        self.snip_idx = 0
        self.custom = None        # center sample of a typed-in time, overrides the snippet
        self.channel = 0
        self.selected = None      # id of the result on view
        self.specs = OrderedDict()
        self.ov_specs = OrderedDict()
        self.envs = {}            # ('y' | 'x' | 'd', result id, channel) -> full-length waveform envelope
        self.full_files = OrderedDict()
        self.peaks = {}           # (result id or 0 = input, sample, channel) -> local peak
        self.shown = {}           # what each image / player on the page shows (see view_keys)

    # ---- files ----------------------------------------------------------------------------------
    @property
    def dir(self):
        if self._dir is None:
            self._dir = os.path.join(self.root, self.id)
            os.makedirs(self._dir, exist_ok=True)
            _created_dirs.add(self._dir)
        return self._dir

    @property
    def single_dir(self):
        return os.path.join(self.dir, "single")

    def close(self):
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
            _created_dirs.discard(self._dir)
            self._dir = None

    def load(self, path):
        """Open a new input (the results of the previous one are discarded; a run of it stops at its next
        step). Until a result restores something, the view shows the input's own peak, in full length."""
        with self.lock:
            shutil.rmtree(self.single_dir, ignore_errors=True)
            self._reset()
            self.loads += 1
            inp = self.input = open_input(path, self.single_dir)
            self.snippets = [Snippet(inp.peak_pos, inp.peak_ch, inp.peak, inp.peak, fallback=True)]
            self.channel = inp.peak_ch
            self.tab = "full"
            self.had.add(os.path.normcase(os.path.abspath(path)))
            return inp

    def load_if_new(self, path):
        """load(path) unless it is the input already: the new InputFile, or None. The upload event and
        Declip (pressed during the upload) can both get here first; the other one waits and gets None."""
        with self.lock:
            if self.input is not None and same_file(self.input.path, path):
                return None
            return self.load(path)

    def take(self, path):
        """The input of a run that was asked for with `path`: 'ok' if it is the input, 'loaded' if it was
        loaded now (Declip came before the upload event), 'stale' if the input changed since (path is an
        earlier input: the run waited in the queue)."""
        with self.lock:
            if self.input is not None and same_file(self.input.path, path):
                return "ok"
            if os.path.normcase(os.path.abspath(path)) in self.had:
                return "stale"
            self.load(path)
            return "loaded"

    def clear(self):
        with self.lock:
            shutil.rmtree(self.single_dir, ignore_errors=True)
            self._reset()
            self.had.clear()

    # ---- results --------------------------------------------------------------------------------
    def process(self, settings, progress=None):
        """Declip the input with these settings; the result is appended to the history and selected.
        Raises Cancelled when Stop was pressed or another file was loaded meanwhile."""
        inp = self.input
        if inp is None:
            raise ValueError("no input")

        def step(i, n, el):
            if self.input is not inp:
                raise Cancelled()
            if progress:
                progress(i, n, el)

        y, sr = sf.read(inp.path, dtype="float64", always_2d=True)
        x, info, label = declip_array(y, sr, settings, step, self.cancel)
        with self.lock:
            if self.input is not inp:
                raise Cancelled()
            rid = self.next_id
            self.next_id += 1
        raw = os.path.join(self.single_dir, f"run{rid}", output_name(inp.stem, label, settings.preset, "wav32f"))
        write_audio(x, sr, raw, "wav32f")
        run = Run(rid, settings.preset, settings.mode, label, raw, float(np.abs(x).max()), info["time"],
                  info.get("device", "cpu"), info["clipped_frac"],
                  "\n".join(report_lines(info, y.shape[1], settings.preset)), RestorationMap(y, x, sr))
        del x, y
        with self.lock:
            if self.input is not inp:
                raise Cancelled()
            self.runs.append(run)
            self.selected = rid
            if not self.snippets or (self.snippets[0].fallback and run.rmap.restored):
                # the first restoring result ranks the snippets; a moment picked by hand stays on view
                self.rank_from(run, keep_view=self.custom is not None)
        return run

    def run(self, rid=None):
        with self.lock:
            rid = self.selected if rid is None else rid
            return next((r for r in self.runs if r.id == rid), None)

    def select(self, rid):
        with self.lock:
            if self.run(rid) is not None:
                self.selected = rid

    def remove(self, rid):
        with self.lock:
            run = self.run(rid)
            if run is None:
                return
            self.runs.remove(run)
            if self.selected == rid:
                self.selected = self.runs[-1].id if self.runs else None
            stale = [self.full_files.pop(k) for k in [k for k in self.full_files if k[1] == rid]]
            for store, i in ((self.specs, 1), (self.ov_specs, 1), (self.envs, 1), (self.peaks, 0)):
                for k in [k for k in store if k[i] == rid]:
                    del store[k]
        shutil.rmtree(os.path.dirname(run.raw_path), ignore_errors=True)
        for p in stale:
            shutil.rmtree(os.path.dirname(p), ignore_errors=True)

    def history(self):
        with self.lock:
            return [(r.choice(), r.id) for r in self.runs], self.selected

    def peak_ref(self):
        """Top of the waveform scale: the highest peak of all results (and the input)."""
        with self.lock:
            return max([self.input.peak] + [r.peak for r in self.runs]) or 1.0

    def export(self, rid, out):
        """(path, note) of a result in the chosen output format, written once per format and gain."""
        run = self.run(rid)
        gain, note = output_gain(run.peak, out)
        key = (out.fmt, round(gain, 9))
        path = run.exports.get(key)
        if path and os.path.exists(path):
            return path, note
        if out.fmt == "wav32f" and gain == 1.0:
            path = run.raw_path
        else:
            x, sr = sf.read(run.raw_path, dtype="float64", always_2d=True)
            sub = out.fmt if gain == 1.0 else f"{out.fmt}_{db(gain):+.2f}dB"
            path = os.path.join(os.path.dirname(run.raw_path), sub,
                                output_name(self.input.stem, run.label, run.preset, out.fmt))
            write_audio(x, sr, path, out.fmt, gain)
        run.exports[key] = path
        return path, note

    # ---- snippets -------------------------------------------------------------------------------
    def rank_from(self, run, keep_view=False):
        """Rank the snippets by this result and show the first one (keep_view: stay on the moment on view)."""
        with self.lock:
            inp = self.input
            sn = rank(run.rmap, inp.sr)
            if not sn:  # nothing restored: look at the input's peak
                sn = [Snippet(inp.peak_pos, inp.peak_ch, inp.peak, inp.peak, fallback=True)]
            self.snippets, self.snip_idx = sn, 0
            if not keep_view:
                self.custom = None
                self.channel = sn[0].channel

    def goto(self, i):
        with self.lock:
            if self.snippets:
                self.snip_idx = int(np.clip(i, 0, len(self.snippets) - 1))
                self.custom = None
                self.channel = self.snippets[self.snip_idx].channel

    def goto_time(self, seconds):
        with self.lock:
            self.custom = int(np.clip(round(seconds * self.input.sr), 0, self.input.frames - 1))

    def _peak_at(self, run, s):
        """Peak of the input (run None) or of a result within 2 ms of a snippet's peak."""
        key = (run.id if run else 0, s.center, s.channel)
        v = self.peaks.get(key)
        if v is None:
            inp = self.input
            w = max(1, int(0.002 * inp.sr))
            seg = read_segment(run.raw_path if run else inp.view_path, s.center - w, s.center + w + 1,
                               inp.frames, inp.channels)
            v = self.peaks[key] = float(np.abs(seg[:, s.channel]).max())
        return v

    def snippet_choices(self):
        """Dropdown entries: the snippets with the peak level of the input and of the result on view there
        (the ranking may come from another result)."""
        with self.lock:
            inp, run = self.input, self.run()
            names = channel_names(inp.channels)
            out = []
            if self.custom is not None:
                out.append((f"Custom · {R.fmt_time(self.custom / inp.sr)}", "custom"))
            n = len(self.snippets)
            for i, s in enumerate(self.snippets):
                t = R.fmt_time(s.center / inp.sr)
                if s.fallback:
                    lab = f"Input peak · {t} · {names[s.channel]}" + (" (nothing was restored)" if self.runs else "")
                else:
                    lab = f"{i + 1}/{n} · {t} · {names[s.channel]} · peak {db(self._peak_at(None, s)):+.1f}"
                    if run is not None:
                        lab += f" → {db(self._peak_at(run, s)):+.1f} dBFS (#{run.id})"
                    else:
                        lab += " dBFS"
                out.append((lab, str(i)))
            return out, ("custom" if self.custom is not None else str(self.snip_idx))

    def region(self):
        """(start, length) in samples of the view: 3 s centered on the snippet, inside the file."""
        with self.lock:
            inp = self.input
            n = min(int(round(VIEW_S * inp.sr)), inp.frames)
            if self.custom is not None:
                center = self.custom
            elif self.snippets:
                center = self.snippets[self.snip_idx].center
            else:
                center = inp.peak_pos
            return int(np.clip(center - n // 2, 0, inp.frames - n)), n

    # ---- views ----------------------------------------------------------------------------------
    def _signals(self, view, run, a, b):
        """(input, shown signal) for samples [a, b) of all channels."""
        inp = self.input
        y = read_segment(inp.view_path, a, b, inp.frames, inp.channels)
        if view == "Before" or run is None:
            return y, y
        x = read_segment(run.raw_path, a, b, inp.frames, inp.channels)
        return y, (x if view == "After" else x - y)

    def render(self, view, scale):
        """The view as an RGB image: view 'Before', 'After' or 'Delta', scale 'log' or 'linear'."""
        with self.lock:
            inp, run, ch, peak = self.input, self.run(), self.channel, self.peak_ref()
            start, n = self.region()
        if run is None:
            view = "Before"
        sr = inp.sr
        pad = R.context_samples(sr)
        y, sig = self._signals(view, run, start - pad, start + n + pad)
        y, sig = y[:, ch], sig[:, ch]
        key = (view, run.id if view != "Before" else 0, start, n, ch, scale)
        with self.lock:
            spec = self.specs.get(key)
            if spec is not None:
                self.specs.move_to_end(key)
        if spec is None:
            spec = R.view_spectrogram(sig, pad, n, sr, scale).astype(np.float16)
            with self.lock:
                self.specs[key] = spec
                while len(self.specs) > SPEC_CACHE:
                    self.specs.popitem(last=False)
        yv, sv = y[pad:pad + n], sig[pad:pad + n]
        if view == "Before":
            layers, title = [(yv, R.BLUE)], "BEFORE  ·  input"
        elif view == "After":
            layers, title = [(sv, R.ORANGE), (yv, R.BLUE)], f"AFTER  ·  {run.name}"
        else:
            layers, title = [(yv, R.GHOST), (sv, R.ORANGE)], f"DELTA  ·  {run.name}  ·  after - before"
        info = (f"{channel_names(inp.channels)[ch]}  ·  {R.fmt_time(start / sr)} - "
                f"{R.fmt_time((start + n) / sr)}")
        return R.render_view(spec.astype(np.float32), layers, sr, start / sr, n, peak, scale, title=title,
                             info=info, title_color=R.VIEW_COLORS[view])

    def snippet_audio(self, view):
        """(WAV path, start, length) of the view (all channels) for listening. All versions share one gain,
        turned down by the highest peak of all results when it exceeds 0 dBFS, so they compare at their true
        levels and nothing clips."""
        with self.lock:
            inp, run, peak = self.input, self.run(), self.peak_ref()
            start, n = self.region()
        if run is None:
            view = "Before"
        _, sig = self._signals(view, run, start, start + n)
        sig = sig * np.float32(1.0 / max(peak, 1.0))
        f = min(int(FADE_S * inp.sr), n // 4)
        if f > 0:
            ramp = np.linspace(0, 1, f, dtype=np.float32)[:, None]
            sig[:f] *= ramp
            sig[-f:] *= ramp[::-1]
        d = os.path.join(self.dir, "clips", str(time.time_ns()))  # a new URL each time (no stale browser cache)
        os.makedirs(d, exist_ok=True)
        tag = "before" if view == "Before" else f"{view.lower()} result {run.id}"
        path = os.path.join(d, f"{inp.stem} {R.fmt_time(start / inp.sr).replace(':', '-')} {tag}.wav")
        sf.write(path, sig, inp.sr, subtype="PCM_16")
        with self.lock:
            self._clips.append(d)
            old, self._clips = self._clips[:-6], self._clips[-6:]
        for p in old:
            shutil.rmtree(p, ignore_errors=True)
        return path, start, n

    # ---- full length ----------------------------------------------------------------------------
    def _reader(self, view, run, ch):
        """read(a, b) of one channel of the input, a result or their difference, zeros outside the file."""
        inp = self.input
        ry = lambda a, b: read_segment(inp.view_path, a, b, inp.frames, inp.channels)[:, ch]
        if view == "Before" or run is None:
            return ry
        rx = lambda a, b: read_segment(run.raw_path, a, b, inp.frames, inp.channels)[:, ch]
        return rx if view == "After" else (lambda a, b: rx(a, b) - ry(a, b))

    def _envelope(self, view, run, ch):
        """Full-length waveform envelope of the input ('Before'), a result ('After') or the delta."""
        key = (view, run.id if run is not None and view != "Before" else 0, ch)
        with self.lock:
            env = self.envs.get(key)
        if env is None:
            env = R.envelope_stream(self._reader(view, run, ch), self.input.frames)
            with self.lock:
                self.envs[key] = env
        return env

    def render_overview(self, view, scale):
        """The whole file as an RGB image, with a numbered flag per snippet and the snippet view's region
        tinted."""
        with self.lock:
            inp, run, ch, peak = self.input, self.run(), self.channel, self.peak_ref()
            start, n = self.region()
            snippets, idx, custom = list(self.snippets), self.snip_idx, self.custom
        if run is None:
            view = "Before"
        sr = inp.sr
        key = (view, run.id if view != "Before" else 0, ch, scale)
        with self.lock:
            spec = self.ov_specs.get(key)
            if spec is not None:
                self.ov_specs.move_to_end(key)
        if spec is None:
            spec = R.overview_spectrogram(self._reader(view, run, ch), inp.frames, sr, scale).astype(np.float16)
            with self.lock:
                self.ov_specs[key] = spec
                while len(self.ov_specs) > OV_CACHE:
                    self.ov_specs.popitem(last=False)
        ey = self._envelope("Before", run, ch)
        if view == "Before":
            layers, title = [(ey, R.BLUE)], "BEFORE  ·  input  ·  full length"
        elif view == "After":
            layers, title = [(self._envelope("After", run, ch), R.ORANGE), (ey, R.BLUE)], f"AFTER  ·  {run.name}"
        else:
            layers = [(ey, R.GHOST), (self._envelope("Delta", run, ch), R.ORANGE)]
            title = f"DELTA  ·  {run.name}  ·  after - before"
        markers = [] if not snippets or snippets[0].fallback else \
            [(s.center / sr, str(i + 1), custom is None and i == idx) for i, s in enumerate(snippets)]
        dur = inp.frames / sr
        dec = 0 if dur >= 60 else 3
        info = f"{channel_names(inp.channels)[ch]}  ·  {R.fmt_time(0, dec)} - {R.fmt_time(dur, dec)}"
        return R.render_view(spec.astype(np.float32), layers, sr, 0.0, inp.frames, peak, scale, title=title,
                             info=info, title_color=R.VIEW_COLORS[view], markers=markers,
                             region=(start / sr, (start + n) / sr))

    def playback_gain_db(self):
        """Gain of both players: turned down by the highest peak of all results when it exceeds 0 dBFS."""
        return -max(db(self.peak_ref()), 0.0)

    def full_audio(self, view):
        """Full-length 16-bit WAV of the view (all channels) for the player, at the snippet player's gain."""
        with self.lock:
            inp, run = self.input, self.run()
            gain_db = self.playback_gain_db()
        if run is None:
            view = "Before"
        key = (view, run.id if view != "Before" else 0, round(gain_db, 6))
        with self.lock:
            path = self.full_files.get(key)
        if path and os.path.exists(path):
            return path
        d = os.path.join(self.single_dir, "full", str(time.time_ns()))
        os.makedirs(d, exist_ok=True)
        tag = "before" if view == "Before" else f"{view.lower()} result {run.id}"
        path = os.path.join(d, f"{inp.stem} {tag}.wav")
        g = np.float32(10 ** (gain_db / 20))
        with sf.SoundFile(path, "w", inp.sr, inp.channels, "PCM_16", format="WAV") as f:
            for a in range(0, inp.frames, 1 << 20):
                b = min(inp.frames, a + (1 << 20))
                _, sig = self._signals(view, run, a, b)
                f.write(sig * g)
        with self.lock:
            self.full_files[key] = path
            old = []
            while len(self.full_files) > FULL_AUDIO_KEEP:
                old.append(self.full_files.popitem(last=False)[1])
        for p in old:
            shutil.rmtree(os.path.dirname(p), ignore_errors=True)
        return path

    def player(self, kind, view):
        """What a player plays, the full length (kind 'full') or the snippet ('snip') of a view: {path, name,
        label, note, tl, file, t0, dur, dl}. tl names the timeline (other views of it play from the same
        position), file the input; t0 and dur are in seconds."""
        with self.lock:
            inp, run, loads = self.input, self.run(), self.loads
            gain_db = self.playback_gain_db()
        v = view if run is not None else "Before"
        if kind == "full":
            path, start, n = self.full_audio(v), 0, inp.frames
            what, tl = "Full length", str(loads)
        else:
            path, start, n = self.snippet_audio(v)
            what, tl = "Snippet", f"{loads}:{start}:{n}"
        which = "Before · the input" if v == "Before" else f"{v} · {run.name}"
        note = f"played at {gain_db:+.1f} dB, so the restored peaks don't clip" if gain_db < -0.05 else ""
        return {"path": path, "name": os.path.basename(path), "label": f"{what} · {which}", "note": note,
                "tl": tl, "file": str(loads), "t0": start / inp.sr, "dur": n / inp.sr, "dl": kind == "snip"}

    def goto_x(self, x):
        """A click on the overview at column x: the snippet whose flag is there, else that moment."""
        with self.lock:
            inp = self.input
            if self.snippets and not self.snippets[0].fallback:
                xs = np.array([s.center for s in self.snippets]) / inp.frames * R.PLOT_W
                i = int(np.argmin(np.abs(xs - x)))
                if abs(xs[i] - x) <= SNAP_PX:
                    self.goto(i)
                    return
            self.goto_time(float(np.clip(x / R.PLOT_W, 0.0, 1.0)) * inp.frames / inp.sr)

    def view_keys(self, view, scale):
        """What each image and player shows, so unchanged ones are not rendered and sent again (and a playing
        full-length player is not restarted by navigating the snippets)."""
        with self.lock:
            run = self.run()
            v = view if run is not None else "Before"
            rid = run.id if run is not None and v != "Before" else 0
            peak = round(self.peak_ref(), 9)
            start, n = self.region()
            snips = tuple((s.center, s.channel, s.fallback) for s in self.snippets)
            return {"ov_img": (v, rid, self.channel, scale, peak, snips, self.snip_idx, self.custom, start),
                    "ov_audio": (v, rid, peak),
                    "sn_img": (v, rid, start, n, self.channel, scale, peak),
                    "sn_audio": (v, rid, start, n, peak)}
