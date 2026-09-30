"""RX-style snippet view: a waveform with a dB amplitude ruler above a spectrogram (log or linear
frequency axis) above an h:m:s time ruler, rendered straight onto the pixel grid with numpy + Pillow.

The spectrogram is multi-resolution: several Hann-window STFTs (8x time overlap, up to 8x zero padding),
each used for the rows it resolves best. For every row the window length is the one whose time blur
(in columns) matches its frequency blur (in rows), so low notes get long windows and sharp harmonics,
and highs get short windows and sharp transients. Bins and frames are reduced onto rows and columns
with a max, so thin harmonics and clicks survive the reduction instead of being averaged away.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import fft as sfft

# ---- layout (pixels) ----------------------------------------------------------------------------
PLOT_W = 1224          # width of the waveform / spectrogram area
RULER_W = 58           # right-hand ruler (dB for the waveform, Hz for the spectrogram)
HEADER_H = 26
WAVE_H = 150
GAP_H = 3
SPEC_H = 420
TIME_H = 26
WIDTH = PLOT_W + RULER_W
HEIGHT = HEADER_H + WAVE_H + GAP_H + SPEC_H + TIME_H
WAVE_Y = HEADER_H
SPEC_Y = HEADER_H + WAVE_H + GAP_H
TIME_Y = SPEC_Y + SPEC_H

# ---- colors -------------------------------------------------------------------------------------
BG = (24, 26, 31)
WAVE_BG = (13, 15, 19)
GRID = (40, 44, 52)
ZERO_DBFS = (120, 60, 60)      # the 0 dBFS line (only visible when restored peaks exceed it)
TEXT = (196, 201, 210)
TEXT_DIM = (128, 134, 145)
TICK = (92, 98, 110)
BLUE = (88, 158, 228)          # the input waveform
ORANGE = (245, 158, 58)        # what the restoration added
GHOST = (44, 70, 102)          # the input behind the delta

VIEW_COLORS = {"Before": (120, 176, 236), "After": ORANGE, "Delta": (238, 110, 90)}

# ---- color map: "cyan to orange", RX-like (black - navy - blue - cyan - pale - orange - yellow - white)
CMAP_STOPS = [
    (0.00, (0, 0, 0)),
    (0.10, (3, 6, 24)),
    (0.20, (8, 20, 62)),
    (0.32, (16, 52, 128)),
    (0.44, (24, 110, 188)),
    (0.57, (36, 172, 216)),
    (0.67, (146, 200, 204)),
    (0.74, (212, 186, 150)),
    (0.81, (244, 160, 76)),
    (0.88, (252, 138, 36)),
    (0.95, (255, 204, 92)),
    (1.00, (255, 248, 226)),
]


def _build_lut(stops, n=1024):
    pos = np.array([p for p, _ in stops])
    col = np.array([c for _, c in stops], float)
    t = np.linspace(0, 1, n)
    return np.stack([np.interp(t, pos, col[:, k]) for k in range(3)], 1).round().astype(np.uint8)


LUT = _build_lut(CMAP_STOPS)

SPEC_TOP_DB = -6.0           # top of the color map (0 dB = full-scale sine; dense music peaks near -10 dB)
DEFAULT_RANGE_DB = 114.0     # color map spans [top - range, top]
TIME_OVERLAP = 8             # hop = window / 8 (or half a column, if that is longer)
ZERO_PAD = 8                 # FFT size = up to 8 x window (less where the rows are much wider than a bin)
WIN_MIN_S = 0.0058           # shortest window (256 samples at 44.1 kHz)
WIN_MAX_S = 0.186            # longest window (8192 samples at 44.1 kHz)
FREQ_BIAS = 1.4              # >1 favors frequency over time resolution (sharper harmonics)
LOG_FMIN = 20.0


def window_sizes(sr):
    """Power-of-two window lengths from ~5.8 ms to ~186 ms at this sample rate."""
    lo = int(round(np.log2(WIN_MIN_S * sr)))
    hi = int(round(np.log2(WIN_MAX_S * sr)))
    return [2 ** k for k in range(max(lo, 6), max(hi, 7) + 1)]


def context_samples(sr):
    """Samples of real audio (or zeros at the file ends) needed on each side of the view."""
    return window_sizes(sr)[-1]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


FONT = _font(13)
FONT_HEAD = _font(14)
HEADER_RIGHT = WIDTH - 110   # the header's right-hand text ends here (clear of the viewer's overlay buttons)


# ---- spectrogram --------------------------------------------------------------------------------
def _row_edges(sr, height, scale):
    fmax = sr / 2
    if scale == "log":
        fmin = min(LOG_FMIN, fmax / 100)
        edges = fmin * (fmax / fmin) ** (np.arange(height + 1) / height)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(0.0, fmax, height + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _reduce_segments(v, lo, hi, centers_idx, axis_len):
    """Max of v (N, M) over index segments [lo, hi) along axis 0 where they are non-empty, linear
    interpolation at the fractional positions centers_idx where they are empty. Segments must be
    contiguous and increasing (they partition a range)."""
    out = np.empty((len(lo), v.shape[1]), v.dtype)
    has = hi > lo
    if has.any():
        a, b = lo[has][0], hi[has][-1]
        out[has] = np.maximum.reduceat(v[a:b], lo[has] - a, axis=0)
    if (~has).any():
        c = np.clip(centers_idx[~has], 0, axis_len - 1)
        k0 = np.minimum(np.floor(c).astype(int), axis_len - 2)
        t = (c - k0)[:, None]
        out[~has] = v[k0] * (1 - t) + v[k0 + 1] * t
    return out


def spectrogram_db(sig, pad, n_view, sr, width, height, scale="log"):
    """dB image (height, width), row 0 = highest frequency, 0 dB = full-scale sine.

    sig: `pad` samples of context + the n_view displayed samples + `pad` samples of context."""
    sig = np.asarray(sig, np.float32)
    edges, centers = _row_edges(sr, height, scale)
    row_hz = np.diff(edges)
    spp = n_view / width
    # equal blur in pixels: time ~ N/2 samples / spp, frequency ~ 2 sr/N Hz / row_hz
    ideal = FREQ_BIAS * 2.0 * np.sqrt(sr * spp / row_hz)
    sizes = window_sizes(sr)
    lg = np.clip(np.log2(ideal), np.log2(sizes[0]), np.log2(sizes[-1]))
    img = np.zeros((height, width), np.float32)
    col_edges = np.linspace(0.0, n_view, width + 1)
    col_centers = 0.5 * (col_edges[:-1] + col_edges[1:])
    for n_win in sizes:
        wrow = np.clip(1.0 - np.abs(lg - np.log2(n_win)), 0.0, 1.0)
        rows = np.flatnonzero(wrow > 0)
        if rows.size == 0:
            continue
        # 8x time overlap, but no more than ~2 frames per column (they are max-reduced onto columns)
        hop = max(1, n_win // TIME_OVERLAP, int(spp / 2))
        # zero padding: up to 8x, enough for ~4 bins across the narrowest row of this window
        pad_f = 1
        while pad_f < ZERO_PAD and sr / (n_win * pad_f) > row_hz[rows].min() / 4:
            pad_f *= 2
        nfft = n_win * pad_f
        df = sr / nfft
        # only the frames that can reach the view
        a = max(0, pad - n_win // 2 - hop)
        b = min(len(sig), pad + n_view + n_win // 2 + hop)
        x = sig[a:b]
        if len(x) < n_win:
            x = np.pad(x, (0, n_win - len(x)))
        win = np.hanning(n_win + 1)[:-1].astype(np.float32)  # periodic Hann
        frames = np.lib.stride_tricks.sliding_window_view(x, n_win)[::hop] * win
        # only the bins these rows need
        k_lo = int(np.floor(edges[rows[0]] / df))
        k_hi = min(int(np.ceil(edges[rows[-1] + 1] / df)) + 2, nfft // 2 + 1)
        spec = sfft.rfft(frames, n=nfft, axis=1, workers=-1)[:, k_lo:k_hi]
        mag = np.abs(spec) * np.float32(2.0 / win.sum())
        db = 20.0 * np.log10(np.maximum(mag, 1e-9))                    # (frames, bins)
        # bins -> rows
        blo = np.clip(np.ceil(edges[rows] / df).astype(int) - k_lo, 0, db.shape[1])
        bhi = np.clip(np.ceil(edges[rows + 1] / df).astype(int) - k_lo, 0, db.shape[1])
        vr = _reduce_segments(db.T, blo, bhi, centers[rows] / df - k_lo, db.shape[1])  # (rows, frames)
        # frames -> columns
        pos = a - pad + np.arange(vr.shape[1]) * hop + n_win / 2.0     # frame centers, view coordinates
        flo = np.searchsorted(pos, col_edges[:-1])
        fhi = np.searchsorted(pos, col_edges[1:])
        cidx = np.interp(col_centers, pos, np.arange(len(pos)))
        vc = _reduce_segments(vr.T, flo, fhi, cidx, vr.shape[1])     # (width, rows)
        img[rows] += wrow[rows, None] * vc.T
    return img[::-1]


def colorize(db, range_db=DEFAULT_RANGE_DB, top_db=SPEC_TOP_DB):
    t = np.clip((db - (top_db - range_db)) / range_db, 0.0, 1.0)
    return LUT[(t * (len(LUT) - 1)).astype(np.int32)]


# ---- waveform -----------------------------------------------------------------------------------
def envelope(x, width):
    """Per-column (min, max) of x over `width` equal columns."""
    x = np.asarray(x, np.float32)
    n = len(x)
    if n == 0:
        z = np.zeros(width, np.float32)
        return z, z
    if n < width:  # fewer samples than columns: nearest sample per column
        idx = np.minimum((np.arange(width) * n / width).astype(int), n - 1)
        return x[idx], x[idx]
    starts = (np.arange(width) * n / width).astype(int)
    return np.minimum.reduceat(x, starts), np.maximum.reduceat(x, starts)


def _amp_to_row(v, peak, height, margin=4):
    half = height / 2.0 - margin
    return height / 2.0 - np.asarray(v) / peak * half


def draw_waveform(layers, width, height, peak):
    """layers: [(signal, rgb), ...] drawn in order (later ones on top). Amplitudes are shown on a
    linear scale where +-peak touches the panel's top and bottom margins."""
    img = np.empty((height, width, 3), np.uint8)
    img[:] = WAVE_BG
    peak = max(float(peak), 1e-9)
    rows = np.arange(height)[:, None]
    for x, color in layers:
        mn, mx = envelope(x, width)
        top = np.floor(_amp_to_row(mx, peak, height)).astype(int)
        bot = np.ceil(_amp_to_row(mn, peak, height)).astype(int)
        bot = np.maximum(bot, top + 1)
        img[(rows >= top[None]) & (rows <= bot[None])] = color
    return img


DB_TICK_TIERS = ([0, -6, -12, 12, 6, -24], [-3, -9, -18, 3, 9, 18], [-1, -2, -4.5, -15, -30, 24, -36, -48, -60])


def db_ticks(peak, height, min_px=15):
    """[(dBFS, row in the positive half)] for the amplitude ruler: round values first (0, -6, -12, ...),
    then finer ones where they fit."""
    half = height / 2.0
    out = []
    for tier in DB_TICK_TIERS:
        for d in tier:
            v = 10 ** (d / 20)
            if v > peak * 1.0001:
                continue
            r = float(_amp_to_row(v, peak, height))
            if half - r < min_px or any(abs(r - q) < min_px for _, q in out):
                continue
            out.append((d, r))
    return sorted(out, key=lambda t: t[1])


# ---- rulers -------------------------------------------------------------------------------------
def fmt_time(t, decimals=3):
    """h:mm:ss.fff"""
    t = max(0.0, t)
    scale = 10 ** decimals
    total = int(round(t * scale))
    h, rem = divmod(total, 3600 * scale)
    m, rem = divmod(rem, 60 * scale)
    s, frac = divmod(rem, scale)
    return f"{h}:{m:02d}:{s:02d}" + (f".{frac:0{decimals}d}" if decimals else "")


def _time_step(span_s, width, min_px=110):
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600):
        if step / span_s * width >= min_px:
            return step
    return 1200


def _freq_ticks(sr, scale):
    fmax = sr / 2
    if scale == "log":
        c = [20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000,
             20000, 30000, 40000]
    else:
        step = 2000 if fmax <= 24000 else 5000 if fmax <= 50000 else 10000
        c = list(np.arange(0, fmax + 1, step))
    return [f for f in c if f <= fmax]


def _fmt_hz(f):
    if f >= 1000:
        v = f / 1000
        return f"{v:g}k"
    return f"{int(f)}"


def _freq_row(f, sr, height, scale):
    fmax = sr / 2
    if scale == "log":
        fmin = min(LOG_FMIN, fmax / 100)
        if f < fmin:
            return None
        pos = np.log(f / fmin) / np.log(fmax / fmin)
    else:
        pos = f / fmax
    return (1.0 - pos) * height


def view_spectrogram(spec_sig, pad, n_view, sr, scale="log"):
    """The spectrogram (dB) of a view: the slow part of render_view, worth caching."""
    return spectrogram_db(spec_sig, pad, n_view, sr, PLOT_W, SPEC_H, scale)


def render_view(db, wave_layers, sr, t0, n_view, peak, scale="log", range_db=DEFAULT_RANGE_DB,
                title="", info="", title_color=TEXT):
    """Full RGB image (HEIGHT, WIDTH, 3).

    db: view_spectrogram(...) of the shown signal; wave_layers: [(signal of n_view samples, rgb)] for the
    waveform, drawn in order; t0: start of the view in seconds (ruler); peak: amplitude at the top of the
    waveform panel (linear, 1.0 = 0 dBFS)."""
    canvas = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    canvas[:] = BG
    wave = draw_waveform(wave_layers, PLOT_W, WAVE_H, peak)
    ticks = db_ticks(peak, WAVE_H)
    for d, r in ticks:  # grid lines at the dB ticks (both halves), behind the waveform
        for rr in (int(round(r)), int(round(WAVE_H - r))):
            if 0 <= rr < WAVE_H:
                line = wave[rr]
                bg = np.all(line == WAVE_BG, axis=1)
                line[bg] = ZERO_DBFS if d == 0 else GRID
    mid = WAVE_H // 2
    bg = np.all(wave[mid] == WAVE_BG, axis=1)
    wave[mid][bg] = GRID
    canvas[WAVE_Y:WAVE_Y + WAVE_H, :PLOT_W] = wave
    canvas[SPEC_Y:SPEC_Y + SPEC_H, :PLOT_W] = colorize(db, range_db)

    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    # header
    dr.text((8, HEADER_H / 2), title, fill=title_color, font=FONT_HEAD, anchor="lm")
    dr.text((HEADER_RIGHT, HEADER_H / 2), info, fill=TEXT_DIM, font=FONT_HEAD, anchor="rm")
    # amplitude ruler
    x0 = PLOT_W + 6
    for d, r in ticks:
        lab = f"{d:+g}" if d > 0 else f"{d:g}"
        for rr in (r, WAVE_H - r):
            dr.line([(PLOT_W, WAVE_Y + rr), (PLOT_W + 4, WAVE_Y + rr)], fill=TICK)
            dr.text((x0, WAVE_Y + rr), lab, fill=TEXT if d == 0 else TEXT_DIM, font=FONT, anchor="lm")
    dr.text((x0, WAVE_Y + mid), "-inf", fill=TEXT_DIM, font=FONT, anchor="lm")
    # frequency ruler
    last = -1e9
    for f in _freq_ticks(sr, scale):
        r = _freq_row(f, sr, SPEC_H, scale)
        if r is None or r < 6 or r > SPEC_H - 6 or abs(r - last) < 22:
            continue
        last = r
        dr.line([(PLOT_W, SPEC_Y + r), (PLOT_W + 4, SPEC_Y + r)], fill=TICK)
        dr.text((x0, SPEC_Y + r), _fmt_hz(f), fill=TEXT_DIM, font=FONT, anchor="lm")
    dr.text((PLOT_W + RULER_W - 4, SPEC_Y + SPEC_H - 2), "Hz", fill=TICK, font=FONT, anchor="rb")
    # time ruler
    span = n_view / sr
    step = _time_step(span, PLOT_W)
    minor = step / 5
    decimals = 3 if step < 1 else 0
    k = np.ceil(t0 / minor - 1e-9)
    while True:
        t = k * minor
        if t > t0 + span + 1e-9:
            break
        x = (t - t0) / span * PLOT_W
        major = abs(t / step - round(t / step)) < 1e-6
        dr.line([(x, TIME_Y), (x, TIME_Y + (7 if major else 3))], fill=TICK)
        if major:
            lab = fmt_time(t, decimals)
            w = dr.textlength(lab, font=FONT)
            xl = min(max(x, w / 2 + 2), PLOT_W - w / 2 - 2)
            dr.text((xl, TIME_Y + 9), lab, fill=TEXT_DIM, font=FONT, anchor="mt")
        k += 1
    return np.asarray(im)
