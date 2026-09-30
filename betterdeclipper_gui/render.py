"""RX-style views: a waveform with a dB amplitude ruler above a spectrogram (linear or RX's log frequency
axis) above an h:m:s time ruler, rendered straight onto the pixel grid with numpy + Pillow.

Colors, frequency axes and resolutions were fitted to iZotope RX screenshots of the same audio, inverted
to dB through RX's own color bar, so the views read like RX's:
- colors: RX's default gradient, black at -120 dB to white at 0 dB (0 dB = a full-scale sine, as in RX);
- log axis: RX's, linear in ln(1 + f / 100 Hz) from 0 Hz up to Nyquist;
- linear 3 s view: one Hann window of ~5.5 columns (13.6 ms for 3 s over 1224 px; the fit to RX's auto
  STFT: correlation 0.994, 2.5 dB rms), 16x time overlap, 8x zero padding;
- log view: RX's multi-resolution at FFT size 512: 512 / 1024 / 2048-sample windows at 44.1 kHz above
  Nyquist / 8, between Nyquist / 32 and Nyquist / 8, and below;
- full-length overview: one 2048 (linear) or 4096 (log) window, the power of its frames averaged per column.
Bins are reduced onto rows and frames onto columns by their maximum, so thin harmonics and clicks survive.
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
MARKER = (150, 156, 168)       # snippet markers in the overview
MARKER_ON = (255, 255, 255)

VIEW_COLORS = {"Before": (120, 176, 236), "After": ORANGE, "Delta": (238, 110, 90)}

# ---- color map: RX's default gradient, sampled from its color bar every 4 dB (max. error 3/255) --
CMAP_STOPS = [
    (0, (255, 255, 248)), (-4, (254, 255, 225)), (-8, (253, 255, 200)), (-12, (252, 255, 174)),
    (-16, (251, 255, 147)), (-20, (250, 243, 123)), (-24, (249, 226, 102)), (-28, (248, 209, 81)),
    (-32, (246, 189, 67)), (-36, (243, 168, 58)), (-40, (241, 150, 50)), (-44, (238, 132, 43)),
    (-48, (229, 117, 38)), (-52, (214, 104, 35)), (-56, (197, 91, 32)), (-60, (176, 84, 32)),
    (-64, (155, 77, 32)), (-68, (133, 72, 37)), (-72, (112, 68, 43)), (-76, (93, 64, 47)),
    (-80, (77, 60, 51)), (-84, (62, 57, 55)), (-88, (52, 53, 59)), (-92, (43, 48, 64)), (-96, (32, 43, 70)),
    (-100, (18, 33, 66)), (-104, (8, 25, 56)), (-108, (2, 17, 43)), (-112, (0, 11, 30)), (-116, (0, 5, 15)),
    (-120, (0, 0, 0)),
]
TOP_DB = 0.0
FLOOR_DB = -120.0


def _build_lut(stops, n=1024):
    pos = np.array([d for d, _ in stops], float)
    col = np.array([c for _, c in stops], float)
    order = np.argsort(pos)
    t = np.linspace(FLOOR_DB, TOP_DB, n)
    return np.stack([np.interp(t, pos[order], col[order, k]) for k in range(3)], 1).round().astype(np.uint8)


LUT = _build_lut(CMAP_STOPS)

# ---- resolutions (fitted to RX, see the module docstring) -----------------------------------------
LIN_WIN_COLS = 5.5         # linear view: window length in columns
MULTI_N = 512              # log view: window of the top band at 44.1 / 48 kHz (doubled per band below)
MULTI_SPLITS = (8, 32)     # log view: band edges at Nyquist / 8 and Nyquist / 32 ...
XFADE_U = 0.04             # ... cross-faded over +-4 % of the axis height (a hard switch shows a seam)
OVERVIEW_N = {"linear": 2048, "log": 4096}
TIME_OVERLAP = 16
ZERO_PAD = 8
LOG_CORNER_HZ = 100.0      # RX's log axis: linear in ln(1 + f / 100 Hz)
CAL_DB = -0.9              # level offsets to RX (measured): snippet views ...
OVERVIEW_CAL_DB = 2.0      # ... and the overview (RX's column reduction reads a little above a power mean)


def _pow2(n):
    return int(2 ** round(np.log2(max(n, 2))))


def context_samples(sr):
    """Samples of real audio (or zeros at the file ends) needed on each side of a snippet view."""
    return 4 * _pow2(MULTI_N * sr / 44100) + 1024


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


FONT = _font(13)
FONT_HEAD = _font(14)
FONT_SMALL = _font(11)
HEADER_RIGHT = WIDTH - 110   # the header's right-hand text ends here (clear of the viewer's overlay buttons)


# ---- spectrogram --------------------------------------------------------------------------------
def _warp(sr, scale):
    """(f -> u, u -> f): the frequency axis, u = 0 at the bottom (0 Hz) and 1 at the top (Nyquist)."""
    nyq = sr / 2
    if scale == "log":
        span = np.log1p(nyq / LOG_CORNER_HZ)
        return (lambda f: np.log1p(np.asarray(f) / LOG_CORNER_HZ) / span,
                lambda u: LOG_CORNER_HZ * np.expm1(np.asarray(u) * span))
    return (lambda f: np.asarray(f) / nyq), (lambda u: np.asarray(u) * nyq)


def _row_edges(sr, height, scale):
    """Frequency edges (height + 1, bottom row first) and centers of the rows."""
    _, inv = _warp(sr, scale)
    edges = inv(np.arange(height + 1) / height)
    centers = inv((np.arange(height) + 0.5) / height)
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
        k0 = np.clip(np.floor(c).astype(int), 0, max(axis_len - 2, 0))
        k1 = np.minimum(k0 + 1, axis_len - 1)
        t = (c - k0)[:, None]
        out[~has] = v[k0] * (1 - t) + v[k1] * t
    return out


def _stft_rows(x, n_win, hop, pad_f, sr, edges, centers, rows, chunk=512):
    """Magnitudes of the Hann STFT of x (frames starting at 0, hop, 2 hop, ...), reduced onto `rows`: the
    max over the bins inside a row, or the interpolated magnitude at its center where it holds no bin.
    Returns (frames, len(rows)) float32; a full-scale sine reads 1.0 (0 dB)."""
    x = np.asarray(x, np.float32)
    if len(x) < n_win:
        x = np.pad(x, (0, n_win - len(x)))
    win = np.hanning(n_win + 1)[:-1].astype(np.float32)  # periodic Hann
    nfft = n_win * pad_f
    df = sr / nfft
    k_lo = int(np.floor(edges[rows[0]] / df))
    k_hi = min(int(np.ceil(edges[rows[-1] + 1] / df)) + 2, nfft // 2 + 1)
    blo = np.clip(np.ceil(edges[rows] / df).astype(int) - k_lo, 0, k_hi - k_lo)
    bhi = np.clip(np.ceil(edges[rows + 1] / df).astype(int) - k_lo, 0, k_hi - k_lo)
    cidx = centers[rows] / df - k_lo
    frames = np.lib.stride_tricks.sliding_window_view(x, n_win)[::hop]
    gain = np.float32(2.0 / win.sum())
    out = np.empty((len(frames), len(rows)), np.float32)
    for a in range(0, len(frames), chunk):
        spec = sfft.rfft(frames[a:a + chunk] * win, n=nfft, axis=1, workers=-1)[:, k_lo:k_hi]
        mag = np.abs(spec) * gain
        out[a:a + chunk] = _reduce_segments(mag.T, blo, bhi, cidx, mag.shape[1]).T
    return out


def _window_plan(sr, spp, centers, scale):
    """[(window length, rows, weights)] of a snippet view: the rows each window renders and their weight in
    the (dB) image. The log view's bands cross-fade around their edges."""
    rows = np.arange(len(centers))
    if scale == "linear":
        return [(int(np.clip(round(LIN_WIN_COLS * spp / 2) * 2, 128, 16384)), rows, np.ones(len(rows)))]
    n0 = _pow2(MULTI_N * sr / 44100)
    fwd, _ = _warp(sr, "log")
    u = fwd(centers)
    ramp = lambda f: np.clip((u - fwd(f) + XFADE_U) / (2 * XFADE_U), 0.0, 1.0)  # 0 below the edge, 1 above
    w_top = ramp(sr / 2 / MULTI_SPLITS[0])
    w_bot = 1.0 - ramp(sr / 2 / MULTI_SPLITS[1])
    w_mid = np.clip(1.0 - w_top - w_bot, 0.0, 1.0)
    return [(n, rows[w > 1e-6], w[w > 1e-6]) for w, n in ((w_top, n0), (w_mid, 2 * n0), (w_bot, 4 * n0))
            if (w > 1e-6).any()]


def spectrogram_db(sig, pad, n_view, sr, width, height, scale="log"):
    """dB image (height, width) of a snippet view, row 0 = highest frequency, 0 dB = full-scale sine.

    sig: `pad` samples of context + the n_view displayed samples + `pad` samples of context."""
    sig = np.asarray(sig, np.float32)
    edges, centers = _row_edges(sr, height, scale)
    spp = n_view / width
    img = np.full((height, width), CAL_DB, np.float32)
    col_edges = np.linspace(0.0, n_view, width + 1)
    col_centers = 0.5 * (col_edges[:-1] + col_edges[1:])
    for n_win, rows, weights in _window_plan(sr, spp, centers, scale):
        # 16x time overlap, and at least 2 frames per column (they are max-reduced onto columns)
        hop = int(max(1, min(n_win // TIME_OVERLAP, spp / 2)))
        a = max(0, pad - n_win // 2 - hop)
        b = min(len(sig), pad + n_view + n_win // 2 + hop)
        mag = _stft_rows(sig[a:b], n_win, hop, ZERO_PAD, sr, edges, centers, rows)
        db = 20.0 * np.log10(np.maximum(mag, 1e-9))                    # (frames, rows)
        pos = a - pad + np.arange(len(db)) * hop + n_win / 2.0          # frame centers, view coordinates
        flo = np.searchsorted(pos, col_edges[:-1])
        fhi = np.searchsorted(pos, col_edges[1:])
        cidx = np.interp(col_centers, pos, np.arange(len(pos)))
        img[rows] += weights[:, None].astype(np.float32) * _reduce_segments(db, flo, fhi, cidx, len(db)).T
    return img[::-1]


def view_spectrogram(spec_sig, pad, n_view, sr, scale="log"):
    """spectrogram_db of a snippet view at the view's size: the slow part of render_view, worth caching."""
    return spectrogram_db(spec_sig, pad, n_view, sr, PLOT_W, SPEC_H, scale)


def overview_spectrogram(read, n_total, sr, scale="log", width=PLOT_W, height=SPEC_H, block=1 << 20):
    """dB image (height, width) of a whole signal, row 0 = highest frequency. read(a, b) returns its samples
    [a, b) (zeros outside); it is read in blocks, so the signal never has to fit in memory at once."""
    edges, centers = _row_edges(sr, height, scale)
    rows = np.arange(height)
    spp = n_total / width
    n_win = min(_pow2(OVERVIEW_N[scale] * sr / 44100), max(128, _pow2(8 * spp)))  # shorter on short files
    # 4x overlap, at most ~16 frames per column on long files and at least 2 on short ones
    hop = int(max(1, min(max(n_win // 4, spp / 16), spp / 2)))
    nf = n_total // hop + 1                                   # frame j is centered on sample j * hop
    acc = np.zeros((width, height))
    cnt = np.zeros(width)
    per_block = max(1, block // hop)
    for j0 in range(0, nf, per_block):
        j1 = min(nf, j0 + per_block)
        a = j0 * hop - n_win // 2
        x = read(a, (j1 - 1) * hop + n_win - n_win // 2)
        mag = _stft_rows(x, n_win, hop, 2, sr, edges, centers, rows).astype(np.float64)
        cols = np.minimum(np.arange(j0, j1) * hop * width // n_total, width - 1)
        starts = np.flatnonzero(np.r_[True, cols[1:] != cols[:-1]])
        acc[cols[starts]] += np.add.reduceat(mag ** 2, starts, axis=0)
        cnt[cols[starts]] += np.diff(np.r_[starts, len(cols)])
    have = np.flatnonzero(cnt > 0)
    power = acc[have] / cnt[have, None]
    if len(have) < width:  # columns without a frame (tiny files): nearest filled column
        idx = np.clip(np.searchsorted(have, np.arange(width)), 0, len(have) - 1)
        power = power[idx]
    return (10.0 * np.log10(np.maximum(power, 1e-18)) + OVERVIEW_CAL_DB).T[::-1].astype(np.float32)


def colorize(db):
    t = np.clip((db - FLOOR_DB) / (TOP_DB - FLOOR_DB), 0.0, 1.0)
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


def envelope_stream(read, n_total, width=PLOT_W, block=1 << 20):
    """envelope() of a whole signal that is read in blocks with read(a, b)."""
    if n_total < 2 * width:
        return envelope(read(0, n_total), width)
    edges = np.arange(width + 1) * n_total // width
    mins = np.full(width, np.inf, np.float32)
    maxs = np.full(width, -np.inf, np.float32)
    for a in range(0, n_total, block):
        b = min(n_total, a + block)
        x = np.asarray(read(a, b), np.float32)
        c0 = int(np.searchsorted(edges, a, "right")) - 1        # columns overlapping [a, b)
        c1 = int(np.searchsorted(edges, b, "left"))
        starts = np.maximum(edges[c0:c1] - a, 0)
        mins[c0:c1] = np.minimum(mins[c0:c1], np.minimum.reduceat(x, starts))
        maxs[c0:c1] = np.maximum(maxs[c0:c1], np.maximum.reduceat(x, starts))
    return mins, maxs


def _amp_to_row(v, peak, height, margin=4):
    half = height / 2.0 - margin
    return height / 2.0 - np.asarray(v) / peak * half


def draw_waveform(layers, width, height, peak):
    """layers: [(signal or (mins, maxs) envelope, rgb), ...] drawn in order (later ones on top). Amplitudes
    are shown on a linear scale where +-peak touches the panel's top and bottom margins."""
    img = np.empty((height, width, 3), np.uint8)
    img[:] = WAVE_BG
    peak = max(float(peak), 1e-9)
    rows = np.arange(height)[:, None]
    for data, color in layers:
        mn, mx = data if isinstance(data, tuple) else envelope(data, width)
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
    for step in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800):
        if step / span_s * width >= min_px:
            return step
    return 3600


LOG_TICKS = [10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000, 7000, 10000, 15000, 20000,
             30000, 40000]


def _freq_ticks(sr, scale):
    fmax = sr / 2
    if scale == "log":
        return [f for f in LOG_TICKS if f <= fmax]
    step = 2000 if fmax <= 24000 else 5000 if fmax <= 50000 else 10000
    return [f for f in np.arange(0, fmax + 1, step)]


def _fmt_hz(f):
    return f"{f / 1000:g}k" if f >= 1000 else f"{int(f)}"


def _freq_row(f, sr, height, scale):
    return (1.0 - float(_warp(sr, scale)[0](f))) * height


def render_view(db, wave_layers, sr, t0, n_view, peak, scale="log", title="", info="", title_color=TEXT,
                markers=(), region=None):
    """Full RGB image (HEIGHT, WIDTH, 3).

    db: spectrogram_db / overview_spectrogram of the shown signal; wave_layers: [(signal of n_view samples
    or (mins, maxs), rgb)] for the waveform, drawn in order; t0: start of the view in seconds (ruler); peak:
    amplitude at the top of the waveform panel (linear, 1.0 = 0 dBFS); markers: [(seconds, label,
    highlighted)] flags at the top of the waveform; region: (start, end) seconds tinted over the view."""
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
    canvas[SPEC_Y:SPEC_Y + SPEC_H, :PLOT_W] = colorize(db)
    span = n_view / sr
    to_x = lambda t: (t - t0) / span * PLOT_W
    if region is not None and region[1] - region[0] < 0.9 * span:  # the snippet view's region: tinted
        xa, xb = int(np.floor(to_x(region[0]))), int(np.ceil(to_x(region[1])))
        xa, xb = max(xa, 0), min(max(xb, xa + 2), PLOT_W)
        for y0, y1 in ((WAVE_Y, WAVE_Y + WAVE_H), (SPEC_Y, SPEC_Y + SPEC_H)):
            part = canvas[y0:y1, xa:xb].astype(np.float32)
            canvas[y0:y1, xa:xb] = (part * 0.7 + 255 * 0.3).astype(np.uint8)

    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    # header
    dr.text((8, HEADER_H / 2), title, fill=title_color, font=FONT_HEAD, anchor="lm")
    dr.text((HEADER_RIGHT, HEADER_H / 2), info, fill=TEXT_DIM, font=FONT_HEAD, anchor="rm")
    # snippet markers: a numbered flag per snippet at the top of the waveform
    for t, label, on in markers:
        x = to_x(t)
        if not 0 <= x < PLOT_W:
            continue
        w = dr.textlength(label, font=FONT_SMALL) + 6
        xl = min(max(x - w / 2, 0), PLOT_W - w)
        col = MARKER_ON if on else MARKER
        dr.line([(x, WAVE_Y + 14), (x, WAVE_Y + WAVE_H - 1)], fill=col if on else (70, 74, 84))
        dr.rectangle([xl, WAVE_Y + 1, xl + w, WAVE_Y + 14], fill=col)
        dr.text((xl + w / 2, WAVE_Y + 8), label, fill=(15, 17, 21), font=FONT_SMALL, anchor="mm")
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
        if r < 6 or r > SPEC_H - 6 or abs(r - last) < 22:
            continue
        last = r
        dr.line([(PLOT_W, SPEC_Y + r), (PLOT_W + 4, SPEC_Y + r)], fill=TICK)
        dr.text((x0, SPEC_Y + r), _fmt_hz(f), fill=TEXT_DIM, font=FONT, anchor="lm")
    dr.text((PLOT_W + RULER_W - 4, SPEC_Y + SPEC_H - 2), "Hz", fill=TICK, font=FONT, anchor="rb")
    # time ruler
    step = _time_step(span, PLOT_W)
    minor = step / 5
    decimals = 3 if step < 1 else 0
    k = np.ceil(t0 / minor - 1e-9)
    while True:
        t = k * minor
        if t > t0 + span + 1e-9:
            break
        x = to_x(t)
        major = abs(t / step - round(t / step)) < 1e-6
        dr.line([(x, TIME_Y), (x, TIME_Y + (7 if major else 3))], fill=TICK)
        if major:
            lab = fmt_time(t, decimals)
            w = dr.textlength(lab, font=FONT)
            xl = min(max(x, w / 2 + 2), PLOT_W - w / 2 - 2)
            dr.text((xl, TIME_Y + 9), lab, fill=TEXT_DIM, font=FONT, anchor="mt")
        k += 1
    return np.asarray(im)
