"""Gradio front-end for BetterDeclipper: declip one file, see the whole result and compare before / after /
delta at its most restored peaks, or declip many files at once."""
import argparse
import atexit
import json
import mimetypes
import os
import shutil
import sys
import time
import urllib.parse
import zipfile

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf
from gradio.route_utils import API_PREFIX
from gradio.utils import get_upload_folder

from . import __version__
from . import render as R
from .processing import (AUDIO_EXTS, PRESETS, Cancelled, Output, Settings, db, declip_array,
                         device_description, list_audio, output_gain, output_name, write_audio)
from .session import Session, channel_names, parse_time, remove_work_dirs

# the players load the WAVs straight from the work folder (Gradio serves its upload folder): as audio/wav,
# which some systems' MIME tables call audio/x-wav, and Gradio would serve as a download then
mimetypes.add_type("audio/wav", ".wav")

WORK_ROOT = os.path.join(get_upload_folder(), "betterdeclipper_gui")
SESSION_ROOT = os.path.join(WORK_ROOT, str(os.getpid()))  # sessions end with the process
STALE_S = 12 * 3600     # other work folders older than this are removed at startup

VIEWS = ["Before", "After", "Delta"]
SCALES = ["Log", "Linear"]

MODE_CHOICES = [
    ("Auto (recommended)", "auto"),
    ("Hard", "hard"),
    ("Soft", "soft"),
    ("Limiter (experimental)", "limiter"),
    ("Legacy", "legacy"),
]
MODE_INFO = ("auto: finds how it was clipped (see Analysis) · hard: ceilings only · soft: also a soft "
             "shoulder below them · limiter: deeper, can add distortion")
FORMAT_CHOICES = [
    ("WAV 32-bit float", "wav32f"),
    ("WAV 24-bit", "wav24"),
    ("WAV 16-bit", "wav16"),
    ("FLAC 24-bit", "flac24"),
    ("FLAC 16-bit", "flac16"),
]
FORMAT_INFO = "32-bit float keeps restored peaks above 0 dBFS as they are."
BATCH_HEADERS = ["File", "Restoration", "Samples restored", "Peak", "Output", "Time"]

IDLE = ("Upload an audio file, then press **Declip** (on the left, under the settings). It can be pressed "
        "while the file is still uploading.")
VIEW_HELP = ("**Before**: the input · **After**: the result, orange where it goes beyond the input · "
             "**Delta**: after - before, what was added. All waveforms share one scale, set by the highest "
             "peak of all results. Keys: **B** / **A** / **D** view, **1** - **9** result, **F** / **S** tab, "
             "**←** / **→** snippet, **Space** play / pause, **Shift** + click play from there.")
OVERVIEW_HELP = ("Click a moment to look at it in **Snippets**, **Shift** + click (or **Ctrl** + click) to play "
                 "from there. The numbered flags are the most restored peaks (1 = most), the light band is the "
                 "snippet on view.")
SNIP_HELP = "Click the view to play from there."
KEYS_NOTE = ("Keys **1** - **9** pick the first nine results: remove the ones you no longer need to reach the "
             "others.")
RESULTS_LABEL = "Results: pick one to view and download it"

HELP = """
### Getting started
1. Upload a clipped file (WAV, FLAC, AIFF, MP3, Ogg) on the **Declip** tab, check the settings on the left
   and press **Declip** under them. It can be pressed while the file is still uploading: the file is
   declipped as soon as it is there, with the settings as they are then.
2. **Full length** shows the whole result: the numbered flags mark the most restored peaks. Click one, or any
   other moment, to look at it closely in **Snippets**, which shows 3 seconds centered on that peak.
   Switch **Before / After / Delta**, the channel and the frequency scale; step through the peaks with
   **◀ ▶**, or type a time (`1:23.5`). A moment picked before the first result stays on view.
3. Try other settings and press **Declip** again: every run is kept under **Results**, and the view stays
   at the same place, so you can compare them. The file is analyzed once: later runs with another preset or
   mode only repeat the restoration. Remove results you don't need; the waveform scale follows the highest
   peak of the results that are left.
4. Listen: **Space** plays and pauses the view on screen. Playback carries on when you switch between
   before, after and delta or between results, so you hear the difference at the same moment. Download the
   selected result (in the output format chosen on the left), or declip many files on the **Batch** tab.

### Keys
- **B** / **A** / **D**: before / after / delta
- **1** - **9**: the first nine results (remove results you no longer need to reach later ones)
- **F** / **S**: full length / snippets
- **←** / **→**: previous / next snippet
- **Space**: play / pause
- **Shift** + click: play from there (also **Ctrl** + click; in Snippets a plain click)

### Settings
- **Preset**: `fast` is quickest, `normal` a good default, `high` and `best` a little more accurate and
  slower. An NVIDIA GPU makes everything much faster.
- **Mode**: `auto` measures how the file was clipped (hard clipping, lossy-encoded or resampled clipping,
  limiting, soft saturation) and restores accordingly; the **Analysis** box explains what it found.
  `soft` also restores a soft shoulder when auto found none (if it still sounds squashed). `limiter` is
  experimental and can add distortion.
- **Advanced**: force the clip level or knee yourself (skips the analysis), cap the gain, pick the device.
- **Output**: restored peaks often go above 0 dBFS. 32-bit float WAV keeps them as they are (turn it down
  later, or tick *Normalize*). PCM WAV and FLAC cannot store them, so those are turned down to the peak
  level instead of clipping again.

### Reading the view
The waveform (top) is drawn like in iZotope RX, with a dB scale; the red line marks 0 dBFS when restored
peaks exceed it. The spectrogram matches RX's default look: its colors (black at -120 dB to white at
0 dB), its log frequency axis, and its resolution (in the log view, RX's multi-resolution at FFT size 512).
Clipping shows up as a haze of distortion between the harmonics and as vertical smears at the peaks; a good
restoration clears them. Both players play at one common gain (turned down only when the highest peak
exceeds 0 dBFS), so the versions compare fairly and nothing clips; the white line is the playhead.
"""

WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def _web(name):
    with open(os.path.join(WEB, name), encoding="utf-8") as f:
        return f.read()


CSS = _web("app.css")
PLAYER_HTML = _web("player.html")
# the browser side (web/app.js), told where the plot area of the rendered views is
HEAD = (f"<script>window.BD_LAYOUT = {json.dumps({'plotW': R.PLOT_W, 'top': R.WAVE_Y, 'bottom': R.TIME_Y})};"
        f"</script>\n<script>\n{_web('app.js')}\n</script>\n")
MOUNT_PLAYER = "if (window.bdMountPlayer) window.bdMountPlayer(element, props, watch);"
# Declip: tell on_run_start whether the input is still uploading, then wait for it in the browser
UPLOADING_JS = "(...a) => { a[a.length - 1] = !!(window.bdUploading && window.bdUploading()); return a; }"
WAIT_UPLOAD_JS = "async () => { if (window.bdWaitForUpload) await window.bdWaitForUpload(); }"
STOP_WAIT_JS = "(...a) => { window.bdStopWait = true; return a; }"


def _pid_alive(pid):
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k32.GetLastError() == 5  # access denied: it exists
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale(root=WORK_ROOT, age_s=STALE_S):
    """Remove work folders left behind by earlier runs of the app (e.g. when its console was closed): those
    of processes that no longer run, and anything else older than age_s."""
    if not os.path.isdir(root):
        return
    now = time.time()
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            if not os.path.isdir(p):
                continue
            if (not _pid_alive(int(name))) if name.isdigit() else now - os.path.getmtime(p) > age_s:
                shutil.rmtree(p, ignore_errors=True)
        except (OSError, ValueError):
            pass


class BadSetting(ValueError):
    """An Advanced setting that can't be used; the message says which one and why."""


def _number(text, what):
    """An optional number typed in a text field: None when empty ('off'), else the value. A comma decimal,
    a typographic minus and a trailing dB / dBFS are fine."""
    t = (text or "").strip().lower().replace("−", "-").replace(",", ".")
    for unit in ("dbfs", "db"):
        t = t[:-len(unit)].strip() if t.endswith(unit) else t
    if t in ("", "off", "none"):
        return None
    try:
        return float(t)
    except ValueError:
        raise BadSetting(f"{what} (Advanced) is '{text.strip()}': type a number, e.g. -12, or leave it empty.")


def _settings(preset, mode, clip, knee, max_gain, device):
    """Settings from the sidebar. The optional Advanced values are text fields: empty means off (Gradio's
    number fields turn empty into 0 when they first show, which forced a clip level of 0 dBFS)."""
    clip_db, knee_db = _number(clip, "Force clip level"), _number(knee, "Force soft knee")
    gain_db = _number(max_gain, "Gain cap")
    for v, what in ((clip_db, "Force clip level"), (knee_db, "Force soft knee")):
        if v is not None and v >= 0:
            raise BadSetting(f"{what} (Advanced) is {v:g} dBFS: it must be below 0 dBFS, e.g. -12, or empty to "
                             f"let the analysis decide.")
    if gain_db is not None and gain_db <= 0:
        raise BadSetting(f"Gain cap (Advanced) is {gain_db:g} dB: it must be above 0 dB, or empty for no cap.")
    return Settings(preset, mode, clip_db, knee_db, gain_db, device)


def _unique(path, taken):
    """path, or 'name (2).ext' etc. if this batch already wrote a file of that name"""
    stem, ext = os.path.splitext(path)
    k = 2
    while path.lower() in taken:
        path = f"{stem} ({k}){ext}"
        k += 1
    taken.add(path.lower())
    return path


def _summary(run, note):
    what = (f"{run.flagged * 100:.1f} % of the samples restored" if run.flagged
            else "nothing to restore, see the analysis below")
    how = run.device + (", the analysis reused" if run.reused else "")
    msg = (f"**Result #{run.id}** in {run.seconds:.1f} s ({how}): *{run.label}*, {what}, "
           f"peak {db(run.peak):+.2f} dBFS.")
    return msg + (f"  \nDownload: {note}." if note else "")


def _player_value(info):
    """A player's value: what it plays (Session.player) with the URL of the file, as JSON."""
    if info is None:
        return ""
    d = dict(info)
    d["src"] = f"{API_PREFIX}/file=" + urllib.parse.quote(d.pop("path"), safe="/")
    return json.dumps(d)


def _unreadable(e):
    return f"Could not read this file ({e}). WAV, FLAC, AIFF, MP3, Ogg and Opus files work."


def _player(pid, elem_id):
    """A player (see web/app.js): its value is a _player_value."""
    return gr.HTML("", html_template=PLAYER_HTML, js_on_load=MOUNT_PLAYER, apply_default_css=False,
                   elem_id=elem_id, pid=pid)


def build(folders=True):
    """The Gradio app. folders: allow reading and writing folders on this computer (local use only)."""
    import betterdeclipper
    dev, dev_name = device_description()
    header = (f"# BetterDeclipper\n"
              f"Restores clipped, limited and lossy-encoded peaks. Upload a file and press **Declip** (on the "
              f"left, under the settings), then look at the whole result and compare *before*, *after* and "
              f"*delta* at the most restored peaks.  \n"
              f"<span class='bd-dim'>core {betterdeclipper.__version__} · GUI {__version__} · {dev_name}</span>")

    # no telemetry or version check (they phone home, and hold the process open when offline)
    with gr.Blocks(title="BetterDeclipper", delete_cache=(3600, 86400), analytics_enabled=False) as demo:
        sess = gr.State(lambda: Session(SESSION_ROOT), delete_callback=Session.close)
        uploading = gr.Checkbox(False, visible=False)  # filled in by the browser when Declip is pressed

        with gr.Sidebar(open=True, width=330):
            gr.Markdown("### Restoration")
            preset = gr.Radio(list(PRESETS), value="normal", label="Preset",
                              info="fast = quickest, normal = good default, high / best = a little better, slower")
            mode = gr.Dropdown(MODE_CHOICES, value="auto", label="Mode", info=MODE_INFO)
            with gr.Accordion("Advanced", open=False) as advanced:
                # text fields: empty = off (see _settings)
                clip = gr.Textbox(label="Force clip level (dBFS)", placeholder="off, e.g. -12", max_lines=1,
                                  info="Skips the analysis and restores only above this level (hard mode).")
                knee = gr.Textbox(label="Force soft knee (dBFS)", placeholder="off, e.g. -9", max_lines=1,
                                  info="Skips the analysis; every sample above it may only grow (soft mode).")
                max_gain = gr.Textbox(label="Gain cap (dB)", placeholder="off, e.g. 12", max_lines=1,
                                      info="Restored samples stay within this much of the clip level.")
                device = gr.Radio(["auto", "cuda", "cpu"] if dev == "cuda" else ["auto", "cpu"], value="auto",
                                  label="Device")
            # the action sits right under the settings it uses: Declip, or Declip all on the Batch tab
            with gr.Column() as run_group:
                run_btn = gr.Button("Declip", variant="primary", size="lg")
                stop_btn = gr.Button("Stop", variant="stop", visible=False)
            with gr.Column(visible=False) as batch_group:
                batch_btn = gr.Button("Declip all", variant="primary", size="lg")
                batch_stop = gr.Button("Stop after this step", variant="stop", visible=False)
            gr.Markdown("### Output")
            fmt = gr.Dropdown(FORMAT_CHOICES, value="wav32f", label="Format", info=FORMAT_INFO)
            normalize = gr.Checkbox(False, label="Normalize the peak")
            target = gr.Number(-0.1, label="Peak level (dBFS)", maximum=0, step=0.1,
                               info="Used to normalize, and when PCM / FLAC could not hold the restored peaks.")
            folder = gr.Textbox(label="Also save to folder", placeholder="optional, e.g. D:\\Music\\declipped",
                                visible=folders)

        gr.Markdown(header)
        # render_children: hidden tabs stay mounted, so switching tabs keeps everything as it was
        with gr.Tabs() as main_tabs:
            # ---- single file --------------------------------------------------------------------
            with gr.Tab("Declip", id="declip", render_children=True):
                with gr.Row(equal_height=False):
                    inp = gr.Audio(label="Input", sources=["upload"], type="filepath", format=None,
                                   editable=False, scale=3, elem_id="bd-input")
                    with gr.Column(scale=1, min_width=250):
                        inp_info = gr.Markdown(IDLE)
                        status = gr.Markdown("", min_height=72)  # run summary; the progress bar shows here
                with gr.Row(visible=False, equal_height=False) as results_row:
                    with gr.Column(scale=3):
                        history = gr.Radio([], label=RESULTS_LABEL, elem_id="bd-history")
                        with gr.Row(equal_height=True):
                            remove_btn = gr.Button("Remove this result", size="sm", scale=0, min_width=160)
                            keys_note = gr.Markdown(KEYS_NOTE, visible=False, elem_classes=["bd-dim"])
                    with gr.Column(scale=1, min_width=250):
                        download = gr.File(label="Download", interactive=False)
                        dl_note = gr.Markdown(elem_classes=["bd-dim"])
                with gr.Column(visible=False) as viewer_col:
                    with gr.Row(elem_id="bd-toolbar"):
                        view = gr.Radio(VIEWS, value="Before", show_label=False, container=False,
                                        elem_id="bd-view", scale=2)
                        chan = gr.Radio(["L", "R"], value="L", show_label=False, container=False, scale=1)
                        scale = gr.Radio(SCALES, value="Log", show_label=False, container=False, scale=1)
                    gr.Markdown(VIEW_HELP, elem_id="bd-help-line")
                    with gr.Tabs(selected="full", elem_id="bd-subtabs") as subtabs:
                        with gr.Tab("Full length", id="full", render_children=True):
                            ov_image = gr.Image(type="numpy", format="png", show_label=False, interactive=False,
                                                buttons=["download", "fullscreen"], elem_id="bd-overview")
                            gr.Markdown(OVERVIEW_HELP, elem_classes=["bd-dim"])
                            ov_player = _player("full", "bd-player-full")
                        with gr.Tab("Snippets", id="snip", render_children=True):
                            image = gr.Image(type="numpy", format="png", show_label=False, interactive=False,
                                             buttons=["download", "fullscreen"], elem_id="bd-image")
                            with gr.Row(equal_height=True):
                                prev_btn = gr.Button("◀ Prev", size="sm", scale=0, min_width=84, elem_id="bd-prev")
                                snip = gr.Dropdown([], show_label=False, container=False, scale=5)
                                next_btn = gr.Button("Next ▶", size="sm", scale=0, min_width=84, elem_id="bd-next")
                                goto = gr.Textbox(show_label=False, container=False,
                                                  placeholder="go to time, e.g. 1:23.5", scale=2, min_width=150)
                                rerank_btn = gr.Button("Re-rank peaks from this result", size="sm", scale=1,
                                                       min_width=150, visible=False)
                            gr.Markdown(SNIP_HELP, elem_classes=["bd-dim"])
                            clip_player = _player("snip", "bd-player-snip")
                    with gr.Accordion("Analysis: what the declipper found", open=False, visible=False) as report_box:
                        report = gr.Textbox(label="Analysis", lines=10, max_lines=40, elem_id="bd-report",
                                            buttons=["copy"], interactive=False)

            # ---- batch --------------------------------------------------------------------------
            with gr.Tab("Batch", id="batch", render_children=True):
                gr.Markdown("Declip many files with the settings on the left: press **Declip all** under them. "
                            "Results are named like the command line names them, e.g. "
                            "`song [auto clip normal].wav`.")
                batch_files = gr.File(label="Audio files", file_count="multiple", type="filepath",
                                      file_types=list(AUDIO_EXTS))
                batch_folder = gr.Textbox(label="... and/or every audio file in this folder", visible=folders,
                                          placeholder="e.g. D:\\Music\\to declip",
                                          info="Results go next to the files, or to 'Also save to folder'.")
                batch_status = gr.Markdown(min_height=72)
                batch_table = gr.Dataframe(headers=BATCH_HEADERS, interactive=False, visible=False, wrap=True)
                batch_zip = gr.File(label="All results (.zip)", visible=False, interactive=False)
                batch_out = gr.File(label="Results", file_count="multiple", visible=False, interactive=False)

            with gr.Tab("Help", id="help", render_children=True):
                gr.Markdown(HELP)

        def on_main_tab(evt: gr.SelectData):
            batch = evt.value == "Batch" or evt.index == 1
            return gr.update(visible=not batch), gr.update(visible=batch)

        main_tabs.select(on_main_tab, None, [run_group, batch_group], queue=False, show_progress="hidden")

        # ---- helpers ----------------------------------------------------------------------------
        def output_of(fmt_v, norm_v, target_v, folder_v):
            t = -0.1 if target_v is None else min(float(target_v), 0.0)
            return Output(fmt_v, bool(norm_v), t, (folder_v or "").strip() if folders else "")

        def viewer(s, view_v, scale_v, full=True):
            """Updates for what changed on the sub-tab on view (the other one catches up when it is opened);
            full: also the selectors, the result list and the report."""
            keys = s.view_keys(view_v, scale_v)
            sc = scale_v.lower()
            d = {}
            if s.tab == "full":
                if s.shown.get("ov_img") != keys["ov_img"]:
                    d[ov_image] = s.render_overview(view_v, sc)
                    s.shown["ov_img"] = keys["ov_img"]
                if s.shown.get("ov_audio") != keys["ov_audio"]:
                    d[ov_player] = _player_value(s.player("full", view_v))
                    s.shown["ov_audio"] = keys["ov_audio"]
            else:
                if s.shown.get("sn_img") != keys["sn_img"]:
                    d[image] = s.render(view_v, sc)
                    s.shown["sn_img"] = keys["sn_img"]
                if s.shown.get("sn_audio") != keys["sn_audio"]:
                    d[clip_player] = _player_value(s.player("snip", view_v))
                    s.shown["sn_audio"] = keys["sn_audio"]
            if full:
                names = channel_names(s.input.channels)
                d[chan] = gr.update(choices=names, value=names[s.channel], visible=len(names) > 1)
                choices, value = s.snippet_choices()
                d[snip] = gr.update(choices=choices, value=value)
                hist, sel = s.history()
                d[history] = gr.update(choices=hist, value=sel)
                d[keys_note] = gr.update(visible=len(hist) > 9)
                run = s.run()
                d[report] = run.report if run else ""
                d[report_box] = gr.update(visible=run is not None)
                d[rerank_btn] = gr.update(visible=run is not None and len(s.runs) > 1)
            return d

        def loaded(s):
            """Updates for a file just loaded: what it is, no results yet, the full length on view."""
            return {inp_info: s.input.describe(), status: "", view: "Before",
                    results_row: gr.update(visible=False), viewer_col: gr.update(visible=True),
                    subtabs: gr.Tabs(selected="full"), clip_player: ""}

        def download_of(s, out):
            run = s.run()
            if run is None:
                return {download: None, dl_note: ""}
            path, note = s.export(run.id, out)
            return {download: path, dl_note: note}

        media = [ov_image, ov_player, image, clip_player]
        view_out = media + [chan, snip, history, keys_note, report, report_box, rerank_btn]
        load_out = [inp_info, status, results_row, viewer_col, view, subtabs]
        out_inputs = [fmt, normalize, target, folder]
        hidden = lambda: {status: "", results_row: gr.update(visible=False), viewer_col: gr.update(visible=False),
                          ov_player: "", clip_player: ""}
        quiet = dict(show_progress="minimal", show_progress_on=[ov_image, image])  # no spinner on the players

        # ---- single file: input and runs --------------------------------------------------------
        def on_upload(s, path, scale_v):
            if not path:
                return gr.skip()
            try:
                inp_file = s.load_if_new(path)
            except Exception as e:  # unreadable / unsupported file
                s.clear()
                return {inp_info: _unreadable(e), **hidden()}
            if inp_file is None:
                return gr.skip()  # loaded already (Declip, pressed during the upload, can get here first)
            d = viewer(s, "Before", scale_v)
            d.update(loaded(s))
            with s.lock:
                if s.input is not inp_file or s.runs:  # a result came first (a tiny file): keep its view
                    s.shown.clear()
                    return gr.skip()
            return d

        def on_clear(s):
            s.cancel.set()  # a run of it stops at its next step, a waiting one does not start
            s.clear()
            return {inp_info: IDLE, **hidden()}

        # upload / clear only: they fire on what the user does, not when the component is shown again
        inp.upload(on_upload, [sess, inp, scale], load_out + view_out)
        inp.clear(on_clear, sess, [inp_info, status, results_row, viewer_col, ov_player, clip_player])

        def on_run_start(s, busy):
            s.cancel.clear()
            msg = ("Waiting for the upload to finish: the file is declipped as soon as it is there, with the "
                   "settings as they are then." if busy else "")
            return gr.update(visible=False), gr.update(visible=True), msg

        def on_run(s, path, preset_v, mode_v, clip_v, knee_v, gain_v, device_v, fmt_v, norm_v, target_v, folder_v,
                   scale_v, progress=gr.Progress()):
            try:
                st = _settings(preset_v, mode_v, clip_v, knee_v, gain_v, device_v)
            except BadSetting as e:  # shown where it can be fixed
                gr.Warning(str(e))
                yield {status: f"**Not declipped.** {e}", advanced: gr.Accordion(open=True)}
                return
            out = output_of(fmt_v, norm_v, target_v, folder_v)
            if s.cancel.is_set():  # Stop, or the input was cleared, while it waited
                yield {status: "Stopped."}
                return
            if path:
                try:
                    got = s.take(path)  # waits while the upload event loads it
                except Exception as e:
                    s.clear()
                    yield {inp_info: _unreadable(e), **hidden()}
                    return
                if got == "stale":
                    yield {status: "The input changed while this run waited: press **Declip** again."}
                    return
                if got == "loaded":  # pressed during the upload, and here before the upload event
                    d = viewer(s, "Before", scale_v)
                    d.update(loaded(s))
                    yield d
            if s.input is None:
                raise gr.Error("Upload an audio file first.")
            if s.cancel.is_set():
                yield {status: "Stopped."}
                return
            progress(0, desc="Declipping (the analysis is reused)" if s.will_reuse_analysis(st)
                     else "Analyzing the clipping")
            try:
                run = s.process(st, lambda i, n, el: progress((i, n), desc="Declipping", unit="steps"))
            except Cancelled:
                if s.cancel.is_set():
                    yield {status: "Stopped."}
                return  # else another file was loaded meanwhile, and is on view
            except Exception as e:
                raise gr.Error(f"Declipping failed: {e}")
            d = viewer(s, "After", scale_v)
            d.update(download_of(s, out))
            msg = _summary(run, "")
            if out.folder:
                try:
                    os.makedirs(out.folder, exist_ok=True)
                    dst = os.path.join(out.folder, os.path.basename(d[download]))
                    shutil.copyfile(d[download], dst)
                    msg += f"  \nSaved to `{dst}`."
                except OSError as e:
                    msg += f"  \nCould not save to the folder: {e}"
            d.update({status: msg, inp_info: s.input.describe(), results_row: gr.update(visible=True),
                      viewer_col: gr.update(visible=True), view: "After"})
            yield d

        def on_run_end():
            return gr.update(visible=True), gr.update(visible=False)

        # Declip can be pressed while the input uploads: the browser waits for the upload, then the run reads
        # the settings (so it uses them as they are when it starts)
        running = run_btn.click(on_run_start, [sess, uploading], [run_btn, stop_btn, status], js=UPLOADING_JS,
                                queue=False).then(
            None, None, None, js=WAIT_UPLOAD_JS).then(
            on_run, [sess, inp, preset, mode, clip, knee, max_gain, device] + out_inputs + [scale],
            load_out + [download, dl_note, advanced] + view_out,
            concurrency_id="declip", concurrency_limit=1, show_progress_on=[status])
        for after in (running.then, running.failure):  # .then does not follow an error
            after(on_run_end, None, [run_btn, stop_btn], queue=False)
        stop_btn.click(lambda s: s.cancel.set(), sess, None, js=STOP_WAIT_JS, queue=False)

        def on_history(s, rid, view_v, scale_v, *out_v):
            s.select(rid)
            view_v = "After" if view_v == "Before" else view_v
            d = viewer(s, view_v, scale_v)
            d.update(download_of(s, output_of(*out_v)))
            d[view] = view_v
            return d

        history.input(on_history, [sess, history, view, scale] + out_inputs,
                      [view, download, dl_note] + view_out, **quiet)

        def on_remove(s, view_v, scale_v, *out_v):
            if s.run() is None:
                return gr.skip()
            s.remove(s.selected)
            if s.run() is None:
                s.goto(0)
                d = viewer(s, "Before", scale_v)
                d.update({results_row: gr.update(visible=False), view: "Before", download: None,
                          status: "All results removed. Press **Declip** to run again."})
                return d
            d = viewer(s, view_v, scale_v)
            d.update(download_of(s, output_of(*out_v)))
            return d

        remove_btn.click(on_remove, [sess, view, scale] + out_inputs,
                         [results_row, view, status, download, dl_note] + view_out, **quiet)

        def on_output(s, *out_v):
            if s.input is None or s.run() is None:
                return gr.skip()
            return download_of(s, output_of(*out_v))

        for ev in (fmt.change, normalize.change, target.submit, target.blur):
            ev(on_output, [sess] + out_inputs, [download, dl_note], show_progress="minimal")

        # ---- single file: the views -------------------------------------------------------------
        def on_view(s, view_v, scale_v):
            if s.input is None:
                return gr.skip()
            if view_v != "Before" and s.run() is None:
                gr.Info("Press Declip first: there is no result to show yet.")
            return viewer(s, view_v, scale_v, full=False) or gr.skip()

        view.input(on_view, [sess, view, scale], media, **quiet)
        scale.input(on_view, [sess, view, scale], media, **quiet)

        def on_chan(s, name, view_v, scale_v):
            if s.input is None:
                return gr.skip()
            names = channel_names(s.input.channels)
            with s.lock:
                s.channel = names.index(name) if name in names else 0
            return viewer(s, view_v, scale_v, full=False) or gr.skip()

        chan.input(on_chan, [sess, chan, view, scale], media, **quiet)

        def on_subtab(s, view_v, scale_v, evt: gr.SelectData):
            s.tab = "snip" if evt.value == "Snippets" or evt.index == 1 else "full"
            if s.input is None:
                return gr.skip()
            return viewer(s, view_v, scale_v, full=False) or gr.skip()

        # the Tabs' select event (a Tab's own select event only fires on its first selection)
        subtabs.select(on_subtab, [sess, view, scale], media, **quiet)

        def on_overview_click(s, view_v, scale_v, evt: gr.SelectData):
            if s.input is None or not evt.index or evt.index[0] >= R.PLOT_W:
                return gr.skip()
            s.goto_x(evt.index[0])
            s.tab = "snip"
            d = viewer(s, view_v, scale_v)
            d[subtabs] = gr.Tabs(selected="snip")
            return d

        ov_image.select(on_overview_click, [sess, view, scale], [subtabs] + view_out, **quiet)

        def on_step(step):
            def fn(s, view_v, scale_v):
                if s.input is None:
                    return gr.skip()
                s.goto(s.snip_idx + step)
                return viewer(s, view_v, scale_v)
            return fn

        prev_btn.click(on_step(-1), [sess, view, scale], view_out, **quiet)
        next_btn.click(on_step(+1), [sess, view, scale], view_out, **quiet)

        def on_snip(s, value, view_v, scale_v):
            if s.input is None or value in (None, "custom"):
                return gr.skip()
            s.goto(int(value))
            return viewer(s, view_v, scale_v)

        snip.input(on_snip, [sess, snip, view, scale], view_out, **quiet)

        def on_goto(s, text, view_v, scale_v):
            if s.input is None or not (text or "").strip():
                return gr.skip()
            try:
                t = parse_time(text)
            except ValueError:
                raise gr.Error("Type a time like 83.5, 1:23.5 or 0:01:23.5")
            dur = s.input.frames / s.input.sr
            if t > dur:
                gr.Warning(f"The file is only {dur:.1f} s long; showing its end.")
            s.goto_time(t)
            return viewer(s, view_v, scale_v)

        goto.submit(on_goto, [sess, goto, view, scale], view_out, **quiet)

        def on_rerank(s, view_v, scale_v):
            run = s.run()
            if run is None:
                return gr.skip()
            s.rank_from(run)
            return viewer(s, view_v, scale_v)

        rerank_btn.click(on_rerank, [sess, view, scale], view_out, **quiet)

        # ---- batch ------------------------------------------------------------------------------
        def on_batch_start(s):
            s.cancel_batch.clear()
            return gr.update(visible=False), gr.update(visible=True)

        def on_batch(s, files, src_v, preset_v, mode_v, clip_v, knee_v, gain_v, device_v, fmt_v, norm_v,
                     target_v, folder_v, progress=gr.Progress()):
            try:
                st = _settings(preset_v, mode_v, clip_v, knee_v, gain_v, device_v)
            except BadSetting as e:
                gr.Warning(str(e))
                yield {batch_status: f"**Not declipped.** {e}", advanced: gr.Accordion(open=True)}
                return
            out = output_of(fmt_v, norm_v, target_v, folder_v)
            items = [(p, None) for p in (files or [])]      # (file, folder to write next to)
            src = (src_v or "").strip() if folders else ""
            if src:
                if not os.path.isdir(src):
                    raise gr.Error(f"Folder not found: {src}")
                items += [(p, src) for p in list_audio(src)]
            if not items:
                raise gr.Error("Add audio files, or type a folder, first.")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            tmp = os.path.join(s.dir, "batch", stamp)
            rows, ready, taken = [], [], set()
            N, t_start = len(items), time.time()
            n_ok = n_err = n_disk = 0
            yield {batch_status: f"Declipping {N} file{'s' * (N > 1)} ...",
                   batch_table: gr.update(value=None, visible=False),
                   batch_zip: gr.update(value=None, visible=False), batch_out: gr.update(value=None, visible=False)}
            for k, (path, src_dir) in enumerate(items):
                name = os.path.basename(path)
                if s.cancel_batch.is_set():
                    rows.append([name, "skipped (stopped)", "", "", "", ""])
                    continue
                dest = out.folder or src_dir or tmp
                progress(k / N, desc=f"{k + 1}/{N} {name}")
                try:
                    y, sr = sf.read(path, dtype="float64", always_2d=True)
                    x, info, label = declip_array(
                        y, sr, st, lambda i, n, el: progress((k + i / n) / N, desc=f"{k + 1}/{N} {name}"),
                        s.cancel_batch)
                    peak = float(np.abs(x).max())
                    gain, _ = output_gain(peak, out)
                    dst = _unique(os.path.join(dest, output_name(os.path.splitext(name)[0], label, st.preset,
                                                                 out.fmt)), taken)
                    write_audio(x, sr, dst, out.fmt, gain)
                    del x, y
                    if dest == tmp:
                        ready.append(dst)
                    else:
                        n_disk += 1
                    pk = f"{db(peak):+.1f} dBFS" + (f" (saved at {db(peak * gain):+.1f})" if gain != 1.0 else "")
                    rows.append([name, label, f"{info['clipped_frac'] * 100:.1f} %", pk,
                                 os.path.basename(dst) if dest == tmp else dst, f"{info['time']:.1f} s"])
                    n_ok += 1
                except Cancelled:
                    rows.append([name, "stopped", "", "", "", ""])
                except Exception as e:
                    rows.append([name, "error", "", "", str(e), ""])
                    n_err += 1
                yield {batch_status: f"{k + 1} of {N} done ...",
                       batch_table: gr.update(value=pd.DataFrame(rows, columns=BATCH_HEADERS), visible=True)}
            took = time.time() - t_start
            if s.cancel_batch.is_set():
                msg = f"**Stopped**: {n_ok} of {N} file{'s' * (N > 1)} declipped."
            else:
                msg = f"**Done**: {n_ok} of {N} file{'s' * (N > 1)} declipped in {took / 60:.1f} min."
            if n_err:
                msg += f" {n_err} failed (see the table)."
            if n_disk:
                msg += f" Saved to `{out.folder}`." if out.folder else " Saved next to the input files."
            if ready:
                msg += " Download the results below."
            zip_path = None
            if len(ready) > 1:
                progress(1.0, desc="Zipping the results")
                zip_path = os.path.join(tmp, f"BetterDeclipper {stamp}.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
                    for p in ready:
                        z.write(p, os.path.basename(p))
            yield {batch_status: msg,
                   batch_table: gr.update(value=pd.DataFrame(rows, columns=BATCH_HEADERS), visible=True),
                   batch_zip: gr.update(value=zip_path, visible=zip_path is not None),
                   batch_out: gr.update(value=ready or None, visible=bool(ready))}

        def on_batch_end():
            return gr.update(visible=True), gr.update(visible=False)

        batching = batch_btn.click(on_batch_start, sess, [batch_btn, batch_stop], queue=False).then(
            on_batch, [sess, batch_files, batch_folder, preset, mode, clip, knee, max_gain, device] + out_inputs,
            [batch_status, batch_table, batch_zip, batch_out, advanced],
            concurrency_id="declip", concurrency_limit=1, show_progress_on=[batch_status])
        for after in (batching.then, batching.failure):
            after(on_batch_end, None, [batch_btn, batch_stop], queue=False)
        batch_stop.click(lambda s: s.cancel_batch.set(), sess, None, queue=False)

    return demo


THEME = gr.themes.Default(
    primary_hue="orange", neutral_hue="slate",
    font=["ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
    font_mono=["ui-monospace", "Cascadia Mono", "Consolas", "monospace"])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="betterdeclipper-gui", description="Web interface for BetterDeclipper.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to listen on (default: this computer only; 0.0.0.0 = the local network)")
    ap.add_argument("--port", type=int, default=None, help="port (default: 7860 or the next free one)")
    ap.add_argument("--share", action="store_true", default=None,
                    help="also create a temporary public link (e.g. to use it from Colab)")
    ap.add_argument("--no-browser", action="store_true", help="don't open the app in the browser")
    ap.add_argument("--allow-folders", action="store_true",
                    help="keep 'Also save to folder' and batch folders when others can reach the app "
                         "(--share or --host); they read and write files on this computer")
    args = ap.parse_args(argv)
    local = args.host in ("127.0.0.1", "localhost", "::1") and not args.share and "google.colab" not in sys.modules
    cleanup_stale()
    atexit.register(remove_work_dirs)
    demo = build(folders=args.allow_folders or local)
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=not args.no_browser,
                theme=THEME, css=CSS, head=HEAD, footer_links=["gradio", "settings"])
    return 0
