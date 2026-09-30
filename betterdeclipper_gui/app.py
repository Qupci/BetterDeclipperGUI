"""Gradio front-end for BetterDeclipper: declip one file, see the whole result and compare before / after /
delta at its most restored peaks, or declip many files at once."""
import argparse
import atexit
import os
import shutil
import sys
import time
import zipfile

import gradio as gr
import numpy as np
import pandas as pd
import soundfile as sf
from gradio.utils import get_upload_folder

from . import __version__
from . import render as R
from .processing import (AUDIO_EXTS, PRESETS, Cancelled, Output, Settings, db, declip_array,
                         device_description, list_audio, output_gain, output_name, write_audio)
from .session import Session, channel_names, parse_time, remove_work_dirs

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

IDLE = "Upload an audio file, then press **Declip**."
VIEW_HELP = ("**Before**: the input · **After**: the result, orange where it goes beyond the input · "
             "**Delta**: after - before, what was added. All waveforms share one scale, set by the highest "
             "peak of all results. Keys: **B** / **A** / **D** view, **F** / **S** tab, **←** / **→** snippet.")
OVERVIEW_HELP = ("Click anywhere to look at that moment in **Snippets**; the numbered flags are the most "
                 "restored peaks (1 = most) and the light band is the snippet on view.")

HELP = """
### Getting started
1. Upload a clipped file (WAV, FLAC, AIFF, MP3, Ogg) on the **Declip** tab and press **Declip**.
2. **Full length** shows the whole result: the numbered flags mark the most restored peaks. Click one, or any
   other moment, to look at it closely in **Snippets**, which shows 3 seconds centered on that peak.
   Switch **Before / After / Delta**, the channel and the frequency scale; step through the peaks with
   **◀ ▶**, or type a time (`1:23.5`).
3. Try other settings and press **Declip** again: every run is kept under **Results**, and the view stays
   at the same place, so you can compare them. Remove results you don't need; the waveform scale follows
   the highest peak of the results that are left.
4. Listen to the whole result or to the snippet, download the selected result (in the output format
   chosen on the left), or declip many files on the **Batch** tab.

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
restoration clears them. Both players play the current view at one common gain (turned down only when the
highest peak exceeds 0 dBFS), so the versions compare fairly and nothing clips.
"""

CSS = """
#bd-report textarea { font-family: ui-monospace, 'Cascadia Mono', Consolas, monospace; font-size: 12px; line-height: 1.35; }
.bd-dim { opacity: 0.65; font-size: 0.92em; }
#bd-help-line { font-size: 0.88em; opacity: 0.8; }
#bd-image img, #bd-overview img { image-rendering: auto; }
#bd-overview img { cursor: crosshair; }
#bd-toolbar { align-items: center; }
"""

# Keyboard shortcuts for quick A/B comparisons (ignored while typing in a text field).
HEAD = """
<script>
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
  const t = e.target, tag = t && t.tagName;
  const typing = tag === 'TEXTAREA' || (t && t.isContentEditable) ||
                 (tag === 'INPUT' && !['radio', 'checkbox', 'button'].includes(t.type));
  if (typing) return;
  const pick = (value) => {
    const el = document.querySelector(`#bd-view input[type="radio"][value="${value}"]`);
    if (el && !el.checked) { el.click(); }
    if (el) e.preventDefault();
  };
  const press = (id) => {
    const b = document.getElementById(id);
    if (b && tag !== 'INPUT') { b.click(); e.preventDefault(); }
  };
  const tab = (label) => {
    const b = [...document.querySelectorAll('button[role="tab"]')].find((x) => x.textContent.trim() === label);
    if (b) { b.click(); e.preventDefault(); }
  };
  switch (e.key) {
    case 'b': case 'B': pick('Before'); break;
    case 'a': case 'A': pick('After'); break;
    case 'd': case 'D': pick('Delta'); break;
    case 'f': case 'F': tab('Full length'); break;
    case 's': case 'S': tab('Snippets'); break;
    case 'ArrowLeft': press('bd-prev'); break;
    case 'ArrowRight': press('bd-next'); break;
  }
});
</script>
"""


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


def _settings(preset, mode, clip, knee, max_gain, device):
    for v, what in ((clip, "clip level"), (knee, "knee")):
        if v is not None and v >= 0:
            raise gr.Error(f"The forced {what} must be below 0 dBFS (for example -12).")
    if max_gain is not None and max_gain < 0:
        raise gr.Error("The gain cap must be 0 dB or more.")
    return Settings(preset, mode, clip, knee, max_gain, device)


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
    msg = (f"**Result #{run.id}** in {run.seconds:.1f} s ({run.device}): *{run.label}*, {what}, "
           f"peak {db(run.peak):+.2f} dBFS.")
    return msg + (f"  \nDownload: {note}." if note else "")


def _player_label(s, name):
    g = s.playback_gain_db()
    return name + (f" (played at {g:+.1f} dB so the restored peaks don't clip)" if g < -0.05 else "")


def build(folders=True):
    """The Gradio app. folders: allow reading and writing folders on this computer (local use only)."""
    import betterdeclipper
    dev, dev_name = device_description()
    header = (f"# BetterDeclipper\n"
              f"Restores clipped, limited and lossy-encoded peaks. Upload a file and press **Declip**, then "
              f"look at the whole result and compare *before*, *after* and *delta* at the most restored peaks.  \n"
              f"<span class='bd-dim'>core {betterdeclipper.__version__} · GUI {__version__} · {dev_name}</span>")

    # no telemetry or version check (they phone home, and hold the process open when offline)
    with gr.Blocks(title="BetterDeclipper", delete_cache=(3600, 86400), analytics_enabled=False) as demo:
        sess = gr.State(lambda: Session(SESSION_ROOT), delete_callback=Session.close)

        with gr.Sidebar(open=True, width=330):
            gr.Markdown("### Restoration")
            preset = gr.Radio(list(PRESETS), value="normal", label="Preset",
                              info="fast = quickest, normal = good default, high / best = a little better, slower")
            mode = gr.Dropdown(MODE_CHOICES, value="auto", label="Mode", info=MODE_INFO)
            with gr.Accordion("Advanced", open=False):
                clip = gr.Number(None, label="Force clip level (dBFS)", placeholder="e.g. -12",
                                 info="Skips the analysis and restores only above this level (hard mode).")
                knee = gr.Number(None, label="Force soft knee (dBFS)", placeholder="e.g. -9",
                                 info="Skips the analysis; every sample above it may only grow (soft mode).")
                max_gain = gr.Number(None, label="Gain cap (dB)", placeholder="off", minimum=0,
                                     info="Restored samples stay within this much of the clip level.")
                device = gr.Radio(["auto", "cuda", "cpu"] if dev == "cuda" else ["auto", "cpu"], value="auto",
                                  label="Device")
            gr.Markdown("### Output")
            fmt = gr.Dropdown(FORMAT_CHOICES, value="wav32f", label="Format", info=FORMAT_INFO)
            normalize = gr.Checkbox(False, label="Normalize the peak")
            target = gr.Number(-0.1, label="Peak level (dBFS)", maximum=0, step=0.1,
                               info="Used to normalize, and when PCM / FLAC could not hold the restored peaks.")
            folder = gr.Textbox(label="Also save to folder", placeholder="optional, e.g. D:\\Music\\declipped",
                                visible=folders)

        gr.Markdown(header)
        # render_children: hidden tabs stay mounted, so switching tabs keeps everything as it was
        with gr.Tabs():
            # ---- single file --------------------------------------------------------------------
            with gr.Tab("Declip", render_children=True):
                with gr.Row(equal_height=False):
                    inp = gr.Audio(label="Input", sources=["upload"], type="filepath", format=None,
                                   editable=False, scale=3)
                    with gr.Column(scale=1, min_width=250):
                        run_btn = gr.Button("Declip", variant="primary", size="lg", interactive=False)
                        stop_btn = gr.Button("Stop", variant="stop", visible=False)
                        inp_info = gr.Markdown(IDLE)
                        status = gr.Markdown("", min_height=72)  # run summary; the progress bar shows here
                with gr.Row(visible=False, equal_height=False) as results_row:
                    with gr.Column(scale=3):
                        history = gr.Radio([], label="Results: pick one to view and download it")
                        with gr.Row():
                            remove_btn = gr.Button("Remove this result", size="sm", scale=0, min_width=160)
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
                    with gr.Tabs(selected="full") as subtabs:
                        with gr.Tab("Full length", id="full", render_children=True):
                            ov_image = gr.Image(type="numpy", format="png", show_label=False, interactive=False,
                                                buttons=["download", "fullscreen"], elem_id="bd-overview")
                            gr.Markdown(OVERVIEW_HELP, elem_classes=["bd-dim"])
                            ov_audio = gr.Audio(label="Full length", type="filepath", interactive=False, buttons=[])
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
                            clip_audio = gr.Audio(label="This snippet", type="filepath", interactive=False,
                                                  buttons=["download"])
                    with gr.Accordion("Analysis: what the declipper found", open=False, visible=False) as report_box:
                        report = gr.Textbox(label="Analysis", lines=10, max_lines=40, elem_id="bd-report",
                                            buttons=["copy"], interactive=False)

            # ---- batch --------------------------------------------------------------------------
            with gr.Tab("Batch", render_children=True):
                gr.Markdown("Declip many files with the settings on the left. Results are named like the "
                            "command line names them, e.g. `song [auto clip normal].wav`.")
                batch_files = gr.File(label="Audio files", file_count="multiple", type="filepath",
                                      file_types=list(AUDIO_EXTS))
                batch_folder = gr.Textbox(label="... and/or every audio file in this folder", visible=folders,
                                          placeholder="e.g. D:\\Music\\to declip",
                                          info="Results go next to the files, or to 'Also save to folder'.")
                with gr.Row():
                    batch_btn = gr.Button("Declip all", variant="primary")
                    batch_stop = gr.Button("Stop after this step", variant="stop", visible=False)
                batch_status = gr.Markdown(min_height=72)
                batch_table = gr.Dataframe(headers=BATCH_HEADERS, interactive=False, visible=False, wrap=True)
                batch_zip = gr.File(label="All results (.zip)", visible=False, interactive=False)
                batch_out = gr.File(label="Results", file_count="multiple", visible=False, interactive=False)

            with gr.Tab("Help", render_children=True):
                gr.Markdown(HELP)

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
                    d[ov_audio] = gr.update(value=s.full_audio(view_v), label=_player_label(s, "Full length"))
                    s.shown["ov_audio"] = keys["ov_audio"]
            else:
                if s.shown.get("sn_img") != keys["sn_img"]:
                    d[image] = s.render(view_v, sc)
                    s.shown["sn_img"] = keys["sn_img"]
                if s.shown.get("sn_audio") != keys["sn_audio"]:
                    d[clip_audio] = gr.update(value=s.snippet_audio(view_v), label=_player_label(s, "This snippet"))
                    s.shown["sn_audio"] = keys["sn_audio"]
            if full:
                names = channel_names(s.input.channels)
                d[chan] = gr.update(choices=names, value=names[s.channel], visible=len(names) > 1)
                choices, value = s.snippet_choices()
                d[snip] = gr.update(choices=choices, value=value)
                hist, sel = s.history()
                d[history] = gr.update(choices=hist, value=sel)
                run = s.run()
                d[report] = run.report if run else ""
                d[report_box] = gr.update(visible=run is not None)
                d[rerank_btn] = gr.update(visible=run is not None and len(s.runs) > 1)
            return d

        def download_of(s, out):
            run = s.run()
            if run is None:
                return {download: None, dl_note: ""}
            path, note = s.export(run.id, out)
            return {download: path, dl_note: note}

        media = [ov_image, ov_audio, image, clip_audio]
        view_out = media + [chan, snip, history, report, report_box, rerank_btn]
        out_inputs = [fmt, normalize, target, folder]
        hidden = lambda: {status: "", results_row: gr.update(visible=False), viewer_col: gr.update(visible=False)}

        # ---- single file: input and runs --------------------------------------------------------
        def on_upload(s, path, scale_v):
            if not path:
                return gr.skip()
            if s.input is not None and os.path.abspath(path) == os.path.abspath(s.input.path):
                return gr.skip()  # the file that is already loaded (its results stay)
            s.cancel.set()  # a run of the previous input stops at its next step (Declip clears this)
            try:
                inp_file = s.load(path)
            except Exception as e:  # unreadable / unsupported file
                s.clear()
                return {inp_info: f"Could not read this file ({e}). WAV, FLAC, AIFF, MP3, Ogg and Opus files work.",
                        run_btn: gr.update(interactive=False), **hidden()}
            d = viewer(s, "Before", scale_v)
            d.update({inp_info: inp_file.describe(), status: "", run_btn: gr.update(interactive=True),
                      view: "Before", results_row: gr.update(visible=False), viewer_col: gr.update(visible=True)})
            return d

        def on_clear(s):
            s.cancel.set()
            s.clear()
            return {inp_info: IDLE, run_btn: gr.update(interactive=False), **hidden()}

        # upload / clear only: they fire on what the user does, not when the component is shown again
        inp.upload(on_upload, [sess, inp, scale],
                   [inp_info, status, run_btn, results_row, viewer_col, view] + view_out)
        inp.clear(on_clear, sess, [inp_info, status, run_btn, results_row, viewer_col])

        def on_run_start(s):
            s.cancel.clear()
            return gr.update(visible=False), gr.update(visible=True), ""

        def on_run(s, preset_v, mode_v, clip_v, knee_v, gain_v, device_v, fmt_v, norm_v, target_v, folder_v,
                   scale_v, progress=gr.Progress()):
            if s.input is None:
                raise gr.Error("Upload an audio file first.")
            st = _settings(preset_v, mode_v, clip_v, knee_v, gain_v, device_v)
            out = output_of(fmt_v, norm_v, target_v, folder_v)
            progress(0, desc="Analyzing the clipping")
            try:
                run = s.process(st, lambda i, n, el: progress((i, n), desc="Declipping", unit="steps"))
            except Cancelled:
                return {status: "Stopped."}
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
            d.update({status: msg, results_row: gr.update(visible=True), view: "After"})
            return d

        def on_run_end():
            return gr.update(visible=True), gr.update(visible=False)

        run_btn.click(on_run_start, sess, [run_btn, stop_btn, status], queue=False).then(
            on_run, [sess, preset, mode, clip, knee, max_gain, device] + out_inputs + [scale],
            [status, results_row, download, dl_note, view] + view_out,
            concurrency_id="declip", concurrency_limit=1, show_progress_on=[status]).then(
            on_run_end, None, [run_btn, stop_btn], queue=False)
        stop_btn.click(lambda s: s.cancel.set(), sess, None, queue=False)

        def on_history(s, rid, view_v, scale_v, *out_v):
            s.select(rid)
            view_v = "After" if view_v == "Before" else view_v
            d = viewer(s, view_v, scale_v)
            d.update(download_of(s, output_of(*out_v)))
            d[view] = view_v
            return d

        history.input(on_history, [sess, history, view, scale] + out_inputs,
                      [view, download, dl_note] + view_out, show_progress="minimal")

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
                         [results_row, view, status, download, dl_note] + view_out)

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

        view.input(on_view, [sess, view, scale], media, show_progress="minimal")
        scale.input(on_view, [sess, view, scale], media, show_progress="minimal")

        def on_chan(s, name, view_v, scale_v):
            if s.input is None:
                return gr.skip()
            names = channel_names(s.input.channels)
            with s.lock:
                s.channel = names.index(name) if name in names else 0
            return viewer(s, view_v, scale_v, full=False) or gr.skip()

        chan.input(on_chan, [sess, chan, view, scale], media, show_progress="minimal")

        def on_subtab(s, view_v, scale_v, evt: gr.SelectData):
            s.tab = "snip" if evt.value == "Snippets" or evt.index == 1 else "full"
            if s.input is None:
                return gr.skip()
            return viewer(s, view_v, scale_v, full=False) or gr.skip()

        # the Tabs' select event (a Tab's own select event only fires on its first selection)
        subtabs.select(on_subtab, [sess, view, scale], media, show_progress="minimal")

        def on_overview_click(s, view_v, scale_v, evt: gr.SelectData):
            if s.input is None or not evt.index or evt.index[0] >= R.PLOT_W:
                return gr.skip()
            s.goto_x(evt.index[0])
            s.tab = "snip"
            d = viewer(s, view_v, scale_v)
            d[subtabs] = gr.Tabs(selected="snip")
            return d

        ov_image.select(on_overview_click, [sess, view, scale], [subtabs] + view_out, show_progress="minimal")

        def on_step(step):
            def fn(s, view_v, scale_v):
                if s.input is None:
                    return gr.skip()
                s.goto(s.snip_idx + step)
                return viewer(s, view_v, scale_v)
            return fn

        prev_btn.click(on_step(-1), [sess, view, scale], view_out, show_progress="minimal")
        next_btn.click(on_step(+1), [sess, view, scale], view_out, show_progress="minimal")

        def on_snip(s, value, view_v, scale_v):
            if s.input is None or value in (None, "custom"):
                return gr.skip()
            s.goto(int(value))
            return viewer(s, view_v, scale_v)

        snip.input(on_snip, [sess, snip, view, scale], view_out, show_progress="minimal")

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

        goto.submit(on_goto, [sess, goto, view, scale], view_out, show_progress="minimal")

        def on_rerank(s, view_v, scale_v):
            run = s.run()
            if run is None:
                return gr.skip()
            s.rank_from(run)
            return viewer(s, view_v, scale_v)

        rerank_btn.click(on_rerank, [sess, view, scale], view_out, show_progress="minimal")

        # ---- batch ------------------------------------------------------------------------------
        def on_batch_start(s):
            s.cancel_batch.clear()
            return gr.update(visible=False), gr.update(visible=True)

        def on_batch(s, files, src_v, preset_v, mode_v, clip_v, knee_v, gain_v, device_v, fmt_v, norm_v,
                     target_v, folder_v, progress=gr.Progress()):
            st = _settings(preset_v, mode_v, clip_v, knee_v, gain_v, device_v)
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

        batch_btn.click(on_batch_start, sess, [batch_btn, batch_stop], queue=False).then(
            on_batch, [sess, batch_files, batch_folder, preset, mode, clip, knee, max_gain, device] + out_inputs,
            [batch_status, batch_table, batch_zip, batch_out],
            concurrency_id="declip", concurrency_limit=1, show_progress_on=[batch_status]).then(
            on_batch_end, None, [batch_btn, batch_stop], queue=False)
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
