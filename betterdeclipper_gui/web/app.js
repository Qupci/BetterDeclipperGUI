/* BetterDeclipper GUI, browser side: keyboard shortcuts, the two players (full length and snippet: playback
   carries on across before / after / delta and results, a playhead runs over the view, a click plays from a
   moment) and Declip's wait for an upload in progress. window.BD_LAYOUT, set before this script, gives the
   plot area of the rendered views. */
(() => {
  'use strict';
  const L = Object.assign({plotW: 1224, top: 26, bottom: 599}, window.BD_LAYOUT || {});
  const VIEW_OF = {full: '#bd-overview', snip: '#bd-image'};
  const VIEW_IMG = '#bd-overview img, #bd-image img';
  const players = {};
  window.bdPlayers = players;
  const FADE_S = 0.05;  // cross-fade between two versions (a hard switch clicks where they differ)
  let startLag = 0.04;  // how long a started element takes to get going (measured at every switch)

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  /* resolves true when `el` fires `ok`, false on an error or after `ms` */
  function when(el, ok, ms = 15000) {
    return new Promise((resolve) => {
      let timer = 0;
      const done = (v) => {
        clearTimeout(timer);
        el.removeEventListener(ok, yes);
        el.removeEventListener('error', no);
        resolve(v);
      };
      const yes = () => done(true);
      const no = () => done(false);
      el.addEventListener(ok, yes);
      el.addEventListener('error', no);
      timer = setTimeout(() => done(false), ms);
    });
  }

  /* m:ss.d, or h:mm:ss.d from an hour on */
  function fmt(t) {
    const d = Math.round(Math.max(0, t || 0) * 10);
    const h = Math.floor(d / 36000), m = Math.floor(d / 600) % 60, s = Math.floor(d / 10) % 60;
    const p2 = (v) => String(v).padStart(2, '0');
    return (h ? `${h}:${p2(m)}` : `${m}`) + `:${p2(s)}.${d % 10}`;
  }

  /* a '/gradio_api/file=...' path -> URL (the app may live below a path, e.g. behind a proxy) */
  function fileUrl(src) {
    const root = (window.gradio_config && window.gradio_config.root) || '';
    try {
      return new URL(root ? root.replace(/\/+$/, '') + src : src.replace(/^\//, ''), location.href).href;
    } catch (e) {
      return src;
    }
  }

  /* where an <img> draws its picture (object-fit scale-down / contain): client left, top and scale */
  function imgBox(img) {
    const r = img.getBoundingClientRect(), nw = img.naturalWidth, nh = img.naturalHeight;
    if (!nw || !nh || !r.width || !r.height) return null;
    let k = Math.min(r.width / nw, r.height / nh);
    if (getComputedStyle(img).objectFit === 'scale-down') k = Math.min(k, 1);
    return {left: r.left + (r.width - nw * k) / 2, top: r.top + (r.height - nh * k) / 2, k};
  }

  const shown = (el) => !!(el && el.isConnected && el.offsetParent !== null);

  // ---- players ------------------------------------------------------------------------------------
  class Player {
    constructor(pid) {
      this.pid = pid;       // 'full' or 'snip'
      this.el = null;       // the controls (the gr.HTML element)
      this.audio = null;
      this.data = null;     // {src, name, label, note, tl (timeline), file, t0, dur, dl}
      this.gen = 0;         // bumped by every new source, so a slower switch still in flight gives up
      this.marked = false;  // the playhead is drawn (once played or sought on this timeline)
      this.dragging = false;
      this.raf = 0;
      this.head = null;
    }

    mount(el, props, watch) {
      this.el = el;
      if (!el.bdBound) {
        el.bdBound = true;
        el.addEventListener('click', (e) => {
          if (e.target.closest('.bd-play')) {
            e.preventDefault();
            this.toggle();
          }
        });
        const seek = el.querySelector('.bd-seek');
        if (seek) {
          seek.addEventListener('input', () => {
            this.dragging = true;
            this.seekTo(seek.value / 1000, false);
          });
          seek.addEventListener('change', () => {
            this.dragging = false;
            this.seekTo(seek.value / 1000, false);
          });
        }
        if (watch) watch('value', () => this.update(props.value));
      }
      this.update(props.value);
    }

    update(json) {
      let d = null;
      try {
        d = json ? JSON.parse(json) : null;
      } catch (e) {
        d = null;
      }
      const prev = this.data;
      if (!d || !d.src) {
        this.data = null;
        this.unload();
        return;
      }
      this.data = d;
      if (prev && prev.src === d.src && this.audio) {
        this.render();
        return;
      }
      this.load(d, !!prev && prev.tl === d.tl, !!prev && prev.file === d.file);
    }

    visible() {
      return shown(this.el);
    }

    duration() {
      const a = this.audio;
      return a && isFinite(a.duration) && a.duration > 0 ? a.duration : (this.data && this.data.dur) || 0;
    }

    make(url) {
      const a = new Audio();
      a.preload = 'auto';
      const sync = () => {
        if (a !== this.audio) return;
        this.render();
        if (!a.paused) this.loop();
      };
      for (const ev of ['play', 'pause', 'ended', 'seeked', 'loadedmetadata', 'timeupdate']) {
        a.addEventListener(ev, sync);
      }
      a.src = url;
      return a;
    }

    release(a) {
      a.pause();
      a.removeAttribute('src');
      try {
        a.load();
      } catch (e) { /* nothing to free */ }
    }

    unload() {
      this.gen++;
      if (this.audio) this.release(this.audio);
      this.audio = null;
      this.marked = false;
      this.render();
    }

    /* Switch to a new source. On the same timeline (another version of the same audio) the position is kept;
       if it is playing, the new version takes over at the same moment (lineUp), so playback carries on. A new
       snippet of the same file starts from its beginning, still playing. */
    async load(d, sameTimeline, sameFile) {
      const gen = ++this.gen;
      const old = this.audio;
      const playing = !!(old && !old.paused && !old.ended);
      if (!sameTimeline) this.marked = false;
      const next = this.make(fileUrl(d.src));
      const ok = await when(next, 'loadedmetadata');
      if (gen !== this.gen) return this.release(next);
      if (ok && old && sameTimeline) {
        if (playing && this.visible()) {
          if (await this.lineUp(old, next, gen)) return;
          if (gen !== this.gen) return this.release(next);
        }
        if (old.currentTime > 0) {  // paused (or could not line up): at the same position
          next.currentTime = Math.min(old.currentTime, next.duration || old.currentTime);
          await when(next, 'seeked', 4000);
          if (gen !== this.gen) return this.release(next);
        }
      }
      const resume = playing && !old.paused && (sameTimeline || sameFile) && this.visible();
      this.audio = next;
      if (resume) this.start();
      if (old) this.release(old);
      this.render();
    }

    /* Start `next` (sought ahead) when the playing `old` reaches that point, less the start-up lag, and
       cross-fade: the new version continues where the old one is. False if it could not keep up. */
    async lineUp(old, next, gen) {
      let lead = 0.15;
      for (let k = 0; k < 4; k++, lead *= 2) {  // on a slow connection, aim further ahead
        const target = old.currentTime + lead;
        next.currentTime = target;
        await when(next, 'seeked', 4000);
        if (next.readyState < 3) await when(next, 'canplay', 4000);
        if (gen !== this.gen || old.paused) return false;
        // when old gets to `at`, on the wall clock: its currentTime moves in steps, so take the median of a few
        const rate = old.playbackRate || 1, offs = [];
        for (let i = 0; i < 5; i++) {
          offs.push(performance.now() / 1000 - old.currentTime / rate);
          await sleep(3);
        }
        const at = (target - startLag) / rate + offs.sort((x, y) => x - y)[2];
        const wait = at - performance.now() / 1000;
        if (wait < 0) continue;
        if (wait > 0.012) await sleep((wait - 0.01) * 1000);
        while (performance.now() / 1000 < at) { /* the last few ms */ }
        if (gen !== this.gen || old.paused) return false;
        next.volume = 0;
        const w0 = performance.now();
        this.audio = next;
        this.start();  // (old plays on and fades out)
        this.crossfade(old, next, gen);
        setTimeout(() => {  // how late did it get going? (next time, start that much earlier)
          if (next.paused || next !== this.audio) return;
          const lag = (performance.now() - w0) / 1000 - (next.currentTime - target) / rate;
          if (lag > -0.05 && lag < 0.4) startLag = Math.max(0, Math.min(0.25, 0.6 * startLag + 0.4 * lag));
        }, 600);
        this.render();
        return true;
      }
      return false;
    }

    crossfade(old, next, gen) {
      const steps = 6;
      let i = 0;
      const step = () => {
        if (gen !== this.gen || next.paused) {  // switched again, or paused: no fade left to do
          next.volume = 1;
          this.release(old);
          return;
        }
        const f = Math.min(1, ++i / steps);
        next.volume = Math.sin((f * Math.PI) / 2);  // equal power
        old.volume = Math.cos((f * Math.PI) / 2);
        if (f < 1) setTimeout(step, (FADE_S * 1000) / steps);
        else this.release(old);
      };
      setTimeout(step, startLag * 1000);  // the new one is getting going: the old one plays on meanwhile
    }

    start() {
      const a = this.audio;
      if (!a) return;
      for (const p of Object.values(players)) {
        if (p !== this && p.audio && !p.audio.paused) p.audio.pause();
      }
      if (a.ended) a.currentTime = 0;
      this.marked = true;
      const pr = a.play();
      if (pr && pr.catch) pr.catch(() => {});
      this.loop();
    }

    toggle() {
      const a = this.audio;
      if (!a) return;
      if (a.paused || a.ended) this.start();
      else a.pause();
    }

    seekTo(frac, play) {
      const a = this.audio, dur = this.duration();
      if (!a || !dur) return;
      a.currentTime = Math.max(0, Math.min(1, frac)) * dur;
      this.marked = true;
      if (play && (a.paused || a.ended)) this.start();
      this.render();
    }

    loop() {
      if (this.raf) return;
      const tick = () => {
        this.raf = 0;
        const a = this.audio;
        if (!a || a.paused) return this.render();
        if (!this.visible()) return a.pause();  // its view was hidden (another tab)
        this.renderTime();
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    }

    render() {
      const el = this.el, d = this.data, a = this.audio;
      const box = el && el.isConnected ? el.querySelector('.bd-player') : null;
      if (box) {
        box.classList.toggle('bd-empty', !d);
        box.classList.toggle('bd-playing', !!(a && !a.paused && !a.ended));
        const text = (sel, t) => {
          const n = box.querySelector(sel);
          if (n && n.textContent !== t) n.textContent = t;
        };
        text('.bd-label', d ? d.label : 'Nothing to play yet');
        text('.bd-note', d && d.note ? d.note : '');
        const dl = box.querySelector('.bd-dl');
        if (dl) {
          dl.hidden = !(d && d.dl);
          if (d && d.dl) {
            dl.href = fileUrl(d.src);
            dl.setAttribute('download', d.name || '');
          }
        }
      }
      this.renderTime();
    }

    renderTime() {
      const el = this.el, d = this.data, a = this.audio;
      if (el && el.isConnected) {
        const dur = this.duration(), t = a ? a.currentTime : 0, t0 = (d && d.t0) || 0;
        const time = el.querySelector('.bd-time');
        const txt = d ? `${fmt(t0 + t)} / ${fmt(t0 + dur)}` : '';
        if (time && time.textContent !== txt) time.textContent = txt;
        const seek = el.querySelector('.bd-seek');
        if (seek && !this.dragging) seek.value = dur ? String(Math.round((t / dur) * 1000)) : '0';
      }
      this.place();
    }

    /* the playhead: a line over the waveform and the spectrogram of the view this player belongs to */
    place() {
      const root = document.querySelector(VIEW_OF[this.pid]);
      const img = root && root.querySelector('img');
      const a = this.audio, dur = this.duration();
      const box = img && a && dur && this.marked && shown(root) ? imgBox(img) : null;
      let head = this.head;
      if (!box) {
        if (head) head.style.display = 'none';
        return;
      }
      if (!head || head.parentNode !== root) {
        if (head) head.remove();
        head = this.head = document.createElement('div');
        head.className = 'bd-playhead';
        root.appendChild(head);
        observe(root);
      }
      const rr = root.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, a.currentTime / dur)) * L.plotW;
      head.style.left = `${box.left - rr.left - root.clientLeft + x * box.k}px`;
      head.style.top = `${box.top - rr.top - root.clientTop + L.top * box.k}px`;
      head.style.height = `${(L.bottom - L.top) * box.k}px`;
      head.style.display = '';
    }
  }

  const ro = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(() => placeAll());
  const observed = new WeakSet();
  function observe(el) {
    if (ro && !observed.has(el)) {
      observed.add(el);
      ro.observe(el);
    }
  }
  function placeAll() {
    for (const p of Object.values(players)) p.place();
  }
  window.addEventListener('resize', placeAll);
  document.addEventListener('load', (e) => {  // a view was rendered again
    const t = e.target;
    if (t && t.tagName === 'IMG' && t.closest && t.closest('#bd-overview, #bd-image')) placeAll();
  }, true);

  /* called by the players' gr.HTML (js_on_load). Which player it is comes from the host's elem_id
     (bd-player-full / bd-player-snip): a component mounted by an update can get its props a moment later
     (they arrive through watch). */
  window.bdMountPlayer = (element, props, watch, tries = 0) => {
    const host = element.closest && element.closest('[id^="bd-player-"]');
    const pid = host ? host.id.slice('bd-player-'.length) : props.pid;
    if (!VIEW_OF[pid]) {  // not in the page yet
      if (tries < 60) requestAnimationFrame(() => window.bdMountPlayer(element, props, watch, tries + 1));
      return;
    }
    const p = players[pid] || (players[pid] = new Player(pid));
    p.mount(element, props, watch);
  };

  // ---- clicks on the views: play from there ----------------------------------------------------
  const modified = (e) => e.shiftKey || e.ctrlKey || e.metaKey || e.altKey;
  document.addEventListener('mousedown', (e) => {  // no text selection or image drag on Shift+click
    if (modified(e) && e.target.closest && e.target.closest(VIEW_IMG)) e.preventDefault();
  }, true);
  document.addEventListener('click', (e) => {
    const img = e.target && e.target.closest ? e.target.closest(VIEW_IMG) : null;
    if (!img) return;
    const pid = img.closest('#bd-overview') ? 'full' : 'snip';
    if (pid === 'full' && !modified(e)) return;  // a plain click on the full length opens that moment in Snippets
    e.preventDefault();
    e.stopPropagation();
    const p = players[pid], box = imgBox(img);
    if (!p || !p.audio || !box) return;
    const x = (e.clientX - box.left) / box.k;
    if (x >= 0 && x < L.plotW) p.seekTo(x / L.plotW, true);
  }, true);

  // ---- keyboard ------------------------------------------------------------------------------------
  const typing = (t) => {
    const tag = t && t.tagName;
    return tag === 'TEXTAREA' || tag === 'SELECT' || !!(t && t.isContentEditable) ||
      (tag === 'INPUT' && !['radio', 'checkbox', 'button', 'range', 'submit', 'reset', 'file'].includes(t.type));
  };
  /* a control reached with the Tab key: Space activates it, as usual (a clicked one plays instead). The
     browser's :focus-visible can't tell: any key press makes the focus "visible". */
  let tabbed = false;
  document.addEventListener('pointerdown', () => { tabbed = false; }, true);
  document.addEventListener('keydown', (e) => { if (e.key === 'Tab') tabbed = true; }, true);
  const keyboardFocused = (t) => tabbed && !!(t && t.matches &&
    t.matches('button, a[href], [role="tab"], [role="button"], input[type="checkbox"], input[type="radio"], summary'));
  const onViewer = () => shown(document.getElementById('bd-view'));
  let spaceTaken = false;

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || typing(e.target)) return;
    if (e.key === ' ' || e.code === 'Space') {
      if (keyboardFocused(e.target) || !onViewer()) return;
      const p = ['full', 'snip'].map((k) => players[k]).find((q) => q && q.visible());
      if (!p) return;
      e.preventDefault();
      spaceTaken = true;
      if (!e.repeat) p.toggle();
      return;
    }
    if (e.repeat || !onViewer()) return;
    const inField = e.target && e.target.tagName === 'INPUT';
    const pick = (value) => {
      const el = document.querySelector(`#bd-view input[type="radio"][value="${value}"]`);
      if (!el) return;
      if (!el.checked) el.click();
      e.preventDefault();
    };
    const press = (id) => {
      const b = document.getElementById(id);
      if (b && !inField && shown(b)) {
        b.click();
        e.preventDefault();
      }
    };
    const tab = (id) => {
      const b = document.querySelector(`button[role="tab"][data-tab-id="${id}"]`);
      if (b) {
        b.click();
        e.preventDefault();
      }
    };
    const result = (n) => {
      const r = document.querySelectorAll('#bd-history input[type="radio"]')[n - 1];
      if (!r) return;
      if (!r.checked) r.click();
      e.preventDefault();
    };
    switch (e.key) {
      case 'b': case 'B': pick('Before'); break;
      case 'a': case 'A': pick('After'); break;
      case 'd': case 'D': pick('Delta'); break;
      case 'f': case 'F': tab('full'); break;
      case 's': case 'S': tab('snip'); break;
      case 'ArrowLeft': press('bd-prev'); break;
      case 'ArrowRight': press('bd-next'); break;
      default:
        if (/^[1-9]$/.test(e.key)) result(Number(e.key));
    }
  }, true);
  document.addEventListener('keyup', (e) => {  // a clicked button keeps the focus: Space must not press it
    if ((e.key === ' ' || e.code === 'Space') && spaceTaken) {
      spaceTaken = false;
      e.preventDefault();
    }
  }, true);

  // ---- Declip during an upload: wait until the file is on the server -----------------------------
  window.bdUploading = () => !!document.querySelector('#bd-input .uploading');
  window.bdWaitForUpload = async () => {
    window.bdStopWait = false;
    while (window.bdUploading() && !window.bdStopWait) await sleep(100);
    await sleep(60);  // the input's new value settles
  };
})();
