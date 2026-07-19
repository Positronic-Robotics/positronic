'use strict';

/* ─────────────────────────────────────────────────────────────────────────
 * Mock hook — `?mock=1` (idle), `?mock=running`, `?mock=console`,
 * `?mock=stopping`, `?mock=error`. Stubs fetch with canned fixtures matching
 * the API contract so the page renders fully with no backend. Kept isolated.
 * ───────────────────────────────────────────────────────────────────────── */
(function installMock() {
  const mock = new URLSearchParams(location.search).get('mock');
  if (!mock) return;

  const now = Date.now() / 1000;
  const presets = {
    error: mock === 'error'
      ? "presets.toml: expected ']' at line 12 (parse error)"
      : null,
    runner: 'LOG_LEVEL=info uv run --extra hardware positronic-inference phail',
    default_task: 'pick up the red block and place it in the bin',
    common: {
      raw: "--output.dir=s3://pos-runs/$(date +%Y-%m-%d)/session --output.fps=30 --session.operator=$USER",
      args: ['--output.dir=s3://pos-runs/$(date +%Y-%m-%d)/session', '--output.fps=30', '--session.operator=$USER'],
    },
    common_resolved: [
      '--output.dir=s3://pos-runs/2026-07-18/session',
      '--output.fps=30',
      '--session.operator=vertix',
    ],
    presets: {
      'pi0_franka_v3': {
        raw: '--policy.host=infer-a.lan --policy.port=8443 --policy.secure=true --policy.history_frames=8 --policy.pad_start=true --policy.checkpoint=s3://ckpt/pi0/v3',
        args: ['--policy.host=infer-a.lan', '--policy.port=8443', '--policy.secure=true', '--policy.history_frames=8', '--policy.pad_start=true', '--policy.checkpoint=s3://ckpt/pi0/v3'],
      },
      'gr00t_n15_droid': {
        raw: '--policy.host=infer-b.lan --policy.port=8443 --policy.secure=true --policy.history_frames=1 --policy.pad_start=false --policy.checkpoint=s3://ckpt/gr00t/n15',
        args: ['--policy.host=infer-b.lan', '--policy.port=8443', '--policy.secure=true', '--policy.history_frames=1', '--policy.pad_start=false', '--policy.checkpoint=s3://ckpt/gr00t/n15'],
      },
      'openpi_fast_local': {
        raw: '--policy.host=localhost --policy.port=9000 --policy.secure=false --policy.history_frames=4 --policy.pad_start=true',
        args: ['--policy.host=localhost', '--policy.port=9000', '--policy.secure=false', '--policy.history_frames=4', '--policy.pad_start=true'],
      },
    },
  };

  const running = {
    preset: 'pi0_franka_v3',
    task: 'pick up the red block and place it in the bin',
    command: presets.runner + ' ' + presets.presets.pi0_franka_v3.raw,
    started_at: now - (mock === 'running' ? 22 : 128),
    console_ready: mock === 'console' || mock === 'stopping',
    stop_requested_at: mock === 'stopping' ? now - 30 : null,
  };

  const state = {
    '1': { status: 'idle', run: null,
           last_run: { preset: 'gr00t_n15_droid', exit_code: 0, started_at: now - 900, ended_at: now - 300, stopped: false },
           console_port: 8009, log_length: 3 },
    'error': { status: 'idle', run: null, last_run: null, console_port: 8009, log_length: 0 },
    'running': { status: 'running', run: running, last_run: null, console_port: 8009, log_length: 6 },
    'console': { status: 'running', run: running, last_run: null, console_port: 8009, log_length: 12 },
    'stopping': { status: 'stopping', run: running, last_run: null, console_port: 8009, log_length: 14 },
  }[mock] || null;

  const logLines = [
    '[12:00:01] launcher: starting subprocess',
    '[12:00:01] positronic-inference phail --policy.host=infer-a.lan …',
    '[12:00:03] desk: opening brakes',
    '[12:00:11] desk: FCI activated',
    '[12:00:12] robot: homing',
    '[12:00:28] policy: connecting to infer-a.lan:8443 (secure)',
    '[12:00:31] policy: session established, warming up model',
    '[12:00:44] policy: model ready (warmup 13.2s)',
    '[12:00:45] console: listening on :8009',
    '[12:00:45] launcher: child console is answering',
    '[12:00:46] loop: waiting for operator',
    '[12:01:02] loop: task received',
  ];

  const json = (obj) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(obj) });

  window.fetch = function (url, opts) {
    const u = String(url);
    if (u.startsWith('/api/presets')) return json(presets.error ? { ...presets, presets: {} } : presets);
    if (u.startsWith('/api/state')) return json(state);
    if (u.startsWith('/api/logs')) {
      const off = Number(new URLSearchParams(u.split('?')[1] || '').get('offset') || 0);
      const count = state && state.status !== 'idle' ? logLines.length : 3;
      return json({ next: count, lines: logLines.slice(off, count), truncated: false });
    }
    if (u.startsWith('/api/start') || u.startsWith('/api/stop')) return json({ ok: true });
    return json({});
  };
})();

/* ─────────────────────────────────────────────────────────────────────────
 * App
 * ───────────────────────────────────────────────────────────────────────── */

const $ = (id) => document.getElementById(id);
const app = document.querySelector('.app');

const els = {
  runPreset: $('runPreset'), runElapsed: $('runElapsed'),
  lampLabel: $('lampLabel'), stopBtn: $('stopBtn'), forceBtn: $('forceBtn'),
  presetList: $('presetList'), presetError: $('presetError'), presetEmpty: $('presetEmpty'),
  policyName: $('policyName'), policyArgs: $('policyArgs'), sessionArgs: $('sessionArgs'),
  taskInput: $('taskInput'), extraInput: $('extraInput'),
  commandText: $('commandText'), copyBtn: $('copyBtn'),
  launchBtn: $('launchBtn'), startError: $('startError'), lastRun: $('lastRun'),
  bootPlaceholder: $('bootPlaceholder'), consoleWrap: $('consoleWrap'),
  consoleFrame: $('consoleFrame'), newTabLink: $('newTabLink'),
  logPanel: $('logPanel'), logBody: $('logBody'), logSub: $('logSub'), jumpLatest: $('jumpLatest'),
};

const LAMP_LABELS = { idle: 'idle', booting: 'booting', running: 'live', stopping: 'stopping', failed: 'failed' };

// Arg-name heuristics: which values distinguish one model from another.
const KEY_ARGS = ['host', 'history_frames', 'pad_start', 'checkpoint', 'model', 'endpoint', 'action_horizon', 'steps'];
const BOILERPLATE_ARGS = ['port', 'secure', 'header', 'headers', 'token', 'api_key', 'apikey', 'timeout', 'retries', 'tls', 'ssl', 'verify'];

let presetsData = null;      // last /api/presets payload
let selectedPreset = null;   // preset name
let userEditedTask = false;  // don't clobber operator's edits on refetch

let stateData = null;
let runStartedAt = null;     // started_at of the run whose logs we're showing
let forceRevealed = false;

// log streaming
let logOffset = 0;
let logsPinned = true;
let truncationShown = false;

/* ── Fetch helpers ─────────────────────────────────────────────── */

async function getJSON(url) {
  const r = await fetch(url);
  return r.json();
}
async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await r.json(); } catch (e) { /* empty body */ }
  return { ok: r.ok, status: r.status, data };
}

/* ── Presets ───────────────────────────────────────────────────── */

async function loadPresets() {
  presetsData = await getJSON('/api/presets');
  renderPresets();
}

function renderPresets() {
  const d = presetsData;
  const hasError = d.error != null;

  els.presetError.hidden = !hasError;
  if (hasError) {
    els.presetError.textContent = d.error + ' — fix presets.toml and refresh.';
  }

  const names = Object.keys(d.presets || {});
  els.presetEmpty.hidden = hasError || names.length > 0;

  // keep current selection if still valid, else first
  if (!selectedPreset || !names.includes(selectedPreset)) {
    selectedPreset = names[0] || null;
  }

  els.presetList.innerHTML = '';
  for (const name of names) {
    const li = document.createElement('li');
    li.className = 'preset-item';
    li.textContent = name;
    li.setAttribute('role', 'button');
    li.tabIndex = 0;
    li.setAttribute('aria-selected', String(name === selectedPreset));
    li.addEventListener('click', () => selectPreset(name));
    li.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectPreset(name); }
    });
    els.presetList.appendChild(li);
  }

  if (!userEditedTask) {
    els.taskInput.value = d.default_task || '';
  }

  renderDetail();
  updateLaunchEnabled();
}

function selectPreset(name) {
  selectedPreset = name;
  for (const li of els.presetList.children) {
    li.setAttribute('aria-selected', String(li.textContent === name));
  }
  renderDetail();
}

function argClass(arg) {
  const m = arg.match(/^--?([^=]+)/);
  if (!m) return 'normal';
  const leaf = m[1].split('.').pop().toLowerCase();
  if (KEY_ARGS.includes(leaf)) return 'key';
  if (BOILERPLATE_ARGS.includes(leaf)) return 'boilerplate';
  return 'normal';
}

function renderArg(arg, cls) {
  const row = document.createElement('div');
  row.className = 'arg ' + cls;
  const eq = arg.indexOf('=');
  if (eq === -1) {
    const n = document.createElement('span');
    n.className = 'arg-name';
    n.textContent = arg;
    row.appendChild(n);
  } else {
    const n = document.createElement('span');
    n.className = 'arg-name';
    n.textContent = arg.slice(0, eq);
    const e = document.createElement('span');
    e.className = 'arg-eq';
    e.textContent = '=';
    const v = document.createElement('span');
    v.className = 'arg-val';
    v.textContent = arg.slice(eq + 1);
    row.append(n, e, v);
  }
  return row;
}

function renderDetail() {
  const d = presetsData;
  const preset = d.presets && selectedPreset ? d.presets[selectedPreset] : null;

  els.policyName.textContent = selectedPreset || '';
  els.policyArgs.innerHTML = '';
  if (preset && preset.args.length) {
    for (const arg of preset.args) els.policyArgs.appendChild(renderArg(arg, argClass(arg)));
  } else {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'Select a preset to see its policy arguments.';
    els.policyArgs.appendChild(e);
  }

  els.sessionArgs.innerHTML = '';
  const common = d.common_resolved || [];
  for (const arg of common) els.sessionArgs.appendChild(renderArg(arg, 'normal'));

  renderCommand();
}

/* ── Command strip ─────────────────────────────────────────────── */

function assembledSegments() {
  const d = presetsData || {};
  const preset = d.presets && selectedPreset ? d.presets[selectedPreset] : null;
  const runner = d.runner || '';
  const commonRaw = (d.common && d.common.raw) || '';
  const task = els.taskInput.value;
  const extra = els.extraInput.value.trim();

  const segs = [];
  if (runner) segs.push(['runner', runner], ['plain', ' ']);
  if (preset) segs.push(['preset', preset.raw], ['plain', ' ']);
  if (commonRaw) segs.push(['session', commonRaw], ['plain', ' ']);
  if (task) {
    // Mirror the backend's shlex.quote so the copied command stays a valid shell line
    // even when the task contains apostrophes; an empty task omits the flag, as the backend does.
    const quoted = "'" + task.replace(/'/g, "'\\''") + "'";
    segs.push(['mach', '--driver.task='], ['task', quoted]);
  }
  if (extra) segs.push(['plain', ' '], ['extra', extra]);
  return segs;
}

function renderCommand() {
  els.commandText.innerHTML = '';
  for (const [cls, text] of assembledSegments()) {
    if (cls === 'plain') {
      els.commandText.appendChild(document.createTextNode(text));
    } else {
      const s = document.createElement('span');
      s.className = cls;
      s.textContent = text;
      els.commandText.appendChild(s);
    }
  }
}

function commandPlainText() {
  return assembledSegments().map(([, t]) => t).join('').trim();
}

/* ── Launch / stop ─────────────────────────────────────────────── */

function updateLaunchEnabled() {
  const d = presetsData;
  const ok = d && d.error == null && selectedPreset != null;
  els.launchBtn.disabled = !ok;
}

async function launch() {
  els.startError.hidden = true;
  els.launchBtn.disabled = true;
  const res = await postJSON('/api/start', {
    preset: selectedPreset,
    task: els.taskInput.value,
    extra_args: els.extraInput.value.trim(),
  });
  if (!res.ok) {
    els.startError.hidden = false;
    els.startError.textContent = (res.data && res.data.error) || `Launch failed (${res.status}).`;
    updateLaunchEnabled();
    return;
  }
  await refreshState();
}

async function stop(force) {
  const res = await postJSON('/api/stop', { force: !!force });
  if (!res.ok && res.data && res.data.error) {
    els.logSub.textContent = res.data.error;
  }
  await refreshState();
}

/* ── State polling + view routing ──────────────────────────────── */

async function refreshState() {
  stateData = await getJSON('/api/state');
  applyState();
}

function applyState() {
  const s = stateData;
  const running = s.status === 'running' || s.status === 'stopping';

  // Effective status for the lamp: booting until the console answers.
  let effStatus = s.status;
  if (s.status === 'running' && s.run && !s.run.console_ready) effStatus = 'booting';
  app.dataset.status = effStatus;
  els.lampLabel.textContent = LAMP_LABELS[effStatus] || effStatus;

  app.dataset.view = running ? 'running' : 'idle';

  if (running) {
    renderRunning(s);
    ensureLogsFor(s.run.started_at);
  } else {
    // The latest run's logs stay available while idle — including a run that died
    // before polling ever saw it as running (bad command, boot crash).
    if (s.last_run) ensureLogsFor(s.last_run.started_at);
    renderIdle(s);
    if (runStartedAt != null) pollLogs();
  }
}

function renderRunning(s) {
  const run = s.run;
  els.runPreset.textContent = run.preset;
  els.runPreset.hidden = false;
  els.runElapsed.hidden = false;

  const stopping = s.status === 'stopping';
  els.stopBtn.hidden = false;
  els.stopBtn.disabled = stopping;
  els.stopBtn.textContent = stopping ? 'Stopping…' : 'Stop';

  // Force kill only after the graceful stop has clearly stalled.
  let showForce = false;
  if (stopping && run.stop_requested_at != null) {
    showForce = (Date.now() / 1000 - run.stop_requested_at) > 25;
  }
  if (showForce) forceRevealed = true;
  els.forceBtn.hidden = !forceRevealed;

  els.logSub.textContent = stopping
    ? 'stopping — robot shutdown and S3 sync can take a while'
    : (run.console_ready ? '' : 'booting — this takes 30–60s');

  // console vs boot placeholder
  if (run.console_ready) {
    const url = `http://${location.hostname}:${s.console_port}/`;
    if (els.consoleFrame.src !== url) els.consoleFrame.src = url;
    els.newTabLink.href = url;
    els.consoleWrap.hidden = false;
    els.bootPlaceholder.hidden = true;
  } else {
    els.consoleWrap.hidden = true;
    els.bootPlaceholder.hidden = false;
  }

  els.logPanel.hidden = false;
}

function renderIdle(s) {
  els.runPreset.hidden = true;
  els.runElapsed.hidden = true;
  els.stopBtn.hidden = true;
  els.forceBtn.hidden = true;
  forceRevealed = false;
  if (els.consoleFrame.src && !els.consoleFrame.src.startsWith('about:')) {
    els.consoleFrame.src = 'about:blank';
  }

  renderLastRun(s.last_run);
  updateLaunchEnabled();

  // Logs from the finished run stay visible until the next launch.
  els.logPanel.hidden = runStartedAt == null;
}

function renderLastRun(lr) {
  if (!lr) { els.lastRun.hidden = true; return; }
  // A run the operator stopped exits by SIGINT (code 130) — that is the intended
  // outcome, not a failure. "Failed" is reserved for exits nobody asked for.
  const clean = lr.exit_code === 0 || lr.stopped;
  els.lastRun.hidden = false;
  els.lastRun.className = 'last-run ' + (clean ? 'clean' : 'failed');
  const code = lr.exit_code == null ? 'no exit code' : `exit ${lr.exit_code}`;
  els.lastRun.innerHTML = '';
  const label = lr.stopped ? 'Last run · stopped' : (clean ? 'Last run · clean' : 'Last run · failed');
  const parts = [
    ['lr-label', label],
    ['lr-preset', lr.preset],
    ['lr-code', code],
    ['lr-when', `ended ${agoText(lr.ended_at)}`],
  ];
  for (const [cls, text] of parts) {
    const span = document.createElement('span');
    span.className = cls;
    span.textContent = text;
    els.lastRun.appendChild(span);
  }
}

/* ── Elapsed / relative time ───────────────────────────────────── */

function fmtElapsed(sec) {
  sec = Math.max(0, Math.floor(sec));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

function agoText(unixSec) {
  const d = Math.max(0, Date.now() / 1000 - unixSec);
  if (d < 60) return 'just now';
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

function tickElapsed() {
  const s = stateData;
  if (s && s.run && (s.status === 'running' || s.status === 'stopping')) {
    els.runElapsed.textContent = fmtElapsed(Date.now() / 1000 - s.run.started_at);
  }
}

/* ── Log streaming ─────────────────────────────────────────────── */

function ensureLogsFor(startedAt) {
  if (runStartedAt !== startedAt) {
    // New run — reset the stream.
    runStartedAt = startedAt;
    logOffset = 0;
    logsPinned = true;
    truncationShown = false;
    els.logBody.innerHTML = '';
    els.jumpLatest.hidden = true;
  }
}

async function pollLogs() {
  if (runStartedAt == null) return;
  const data = await getJSON(`/api/logs?offset=${logOffset}`);

  if (data.truncated && !truncationShown) {
    truncationShown = true;
    const row = document.createElement('span');
    row.className = 'log-line log-truncated';
    row.textContent = '… earlier output dropped …';
    els.logBody.appendChild(row);
  }

  if (data.lines && data.lines.length) {
    for (const line of data.lines) {
      const row = document.createElement('span');
      row.className = 'log-line';
      row.textContent = line;
      els.logBody.appendChild(row);
    }
    if (logsPinned) els.logBody.scrollTop = els.logBody.scrollHeight;
  }

  if (!els.logBody.children.length) {
    const row = document.createElement('span');
    row.className = 'log-line log-empty';
    row.textContent = 'Waiting for output…';
    els.logBody.appendChild(row);
  } else {
    const empty = els.logBody.querySelector('.log-empty');
    if (empty && els.logBody.children.length > 1) empty.remove();
  }

  logOffset = data.next != null ? data.next : logOffset;
}

function onLogScroll() {
  const b = els.logBody;
  const atBottom = b.scrollHeight - b.scrollTop - b.clientHeight < 24;
  logsPinned = atBottom;
  els.jumpLatest.hidden = atBottom;
}

/* ── Wiring ────────────────────────────────────────────────────── */

els.taskInput.addEventListener('input', () => { userEditedTask = true; renderCommand(); });
els.extraInput.addEventListener('input', renderCommand);
els.launchBtn.addEventListener('click', launch);
els.stopBtn.addEventListener('click', () => stop(false));
els.forceBtn.addEventListener('click', () => stop(true));
els.logBody.addEventListener('scroll', onLogScroll);
els.jumpLatest.addEventListener('click', () => {
  els.logBody.scrollTop = els.logBody.scrollHeight;
  logsPinned = true;
  els.jumpLatest.hidden = true;
});
els.copyBtn.addEventListener('click', async () => {
  const text = commandPlainText();
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); ta.remove();
  }
  els.copyBtn.textContent = 'Copied';
  setTimeout(() => { els.copyBtn.textContent = 'Copy'; }, 1200);
});

// Presets can change on disk; refetch when the operator returns to the tab.
window.addEventListener('focus', () => { if (app.dataset.view === 'idle') loadPresets(); });

/* ── Boot ──────────────────────────────────────────────────────── */

async function init() {
  await Promise.all([loadPresets(), refreshState()]);
  setInterval(refreshState, 1000);
  setInterval(() => { if (app.dataset.view === 'running') pollLogs(); }, 700);
  setInterval(tickElapsed, 250);
}

init();
