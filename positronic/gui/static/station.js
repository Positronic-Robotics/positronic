'use strict';

// The page holds no state of its own: it draws the newest answer of the console, and a reloaded tab draws the same.
const POLL_MS = 500;
const EDIT_DELAY_MS = 300;

// Every name this page shares with the console's server, and the only place the page spells one. The server's
// side is the routes of StationConsole.build_app, the models Status, RunView, Episode, CameraView, InstructionBody
// and EndBody, the enums Phase and Outcome, and the `detail` of a FastAPI error answer.
const ROUTES = {
  status: '/status',
  instruction: '/instruction',
  start: '/episode/start',
  end: '/episode/end',
  endRun: '/run/end',
  video: '/video',
};
const PHASE = { ready: 'ready', running: 'running', ending: 'ending', runEnded: 'run_ended' };
const OUTCOME = { pass: 'pass', fail: 'fail', timeout: 'timeout', error: 'error' };
const REQUEST = {
  instruction: (text) => ({ override: text }),
  end: (verdict) => ({ verdict: verdict }),
};
const errorDetail = (answer) => answer.detail;

// The page's copy of an answer of GET /status.
function readStatus(answer) {
  const run = answer.run;
  return {
    phase: run.phase,
    configured: run.configured,
    override: run.override,
    overrideSince: run.override_since,
    generation: run.generation,
    now: run.now,
    episodes: run.episodes.map((e) => ({
      number: e.number,
      instruction: e.instruction,
      overridden: e.overridden,
      startedAt: e.started_at,
      endedAt: e.ended_at,
      outcome: e.outcome,
      error: e.error,
    })),
    cameras: answer.cameras.map((c) => ({
      name: c.name,
      label: c.label,
      live: c.live,
      fps: c.fps,
      width: c.width,
      height: c.height,
    })),
    policy: answer.policy,
    host: answer.host,
  };
}

const ICON = {
  check:
    '<svg class="i" viewBox="0 0 16 16"><path d="M3 8.5l3.2 3.2L13 4.8" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cross:
    '<svg class="i" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round"/></svg>',
};

// How the page draws each phase and each outcome: `css` names the style in station.css.
const PHASE_VIEW = {
  [PHASE.ready]: { text: 'Ready', css: 'ready' },
  [PHASE.running]: { text: 'Running', css: 'running', doing: 'recording' },
  [PHASE.ending]: { text: 'Ending', css: 'ending', doing: 'ending' },
  [PHASE.runEnded]: { text: 'Run ended', css: 'run-ended' },
};
const OUTCOME_VIEW = {
  [OUTCOME.pass]: { label: `${ICON.check}Pass`, text: 'pass', css: 'pass' },
  [OUTCOME.fail]: { label: `${ICON.cross}Fail`, text: 'fail', css: 'fail' },
  [OUTCOME.timeout]: { label: 'Timed out', text: 'timed out', css: 'timeout' },
  [OUTCOME.error]: { label: 'Error', text: 'error', css: 'error' },
};
// An open episode's row shows the phase in place of an outcome.
const OPEN_VIEW = {
  [PHASE.running]: { label: '<span class="dot"></span>Running', css: 'running' },
  [PHASE.ending]: { label: 'Ending', css: 'ending' },
};

const $ = (id) => document.getElementById(id);
const textarea = $('instruction');
const verdictButtons = [
  [$('finish-pass'), OUTCOME.pass],
  [$('finish-fail'), OUTCOME.fail],
];

let current = null;
let offline = false;
let busy = false;
// The field holds text that the console does not have yet.
let editing = false;
let editTimer = null;
const tiles = new Map();

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mm}:${ss}` : `${mm}:${ss}`;
}

const isOpen = (phase) => phase === PHASE.running || phase === PHASE.ending;
const runHasEnded = () => current !== null && current.phase === PHASE.runEnded;
const openEpisode = (status) => (isOpen(status.phase) ? status.episodes[status.episodes.length - 1] : null);
const nextNumber = (status) => status.episodes.length + 1;
const duration = (episode, status) => (episode.endedAt ?? status.now) - episode.startedAt;

async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const answer = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = errorDetail(answer);
    throw new Error(typeof detail === 'string' ? detail : response.statusText);
  }
  showRefusal(null);
  render(readStatus(answer));
}

// What the console refused last, on the page until the next request it accepts. `null` clears it.
function showRefusal(error) {
  $('refusal').textContent = error === null ? '' : `The console refused: ${error.message}`;
  $('refusal').hidden = error === null;
}

// The last instruction request in the queue. Each waits for the one before it, so the console applies the edits and
// the reset in the order the operator made them, and Start waits for all of them. It never rejects.
let instructionQueue = Promise.resolve();

function postInstruction(text) {
  const sent = instructionQueue.then(() => post(ROUTES.instruction, REQUEST.instruction(text)));
  instructionQueue = sent.catch(() => {});
  return sent;
}

async function act(send) {
  busy = true;
  renderButtons();
  try {
    await send();
  } catch (error) {
    showRefusal(error);
  } finally {
    busy = false;
    renderButtons();
  }
}

// Send the text in the field, if the console does not have it. A refused text stays in the field and stays unsent,
// so the next Start sends it again or refuses.
async function sendEdit() {
  clearTimeout(editTimer);
  editTimer = null;
  if (!editing) return;
  const text = textarea.value;
  await postInstruction(text);
  if (textarea.value === text) editing = false;
}

async function poll() {
  try {
    const response = await fetch(ROUTES.status, { cache: 'no-store' });
    if (!response.ok) throw new Error(response.statusText);
    offline = false;
    render(readStatus(await response.json()));
  } catch (error) {
    if (runHasEnded()) return;
    offline = true;
    $('offline').hidden = false;
    renderButtons();
  } finally {
    // The console stops once the run ends, so the page stops asking.
    if (!runHasEnded()) setTimeout(poll, POLL_MS);
  }
}

// A poll and a request can cross, so an answer older than the one on the page is dropped.
function render(status) {
  $('offline').hidden = !offline;
  if (current !== null && status.generation < current.generation) return;
  current = status;
  $('policy').textContent = status.policy;
  $('host').textContent = status.host;
  $('cameras-live').textContent = `${status.cameras.filter((c) => c.live).length} / ${status.cameras.length} live`;
  renderTiles(status);
  renderStatusRow(status);
  renderInstruction(status);
  renderButtons();
  renderFailure(status);
  renderEpisodes(status);
}

function renderStatusRow(status) {
  const episode = openEpisode(status);
  const view = PHASE_VIEW[status.phase];
  $('status-row').className = `status-row is-${view.css}`;
  $('status').className = `status status--${view.css}`;
  $('status-text').textContent = view.text;
  $('elapsed').hidden = episode === null;
  $('counters').hidden = episode !== null;
  if (episode) {
    $('status-detail').innerHTML = `Episode <b>${episode.number}</b> · ${view.doing}`;
    $('elapsed-value').textContent = clock(duration(episode, status));
    return;
  }
  $('status-detail').innerHTML =
    status.phase === PHASE.runEnded
      ? 'The program releases the rig and exits. You can close this tab.'
      : `Episode <b>${nextNumber(status)}</b> starts on Start`;
  const last = status.episodes[status.episodes.length - 1];
  $('counters').textContent = last
    ? `Last: #${last.number} ${OUTCOME_VIEW[last.outcome].text} · ${clock(duration(last, status))}`
    : 'No episodes in this run';
}

function renderInstruction(status) {
  const episode = openEpisode(status);
  const locked = episode !== null;
  textarea.readOnly = locked || status.phase === PHASE.runEnded;
  if (locked) textarea.value = episode.instruction;
  else if (!editing) textarea.value = status.override ?? status.configured;
  const overridden = locked ? episode.overridden : textarea.value !== status.configured;
  const source = overridden ? 'override' : 'configured';

  $('instr-badge').className = `badge badge--${source}`;
  $('instr-badge').textContent = overridden ? 'Overridden' : 'Configured';
  $('cfg-bar').hidden = !overridden;
  $('cfg-text').textContent = status.configured;
  $('field').className = `field${overridden ? ' is-override' : ''}${locked ? ' is-locked' : ''}`;
  $('field-k').hidden = !overridden;
  $('lock').hidden = !locked;
  $('reset').disabled = locked || !overridden || status.phase === PHASE.runEnded;
  $('reset').classList.toggle('is-live', !locked && overridden);

  let hint = '';
  let aside = 'An override stays until you reset it';
  if (locked) aside = `Unlocks when episode ${episode.number} ends`;
  else if (!overridden) hint = 'From the run configuration. Type here to override it.';
  else if (status.override === textarea.value && status.overrideSince !== null)
    aside = `Override in use since episode ${status.overrideSince}`;
  else aside = 'No episode has used this override yet';
  $('instr-hint').textContent = hint;
  $('instr-aside').textContent = aside;

  const named = `<span class="src-${source}">${source}</span>`;
  $('sends').classList.toggle('is-running', locked);
  $('sends-text').innerHTML = locked
    ? `Episode <b>${episode.number}</b> received the ${named} text at Start.`
    : `Start sends the ${named} text to the policy as episode <b>${nextNumber(status)}</b>.`;
}

function renderButtons() {
  const phase = current ? current.phase : null;
  const idle = !offline && !busy;
  $('start').disabled = !(idle && phase === PHASE.ready);
  for (const [button] of verdictButtons) button.disabled = !(idle && phase === PHASE.running);
  $('end-run').disabled = !idle || phase === null || phase === PHASE.runEnded;
}

function renderFailure(status) {
  const last = status.episodes[status.episodes.length - 1];
  const failed = status.phase === PHASE.ready && last !== undefined && last.outcome === OUTCOME.error;
  $('failure').hidden = !failed;
  if (!failed) return;
  $('failure-head').textContent = `Episode ${last.number} failed. Start runs the next episode.`;
  $('failure-text').textContent = last.error ?? '';
}

function renderEpisodes(status) {
  const episodes = [...status.episodes].reverse();
  const count = (outcome) => status.episodes.filter((e) => e.outcome === outcome).length;
  $('ep-empty').hidden = episodes.length > 0;
  $('ep-list').hidden = episodes.length === 0;
  $('ep-order').hidden = episodes.length === 0;
  $('ep-legend').hidden = episodes.length === 0;
  let sum = '<span>0 episodes</span>';
  if (episodes.length > 0) {
    sum = `<span class="n-pass">${count(OUTCOME.pass)} pass</span>`;
    sum += `<span class="n-fail">${count(OUTCOME.fail)} fail</span>`;
    if (count(OUTCOME.timeout)) sum += `<span>${count(OUTCOME.timeout)} timed out</span>`;
    if (count(OUTCOME.error)) sum += `<span>${count(OUTCOME.error)} error</span>`;
  }
  $('ep-sum').innerHTML = sum;
  const rows = episodes.map((episode) => row(episode, status)).join('');
  if ($('ep-list').innerHTML !== rows) $('ep-list').innerHTML = rows;
}

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

function row(episode, status) {
  const view = episode.outcome === null ? OPEN_VIEW[status.phase] : OUTCOME_VIEW[episode.outcome];
  const classes = ['ep'];
  if (episode.outcome === null) classes.push('is-open');
  if (!episode.overridden) classes.push('is-configured');
  const tag = episode.overridden ? '<span class="tag">override</span>' : '';
  const error = episode.error === null ? '' : ` title="${escapeHtml(episode.error)}"`;
  return (
    `<li class="${classes.join(' ')}"><span class="ep-n">#${episode.number}</span>` +
    `<span class="out out--${view.css}"${error}>${view.label}</span>` +
    `<span class="ep-dur">${clock(duration(episode, status))}</span>` +
    `<span class="ep-text"><span class="t">${escapeHtml(episode.instruction)}</span>${tag}</span></li>`
  );
}

function renderTiles(status) {
  const grid = $('cams');
  const cameras = status.cameras;
  if (tiles.size === 0) {
    grid.style.gridTemplateColumns = `repeat(${Math.max(1, cameras.length)}, minmax(0, 1fr))`;
    if (cameras.length === 0) grid.innerHTML = '<div class="cams-empty">This embodiment has no cameras.</div>';
    for (const camera of cameras) tiles.set(camera.name, makeTile(grid, camera));
  }
  const recording = isOpen(status.phase);
  for (const camera of cameras) {
    const tile = tiles.get(camera.name);
    if (!tile) continue;
    tile.figure.classList.toggle('is-recording', recording);
    tile.state.className = `tile-state${recording ? ' rec' : camera.live ? '' : ' off'}`;
    tile.stateText.textContent = recording ? 'REC' : camera.live ? 'LIVE' : 'NO SIGNAL';
    tile.meta.textContent =
      camera.width === null ? 'no frames' : `${camera.width}×${camera.height} · ${Math.round(camera.fps)} fps`;
  }
}

function makeTile(grid, camera) {
  const figure = document.createElement('figure');
  figure.className = 'tile';
  figure.innerHTML =
    '<video muted autoplay playsinline></video><div class="tile-wait">Waiting for frames</div>' +
    '<span class="tile-label"></span><span class="tile-state"><span class="dot"></span><span></span></span>' +
    '<span class="tile-meta"></span>';
  figure.querySelector('.tile-label').textContent = camera.label;
  grid.appendChild(figure);
  const state = figure.querySelector('.tile-state');
  const wait = figure.querySelector('.tile-wait');
  startStream(figure.querySelector('video'), `${ROUTES.video}/${encodeURIComponent(camera.name)}`, wait);
  return { figure, state, stateText: state.lastElementChild, meta: figure.querySelector('.tile-meta') };
}

function startStream(video, path, wait) {
  if (!('MediaSource' in window)) {
    wait.textContent = 'This browser cannot play the stream';
    return;
  }
  const MAX_BUFFER_SECONDS = 10;
  const STALL_MS = 5000;
  let socket = null;
  let mediaSource = null;
  let sourceBuffer = null;
  let pending = [];
  let objectUrl = null;
  let watchdog = null;
  let reconnectTimer = null;
  let backoff = 500;
  let lastActivity = 0;

  // A new load on reconnect rejects play() with AbortError. The tile shows any other rejection.
  const ensurePlaying = () => {
    if (!video.paused) return;
    video.play().catch((error) => {
      if (error.name === 'AbortError') return;
      wait.hidden = false;
      wait.textContent = `This browser did not play the stream: ${error.message}`;
    });
  };
  video.addEventListener('canplay', ensurePlaying);
  video.addEventListener('playing', () => {
    wait.hidden = true;
  });

  const pump = () => {
    if (!sourceBuffer || sourceBuffer.updating) return;
    const buffered = sourceBuffer.buffered;
    if (buffered.length > 0) {
      // MSE does not start by itself, and playback falls behind after a stall: keep it near the live edge.
      const liveEdge = buffered.end(buffered.length - 1);
      if (liveEdge - video.currentTime > 1.5) video.currentTime = Math.max(buffered.start(0), liveEdge - 0.3);
      const dropTo = video.currentTime - MAX_BUFFER_SECONDS;
      if (dropTo > buffered.start(0) + 1) {
        sourceBuffer.remove(buffered.start(0), dropTo);
        return;
      }
    }
    if (pending.length > 0) {
      sourceBuffer.appendBuffer(pending.shift());
      ensurePlaying();
    }
  };

  // A half-open socket fires no close, so a socket with no fragment for STALL_MS is closed here.
  const scheduleReconnect = () => {
    if (reconnectTimer || runHasEnded()) return;
    if (watchdog) {
      clearInterval(watchdog);
      watchdog = null;
    }
    wait.hidden = false;
    wait.textContent = 'Reconnecting…';
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, backoff);
    backoff = Math.min(backoff * 2, 4000);
  };

  const connect = () => {
    sourceBuffer = null;
    pending = [];
    lastActivity = Date.now();
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${scheme}://${location.host}${path}`);
    socket.binaryType = 'arraybuffer';
    socket.addEventListener('message', (event) => {
      lastActivity = Date.now();
      if (typeof event.data !== 'string') {
        pending.push(new Uint8Array(event.data));
        pump();
        return;
      }
      const codec = `video/mp4; codecs="${event.data}"`;
      if (!MediaSource.isTypeSupported(codec)) {
        wait.textContent = `This browser cannot play ${event.data}`;
        socket.close();
        return;
      }
      backoff = 500;
      mediaSource = new MediaSource();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
      objectUrl = URL.createObjectURL(mediaSource);
      video.src = objectUrl;
      mediaSource.addEventListener('sourceopen', () => {
        sourceBuffer = mediaSource.addSourceBuffer(codec);
        sourceBuffer.mode = 'sequence';
        sourceBuffer.addEventListener('updateend', pump);
        pump();
      });
    });
    socket.addEventListener('close', scheduleReconnect);
    socket.addEventListener('error', () => socket.close());
    watchdog = setInterval(() => {
      if (Date.now() - lastActivity > STALL_MS) socket.close();
    }, 1000);
  };

  connect();
}

textarea.addEventListener('input', () => {
  editing = true;
  clearTimeout(editTimer);
  editTimer = setTimeout(() => sendEdit().catch(showRefusal), EDIT_DELAY_MS);
  if (current) renderInstruction(current);
});

$('reset').addEventListener('click', () => {
  clearTimeout(editTimer);
  editing = false;
  act(() => postInstruction(null));
});

// Start runs only once the console has the text in the field: a refused instruction stops it in READY.
$('start').addEventListener('click', () =>
  act(async () => {
    await instructionQueue;
    await sendEdit();
    await post(ROUTES.start);
  }),
);

for (const [button, verdict] of verdictButtons) {
  button.addEventListener('click', () => act(() => post(ROUTES.end, REQUEST.end(verdict))));
}

$('end-run').addEventListener('click', () => {
  if (window.confirm('End the run? The program releases the rig and exits.')) act(() => post(ROUTES.endRun));
});

poll();
