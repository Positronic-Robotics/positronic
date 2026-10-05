'use strict';

// The page holds no state of its own: it draws each answer of GET /status, and a reloaded tab draws the same.
const POLL_MS = 500;
const EDIT_DELAY_MS = 300;

const ICON = {
  check:
    '<svg class="i" viewBox="0 0 16 16"><path d="M3 8.5l3.2 3.2L13 4.8" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cross:
    '<svg class="i" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round"/></svg>',
};

const OUTCOME_LABEL = {
  pass: `${ICON.check}Pass`,
  fail: `${ICON.cross}Fail`,
  discarded: 'Discarded',
  timeout: 'Timed out',
  error: 'Error',
  running: '<span class="dot"></span>Running',
  ending: 'Ending',
};
const OUTCOME_TEXT = { pass: 'pass', fail: 'fail', discarded: 'discarded', timeout: 'timed out', error: 'error' };

const $ = (id) => document.getElementById(id);
const textarea = $('instruction');
const verdictButtons = [...document.querySelectorAll('[data-verdict]')];

let current = null;
let offline = false;
let busy = false;
// The field holds text that the console does not have yet.
let editing = false;
let editTimer = null;
const tiles = new Map();

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return s >= 3600 ? `${Math.floor(s / 3600)}:${mm}:${ss}` : `${mm}:${ss}`;
}

const openEpisode = (run) => (run.phase === 'ready' ? null : run.episodes[run.episodes.length - 1]);
const nextNumber = (run) => run.episodes.length + 1;
const duration = (episode, run) => (episode.ended_at ?? run.now) - episode.started_at;

async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const answer = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof answer.detail === 'string' ? answer.detail : response.statusText);
  render(answer);
}

async function act(path, body) {
  busy = true;
  renderButtons();
  try {
    await post(path, body);
  } catch (error) {
    window.alert(`The console refused: ${error.message}`);
  } finally {
    busy = false;
    renderButtons();
  }
}

async function sendEdit() {
  clearTimeout(editTimer);
  editTimer = null;
  if (!editing) return;
  const text = textarea.value;
  try {
    await post('/instruction', { override: text });
  } catch (error) {
    window.alert(`The console refused the instruction: ${error.message}`);
  }
  if (textarea.value === text) editing = false;
}

async function poll() {
  try {
    const response = await fetch('/status', { cache: 'no-store' });
    if (!response.ok) throw new Error(response.statusText);
    offline = false;
    render(await response.json());
  } catch (error) {
    offline = true;
    $('offline').hidden = false;
    renderButtons();
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

function render(status) {
  current = status;
  const run = status.run;
  $('offline').hidden = !offline;
  $('policy').textContent = status.policy;
  $('host').textContent = status.host;
  $('cameras-live').textContent = `${status.cameras.filter((c) => c.live).length} / ${status.cameras.length} live`;
  renderTiles(status.cameras, run.phase);
  renderStatusRow(run);
  renderInstruction(run);
  renderButtons();
  renderEpisodes(run);
}

function renderStatusRow(run) {
  const episode = openEpisode(run);
  $('status-row').className = `status-row is-${run.phase}`;
  $('status').className = `status status--${run.phase}`;
  $('status-text').textContent = { ready: 'Ready', running: 'Running', ending: 'Ending' }[run.phase];
  $('elapsed').hidden = episode === null;
  $('counters').hidden = episode !== null;
  if (episode) {
    const doing = run.phase === 'running' ? 'recording' : 'ending';
    $('status-detail').innerHTML = `Episode <b>${episode.number}</b> · ${doing}`;
    $('elapsed-value').textContent = clock(duration(episode, run));
    return;
  }
  $('status-detail').innerHTML = `Episode <b>${nextNumber(run)}</b> starts on Start`;
  const last = run.episodes[run.episodes.length - 1];
  $('counters').textContent = last
    ? `Last: #${last.number} ${OUTCOME_TEXT[last.outcome]} · ${clock(duration(last, run))}`
    : 'No episodes in this run';
}

function renderInstruction(run) {
  const episode = openEpisode(run);
  const locked = episode !== null;
  textarea.readOnly = locked;
  if (locked) textarea.value = episode.instruction;
  else if (!editing) textarea.value = run.override ?? run.configured;
  const overridden = locked ? episode.overridden : textarea.value !== run.configured;
  const source = overridden ? 'override' : 'configured';

  $('instr-badge').className = `badge badge--${overridden ? 'override' : 'configured'}`;
  $('instr-badge').textContent = overridden ? 'Overridden' : 'Configured';
  $('cfg-bar').hidden = !overridden;
  $('cfg-text').textContent = run.configured;
  $('field').className = `field${overridden ? ' is-override' : ''}${locked ? ' is-locked' : ''}`;
  $('field-k').hidden = !overridden;
  $('lock').hidden = !locked;
  $('reset').disabled = locked || !overridden;
  $('reset').classList.toggle('is-live', !locked && overridden);

  let hint = '';
  let aside = 'An override stays until you reset it';
  if (locked) aside = `Unlocks when episode ${episode.number} ends`;
  else if (!overridden) hint = 'From the run configuration. Type here to override it.';
  else if (run.override === textarea.value && run.override_since !== null)
    aside = `Override in use since episode ${run.override_since}`;
  else aside = 'No episode has used this override yet';
  $('instr-hint').textContent = hint;
  $('instr-aside').textContent = aside;

  const named = `<span class="src-${source}">${source}</span>`;
  $('sends').classList.toggle('is-running', locked);
  $('sends-text').innerHTML = locked
    ? `Episode <b>${episode.number}</b> received the ${named} text at Start.`
    : `Start sends the ${named} text to the policy as episode <b>${nextNumber(run)}</b>.`;
}

function renderButtons() {
  const phase = current ? current.run.phase : null;
  const idle = !offline && !busy;
  $('start').disabled = !(idle && phase === 'ready');
  for (const button of verdictButtons) button.disabled = !(idle && phase === 'running');
}

function renderEpisodes(run) {
  const episodes = [...run.episodes].reverse();
  const count = (outcome) => run.episodes.filter((e) => e.outcome === outcome).length;
  $('ep-empty').hidden = episodes.length > 0;
  $('ep-list').hidden = episodes.length === 0;
  $('ep-order').hidden = episodes.length === 0;
  $('ep-legend').hidden = episodes.length === 0;
  let sum = '<span>0 episodes</span>';
  if (episodes.length > 0) {
    sum = `<span class="n-pass">${count('pass')} pass</span><span class="n-fail">${count('fail')} fail</span>`;
    sum += `<span>${count('discarded')} discarded</span>`;
    if (count('timeout')) sum += `<span>${count('timeout')} timed out</span>`;
    if (count('error')) sum += `<span>${count('error')} error</span>`;
  }
  $('ep-sum').innerHTML = sum;
  const rows = episodes.map((episode) => row(episode, run)).join('');
  if ($('ep-list').innerHTML !== rows) $('ep-list').innerHTML = rows;
}

function row(episode, run) {
  const state = episode.outcome ?? run.phase;
  const classes = ['ep'];
  if (episode.outcome === null) classes.push('is-open');
  if (episode.outcome === 'discarded') classes.push('is-discarded');
  if (!episode.overridden) classes.push('is-configured');
  const tag = episode.overridden ? '<span class="tag">override</span>' : '';
  return (
    `<li class="${classes.join(' ')}"><span class="ep-n">#${episode.number}</span>` +
    `<span class="out out--${state}">${OUTCOME_LABEL[state]}</span>` +
    `<span class="ep-dur">${clock(duration(episode, run))}</span>` +
    `<span class="ep-text"><span class="t">${escapeHtml(episode.instruction)}</span>${tag}</span></li>`
  );
}

function renderTiles(cameras, phase) {
  const grid = $('cams');
  if (tiles.size === 0) {
    grid.style.gridTemplateColumns = `repeat(${Math.max(1, cameras.length)}, minmax(0, 1fr))`;
    if (cameras.length === 0) grid.innerHTML = '<div class="cams-empty">This embodiment has no cameras.</div>';
    for (const camera of cameras) tiles.set(camera.name, makeTile(grid, camera));
  }
  const recording = phase !== 'ready';
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
  startStream(figure.querySelector('video'), `/video/${encodeURIComponent(camera.name)}`, wait);
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

  const ensurePlaying = () => {
    if (video.paused) video.play().catch(() => {});
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
    if (reconnectTimer) return;
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
  editTimer = setTimeout(sendEdit, EDIT_DELAY_MS);
  if (current) renderInstruction(current.run);
});

$('reset').addEventListener('click', () => {
  clearTimeout(editTimer);
  editing = false;
  act('/instruction', { override: null });
});

$('start').addEventListener('click', async () => {
  await sendEdit();
  await act('/episode/start');
});

for (const button of verdictButtons) {
  button.addEventListener('click', () => {
    const verdict = button.dataset.verdict;
    const episode = current && openEpisode(current.run);
    if (verdict === 'discarded' && episode) {
      const question = `Discard episode ${episode.number}? The recording stays, marked as discarded.`;
      if (!window.confirm(question)) return;
    }
    act('/episode/end', { verdict });
  });
}

poll();
