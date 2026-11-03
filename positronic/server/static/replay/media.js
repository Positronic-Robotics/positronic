// The player of one episode: its cameras on one clock, its phone video, or its 3D replay in the rerun viewer.
//
// The player loads only the URLs a row names. It loads a recording only in `full` mode, on a wide screen.
import { element } from './cells.js';

// A follower camera further than this from the lead camera's time is moved back into step.
const DRIFT_S = 0.1;
const SPEEDS = [0.25, 0.5, 1, 2, 4];
const PLAY_LABEL = 'Play';
const PAUSE_LABEL = 'Pause';
// The wait before the next request for a file that is building, where the server names none.
const DEFAULT_RETRY_S = 5;
// The quiet time after a visitor's last seek on the native seek bar, before it reports the seek.
const SEEK_SETTLE_MS = 300;
// The rerun release whose viewer sits beside the element, under `rerun/<release>/`.
export const RERUN_VERSION = '0.30.0';
const VIEWER_URL = new URL(`../rerun/${RERUN_VERSION}/embedded.html`, import.meta.url).href;

export const MediaState = Object.freeze({
  PENDING: 'pending',
  READY: 'ready',
  BUILDING: 'building',
  GONE: 'gone',
  FAILED: 'failed',
});

const NOTES = {
  none: 'The replay of this episode is not available.',
  replayBuilding: 'Preparing the 3D replay…',
  replayGone: 'The 3D replay of this episode is gone.',
  replayFailed: 'The 3D replay did not load.',
};

// The note for each state of a video that is not there, for all the videos of an episode and for some of them.
const VIDEO_NOTES = [
  [MediaState.GONE, 'The video of this episode is gone.', (names) => `The video of ${names} is gone.`],
  [MediaState.FAILED, 'The video did not load.', (names) => `The video of ${names} did not load.`],
  [
    MediaState.BUILDING,
    'The video is building. It plays when it is ready.',
    (names) => `The video of ${names} is building. It plays when it is ready.`,
  ],
];

/** The note for a set of `[name, state]` videos, from the states of all of them; null where none is missing. */
function cameraNote(videos) {
  const notes = VIDEO_NOTES.map(([state, all, some]) => {
    const named = videos.filter(([, each]) => each === state).map(([name]) => name);
    if (!named.length) return null;
    return named.length === videos.length ? all : some(named.join(' and '));
  }).filter(Boolean);
  return notes.length ? notes.join(' ') : null;
}

// The playback state of a new episode: paused, at its start, at the normal rate.
const START = Object.freeze({ playing: false, at: 0, rate: 1 });

/** The length of the episode: the lead's own length once it knows it, else the length the row names, else the
 * largest length a camera names. */
function episodeLength(episode, lead) {
  if (lead && Number.isFinite(lead.duration)) return lead.duration;
  return episode.row.durationS ?? Math.max(0, ...episode.cameras.map((camera) => camera.durationS));
}

/** The state of `url`, from one request for its first byte, and the wait a building file asks for.
 * Null where `signal` aborts the request. */
async function probe(url, signal) {
  let answer;
  try {
    answer = await fetch(url, { headers: { Range: 'bytes=0-0' }, signal });
  } catch (error) {
    // The player aborts the requests of an episode it no longer shows.
    if (error.name === 'AbortError') return null;
    console.error(`The request for ${url} failed: ${error.message}`);
    return { state: MediaState.FAILED };
  }
  answer.body?.cancel();
  if (answer.status === 202) {
    return { state: MediaState.BUILDING, retryS: Number(answer.headers.get('Retry-After')) || DEFAULT_RETRY_S };
  }
  if (answer.status === 404 || answer.status === 410) return { state: MediaState.GONE };
  return { state: answer.ok ? MediaState.READY : MediaState.FAILED };
}

function clock(value) {
  const whole = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${(whole - minutes * 60).toFixed(1).padStart(4, '0')}`;
}

// The browser refuses to play a video with no visitor gesture, a new `src` interrupts a play that has not
// started, and a file that is gone or building has no source; its note says which.
const EXPECTED_PLAY_REFUSALS = new Set(['NotAllowedError', 'AbortError', 'NotSupportedError']);

// The element's own calls on each video, counted by the event each one fires, so a listener on a video with
// native controls tells them from a visitor's use of those controls. A call that fires no event counts none.
const ownCalls = new WeakMap();

function expectOwn(video, event) {
  const calls = ownCalls.get(video) ?? new Map();
  calls.set(event, (calls.get(event) ?? 0) + 1);
  ownCalls.set(video, calls);
}

/** Whether `event` on `video` is the element's own; it takes the count of that call. */
function takeOwn(video, event) {
  const calls = ownCalls.get(video);
  if (!calls?.get(event)) return false;
  calls.set(event, calls.get(event) - 1);
  return true;
}

/** Plays a video and names it in the console where it does not play. */
export function start(video) {
  if (!video.paused) return;
  expectOwn(video, 'play');
  video.play().catch((error) => {
    const log = EXPECTED_PLAY_REFUSALS.has(error.name) ? console.warn : console.error;
    log(`${video.dataset.name} did not play: ${error.message}`);
  });
  // A play the browser refuses at once leaves the video paused and fires no `play` event.
  if (video.paused) takeOwn(video, 'play');
}

export function pause(video) {
  if (video.paused) return;
  expectOwn(video, 'pause');
  video.pause();
}

// A seek before the video knows its duration fires no `seeking` until the load seeks to a later start.
function seek(video, seconds) {
  if (video.readyState > HTMLMediaElement.HAVE_NOTHING || seconds > 0) expectOwn(video, 'seeking');
  video.currentTime = seconds;
}

function setRate(video, rate) {
  if (video.playbackRate === rate) return;
  expectOwn(video, 'ratechange');
  video.playbackRate = rate;
}

export class EpisodePlayer {
  #root;
  #stage;
  #note;
  #notes = { replay: null, video: null };
  #summary;
  #frame = null;
  // The element whose focus the viewer takes when it next shows a recording: the player root where the viewer held
  // the focus when it was hidden, or the element that held it when a host asked for it while it was hidden. A focus
  // that moved elsewhere in the meantime stays where the visitor put it.
  #focusForViewer = null;
  #full;
  #narrow;
  #act;
  #frameKeys;
  #cameraName;
  #turn = 0;
  // The playback state of the episode on show: set when a row opens, updated by the visitor and the element, and
  // kept while its media elements change, also while the 3D viewer, whose time is not readable, owns the episode.
  #state = { ...START };
  // Stops the listeners of the lead camera that update `#state`, before a lead change or a clear.
  #detachMedia = () => {};
  // The signal of every listener on the videos on show that updates `#state` or reports a visitor's action. A
  // clear aborts it before it changes the videos, so their own events report nothing.
  #media = new AbortController();
  // The waits and the requests of the episode on show; `stop` and the next `show` end them.
  #timers = new Set();
  #requests = new AbortController();
  lead = null;

  /** `act(action, detail)` reports a visitor's use of a control, and `frameKeys` takes the viewer's key presses. */
  constructor(root, { full, narrow, act, frameKeys, cameraName }) {
    this.#root = root;
    this.#full = full;
    this.#narrow = narrow;
    this.#act = act;
    this.#frameKeys = frameKeys;
    this.#cameraName = cameraName;
    this.#stage = element('div', { part: 'stage' });
    this.#note = element('p', { part: 'note', hidden: '' });
    this.#summary = element('p', { part: 'summary' });
    root.replaceChildren(this.#stage, this.#note, this.#summary);
    // The root takes the focus from a viewer frame the element hides, so the element's keys still work.
    root.tabIndex = -1;
  }

  /** Shows `row` in the `playback` state: whether it plays, from which time, at which rate. A row opened later stops
   * the waits of this one. */
  show(row, summary, playback = {}) {
    this.#state = { ...START, ...playback };
    this.#endWaits();
    const turn = ++this.#turn;
    const wanted = () => turn === this.#turn;
    this.#summary.textContent = summary;
    this.#clearVideos();
    // The recording of the row before goes at once, so it never shows under this row's title.
    this.#hideFrame();
    this.#setNote('replay', null);
    const startVideo = this.#videoStarter(row);
    if (this.#full && row.rrd && !this.#narrow.matches) {
      this.#playRecording(row, startVideo, wanted);
      return;
    }
    if (startVideo) startVideo();
    else this.#setNote('replay', NOTES.none);
  }

  /** The playback state of the episode on show, which a new player for the same row takes whole. */
  playback() {
    return { ...this.#state };
  }

  /** Stops every video and empties the viewer. */
  stop() {
    this.#endWaits();
    this.#turn += 1;
    this.#clearVideos();
    this.#hideFrame();
    // A stopped player shows no recording, so no later one takes the focus for it.
    this.#focusForViewer = null;
    this.lead = null;
  }

  #endWaits() {
    for (const timer of this.#timers) clearTimeout(timer);
    this.#timers.clear();
    this.#requests.abort();
    this.#requests = new AbortController();
  }

  /** A wait of `seconds` that `#endWaits` cancels, so it never resolves. */
  #pause(seconds) {
    return new Promise((done) => {
      const timer = setTimeout(() => {
        this.#timers.delete(timer);
        done();
      }, 1000 * seconds);
      this.#timers.add(timer);
    });
  }

  /** The final state of `url`: it asks again while the server builds the file. Null once `wanted()` turns false. */
  async #settled(url, { building, wanted }) {
    for (;;) {
      const answer = await probe(url, this.#requests.signal);
      if (answer === null || !wanted()) return null;
      if (answer.state !== MediaState.BUILDING) return answer.state;
      building();
      await this.#pause(answer.retryS);
      if (!wanted()) return null;
    }
  }

  /** Gives the viewer the keyboard, at once where it shows, else when it next shows a recording. */
  focusViewer() {
    if (!this.#frame || this.#frame.hidden) {
      this.#focusForViewer = this.#focused();
      return;
    }
    this.#frame.focus({ preventScroll: true });
    this.#frame.contentDocument?.getElementById('the_canvas_id')?.focus({ preventScroll: true });
  }

  /** The function that starts the video `row` plays on this screen; null for a row with no video. */
  #videoStarter(row) {
    if (row.phone && (this.#narrow.matches || !row.cameras.length)) return () => this.#playPhone(row);
    if (row.cameras.length) return () => this.#playCameras(row);
    return null;
  }

  async #playRecording(row, startVideo, wanted) {
    // The video starts once: a later 202 of the recording keeps it, where a camera is not ready yet either.
    let videoStarted = false;
    const fallBackToVideo = (note) => {
      this.#hideFrame();
      this.#setNote('replay', note);
      if (videoStarted || !startVideo) return;
      videoStarted = true;
      startVideo();
    };
    // The row's note shows while the element asks for the recording.
    this.#setNote('replay', NOTES.replayBuilding);
    const state = await this.#settled(row.rrd, { building: () => fallBackToVideo(NOTES.replayBuilding), wanted });
    if (state === null) return;
    if (state !== MediaState.READY) {
      fallBackToVideo(state === MediaState.GONE ? NOTES.replayGone : NOTES.replayFailed);
      return;
    }
    this.#clearVideos();
    this.#setNote('replay', null);
    const url = new URL(VIEWER_URL);
    url.searchParams.set('url', row.rrd);
    url.searchParams.set('hide_welcome_screen', '');
    this.#showFrame(url.href);
  }

  /** Sets the note about the 3D replay or about the video; the note line shows both. */
  #setNote(kind, text) {
    this.#notes[kind] = text;
    const shown = [this.#notes.replay, this.#notes.video].filter(Boolean).join(' ');
    this.#note.hidden = !shown;
    this.#note.textContent = shown;
  }

  // A new `src` would add a browser history entry, so the frame changes its page in place.
  #showFrame(url) {
    if (!this.#frame) {
      this.#frame = element('iframe', { part: 'viewer', title: '3D replay', allow: 'fullscreen' });
      this.#frame.addEventListener('load', () => this.#watchFrame());
      this.#stage.before(this.#frame);
    }
    this.#frame.hidden = false;
    this.#frame.contentWindow.location.replace(url);
    const holder = this.#focusForViewer;
    this.#focusForViewer = null;
    if (holder && holder === this.#focused()) this.focusViewer();
  }

  /** The element with the focus, in the element's own tree where the focus is there. */
  #focused() {
    return this.#root.getRootNode().activeElement ?? this.#root.ownerDocument.activeElement;
  }

  #hideFrame() {
    if (!this.#frame || this.#frame.hidden) return;
    if (this.#focused() === this.#frame) {
      this.#root.focus({ preventScroll: true });
      this.#focusForViewer = this.#root;
    }
    this.#frame.hidden = true;
    this.#frame.contentWindow.location.replace('about:blank');
  }

  // The viewer's page loads anew for each episode: give it the element's keys, and the keyboard once it starts.
  // A viewer on another origin than the page keeps its own keys.
  #watchFrame() {
    const view = this.#frame.contentWindow;
    let blank;
    try {
      blank = view.location.href === 'about:blank';
    } catch (error) {
      if (error.name !== 'SecurityError') throw error;
      // A cross-origin host refuses this step on every open.
      console.warn(`The viewer on another origin keeps its own keys: ${error.message}`);
      return;
    }
    if (blank) return;
    view.addEventListener('keydown', this.#frameKeys, true);
    const canvas = this.#frame.contentDocument.getElementById('the_canvas_id');
    if (!canvas) return;
    // rerun shows its canvas and makes it focusable when it starts, after the page loads.
    new view.MutationObserver((_, observer) => {
      if (!canvas.hasAttribute('tabindex') || canvas.classList.contains('hidden')) return;
      observer.disconnect();
      if (this.#focused() === this.#frame) this.focusViewer();
    }).observe(canvas, { attributes: true });
  }

  // The videos go with their retries and requests, also where the 3D replay takes their place.
  #clearVideos() {
    this.#endWaits();
    this.#detachMedia();
    this.#detachMedia = () => {};
    this.#media.abort();
    this.#media = new AbortController();
    for (const video of this.#stage.querySelectorAll('video')) {
      pause(video);
      video.removeAttribute('src');
      video.load();
    }
    this.#stage.replaceChildren();
    this.#setNote('video', null);
    this.lead = null;
  }

  // A video that fails is asked for once more: a 202 means it is building, and it loads again after the wait.
  // `onState` takes each state of the video: building, ready, gone or failed.
  #watch(video, url, onState) {
    video.addEventListener('error', async () => {
      if (video.getAttribute('src') !== url) return;
      const turn = this.#turn;
      const answer = await probe(url, this.#requests.signal);
      if (answer === null || turn !== this.#turn) return;
      if (answer.state !== MediaState.BUILDING) {
        onState(answer.state === MediaState.READY ? MediaState.FAILED : answer.state);
        return;
      }
      onState(MediaState.BUILDING);
      await this.#pause(answer.retryS);
      if (turn !== this.#turn) return;
      video.src = url;
    });
    video.addEventListener('loadedmetadata', () => onState(MediaState.READY));
  }

  /** Moves `video` to `at` seconds once it knows its duration; a shorter video rests on its last frame. */
  static #startAt(video, at) {
    if (!at) return;
    video.addEventListener('loadedmetadata', () => seek(video, Math.min(at, video.duration)), { once: true });
  }

  #playPhone(row) {
    const state = this.#state;
    const video = element('video', { part: 'phone', src: row.phone, controls: '', playsinline: '', muted: '' });
    video.muted = true;
    video.dataset.name = 'The phone video';
    EpisodePlayer.#startAt(video, state.at);
    // A load sets the rate back to the default, so the video takes it once it knows its length.
    video.addEventListener('loadedmetadata', () => setRate(video, state.rate), { once: true });
    const { signal } = this.#media;
    video.addEventListener('play', () => (state.playing = true), { signal });
    // A failed load pauses the video with no choice of the visitor's; the end of the video stops the episode.
    video.addEventListener('pause', () => video.error || (state.playing = false), { signal });
    video.addEventListener('timeupdate', () => video.error || (state.at = video.currentTime), { signal });
    video.addEventListener('ratechange', () => video.readyState && (state.rate = video.playbackRate), { signal });
    this.#reportNativeControls(row, video, signal);
    this.#watch(video, row.phone, (each) => {
      this.#setNote('video', cameraNote([[null, each]]));
      // A row that opened playing plays once its building video is there.
      if (each === MediaState.READY && state.playing) start(video);
    });
    this.#stage.replaceChildren(video);
    this.lead = video;
    if (state.playing) start(video);
  }

  // The browser's own controls fire only media events. An event the element did not cause is a visitor's use; a
  // drag of the seek bar fires many seeks, so a seek reports once it has settled.
  #reportNativeControls(row, video, signal) {
    const report = (action, detail) => this.#act(action, { row: row.id, ...detail, via: 'Native' });
    // A media event that a script dispatches is no visitor's, and it takes no count of an own call.
    const listen = (type, handle) => video.addEventListener(type, (event) => event.isTrusted && handle(), { signal });
    listen('play', () => takeOwn(video, 'play') || report('play', { playing: true }));
    // A video ends, or fails to load, with a `pause` that no visitor made.
    listen('pause', () => takeOwn(video, 'pause') || video.ended || video.error || report('play', { playing: false }));
    listen('ratechange', () => takeOwn(video, 'ratechange') || report('speed', { value: String(video.playbackRate) }));
    let settling = null;
    listen('seeking', () => {
      if (takeOwn(video, 'seeking')) return;
      clearTimeout(settling);
      this.#timers.delete(settling);
      settling = setTimeout(() => {
        this.#timers.delete(settling);
        report('seek', {});
      }, SEEK_SETTLE_MS);
      this.#timers.add(settling);
    });
  }

  /** Plays or pauses `lead`, and reports it as the visitor's use of the control `via`; null reports nothing. */
  #toggle(row, lead, via) {
    if (via) this.#act('play', { row: row.id, playing: lead.paused, via });
    if (lead.paused) start(lead);
    else pause(lead);
  }

  // Each camera holds its own state, and `#settle` derives from all of them which camera leads and the note.
  #playCameras(row) {
    const cameras = row.cameras.map((camera) => {
      const name = camera.label ?? this.#cameraName(camera.camera);
      const video = element('video', {
        part: 'camera',
        src: camera.src,
        preload: 'metadata',
        playsinline: '',
        muted: '',
        width: camera.width,
        height: camera.height,
        'data-camera': camera.camera,
      });
      video.muted = true;
      video.dataset.name = name;
      const caption = element('figcaption', { part: 'caption' }, [name]);
      const figure = element('figure', { part: 'figure', 'data-state': MediaState.PENDING }, [video, caption]);
      return { name, video, caption, figure, durationS: camera.durationS, state: MediaState.PENDING };
    });
    // `setLength` sets the end of the lead's seek bar, while a lead has controls.
    const episode = { row, cameras, setLength: null };
    for (const camera of cameras) {
      camera.video.addEventListener(
        'click',
        (event) => this.lead && this.#toggle(row, this.lead, event.isTrusted ? 'Video' : null),
        { signal: this.#media.signal },
      );
      this.#watch(camera.video, camera.video.getAttribute('src'), (state) => {
        camera.state = state;
        this.#settle(episode);
      });
    }
    this.#stage.replaceChildren(
      element(
        'div',
        { part: 'cameras' },
        cameras.map((camera) => camera.figure),
      ),
    );
    this.#settle(episode);
  }

  /** The longest ready camera: by the length of its video once it knows it, else by the length the row names, and
   * on a tie by the named length, then the first. Null while no camera is ready. */
  static #longestReady(cameras) {
    const length = (camera) => (Number.isFinite(camera.video.duration) ? camera.video.duration : camera.durationS);
    const ready = cameras.filter((camera) => camera.state === MediaState.READY);
    return ready.reduce(
      (best, camera) =>
        length(camera) > length(best) || (length(camera) === length(best) && camera.durationS > best.durationS)
          ? camera
          : best,
      ready[0] ?? null,
    );
  }

  /** Derives from the state of every camera the lead (the preferred ready camera, none while none is ready) and the
   * note, and hands the controls to a new lead at the visitor's play state and time. */
  #settle(episode) {
    for (const camera of episode.cameras) {
      camera.figure.dataset.state = camera.state;
      const shown = camera.state === MediaState.PENDING || camera.state === MediaState.READY;
      camera.caption.textContent = shown ? camera.name : `${camera.name} · ${camera.state}`;
    }
    this.#setNote('video', cameraNote(episode.cameras.map((camera) => [camera.name, camera.state])));
    const lead = EpisodePlayer.#longestReady(episode.cameras)?.video ?? null;
    if (lead !== this.lead) this.#lead(episode, lead);
    episode.setLength?.(episodeLength(episode, lead));
    for (const camera of episode.cameras) {
      if (camera.state === MediaState.READY) setRate(camera.video, this.#state.rate);
    }
    // A camera that is ready under a lead it has not joined takes the lead's time and play state.
    for (const camera of episode.cameras) {
      if (camera.state !== MediaState.READY || camera.video === lead || camera.joined === lead) continue;
      camera.joined = lead;
      if (!lead) continue;
      seek(camera.video, Math.min(lead.currentTime, camera.video.duration));
      if (lead.paused) pause(camera.video);
      else start(camera.video);
    }
  }

  /** Hands the controls to `lead`, at the visitor's play state and time; none where `lead` is null. */
  #lead(episode, lead) {
    this.#detachMedia();
    this.#detachMedia = () => {};
    episode.setLength = null;
    this.lead = lead;
    this.#stage.querySelector('[part~="controls"]')?.remove();
    if (!lead) return;
    const listening = new AbortController();
    const { signal } = listening;
    const state = this.#state;
    lead.addEventListener('play', () => (state.playing = true), { signal });
    lead.addEventListener('pause', () => lead.error || (state.playing = false), { signal });
    lead.addEventListener('timeupdate', () => lead.error || (state.at = lead.currentTime), { signal });
    const followers = episode.cameras.map((camera) => camera.video).filter((video) => video !== lead);
    follow(lead, followers, signal);
    const speed = {
      rate: state.rate,
      choose: (rate) => {
        state.rate = rate;
        this.#settle(episode);
      },
    };
    const controls = this.#controls(episode.row, lead, episodeLength(episode, lead), speed, signal);
    episode.setLength = controls.setLength;
    this.#stage.append(controls.element);
    this.#detachMedia = () => listening.abort();
    if (Math.abs(lead.currentTime - state.at) > DRIFT_S) seek(lead, Math.min(state.at, lead.duration));
    if (state.playing) start(lead);
  }

  // One control set for the episode, which drives `lead` until `signal` aborts, with its element and `setLength`,
  // which sets the end of its seek bar. The speed control shows `speed.rate`, and `speed.choose` sets the rate.
  #controls(row, lead, durationS, speed, signal) {
    const play = element('button', { type: 'button', part: 'play', 'aria-label': PLAY_LABEL }, ['▶']);
    const slider = element('input', {
      type: 'range',
      part: 'seek',
      min: '0',
      max: String(durationS ?? 0),
      step: '0.01',
      value: '0',
      'aria-label': 'Position',
    });
    const time = element('span', { part: 'clock' }, [`${clock(0)} / ${clock(durationS)}`]);
    const rates = element(
      'select',
      { part: 'speed', 'aria-label': 'Speed' },
      SPEEDS.map((rate) =>
        element('option', { value: String(rate), selected: rate === speed.rate ? '' : null }, [`${rate}×`]),
      ),
    );
    let total = durationS;
    const showState = () => {
      play.textContent = lead.paused ? '▶' : '❚❚';
      play.setAttribute('aria-label', lead.paused ? PLAY_LABEL : PAUSE_LABEL);
    };
    const showTime = () => {
      slider.value = String(lead.currentTime);
      time.textContent = `${clock(lead.currentTime)} / ${clock(total)}`;
    };
    play.addEventListener('click', (event) => this.#toggle(row, lead, event.isTrusted ? 'Button' : null));
    slider.addEventListener('input', () => {
      seek(lead, Number(slider.value));
    });
    slider.addEventListener('change', (event) => event.isTrusted && this.#act('seek', { row: row.id }));
    rates.addEventListener('change', (event) => {
      speed.choose(Number(rates.value));
      if (event.isTrusted) this.#act('speed', { row: row.id, value: rates.value });
    });
    for (const type of ['play', 'pause', 'ended']) lead.addEventListener(type, showState, { signal });
    for (const type of ['timeupdate', 'seeked']) lead.addEventListener(type, showTime, { signal });
    showState();
    showTime();
    const setLength = (seconds) => {
      if (seconds === total) return;
      total = seconds;
      slider.max = String(total);
      showTime();
    };
    return { element: element('div', { part: 'controls' }, [play, slider, time, rates]), setLength };
  }
}

// Every video of an episode starts on one clock, so each other video plays, pauses and seeks with the lead
// until its own end.
function follow(lead, followers, signal) {
  const align = () => {
    for (const video of followers) {
      if (Number.isNaN(video.duration) || video.error) continue;
      // Past its own end, a shorter camera rests on its last frame.
      const wanted = Math.min(lead.currentTime, video.duration);
      if (Math.abs(video.currentTime - wanted) > DRIFT_S) seek(video, wanted);
      if (!lead.paused && video.paused && wanted < video.duration) start(video);
    }
  };
  for (const type of ['play', 'seeking', 'timeupdate']) lead.addEventListener(type, align, { signal });
  const rest = () => {
    // A paused episode shows one moment on every camera, so each follower takes the lead's time exactly.
    for (const video of followers) {
      pause(video);
      if (!Number.isNaN(video.duration)) seek(video, Math.min(lead.currentTime, video.duration));
    }
  };
  lead.addEventListener('pause', rest, { signal });
}
