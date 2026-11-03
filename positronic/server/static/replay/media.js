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

export const MediaState = Object.freeze({ READY: 'ready', BUILDING: 'building', GONE: 'gone', FAILED: 'failed' });

const NOTES = {
  none: 'The replay of this episode is not available.',
  videoBuilding: 'The video is building. It plays when it is ready.',
  videoGone: 'The video of this episode is gone.',
  videoFailed: 'The video did not load.',
  replayBuilding: 'Preparing the 3D replay…',
  replayGone: 'The 3D replay of this episode is gone.',
  replayFailed: 'The 3D replay did not load.',
};

/** The state of `url`, from one request for its first byte, and the wait a building file asks for. */
export async function probe(url) {
  let answer;
  try {
    answer = await fetch(url, { headers: { Range: 'bytes=0-0' } });
  } catch (error) {
    console.warn(`${url} did not answer: ${error.message}`);
    return { state: MediaState.FAILED };
  }
  answer.body?.cancel();
  if (answer.status === 202) {
    return { state: MediaState.BUILDING, retryS: Number(answer.headers.get('Retry-After')) || DEFAULT_RETRY_S };
  }
  if (answer.status === 404 || answer.status === 410) return { state: MediaState.GONE };
  return { state: answer.ok ? MediaState.READY : MediaState.FAILED };
}

const pause = (seconds) => new Promise((done) => setTimeout(done, 1000 * seconds));

/** The final state of `url`: it asks again while the server builds the file. Null once `wanted()` turns false. */
async function settled(url, { building, wanted }) {
  for (;;) {
    const answer = await probe(url);
    if (!wanted()) return null;
    if (answer.state !== MediaState.BUILDING) return answer.state;
    building();
    await pause(answer.retryS);
    if (!wanted()) return null;
  }
}

function clock(value) {
  const whole = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(whole / 60);
  return `${minutes}:${(whole - minutes * 60).toFixed(1).padStart(4, '0')}`;
}

/** Plays a video and names it in the console where the browser refuses. */
function start(video) {
  video.play().catch((error) => console.warn(`${video.dataset.name} did not play: ${error.message}`));
}

export class EpisodePlayer {
  #root;
  #stage;
  #note;
  #notes = { replay: null, video: null };
  #summary;
  #frame = null;
  #viewerUrl;
  #full;
  #narrow;
  #act;
  #frameKeys;
  #cameraName;
  #turn = 0;
  lead = null;

  /** `act(action, detail)` reports a visitor's use of a control, and `frameKeys` takes the viewer's key presses. */
  constructor(root, { viewerUrl, full, narrow, act, frameKeys, cameraName }) {
    this.#root = root;
    this.#viewerUrl = viewerUrl;
    this.#full = full;
    this.#narrow = narrow;
    this.#act = act;
    this.#frameKeys = frameKeys;
    this.#cameraName = cameraName;
    this.#stage = element('div', { part: 'stage' });
    this.#note = element('p', { part: 'note', hidden: '' });
    this.#summary = element('p', { part: 'summary' });
    root.replaceChildren(this.#stage, this.#note, this.#summary);
  }

  /** Shows `row`, and plays it where `autoplay`. A row opened later stops the waits of this one. */
  show(row, summary, { autoplay = false } = {}) {
    const turn = ++this.#turn;
    const wanted = () => turn === this.#turn;
    this.#summary.textContent = summary;
    this.#clearVideos();
    this.#setNote('replay', null);
    const video = this.#videoOf(row);
    if (this.#full && row.rrd && !this.#narrow.matches) {
      this.#playRecording(row, video, autoplay, wanted);
      return;
    }
    this.#hideFrame();
    if (video) video(autoplay);
    else this.#setNote('replay', NOTES.none);
  }

  /** Stops every video and empties the viewer. */
  stop() {
    this.#turn += 1;
    this.#clearVideos();
    this.#hideFrame();
    this.lead = null;
  }

  focusViewer() {
    if (!this.#frame || this.#frame.hidden) return;
    this.#frame.focus({ preventScroll: true });
    this.#frame.contentDocument?.getElementById('the_canvas_id')?.focus({ preventScroll: true });
  }

  /** The video `row` plays on this screen, as a function of `autoplay`; null for a row with no video. */
  #videoOf(row) {
    if (row.phone && (this.#narrow.matches || !row.cameras.length)) return (autoplay) => this.#playPhone(row, autoplay);
    if (row.cameras.length) return (autoplay) => this.#playCameras(row, autoplay);
    return null;
  }

  async #playRecording(row, video, autoplay, wanted) {
    const playVideo = (note) => {
      this.#hideFrame();
      this.#setNote('replay', note);
      if (!this.lead && video) video(autoplay);
    };
    const state = await settled(row.rrd, { building: () => playVideo(NOTES.replayBuilding), wanted });
    if (state === null) return;
    if (state !== MediaState.READY) {
      playVideo(state === MediaState.GONE ? NOTES.replayGone : NOTES.replayFailed);
      return;
    }
    this.#clearVideos();
    this.#setNote('replay', null);
    const url = new URL(this.#viewerUrl);
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
  }

  #hideFrame() {
    if (!this.#frame || this.#frame.hidden) return;
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
      if (error.name === 'SecurityError') return;
      throw error;
    }
    if (blank) return;
    view.addEventListener('keydown', this.#frameKeys, true);
    const canvas = this.#frame.contentDocument.getElementById('the_canvas_id');
    if (!canvas) return;
    // rerun shows its canvas and makes it focusable when it starts, after the page loads.
    new view.MutationObserver((_, observer) => {
      if (!canvas.hasAttribute('tabindex') || canvas.classList.contains('hidden')) return;
      observer.disconnect();
      if (this.#root.getRootNode().activeElement === this.#frame) this.focusViewer();
    }).observe(canvas, { attributes: true });
  }

  #clearVideos() {
    for (const video of this.#stage.querySelectorAll('video')) {
      video.pause();
      video.removeAttribute('src');
      video.load();
    }
    this.#stage.replaceChildren();
    this.#setNote('video', null);
    this.lead = null;
  }

  // A video that fails is asked for once more: a 202 means it is building, and it loads again after the wait.
  #watch(video, url, onNote) {
    video.addEventListener('error', async () => {
      if (video.getAttribute('src') !== url) return;
      const turn = this.#turn;
      const answer = await probe(url);
      if (turn !== this.#turn) return;
      if (answer.state === MediaState.BUILDING) {
        onNote(NOTES.videoBuilding, 'building');
        await pause(answer.retryS);
        if (turn !== this.#turn) return;
        video.src = url;
        return;
      }
      onNote(answer.state === MediaState.GONE ? NOTES.videoGone : NOTES.videoFailed, answer.state);
    });
    video.addEventListener('loadeddata', () => onNote(null, MediaState.READY));
  }

  #playPhone(row, autoplay) {
    const video = element('video', { part: 'phone', src: row.phone, controls: '', playsinline: '', muted: '' });
    video.muted = true;
    video.dataset.name = 'The phone video';
    this.#watch(video, row.phone, (note) => this.#setNote('video', note));
    this.#stage.replaceChildren(video);
    this.lead = video;
    if (autoplay) start(video);
  }

  #playCameras(row, autoplay) {
    const lead = row.cameras.reduce(
      (best, camera, at) => (camera.durationS > row.cameras[best].durationS ? at : best),
      0,
    );
    const figures = row.cameras.map((camera) => {
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
      const figure = element('figure', { part: 'figure' }, [video, caption]);
      this.#watch(video, camera.src, (note, state) => {
        figure.dataset.state = state;
        caption.textContent = note ? `${name} · ${state}` : name;
        if (note) this.#setNote('video', note);
        else if ([...this.#stage.querySelectorAll('figure')].every((each) => each.dataset.state === MediaState.READY)) {
          this.#setNote('video', null);
        }
      });
      return figure;
    });
    const videos = figures.map((figure) => figure.querySelector('video'));
    const durationS = row.durationS ?? row.cameras[lead].durationS;
    const controls = this.#controls(row, videos[lead], durationS);
    for (const video of videos) video.addEventListener('click', () => controls.querySelector('[part~="play"]').click());
    this.#stage.replaceChildren(element('div', { part: 'cameras' }, figures), controls);
    follow(
      videos[lead],
      videos.filter((_, at) => at !== lead),
    );
    this.lead = videos[lead];
    if (autoplay) start(videos[lead]);
  }

  // One control set for the episode. It drives the lead, the longest video, and `follow` moves the others.
  #controls(row, lead, durationS) {
    const play = element('button', { type: 'button', part: 'play', 'aria-label': PLAY_LABEL }, ['▶']);
    const seek = element('input', {
      type: 'range',
      part: 'seek',
      min: '0',
      max: String(durationS),
      step: '0.01',
      value: '0',
      'aria-label': 'Position',
    });
    const time = element('span', { part: 'clock' }, [`${clock(0)} / ${clock(durationS)}`]);
    const speed = element(
      'select',
      { part: 'speed', 'aria-label': 'Speed' },
      SPEEDS.map((rate) => element('option', { value: String(rate), selected: rate === 1 ? '' : null }, [`${rate}×`])),
    );
    const showState = () => {
      play.textContent = lead.paused ? '▶' : '❚❚';
      play.setAttribute('aria-label', lead.paused ? PLAY_LABEL : PAUSE_LABEL);
    };
    const showTime = () => {
      seek.value = String(lead.currentTime);
      time.textContent = `${clock(lead.currentTime)} / ${clock(durationS)}`;
    };
    play.addEventListener('click', (event) => {
      if (event.isTrusted) this.#act('play', { row: row.id, playing: lead.paused });
      if (lead.paused) start(lead);
      else lead.pause();
    });
    seek.addEventListener('input', () => {
      lead.currentTime = Number(seek.value);
    });
    seek.addEventListener('change', (event) => event.isTrusted && this.#act('seek', { row: row.id }));
    speed.addEventListener('change', (event) => {
      lead.playbackRate = Number(speed.value);
      if (event.isTrusted) this.#act('speed', { row: row.id, value: speed.value });
    });
    for (const type of ['play', 'pause', 'ended']) lead.addEventListener(type, showState);
    for (const type of ['timeupdate', 'seeked']) lead.addEventListener(type, showTime);
    return element('div', { part: 'controls' }, [play, seek, time, speed]);
  }
}

// Every video of an episode starts on one clock, so each other video plays, pauses and seeks with the lead
// until its own end.
function follow(lead, followers) {
  const align = () => {
    for (const video of followers) {
      if (Number.isNaN(video.duration)) continue;
      // Past its own end, a shorter camera rests on its last frame.
      const wanted = Math.min(lead.currentTime, video.duration);
      if (Math.abs(video.currentTime - wanted) > DRIFT_S) video.currentTime = wanted;
      if (!lead.paused && video.paused && wanted < video.duration) start(video);
    }
  };
  for (const type of ['play', 'seeking', 'timeupdate']) lead.addEventListener(type, align);
  lead.addEventListener('pause', () => {
    // A paused episode shows one moment on every camera, so each follower takes the lead's time exactly.
    for (const video of followers) {
      video.pause();
      if (!Number.isNaN(video.duration)) video.currentTime = Math.min(lead.currentTime, video.duration);
    }
  });
  lead.addEventListener('ratechange', () => {
    for (const video of followers) video.playbackRate = lead.playbackRate;
  });
}
