// The headset video over WebRTC: one peer connection to the service, which receives one video track into a <video>
// element. The setup messages go over `signaling` (`SocketSignaling` of core.js). A data channel carries the timing of
// the frames, so the page measures the age of each frame and the stalls, and reports them to the service, which logs
// them (`positronic.drivers.webrtc`)

// The types of the setup messages, and the label of the timing channel
const SIGNAL_TYPE = { offer: 'offer', answer: 'answer', candidate: 'candidate' };
const TIMING_CHANNEL = 'timing';
// With no new decoded frame for this long, the page connects again
const LOST_MS = 5000;

export class WebRTCVideo {
  constructor(signaling, video) {
    this.signaling = signaling;
    this.video = video;
    this.connection = null;
    this.timing = null;
    this.offerId = 0;
    this.framesDecoded = 0;
    this.lastDecodedAt = performance.now();
    signaling.onopen = () => this.connect();
    signaling.onmessage = (message) => this.takeSignal(message);
    if (signaling.isOpen) this.connect();
    this.watchdog = setInterval(() => this.replaceLost(), 1000);
  }

  // A new peer connection, in place of the one before it
  connect() {
    this.closeConnection();
    const id = ++this.offerId;
    const connection = new RTCPeerConnection();
    this.connection = connection;
    const transceiver = connection.addTransceiver('video', { direction: 'recvonly' });
    preferH264(transceiver);
    this.timing = new FrameTiming(connection, connection.createDataChannel(TIMING_CHANNEL), this.video);
    connection.ontrack = (event) => {
      this.video.srcObject = new MediaStream([event.track]);
      this.video.play().catch((error) => console.error('Cannot play the video:', error));
    };
    connection.onicecandidate = ({ candidate }) => {
      if (candidate) this.signaling.send({ type: SIGNAL_TYPE.candidate, id, ...candidate.toJSON() });
    };
    connection.onconnectionstatechange = () => console.log(`Video connection ${id}: ${connection.connectionState}`);
    this.framesDecoded = 0;
    this.lastDecodedAt = performance.now();
    connection.createOffer()
      .then((offer) => connection.setLocalDescription(offer))
      .then(() => {
        if (this.connection === connection) {
          this.signaling.send({ type: SIGNAL_TYPE.offer, id, sdp: connection.localDescription.sdp });
        }
      })
      .catch((error) => console.error('Cannot make the video offer:', error));
  }

  takeSignal(message) {
    if (message.type !== SIGNAL_TYPE.answer || message.id !== this.offerId || this.connection === null) return;
    this.connection.setRemoteDescription({ type: 'answer', sdp: message.sdp })
      .catch((error) => console.error('Cannot take the video answer:', error));
  }

  // Connect again when the connection failed or decodes no frame. The decoder count does not need the frame callbacks
  async replaceLost() {
    const connection = this.connection;
    if (connection === null || !this.signaling.isOpen) return;
    const stats = await connection.getStats();
    stats.forEach((entry) => {
      const decoded = entry.type === 'inbound-rtp' && entry.kind === 'video' ? entry.framesDecoded : undefined;
      if (typeof decoded === 'number' && decoded !== this.framesDecoded) {
        this.framesDecoded = decoded;
        this.lastDecodedAt = performance.now();
      }
    });
    const silentMs = performance.now() - this.lastDecodedAt;
    if (connection !== this.connection || (silentMs < LOST_MS && connection.connectionState !== 'failed')) return;
    console.log(`Video lost (no frame for ${Math.round(silentMs)} ms, ${connection.connectionState}), connecting again`);
    this.connect();
  }

  closeConnection() {
    if (this.timing !== null) this.timing.close();
    if (this.connection !== null) this.connection.close();
    this.timing = null;
    this.connection = null;
  }

  close() {
    clearInterval(this.watchdog);
    this.signaling.onopen = null;
    this.signaling.onmessage = null;
    this.closeConnection();
  }
}

// Put H.264 first, which the headset decodes in hardware. A browser with no codec choice keeps its own order
function preferH264(transceiver) {
  if (typeof RTCRtpReceiver.getCapabilities !== 'function' || typeof transceiver.setCodecPreferences !== 'function') {
    return;
  }
  const codecs = RTCRtpReceiver.getCapabilities('video').codecs;
  const h264 = codecs.filter((codec) => codec.mimeType === 'video/H264');
  transceiver.setCodecPreferences([...h264, ...codecs.filter((codec) => codec.mimeType !== 'video/H264')]);
}

// A gap between two shown frames above this is a stall
const STALL_MS = 250;
const PING_MS = 1000;
const REPORT_MS = 5000;
// The pings whose round trip the clock offset comes from: the one with the shortest round trip among them
const PING_SAMPLES = 10;
// The RTP clock of video, in ticks per ms
const RTP_TICKS_PER_MS = 90;
// The number of shown frames that must agree on the RTP offset before the page uses it
const OFFSET_VOTES = 3;
// The sent times that the page keeps to find the RTP offset: 3 s of frames at 30 frames per second
const SENT_KEPT = 90;

// The page end of the timing channel. An age goes from the hand-over of a frame to the encoder to its show on the page.
// The RTP timestamp of a frame is its sent time plus an offset that only the sender knows, so shown frames vote for it
class FrameTiming {
  constructor(connection, channel, video) {
    this.connection = connection;
    this.channel = channel;
    this.video = video;
    this.closed = false;
    this.clockSamples = [];
    this.clockOffsetMs = null;
    this.sent = [];
    this.votes = new Map();
    this.rtpOffset = null;
    this.lastShownAt = null;
    this.startReport();
    channel.onmessage = (event) => this.take(JSON.parse(event.data));
    channel.onopen = () => {
      this.ping();
      this.pinger = setInterval(() => this.ping(), PING_MS);
      this.reporter = setInterval(() => this.report(), REPORT_MS);
    };
    const onFrame = (now, metadata) => {
      if (this.closed) return;
      this.shown(metadata);
      video.requestVideoFrameCallback(onFrame);
    };
    video.requestVideoFrameCallback(onFrame);
  }

  startReport() {
    this.since = performance.now();
    this.frames = 0;
    this.ages = [];
    this.stalls = [];
  }

  ping() {
    if (this.channel.readyState === 'open') this.channel.send(JSON.stringify({ ping: performance.now() }));
  }

  take(message) {
    if ('pong' in message) {
      const now = performance.now();
      this.clockSamples.push({ rttMs: now - message.pong, offsetMs: message.service_ms - (message.pong + now) / 2 });
      this.clockSamples = this.clockSamples.slice(-PING_SAMPLES);
      this.clockOffsetMs = this.clockSamples.reduce((best, sample) => (sample.rttMs < best.rttMs ? sample : best))
        .offsetMs;
    } else if ('frame' in message) {
      this.sent.push(message.frame % 2 ** 32);
      if (this.sent.length > SENT_KEPT) this.sent.shift();
    }
  }

  shown(metadata) {
    const at = metadata.presentationTime;
    if (this.lastShownAt !== null && at - this.lastShownAt > STALL_MS) this.stalls.push(at - this.lastShownAt);
    this.lastShownAt = at;
    this.frames++;
    const rtp = metadata.rtpTimestamp;
    if (rtp === undefined || this.clockOffsetMs === null) return;
    if (this.rtpOffset === null) this.vote(rtp);
    if (this.rtpOffset === null) return;
    const encodedMs = ((rtp - this.rtpOffset) >>> 0) / RTP_TICKS_PER_MS;
    this.ages.push(at + this.clockOffsetMs - encodedMs);
  }

  // The true offset is the same for each shown frame. Another difference repeats only by chance
  vote(rtp) {
    for (const ticks of this.sent) {
      const offset = (rtp - ticks) >>> 0;
      const votes = (this.votes.get(offset) || 0) + 1;
      this.votes.set(offset, votes);
      if (votes >= OFFSET_VOTES) {
        this.rtpOffset = offset;
        this.votes.clear();
        return;
      }
    }
  }

  async report() {
    const stats = await this.connection.getStats();
    let inbound = null;
    let pair = null;
    const codecs = new Map();
    stats.forEach((entry) => {
      if (entry.type === 'inbound-rtp' && entry.kind === 'video') inbound = entry;
      if (entry.type === 'candidate-pair' && entry.nominated && entry.state === 'succeeded') pair = entry;
      if (entry.type === 'codec') codecs.set(entry.id, entry.mimeType);
    });
    const report = {
      seconds: (performance.now() - this.since) / 1000,
      frames: this.frames,
      ages_ms: this.ages,
      stalls_ms: this.stalls,
      packets_lost: inbound?.packetsLost ?? 0,
      nacks: inbound?.nackCount ?? 0,
      plis: inbound?.pliCount ?? 0,
      rtt_ms: pair?.currentRoundTripTime === undefined ? null : pair.currentRoundTripTime * 1000,
      codec: codecs.get(inbound?.codecId) ?? null,
    };
    this.startReport();
    if (this.channel.readyState === 'open') this.channel.send(JSON.stringify({ report }));
  }

  close() {
    this.closed = true;
    clearInterval(this.pinger);
    clearInterval(this.reporter);
  }
}
