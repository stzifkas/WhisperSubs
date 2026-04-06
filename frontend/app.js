'use strict';

const SAMPLE_RATE = 24000;  // Required by OpenAI Realtime API

const startBtn = document.getElementById('start-btn');
const audioSourceSelect = document.getElementById('audio-source-select');
const srcLangSelect = document.getElementById('src-lang-select');
const langSelect = document.getElementById('lang-select');
const modeSelect = document.getElementById('mode-select');

// Commit delay by mode — how long to wait for translation before showing
// source caption on its own. Must mirror ModeProfile.commit_delay_ms in backend.
const MODE_COMMIT_DELAY_MS = {
  fast:      500,
  balanced:  2000,
  broadcast: 800,
  precision: 3000,
};
const captionsEl = document.getElementById('captions');
const captionsWrap = document.getElementById('captions-wrap');
const micLevelWrap = document.getElementById('mic-level-wrap');
const micBar = document.getElementById('mic-bar');
const statusEl = document.getElementById('status');
const toast = document.getElementById('toast');
const advancedBtn = document.getElementById('advanced-btn');
const advancedPanel = document.getElementById('advanced-panel');
const silenceSlider = document.getElementById('silence-threshold');
const silenceVal = document.getElementById('silence-threshold-val');
const noSpeechSlider = document.getElementById('no-speech-threshold');
const noSpeechVal = document.getElementById('no-speech-threshold-val');
const overlayBtn = document.getElementById('overlay-btn');
let subPopup = null;
const chatBtn = document.getElementById('chat-btn');
const chatPanel = document.getElementById('chat-panel');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');

let ws = null;
let recording = false;
let audioContext = null;
let analyser = null;
let animFrameId = null;
let maxCaptionBlocks = 30;
let currentSessionId = null;

// ── Caption display ───────────────────────────────────────────────────────────

function addCaption(transcript, translation, subtitleState) {
  // Remove live caption block if present
  const live = captionsEl.querySelector('.caption-block.live');
  if (live) live.remove();

  const block = document.createElement('div');
  block.className = 'caption-block' + (subtitleState ? ' subtitle-' + subtitleState : '');

  const tEl = document.createElement('p');
  tEl.className = 'transcript';
  tEl.textContent = transcript;
  block.appendChild(tEl);

  if (translation) {
    const trEl = document.createElement('p');
    trEl.className = 'translation';
    trEl.textContent = translation;
    block.appendChild(trEl);
  }

  captionsEl.appendChild(block);

  while (captionsEl.children.length > maxCaptionBlocks) {
    captionsEl.removeChild(captionsEl.firstChild);
  }

  captionsWrap.scrollTop = captionsWrap.scrollHeight;
}

// ── Live delta display ────────────────────────────────────────────────────────

let liveText = '';

function handleTranscriptDelta(delta) {
  liveText += delta;

  let liveBlock = captionsEl.querySelector('.caption-block.live');
  if (!liveBlock) {
    liveBlock = document.createElement('div');
    liveBlock.className = 'caption-block live';
    const p = document.createElement('p');
    p.className = 'transcript';
    liveBlock.appendChild(p);
    captionsEl.appendChild(liveBlock);
  }
  liveBlock.querySelector('.transcript').textContent = liveText;
  captionsWrap.scrollTop = captionsWrap.scrollHeight;
  updateSubPopup(liveText, '');
}

// ── Toast notifications ───────────────────────────────────────────────────────

let toastTimeout = null;
function showToast(msg, type = '') {
  toast.textContent = msg;
  toast.className = 'show ' + (type === 'warn' ? 'warn' : type === 'error' ? 'error-toast' : '');
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => { toast.className = ''; }, 4000);
}

// ── Status badge ──────────────────────────────────────────────────────────────

function setStatus(text, cls = '') {
  statusEl.textContent = text;
  statusEl.className = cls;
}

// ── Mic level animation ───────────────────────────────────────────────────────

function animateMicLevel() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  let max = 0;
  for (let i = 0; i < data.length; i++) {
    const v = Math.abs(data[i] - 128) / 128;
    if (v > max) max = v;
  }
  micBar.style.width = Math.min(100, max * 250) + '%';
  animFrameId = requestAnimationFrame(animateMicLevel);
}

// ── Subtitle lifecycle ────────────────────────────────────────────────────────
// Each segment has an integer id (= backend segment_index).
// State machine per block: tentative → (update) → stable | locked
//
// Events:  subtitle_tentative → subtitle_update → subtitle_commit
//          subtitle_lock (may arrive for older blocks after new segments come in)

// segment_id → DOM element for all committed+tentative blocks
const subtitleBlocks = new Map();

// Safety timeout — if subtitle_commit never arrives (e.g. LLM error), promote
// the tentative block to stable after this delay so it doesn't hang forever.
let TRANSLATION_WAIT_MS = MODE_COMMIT_DELAY_MS[modeSelect.value] ?? 2000;
const COMMIT_SAFETY_BUFFER_MS = 4000;

// Accumulated streaming translation tokens for the current tentative block.
let streamingTranslation = '';

function _tentativeBlock() {
  return captionsEl.querySelector('.caption-block.subtitle-tentative');
}

function handleSubtitleTentative(id, text, lang) {
  // Dismiss live-delta block — the tentative block supersedes it.
  liveText = '';
  streamingTranslation = '';
  const live = captionsEl.querySelector('.caption-block.live');
  if (live) live.remove();

  const block = document.createElement('div');
  block.className = 'caption-block subtitle-tentative';
  block.dataset.segmentId = id;

  const tEl = document.createElement('p');
  tEl.className = 'transcript';
  tEl.textContent = text;
  block.appendChild(tEl);

  captionsEl.appendChild(block);

  // Evict oldest blocks once we exceed the cap
  while (captionsEl.children.length > maxCaptionBlocks) {
    const first = captionsEl.firstChild;
    const evictedId = first.dataset?.segmentId ? parseInt(first.dataset.segmentId) : null;
    if (evictedId !== null) subtitleBlocks.delete(evictedId);
    captionsEl.removeChild(first);
  }

  subtitleBlocks.set(id, block);
  captionsWrap.scrollTop = captionsWrap.scrollHeight;

  // Safety: if subtitle_commit never fires, promote to stable
  block._commitTimer = setTimeout(() => {
    if (block.classList.contains('subtitle-tentative')) {
      block.classList.replace('subtitle-tentative', 'subtitle-stable');
    }
  }, TRANSLATION_WAIT_MS + COMMIT_SAFETY_BUFFER_MS);
}

function handleSubtitleUpdate(id, text) {
  const block = subtitleBlocks.get(id);
  if (!block) return;
  const tEl = block.querySelector('.transcript');
  if (tEl) tEl.textContent = text;
  captionsWrap.scrollTop = captionsWrap.scrollHeight;
  updateSubPopup(text, '');
}

function handleTranslationDelta(delta) {
  streamingTranslation += delta;
  const block = _tentativeBlock();
  if (!block) return;
  let trEl = block.querySelector('.translation.streaming');
  if (!trEl) {
    trEl = document.createElement('p');
    trEl.className = 'translation streaming';
    block.appendChild(trEl);
  }
  trEl.textContent = streamingTranslation;
  captionsWrap.scrollTop = captionsWrap.scrollHeight;
}

function handleSubtitleCommit(id, text, translation, locked) {
  streamingTranslation = '';
  const block = subtitleBlocks.get(id);
  if (!block) {
    // Revision-blocked path: no tentative block was created — render directly.
    addCaption(text, translation || null, locked ? 'locked' : 'stable');
    updateSubPopup(text, translation || '');
    return;
  }

  clearTimeout(block._commitTimer);

  const tEl = block.querySelector('.transcript');
  if (tEl) tEl.textContent = text;

  // Replace streaming translation element with the final committed one.
  const streamingTr = block.querySelector('.translation.streaming');
  if (streamingTr) streamingTr.remove();
  if (translation) {
    const trEl = document.createElement('p');
    trEl.className = 'translation';
    trEl.textContent = translation;
    block.appendChild(trEl);
  }

  block.className = 'caption-block ' + (locked ? 'subtitle-locked' : 'subtitle-stable');
  updateSubPopup(text, translation || '');
  captionsWrap.scrollTop = captionsWrap.scrollHeight;
}

function handleSubtitleLock(id) {
  const block = subtitleBlocks.get(id);
  if (!block) return;
  clearTimeout(block._commitTimer);
  block.classList.remove('subtitle-tentative', 'subtitle-stable');
  block.classList.add('subtitle-locked');
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setStatus('Connected', 'connected');
    sendConfig();
    sendMode();
  };

  ws.onclose = () => {
    setStatus('Disconnected');
    if (recording) stopRecording();
  };

  ws.onerror = () => {
    setStatus('Error', 'error');
    showToast('WebSocket error', 'error');
  };

  ws.onmessage = (evt) => {
    let msg;
    try { msg = JSON.parse(evt.data); } catch { return; }

    switch (msg.type) {
      case 'session_id':
        currentSessionId = msg.id;
        updateDownloadBtn();
        break;
      case 'transcript_delta':
        handleTranscriptDelta(msg.delta);
        break;
      case 'subtitle_tentative':
        handleSubtitleTentative(msg.id, msg.text, msg.lang);
        break;
      case 'subtitle_update':
        handleSubtitleUpdate(msg.id, msg.text);
        break;
      case 'translation_delta':
        handleTranslationDelta(msg.delta);
        break;
      case 'subtitle_commit':
        handleSubtitleCommit(msg.id, msg.text, msg.translation, msg.locked);
        break;
      case 'subtitle_lock':
        handleSubtitleLock(msg.id);
        break;
      case 'warning':
        showToast(msg.message, 'warn');
        break;
      case 'error':
        showToast(msg.message, 'error');
        break;
    }
  };
}

// ── Recording ─────────────────────────────────────────────────────────────────

async function getAudioStream() {
  const mode = audioSourceSelect.value;

  if (mode === 'tab') {
    const displayStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: true,
    });
    displayStream.getVideoTracks().forEach(t => t.stop());
    const audioTracks = displayStream.getAudioTracks();
    if (audioTracks.length === 0) {
      throw new Error('No audio track captured. Make sure to check "Share tab audio" in Chrome\'s dialog.');
    }
    return new MediaStream(audioTracks);
  }

  return navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true },
  });
}

async function startRecording() {
  try {
    const stream = await getAudioStream();

    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(stream);

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    source.connect(processor);
    processor.connect(audioContext.destination);

    // Stream raw PCM16 directly — no chunking, no WAV headers
    processor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const float32 = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        pcm16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
      }
      ws.send(pcm16.buffer);
    };

    stream.getAudioTracks()[0].addEventListener('ended', () => {
      if (recording) stopRecording();
    });

    micLevelWrap.classList.add('visible');
    animateMicLevel();

    connectWS();
    recording = true;
    startBtn.textContent = 'Stop';
    startBtn.classList.add('recording');
    audioSourceSelect.disabled = true;

  } catch (err) {
    showToast((audioSourceSelect.value === 'tab' ? 'Tab capture error: ' : 'Microphone error: ') + err.message, 'error');
  }
}

function stopRecording() {
  recording = false;
  cancelAnimationFrame(animFrameId);

  // Dismiss live-delta and any in-flight tentative blocks
  captionsEl.querySelectorAll('.caption-block.live, .caption-block.subtitle-tentative')
    .forEach(el => {
      const sid = el.dataset?.segmentId ? parseInt(el.dataset.segmentId) : null;
      if (sid !== null) subtitleBlocks.delete(sid);
      clearTimeout(el._commitTimer);
      el.remove();
    });
  liveText = '';
  streamingTranslation = '';

  if (audioContext) {
    audioContext.close();
    audioContext = null;
    analyser = null;
  }

  if (ws) {
    ws.close();
    ws = null;
  }

  micBar.style.width = '0%';
  micLevelWrap.classList.remove('visible');
  startBtn.textContent = 'Start';
  startBtn.classList.remove('recording');
  audioSourceSelect.disabled = false;
  setStatus('Idle');
  flushPending();
}

// ── Subtitle popup ────────────────────────────────────────────────────────────

const POPUP_HTML = `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<title>WhisperSubs</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: rgba(0,0,0,0.82);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 12px 24px 16px;
    overflow: hidden;
  }
  #controls {
    display: flex;
    gap: 6px;
    margin-bottom: 10px;
    opacity: 0.25;
    transition: opacity 0.2s;
  }
  body:hover #controls { opacity: 1; }
  #controls button {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 5px;
    color: #cbd5e1;
    font-size: 11px;
    padding: 3px 10px;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  #controls button:hover { background: rgba(255,255,255,0.15); }
  #controls button.active { background: rgba(96,165,250,0.25); border-color: #60a5fa; color: #93c5fd; }
  #t {
    font-size: 32px;
    font-weight: 600;
    color: #f1f5f9;
    text-align: center;
    line-height: 1.4;
    text-shadow: 0 2px 6px rgba(0,0,0,0.9);
    word-break: break-word;
  }
  #tr {
    font-size: 26px;
    color: #fbbf24;
    font-style: italic;
    text-align: center;
    line-height: 1.4;
    margin-top: 6px;
    text-shadow: 0 2px 6px rgba(0,0,0,0.9);
    word-break: break-word;
  }
  .hidden { display: none !important; }
</style>
</head>
<body>
<div id="controls">
  <button id="btn-both" class="active" onclick="setMode('both')">Both</button>
  <button id="btn-source" onclick="setMode('source')">Original</button>
  <button id="btn-translated" onclick="setMode('translated')">Translated</button>
</div>
<div id="t"></div>
<div id="tr"></div>
<script>
  var mode = 'both';
  function setMode(m) {
    mode = m;
    document.getElementById('btn-both').className = m === 'both' ? 'active' : '';
    document.getElementById('btn-source').className = m === 'source' ? 'active' : '';
    document.getElementById('btn-translated').className = m === 'translated' ? 'active' : '';
    render();
  }
  function render() {
    var t = document.getElementById('t');
    var tr = document.getElementById('tr');
    t.className = (mode === 'translated') ? 'hidden' : '';
    tr.className = (mode === 'source') ? 'hidden' : '';
  }
</script>
</body>
</html>`;

function openSubPopup() {
  if (subPopup && !subPopup.closed) { subPopup.focus(); return; }
  subPopup = window.open(
    '', 'whispersubs_overlay',
    'width=720,height=160,toolbar=no,menubar=no,location=no,status=no,resizable=yes'
  );
  subPopup.document.write(POPUP_HTML);
  subPopup.document.close();
  subPopup.onunload = () => {
    subPopup = null;
    overlayBtn.classList.remove('open');
  };
  overlayBtn.classList.add('open');
}

function updateSubPopup(transcript, translation) {
  if (!subPopup || subPopup.closed) return;
  subPopup.document.getElementById('t').textContent = transcript;
  subPopup.document.getElementById('tr').textContent = translation || '';
}

// ── SRT download ──────────────────────────────────────────────────────────────

const downloadBtn = document.getElementById('download-btn');

function updateDownloadBtn() {
  if (currentSessionId) {
    downloadBtn.href = `/srt/${currentSessionId}`;
    downloadBtn.classList.remove('hidden');
  }
}

// ── Config helpers ────────────────────────────────────────────────────────────

function sendConfig() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: 'config',
    target_language: langSelect.value,
    whisper_language: srcLangSelect.value,
    silence_threshold: parseFloat(silenceSlider.value),
    no_speech_threshold: parseFloat(noSpeechSlider.value),
  }));
}

function sendMode() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'mode', mode: modeSelect.value }));
}

// ── Chat ──────────────────────────────────────────────────────────────────────

function appendChatMsg(text, role) {
  const el = document.createElement('div');
  el.className = `chat-msg ${role}`;
  el.textContent = text;
  chatMessages.appendChild(el);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return el;
}

async function sendChatMessage() {
  const text = chatInput.value.trim();
  if (!text || !currentSessionId) return;

  chatInput.value = '';
  chatInput.disabled = true;
  chatSendBtn.disabled = true;

  appendChatMsg(text, 'user');
  const thinking = appendChatMsg('Thinking…', 'thinking');

  try {
    const res = await fetch(`/chat/${currentSessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    thinking.remove();
    appendChatMsg(res.ok ? data.reply : (data.detail || 'Error'), res.ok ? 'assistant' : 'thinking');
  } catch {
    thinking.textContent = 'Network error.';
  } finally {
    chatInput.disabled = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────

startBtn.addEventListener('click', () => {
  if (recording) stopRecording();
  else startRecording();
});

langSelect.addEventListener('change', sendConfig);
srcLangSelect.addEventListener('change', sendConfig);

modeSelect.addEventListener('change', () => {
  // Update safety-timer duration to match the new mode's expected commit latency.
  TRANSLATION_WAIT_MS = MODE_COMMIT_DELAY_MS[modeSelect.value] ?? 2000;
  sendMode();
});

advancedBtn.addEventListener('click', () => {
  advancedPanel.classList.toggle('open');
  advancedBtn.classList.toggle('open');
});

silenceSlider.addEventListener('input', () => {
  silenceVal.textContent = silenceSlider.value;
  sendConfig();
});

noSpeechSlider.addEventListener('input', () => {
  noSpeechVal.textContent = parseFloat(noSpeechSlider.value).toFixed(2);
  sendConfig();
});

overlayBtn.addEventListener('click', openSubPopup);

chatBtn.addEventListener('click', () => {
  chatPanel.classList.toggle('open');
  chatBtn.classList.toggle('open');
});

chatSendBtn.addEventListener('click', sendChatMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
});
