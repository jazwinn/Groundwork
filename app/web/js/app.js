// Bootstrap: wires the filmstrip, the editor, the step panels, the job bar
// and the keyboard shortcuts together.

import { api, connect } from './api.js';
import { Editor } from './canvas.js';
import { labelPanel, mediaPanel, scriptOutcome, trainPanel } from './panels.js';
import { confirmDialog, el, nativePicker, openModal, closeModal, plural, toast } from './ui.js';

const UI_DEFAULTS = {
  importStride: 1,
  policy: 'replace',
  method: 'describe',
  concept: '',
  promptText: '',
  classId: 0,
  start: 0,
  stride: 1,
  trackRange: 0,
  epochs: 50,
  imgsz: 640,
  cache: 'off',
  workers: 4,
  model: '',
  datasetDir: '',
  precision: 'FP32',
  shape: 'dynamic',
  skipUnreviewed: false,
  filter: 'all',
  nativeDialogs: true,
  mergeThreshold: 0.8,
  mergeSameClass: true,
  wrapperBoxes: true,
  script: '',
  // Per script, and only the settings moved off that script's own
  // defaults — editing a default in the file is then picked up.
  scriptParams: {},
  // Only the settings moved off their default live here; the rest come from
  // the server's spec, so a change to a default is picked up rather than
  // frozen into everyone's browser.
  augment: {},
  open: {},
};

const UI_KEY = 'groundwork.ui';
const UI_KEY_LEGACY = 'modeltrainer.ui';   // what it was called before the rename

const app = {
  state: { classes: ['object'], task: 'detect', samModel: '', samModels: [], threshold: 0.5, frames: [], stats: {}, device: {}, augmentations: [], scripts: [] },
  models: null,
  ui: loadUi(),
  step: 'media',
  currentIndex: null,
  // Frames picked in the filmstrip. Always holds currentIndex; Clear labels and
  // Drop frames act on all of it.
  frameSelection: new Set(),
  frameDetail: null,
  examples: { positive: [], negative: [] },
  jobs: new Map(),
  jobLogs: new Map(),
  editor: null,
  suppressReload: 0,
  // The scripts folder is read on demand rather than with the rest of the
  // state: it imports every script, and a script is free to be slow.
  scriptsState: 'idle',   // 'idle' | 'loading' | 'ready' | 'error'
  scriptsDir: '',
  scriptReport: null,
};

function loadUi() {
  try {
    const saved = localStorage.getItem(UI_KEY) ?? localStorage.getItem(UI_KEY_LEGACY);
    return { ...UI_DEFAULTS, ...JSON.parse(saved || '{}') };
  } catch { return { ...UI_DEFAULTS }; }
}

function saveUi() {
  try {
    localStorage.setItem(UI_KEY, JSON.stringify(app.ui));
    localStorage.removeItem(UI_KEY_LEGACY);
  } catch { /* private mode */ }
}

// ── Elements ───────────────────────────────────────────────────

const dom = {
  steps: document.getElementById('steps'),
  panel: document.getElementById('panel'),
  strip: document.getElementById('strip'),
  stripEmpty: document.getElementById('filmstripEmpty'),
  stripCount: document.getElementById('filmstripCount'),
  filter: document.getElementById('frameFilter'),
  canvas: document.getElementById('canvas'),
  host: document.getElementById('canvasHost'),
  stageEmpty: document.getElementById('stageEmpty'),
  toolbar: document.getElementById('stageToolbar'),
  tools: document.getElementById('toolButtons'),
  activeClass: document.getElementById('activeClass'),
  frameLabel: document.getElementById('frameLabel'),
  boxSummary: document.getElementById('boxSummary'),
  saveState: document.getElementById('saveState'),
  selectionInfo: document.getElementById('selectionInfo'),
  mergeBtn: document.getElementById('mergeBtn'),
  wrapperBtn: document.getElementById('wrapperBtn'),
  clearFrameBtn: document.getElementById('clearFrameBtn'),
  dropFrameBtn: document.getElementById('dropFrameBtn'),
  stripSelection: document.getElementById('stripSelection'),
  jobbar: document.getElementById('jobbar'),
  device: document.getElementById('deviceInfo'),
  frameCount: document.getElementById('frameCount'),
};

// ── App methods used by the panels ─────────────────────────────

app.currentFrame = () => app.state.frames.find((f) => f.index === app.currentIndex) || null;

app.goStep = (step) => {
  app.step = step;
  for (const btn of dom.steps.querySelectorAll('.step')) {
    btn.setAttribute('aria-current', btn.dataset.step === step ? 'true' : 'false');
  }
  dom.toolbar.style.display = step === 'label' ? '' : 'none';
  app.renderPanel();
};

app.renderPanel = () => {
  const builder = { media: mediaPanel, label: labelPanel, train: trainPanel }[app.step];
  // The panel is rebuilt on almost every event, so hold the scroll position —
  // otherwise it jumps to the top while you are working in a section.
  const scroll = dom.panel.scrollTop;
  dom.panel.innerHTML = '';
  dom.panel.appendChild(builder(app));
  dom.panel.scrollTop = scroll;
  saveUi();
};

app.onMethodChange = () => {
  // Selecting the example method puts the right tool in the user's hand.
  if (app.ui.method === 'example') setTool('pos');
  else if (app.editor?.tool === 'pos' || app.editor?.tool === 'neg') setTool('select');
  app.renderPanel();
};

// Picking a class in the toolbar also recolours whatever box is selected.
app.setClass = (id) => {
  app.ui.classId = id;
  dom.activeClass.value = String(id);
  app.editor.setClassId(id);
};

// Picking a class inside an auto-label method only says where results should
// land — it must not relabel the box the user happens to have selected.
app.setSearchClass = (id) => {
  app.ui.classId = id;
  dom.activeClass.value = String(id);
  app.editor.classId = id;
};

app.reloadFrame = () => reloadCurrentFrame();

/** Read the scripts folder. Cheap after the first time, so panels may ask freely. */
app.refreshScripts = async (force = false) => {
  if (app.scriptsState === 'loading') return;
  if (app.scriptsState === 'ready' && !force) return;
  app.scriptsState = 'loading';
  try {
    const res = await api.scripts();
    app.state.scripts = res.scripts || [];
    app.scriptsDir = res.dir || '';
    app.scriptsState = 'ready';
  } catch {
    app.scriptsState = 'error';
  }
  app.renderPanel();
};

/** Settle the frame before something else edits it, and leave a way back. */
app.beforeFrameEdit = async () => {
  if (app.editor) app.editor.pushUndo();
  await flushSave();
};

app.onTaskChange = () => {
  syncMaskTools();
  app.renderPanel();
};

app.clearExamples = () => {
  app.editor.clearExamples();
  app.examples = { positive: [], negative: [] };
  app.renderPanel();
};

// ── Filmstrip ──────────────────────────────────────────────────

function visibleFrames() {
  const filter = app.ui.filter;
  return app.state.frames.filter((f) => (
    filter === 'labelled' ? f.boxes > 0 : filter === 'empty' ? f.boxes === 0 : true
  ));
}

// A timer rather than requestAnimationFrame: rAF is paused while the tab is in
// the background, and a long import in another tab must still land in the strip.
let stripTimer = null;
function scheduleStrip() {
  if (stripTimer) return;
  stripTimer = setTimeout(() => { stripTimer = null; renderStrip(); }, 60);
}

function renderStrip() {
  const frames = visibleFrames();
  dom.stripCount.textContent = frames.length
    ? `${frames.length} of ${app.state.frames.length} shown`
    : '';
  dom.stripEmpty.hidden = app.state.frames.length > 0;
  dom.frameCount.textContent = app.state.frames.length
    ? `${app.state.stats.labelled || 0}/${app.state.frames.length} labelled`
    : 'no frames';

  const existing = new Map();
  for (const node of dom.strip.children) existing.set(Number(node.dataset.index), node);

  const fragment = document.createDocumentFragment();
  for (const frame of frames) {
    let node = existing.get(frame.index);
    if (node) {
      existing.delete(frame.index);
      updateThumb(node, frame);
    } else {
      node = makeThumb(frame);
    }
    fragment.appendChild(node);
  }
  for (const node of existing.values()) node.remove();
  dom.strip.appendChild(fragment);
}

function makeThumb(frame) {
  const img = el('img', { loading: 'lazy', alt: `Frame ${frame.index}`, src: `/api/frames/${frame.index}/thumb` });
  const node = el('div', {
    class: 'thumb', 'data-index': String(frame.index), role: 'option', tabindex: '0',
    onclick: (e) => pickFrame(frame.index, e),
    onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickFrame(frame.index, e); } },
  }, [img, el('span', { class: 'thumb-idx', text: String(frame.index) }), el('span', { class: 'thumb-n' })]);
  updateThumb(node, frame);
  return node;
}

function updateThumb(node, frame) {
  node.setAttribute('aria-selected', frame.index === app.currentIndex ? 'true' : 'false');
  const multi = app.frameSelection.size > 1;
  node.setAttribute('data-picked', multi && app.frameSelection.has(frame.index) ? 'true' : 'false');
  const badge = node.querySelector('.thumb-n');
  badge.textContent = frame.boxes ? String(frame.boxes) : '';
  badge.className = `thumb-n ${frame.status}`;
  node.setAttribute('data-exported', frame.exported ? 'true' : 'false');
}

/** Filmstrip click: plain picks one, Shift takes a range, Ctrl adds or removes. */
function pickFrame(index, event) {
  const frames = visibleFrames().map((f) => f.index);
  if (event?.shiftKey && app.currentIndex !== null) {
    const from = frames.indexOf(app.currentIndex);
    const to = frames.indexOf(index);
    if (from >= 0 && to >= 0) {
      const [lo, hi] = from < to ? [from, to] : [to, from];
      app.frameSelection = new Set(frames.slice(lo, hi + 1));
    }
  } else if (event?.ctrlKey || event?.metaKey) {
    if (app.frameSelection.has(index) && app.frameSelection.size > 1) {
      app.frameSelection.delete(index);
      if (index === app.currentIndex) {
        selectFrame([...app.frameSelection][0], { keepSelection: true });
        return;
      }
      renderStrip();
      updateFrameActions();
      return;
    }
    app.frameSelection.add(index);
  } else {
    app.frameSelection = new Set([index]);
  }
  selectFrame(index, { keepSelection: true });
}

function selectedFrames() {
  return [...app.frameSelection].sort((a, b) => a - b);
}

/** Keep the buttons honest about how many frames they are about to change. */
function updateFrameActions() {
  const count = app.frameSelection.size;
  const many = count > 1;
  dom.clearFrameBtn.textContent = many ? `Clear ${count} frames` : 'Clear frame';
  dom.dropFrameBtn.textContent = many ? `Drop ${count} frames` : 'Drop frame';
  dom.clearFrameBtn.title = many
    ? `Remove every box on the ${count} selected frames`
    : 'Remove every box on this frame';
  dom.dropFrameBtn.title = many
    ? `Drop the ${count} selected frames from the session`
    : 'Drop this frame from the session';

  dom.stripSelection.hidden = !many;
  dom.stripSelection.innerHTML = '';
  if (many) {
    dom.stripSelection.append(
      el('span', { text: `${count} selected` }),
      el('button', {
        class: 'linky', text: 'clear', title: 'Go back to a single frame',
        onclick: () => {
          app.frameSelection = new Set(app.currentIndex === null ? [] : [app.currentIndex]);
          renderStrip();
          updateFrameActions();
        },
      }),
    );
  }
}

// ── Frame selection ────────────────────────────────────────────

async function selectFrame(index, { scroll = false, keepSelection = false } = {}) {
  if (index === null || index === undefined) return;
  await flushSave();
  app.currentIndex = index;
  if (!keepSelection) app.frameSelection = new Set([index]);
  else app.frameSelection.add(index);

  let detail;
  try {
    detail = await api.frame(index);
  } catch {
    return;
  }
  app.frameDetail = detail;
  dom.stageEmpty.hidden = true;
  await app.editor.load(`/api/frames/${index}/image`, detail.boxes);
  app.examples = { positive: [], negative: [] };

  dom.frameLabel.textContent = `Frame ${index}`;
  updateBoxSummary();
  updateReviewButton();
  updateSelectionInfo();
  updateFrameActions();
  for (const node of dom.strip.children) {
    const at = Number(node.dataset.index);
    node.setAttribute('aria-selected', at === index ? 'true' : 'false');
    node.setAttribute('data-picked',
      app.frameSelection.size > 1 && app.frameSelection.has(at) ? 'true' : 'false');
  }
  if (scroll) {
    const node = dom.strip.querySelector(`[data-index="${index}"]`);
    node?.scrollIntoView({ block: 'nearest' });
  }
  prewarmSnap();
  app.renderPanel();
}

function updateReviewButton() {
  const btn = document.getElementById('reviewBtn');
  const reviewed = app.frameDetail?.status === 'verified';
  btn.textContent = reviewed ? 'Reviewed ✓' : 'Mark reviewed';
  btn.disabled = app.currentIndex === null;
  btn.title = reviewed
    ? 'You have checked this frame. Click to put it back on the pile.'
    : 'Mark this frame as checked by you — export can then skip the rest.';
}

function updateSelectionInfo() {
  const count = app.editor?.selection.length || 0;
  dom.selectionInfo.hidden = count === 0;
  dom.selectionInfo.textContent = count === 1
    ? '1 box selected'
    : `${count} boxes selected — press M to merge`;
  dom.mergeBtn.disabled = count < 2;
}

/** Replace the selected boxes with one that covers them all. */
async function mergeSelected() {
  const indices = app.editor.selectionIndices();
  if (indices.length < 2) {
    toast('Select two or more boxes first — Shift-click them, or drag a lasso around them.', 'warn');
    return;
  }
  app.editor.pushUndo();
  await flushSave();
  try {
    const res = await api.mergeBoxes(app.currentIndex, indices);
    app.editor.setBoxes(res.boxes);
    app.state.stats = res.stats;
    updateBoxSummary();
    updateSelectionInfo();
    scheduleStrip();
    toast(`${indices.length} boxes merged into one.`, 'ok');
  } catch (err) {
    toast(err.message, 'error', 'Could not merge');
  }
}

function updateBoxSummary() {
  const boxes = app.editor?.boxes || [];
  const detail = app.frameDetail;
  const parts = [];
  parts.push(`${boxes.length} object${boxes.length === 1 ? '' : 's'}`);
  if (detail?.source) parts.push(detail.source.split(/[\\/]/).pop());
  if (detail?.width) parts.push(`${detail.width}×${detail.height}`);
  dom.boxSummary.textContent = parts.join('  ·  ');
}

function step(delta, extend = false) {
  const frames = visibleFrames();
  if (!frames.length) return;
  const at = frames.findIndex((f) => f.index === app.currentIndex);
  const next = frames[Math.min(frames.length - 1, Math.max(0, (at < 0 ? 0 : at) + delta))];
  if (!next) return;
  if (extend) app.frameSelection.add(next.index);
  selectFrame(next.index, { scroll: true, keepSelection: extend });
}

/** Strip the labels off every selected frame. */
async function clearSelectedFrames() {
  const frames = selectedFrames();
  if (!frames.length) return;

  if (frames.length === 1) {
    app.editor.clearAll();   // one frame stays undoable in the editor
    return;
  }
  const carrying = frames.filter((i) => (app.state.frames.find((f) => f.index === i)?.boxes || 0) > 0);
  if (!carrying.length) {
    toast('Those frames have no labels to clear.', 'warn');
    return;
  }
  const ok = await confirmDialog({
    title: `Clear labels on ${plural(frames.length, 'frame')}?`,
    message: `${plural(carrying.length, 'frame')} carry labels. The frames stay; their boxes are removed. This cannot be undone.`,
    confirmLabel: 'Clear them',
    danger: true,
  });
  if (!ok) return;

  await flushSave();
  try {
    const res = await api.clearLabels(frames);
    app.state.stats = res.stats;
    await refreshStats();
    await reloadCurrentFrame();
    toast(`Labels cleared on ${plural(frames.length, 'frame')}.`, 'ok');
  } catch (err) {
    toast(err.message, 'error', 'Could not clear');
  }
}

/** Remove every selected frame from the session. */
async function dropSelectedFrames() {
  const frames = selectedFrames();
  if (!frames.length) return;

  const ok = await confirmDialog({
    title: frames.length === 1 ? `Drop frame ${frames[0]}?` : `Drop ${plural(frames.length, 'frame')}?`,
    message: frames.length === 1
      ? 'The frame and its labels leave this session. The original file on disk is untouched.'
      : `Those ${frames.length} frames and their labels leave this session. The original files on disk are untouched.`,
    confirmLabel: frames.length === 1 ? 'Drop it' : 'Drop them',
    danger: true,
  });
  if (!ok) return;

  // Pick what to show afterwards: the first frame below the ones going away.
  const visible = visibleFrames().map((f) => f.index);
  const doomed = new Set(frames);
  const after = visible.find((i) => i > Math.max(...frames) && !doomed.has(i));
  const before = [...visible].reverse().find((i) => i < Math.min(...frames) && !doomed.has(i));
  const next = after ?? before ?? null;

  try {
    await api.deleteFrames(frames);
  } catch (err) {
    toast(err.message, 'error', 'Could not drop');
    return;
  }
  app.state.frames = app.state.frames.filter((f) => !doomed.has(f.index));
  app.frameSelection = new Set();
  app.currentIndex = null;
  renderStrip();

  if (next !== null) {
    selectFrame(next, { scroll: true });
  } else {
    app.frameDetail = null;
    app.editor.load(null, []);
    dom.stageEmpty.hidden = false;
    dom.frameLabel.textContent = '—';
    updateFrameActions();
    app.renderPanel();
  }
  toast(`${plural(frames.length, 'frame')} dropped.`, 'ok');
}

// ── Saving ─────────────────────────────────────────────────────

let saveTimer = null;
let pendingSave = null;

function queueSave(boxes) {
  pendingSave = { index: app.currentIndex, boxes: JSON.parse(JSON.stringify(boxes)) };
  dom.saveState.textContent = 'Saving…';
  clearTimeout(saveTimer);
  saveTimer = setTimeout(flushSave, 350);
}

async function flushSave() {
  clearTimeout(saveTimer);
  const job = pendingSave;
  pendingSave = null;
  if (!job) return;
  app.suppressReload = Date.now();
  try {
    const res = await api.putBoxes(job.index, job.boxes, 'verified');
    if (res.stats) app.state.stats = res.stats;
    dom.saveState.textContent = 'Saved';
    setTimeout(() => { if (dom.saveState.textContent === 'Saved') dom.saveState.textContent = ''; }, 1400);
  } catch (err) {
    dom.saveState.textContent = '';
    toast(err.message, 'error', 'Could not save labels');
  }
}

// ── Tools & HUD ────────────────────────────────────────────────

function maskMode() {
  return app.state.task === 'segment';
}

/**
 * Show or hide the rectangle drawn around an outlined object.
 *
 * Tracing a shape means watching its edge, and a bright rectangle following it
 * around competes for exactly that attention. Boxes with no outline keep their
 * rectangle regardless — it is all they have.
 */
function setWrapperBoxes(on) {
  app.ui.wrapperBoxes = on;
  app.editor.showWrappers = on;
  dom.wrapperBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
  dom.wrapperBtn.title = on
    ? 'Hide the box drawn around an outlined object'
    : 'Show the box drawn around an outlined object';
  app.editor.render();
  // Saved here rather than left to the next panel render: this is a toolbar
  // button and can easily be the last thing touched before the tab closes.
  saveUi();
}

function setTool(tool) {
  if ((tool === 'outline' || tool === 'snap') && !maskMode()) {
    toast('Switch “What you are labelling” to Outlines first.', 'warn');
    return;
  }
  app.editor.setTool(tool);
  for (const btn of dom.tools.querySelectorAll('.tool')) {
    btn.setAttribute('aria-pressed', btn.dataset.tool === tool ? 'true' : 'false');
  }
  prewarmSnap();
}

/**
 * Ask the server to read this frame before a box is drawn on it.
 *
 * Reading the picture is nearly all of what a snap costs, and it depends only
 * on the picture — so it can happen while the user is still choosing where to
 * drag.  Fire and forget: if it fails, the snap reads the frame itself exactly
 * as it used to.
 */
function prewarmSnap() {
  if (app.editor?.tool !== 'snap' || app.currentIndex === null) return;
  api.prewarm({ frame: app.currentIndex }).catch(() => {});
}

/** Outline and Snap are meaningless for box or pose datasets, so they hide. */
function syncMaskTools() {
  const masks = maskMode();
  for (const btn of dom.tools.querySelectorAll('.tool-mask')) btn.hidden = !masks;
  if (!masks && (app.editor?.tool === 'outline' || app.editor?.tool === 'snap')) {
    setTool('select');
  }
}


// ── Keyboard reference ─────────────────────────────────────────

const SHORTCUTS = [
  ['Tools', [
    ['V', 'Select and edit boxes'],
    ['B', 'Draw a box'],
    ['P', 'Trace an outline point by point (mask labelling)'],
    ['G', 'Snap an outline to the object in a rough box (mask labelling)'],
    ['E', 'Mark an example for SAM 3 to match'],
    ['X', 'Mark something to exclude'],
  ]],
  ['Outlines', [
    ['Click · Enter', 'Drop a point · close the outline you are tracing'],
    ['Right-click', 'Take back the last point while tracing'],
    ['Esc', 'Abandon the outline you are tracing'],
    ['Drag a point', 'Move it — the box follows the outline'],
    ['Click an edge', 'Add a point there'],
    ['Right-click a point', 'Remove it'],
    ['H', 'Show or hide the box around an outlined object'],
  ]],
  ['Selecting', [
    ['Click', 'Select one box'],
    ['Shift-click', 'Add a box to the selection, or take it out'],
    ['Drag on empty space', 'Lasso every box the rectangle touches'],
    ['Ctrl+A', 'Select every box on this frame'],
    ['Esc', 'Select nothing'],
  ]],
  ['Editing', [
    ['M', 'Merge the selected boxes into one'],
    ['Delete', 'Remove the selected boxes'],
    ['0 – 9', 'Put the selected boxes in that class'],
    ['Drag a box', 'Move it — moves the whole selection'],
    ['Drag a corner', 'Resize (one box at a time)'],
    ['Right-click a box', 'Delete it'],
    ['Right-click a keypoint', 'Remove it, or bring a removed one back'],
    ['Ctrl+Z', 'Undo on this frame'],
  ]],
  ['Frames', [
    ['Click a frame', 'Open it'],
    ['Shift-click a frame', 'Select everything between it and the current one'],
    ['Ctrl-click a frame', 'Add it to the selection, or take it out'],
    ['Shift + ← →', 'Extend the selection frame by frame'],
    ['Clear frames · Drop frames', 'Act on every selected frame'],
  ]],
  ['Getting around', [
    ['← →', 'Previous / next frame'],
    ['Wheel', 'Zoom'],
    ['Space-drag · middle-drag · Alt-drag', 'Pan'],
    ['F', 'Fit the image to the window'],
    ['Ctrl+S', 'Save the session now'],
    ['?', 'Show this list'],
  ]],
];

function showShortcuts() {
  const body = el('div', { class: 'shortcuts' },
    SHORTCUTS.map(([group, rows]) => el('div', { class: 'shortcut-group' }, [
      el('h4', { text: group }),
      ...rows.map(([keys, what]) => el('div', { class: 'shortcut' }, [
        el('span', { class: 'shortcut-keys' },
          keys.split(' · ').map((k) => el('kbd', { text: k }))),
        el('span', { text: what }),
      ])),
    ])));
  openModal({
    title: 'Keyboard shortcuts',
    subtitle: 'Keys work whenever the canvas has focus and you are not typing in a field.',
    body,
    actions: [{ label: 'Close', kind: 'primary', onClick: () => closeModal() }],
  });
}

// ── Job bar ────────────────────────────────────────────────────

function renderJobs() {
  const jobs = [...app.jobs.values()].filter((j) => j.status === 'running' || j.status === 'pending');
  dom.jobbar.hidden = jobs.length === 0;
  dom.jobbar.innerHTML = '';

  for (const job of jobs) {
    const pct = job.total > 0 ? Math.min(100, (job.current / job.total) * 100) : 0;
    const fill = el('div', { class: `job-fill ${job.total > 0 ? '' : 'indet'}`, style: `width:${job.total > 0 ? pct : 35}%` });

    const stop = el('button', {
      class: 'ghost job-stop',
      text: job.cancelRequested ? 'Stopping…' : 'Stop',
      disabled: !job.cancellable || job.cancelRequested ? 'disabled' : null,
      onclick: () => api.cancelJob(job.id).catch((err) => toast(err.message, 'error')),
    });

    const node = el('div', { class: `job ${job.kind}` }, [
      el('div', { class: 'job-top' }, [
        el('span', { class: 'job-label', text: job.label }),
        el('span', { class: 'job-msg', text: job.total > 0 ? `${job.message || ''}` : (job.message || 'Working…') }),
        stop,
      ]),
      el('div', { class: 'job-track' }, [fill]),
    ]);

    const lines = app.jobLogs.get(job.id);
    if (lines && lines.length && (job.kind === 'train' || job.kind === 'convert' || job.kind === 'script')) {
      const log = el('div', { class: 'joblog', text: lines.slice(-8).join('\n') });
      node.appendChild(log);
    }
    dom.jobbar.appendChild(node);
  }
}

function onJobFinished(job) {
  if (job.status === 'error') {
    toast(job.error || 'Something went wrong.', 'error', job.label);
    return;
  }
  const r = job.result || {};
  const stopped = job.status === 'cancelled';
  // Stopping does not undo: a run keeps every frame it reached, and one that
  // returned rather than raised handed back a report of exactly that.  Only
  // fall back to the bare message when there is genuinely nothing to show.
  if (stopped && !Object.keys(r).length) {
    toast('Stopped. Everything finished so far is kept.', 'warn', job.label);
    return;
  }
  const title = (done) => (stopped ? 'Stopped' : done);
  if (job.kind === 'import') {
    toast(`${plural(r.frames ?? 0, 'frame')} ready${r.labelled ? `, ${r.labelled} with labels from the dataset` : ''}.`, 'ok', title('Import complete'));
    refreshFrames();
  } else if (job.kind === 'autolabel') {
    toast(r.objects
      ? `${plural(r.objects, 'object')} found across ${plural(r.framesWithObjects, 'frame')}.`
      : 'No matches. Try a lower confidence, or describe the object differently.',
      r.objects ? 'ok' : 'warn', title('Auto-label complete'));
  } else if (job.kind === 'track') {
    toast(`${plural(r.objects ?? 0, 'object')} tracked through ${plural(r.frames ?? 0, 'frame')}.`, 'ok', title('Tracking complete'));
  } else if (job.kind === 'prompt') {
    toast(r.objects
      ? `${plural(r.objects, 'match', 'matches')} added to this frame.`
      : 'No matches — try another example box, or lower the confidence.',
      r.objects ? 'ok' : 'warn', 'Search complete');
  } else if (job.kind === 'snap') {
    toast(r.objects
      ? `Outline added — ${plural(r.points || 0, 'point')}.`
      : 'Nothing to outline there. Draw the box a little tighter around the object.',
      r.objects ? 'ok' : 'warn', 'Snap');
  } else if (job.kind === 'script') {
    app.scriptReport = r;
    toast(scriptOutcome(r, app.state.frames.length),
      (r.errors || []).length ? 'error' : (r.changed ? 'ok' : 'warn'), job.label);
  } else if (job.kind === 'train') {
    const where = r.best || r.save_dir || '';
    toast(where ? `Weights saved to ${where}` : 'Training finished.', 'ok', title('Training complete'));
  } else if (job.kind === 'convert') {
    toast(`${r.converted_items ?? 0} converted, ${r.fallback_items ?? 0} fell back, ${r.failed_items ?? 0} failed → ${r.output_dir || ''}`, 'ok', title('Conversion complete'));
  } else if (job.kind === 'onnx') {
    toast(`Saved to ${r.path}`, 'ok', title('ONNX export complete'));
  }
}

// ── Events ─────────────────────────────────────────────────────

function handleEvent(msg) {
  switch (msg.type) {
    case 'hello':
      applyState(msg.state);
      break;
    case 'job': {
      const job = msg.job;
      app.jobs.set(job.id, job);
      if (msg.event === 'finished') {
        onJobFinished(job);
        setTimeout(() => { app.jobs.delete(job.id); app.jobLogs.delete(job.id); renderJobs(); }, 600);
        if (['autolabel', 'track', 'prompt', 'snap', 'script'].includes(job.kind)) {
          reloadCurrentFrame();
          refreshStats();
        }
        if (job.kind === 'import') app.renderPanel();
      }
      renderJobs();
      break;
    }
    case 'log': {
      const lines = app.jobLogs.get(msg.jobId) || [];
      lines.push(msg.line);
      app.jobLogs.set(msg.jobId, lines.slice(-60));
      renderJobs();
      break;
    }
    case 'frames': {
      if (msg.event === 'added') {
        for (const frame of msg.frames) upsertFrame(frame);
        scheduleStrip();
        if (app.currentIndex === null && msg.frames.length) selectFrame(msg.frames[0].index);
      } else if (msg.event === 'cleared') {
        app.state.frames = [];
        app.currentIndex = null;
        app.frameDetail = null;
        app.editor.load(null, []);
        dom.stageEmpty.hidden = false;
        dom.frameLabel.textContent = '—';
        renderStrip();
        app.renderPanel();
      } else if (msg.event === 'removed') {
        app.state.frames = app.state.frames.filter((f) => !msg.indices.includes(f.index));
        renderStrip();
      } else if (msg.event === 'reload') {
        refreshFrames();
      }
      break;
    }
    case 'frame': {
      upsertFrame(msg.frame);
      scheduleStrip();
      if (msg.frame.index === app.currentIndex && Date.now() - app.suppressReload > 1200) {
        reloadCurrentFrame();
      }
      break;
    }
    case 'classes':
      app.state.classes = msg.classes;
      app.editor.setClassNames(msg.classes);
      renderClassSelect();
      app.renderPanel();
      break;
    case 'task':
      app.state.task = msg.task;
      syncMaskTools();
      app.renderPanel();
      break;
    case 'toast':
      toast(msg.message, msg.level === 'warn' ? 'warn' : msg.level === 'error' ? 'error' : 'info');
      break;
    default:
      break;
  }
}

function upsertFrame(meta) {
  const at = app.state.frames.findIndex((f) => f.index === meta.index);
  if (at >= 0) app.state.frames[at] = meta;
  else {
    app.state.frames.push(meta);
    app.state.frames.sort((a, b) => a.index - b.index);
  }
}

async function reloadCurrentFrame() {
  if (app.currentIndex === null) return;
  try {
    const detail = await api.frame(app.currentIndex);
    app.frameDetail = detail;
    app.editor.setBoxes(detail.boxes);
    updateBoxSummary();
    app.renderPanel();
  } catch { /* frame went away */ }
}

async function refreshFrames() {
  const data = await api.frames();
  app.state.frames = data.frames;
  app.state.stats = data.stats;
  renderStrip();
  if (app.currentIndex === null && data.frames.length) selectFrame(data.frames[0].index, { scroll: true });
  else if (app.currentIndex !== null) reloadCurrentFrame();
  app.renderPanel();
}

async function refreshStats() {
  try {
    const data = await api.frames();
    app.state.frames = data.frames;
    app.state.stats = data.stats;
    renderStrip();
    app.renderPanel();
  } catch { /* ignore */ }
}

function applyState(state) {
  Object.assign(app.state, state);
  nativePicker.available = !!state.nativeDialogs;
  nativePicker.enabled = app.ui.nativeDialogs !== false;
  app.editor.setClassNames(state.classes);
  renderClassSelect();
  renderStrip();
  renderDevice();
  syncMaskTools();
  for (const job of state.jobs || []) app.jobs.set(job.id, job);
  renderJobs();
  if (app.currentIndex === null && state.frames.length) {
    selectFrame(state.frames[0].index);
    app.goStep('label');
  } else {
    app.renderPanel();
  }
}

function renderClassSelect() {
  dom.activeClass.innerHTML = '';
  app.state.classes.forEach((name, i) => {
    dom.activeClass.appendChild(el('option', { value: String(i), text: `${i} · ${name}` }));
  });
  const id = Math.min(app.ui.classId, app.state.classes.length - 1);
  app.ui.classId = Math.max(0, id);
  dom.activeClass.value = String(app.ui.classId);
  app.editor.classId = app.ui.classId;
}

function renderDevice() {
  const d = app.state.device || {};
  if (d.cuda) {
    dom.device.textContent = `${d.gpu} · ${d.vram} GB`;
    dom.device.className = 'chip gpu';
    dom.device.title = `Running on GPU · PyTorch ${d.torch}`;
  } else {
    dom.device.textContent = 'CPU only';
    dom.device.className = 'chip cpu';
    dom.device.title = d.error || 'No CUDA GPU detected — SAM 3 and training will be slow.';
  }
}

// ── Wiring ─────────────────────────────────────────────────────

function bind() {
  dom.steps.addEventListener('click', (e) => {
    const btn = e.target.closest('.step');
    if (btn) app.goStep(btn.dataset.step);
  });

  dom.tools.addEventListener('click', (e) => {
    const btn = e.target.closest('.tool');
    if (btn) setTool(btn.dataset.tool);
  });

  dom.activeClass.addEventListener('change', () => app.setClass(+dom.activeClass.value));
  dom.filter.value = app.ui.filter;
  // Assigning a value the select does not offer leaves it showing blank,
  // so a setting remembered from an older build falls back to All frames.
  if (!dom.filter.value) { dom.filter.value = 'all'; app.ui.filter = 'all'; }
  dom.filter.addEventListener('change', () => { app.ui.filter = dom.filter.value; saveUi(); renderStrip(); });

  document.getElementById('prevFrame').onclick = () => step(-1);
  document.getElementById('nextFrame').onclick = () => step(1);
  document.getElementById('zoomIn').onclick = () => app.editor.zoomBy(1.25);
  document.getElementById('zoomOut').onclick = () => app.editor.zoomBy(1 / 1.25);
  document.getElementById('zoomFit').onclick = () => app.editor.fit();
  dom.wrapperBtn.onclick = () => setWrapperBoxes(!app.ui.wrapperBoxes);
  document.getElementById('undoBtn').onclick = () => { if (!app.editor.undo()) toast('Nothing left to undo on this frame.', 'warn'); };
  dom.clearFrameBtn.onclick = () => clearSelectedFrames();
  dom.mergeBtn.onclick = () => mergeSelected();
  document.getElementById('helpBtn').onclick = () => showShortcuts();
  document.getElementById('reviewBtn').onclick = async () => {
    if (app.currentIndex === null) return;
    const reviewed = app.frameDetail?.status === 'verified';
    const res = await api.putStatus(app.currentIndex, reviewed ? 'pending' : 'verified');
    app.frameDetail.status = res.frame.status;
    app.state.stats = res.stats;
    updateReviewButton();
    scheduleStrip();
  };
  dom.dropFrameBtn.onclick = () => dropSelectedFrames();

  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName);
    if (typing) return;

    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      app.editor.undo();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      flushSave().then(() => api.saveSession()).then(() => toast('Session saved.', 'ok'));
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'a') {
      e.preventDefault();
      app.editor.selectAll();
      return;
    }
    switch (e.key) {
      case 'ArrowLeft': case 'ArrowUp': e.preventDefault(); step(-1, e.shiftKey); break;
      case 'ArrowRight': case 'ArrowDown': e.preventDefault(); step(1, e.shiftKey); break;
      case 'Delete': case 'Backspace': app.editor.deleteSelected(); break;
      case 'v': case 'V': setTool('select'); break;
      case 'p': case 'P': setTool('outline'); break;
      case 'g': case 'G': setTool('snap'); break;
      case 'Enter': if (app.editor.pending) { e.preventDefault(); app.editor.finishOutline(); } break;
      case 'b': case 'B': setTool('draw'); break;
      case 'e': case 'E': setTool('pos'); break;
      case 'x': case 'X': setTool('neg'); break;
      case 'f': case 'F': app.editor.fit(); break;
      case 'h': case 'H': setWrapperBoxes(!app.ui.wrapperBoxes); break;
      case 'm': case 'M': mergeSelected(); break;
      case 'Escape':
        if (!app.editor.cancelOutline()) app.editor.select(null);
        break;
      case '?': showShortcuts(); break;
      case ' ':
        e.preventDefault();
        app.editor.setSpaceHeld(true);
        break;
      default:
        if (/^[0-9]$/.test(e.key)) {
          const id = e.key === '0' ? 9 : Number(e.key) - 1;
          if (id < app.state.classes.length) app.setClass(id);
        }
    }
  });

  document.addEventListener('keyup', (e) => {
    if (e.key === ' ') app.editor.setSpaceHeld(false);
  });
  window.addEventListener('blur', () => app.editor?.setSpaceHeld(false));

  window.addEventListener('beforeunload', () => { flushSave(); saveUi(); });
}

// ── Start ──────────────────────────────────────────────────────

async function main() {
  app.editor = new Editor(dom.canvas, dom.host);
  app.editor.onChange = (boxes) => { queueSave(boxes); updateBoxSummary(); scheduleStrip(); };
  app.editor.onSelect = (selection) => {
    const boxes = selection || [];
    const primary = boxes[0];
    if (primary) {
      dom.activeClass.value = String(primary.class_id);
      app.ui.classId = primary.class_id;
      app.editor.classId = primary.class_id;
    }
    updateSelectionInfo();
  };
  app.editor.onSnap = async (rect) => {
    if (app.currentIndex === null) return;
    try {
      await api.snap({
        frame: app.currentIndex,
        box: [rect.x1, rect.y1, rect.x2, rect.y2],
        classId: app.ui.classId,
      });
    } catch (err) {
      toast(err.message, 'error', 'Could not outline');
    }
  };

  app.editor.onExemplars = (positive, negative) => {
    app.examples = {
      positive: positive.map((r) => [r.x1, r.y1, r.x2, r.y2]),
      negative: negative.map((r) => [r.x1, r.y1, r.x2, r.y2]),
    };
    app.renderPanel();
  };

  bind();
  setTool('select');
  setWrapperBoxes(app.ui.wrapperBoxes !== false);
  app.goStep('media');

  try {
    app.models = await api.models();
    const state = await api.state();
    applyState(state);
  } catch (err) {
    toast(err.message, 'error', 'Could not reach the server');
  }

  connect(handleEvent, (online) => {
    if (!online) dom.saveState.textContent = 'Reconnecting…';
    else if (dom.saveState.textContent === 'Reconnecting…') dom.saveState.textContent = '';
  });
}

main();
