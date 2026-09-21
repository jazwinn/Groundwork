// The three step panels. Everything the old Qt side panel did, reorganised so
// each auto-label method states plainly what it does and when to use it.

import { api } from './api.js';
import { classColor } from './canvas.js';
import { confirmDialog, el, openModal, closeModal, nativePicker, pickPath, plural, toast } from './ui.js';

// ── small builders ─────────────────────────────────────────────

function append(parent, children) {
  for (const child of children) if (child) parent.appendChild(child);
}

/**
 * A section that folds away. The header keeps a one-line summary so a closed
 * section still tells you what it is set to, and the open/closed state is
 * remembered — the panel rebuilds constantly, and sections must not spring
 * back open every time.
 */
function disclosure(app, { id, title, summary = '', defaultOpen = false, children = [] }) {
  const open = app.ui.open?.[id] ?? defaultOpen;
  const head = el('button', {
    class: 'disclosure-head',
    'aria-expanded': String(open),
    onclick: () => {
      app.ui.open = { ...(app.ui.open || {}), [id]: !open };
      app.renderPanel();
    },
  }, [
    el('span', { class: 'disclosure-chevron', text: '›' }),
    el('span', { class: 'disclosure-title', text: title }),
    summary ? el('span', { class: 'disclosure-summary', text: summary }) : null,
  ]);

  const node = el('section', { class: 'disclosure', 'data-open': open ? 'true' : 'false' }, [head]);
  if (open) node.appendChild(el('div', { class: 'disclosure-body' }, children));
  return node;
}

function field(label, input, hint) {
  return el('label', { class: 'field' }, [
    el('span', { text: label }),
    input,
    hint ? el('span', { class: 'block-note', text: hint }) : null,
  ]);
}

function num(value, { min = 0, max = 999999, step = 1, id = null } = {}) {
  return el('input', { type: 'number', value: String(value), min, max, step, id });
}

function select(options, value, id) {
  const node = el('select', { id });
  for (const opt of options) {
    const [val, label] = Array.isArray(opt) ? opt : [opt, opt];
    node.appendChild(el('option', { value: val, text: label, selected: val === value ? 'selected' : null }));
  }
  node.value = value;
  return node;
}

/**
 * One augmentation setting, as a slider that says what its number means.
 *
 * The readout updates as the slider moves but the panel is not rebuilt, because
 * rebuilding mid-drag takes the slider out from under the pointer. The summary
 * on the closed section catches up the next time the panel renders.
 */
function augSlider(app, spec) {
  const value = augValue(app, spec);
  const input = el('input', {
    type: 'range',
    min: String(spec.min), max: String(spec.max), step: String(spec.step),
    value: String(value),
  });
  const readout = el('span', { class: 'block-note', text: augReadout(spec, value) });

  input.addEventListener('input', () => { readout.textContent = augReadout(spec, +input.value); });
  input.addEventListener('change', () => {
    app.ui.augment = { ...(app.ui.augment || {}), [spec.key]: +input.value };
  });

  return el('label', { class: 'field' }, [
    el('span', { text: spec.label }),
    input,
    readout,
    spec.hint ? el('span', { class: 'block-note dim', text: spec.hint }) : null,
  ]);
}

function augValue(app, spec) {
  const set = app.ui.augment || {};
  return set[spec.key] === undefined ? spec.default : set[spec.key];
}

/** Say what the number does, not just what it is. */
function augReadout(spec, value) {
  const isDefault = Math.abs(value - spec.default) < 1e-9;
  const tail = isDefault ? ' · default' : '';
  if (value === 0 && spec.format !== 'epochs') return `off${tail}`;
  switch (spec.format) {
    case 'chance':  return `${Math.round(value * 100)}% of images${tail}`;
    case 'degrees': return `up to ±${value}°${tail}`;
    case 'epochs':  return value === 0 ? `never${tail}` : `for the last ${plural(value, 'epoch')}${tail}`;
    default:        return `±${value}${tail}`;
  }
}

function pathField(state, key, { placeholder, onPick }) {
  const label = el('button', {
    class: `picker-path ${state[key] ? 'set' : ''}`,
    title: state[key] || placeholder,
    text: state[key] || placeholder,
    onclick: onPick,
  });
  return el('div', { class: 'picker-row' }, [label, el('button', { class: 'ghost', text: 'Browse', onclick: onPick })]);
}

// ── 1 · Media ──────────────────────────────────────────────────

export function mediaPanel(app) {
  const stride = num(app.ui.importStride, { min: 1, max: 120 });
  stride.addEventListener('change', () => { app.ui.importStride = Math.max(1, +stride.value || 1); });

  const startImport = async (mode) => {
    const path = await pickPath({
      title: mode === 'dataset' ? 'Open a YOLO dataset' : 'Choose a media folder',
      subtitle: mode === 'dataset'
        ? 'Pick the folder holding images/, labels/ and data.yaml.'
        : 'Every image and video inside the folder is imported.',
      want: 'media',
      confirmLabel: 'Use this folder',
      describe: mode === 'dataset' ? async (p) => {
        try {
          const info = await api.inspect(p);
          if (!info.isDataset) return 'No labels/ or data.yaml here — images would be imported without labels.';
          const classes = info.classes?.length ? ` · ${info.classes.length} classes` : '';
          return `YOLO dataset · task ${info.task || 'detect'}${classes}`;
        } catch { return null; }
      } : null,
    });
    if (!path) return;

    if (mode === 'dataset') {
      let info = null;
      try { info = await api.inspect(path); } catch { /* let the import report it */ }
      if (info && !info.isDataset) {
        const ok = await confirmDialog({
          title: 'That folder has no labels',
          message: 'There is no labels/ folder or data.yaml here, so only the images can come in. Import them anyway?',
          confirmLabel: 'Import the images',
        });
        if (!ok) return;
      }
    }

    if (mode === 'replace' && app.state.frames.length) {
      const ok = await confirmDialog({
        title: 'Replace everything?',
        message: `This clears the ${app.state.frames.length} frames you have loaded, along with their labels, and imports the new folder instead.`,
        confirmLabel: 'Replace',
        danger: true,
      });
      if (!ok) return;
    }

    try {
      await api.import({ path, mode, stride: Math.max(1, +stride.value || 1) });
      app.goStep('label');
    } catch (err) {
      toast(err.message, 'error', 'Import failed');
    }
  };

  const stats = app.state.stats || {};

  return el('div', { class: 'panel-inner' }, [
    el('div', { class: 'block' }, [
      el('h3', {}, ['Bring in your media']),
      el('p', { class: 'block-note', text: 'Images are kept as they are. Videos are split into frames.' }),
      el('button', { class: 'btn primary block-btn', text: 'Import images or video', onclick: () => startImport('replace') }),
      el('button', { class: 'btn block-btn', text: 'Add more to what is loaded', onclick: () => startImport('add') }),
      el('button', { class: 'btn block-btn', text: 'Open an existing YOLO dataset', onclick: () => startImport('dataset') }),
      el('p', { class: 'block-note', text: 'Opening a dataset also loads its class names and existing boxes.' }),
    ]),

    disclosure(app, {
      id: 'sampling',
      title: 'Video frames',
      summary: app.ui.importStride > 1 ? `1 in ${app.ui.importStride}` : 'every frame',
      children: [
        field('Keep 1 frame in every', stride,
          'A 30 fps clip at 5 gives you 6 frames per second — plenty for labelling, and far fewer files. Tracking needs 1.'),
      ],
    }),

    nativePicker.available ? disclosure(app, {
      id: 'picker',
      title: 'Choosing folders',
      summary: app.ui.nativeDialogs !== false ? 'Windows dialog' : 'built-in browser',
      children: [
        (() => {
          const box = el('input', { type: 'checkbox', ...(app.ui.nativeDialogs !== false ? { checked: 'checked' } : {}) });
          box.addEventListener('change', () => {
            app.ui.nativeDialogs = box.checked;
            nativePicker.enabled = box.checked;
            app.renderPanel();
          });
          return el('label', { class: 'inline-field' }, [box, 'Use the Windows file dialog']);
        })(),
        el('p', { class: 'block-note', text: app.ui.nativeDialogs !== false
          ? 'Browse buttons open Explorer, the same dialog every other Windows app uses.'
          : 'Browse buttons open the folder browser built into this page.' }),
      ],
    }) : null,

    disclosure(app, {
      id: 'session',
      title: 'This session',
      summary: `${stats.frames ?? 0} frames · ${stats.boxes ?? 0} objects`,
      defaultOpen: true,
      children: [
        el('div', { class: 'stats' }, [
          el('div', { class: 'stat' }, [el('b', { text: String(stats.frames ?? 0) }), el('span', { text: 'frames' })]),
          el('div', { class: 'stat' }, [el('b', { text: String(stats.labelled ?? 0) }), el('span', { text: 'labelled' })]),
          el('div', { class: 'stat' }, [el('b', { text: String(stats.boxes ?? 0) }), el('span', { text: 'objects' })]),
        ]),
        el('p', { class: 'block-note', text: 'Your work saves itself and comes back when you reopen Groundwork.' }),
        el('button', {
          class: 'btn danger block-btn',
          text: 'Clear this session',
          onclick: async () => {
            const ok = await confirmDialog({
              title: 'Clear everything?',
              message: 'Every frame and label in this session is removed. Exported datasets on disk are untouched.',
              confirmLabel: 'Clear session',
              danger: true,
            });
            if (ok) { await api.resetSession(); toast('Session cleared.', 'ok'); }
          },
        }),
      ],
    }),
  ]);
}

// ── 2 · Label ──────────────────────────────────────────────────

const METHODS = [
  {
    id: 'describe',
    title: 'Describe what to find',
    sub: 'Type what the object is. SAM 3 finds every match, frame by frame.',
    when: 'Best when you can name the thing: "car", "solar panel", "yellow helmet".',
  },
  {
    id: 'example',
    title: 'Point at an example',
    sub: 'Box one object on this frame; SAM 3 finds the others like it here.',
    when: 'Best when the object is hard to name, or a description brings back the wrong things.',
  },
  {
    id: 'reuse',
    title: 'Reuse this frame’s labels',
    sub: 'Takes the classes you used here and searches for them in the other frames.',
    when: 'Best for photo sets and changing scenes, where each frame stands on its own.',
  },
  {
    id: 'track',
    title: 'Track through the video',
    sub: 'Follows the exact objects you boxed here into the frames that come after.',
    when: 'Best for continuous video of the same moving objects. Needs frames imported at stride 1.',
  },
];

export function labelPanel(app) {
  const frame = app.currentFrame();
  const detail = app.frameDetail;
  const boxCount = detail?.boxes?.length ?? 0;

  // ── Classes ──
  const classTags = el('div', { class: 'classlist' },
    app.state.classes.map((name, i) => el('span', { class: 'classtag' }, [
      el('i', { style: `background:${classColor(i)}` }),
      el('span', { text: name }),
    ])));

  const taskNames = { detect: 'Boxes', segment: 'Outlines', pose: 'Corner points' };
  const task = select([
    ['detect', 'Boxes — a rectangle per object'],
    ['segment', 'Outlines — the shape of each object'],
    ['pose', 'Corner points — four points per object'],
  ], app.state.task);
  task.addEventListener('change', async () => {
    await api.settings({ task: task.value });
    app.state.task = task.value;
    app.onTaskChange();
  });

  const detailSelect = select(
    (app.state.maskDetails || ['coarse', 'medium', 'fine']).map((d) => [d, {
      coarse: 'Coarse — straight edges, fewest points',
      medium: 'Medium — follows most shapes',
      fine: 'Fine — keeps every wobble',
    }[d] || d]),
    app.state.maskDetail || 'medium',
  );
  detailSelect.addEventListener('change', () => api.settings({ maskDetail: detailSelect.value }));

  const taskBlock = disclosure(app, {
    id: 'task',
    title: 'What you are labelling',
    summary: taskNames[app.state.task] || app.state.task,
    defaultOpen: true,
    children: [
      task,
      app.state.task === 'segment' ? field('Outline detail', detailSelect,
        'How closely an outline follows the mask. Panels and other straight-edged things need only a few points.') : null,
      app.state.task === 'segment'
        ? el('p', { class: 'block-note', text: 'Auto-label and tracking now record outlines, and the Outline and Snap tools appear in the toolbar.' })
        : el('p', { class: 'block-note', text: 'This drives both the tools you get and the format Export writes.' }),
    ].filter(Boolean),
  });

  const classBlock = disclosure(app, {
    id: 'classes',
    title: 'Classes',
    summary: app.state.classes.length === 1 ? app.state.classes[0] : `${app.state.classes.length} defined`,
    defaultOpen: true,
    children: [
      classTags,
      el('button', { class: 'btn block-btn', text: 'Edit classes', onclick: () => editClasses(app) }),
    ],
  });

  // ── Shared auto-label settings ──
  const samModel = select(app.state.samModels, app.state.samModel);
  samModel.addEventListener('change', () => api.settings({ samModel: samModel.value }));

  const threshold = el('input', {
    type: 'range', min: '0.05', max: '0.95', step: '0.05', value: String(app.state.threshold),
  });
  const thresholdOut = el('span', { class: 'block-note', text: confidenceLabel(app.state.threshold) });
  threshold.addEventListener('input', () => { thresholdOut.textContent = confidenceLabel(+threshold.value); });
  threshold.addEventListener('change', () => api.settings({ threshold: +threshold.value }));

  const policy = select([
    ['replace', 'Replace them'],
    ['merge', 'Keep them and add'],
    ['skip', 'Leave that frame alone'],
  ], app.ui.policy);
  policy.addEventListener('change', () => { app.ui.policy = policy.value; app.renderPanel(); });

  const policyWord = { replace: 'replace', merge: 'add to', skip: 'skip' }[app.ui.policy] || app.ui.policy;
  const settingsBlock = disclosure(app, {
    id: 'autolabel-settings',
    title: 'Auto-label settings',
    summary: `${app.state.samModel.split(' (')[0]} · ${Math.round(app.state.threshold * 100)}% · ${policyWord}`,
    children: [
      field('Model', samModel),
      field('Confidence', threshold, null),
      thresholdOut,
      field('If a frame already has labels', policy,
        app.ui.policy === 'replace'
          ? 'Careful: boxes you drew by hand on those frames are overwritten.'
          : app.ui.policy === 'skip'
            ? 'Frames you have already labelled are skipped entirely.'
            : 'New findings are added next to what is already there.'),
    ],
  });

  // ── Methods ──
  const methods = el('div', { class: 'methods' },
    METHODS.map((method) => methodCard(app, method, { frame, detail, boxCount })));

  const chosen = METHODS.find((m) => m.id === app.ui.method);
  const methodBlock = disclosure(app, {
    id: 'autolabel',
    title: 'Label automatically',
    summary: chosen ? chosen.title : '',
    defaultOpen: true,
    children: [
      el('p', { class: 'block-note', text: 'Pick one way to work. Each one runs on its own and can be stopped at any time.' }),
      methods,
    ],
  });

  return el('div', { class: 'panel-inner' }, [
    taskBlock,
    classBlock,
    methodBlock,
    settingsBlock,
    tidyBlock(app, boxCount),
    scriptBlock(app),
  ]);
}

// Auto-labelling often stacks several boxes on one object. This folds them back
// into one without having to click through every frame by hand.
function tidyBlock(app, boxCount) {
  const ui = app.ui;

  const threshold = el('input', {
    type: 'range', min: '0.3', max: '1', step: '0.05', value: String(ui.mergeThreshold),
  });
  const readout = el('p', { class: 'block-note', text: overlapLabel(ui.mergeThreshold) });
  threshold.addEventListener('input', () => { readout.textContent = overlapLabel(+threshold.value); });
  threshold.addEventListener('change', () => { ui.mergeThreshold = +threshold.value; });

  const sameClass = el('input', {
    type: 'checkbox', ...(ui.mergeSameClass !== false ? { checked: 'checked' } : {}),
  });
  sameClass.addEventListener('change', () => { ui.mergeSameClass = sameClass.checked; });

  const run = async (scope) => {
    const payload = {
      scope,
      frame: app.currentIndex,
      threshold: +threshold.value,
      sameClassOnly: sameClass.checked,
    };
    if (scope === 'all') {
      const ok = await confirmDialog({
        title: 'Merge across every frame?',
        message: `Any boxes overlapping by ${Math.round(+threshold.value * 100)}% or more become one box, on all ${app.state.frames.length} frames. Undo only covers the frame you are looking at, so this is worth exporting before.`,
        confirmLabel: 'Merge them',
      });
      if (!ok) return;
    }
    try {
      const res = await api.mergeOverlaps(payload);
      app.state.stats = res.stats;
      await app.reloadFrame();
      toast(res.removed
        ? `${plural(res.removed, 'box', 'boxes')} folded away across ${plural(res.frames, 'frame')}.`
        : 'Nothing overlapped that much — try a lower percentage.',
        res.removed ? 'ok' : 'warn', 'Merge overlapping');
    } catch (err) {
      toast(err.message, 'error', 'Could not merge');
    }
  };

  return disclosure(app, {
    id: 'tidy',
    title: 'Tidy up boxes',
    summary: `${Math.round(ui.mergeThreshold * 100)}% overlap`,
    children: [
      el('p', { class: 'block-note', text: 'Select boxes on the image and press M to merge them by hand, or fold overlapping duplicates together here.' }),
      field('Merge when they overlap by', threshold),
      readout,
      el('label', { class: 'inline-field' }, [sameClass, 'Only merge boxes of the same class']),
      el('button', {
        class: 'btn block-btn',
        text: 'Merge on this frame',
        disabled: boxCount > 1 ? null : 'disabled',
        onclick: () => run('frame'),
      }),
      el('button', {
        class: 'btn block-btn',
        text: `Merge across all ${app.state.frames.length} frames`,
        disabled: app.state.frames.length ? null : 'disabled',
        onclick: () => run('all'),
      }),
    ],
  });
}

function overlapLabel(value) {
  const pct = Math.round(value * 100);
  if (value >= 0.95) return `${pct}% — only boxes sitting almost exactly on top of each other`;
  if (value <= 0.5) return `${pct}% — merges freely, and will join neighbours that only touch`;
  return `${pct}% — when this much of the smaller box is inside the bigger one`;
}

function confidenceLabel(value) {
  const pct = Math.round(value * 100);
  if (value <= 0.3) return `${pct}% — more objects found, more mistakes`;
  if (value >= 0.7) return `${pct}% — only confident matches`;
  return `${pct}% — balanced`;
}

function methodCard(app, method, ctx) {
  const open = app.ui.method === method.id;
  const head = el('button', { class: 'method-head', onclick: () => { app.ui.method = method.id; app.onMethodChange(); } }, [
    el('span', { class: 'method-dot' }),
    el('span', {}, [
      el('div', { class: 'method-title', text: method.title }),
      el('div', { class: 'method-sub', text: method.sub }),
    ]),
  ]);

  const card = el('div', { class: 'method', 'data-open': open ? 'true' : 'false' }, [head]);
  if (open) card.appendChild(methodBody(app, method, ctx));
  return card;
}

function methodBody(app, method, { frame, detail, boxCount }) {
  const body = el('div', { class: 'method-body' }, [el('p', { class: 'method-when', text: method.when })]);
  const ui = app.ui;

  if (method.id === 'describe') {
    const concept = el('input', { type: 'text', placeholder: 'e.g. solar panel', value: ui.concept || '' });
    concept.addEventListener('input', () => { ui.concept = concept.value; });
    const cls = select(app.state.classes.map((n, i) => [String(i), n]), String(ui.classId));
    cls.addEventListener('change', () => app.setSearchClass(+cls.value));
    const start = num(ui.start, { min: 0 });
    const every = num(ui.stride, { min: 1, max: 120 });
    const count = () => countFrames(app, +start.value, +every.value);

    const run = el('button', { class: 'btn primary block-btn', text: `Search ${count()} frames` });
    const refresh = () => { run.textContent = `Search ${count()} frames`; };
    start.addEventListener('input', refresh);
    every.addEventListener('input', refresh);

    run.onclick = async () => {
      ui.start = +start.value; ui.stride = +every.value;
      try {
        await api.describe({
          concept: concept.value.trim(),
          classId: +cls.value,
          start: +start.value,
          stride: +every.value,
          policy: ui.policy,
        });
      } catch (err) { toast(err.message, 'error', 'Could not start'); }
    };

    body.append(
      field('What to find', concept, 'Leave blank to search for the class name.'),
      field('Save matches as', cls),
      el('div', { class: 'field-row' }, [
        field('Start at frame', start),
        field('Every Nth frame', every),
      ]),
      run,
    );
  }

  if (method.id === 'example') {
    const tally = el('div', { class: 'tally' }, [
      el('span', { class: 'pos' }, [el('b', { text: String(app.examples.positive.length) }), ' to find']),
      el('span', { class: 'neg' }, [el('b', { text: String(app.examples.negative.length) }), ' to exclude']),
    ]);

    const text = el('input', { type: 'text', placeholder: 'optional: also describe it', value: ui.promptText || '' });
    text.addEventListener('input', () => { ui.promptText = text.value; });

    const run = el('button', {
      class: 'btn primary block-btn',
      text: 'Find matches on this frame',
      disabled: (!app.examples.positive.length && !(ui.promptText || '').trim()) || !frame ? 'disabled' : null,
    });
    run.onclick = async () => {
      try {
        await api.prompt({
          frame: app.currentIndex,
          positive: app.examples.positive,
          negative: app.examples.negative,
          text: text.value.trim(),
          classId: ui.classId,
          policy: 'merge',
        });
      } catch (err) { toast(err.message, 'error', 'Could not start'); }
    };

    body.append(
      el('ol', { class: 'steplist' }, [
        el('li', {}, [el('b', { text: 'Example' }), ' in the toolbar, then drag a box around one object you want.']),
        el('li', {}, [el('b', { text: 'Exclude' }), ' and drag around anything it should not match (optional).']),
        el('li', {}, ['Run it. Matches land on this frame as ', el('b', { text: 'new boxes' }), '.']),
      ]),
      tally,
      field('Describe it too', text, 'A word or two sharpens the search when examples alone are ambiguous.'),
      run,
      el('button', { class: 'btn block-btn', text: 'Clear examples', onclick: () => app.clearExamples() }),
      el('p', { class: 'block-note', text: 'This works on the frame you are looking at. To spread labels, use one of the other methods afterwards.' }),
    );
  }

  if (method.id === 'reuse') {
    const start = num(ui.start, { min: 0 });
    const names = detail?.boxes?.length
      ? [...new Set(detail.boxes.map((b) => app.state.classes[b.class_id] || `class ${b.class_id}`))]
      : [];

    const run = el('button', {
      class: 'btn primary block-btn',
      text: `Copy to ${Math.max(0, countFrames(app, +start.value, 1) - 1)} frames`,
      disabled: boxCount ? null : 'disabled',
    });
    start.addEventListener('input', () => {
      run.textContent = `Copy to ${Math.max(0, countFrames(app, +start.value, 1) - 1)} frames`;
    });
    run.onclick = async () => {
      ui.start = +start.value;
      try {
        await api.propagate({ seed: app.currentIndex, start: +start.value, policy: ui.policy });
      } catch (err) { toast(err.message, 'error', 'Could not start'); }
    };

    body.append(
      boxCount
        ? el('p', { class: 'block-note' }, [`Searches other frames for: `, el('b', { text: names.join(', ') })])
        : el('p', { class: 'block-note', text: 'Label this frame first — its classes become the search terms.' }),
      field('Start at frame', start, 'Useful after adding new media: skip the frames you already did.'),
      run,
    );
  }

  if (method.id === 'track') {
    const range = num(ui.trackRange, { min: 0, max: 100000 });
    const isVideo = detail?.isVideo;
    const stride = detail?.importStride ?? 1;

    const run = el('button', {
      class: 'btn primary block-btn',
      text: frame ? `Track forward from frame ${app.currentIndex}` : 'Track forward',
      disabled: boxCount && isVideo ? null : 'disabled',
    });
    run.onclick = async () => {
      ui.trackRange = +range.value;
      try {
        const res = await api.track({ seed: app.currentIndex, range: +range.value, policy: ui.policy });
        if (res.needsConfirm) {
          const ok = await confirmDialog({
            title: 'Frame gaps may break tracking',
            message: res.reason,
            confirmLabel: 'Track anyway',
            danger: true,
          });
          if (ok) await api.track({ seed: app.currentIndex, range: +range.value, policy: ui.policy, confirm: true });
        }
      } catch (err) { toast(err.message, 'error', 'Could not start'); }
    };

    // Node.append() stringifies null, so drop the branches that produced none.
    append(body, [
      !isVideo
        ? el('p', { class: 'block-note', text: 'This frame came from a still image, so there is no video to follow. Use “Reuse this frame’s labels” instead.' })
        : stride > 1
          ? el('p', { class: 'block-note', text: `This video was imported keeping 1 frame in every ${stride}. The tracker works best at stride 1.` })
          : null,
      !boxCount ? el('p', { class: 'block-note', text: 'Draw a box around each object you want followed, then run this.' }) : null,
      field('How many frames forward', range, '0 follows the objects to the end of the video.'),
      run,
    ]);
  }

  return body;
}

function countFrames(app, start, stride) {
  const step = Math.max(1, stride || 1);
  return app.state.frames.filter((f) => f.index >= (start || 0) && f.index % step === 0).length;
}

function editClasses(app) {
  const input = el('textarea', { rows: '6', spellcheck: 'false' });
  input.value = app.state.classes.join('\n');
  openModal({
    title: 'Edit classes',
    subtitle: 'One class per line. The order sets the class id used in the exported labels.',
    body: el('div', { class: 'block' }, [
      input,
      el('p', { class: 'block-note', text: 'Renaming a class keeps existing boxes pointing at the same row.' }),
    ]),
    actions: [
      { label: 'Cancel', onClick: () => closeModal() },
      {
        label: 'Save classes',
        kind: 'primary',
        onClick: async () => {
          const names = input.value.split('\n').map((s) => s.trim()).filter(Boolean);
          if (!names.length) { toast('Add at least one class.', 'warn'); return; }
          await api.setClasses(names);
          closeModal();
        },
      },
    ],
  });
}

// ── Your own scripts ───────────────────────────────────────────
//
// "Tidy up boxes" above is one rule, fixed. This is the same shape with the
// rule left to you: whatever .py files are in the scripts folder show up here,
// each with the controls it asked for, to run on this frame or on all of them.

function scriptBlock(app) {
  if (app.scriptsState === 'idle') app.refreshScripts();

  const list = app.state.scripts || [];
  const runnable = list.filter((s) => !s.error);
  const broken = list.filter((s) => s.error);
  const chosen = runnable.find((s) => s.id === app.ui.script) || runnable[0] || null;
  const frames = app.state.frames.length;
  const body = [];

  if (!list.length) {
    body.push(el('p', { class: 'block-note', text: app.scriptsState === 'loading'
      ? 'Reading the scripts folder…'
      : 'The scripts folder is empty. A .py file with a NAME and an apply(frame, p) function appears here, along with whatever settings it declares.' }));
  } else if (!chosen) {
    body.push(el('p', { class: 'block-note', text: 'None of the files in the scripts folder will load. What is wrong with each one is below.' }));
  } else {
    const picker = select(runnable.map((s) => [s.id, s.name]), chosen.id);
    picker.addEventListener('change', () => {
      app.ui.script = picker.value;
      app.scriptReport = null;
      app.renderPanel();
    });
    body.push(field('Script', picker, chosen.description));

    for (const spec of chosen.params) body.push(scriptControl(app, chosen, spec));

    const run = async (scope) => {
      if (scope === 'all') {
        const ok = await confirmDialog({
          title: `Run ${chosen.name} on every frame?`,
          message: `It decides the labels on all ${plural(frames, 'frame')}. Undo only covers the frame you are looking at, so a copy of the session is saved to .groundwork/session/backups first — putting that file back is how this is undone.`,
          confirmLabel: 'Run it',
        });
        if (!ok) return;
      } else {
        await app.beforeFrameEdit();     // so Ctrl+Z still puts the frame back
      }
      try {
        const res = await api.runScript({
          scriptId: chosen.id,
          scope,
          frame: app.currentIndex,
          params: scriptValues(app, chosen),
        });
        if (scope === 'all') {
          app.scriptReport = null;       // the finished job brings the real one
          app.renderPanel();
          return;
        }
        app.scriptReport = res;
        if (res.stats) app.state.stats = res.stats;
        await app.reloadFrame();
        toast(scriptOutcome(res, 1),
          res.errors.length ? 'error' : (res.changed ? 'ok' : 'warn'), chosen.name);
      } catch (err) {
        toast(err.message, 'error', `Could not run ${chosen.name}`);
      }
    };

    body.push(el('button', {
      class: 'btn block-btn',
      text: 'Run on this frame',
      title: chosen.canFrame ? '' : `${chosen.name} only defines apply_all, so it runs across every frame rather than on one.`,
      disabled: (chosen.canFrame && app.currentIndex !== null) ? null : 'disabled',
      onclick: () => run('frame'),
    }));
    body.push(el('button', {
      class: 'btn block-btn',
      text: `Run on all ${frames} frames`,
      disabled: frames ? null : 'disabled',
      onclick: () => run('all'),
    }));
  }

  const report = app.scriptReport;
  if (report) {
    body.push(el('p', { class: 'block-note', text: scriptOutcome(report, frames) }));
    if (report.backup) {
      body.push(el('p', { class: 'block-note dim', text: `The session as it was before this run: ${report.backup}` }));
    }
    if (report.output && report.output.length) {
      body.push(el('div', { class: 'joblog', text: report.output.slice(-10).join('\n') }));
    }
    const errors = report.errors || [];
    for (const problem of errors.slice(0, 3)) {
      body.push(el('div', { class: 'joblog bad',
        text: (problem.frame === null ? '' : `Frame ${problem.frame}\n`) + problem.error }));
    }
    if (errors.length > 3) {
      body.push(el('p', { class: 'block-note', text: `…and ${errors.length - 3} more frames failed the same way.` }));
    }
  }

  for (const bad of broken) {
    body.push(el('div', { class: 'joblog bad', text: `${bad.id}.py\n${bad.error}` }));
  }

  body.push(el('div', { class: 'picker-row' }, [
    el('button', { class: 'ghost', text: 'Reload', title: 'Read the folder again',
      onclick: () => app.refreshScripts(true) }),
    el('button', { class: 'ghost', text: 'Open folder',
      onclick: async () => {
        try { await api.openScripts(); } catch (err) { toast(err.message, 'error', 'Could not open the folder'); }
      } }),
  ]));

  return disclosure(app, {
    id: 'scripts',
    title: 'Run a script',
    summary: chosen ? chosen.name
      : (list.length ? `${plural(broken.length, 'file')} will not load` : 'none yet'),
    children: body,
  });
}

/** What a run did, in one line. */
export function scriptOutcome(report, scanned) {
  const errors = report.errors || [];
  if (errors.length) {
    const where = errors[0].frame === null
      ? 'It stopped before changing anything'
      : `${plural(errors.length, 'frame')} failed and kept the labels they had`;
    const rest = report.changed ? ` ${plural(report.changed, 'frame')} changed.` : '';
    return `${where}.${rest}`;
  }
  if (!report.changed) return `Nothing changed on ${plural(report.scanned ?? scanned, 'frame')}.`;
  const moves = [];
  if (report.removed) moves.push(`${plural(report.removed, 'box', 'boxes')} removed`);
  if (report.added) moves.push(`${plural(report.added, 'box', 'boxes')} added`);
  if (report.dropped) moves.push(`${plural(report.dropped, 'box', 'boxes')} fell outside the picture`);
  const tail = moves.length ? ` — ${moves.join(', ')}` : '';
  return `${plural(report.changed, 'frame')} changed${tail}.${report.stopped ? ' Stopped early.' : ''}`;
}

/** The settings for one script: what it declared, with anything you changed. */
function scriptValues(app, script) {
  const stored = (app.ui.scriptParams || {})[script.id] || {};
  const out = {};
  for (const spec of script.params) {
    out[spec.key] = stored[spec.key] === undefined ? spec.default : stored[spec.key];
  }
  return out;
}

function setScriptValue(app, script, key, value) {
  const all = { ...(app.ui.scriptParams || {}) };
  all[script.id] = { ...(all[script.id] || {}), [key]: value };
  app.ui.scriptParams = all;
}

/** One PARAMS entry, as the control it asked for. */
function scriptControl(app, script, spec) {
  const value = scriptValues(app, script)[spec.key];
  const hint = spec.hint ? el('span', { class: 'block-note dim', text: spec.hint }) : null;
  const keep = (v) => setScriptValue(app, script, spec.key, v);

  if (spec.type === 'bool') {
    const input = el('input', { type: 'checkbox', ...(value ? { checked: 'checked' } : {}) });
    input.addEventListener('change', () => keep(input.checked));
    return el('div', { class: 'field' }, [el('label', { class: 'inline-field' }, [input, spec.label]), hint]);
  }

  if (spec.type === 'text') {
    const input = el('input', { type: 'text', value: String(value ?? ''), placeholder: spec.placeholder || '' });
    input.addEventListener('input', () => keep(input.value));
    return field(spec.label, input, spec.hint);
  }

  if (spec.type === 'choice') {
    const input = select(spec.options, String(value));
    input.addEventListener('change', () => keep(input.value));
    return field(spec.label, input, spec.hint);
  }

  if (spec.type === 'class') {
    const input = select(app.state.classes.map((name, i) => [String(i), name]), String(value));
    input.addEventListener('change', () => keep(+input.value));
    return field(spec.label, input, spec.hint);
  }

  // A number with both ends is a slider; one open at either end is a box to type in.
  if (spec.min === null || spec.max === null) {
    const input = num(value, { min: spec.min ?? -1e9, max: spec.max ?? 1e9, step: spec.step });
    input.addEventListener('change', () => keep(+input.value));
    return field(spec.label, input, spec.hint);
  }

  const input = el('input', {
    type: 'range', min: String(spec.min), max: String(spec.max),
    step: String(spec.step), value: String(value),
  });
  const readout = el('span', { class: 'block-note', text: scriptReadout(spec, value) });
  // As with the augmentation sliders: update the readout, but do not rebuild
  // the panel mid-drag or the slider goes out from under the pointer.
  input.addEventListener('input', () => { readout.textContent = scriptReadout(spec, +input.value); });
  input.addEventListener('change', () => keep(+input.value));
  return el('label', { class: 'field' }, [el('span', { text: spec.label }), input, readout, hint]);
}

function scriptReadout(spec, value) {
  const shown = spec.integer ? String(Math.round(value)) : String(+(+value).toFixed(4));
  const unit = spec.format ? ` ${spec.format}` : '';
  const tail = Math.abs(value - spec.default) < 1e-9 ? ' · default' : '';
  return `${shown}${unit}${tail}`;
}

// ── 3 · Train ──────────────────────────────────────────────────

export function trainPanel(app) {
  const ui = app.ui;
  const stats = app.state.stats || {};

  // Export
  const task = select([['detect', 'Detection — boxes'], ['segment', 'Segmentation — masks'], ['pose', 'Pose — 4 corner points']], app.state.task);
  task.addEventListener('change', async () => {
    await api.settings({ task: task.value });
    app.state.task = task.value;
    app.renderPanel();
  });

  const skip = el('input', { type: 'checkbox', ...(ui.skipUnreviewed ? { checked: 'checked' } : {}) });
  skip.addEventListener('change', () => { ui.skipUnreviewed = skip.checked; });

  const exportBtn = el('button', {
    class: 'btn go block-btn',
    text: 'Export labels to a folder',
    disabled: stats.labelled ? null : 'disabled',
    onclick: async () => {
      const outDir = await pickPath({
        title: 'Where should the dataset go?',
        subtitle: 'Writes images/, labels/ and data.yaml into the folder you choose.',
        confirmLabel: 'Export here',
      });
      if (!outDir) return;
      try {
        const res = await api.export({ outDir, task: task.value, skipUnreviewed: skip.checked });
        ui.datasetDir = res.outDir;
        toast(`${plural(res.frames, 'frame')} written to ${res.outDir}`, 'ok', 'Exported');
        app.renderPanel();
      } catch (err) { toast(err.message, 'error', 'Export failed'); }
    },
  });

  const exportBlock = el('div', { class: 'block' }, [
    el('h3', {}, ['Export a dataset']),
    field('Label format', task),
    el('label', { class: 'inline-field' }, [skip, 'Only export frames I have reviewed']),
    exportBtn,
    stats.labelled
      ? el('p', { class: 'block-note', text: `${stats.labelled} of ${stats.frames} frames carry labels. Frames with no labels are never exported.` })
      : el('p', { class: 'block-note', text: 'Nothing is labelled yet — export unlocks once you have at least one box.' }),
  ]);

  // Convert
  const convertBlock = disclosure(app, {
    id: 'convert',
    title: 'Convert a dataset',
    summary: 'on disk',
    children: [
    el('p', { class: 'block-note', text: 'Upgrades an exported dataset in place, writing a new folder next to it.' }),
    el('button', {
      class: 'btn block-btn',
      text: 'Boxes → masks (segmentation)',
      onclick: async () => {
        const source = await pickPath({ title: 'Pick a detection dataset', subtitle: 'The folder must contain data.yaml.', confirmLabel: 'Convert this' });
        if (!source) return;
        try { await api.convertSeg({ source }); } catch (err) { toast(err.message, 'error', 'Could not start'); }
      },
    }),
    el('button', {
      class: 'btn block-btn',
      text: 'Masks → 4 corner points (pose)',
      onclick: () => convertPoseDialog(),
    }),
    ],
  });

  // Train
  const models = app.models?.[app.state.task] || [];
  const model = select(models, models.includes(ui.model) ? ui.model : models[0]);
  model.addEventListener('change', () => { ui.model = model.value; });
  ui.model = model.value;

  const epochs = num(ui.epochs, { min: 1, max: 2000 });
  epochs.addEventListener('change', () => { ui.epochs = +epochs.value || 50; });
  const imgsz = select([['320', '320'], ['512', '512'], ['640', '640 (default)'], ['960', '960'], ['1280', '1280']], String(ui.imgsz));
  imgsz.addEventListener('change', () => { ui.imgsz = +imgsz.value; });
  const cache = select([['off', 'Off — lowest memory'], ['disk', 'Disk'], ['ram', 'RAM — fastest, heaviest']], ui.cache);
  cache.addEventListener('change', () => { ui.cache = cache.value; });
  const workers = num(ui.workers, { min: 0, max: 16 });
  workers.addEventListener('change', () => { ui.workers = +workers.value; });

  const datasetRow = pathField(ui, 'datasetDir', {
    placeholder: 'Choose the exported dataset folder',
    onPick: async () => {
      const dir = await pickPath({ title: 'Pick the dataset to train on', subtitle: 'The folder that holds data.yaml.', confirmLabel: 'Train on this' });
      if (dir) { ui.datasetDir = dir; app.renderPanel(); }
    },
  });

  const startTraining = async (confirm = false) => {
    if (!ui.datasetDir) { toast('Choose a dataset folder first.', 'warn'); return; }
    try {
      const res = await api.train({
        model: model.value,
        epochs: +epochs.value,
        imgsz: +imgsz.value,
        cache: cache.value,
        workers: +workers.value,
        augment: augmentValues(app),
        datasetDir: ui.datasetDir,
        confirm,
      });
      if (res.needsConfirm) {
        const ok = await confirmDialog({
          title: 'Model and dataset do not match',
          message: res.warnings.join('\n\n'),
          confirmLabel: 'Train anyway',
          danger: true,
        });
        if (ok) startTraining(true);
      }
    } catch (err) { toast(err.message, 'error', 'Could not start training'); }
  };

  const settingsBlock = disclosure(app, {
    id: 'trainsettings',
    title: 'Training settings',
    summary: `${plural(ui.epochs, 'epoch')} · ${ui.imgsz}px`,
    children: [
      el('div', { class: 'field-row' }, [field('Epochs', epochs), field('Image size', imgsz)]),
      el('div', { class: 'field-row' }, [field('Image cache', cache), field('Loader workers', workers)]),
      el('p', { class: 'block-note', text: 'Cache trades memory for speed. Fewer loader workers if the machine struggles.' }),
    ],
  });

  const augmentBlock = augmentDisclosure(app);

  const trainBlock = el('div', { class: 'block' }, [
    el('h3', {}, ['Train a model']),
    field('Dataset', datasetRow),
    field('Model', model),
    settingsBlock,
    augmentBlock,
    el('button', { class: 'btn go block-btn', text: 'Start training', onclick: () => startTraining(false) }),
    el('p', { class: 'block-note', text: 'Progress shows at the bottom of the window, with a Stop button that keeps the epochs already finished.' }),
  ]);

  // ONNX
  const prec = select(['FP32', 'FP16'], ui.precision);
  prec.addEventListener('change', () => { ui.precision = prec.value; });
  const shape = select([['dynamic', 'Dynamic — any input size'], ['static', 'Static — fixed input size']], ui.shape);
  shape.addEventListener('change', () => { ui.shape = shape.value; });

  const onnxBlock = disclosure(app, {
    id: 'onnx',
    title: 'Export to ONNX',
    summary: `${ui.precision} · ${ui.shape}`,
    children: [
    el('div', { class: 'field-row' }, [field('Precision', prec), field('Input shape', shape)]),
    el('button', {
      class: 'btn block-btn',
      text: 'Pick a .pt file and export',
      onclick: async () => {
        const ptPath = await pickPath({
          title: 'Pick a trained checkpoint',
          subtitle: 'Trained weights live under runs/train/…/weights/best.pt',
          want: '.pt',
          confirmLabel: 'Export this',
        });
        if (!ptPath) return;
        try {
          await api.onnx({ ptPath, precision: prec.value, dynamic: shape.value === 'dynamic', imgsz: +imgsz.value });
        } catch (err) { toast(err.message, 'error', 'Export failed'); }
      },
    }),
    el('p', { class: 'block-note', text: 'Exports at the image size set above. FP16 needs a CUDA GPU, and an export runs to completion once started.' }),
    ],
  });

  return el('div', { class: 'panel-inner' }, [
    exportBlock,
    el('hr', { class: 'rule' }),
    trainBlock,
    convertBlock,
    onnxBlock,
  ]);
}

// ── Image augmentation ─────────────────────────────────────────
//
// Training already distorts every picture it shows the model — a little colour
// shift, a flip, a crop, four images stitched together — so that 40 labelled
// frames stretch further than 40. That happens whether or not anyone asks for
// it; all this does is show the settings and let them be changed.

const AUG_GROUPS = [
  ['geometric', 'Shape and position', 'Moving the picture teaches the model that an object is the same object wherever it lands.'],
  ['photometric', 'Light and colour', 'Changing the light teaches it that an object is the same object in a different exposure.'],
  ['composition', 'Combining pictures', 'Building new pictures out of old ones. Strong medicine, and the first thing to turn down if training goes strange.'],
];

/** Every value, including the ones left at their default. */
function augmentValues(app) {
  const specs = app.state.augmentations || [];
  const set = app.ui.augment || {};
  const out = {};
  for (const spec of specs) {
    out[spec.key] = set[spec.key] === undefined ? spec.default : set[spec.key];
  }
  return out;
}

function augmentChangedCount(app) {
  const specs = app.state.augmentations || [];
  const set = app.ui.augment || {};
  return specs.filter((s) => set[s.key] !== undefined
                          && Math.abs(set[s.key] - s.default) > 1e-9).length;
}

function augmentDisclosure(app) {
  const specs = app.state.augmentations || [];
  const changed = augmentChangedCount(app);

  const children = [
    el('p', { class: 'block-note', text: 'Applied fresh every epoch, so the model rarely sees the same picture twice. Your labels are not touched — these only affect what training sees.' }),
  ];

  for (const [group, title, note] of AUG_GROUPS) {
    const inGroup = specs.filter((s) => s.group === group);
    if (!inGroup.length) continue;
    children.push(el('div', { class: 'aug-group' }, [
      el('h4', { text: title }),
      el('p', { class: 'block-note', text: note }),
      ...inGroup.map((spec) => augSlider(app, spec)),
    ]));
  }

  children.push(el('button', {
    class: 'btn block-btn',
    text: 'Put everything back to its default',
    disabled: changed ? null : 'disabled',
    onclick: () => { app.ui.augment = {}; app.renderPanel(); },
  }));

  return disclosure(app, {
    id: 'augment',
    title: 'Image augmentation',
    summary: changed ? `${changed} changed` : 'defaults',
    children,
  });
}

function convertPoseDialog() {
  const margin = num(2, { min: 0, max: 49, step: 0.5 });
  openModal({
    title: 'Masks → 4 corner points',
    subtitle: 'Reads polygon labels and writes the four corners of each shape as pose keypoints.',
    body: el('div', { class: 'block' }, [
      field('Drop points within this % of an image edge', margin,
        'Corners touching the border are usually cut off, so they make poor keypoints. They are kept as removable ghosts you can restore while labelling.'),
    ]),
    actions: [
      { label: 'Cancel', onClick: () => closeModal() },
      {
        label: 'Choose dataset',
        kind: 'primary',
        onClick: async () => {
          const edgeMargin = +margin.value;
          closeModal();
          const source = await pickPath({ title: 'Pick a segmentation dataset', subtitle: 'The folder must contain data.yaml with task: segment.', confirmLabel: 'Convert this' });
          if (!source) return;
          try { await api.convertPose({ source, edgeMargin }); } catch (err) { toast(err.message, 'error', 'Could not start'); }
        },
      },
    ],
  });
}
