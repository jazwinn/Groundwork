// Server calls and the live event feed.

async function request(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const message = (data && (data.error || data.detail)) || `${res.status} ${res.statusText}`;
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }
  return data;
}

export const api = {
  state:        ()            => request('GET', '/api/state'),
  models:       ()            => request('GET', '/api/models'),
  settings:     (patch)       => request('POST', '/api/settings', patch),
  setClasses:   (names)       => request('POST', '/api/classes', { names }),

  browse:       (path, want)  => request('GET', `/api/fs?want=${encodeURIComponent(want || 'dir')}${path ? `&path=${encodeURIComponent(path)}` : ''}`),
  inspect:      (path)        => request('GET', `/api/fs/inspect?path=${encodeURIComponent(path)}`),
  nativePick:   (payload)     => request('POST', '/api/fs/native', payload),
  nativeCancel: ()            => request('POST', '/api/fs/native/cancel'),

  import:       (payload)     => request('POST', '/api/import', payload),
  frames:       ()            => request('GET', '/api/frames'),
  frame:        (i)           => request('GET', `/api/frames/${i}`),
  putBoxes:     (i, boxes, status) => request('PUT', `/api/frames/${i}/boxes`, { boxes, status }),
  putStatus:    (i, status)   => request('PUT', `/api/frames/${i}/status`, { status }),
  deleteFrame:  (i)           => request('DELETE', `/api/frames/${i}`),
  deleteFrames: (indices)     => request('POST', '/api/frames/delete', { indices }),
  clearLabels:  (indices)     => request('POST', '/api/frames/clear-labels', { indices }),
  mergeBoxes:   (i, indices)  => request('POST', `/api/frames/${i}/merge`, { indices }),
  mergeOverlaps:(payload)     => request('POST', '/api/labels/merge-overlaps', payload),

  describe:     (payload)     => request('POST', '/api/autolabel/describe', payload),
  propagate:    (payload)     => request('POST', '/api/autolabel/propagate', payload),
  track:        (payload)     => request('POST', '/api/autolabel/track', payload),
  prompt:       (payload)     => request('POST', '/api/autolabel/prompt', payload),
  snap:         (payload)     => request('POST', '/api/autolabel/snap', payload),
  prewarm:      (payload)     => request('POST', '/api/autolabel/prewarm', payload),

  scripts:      ()            => request('GET', '/api/scripts'),
  runScript:    (payload)     => request('POST', '/api/scripts/run', payload),
  openScripts:  ()            => request('POST', '/api/scripts/open', {}),

  jobs:         ()            => request('GET', '/api/jobs'),
  jobLog:       (id)          => request('GET', `/api/jobs/${id}/log`),
  cancelJob:    (id)          => request('POST', `/api/jobs/${id}/cancel`),

  export:       (payload)     => request('POST', '/api/export', payload),
  convertSeg:   (payload)     => request('POST', '/api/convert/seg', payload),
  convertPose:  (payload)     => request('POST', '/api/convert/pose', payload),
  train:        (payload)     => request('POST', '/api/train', payload),
  onnx:         (payload)     => request('POST', '/api/onnx', payload),

  saveSession:  ()            => request('POST', '/api/session/save'),
  resetSession: ()            => request('POST', '/api/session/reset'),
};

// ── Live events ────────────────────────────────────────────────

export function connect(onMessage, onStatus) {
  let socket = null;
  let retry = 0;

  const open = () => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${proto}://${location.host}/ws`);

    socket.onopen = () => { retry = 0; onStatus?.(true); };
    socket.onmessage = (event) => {
      try { onMessage(JSON.parse(event.data)); } catch { /* ignore */ }
    };
    socket.onclose = () => {
      onStatus?.(false);
      retry = Math.min(retry + 1, 6);
      setTimeout(open, 400 * retry);
    };
    socket.onerror = () => socket.close();
  };

  open();
  return () => socket && socket.close();
}
