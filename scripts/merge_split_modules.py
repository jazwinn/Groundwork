"""Join module halves that auto-labelling cut in two.

The segmenter sometimes draws one module as two shapes meeting along a line that
is not there.  Merging every pair that touches would be wrong: in a row of
three, two of them belong together and the third does not, and nothing about
"touching" says which two.

So every abutting pair is scored, the best pairing wins, and each shape is
allowed into at most one merge — which is what settles the row of three.

What decides it is the seam itself, read off the picture.  Modules are held
apart by their frames: a real boundary is a dark gutter with a lit aluminium
edge down each side of it.  Both marks are looked for, and either one on its own
settles the question — there is a module frame here, so there is nothing to
join, whatever the rest of the scoring thinks.

Looking for the dark line alone is not enough, which is worth knowing before
loosening anything.  Modules have busbars running down them, and a busbar is a
dark line too: on the frames this was built against, seams cut along one darken
about as much as a third of the real boundaries do.  What a busbar has not got
is the bright frame either side, and that is what separates the two cleanly.

How far apart the two shapes were drawn is worth much less than it looks.  An
outline traced round a module does not reliably stop at its edge, so two halves
of one module can be drawn ten pixels apart while two genuinely separate modules
are drawn touching.  It is scored lightly, along with whether joining the pair
gives back something the size of the whole modules around it.

Read the report before trusting it.  What counts as a convincing seam differs
from one set of pictures to the next, and the numbers printed for each pair are
the ones to set the thresholds against.
"""

from __future__ import annotations

import math

import numpy as np

from app.core.boxops import merge_boxes
from app.core.yolo_pose_converter import extract_four_corners

NAME = "Merge split modules"

PARAMS = [
    {"key": "dry_run", "label": "Report only", "type": "bool", "default": True,
     "hint": "On, nothing is changed — the report says what would be merged."},
    {"key": "min_score", "label": "Confidence needed to merge", "default": 0.25,
     "min": 0.0, "max": 1.0, "step": 0.05,
     "hint": "Only sifts the pairs with no gutter between them; a pair with one "
             "is left alone whatever this says. Set it against the report."},
    {"key": "gap_ratio", "label": "A split is a gap under", "default": 0.8,
     "min": 0.1, "max": 1.5, "step": 0.05,
     "hint": "As a fraction of the gap between the picture's other modules."},
    {"key": "max_dip", "label": "Darkest a split's seam may be", "default": 0.28,
     "min": 0.05, "max": 1.0, "step": 0.01,
     "hint": "Past this the seam is a gutter, and the pair is left alone."},
    {"key": "max_rise", "label": "Brightest a split's edges may be", "default": 0.10,
     "min": 0.02, "max": 1.0, "step": 0.01,
     "hint": "Module frames catch the light either side of a real gutter. "
             "This is what tells one from a busbar running down a module."},
    {"key": "spare_edited", "label": "Leave edited shapes alone", "type": "bool",
     "default": True,
     "hint": "Skips any pair you have already drawn or corrected by hand. Also "
             "what stops a joined module being joined again on a second run."},
    {"key": "min_overlap", "label": "Least two edges must share", "default": 0.6,
     "min": 0.1, "max": 1.0, "step": 0.05,
     "hint": "Stops a corner grazing another shape counting as a seam."},
    {"key": "search_gap", "label": "Furthest apart to consider", "default": 40,
     "min": 2, "max": 300, "step": 1, "integer": True, "format": "pixels",
     "hint": "Only needs to be wider than the gutter between real modules."},
    {"key": "use_pixels", "label": "Look at the picture", "type": "bool", "default": True,
     "hint": "This is what tells a gutter from an invented line. Off, the shapes "
             "have to speak for themselves, and on aerial rows they rarely do."},
    {"key": "clean_quad", "label": "Square the merged outline off", "type": "bool",
     "default": True,
     "hint": "Reduces the joined outline back to four corners."},
]

PARALLEL = 0.96     # how aligned two edges must be before they can be one seam
SAMPLES = 41        # points taken along a seam, and again across it
MAX_REACH = 60.0    # how far either side of a seam to look, in pixels
MAX_PASSES = 4      # sweeps over one frame before calling it settled


# ── small vector helpers ────────────────────────────────────────────────────

def _sub(a, b): return (a[0] - b[0], a[1] - b[1])
def _add(a, b): return (a[0] + b[0], a[1] + b[1])
def _mul(a, k): return (a[0] * k, a[1] * k)
def _dot(a, b): return a[0] * b[0] + a[1] * b[1]


class Seam:
    """Where two shapes meet: the line between them, and how far apart they are."""

    __slots__ = ("gap", "length", "mid", "along", "across")

    def __init__(self, gap, length, mid, along, across):
        self.gap = gap        # perpendicular distance between the two edges
        self.length = length  # how long the shared run is, in pixels
        self.mid = mid        # the middle of the seam
        self.along = along    # unit vector down the seam
        self.across = across  # unit vector through it


# ── shapes ──────────────────────────────────────────────────────────────────

def _outline(box, width, height):
    """The outline as drawn, in pixels.

    Seams are measured off this rather than off the squared-up quad below.
    Reducing an outline to four corners moves its edges by a few pixels, which
    is nothing next to a module but is most of the gap being measured.
    """
    if box.polygon:
        flat = box.polygon
        points = [(flat[i] * width, flat[i + 1] * height)
                  for i in range(0, len(flat) - 1, 2)]
        if len(points) >= 3:
            return points
    return [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)]


def _edges(points):
    return [(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]


def _long_side(box, width, height):
    """How long the shape is, once squared off to four corners.

    Only ever compared with other shapes' long sides, so the squaring-up that
    would spoil a gap measurement does no harm here.
    """
    corners = None
    if box.polygon:
        try:
            corners = extract_four_corners(list(box.polygon))
        except Exception:
            corners = None
    if corners and len(corners) == 4:
        quad = [(x * width, y * height) for x, y in corners]
    else:
        quad = [(box.x1, box.y1), (box.x2, box.y1), (box.x2, box.y2), (box.x1, box.y2)]
    sides = sorted(math.dist(a, b) for a, b in _edges(quad))
    return (sides[-1] + sides[-2]) / 2


def _seam_of(edge_a, edge_b, min_overlap, search_gap):
    """The seam these two edges form, or None if they do not form one."""
    a1, a2 = edge_a
    b1, b2 = edge_b
    da, db = _sub(a2, a1), _sub(b2, b1)
    la, lb = math.hypot(*da), math.hypot(*db)
    if la < 1 or lb < 1:
        return None

    along = _mul(da, 1 / la)
    if abs(_dot(along, _mul(db, 1 / lb))) < PARALLEL:
        return None

    mid_a = _mul(_add(a1, a2), 0.5)
    mid_b = _mul(_add(b1, b2), 0.5)
    across = (-along[1], along[0])
    offset = _dot(_sub(mid_b, mid_a), across)
    gap = abs(offset)
    if gap > search_gap:
        return None

    # How much of the shorter edge the two actually run alongside.
    span_a = sorted(_dot(_sub(p, mid_a), along) for p in (a1, a2))
    span_b = sorted(_dot(_sub(p, mid_a), along) for p in (b1, b2))
    low, high = max(span_a[0], span_b[0]), min(span_a[1], span_b[1])
    if (high - low) / min(la, lb) < min_overlap:
        return None

    centre = _add(mid_a, _mul(along, (low + high) / 2))
    return Seam(gap, high - low,
                _add(centre, _mul(across, offset / 2)), along, across)


def _seam_between(outline_a, outline_b, min_overlap, search_gap):
    """The tightest seam between two shapes, out of every pairing of edges."""
    best = None
    for edge_a in _edges(outline_a):
        for edge_b in _edges(outline_b):
            seam = _seam_of(edge_a, edge_b, min_overlap, search_gap)
            if seam and (best is None or seam.gap < best.gap):
                best = seam
    return best


def _neighbours(pairs, i, j):
    """The shapes that meet this pair, without the pair itself."""
    out = set()
    for a, b, _ in pairs:
        if a in (i, j) and b not in (i, j):
            out.add(b)
        if b in (i, j) and a not in (i, j):
            out.add(a)
    return out


# ── the picture ─────────────────────────────────────────────────────────────

def _across_seam(grey, seam, reference):
    """What the picture does as you cross the seam: how far it darkens, and how
    far it brightens either side of that.

    A boundary between two modules is a dark gutter with a lit aluminium frame
    down each side of it, so it both darkens and brightens sharply.  A busbar
    running down the middle of one module darkens just as convincingly and does
    not brighten at all, which is the only thing telling the two apart.

    The crossing is sampled a pixel at a time, and how far out to look for the
    trough, for the frames, and for the module itself all follow the width of
    this picture's gutters.  Reading those three off a fixed count of samples
    instead would quietly measure a different width in every frame, and the two
    marks are close enough together that it changes the answer.
    """
    height, width = grey.shape
    reach = min(max(3.0 * reference, 12.0), MAX_REACH)
    trough = max(3.0, 0.6 * reference)      # where the gutter itself would be
    frames = max(6.0, 1.3 * reference)      # and the lit frames either side of it

    offsets = np.arange(-reach, reach + 1.0)
    steps = np.linspace(-0.4 * seam.length, 0.4 * seam.length, SAMPLES)
    profile = []
    for offset in offsets:
        base = _add(seam.mid, _mul(seam.across, float(offset)))
        xs = np.clip(np.rint(base[0] + seam.along[0] * steps).astype(int), 0, width - 1)
        ys = np.clip(np.rint(base[1] + seam.along[1] * steps).astype(int), 0, height - 1)
        profile.append(float(grey[ys, xs].mean()))

    profile = np.asarray(profile)
    away = np.abs(offsets) >= reach - 8.0
    interior = float(profile[away].mean())
    if interior <= 1.0:
        return 0.0, 0.0
    dip = (interior - float(profile[np.abs(offsets) <= trough].min())) / interior
    rise = (float(profile[np.abs(offsets) <= frames].max()) - interior) / interior
    return max(0.0, dip), max(0.0, rise)


# ── scoring ─────────────────────────────────────────────────────────────────

def _falling(value, limit):
    """1 where value is 0, sliding to 0 by the time it reaches limit."""
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - value / limit))


def _clipped(box, width, height, margin=4.0):
    """Whether the edge of the picture cut this shape short."""
    return (box.x1 <= margin or box.y1 <= margin
            or box.x2 >= width - margin or box.y2 >= height - margin)


def _size_score(merged_long, neighbours):
    """Whether joining these two gives back something the size of its neighbours.

    Only whole modules are worth comparing against, so the shapes the edge of
    the picture cut short are left out.  A row can easily have nothing but those
    either side of it, and half a module is no evidence about the size of a
    whole one; with none left to compare against this says so rather than
    guessing, and the other signals decide.
    """
    if not neighbours:
        return 0.5
    closest = min(abs(merged_long - n) / max(n, 1.0) for n in neighbours)
    return _falling(closest, 0.35)


# ── the script ──────────────────────────────────────────────────────────────

def apply(frame, p):
    """Join what this frame has to join, and keep going until nothing is left.

    Joining one pair sharpens the picture for the rest: the gutters get measured
    against a truer set of shapes, and a half that had only halves to compare
    itself with finally has a whole module next to it.  So a second look often
    finds a pair the first could not, and running once should mean running until
    there is nothing more rather than leaving the rest for whoever thinks to
    press the button again.  Nothing already joined is reconsidered, so this
    settles after a pass or two.
    """
    joined = 0
    boxes = frame.boxes
    for _ in range(MAX_PASSES):
        boxes, made = _one_pass(frame, boxes, p, first=joined == 0)
        joined += made
        if not made:
            break

    if p.dry_run:
        if joined:
            print(f"  {joined} would be merged. Turn off Report only to apply.")
        return frame.boxes
    return boxes


def _one_pass(frame, boxes, p, first):
    """One sweep: score every pair that meets, join the best, and say how many."""
    if len(boxes) < 2:
        return boxes, 0

    outlines = [_outline(box, frame.width, frame.height) for box in boxes]
    lengths = [_long_side(box, frame.width, frame.height) for box in boxes]
    # A joined module is stored as hand-made, so sparing edited shapes is also
    # what keeps a second run from joining it to the next module along.
    settled = [p.spare_edited and box.source == "manual" for box in boxes]

    pairs = []
    for i in range(len(outlines)):
        for j in range(i + 1, len(outlines)):
            if settled[i] or settled[j]:
                continue
            seam = _seam_between(outlines[i], outlines[j], p.min_overlap, p.search_gap)
            if seam:
                pairs.append((i, j, seam))
    if not pairs:
        return boxes, 0

    gaps = sorted(seam.gap for _, _, seam in pairs)
    reference = gaps[len(gaps) // 2]
    # Under three shapes meeting, there is no telling what this picture's
    # gutters look like, so fall back to calling anything nearly touching a split.
    calibrated = len(gaps) >= 3 and reference >= 2.0
    if not calibrated:
        reference = p.search_gap / 4

    grey = np.asarray(frame.image().convert("L"), dtype=np.float32) if p.use_pixels else None

    scored = []
    for i, j, seam in pairs:
        gap_score = _falling(seam.gap, p.gap_ratio * reference)

        gutter = None
        if grey is not None:
            dip, rise = _across_seam(grey, seam, reference)
            # Either mark on its own is enough to call this a real boundary.
            gutter = max(dip / p.max_dip, rise / p.max_rise)

        merged = merge_boxes([boxes[i], boxes[j]], frame.width, frame.height)
        size_score = _size_score(
            _long_side(merged, frame.width, frame.height),
            [lengths[k] for k in _neighbours(pairs, i, j)
             if not _clipped(boxes[k], frame.width, frame.height)])

        if gutter is None:
            total = 0.7 * gap_score + 0.3 * size_score
        elif gutter >= 1.0:
            # A module frame runs down this seam, so there is nothing to join.
            total = 0.0
        else:
            total = 0.4 * (1.0 - gutter) + 0.2 * gap_score + 0.4 * size_score
        gated = gutter is not None and gutter >= 1.0
        scored.append((total, i, j, seam, gutter, gap_score, size_score, gated))

    scored.sort(key=lambda row: row[0], reverse=True)

    taken = set()
    accepted = []
    for row in scored:
        total, i, j, gated = row[0], row[1], row[2], row[7]
        if not gated and total >= p.min_score and i not in taken and j not in taken:
            taken.update((i, j))
            accepted.append(row)

    _report(frame, scored, accepted, reference, calibrated, first)

    if not accepted:
        return boxes, 0

    replace = {}
    remove = set()
    for _, i, j, *_rest in accepted:
        merged = merge_boxes([boxes[i], boxes[j]], frame.width, frame.height)
        if p.clean_quad and merged.polygon:
            _square_off(merged, frame.width, frame.height)
        replace[i] = merged
        remove.add(j)

    kept = [replace.get(k, box) for k, box in enumerate(boxes) if k not in remove]
    return kept, len(accepted)


def _square_off(box, width, height):
    """Reduce a joined outline to four corners, keeping the rectangle on it.

    The outline is the truth and the rectangle wraps it, which is the rule the
    editor works to, so the rectangle comes back off the corners rather than
    staying as the union of the two shapes that went in.
    """
    try:
        corners = extract_four_corners(list(box.polygon))
    except Exception:
        return
    if not corners or len(corners) != 4:
        return
    box.polygon = [value for corner in corners for value in corner]
    xs = [x * width for x, _ in corners]
    ys = [y * height for _, y in corners]
    box.x1, box.x2 = min(xs), max(xs)
    box.y1, box.y2 = min(ys), max(ys)


def _report(frame, scored, accepted, reference, calibrated, first):
    """What was considered and why, in the order it was considered.

    This is the part that makes the thresholds settable: the gap and the dip are
    printed as measured, next to what the current settings made of them.

    Only the pairs that came anywhere near merging are listed.  Running across
    every frame would otherwise print several hundred lines of pairs that were
    never in question, and the panel keeps the first few hundred lines of a run
    — so the frames worth looking at would be the ones scrolled away.
    """
    if not scored:
        return

    close = [row for row in scored if row[0] > 0.0]
    if not close:
        if first:
            print(f"Frame {frame.index}: {len(scored)} pairs meet, all with a gutter.")
        return

    chosen = {(row[1], row[2]) for row in accepted}
    print(f"Frame {frame.index}: {len(scored)} pairs meet, "
          f"typical gap {reference:.1f}px" + ("" if calibrated else " (guessed)")
          + ("" if first else ", looking again"))
    for total, i, j, seam, gutter, gap_score, size_score, _gated in close:
        seam_text = "gutter    -" if gutter is None else f"gutter {gutter:4.2f}"
        print(f"  {i:>2}+{j:<2} gap {seam.gap:5.1f}px  fit {gap_score:.2f}  "
              f"{seam_text}  size {size_score:.2f}  -> {total:.2f}  "
              f"{'MERGE' if (i, j) in chosen else 'keep'}")
    rest = len(scored) - len(close)
    if rest:
        print(f"  and {rest} with a gutter down them.")
