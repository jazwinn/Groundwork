---
name: groundwork
description: Drive the Groundwork labelling and training session from the command line — inspect and render frames, run label scripts, hold frames out, export, train, and put changes through the quality gate. Use when working on labels or models in this repo without the browser, or when asked to clean up labels, check labelling quality, or train and score a model.
---

# Groundwork from the command line

Groundwork is a labelling and training tool. The browser UI is one way to drive
it; `python -m app.cli` is the other, and the one to use here.

```bash
python -m app.cli stats
```

Everything prints JSON on **stdout** and nothing else. Progress, training
chatter and notes go to **stderr**. Exit codes: `0` it worked, `1` it ran and
the answer was no (a gate put a change back, a script errored), `2` it could not
run at all. So `... | jq` is always safe, and so is checking `$?` in CI.

## Before touching anything

Two processes writing one session lose each other's work, so every writing
command refuses while the browser server is up, and says so. Either close the
browser, or pass `--force` if you are certain nothing else is editing. Reading
always works.

`python -m app.cli backup --tag whatever` takes a restore point first;
`restore <path>` puts one back. Sweeping script runs take one automatically.

**Check what the app is already doing before you start.** Importing,
auto-labelling and training all run on the app's own worker threads, and a
second heavy job means two processes competing for one GPU.

```bash
python -m app.cli jobs                  # what is running, and how far along
python -m app.cli jobs --cancel all     # stop it
python -m app.cli jobs --log <id>       # what one of them said
```

Jobs only exist inside the running app, so with it closed there are none and
the command says so rather than printing an empty list.

## Starting a session

```bash
python -m app.cli import /path/to/footage            # video, pictures, or one file
python -m app.cli import /path/to/dataset --mode dataset
python -m app.cli classes SolarModules Defect
python -m app.cli settings --threshold 0.4
```

`import` defaults to `--mode add`, which keeps what is already there. The
browser defaults the same operation to *replace*, because someone who just
picked a folder means that folder — but a line in a script has nobody watching
it, so **emptying a session that already has frames needs `--yes`**, and the
refusal tells you how many labels are at stake. `--mode dataset` reads labels
and class names out of a YOLO dataset's `data.yaml`. `--stride N` keeps one
video frame in every N.

`classes` takes names in class-id order. Labels on disk hold ids, not names, so
renaming is only ever a relabelling of the same numbers — and shortening the
list leaves any label past the end pointing at a class that no longer has a
name, which it warns about rather than fixing.

`settings` with no arguments prints what the auto-labeller is set to and what
the valid choices are. `--threshold` is the one that changes results most:
lower finds more and invents more.

## Looking before changing

The single most useful habit: **render the frame and look at it.**

```bash
python -m app.cli render 24 --out /tmp/f24.png    # then read the image
```

Label quality is a visual judgement — whether an outline follows the object,
whether one object has been cut in two, whether a seam is a real gap or an
invented one. A list of coordinates does not answer any of that. Each shape is
drawn with its index, class and score, so a finding can be reported as
"box 2 and box 7 are halves of one module" and acted on directly.

The same picture over HTTP when the server is up:
`GET /api/frames/{index}/render`.

## Changing labels

Label scripts live in `scripts/` and appear in both the UI and the CLI.

```bash
python -m app.cli scripts
python -m app.cli run merge_split_modules --all --set dry_run=true
```

**Run with the script's dry-run setting on first, read the report, then apply.**
The reports print the measurements behind each decision, and the thresholds that
suit one set of pictures often do not suit another. Changing a threshold because
the report justifies it is good; changing it until something merges is not.

`--set key=value` takes JSON values (`true`, `12`, `"text"`).

## Tidying up

```bash
python -m app.cli tidy --all                 # fold stacked duplicate boxes together
python -m app.cli status 3 9 --pending       # mark frames unchecked
python -m app.cli clear-labels 12 --yes      # drop the labels, keep the picture
python -m app.cli delete 12 13 --yes         # drop the frames
```

`tidy` clusters boxes that sit *on top of* each other, which is what two passes
of an auto-labeller leave behind. Halves of one module abut rather than overlap,
so it never fires on those — `run merge_split_modules` is what does.

`clear-labels` and `delete` both take `--yes`, and the refusal without it says
exactly how many labels are at stake. Both take a restore point first, as does
`tidy`. All three refuse held-out frames and report which they skipped.

**`status` deserves care.** `verified` means a person looked at the frame and
stands behind it — that is the whole reason a held-out score means anything.
Setting it from a script claims a check nobody performed. Use it to record a
decision the user actually made, and `--pending` to undo one.

## Letting the model label

```bash
python -m app.cli autolabel --seed 12 --dry-run     # say what it would touch
python -m app.cli autolabel --seed 12               # then do it
```

Three ways to ask:

- `--seed N` copies frame N's labels everywhere — the class names on that frame
  become the search terms. Best quality, and the one to reach for first.
- `--concept "solar panel"` searches by description when no frame is a good
  seed. `--class-id` says which class the hits belong to.
- `--seed N --track` follows the exact shapes from one frame through the video.
  Needs frames imported at stride 1; it refuses and explains if they were not.

`--policy` decides what happens to labels already there. **`skip` is the
default and only fills empty frames.** `merge` appends, which duplicates
anything already drawn; `replace` overwrites, which throws away hand-corrected
work. Ask before using either.

**Run `--dry-run` first.** It prints the frames it would write to, the held-out
ones it will not touch, and the ones already labelled that `skip` will pass
over — before the model spends ten minutes. A sweep over "everything" often
comes down to very little, and it is worth knowing which.

Two things this command does differently from the rest:

- **It does not refuse while the browser is open.** It hands the work to the
  server instead, where the model is already in memory. Everything else refuses
  because two processes writing one session lose each other's work; this keeps
  that promise by making sure only one of them writes.
- **It takes a restore point by itself**, before anything runs, and prints the
  path.

Held-out frames are refused, so a sweep cannot touch the measuring set. Every
frame it writes is marked `pending` — auto-labelled means nobody has looked at
it yet, and that is the whole meaning of the status field. **Never mark
auto-labelled frames `verified` on the model's behalf.**

Afterwards, render a few and look. The labeller has a known habit of cutting
one module into two abutting shapes; `merge_split_modules` is what puts those
back together.

## The held-out half

Some frames are held out to measure against. They are exported as the validation
half, and **everything that writes labels refuses them** — the editor, the
auto-labeller, every script, even frame deletion.

```bash
python -m app.cli frames --status verified    # candidates: someone has checked these
python -m app.cli hold 3 9 17 24
python -m app.cli frames --held-out
python -m app.cli hold 3 --release
```

`status` is `pending` or `verified` and means only whether a person has checked
the frame. Whether it has been written into a dataset is a separate `exported`
flag — the two used to share one field, so exporting erased the record of what
had been reviewed. **Render a candidate and look at it before holding it out**:
a frame with objects nobody labelled will score every correct detection as a
false positive for as long as it stays in the held-out half.

The rule that makes them worth anything: **hold out frames a person has
checked, and never hold out frames you labelled yourself.** A model measured
against labels the same process produced is measuring its own opinion. If asked
to expand the held-out set, say which frames and why, and let the user decide —
this is the one part of the loop that should not be automated.

## Training and scoring

```bash
python -m app.cli export --out runs/ds
python -m app.cli train --data runs/ds --model YOLO11n-seg --epochs 50
python -m app.cli score --data runs/ds --weights runs/train/cli/weights/best.pt
```

Export writes `train.txt` and `val.txt`; held-out frames go in val. With nothing
held out it still exports, but scores against the training pictures and returns
a `warning` saying so. **If a result carries that warning, the number is not a
measurement** — say so rather than reporting it.

`score` reports mAP50-95 as `score`, because mAP50 saturates and stops
separating a good change from a bad one.

Two more, for getting a dataset or a model out of here:

```bash
python -m app.cli convert runs/ds --to pose     # or --to seg
python -m app.cli onnx runs/train/cli/weights/best.pt
```

`convert` reads one exported dataset and writes another in a different task's
shape, carrying the train/val split across as its own `train.txt`/`val.txt`.
`--to seg` runs SAM to turn boxes into outlines, so it goes to the server when
one is up and the model is already loaded.

## The gate

Change the labels, and keep the change only if the model still scores:

```bash
python -m app.cli gate --script merge_split_modules --set dry_run=false \
    --model YOLO11n-seg --epochs 30
```

It scores the labels as they are, takes a restore point, applies the change,
scores again, and keeps or puts back. Exit `0` kept, `1` put back. A crash
anywhere after the change puts it back too.

Two guards worth knowing:

- **Label loss.** If a change makes more than `--max-loss` (default 20%) of the
  labels disappear, it is put back without being scored at all. On a small
  held-out set, deleting most of the labels can pass for a change that did
  nothing; that is not something a score should be asked to adjudicate.
- **Noise.** With few held-out frames, a `caution` appears next to a small
  delta. Read it. At this size the gate answers "did I break something badly",
  not "did this help by 1%". Report a small delta as *nothing broke*, never as
  evidence the change was an improvement.

## A scheduled run

For cron, Task Scheduler or CI — clean up, gate it, and fail the job if the
change was put back:

```bash
cd /path/to/Groundwork
python -m app.cli backup --tag nightly
python -m app.cli gate --script merge_split_modules --set dry_run=false \
    --model YOLO11n-seg --epochs 30 > gate.json
test $? -eq 0 || echo "change was put back — see gate.json"
```

Nothing is committed automatically. Label changes are data, not code; show the
user what moved and let them decide.

## What not to do

- Do not relax a threshold to make something merge. Change it because the
  report justifies it, and say which numbers justified it.
- Do not hold out frames to make a score look better.
- Do not report a score from an export that carried the no-held-out warning.
- Do not run writing commands while the browser is open, or reach for `--force`
  to get past the refusal without knowing why it appeared.
