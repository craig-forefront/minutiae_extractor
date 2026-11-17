# Using the extractor: options and reference

This document explains each option of the `Extractor`, when to use it, and what it changes
in the result. Installation and a first run are in the [README](../README.md). The
measurements behind every figure quoted here, and their limits, are in
[measurements.md](measurements.md).

Contents:

1. [Preparing the image](#preparing-the-image)
2. [Direction sense](#direction-sense)
3. [Finding more minutiae](#finding-more-minutiae)
4. [Leaving out ClassifyNet and CoreNet](#leaving-out-classifynet-and-corenet)
5. [Running over many images, and use with DeepPrint](#running-over-many-images-and-use-with-deepprint)
6. [Reference](#reference)

Terms used here: a *minutia* (plural *minutiae*) is a point where a ridge of the
fingerprint ends or splits. *ppi* is pixels per inch, the resolution of the image.
*CoarseNet* proposes minutiae for the whole image, *FineNet* scores the image patch around
each proposed minutia, *ClassifyNet* assigns a type to each minutia, and *CoreNet* detects
the core (the centre of the ridge pattern).

## Preparing the image

`extract_minutiae` expects a numpy array that is:

- **8-bit** (`uint8`, values 0 to 255). Any other type raises `ValueError`.
- **Grayscale (2D), BGR or BGRA**, as `cv2.imread` returns it.
- **Dark ridges on a light background.** Some sensors deliver light ridges on a dark
  background; invert such an image first (`image = 255 - image`).
- **At about 500 ppi.** The minutiae networks work at this one scale and the extractor
  does not rescale anything. An image at another resolution has to be resampled first,
  for example:

  ```python
  import cv2

  ppi = 1000  # the true resolution of the file
  scale = 500 / ppi
  image = cv2.resize(image, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
  ```

  This matters a great deal: 1000 ppi prints passed in unchanged gave under 10% of the
  examiner-marked minutiae, against 46% for the same prints at 500 ppi
  ([measurements.md, section 5](measurements.md#5-image-resolution)).

If you extract minutiae for DeepPrint training with the script of the DeepPrint
repository, do **not** resample or invert the files yourself: the script does both (see
[below](#running-over-many-images-and-use-with-deepprint)).

## Direction sense

Every minutia has a direction (`angle`). CoarseNet reports the ridge orientation of a
minutia reliably, but which of the two opposite directions it reports is not reliable:
with the released weights it changes for roughly a quarter to a half of the minutiae when
the image is shifted by a few pixels or rotated.

By default the extractor therefore takes the direction sense from the ridge pattern in the
image (`refine_direction=True`). The pattern has one more ridge on one side of a minutia
than on the other, and the direction points to that side. Where the CoarseNet orientation
differs by more than 15 degrees from the orientation measured in the image, the measured
one is used.

On examiner-marked exemplar prints the sense then agreed with the examiner for 96.6% of
the minutiae (CoarseNet alone: 68.6%), on latents for 91.8% (59.3%). See
[measurements.md, section 3](measurements.md#3-direction-sense).

**`direction_confidence`** (a column of the result) tells how clear the pattern was. It is
typically between 0 and 1 and occasionally higher. On the test prints the sense was right
for 95 to 99% of the minutiae with a value of 0.25 or more, much less often below that,
and no better than CoarseNet below 0.1. To keep only the minutiae with a clear direction:

```python
sure = minutiae[minutiae["direction_confidence"] >= 0.25]
```

**`image_dpi`** sets the scale of this step only. It does not rescale the image and has
no effect on which minutiae are found. Resample the image to 500 ppi first and leave
`image_dpi` at 500. Do not pass the resolution of the original file here.

The one case for another value: the image you pass in is not truly at 500 ppi and you
cannot correct that. What matters to the step is the distance between neighbouring ridges
in pixels, which is about 8 to 11 px in ordinary 500 ppi prints. A suitable value is then
roughly `500 * period_px / 9.5`, where `period_px` is that distance in your image. The
setting is tolerant: on 500 ppi test prints, values from 450 to 600 gave the same
agreement with the examiner.

**Switching it off.** `refine_direction=False` returns the CoarseNet direction unchanged,
and the result has no `direction_confidence` column.

## Finding more minutiae

With the released weights and the default settings, the extractor finds well under half of
the minutiae that a fingerprint examiner marks (42% on the test prints), and about a
quarter of what it reports has no examiner mark nearby. Two options of the constructor
improve on this. Both are off by default.

### `crop_to_print=True`

Runs the minutiae steps on the part of the image that holds the print, and reports `x` and
`y` in the full image, as usual.

Why it helps: CoarseNet scales its input by the mean and variance of the whole image, so a
large empty margin, as rolled prints on a card have, changes what it sees of the print.

Details:

- The print is taken to be where the image has texture. The box around it is widened by
  16 px.
- If the box is narrower or lower than 128 px, or no textured region is found, the whole
  image is used. No message is printed in that case.
- The core detection still uses the whole image.
- On images without a margin (slap prints in the test, latents cut close to the print) it
  has nothing to remove.

### `reweight_score=True`

Gives more say to FineNet. By default a minutia needs a CoarseNet score above 0.5 and
FineNet can hardly change the outcome. With the option:

1. every CoarseNet cell scoring above `candidate_threshold` (default 0.1) becomes a
   candidate;
2. its score is `(1 - fine_net_weight) * CoarseNet + fine_net_weight * FineNet`
   (`fine_net_weight` default 0.5);
3. candidates scoring at least `score_threshold` (default 0.55) are returned.

Two things change with it. Read them before you switch it on:

- **The `score` column means something else.** By default it is
  `(4 * CoarseNet + FineNet) / 5` and lies above 0.45 (lower only where fewer than five
  minutiae were found). With the option it is the mix above and lies at or above
  `score_threshold`. Scores from the two modes cannot be compared, and a threshold that
  later code applies to `score` has to be chosen again.
- **An image can come back with no minutiae.** The default lowers its threshold step by
  step when it finds fewer than five minutiae. The option does not.

It also takes more time, because more patches go through FineNet.

Raising `score_threshold` gives the same result as dropping the rows with a lower `score`
afterwards, so a stricter setting needs no second run:

```python
stricter = minutiae[minutiae["score"] >= 0.575]
```

### What was measured, in short

On 130 examiner-marked rolled and slap prints
([measurements.md, section 4](measurements.md#4-finding-more-minutiae-crop_to_print-and-reweight_score)):

| Options | Reported minutiae that have an examiner mark | Examiner marks that are found |
|---|---|---|
| none (default) | 76.8% | 42.2% |
| `crop_to_print` | 82.4% | 44.7% |
| both | 81.7% | 50.3% |
| both, keeping `score` of 0.575 or more | 83.4% | 48.3% |

### Choosing

```python
extractor = Extractor("models/CoarseNet.h5", "models/FineNet.h5", None, None,
                      crop_to_print=True, reweight_score=True)
```

- Both options together found more of the marked minutiae than the default on every
  group of rolled and slap prints tested. On the rolled prints of every device they also
  gave fewer unmarked detections. On the 17 slap prints the precision went from 77.7% to
  76.2%, a change that 17 prints cannot settle.
- Keep `score_threshold` at 0.55 to get the most minutiae. Keep only the rows with a
  `score` of at least 0.575 if detections without a mark matter more to you.
- Do not take 0.575 as a value that keeps the precision of cropping alone on every
  sensor: it did on the 130 prints as a whole and did not on one capture device (device V)
  in the further prints. If unmarked detections matter most, note that on two of the
  three devices cropping alone was as precise as both options, or more so.
- **Whether more minutiae give a better DeepPrint model was not measured.** Nor was the
  effect on other sensors or on latent prints.
- These figures come from one collection. Before a large run, check a sample of your own
  images from each sensor type with and without the options, at least by looking at the
  minutiae drawn on the prints (the notebook [demo.ipynb](../demo.ipynb) draws them), and
  choose `score_threshold` again if you have marked minutiae to compare with.

## Leaving out ClassifyNet and CoreNet

Pass `None` instead of a path:

```python
extractor = Extractor("models/CoarseNet.h5", "models/FineNet.h5", None, None)
```

- `classify_net_path=None`: the minutiae are not classified. The `class` column is `NaN`
  for every row, so a filter such as `minutiae["class"] == 0` selects nothing.
- `core_net_path=None`: no core is detected, and `result["core"]` is `None` (not an empty
  table).
- Position, angle, score and direction confidence are computed the same way with and
  without the two networks.
- It saves time: ClassifyNet is called once per minutia. In the one timing made, 8 prints
  took 153 s without ClassifyNet and 263 s with it
  ([measurements.md, section 6](measurements.md#6-time-and-memory-on-a-cpu)).
- It saves download and memory: the two files are 0.9 GB of the 1.6 GB of weights.

How often the `class` is right, and how well the core is found, was not measured for this
port. DeepPrint training uses neither.

## Running over many images, and use with DeepPrint

This repository has no batch tool. The `Extractor` handles one image per call; load it
once and call `extract_minutiae` for every image.

For DeepPrint training, use `scripts/extract_minutiae.py` of the
[fixed-length-fingerprint-extractors](https://github.com/craig-forefront/fixed-length-fingerprint-extractors)
repository (the "DeepPrint repository"). It reads a directory of images, calls this
package, and writes one `.mnt` file per image. Its training guide, linked from its README,
describes the options. What you need to know on this side:

- **What it needs from here:** this repository at `main` (commit `420b6e5` or later),
  importable as `fingerflow` (see the README), and the files `CoarseNet.h5` and
  `FineNet.h5`.
- **Give it the original files.** The script resamples every image to 500 ppi itself
  (you tell it the resolution of the files) and can invert light-ridge images. The
  training loads the images through the same preparation. Do not resample beforehand.
- **What reaches the `.mnt` file:** `x`, `y`, `angle`, `class` (written as -1 when
  ClassifyNet is left out) and `direction_confidence`. The conventions are the same on
  both sides, so nothing has to be converted: `x` is the column, `y` the row pointing
  down, and `angle` is in radians with the direction vector `(cos(angle), sin(angle))`.
- **What does not reach it:** `score` and the core. DeepPrint uses every minutia the
  extractor returns and reads only `x`, `y` and `angle`. A selection by `score` after the
  extraction is therefore not possible there.
- **The script runs the default extractor.** When this was written (DeepPrint repository
  at commit `b118637`), it called `Extractor(coarse, fine, classify_or_None, None,
  image_dpi=...)` and had no way to switch on `crop_to_print` or `reweight_score` or to
  set `score_threshold`. To use them there, `make_extractor` in
  `flx/preparation/minutiae_extraction.py` of that repository has to be changed, and all
  `.mnt` files of a dataset then have to be extracted again with the same settings.

Throughput on a large corpus was not measured here. The largest timed runs are a handful
of prints on a CPU ([measurements.md, section 6](measurements.md#6-time-and-memory-on-a-cpu)).
Time a few hundred of your own images on your own hardware before you plan a large run.

## Reference

The same information is in the docstrings: `help(Extractor)` in Python.

### `Extractor(...)`

```python
from fingerflow.extractor import Extractor

Extractor(coarse_net_path, fine_net_path, classify_net_path, core_net_path,
          refine_direction=True, image_dpi=500, crop_to_print=False,
          reweight_score=False, candidate_threshold=0.1, fine_net_weight=0.5,
          score_threshold=0.55)
```

| Parameter | Type, default | Meaning |
|---|---|---|
| `coarse_net_path` | str | Path of `CoarseNet.h5` |
| `fine_net_path` | str | Path of `FineNet.h5` |
| `classify_net_path` | str or `None` | Path of `ClassifyNet_6_classes.h5`. `None`: no classification, `class` is `NaN` |
| `core_net_path` | str or `None` | Path of `CoreNet.weights`. `None`: no core detection, `core` is `None` |
| `refine_direction` | bool, `True` | Take the direction sense from the image instead of from CoarseNet |
| `image_dpi` | number above 0, `500` | Scale of the direction step only; does not rescale the image. Leave at 500 |
| `crop_to_print` | bool, `False` | Run the minutiae steps on the print area only |
| `reweight_score` | bool, `False` | Give FineNet more weight; changes the meaning of `score` |
| `candidate_threshold` | 0 to 1, `0.1` | Used by `reweight_score` only |
| `fine_net_weight` | 0 to 1, `0.5` | Used by `reweight_score` only |
| `score_threshold` | 0 to 1, `0.55` | Used by `reweight_score` only |

The four paths have no default and `None` has to be written out for a network that is
left out. A wrong `image_dpi` or a threshold outside 0 to 1 raises `ValueError`.

### `extract_minutiae(image_data)`

Takes one image (see [Preparing the image](#preparing-the-image)) and returns a
dictionary with two entries.

`result["minutiae"]` is a pandas DataFrame with one row per minutia:

| Column | Meaning |
|---|---|
| `x` | Column of the minutia in the image passed in, in px |
| `y` | Row of the minutia, in px (y points down) |
| `angle` | Direction in radians, 0 to 2π. The direction vector is `(cos(angle), sin(angle))` in these axes. It points back along the ridge for an ending and into the valley between the two branches for a bifurcation (the convention of ISO 19794-2) |
| `score` | 0 to 1. By default `(4 * CoarseNet + FineNet) / 5`; with `reweight_score=True` a different mix that is not comparable (see [Finding more minutiae](#finding-more-minutiae)) |
| `class` | Minutia type as a number: `0` ending, `1` bifurcation, `2` fragment, `3` enclosure, `4` crossbar, `5` other (the meaning of the numbers is taken from the FingerFlow project). `NaN` without ClassifyNet |
| `direction_confidence` | How clear the ridge pattern was that the direction sense was taken from (see [Direction sense](#direction-sense)). Absent with `refine_direction=False` |

`result["core"]` is `None` without CoreNet. With CoreNet it is a pandas DataFrame with one
row per detected core box, which can have no rows:

| Column | Meaning |
|---|---|
| `x1`, `y1` | Left and top edge of the box, in px |
| `x2`, `y2` | Right and bottom edge of the box, in px |
| `score` | Detection score, 0 to 1 |
| `w`, `h` | Width and height of the box, in px |
