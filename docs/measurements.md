# Measurements and their limits

This document holds every figure that was measured for this extractor, the sample each
figure comes from, and what the figures do not cover. How to use the options is in
[usage.md](usage.md); installation and a first run are in the [README](../README.md).

Contents:

1. [What was measured, and on what](#1-what-was-measured-and-on-what)
2. [What was not measured](#2-what-was-not-measured)
3. [Direction sense](#3-direction-sense)
4. [Finding more minutiae: `crop_to_print` and `reweight_score`](#4-finding-more-minutiae-crop_to_print-and-reweight_score)
5. [Image resolution](#5-image-resolution)
6. [Time and memory on a CPU](#6-time-and-memory-on-a-cpu)

## 1. What was measured, and on what

- **Data.** NIST Special Database 302 (SD 302), a public collection of fingerprints in
  which a fingerprint examiner marked the minutiae by hand. Two kinds of prints were used:
  *exemplars* (prints taken on purpose: *rolled* prints, where the finger is rolled from
  nail edge to nail edge, and *slap* prints, where it is pressed flat) and *latents*
  (prints left on a surface, usually partial and of poor quality).
- **Resolution.** All images were at 500 ppi (pixels per inch), either as captured or
  downsampled from 1000 ppi. The exception is the resolution test in section 5.
- **How a detection is compared with the examiner.** A detected minutia counts as right
  if an examiner mark lies within 10 px of it. In section 4 each mark can be paired with
  one detection only.
  - "Reported that are marked" is the share of detected minutiae that have such a mark
    (also called precision).
  - "Marked that are found" is the share of examiner marks that have a detection (also
    called recall).
- **Hardware.** No graphics card (GPU) was used for any figure. The number of CPU cores
  is recorded only for some of the timings; section 6 says for which.
- **Samples differ between sections.** Each section names its prints, subjects and
  minutiae. Figures from different samples are not expected to be equal: for example, the
  default extractor finds 41% of the marked minutiae on the 259 exemplar prints of
  section 3 and 42.2% on the 130 prints of section 4.

## 2. What was not measured

- Speed or memory use on a GPU. No GPU was available, and the package was never run on
  one for this port.
- Throughput on more than a few hundred prints, running several processes side by side,
  or any behaviour at the scale of millions of images.
- Prints from sensors other than the five capture devices of SD 302, and images that hold
  several fingers (covered by a synthetic test only).
- How often the minutia type (`class` column, from ClassifyNet) is right.
- How well the core detection (CoreNet) works.
- The effect of any option on a matcher, or on a DeepPrint model trained with these
  minutiae.
- All marks come from one examiner. Some detections without a mark are real minutiae
  that the examiner did not mark, so the "reported that are marked" figures are lower
  bounds.

## 3. Direction sense

Background: CoarseNet (the network that proposes minutiae) reports the ridge orientation
of a minutia reliably, but which of the two opposite directions it reports is not
reliable. By default the extractor takes the direction sense from the ridge pattern in the
image instead (`refine_direction=True`, see [usage.md](usage.md#direction-sense)).

Sample: examiner-marked minutiae from SD 302 at 500 ppi (native or downsampled from
1000 ppi). Counted are detected minutiae that have an examiner mark within 10 px.

| | Sense agrees with the examiner | CoarseNet | Reverses under a small crop or 180 degree rotation | CoarseNet |
|---|---|---|---|---|
| Rolled and slap exemplars (259 prints, 119 subjects, 11,577 minutiae) | 96.6% | 68.6% | 4.0% | 31% |
| Latents (325 prints, 118 subjects, 1,863 minutiae) | 91.8% | 59.3% | 6.0% | 43% |

The columns headed "CoarseNet" give the same figure for the direction that CoarseNet
reports, which is what you get with `refine_direction=False`.

Some of these pairings are chance matches (about 6% on exemplars, 9% on latents), so the
true agreement is a little higher. On synthetic (SFinGe) prints the step is right for
about 98%.

**`direction_confidence`.** This column tells how clear the ridge pattern was. A value of
at least 0.25 is reached by 92% of these minutiae on exemplars and 88% on latents, and the
sense is then right for 99% and 95%. Below 0.25 it is much less reliable, and below 0.1 no
better than CoarseNet.

**`image_dpi`.** The setting is tolerant. On the exemplar prints above (which are at
500 ppi), values from 450 to 600 gave 96% agreement, 400 and 650 gave 95%, and at 800 it
fell to 87%.

**Limits of this evidence.**

- All marks come from one examiner.
- The filter sizes and the 15 degree threshold of the step were chosen on half of the
  exemplar subjects, so the exemplar figures are slightly optimistic. The latents were
  not used for the choice.
- The figures cover only minutiae that the extractor finds. With the default settings it
  finds about 41% of the marked minutiae on these exemplars (76% of its detections have a
  mark) and 29% on these latents (64%), where it returns a median of 7 minutiae per
  print.

## 4. Finding more minutiae: `crop_to_print` and `reweight_score`

Both options are off by default. What they do is described in
[usage.md](usage.md#finding-more-minutiae).

### The print box of `crop_to_print`

On the 1,708 SD 302 exemplar records at 500 ppi, none of the 191,610 examiner-marked
minutiae fell outside the box that `crop_to_print` cuts out. The box kept about half of
the image area (median 48%).

### Sample and method

SD 302 rolled and slap exemplars at 500 ppi, compared with the minutiae marked by an
examiner. A detected minutia counts as right if it can be paired one-to-one with a mark
within 10 px.

The subjects were split in two halves. The settings of `reweight_score` (the defaults of
`candidate_threshold`, `fine_net_weight` and `score_threshold`) were chosen on 40 prints
of one half (32 subjects). All figures below are from the other half: 130 prints
(113 rolled, 17 slap) of 60 subjects with 14,202 marked minutiae. The ranges in brackets
are 95% intervals from resampling the subjects 2,000 times.

### Results on the 130 prints

| Options | Reported that are marked | Marked that are found | Direction sense agrees | Mean distance to the mark |
|---|---|---|---|---|
| none (default) | 76.8% (73.6 to 79.4) | 42.2% (38.8 to 45.1) | 96.8% | 4.14 px |
| `crop_to_print` | 82.4% (79.9 to 84.7) | 44.7% (41.3 to 47.6) | 97.0% | 4.00 px |
| `reweight_score` | 78.0% (74.4 to 81.0) | 48.0% (45.1 to 50.5) | 96.5% | 4.12 px |
| both | 81.7% (78.6 to 84.3) | 50.3% (47.3 to 53.0) | 96.8% | 4.01 px |
| both, `score_threshold=0.575` | 83.4% (80.5 to 85.9) | 48.3% (45.2 to 51.1) | 97.0% | 3.98 px |

The ranges of the rows overlap because prints differ a lot from one another. The change on
the same prints is known more precisely:

| Change, in percentage points | Reported that are marked | Marked that are found |
|---|---|---|
| default to `crop_to_print` | +5.6 (+4.5 to +6.9) | +2.5 (+1.8 to +3.3) |
| default to `reweight_score` | +1.2 (+0.3 to +2.2) | +5.8 (+4.9 to +6.8) |
| default to both | +4.9 (+3.9 to +6.0) | +8.2 (+7.0 to +9.3) |
| `crop_to_print` to both | -0.7 (-1.8 to +0.3) | +5.6 (+4.8 to +6.5) |
| `crop_to_print` to both with `score_threshold=0.575` | +1.0 (+0.1 to +1.9) | +3.6 (+2.9 to +4.4) |

### What this shows

- Cropping helps on both counts, and both options together find about a fifth more of the
  marked minutiae than the default (42.2% to 50.3%) with fewer unmarked detections.
- The re-weighting finds more of the marked minutiae on every group of prints looked at
  (about 5 to 8 points). What it does to the precision depends on the capture device, as
  the next table shows: on these 130 prints as a whole it costs little against cropping
  alone at 0.55 (-0.7 points, the range includes zero) and nothing at 0.575, but on one
  of the devices it costs about 3 points at 0.55.
- The options hardly change how often the direction sense agrees with the examiner
  (changes between -0.3 and +0.2 points on the 130 prints, all ranges include zero; see
  the further prints below for a small loss) and do not make the position worse (mean
  distance to the mark 4.14 px by default, 4.01 px with both).
- No print came back empty with any setting.
- Cropping does nothing for the slap images, which have no margin. With both options the
  share found there rose from 34.0% to 39.6% (+3.8 to +8.0 points) and the precision went
  from 77.7% to 76.2%, a change that 17 prints cannot settle (-4.8 to +1.9 points).
- The re-weighting sends about 2.4 times as many patches through FineNet (18,956 against
  7,800 on the 130 prints), which costs time; cropping saves time on prints with a
  margin. The timings are in [section 6](#6-time-and-memory-on-a-cpu).

### Split by capture device

The effect of the re-weighting on precision differs between capture devices. The table
splits the rolled prints of the 130 by the device code of the collection (C, U, V) and
adds 18 further rolled prints of 18 subjects that were measured afterwards as an
independent check (the last two rows; 12 of these subjects appear nowhere in the 130
prints or in the 40 used to choose the settings). The first three columns are "reported
that are marked"; the changes are in percentage points of it, with the same 95% intervals
as above. The split by device was made after the results were seen, so it describes these
prints and is not a tested prediction.

| Rolled prints | Cropping alone | Both | Both, 0.575 | Cropping to both | Cropping to both, 0.575 | Default to `reweight_score` |
|---|---|---|---|---|---|---|
| Device C, 50 prints, 28 subjects | 81.6% | 82.8% | 84.2% | +1.2 (+0.1 to +2.5) | +2.6 (+1.6 to +3.9) | +4.3 (+3.2 to +5.7) |
| Device U, 18 prints, 18 subjects | 85.2% | 84.5% | 85.6% | -0.7 (-2.5 to +1.1) | +0.5 (-1.4 to +2.3) | +1.1 (-1.5 to +3.5) |
| Device V, 45 prints, 45 subjects | 83.0% | 80.2% | 82.5% | -2.8 (-4.8 to -1.2) | -0.5 (-2.2 to +0.9) | -1.7 (-3.9 to +0.2) |
| Further prints, device U, 6 prints | 83.9% | 83.6% | 83.8% | -0.3 (-2.4 to +1.8) | -0.1 (-2.2 to +1.8) | +5.7 (+1.8 to +9.8) |
| Further prints, device V, 12 prints | 83.8% | 79.6% | 81.4% | -4.2 (-6.0 to -2.4) | -2.4 (-3.5 to -1.1) | -3.2 (-5.5 to -0.9) |

On device V the re-weighting lowers the precision that cropping reaches by about 3 to 4
points at 0.55, and on the 12 further prints by 2.4 points even at 0.575; on device C it
raises it. In every row the share of marked minutiae found rises by 5.5 to 8.4 points at
0.55 and by 3.1 to 6.4 points at 0.575, and both options together stay ahead of the default
on both counts (precision +2.7 to +10.3 points).

On the 18 further prints the direction sense agreed with the examiner slightly less often
with both options than with cropping alone: 95.4% against 96.7%, a change of -1.3 points
(-2.1 to -0.5), where the 130 prints show -0.2 (-0.6 to +0.1). The further prints were run
through the unmodified package.

With all 148 prints taken together (72 subjects, 16,207 marked minutiae), cropping to both
changes precision by -1.0 points (-2.0 to -0.1) and the share found by +5.9 (+5.1 to
+6.7); with 0.575 the changes are +0.7 (-0.2 to +1.5) and +3.9 (+3.2 to +4.6).

### Limits of this evidence

- All marks come from one examiner. Some unmarked detections are real minutiae the
  examiner did not mark, so the precision figures are lower bounds.
- The default figures come from stored earlier runs of the default path, which fresh runs
  reproduced on the 12 prints that were re-run (same minutiae, scores equal to within
  0.000001).
- The measurement called FineNet and ClassifyNet through a faster Keras call than the
  package uses, which gave the same minutiae on those 12 prints (1,438 minutiae over the
  three option settings).
- The prints come from five capture devices of one collection. The effect on other
  sensors, on latent prints, on the minutia type, and on a matcher or a network trained
  from these minutiae was not measured.
- Images with several fingers were not available: the box encloses every textured region
  of at least a tenth of the largest one, which is covered by a synthetic test only.
- The slap figures rest on 17 prints: the collection has only 33 marked slap images, so
  no further slap sample was available for the independent check.
- On latent prints cut close to the print, cropping has nothing to remove.

## 5. Image resolution

The minutiae networks work at one fixed scale, about 500 ppi, and the extractor does not
rescale the image. On SD 302 prints:

| What was passed in | Marked that are found | Same prints at 500 ppi |
|---|---|---|
| 1000 ppi images, unchanged (7 exemplars) | under 10% | 46% |
| 1000 ppi images, unchanged (12 latents) | under 10% | 24% |
| 500 ppi images shrunk to 80% of their size (18 exemplars) | 31% | 41% |
| 500 ppi images shrunk to 80% of their size (30 latents) | 18% | 25% |

These samples are small. They show that the resolution matters a great deal; they do not
give a precise loss for a given resolution error.

## 6. Time and memory on a CPU

All times are from a CPU without a GPU, with other jobs running at the same time, so they
are rough. FineNet, and ClassifyNet if it is loaded, are called once per minutia, so the
time for a print grows with the number of minutiae.

**Six rolled SD 302 prints, four option settings** (one CPU thread; which networks were
loaded and the number of cores of the machine are not recorded). The measurement they belong to called
FineNet and ClassifyNet through a faster Keras call than the package uses
([section 4, limits](#limits-of-this-evidence)), so the package itself may take longer:

| Options | Total for the six prints |
|---|---|
| none (default) | 454 s |
| `crop_to_print` | 230 s |
| `reweight_score` | 517 s |
| both | 336 s |

That is about 76 s per print by default. Cropping saves time on prints with a margin,
because CoarseNet and the orientation step have less to process. The re-weighting costs
time, because more patches go through FineNet.

**Eight prints, with and without ClassifyNet.** Quoted from the measurements of the
DeepPrint repository, made with its batch script (one process on 4 CPU cores, 8 prints at 500 ppi with 369 minutiae, CoreNet not
loaded): 153 s without ClassifyNet and 263 s with it, with the same 369 positions, angles
and confidences in both runs.

**The sample image of this repository** (`sample/sample_fingerprint.tif`, 296 x 560 px,
4 CPU cores, time of the `extract_minutiae` call only; the first call in a process is
slower than later ones):

| Networks and options | Minutiae | Time per call |
|---|---|---|
| All four networks, default options | 51 | 57 s (first call), 39 s (second call) |
| CoarseNet and FineNet only, default options | 51 | 21 s, 32 s (first calls of two runs), 26 s (second call) |
| CoarseNet and FineNet only, `crop_to_print` and `reweight_score` | 60 | 31 s (one call) |

The spread between the calls of the second row shows how rough these times are. Starting
TensorFlow and loading the networks comes on top, once per process: 17 s with four
networks and 10 s with two.

**Memory.** Measured only for the sample image above: the peak memory of the Python
process was 2.4 GB with four networks and 1.8 GB with two. Larger images were not
measured.

**Not measured:** any run on a GPU, more than one process at a time, and images much
larger than the ones above.
