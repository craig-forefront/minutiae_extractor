# Minutiae Extractor
This patch updates and fixes issues in the legacy version of **MinutiaeNet** and **FingerFlow** improving compatibility with newer dependencies and ensuring smoother model loading, inference, and fingerprint preprocessing. It preserves the original architecture and training logic while resolving broken imports and outdated function calls.

![Detection Results](https://github.com/tourmii/minutiae_extractor/blob/main/assets/result.png)

## Features

- **Minutiae Detection**: Extract minutiae points from fingerprint images with high accuracy
- **Minutiae Classification**: Classify minutiae into 6 types:
  - Ending
  - Bifurcation
  - Fragment
  - Enclosure
  - Crossbar
  - Other
- **Core Detection**: Detect and localize fingerprint core regions
- **Easy Integration**: Simple API for seamless integration into your projects

### Minutiae Patches Examples

FingerFlow extracts detailed minutiae patches with precise location and orientation information:

![Minutiae Patches](https://github.com/tourmii/minutiae_extractor/blob/main/assets/patches.png)

## Requirements

- Python >= 3.11
- uv (recommended) or pip for package management

## Installation

### Step 1: Install Models

Before installing the package, you need to download the pre-trained models from Hugging Face:

```bash
python scripts/install_model.py
```

This will download the MinutiaeNet models to the `models/` directory.

### Step 2: Set Up Environment with uv

We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management.

#### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install FingerFlow

```bash
git clone https://github.com/tourmii/minutiae_extractor
cd minutiae_extractor

uv sync
```

Alternatively, if you prefer using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import cv2
import numpy as np
from fingerflow.extractor import Extractor

extractor = Extractor("coarse_net", "fine_net", "classify_net", "core_net")

image = cv2.imread("path/to/your/fingerprint.png")

extracted_minutiae = extractor.extract_minutiae(image)
```

### Detection Pipeline

The extraction process includes multiple stages:

![Detection Pipeline](https://github.com/tourmii/minutiae_extractor/blob/main/assets/minutiae_extract.png)

1. **Original Detection**: Initial minutiae detection identifies all potential points
2. **Otsu Thresholding**: Image segmentation to isolate fingerprint region
3. **Filtered Result**: Final minutiae with accurate orientations and classifications

### Working with Results

The `extract_minutiae` method returns an object containing:

#### Minutiae DataFrame

A Pandas DataFrame with extracted and classified minutiae points:

| Column | Description |
|--------|-------------|
| `x` | X coordinate of the minutiae point |
| `y` | Y coordinate of the minutiae point |
| `angle` | Direction of the minutia in radians (0 to 2π), in image axes: the direction vector is `(cos(angle), sin(angle))` with y pointing down. It points back along the ridge for an ending and into the valley between the two branches for a bifurcation (ISO 19794-2 convention) |
| `score` | Extraction confidence score (0-1). By default it is `(4 * CoarseNet + FineNet) / 5`; with `reweight_score=True` it is a different mix of the two network scores and is not comparable with a default score (see Finding more minutiae) |
| `class` | Minutiae type as a numeric code: `0` ending, `1` bifurcation, `2` fragment, `3` enclosure, `4` crossbar, `5` other |
| `direction_confidence` | Confidence of the direction sense, typically 0-1 and occasionally higher. On real test prints the sense was right for 95-99% of minutiae at 0.25 and above, and much less often below that (see Direction sense). Present only when `refine_direction` is on |

Example:
```python
minutiae_df = extracted_minutiae['minutiae']
print(f"Found {len(minutiae_df)} minutiae points")

high_confidence = minutiae_df[minutiae_df['score'] > 0.8]

endings = minutiae_df[minutiae_df['class'] == 0]
bifurcations = minutiae_df[minutiae_df['class'] == 1]
```

##### Direction sense

The ridge orientation that CoarseNet reports for a minutia is reliable, but which of the two
opposite directions it reports is not: with the released weights it changes for roughly a
quarter to a half of the minutiae when the image is shifted by a few pixels or rotated. By
default the extractor therefore takes the sense from the ridge pattern in the image
(`refine_direction=True`). The pattern has one more ridge on one side of a minutia than on
the other, and the direction points to that side. Where the CoarseNet orientation differs by
more than 15 degrees from the orientation measured in the image, the measured one is used.

`image_dpi` sets the scale of this step: pass the resolution of the image. What matters is
the ridge period in pixels. The default of 500 suits ordinary 500 ppi prints, whose ridges
are about 8 to 11 px apart; for other images a suitable value is roughly
`500 * period_px / 9.5`. The setting is tolerant: on the exemplar prints described below,
values from 450 to 600 gave 96%, 400 and 650 gave 95%, and at 800 it fell to 87%.

Measured against examiner-marked minutiae from NIST SD 302, at 500 ppi (native or
downsampled from 1000 ppi), counting detected minutiae that have an examiner mark within
10 px:

| | Sense agrees with the examiner | CoarseNet | Reverses under a small crop or 180 degree rotation | CoarseNet |
|---|---|---|---|---|
| Rolled and slap exemplars (259 prints, 119 subjects, 11,577 minutiae) | 96.6% | 68.6% | 4.0% | 31% |
| Latents (325 prints, 118 subjects, 1,863 minutiae) | 91.8% | 59.3% | 6.0% | 43% |

Some of these pairings are chance matches (about 6% on exemplars, 9% on latents), so the
true agreement is a little higher. On synthetic (SFinGe) prints the step is right for
about 98%.

`direction_confidence` tells how clear the pattern was. A value of at least 0.25 is reached
by 92% of these minutiae on exemplars and 88% on latents, and the sense is then right for
99% and 95%; below 0.25 it is much less reliable, and below 0.1 no better than CoarseNet.

Limits of this evidence: all marks come from one examiner; the filter sizes and the
15 degree threshold were chosen on half of the exemplar subjects, so the exemplar figures
are slightly optimistic (the latents were not used for the choice); and the figures cover
only minutiae the extractor finds. It finds about 41% of the marked minutiae on exemplars
(76% of its detections have a mark) and 29% on latents (64%), where it returns a median of
7 minutiae per print (two options that find more are described under Finding more
minutiae). Pass `refine_direction=False` to get the CoarseNet direction unchanged.

##### Finding more minutiae

With the released weights the extractor finds well under half of the minutiae that a
fingerprint examiner marks, and about a quarter of what it reports has no examiner mark
nearby. Two options of the `Extractor` constructor improve on this. Both are off by
default, so the default output is what it was before they existed.

`crop_to_print=True` runs the minutiae steps on the part of the image that holds the print
and reports `x` and `y` in the full image, as usual. CoarseNet (the network that proposes
minutiae) scales its input by the mean and variance of the whole image, so a large empty
margin, as rolled prints on a card have, changes what it sees of the print. The print is
taken to be where the image has texture; the box around it is widened by 16 px. On the
1,708 NIST SD 302 exemplar records at 500 ppi none of the 191,610 examiner-marked minutiae
fell outside the box, which kept about half of the image area (median 48%). The core
detection still uses the whole image.

`reweight_score=True` gives more say to FineNet, the network that looks at the image patch
around a proposed minutia. By default a minutia needs a CoarseNet score above 0.5 and
FineNet can hardly change the outcome. With the option, every CoarseNet cell scoring above
`candidate_threshold` (0.1) becomes a candidate, its score is
`(1 - fine_net_weight) * CoarseNet + fine_net_weight * FineNet` (weight 0.5), and candidates
scoring at least `score_threshold` (0.55) are returned. Two things change with it:

- **The `score` column means something else.** By default it is
  `(4 * CoarseNet + FineNet) / 5` and lies above 0.45 (lower only where fewer than five
  minutiae were found); with the option it is the mix above and lies at or above
  `score_threshold`. Scores from the two modes cannot be compared, and a threshold that
  later code applies to `score` has to be chosen again.
- The default lowers its threshold step by step when it finds fewer than five minutiae.
  The option does not, so an image can come back with no minutiae.

Raising `score_threshold` gives the same result as dropping the rows with a lower `score`
afterwards, so the stricter setting in the table needs no second run.

Measured on NIST SD 302 rolled and slap exemplars at 500 ppi against the minutiae marked by
an examiner. A detected minutia counts as right if it can be paired one-to-one with a mark
within 10 px. "Reported that are marked" is the share of detected minutiae with such a
mark (precision), "marked that are found" the share of marks with a detection (recall).
The subjects were split in two halves. The settings of `reweight_score` were chosen on 40
prints of one half (32 subjects); all figures below are from the other half: 130 prints
(113 rolled, 17 slap) of 60 subjects with 14,202 marked minutiae. The ranges are 95%
intervals from resampling the subjects 2,000 times.

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

What this shows:

- Each option helps against the default, and together they find about a fifth more of the
  marked minutiae (42.2% to 50.3%) with fewer unmarked detections.
- The re-weighting buys its extra minutiae at about the precision that cropping alone
  reaches: at 0.55 it may cost a little against cropping alone (-0.7 points, the range
  includes zero), at 0.575 it costs nothing and still finds 3.6 points more.
- Neither option changes how often the direction sense agrees with the examiner (changes
  between -0.3 and +0.2 points, all ranges include zero) or makes the position worse (mean
  distance to the mark 4.14 px by default, 4.01 px with both).
- No print came back empty with any setting.
- Cropping does nothing for the slap images, which have no margin. With both options the
  share found there rose from 34.0% to 39.6% (+3.8 to +8.0 points) and the precision went
  from 77.7% to 76.2%, a change that 17 prints cannot settle (-4.8 to +1.9 points).
- Cropping also saves time on prints with a margin, because CoarseNet and the orientation
  step have less to process, and the re-weighting costs time, because it sends about 2.4
  times as many patches through FineNet (18,956 against 7,800 on the 130 prints). Six of
  the rolled prints were timed with all four settings (one CPU thread, other jobs running,
  so the times are rough): 454 s in total by default, 230 s with `crop_to_print`, 517 s
  with `reweight_score`, 336 s with both.

For preparing rolled and slap prints as training data, switch both options on:

```python
extractor = Extractor("coarse_net", "fine_net", "classify_net", "core_net",
                      crop_to_print=True, reweight_score=True)
```

Keep `score_threshold` at 0.55 to get the most minutiae; keep only the rows with a `score`
of at least 0.575 if detections without a mark matter more. These figures come from one
collection, so before a large run check a sample of your own images with and without the
options, at least by looking at the minutiae drawn on the prints.

Limits of this evidence: all marks come from one examiner; some unmarked detections are
real minutiae the examiner did not mark, so the precision figures are lower bounds; the
default figures come from stored earlier runs of the default path, which fresh runs
reproduced on the 12 prints that were re-run (same minutiae, scores equal to within
0.000001); the measurement called FineNet and ClassifyNet through a faster Keras call
than the package uses, which gave the same minutiae on those 12 prints (1,438 minutiae
over the three option settings); the prints come from five capture devices of one collection, and the effect on other
sensors, on latent prints, on the minutia type and on a matcher or a network trained from
these minutiae was not measured. Images with several fingers were not available: the box encloses every
textured region of at least a tenth of the largest one, which is covered by a synthetic
test only. On latent prints cut close to the print, cropping has nothing to remove.

#### Core DataFrame

A Pandas DataFrame with detected fingerprint cores:

| Column | Description |
|--------|-------------|
| `x1` | Left coordinate of bounding box |
| `y1` | Top coordinate of bounding box |
| `x2` | Right coordinate of bounding box |
| `y2` | Bottom coordinate of bounding box |
| `score` | Core detection confidence score (0-1) |
| `w` | Width of bounding box |
| `h` | Height of bounding box |

Example:
```python
cores_df = extracted_minutiae['core']
print(f"Found {len(cores_df)} core region(s)")

if len(cores_df) > 0:
    best_core = cores_df.loc[cores_df['score'].idxmax()]
    print(f"Core at ({best_core['x1']}, {best_core['y1']})")
```

## API Reference

### Extractor

The main class for fingerprint feature extraction.

#### Constructor

```python
Extractor(coarse_net, fine_net, classify_net, core_net, refine_direction=True, image_dpi=500,
          crop_to_print=False, reweight_score=False, candidate_threshold=0.1,
          fine_net_weight=0.5, score_threshold=0.55)
```

**Parameters:**
- `coarse_net` (str): Path to the coarse network model
- `fine_net` (str): Path to the fine network model
- `classify_net` (str): Path to the classification network model
- `core_net` (str): Path to the core detection network model
- `refine_direction` (bool): Take the direction sense of the minutiae from the image instead of from CoarseNet (default `True`)
- `image_dpi` (float): Resolution of the input images, used by the direction step only (default `500`)
- `crop_to_print` (bool): Run the minutiae steps on the part of the image that holds the print and report coordinates in the full image (default `False`; see Finding more minutiae)
- `reweight_score` (bool): Give the FineNet score more weight, which finds more minutiae and changes the meaning of the `score` column (default `False`; see Finding more minutiae)
- `candidate_threshold`, `fine_net_weight`, `score_threshold` (float, 0 to 1): Settings of `reweight_score`, ignored while it is off (defaults `0.1`, `0.5`, `0.55`)

#### Methods

##### `extract_minutiae(image_data)`

Extracts minutiae points and detects fingerprint cores from an input image.

**Parameters:**
- `image_data` (numpy.ndarray): 8-bit (`uint8`) image with dark ridges on a light background, e.g. the output of `cv2.imread()`. Grayscale (2D), BGR and BGRA arrays are accepted

The minutiae networks work at one fixed scale, about 500 ppi, and the extractor does not
rescale the image (`image_dpi` is used by the direction step only). Resample images of any
other resolution to 500 ppi before calling. On NIST SD 302 prints, 1000 ppi images passed in
unchanged gave under 10% of the examiner-marked minutiae (7 exemplars, 12 latents), against
46% and 24% for the same prints at 500 ppi; shrinking 500 ppi images to 80% of their size
lowered the share found from 41% to 31% (18 exemplars) and from 25% to 18% (30 latents).

**Returns:**
- A dictionary containing:
  - `minutiae`: Pandas DataFrame with minutiae information
  - `core`: Pandas DataFrame with core detection information

## Project Structure

```
fingerflow/
├── models/              
├── minutiae_patches/    
├── sample/              
├── scripts/             
│   └── install_model.py 
├── src/                 
│   └── fingerflow/      
├── test/                
├── demo.ipynb           
├── pyproject.toml       
├── requirements.txt     
└── README.md            
```

## Development
### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use FingerFlow in your research, please cite:

https://arxiv.org/pdf/1712.09401

## Acknowledgments

Models are hosted on Hugging Face: [tourmaline05/MinutiaeNet](https://huggingface.co/tourmaline05/MinutiaeNet)

## Support

For issues, questions, or contributions, please open an issue on the project repository.

## License

[MIT](https://choosealicense.com/licenses/mit/)