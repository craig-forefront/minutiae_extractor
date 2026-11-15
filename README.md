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
| `score` | Extraction confidence score (0-1) |
| `class` | Minutiae type as a numeric code: `0` ending, `1` bifurcation, `2` fragment, `3` enclosure, `4` crossbar, `5` other |
| `direction_confidence` | Confidence of the direction sense, typically 0-1 and occasionally higher. On real test prints the sense was right for about 98% of minutiae at 0.25 and above, and much less often below that (see Direction sense). Present only when `refine_direction` is on |

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
more than 25 degrees from the orientation measured in the image, the measured one is used.

`image_dpi` sets the scale of this step: pass the resolution of the image. What matters is
the ridge period in pixels. The default of 500 suits ordinary 500 ppi prints, whose ridges
are about 8 to 11 px apart; for other images a suitable value is roughly
`500 * period_px / 9.5`. The setting is tolerant: on the real prints described below, values
from 350 to 550 gave 95 to 96% and 300 and 600 gave 93%; at 700 it fell to 87%.

Measured against examiner-marked minutiae on 259 rolled and slap prints of 119 subjects from
NIST SD 302 (500 ppi, native or downsampled from 1000 ppi), the sense agreed with the
examiner for 95.7% of the detected minutiae that have an examiner mark within 10 px, against
68.6% for CoarseNet. About 6% of such pairings are chance matches, so the true figures are a
little higher for this step and about 70% for CoarseNet. When the image is cropped by a few
pixels or rotated by 180 degrees, 4.8% of the re-found minutiae reverse direction, against
31% for CoarseNet. On synthetic (SFinGe) prints the step is right for about 97%.

`direction_confidence` tells how clear the pattern was. On the same data 90% of the minutiae
had a value of at least 0.25 and the sense was right for 98% of those; below 0.25 it is
much less reliable, and below 0.1 no better than CoarseNet.

Limits of this evidence: all marks come from one examiner, the figures cover only minutiae
the extractor finds (it finds about 41% of the marked ones, at 76% precision), and latent
prints have not been measured beyond the two latent images that ship with MinutiaeNet, where
CoarseNet was right for 24 of 25 minutiae and the earlier, longer filters of this step for
21 of 25. Pass `refine_direction=False` to get the CoarseNet direction unchanged.

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
Extractor(coarse_net, fine_net, classify_net, core_net, refine_direction=True, image_dpi=500)
```

**Parameters:**
- `coarse_net` (str): Path to the coarse network model
- `fine_net` (str): Path to the fine network model
- `classify_net` (str): Path to the classification network model
- `core_net` (str): Path to the core detection network model
- `refine_direction` (bool): Take the direction sense of the minutiae from the image instead of from CoarseNet (default `True`)
- `image_dpi` (float): Resolution of the input images, used by the direction step only (default `500`)

#### Methods

##### `extract_minutiae(image_data)`

Extracts minutiae points and detects fingerprint cores from an input image.

**Parameters:**
- `image_data` (numpy.ndarray): 8-bit (`uint8`) image with dark ridges on a light background, e.g. the output of `cv2.imread()`. Grayscale (2D), BGR and BGRA arrays are accepted

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