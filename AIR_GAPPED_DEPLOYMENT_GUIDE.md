# Air-Gapped Network Deployment Guide

How to install this repository on a machine that has no internet access (an "air-gapped"
machine). You prepare everything on a machine with internet access, carry one archive
over, and install from it.

For a machine with internet access, use the [README](README.md) instead.

> **Coming from the installation guide of the DeepPrint repository?** You need only three
> things from this guide: the clone (Step 1.1), the model files `CoarseNet.h5` and
> `FineNet.h5` (Step 1.2), and the Python packages (Steps 1.3 and 1.4). Put the packages
> **and** the file `requirements-locked.txt` into the package directory that the DeepPrint
> guide names, then go back to that guide. Phases 2 and 3 below are for installing this
> repository on its own.

## Overview

| What is transferred | Size |
|---|---|
| Python packages (109 wheel files, locked versions) | 3.6 GB |
| Model weights, all four files | 1.6 GB |
| Source code of this repository | under 10 MB |
| **Archive in total** | **about 5.2 GB** |

2.7 GB of the packages are NVIDIA GPU libraries. The dependency list of this repository
(`tensorflow[and-cuda]`) includes them, so they are downloaded and installed on a machine
without a GPU as well.

If you only need minutiae positions and directions (which is all DeepPrint training
uses), `CoarseNet.h5` and `FineNet.h5` are enough: 0.7 GB instead of 1.6 GB of weights.

## Prerequisites

Both machines:

- Linux on x86_64. Nothing else was tried.
- **The same Python version, 3.11 or later, down to the second number** (for example 3.12
  on both). Check with `python3 --version`. The package files are specific to it.

Machine with internet access:

- `git`, [uv](https://docs.astral.sh/uv/), and `pip` for that `python3`
  (`python3 -m pip --version` must work).
- About 20 GB of free disk space: the installed environment (6.4 GB), the weights
  (1.6 GB), the package files (3.6 GB, and the same again in temporary files while
  they download), and the archive (5.2 GB).

Air-gapped machine:

- `python3` with the `venv` module and `pip`. On Debian and Ubuntu these are the system
  packages `python3-venv` and `python3-pip`.
- About 18 GB of free disk space: the archive (5.2 GB), its unpacked content (the same
  again) and the installed environment (about 6.7 GB). The archive can be deleted after
  unpacking.
- uv is not needed there.

## Phase 1: Preparation on the internet-connected machine

### Step 1.1: Clone the repository and install it

```bash
git clone https://github.com/craig-forefront/minutiae_extractor
cd minutiae_extractor
uv sync
uv pip install -e .
```

This is the installation of the README. It is needed here to download the weights and to
write the package list. All following steps start in this `minutiae_extractor` directory.

### Step 1.2: Download the model weights

```bash
uv run python scripts/install_model.py
ls -l models
```

`models/` must then hold `CoarseNet.h5` (81 MB), `FineNet.h5` (654 MB),
`ClassifyNet_6_classes.h5` (619 MB) and `CoreNet.weights` (256 MB).

### Step 1.3: Write the list of locked packages

```bash
uv export --frozen --no-hashes --no-emit-project -o requirements-locked.txt
```

This writes the exact package versions of `uv.lock` (for example TensorFlow 2.20.0 and
Keras 3.11.3) into `requirements-locked.txt`, a file that `pip` can read. Do not use
`uv pip compile pyproject.toml` for this: it picks the newest versions instead of the
locked ones.

### Step 1.4: Download All Python Dependencies

```bash
python3 --version    # must show the same 3.x as on the air-gapped machine
python3 -m pip download --no-cache-dir -r requirements-locked.txt -d ../offline_packages
```

This puts 109 files (3.6 GB) into the directory `offline_packages` next to
the clone. It took 4 minutes here. `pip` first collects the files in a temporary
directory, so about twice that size has to be free while it runs.

The files are for the Python version and platform that ran the command. That is why
`python3` on this machine has to match the air-gapped machine. (uv has no command that
downloads package files; `uv pip download` does not exist.)

### Step 1.5: Pack the transfer archive

```bash
cd ..
tar cf minutiae_extractor_airgap.tar --exclude=.git --exclude=.venv --exclude=__pycache__ \
    minutiae_extractor offline_packages
sha256sum minutiae_extractor_airgap.tar > minutiae_extractor_airgap.tar.sha256
```

The archive holds the source code with `models/` and `requirements-locked.txt`, and the
package files. It is not compressed, because model weights and package files hardly
shrink. The second command writes a checksum, a short fingerprint of the archive that
shows on the other side whether the file arrived unchanged.

## Phase 2: Transfer

Copy `minutiae_extractor_airgap.tar` and `minutiae_extractor_airgap.tar.sha256` to the
air-gapped machine by whatever route your organisation allows (for example a USB drive
with at least 5.2 GB free). Put both files into the directory where the
installation should live.

## Phase 3: Installation on the air-gapped machine

```bash
sha256sum -c minutiae_extractor_airgap.tar.sha256    # must print: minutiae_extractor_airgap.tar: OK
tar xf minutiae_extractor_airgap.tar
cd minutiae_extractor

python3 -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links ../offline_packages -r requirements-locked.txt
pip install --no-index --find-links ../offline_packages --no-deps -e .
```

- `--no-index` forbids any access to the internet; `--find-links` names the directory to
  take the packages from.
- The first `pip install` installs the dependencies. The second makes this repository
  importable as `fingerflow`.
- The environment takes about 6.7 GB: everything except the NVIDIA libraries was 2.8 GB
  here, and the NVIDIA libraries unpack to 3.9 GB (measured in the environment of
  Step 1.1).
- In every new shell, run `source .venv/bin/activate` in this directory again before you
  use the package.

## Verification

In the `minutiae_extractor` directory, with the environment activated:

```bash
python - <<'EOF'
import cv2
from fingerflow.extractor import Extractor, direction_sense

extractor = Extractor("models/CoarseNet.h5", "models/FineNet.h5", None, None)
image = cv2.imread("sample/sample_fingerprint.tif")
minutiae = extractor.extract_minutiae(image)["minutiae"]
print(len(minutiae), "minutiae; columns:", list(minutiae.columns))
EOF
```

After many TensorFlow log lines the last line must be:

```
51 minutiae; columns: ['x', 'y', 'angle', 'score', 'class', 'direction_confidence']
```

This took under a minute on 4 CPU cores. It checks what the DeepPrint extraction script
needs: the import, an `Extractor` that accepts `None` for ClassifyNet and CoreNet, and the
`direction_confidence` column. To check all four networks as well, run the quick start of
the [README](README.md#quick-start).

On a machine without a GPU, TensorFlow prints a message that no CUDA device was found.
That is expected and not an error.

## Troubleshooting

Only failures that were seen while this guide was written or tested are listed.

**`pip install` stops with "Could not find a version that satisfies the requirement" or
"No matching distribution found".** The package files do not fit this Python. Compare
`python3 --version` on both machines; if the second number differs, repeat Step 1.4 with
the matching Python and transfer again.

**`ModuleNotFoundError: No module named 'fingerflow'`.** Either the environment is not
activated (`source .venv/bin/activate`) or the second `pip install` line of Phase 3 was
not run.

**`ValueError: File format not supported: filepath=...`** when the `Extractor` is made.
The path of a model file is wrong or the file is missing. Check `ls -l models`.

**Step 1.4 stops part of the way with a long error.** The disk was full. `pip` needs
about twice the size of the packages while it downloads. Free space, delete
`../offline_packages` and run the step again.

## What was and was not tested

Tested on 2026-11-17 on one Linux x86_64 machine with 4 CPU cores, no GPU, Python 3.13
and uv 0.11.32:

- Steps 1.1 to 1.4 as written. In Step 1.2 three of the four weight files were already
  present, so only `CoarseNet.h5` was downloaded.
- Phase 3 from `python3 -m venv .venv` on, with the network blocked for `pip`. The first
  `pip install` was run as written with `--dry-run` added: `pip` found all 109 packages
  in `offline_packages`. It was then run for real without the 12 NVIDIA libraries, for
  lack of disk space (101 packages, 50 seconds). The second `pip install` and the
  verification were run as written and gave the output shown.
- Step 1.5, the checksum test and `tar xf` with a small stand-in for the package
  directory and the weights, to check the commands. The full archive was not built, for
  lack of disk space; its size above is the sum of its parts.

Not tested:

- A second, truly disconnected machine. Phase 3 ran on the machine that had done Phase 1,
  in the clone of Step 1.1 after its `.venv` was deleted, not in an unpacked archive.
- Unpacking and loading the NVIDIA libraries from `offline_packages` (see above).
- Any GPU. TensorFlow was never run on a GPU for this port, so nothing is known about
  speed, drivers or supported graphics cards. The NVIDIA libraries come with the Python
  packages; an NVIDIA driver on the system would be needed in addition.
- Other platforms (macOS, Windows, ARM) and other Python versions than 3.13.
- A minimal server installation. `opencv-python` needs the system library `libGL.so.1`;
  it was present here. If `import cv2` fails with a message about `libGL.so.1`, that
  system library is missing (package `libgl1` on Debian and Ubuntu, `mesa-libGL` on Red
  Hat family systems); this case was not reproduced here.
- The notebook `demo.ipynb` on the air-gapped machine. Jupyter is not among the
  dependencies and is not in the archive.

How long the extraction takes and how much memory it needs is in
[docs/measurements.md](docs/measurements.md#6-time-and-memory-on-a-cpu).
