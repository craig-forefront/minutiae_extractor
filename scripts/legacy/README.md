# Legacy scripts (not maintained)

These files come from the original FingerFlow project and were moved here unchanged.
They are kept for reference only:

- Nearly all of them contain hard-coded paths of their author's machine (`/home/jakub/...`) and take
  no command-line arguments.
- They name model files that this repository does not have (`ClassifyNet.h5`,
  `yolo-kernel_best.weights`), and some import modules that do not exist
  (`scripts.matcher_evaluation`) or packages that are not installed (`seaborn`).
- None of them was run for this port, and no test or document uses them.
- The two files named `*_test.py` are timing scripts, not tests. The unit tests are in
  `test/`.

What you probably want instead:

- Download the model weights: `scripts/install_model.py` (one directory up).
- Extract minutiae from one image: see the [README](../../README.md).
- Extract minutiae from a directory of images for DeepPrint training:
  `scripts/extract_minutiae.py` in the
  [fixed-length-fingerprint-extractors](https://github.com/craig-forefront/fixed-length-fingerprint-extractors)
  repository.
