# Dataset preparation

## Static CIFAR-10

Run the static script from the package root. It changes to `static_cifar10/`,
where the original loader downloads CIFAR-10 into `static_cifar10/data/`.
The loader uses the original CIFAR-10 train/test splits. Inputs default to
224 x 224, as in the recovered training arguments.

Burst calibration in `eval_cifar10_snn_checkpoint.py` uses a training-loader
dataset. Dataset and model code are included, not the downloaded images.

## CIFAR10-DVS and N-Caltech101

The original experiments used pre-integrated NPZ frame caches. Expected layout:

```text
DATA_ROOT/
  CIFAR10DVS/
    frames_number_16_split_by_number/
      <class>/<sample>.npz
  NCaltech101/
    frames_number_16_split_by_number/
      <class>/<sample>.npz
```

NPZ files contain a `frames` array of shape `[time, polarity, height, width]`.
The polarity dimension has length two. The event loader implements its own
deterministic split and ordering; do not substitute a different split when
comparing runs. Consult `EventFrameStaticDataset` in `synax_neuro.py` for details.

The original frame-cache input is interpolated to nine temporal bins and
flattened to 18 ANN input channels, with spatial size 128 x 128. This input
representation is distinct from the eight SNN simulation timesteps (`T=8`).
Do not describe the ANN input cache as eight bins solely because `T=8`.

The package contains the cache reader but no new raw-event preprocessing
pipeline. To reproduce the input representation, prepare the named cache with
the original dataset tooling or use an existing cache. Neither private files
nor public datasets are included in the release.
