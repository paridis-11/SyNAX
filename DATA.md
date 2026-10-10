# Dataset preparation

Run the preparation commands below from the repository root after installing
the dependencies in `requirements.txt`. Download links are listed in the
[README Data section](README.md#data).

## Static CIFAR-10

Run the static script from the package root. It changes to `static/`,
where the original loader downloads CIFAR-10 into `static/data/`.
The loader uses the original CIFAR-10 train/test splits. The training entry
point uses 32 x 32 inputs for ResNet-18/VGG and 224 x 224 inputs for ViTs.

For a manual download, select the **CIFAR-10 Python version** on the
[official page](https://www.cs.toronto.edu/~kriz/cifar.html) and extract
`cifar-10-batches-py/` into `static/data/`.

Burst calibration in `eval_cifar10_snn_checkpoint.py` uses a training-loader
dataset. Dataset and model code are included, not the downloaded images.

## ImageNet-1K

Set `DATASET=imagenet` in the static launcher and prepare class-organized images:

Follow the access instructions on the [ImageNet download page](https://www.image-net.org/download.php)
to obtain the ILSVRC2012 classification training images, validation images and
validation labels. Extract the per-class training archives and group validation
images into the same class folders using their ground-truth labels.

```text
static/data/imagenet/
  train/<class>/<image>
  val/<class>/<image>
```

The loader uses 1,000 classes and 224 x 224 inputs. The checkpoint evaluator
uses `--dataset imagenet`; ImageNet conversion thresholds use training data
and accuracy is evaluated on the validation split.

## CIFAR10-DVS and N-Caltech101

Use SpikingJelly 0.0.0.0.14 to download or convert the source recordings into
16-bin, event-count-split frame caches. CIFAR10-DVS can be downloaded
automatically from [Figshare](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671).
For N-Caltech101, download the following files from the
[dataset page](https://www.garrickorchard.com/datasets/n-caltech101) into
`data/NCaltech101/download/`:

- `Caltech101.zip`
- `Caltech101_annotations.zip`
- `ReadMe(Caltech101)-SINAPSE-G.txt`
- `ReadMe(Caltech101).txt`

Prepare the frame caches:

```python
import numpy as np
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_caltech101 import NCaltech101

# Compatibility with the NumPy alias used by SpikingJelly 0.0.0.0.14.
np.bool = np.bool_

CIFAR10DVS(root="data/CIFAR10DVS", data_type="frame",
           frames_number=16, split_by="number")
NCaltech101(root="data/NCaltech101", data_type="frame",
            frames_number=16, split_by="number")
```

These calls download or extract the source recordings and write the caches.
The dataset constructors and archive filenames follow the pinned
[CIFAR10-DVS](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/_modules/spikingjelly/datasets/cifar10_dvs.html)
and [N-Caltech101](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/_modules/spikingjelly/datasets/n_caltech101.html)
implementations. Expected layout:

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

Set `DATA_ROOT` to the absolute path of the parent directory, for example
`DATA_ROOT="$PWD/data"` when invoking a launcher from the repository root.

## DVS Gesture

DVS Gesture and N-MNIST are used for the Speck hardware experiments. The
ResNet event launcher uses CIFAR10-DVS and N-Caltech101.

Download the following files from the [IBM dataset page](https://ibm.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8)
into `data/DvsGesture/download/`:

- `DvsGesture.tar.gz`
- `gesture_mapping.csv`
- `LICENSE.txt`
- `README.txt`

Convert the recordings into event NPZ files using the pinned
[SpikingJelly DVS128Gesture loader](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/_modules/spikingjelly/datasets/dvs128_gesture.html):

```python
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

for train in (True, False):
    DVS128Gesture(root="data/DvsGesture", train=train, data_type="event")
```

The generated files contain `t`, `x`, `y` and `p` event arrays:

```text
data/DvsGesture/events_np/
  train/<class>/<sample>.npz
  test/<class>/<sample>.npz
```

The Speck experiment loader receives the `events_np/` directory through
`--data-root data/DvsGesture/events_np`. Use the supplied trial-based
train/test split.

## N-MNIST

The Speck experiments use the [Tonic N-MNIST loader](https://github.com/neuromorphs/tonic/blob/v1.6.0/tonic/datasets/nmnist.py).
Download and extract both splits with:

```python
from tonic.datasets import NMNIST

for train in (True, False):
    NMNIST(save_to="data/NMNIST", train=train)
```

Tonic adds an `NMNIST/` subdirectory beneath `save_to`:

```text
data/NMNIST/NMNIST/
  Train/<digit>/<sample>.bin
  Test/<digit>/<sample>.bin
```

The Speck loader receives `--data-root data/NMNIST`, the same value used for
`save_to`. It converts raw events to temporal frames when loading samples.

For a manual download, obtain `Train.zip` and `Test.zip` from the
[official dataset page](https://www.garrickorchard.com/datasets/n-mnist).
Save them as `train.zip` and `test.zip` in `data/NMNIST/NMNIST/` and extract
`Train/` and `Test/` alongside the archives.
