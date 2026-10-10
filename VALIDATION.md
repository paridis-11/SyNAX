# Validation record

Checks performed on 2026-10-10:

- All copied source/configuration/license files match `SOURCE_MANIFEST.json`.
- All packaged Python files parse successfully.
- All launch scripts pass `bash -n`.
- No dataset, checkpoint, private-key, or environment credential files are included.
- Common GitHub token, AWS access-key, and private-key patterns were checked;
  no matches were found in the packaged Python code.
- All six principal entry points successfully show `--help` in the observed
  server environment with the documented protobuf setting.
- Static ResNet-18, ViT-Small, and ViT-Base build and produce finite `[1,10]`
  logits for a synthetic `[1,3,224,224]` tensor on CPU.
- Event Burst-ResNet18 builds and produces finite `[1,10]` and `[1,101]`
  logits for synthetic `[1,18,128,128]` inputs, using the actual server Burst source.
- The external Burst installation script successfully checks out its fixed
  upstream commit, applies the compatibility patch, and verifies three
  patched file hashes against the server snapshot.

This validates source integrity, importability and basic model construction.
It does not establish numerical reproduction of the full historical runs.
No full training, new multi-seed benchmark, or accuracy measurement was run
during packaging. The clean public-wheel environment recipe has not been
separately installed and tested from scratch.

Memory-code addition on 2026-10-10: the local Figure 2
`benchmark_gpu_memory.py` was added without source edits and passes syntax
checks. Only this standardized memory benchmark is included. No new GPU
memory benchmark has been run during packaging.
The standardized benchmark successfully shows `--help` using the observed
server environment, including its `.deps-synax-h100` Python dependency path
and the documented protobuf setting.

Directory rename checks on 2026-10-10: `static/` and `event/` retain all 62
source hashes. All 61 Python files and all shell launchers pass syntax checks.
Thirteen launcher dry runs checked 16 Python entry-point paths, covering all
static models, both event datasets and all four event methods, plus the GPU
memory launcher. Working directories, GPU masking, memory `PYTHONPATH` and
the S-TLLR dependency path were checked without starting training.

## Historical Run Context

The static experiments span code revisions; the original
`SNN_multiseed/results.tsv` has not been recovered. Evaluation uses the saved
best ANN checkpoint. Figure 2 event curves use seeds 42--44 for SyNAX and
SG-BPTT, and seeds 42--46 for S-TLLR.

The recorded server environment uses an NVIDIA PyTorch build. The README
installation recipe uses the corresponding public PyTorch release. Launchers
set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` for compatibility with the
older ONNX installation in the server image.

## Static Dataset Selection

The static launcher now accepts `DATASET=cifar10` or `DATASET=imagenet` and
passes the selection to both training and checkpoint evaluation. The evaluator
uses dataset-specific class counts and input sizes. ImageNet Dirichlet
partitions use ImageFolder targets; ImageNet native conversion is calibrated
on training data. Original server hashes for the two adapted source files are
retained in `SOURCE_MANIFEST.json`.

Syntax and source integrity checks passed. Sixteen launcher dry runs checked
22 entry-point paths, including both static datasets, checkpoint names and
invalid-dataset rejection. Six evaluator argument cases checked class counts
and input sizes for ResNet-18 and ViT-Small/Base. No full ImageNet training or
checkpoint accuracy run was performed for this launcher update.
