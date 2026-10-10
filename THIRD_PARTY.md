# Third-party components

- Burst ANN2SNN: https://github.com/bic-L/burst-ann2snn . Installed separately
  by `scripts/fetch_burst_dependency.py`. Three server compatibility changes
  are supplied as a patch, with output hashes for verification. Preserve
  upstream notices and check upstream distribution terms.
- S-TLLR: https://github.com/mapolinario94/S-TLLR . The relevant source is
  included under `event/third_party/S-TLLR/` with its original MIT license.
- SEW-ResNet: provided by the pinned SpikingJelly dependency. Reference:
  Fang et al., *Deep Residual Learning in Spiking Neural Networks*, NeurIPS 2021.
- Static ViT modules retain their original attribution to timm and EVA in
  source comments. The package keeps these headers.

