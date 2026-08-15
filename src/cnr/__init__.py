# Copyright 2026 Coordinates Need a Ruler contributors.
#
# Licensed under the MIT License; see LICENSE at the repository root.
"""Coordinates Need a Ruler -- atomic coordinate tokens and a number token loss.

Reference implementation for VizWiz answer grounding with a 2B vision--language
model that writes each coordinate as a single token from a contiguous ordered
block, and is trained with a Wasserstein-1 term over that block alongside
cross-entropy.

Module map, in the order the pipeline uses them:

===========================  =====================================================
:mod:`cnr.coord_tokens`      the ``<coord_k>`` vocabulary block (Eq. 1)
:mod:`cnr.desc_init`         numeric-description initialisation (Eq. 2)
:mod:`cnr.grad_mask`         gradient-masked embedding training (Eq. 3)
:mod:`cnr.ntl`               the number token loss (Eq. 4-6)
:mod:`cnr.prompts`           box and click targets from a ground-truth mask
:mod:`cnr.dataset`           the supervised record; the one canonical prompt
:mod:`cnr.crops`             multi-scale crop augmentation
:mod:`cnr.infer`             test-set generation
:mod:`cnr.sam2_decode`       geometry -> binary mask
:mod:`cnr.geometry`          boxes, polygons, the ``[0, 1000]`` <-> pixel maps
:mod:`cnr.metrics`           bounding-box IoU and the benchmark's mask IoU
===========================  =====================================================

Only :mod:`cnr.ntl`, :mod:`cnr.desc_init` and :mod:`cnr.grad_mask` need
``torch``; only :mod:`cnr.sam2_decode` needs SAM 2; only :mod:`cnr.infer` needs
LlamaFactory. Everything else is importable with numpy and Pillow alone, which
is what keeps the test suite runnable on a laptop.
"""

__version__ = "1.0.0"

__all__ = [
    "coord_tokens",
    "crops",
    "dataset",
    "desc_init",
    "geometry",
    "grad_mask",
    "infer",
    "metrics",
    "ntl",
    "prompts",
    "sam2_decode",
]
