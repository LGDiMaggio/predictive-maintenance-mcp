"""The CPU image rule: no NVIDIA, CUDA or Triton component in the official image.

``python -m tools.audit artifact-licensing --image - --cpu-only`` adds these
rules to the image mode. The official image is built with
``--build-arg TORCH_VARIANT=cpu`` from ``docker/requirements-cpu.txt``; a build
that forgot the argument, or any later change that brings a GPU component back,
fails here. Each rule reports one finding, with the number of matching files
and the first of them.

The rules read only the member paths of the exported filesystem:

* ``gpu-package-folder``: a file under ``site-packages/nvidia/``,
  ``site-packages/triton/`` or ``site-packages/cuda/`` (also ``dist-packages``);
* ``gpu-distribution``: a dist-info folder of a distribution that
  :func:`is_gpu_distribution` names (``nvidia-*``, ``cuda-*``, ``triton``,
  ``triton-*``, ``pytorch-triton*``, ``*-cuda<N>*``, ``*-cu<N>``, ``*-gpu``);
* ``cuda-library``: a file named ``libcuda*``, ``libcudart*``, ``libcublas*``,
  ``libcudnn*``, ``libnccl*``, ``libnvrtc*``, ``libnvJitLink*``, ``libcufile*``,
  ``libnvshmem*`` or ``libcupti*``, anywhere;
* ``cuda-tool``: a file named ``ptxas*``, ``cuobjdump`` or ``nvdisasm``, anywhere.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable, Iterable

from tools.audit.results import Finding

__all__ = [
    "CPU_RULES",
    "RULE",
    "cpu_image_findings",
    "gpu_component_counts",
    "is_gpu_distribution",
    "normalize_name",
]

#: Rule id of every finding this module reports.
RULE = "image-gpu-component"

#: Distribution names (PEP 503 normalized) that may not enter the CPU image:
#: NVIDIA and CUDA packages, Triton under any name, and wheels whose name
#: marks a CUDA or GPU build (``cupy-cuda12x``, ``*-cu13``, ``onnxruntime-gpu``).
_GPU_NAME = re.compile(
    r"(?:nvidia|cuda)-.*|triton(?:-.*)?|pytorch-triton(?:-.*)?"
    r"|.*-cuda\d.*|.*-cu\d+|.*-gpu"
)


def normalize_name(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_gpu_distribution(name: str) -> bool:
    """True for a distribution the CPU image may not hold (see ``_GPU_NAME``)."""
    return _GPU_NAME.fullmatch(normalize_name(name)) is not None


_SITE = r"(?:^|/)(?:site|dist)-packages/"
_PACKAGE_FOLDER = re.compile(_SITE + r"(?:nvidia|triton|cuda)/", re.IGNORECASE)
_DIST_INFO = re.compile(_SITE + r"([^/]+?)-[^/-]+\.dist-info/", re.IGNORECASE)
_LIBRARY = re.compile(
    r"lib(?:cuda|cudart|cublas|cudnn|nccl|nvrtc|nvjitlink|cufile|nvshmem|cupti)",
    re.IGNORECASE,
)
_TOOL = re.compile(r"ptxas.*|cuobjdump(?:\.exe)?|nvdisasm(?:\.exe)?", re.IGNORECASE)


def _package_folder(path: str) -> bool:
    return _PACKAGE_FOLDER.search(path) is not None


def _distribution(path: str) -> bool:
    match = _DIST_INFO.search(path)
    return match is not None and is_gpu_distribution(match.group(1))


def _library(path: str) -> bool:
    return _LIBRARY.match(posixpath.basename(path)) is not None


def _tool(path: str) -> bool:
    return _TOOL.fullmatch(posixpath.basename(path)) is not None


#: Each rule: (name, predicate on a member path, what a match means).
CPU_RULES: tuple[tuple[str, Callable[[str], bool], str], ...] = (
    ("gpu-package-folder", _package_folder, "an NVIDIA, CUDA or Triton package folder"),
    ("gpu-distribution", _distribution, "an NVIDIA, CUDA or Triton distribution"),
    ("cuda-library", _library, "a CUDA library"),
    ("cuda-tool", _tool, "a CUDA compiler or disassembler"),
)

#: How many matching paths a finding lists.
EXAMPLES = 20


def gpu_component_counts(paths: Iterable[str]) -> dict[str, list[str]]:
    """Map each CPU rule to the sorted member paths it matches."""
    members = sorted(paths)
    return {
        name: [path for path in members if matches(path)]
        for name, matches, _ in CPU_RULES
    }


def cpu_image_findings(paths: Iterable[str], image_name: str) -> list[Finding]:
    """One finding per CPU rule that matches any member path of the image."""
    from tools.audit.artifact_licensing import _finding

    findings: list[Finding] = []
    counts = gpu_component_counts(paths)
    for name, _, meaning in CPU_RULES:
        matched = counts[name]
        if not matched:
            continue
        findings.append(
            _finding(
                RULE,
                f"image {image_name}",
                f"The CPU image holds {len(matched)} file(s) of {meaning} ({name}), "
                f"for example {matched[0]}",
                image=image_name,
                cpu_rule=name,
                count=len(matched),
                examples=matched[:EXAMPLES],
            )
        )
    return findings
