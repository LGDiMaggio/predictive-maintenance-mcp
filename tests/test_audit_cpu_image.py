"""The CPU image rule of the image mode (``artifact-licensing --image - --cpu-only``).

Each rule has its own fixture path, and dropping that rule from
:data:`tools.audit.cpu_image.CPU_RULES` lets the fixture through (removal seam),
so no rule is redundant. Paths a CPU torch install legitimately holds (headers
and modules with ``cuda`` or ``triton`` in a deeper path) must not match.
"""

from __future__ import annotations

import io
import tarfile
import tomllib
from pathlib import Path

import pytest

from tools.audit import __main__ as audit_main
from tools.audit import artifact_licensing, cpu_image

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = artifact_licensing.IMAGE_SITE_PACKAGES

#: One fixture path per CPU rule.
RULE_FIXTURES = {
    "gpu-package-folder": f"{SITE}/nvidia/cublas/__init__.py",
    "gpu-distribution": f"{SITE}/triton-3.7.1.dist-info/METADATA",
    "cuda-library": "usr/local/lib/libcudart.so.13",
    "cuda-tool": f"{SITE}/somewhere/bin/ptxas-blackwell",
}

#: More paths each rule must catch.
CAUGHT = [
    (f"{SITE}/triton/backends/nvidia/bin/ptxas", {"gpu-package-folder", "cuda-tool"}),
    (f"{SITE}/cuda/bindings/driver.py", {"gpu-package-folder"}),
    ("usr/lib/python3/dist-packages/nvidia/x.py", {"gpu-package-folder"}),
    (f"{SITE}/nvidia_cudnn_cu13-9.20.0.48.dist-info/RECORD", {"gpu-distribution"}),
    (f"{SITE}/cuda_toolkit-13.0.3.0.dist-info/METADATA", {"gpu-distribution"}),
    (f"{SITE}/cuda_bindings-13.3.1.dist-info/METADATA", {"gpu-distribution"}),
    (f"{SITE}/cuda_pathfinder-1.6.0.dist-info/METADATA", {"gpu-distribution"}),
    (f"{SITE}/cuda_python-13.0.0.dist-info/METADATA", {"gpu-distribution"}),
    (f"{SITE}/cupy_cuda12x-13.6.0.dist-info/METADATA", {"gpu-distribution"}),
    (f"{SITE}/pytorch_triton-3.4.0.dist-info/RECORD", {"gpu-distribution"}),
    (f"{SITE}/triton_windows-3.2.0.dist-info/METADATA", {"gpu-distribution"}),
    (f"{SITE}/onnxruntime_gpu-1.20.0.dist-info/METADATA", {"gpu-distribution"}),
    ("opt/x/libcublasLt.so.13", {"cuda-library"}),
    ("opt/x/libcudnn_ops.so.9", {"cuda-library"}),
    ("opt/x/libnccl.so.2", {"cuda-library"}),
    ("opt/x/libnvrtc.so.13", {"cuda-library"}),
    ("opt/x/libnvJitLink.so.13", {"cuda-library"}),
    ("opt/x/libcufile.so.0", {"cuda-library"}),
    ("opt/x/libnvshmem_host.so.3", {"cuda-library"}),
    ("opt/x/libcupti.so.13", {"cuda-library"}),
    ("opt/x/libcuda.so.1", {"cuda-library"}),
    ("opt/x/cuobjdump", {"cuda-tool"}),
    ("opt/x/nvdisasm", {"cuda-tool"}),
]

#: Paths of a CPU-only install that mention CUDA or Triton deeper down.
NOT_CAUGHT = [
    f"{SITE}/torch/_inductor/codegen/triton.py",
    f"{SITE}/torch/utils/_triton.py",
    f"{SITE}/torch/include/ATen/cuda/CUDAContext.h",
    f"{SITE}/torch/cuda/__init__.py",
    f"{SITE}/numba/cuda/__init__.py",
    f"{SITE}/torch/lib/libtorch_cpu.so",
    f"{SITE}/torch-2.13.0+cpu.dist-info/METADATA",
    f"{SITE}/faiss_cpu-1.15.0.dist-info/METADATA",
    f"{SITE}/sentence_transformers-5.7.0.dist-info/RECORD",
    f"{SITE}/transformers/kernels/cuda_kernel.cu",
    "usr/share/doc/libcurl4/copyright",
]


def _matches(path: str) -> set:
    return {
        rule for rule, paths in cpu_image.gpu_component_counts([path]).items() if paths
    }


class TestRules:
    @pytest.mark.parametrize("path,expected", CAUGHT)
    def test_a_gpu_component_is_caught(self, path, expected):
        assert _matches(path) == expected

    @pytest.mark.parametrize("path", NOT_CAUGHT)
    def test_a_cpu_install_path_is_not_caught(self, path):
        assert _matches(path) == set()

    @pytest.mark.parametrize("rule", sorted(RULE_FIXTURES))
    def test_each_rule_catches_its_fixture_and_is_needed(self, rule, monkeypatch):
        fixture = RULE_FIXTURES[rule]
        assert rule in _matches(fixture)
        monkeypatch.setattr(
            cpu_image,
            "CPU_RULES",
            tuple(entry for entry in cpu_image.CPU_RULES if entry[0] != rule),
        )
        assert _matches(fixture) == set()

    def test_one_finding_per_rule_with_count_and_examples(self):
        paths = [f"{SITE}/nvidia/lib/{i}.so" for i in range(30)]
        findings = cpu_image.cpu_image_findings(paths, "img")
        assert [f.rule for f in findings] == [cpu_image.RULE]
        evidence = findings[0].evidence
        assert evidence["cpu_rule"] == "gpu-package-folder"
        assert evidence["count"] == 30
        assert len(evidence["examples"]) == cpu_image.EXAMPLES
        assert evidence["examples"] == sorted(paths)[: cpu_image.EXAMPLES]


def _project_version() -> str:
    with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _clean_image() -> dict:
    """A minimal filesystem the image mode accepts against the real repository."""
    license_bytes = (REPO_ROOT / artifact_licensing.LICENSE_FILE).read_bytes()
    dist = f"{SITE}/predictive_maintenance_mcp-{_project_version()}.dist-info"
    return {
        "app/LICENSE": license_bytes,
        f"{dist}/licenses/LICENSE": license_bytes,
        f"{dist}/METADATA": b"Metadata-Version: 2.4\n",
        "app/resources/bearing_catalogs/README.md": b"# Bearing catalogs\n",
        "app/resources/bearing_catalogs/common_bearings_catalog.json": b"{}\n",
        "app/src/server.py": b"print('server')\n",
    }


def _tar(path: Path, files: dict) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return path


def _run(tmp_path: Path, extra: dict, cpu_only: bool):
    image = _tar(tmp_path / "image.tar", {**_clean_image(), **extra})
    return artifact_licensing.run_image_licensing(
        artifact_licensing.ImageLicensingConfig(
            image=str(image), repo_root=REPO_ROOT, cpu_only=cpu_only
        )
    )


class TestImageMode:
    def test_a_clean_image_passes_with_zero_counts(self, tmp_path):
        result = _run(tmp_path, {}, cpu_only=True)
        assert result.status == "pass", result.findings
        assert result.facts["cpu_only"] == {name: 0 for name in RULE_FIXTURES}

    @pytest.mark.parametrize("rule", sorted(RULE_FIXTURES))
    def test_a_gpu_file_fails_only_with_cpu_only(self, tmp_path, rule):
        extra = {RULE_FIXTURES[rule]: b"x"}
        assert _run(tmp_path, extra, cpu_only=False).status == "pass"
        result = _run(tmp_path, extra, cpu_only=True)
        assert result.status == "fail"
        assert {(f.rule, f.evidence["cpu_rule"]) for f in result.findings} >= {
            (cpu_image.RULE, rule)
        }
        assert result.facts["cpu_only"][rule] == 1


class TestCli:
    def test_cpu_only_fails_the_cli_on_a_gpu_file(self, tmp_path, capsys):
        image = _tar(
            tmp_path / "image.tar",
            {**_clean_image(), RULE_FIXTURES["gpu-distribution"]: b"x"},
        )
        code = audit_main.main(
            [
                "artifact-licensing",
                "--image",
                str(image),
                "--cpu-only",
                "--repo-root",
                str(REPO_ROOT),
            ]
        )
        assert code == 1
        assert f"[high] {cpu_image.RULE}:" in capsys.readouterr().out

    def test_cpu_only_without_image_is_a_usage_error(self, tmp_path, capsys):
        code = audit_main.main(
            ["artifact-licensing", "--dist", str(tmp_path), "--cpu-only"]
        )
        assert code == 2
        assert "--cpu-only applies to --image only" in capsys.readouterr().err
