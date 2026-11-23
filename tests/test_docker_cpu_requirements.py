"""The official CPU image's dependency file and how the image uses it.

* T4 guard: ``docker/requirements-cpu.txt`` follows ``uv.lock`` (every pin a
  locked version with locked hashes), holds no NVIDIA, CUDA or Triton
  distribution, and takes torch as the SHA-256-pinned ``+cpu`` wheel of the
  lock's torch version. Each rule has a removal proof: a temporary copy with
  one violation injected must fail with that rule's message.
* The generator, run on fake export, lock and index inputs (no network).
* ``check-freeze``: what a built image installs against the file.
* Static guards on the Dockerfile, ``.dockerignore`` and the image workflow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from _workflows import load_workflows
from tools import cpu_requirements as cr

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / cr.REQUIREMENTS_RELPATH
TORCH_LINE = re.compile(
    r"^torch @ \S+ \\\n    --hash=sha256:[0-9a-f]{64}$", re.MULTILINE
)


@pytest.fixture(scope="module")
def lock() -> dict:
    return cr.load_lock(REPO_ROOT / cr.LOCK_RELPATH)


@pytest.fixture(scope="module")
def text() -> str:
    return REQUIREMENTS.read_text(encoding="utf-8")


def _torch_version(lock: dict) -> str:
    (version,) = cr.lock_index(lock)["torch"]
    return version


class TestCommittedFile:
    def test_it_follows_the_lock(self, text, lock):
        assert cr.verify_requirements(text, lock) == []

    def test_it_holds_no_gpu_distribution(self, text):
        names = [entry.name for entry in cr.parse_requirements(text)]
        assert names and not [name for name in names if cr.is_gpu_distribution(name)]

    def test_torch_is_the_cpu_wheel_of_the_lock_version(self, text, lock):
        (torch,) = [e for e in cr.parse_requirements(text) if e.key == "torch"]
        assert torch.url is not None and len(torch.hashes) == 1
        filename = torch.url.rsplit("/", 1)[-1].replace("%2B", "+")
        assert filename == (
            f"torch-{_torch_version(lock)}+cpu-cp311-cp311-manylinux_2_28_x86_64.whl"
        )

    def test_the_header_names_the_lock_blob(self, text):
        assert re.search(r"\(git blob [0-9a-f]{40}\)", text)


def _first_pin(text: str) -> cr.Entry:
    return next(e for e in cr.parse_requirements(text) if e.key != "torch")


def _inject_gpu(text: str, lock: dict) -> str:
    hashes = sorted(cr.lock_index(lock)["nvidia-cublas"].popitem()[1])
    return text + f"nvidia-cublas==13.1.1.3 \\\n    --hash=sha256:{hashes[0]}\n"


def _drop_entry(text: str, name: str) -> str:
    """Remove one pin (its line and its continuation lines) from the file text."""
    lines, out, skipping = text.splitlines(keepends=True), [], False
    for line in lines:
        if line.startswith(f"{name}=="):
            skipping = True
            continue
        if skipping and line.startswith("    "):
            continue
        skipping = False
        out.append(line)
    return "".join(out)


def _outside_full(lock: dict) -> str:
    version, hashes = sorted(cr.lock_index(lock)["black"].items())[0]
    return f"black=={version} \\\n    --hash=sha256:{sorted(hashes)[0]}\n"


def _swap_torch(text: str, replacement: str) -> str:
    return TORCH_LINE.sub(lambda _: replacement, text, count=1)


def _cpu_url(version: str, host: str = "download-r2.pytorch.org") -> str:
    return (
        f"https://{host}/whl/cpu/torch-{version}%2Bcpu-cp311-cp311-"
        "manylinux_2_28_x86_64.whl"
    )


HASH = "--hash=sha256:" + "a" * 64


def _mutations():
    yield "gpu-line", _inject_gpu, "is an NVIDIA, CUDA or Triton distribution"
    yield (
        "torch-not-cpu",
        lambda t, lk: _swap_torch(
            t,
            "torch @ https://download.pytorch.org/whl/torch-"
            f"{_torch_version(lk)}-cp311-cp311-manylinux_2_28_x86_64.whl \\\n    {HASH}",
        ),
        "is not a +cpu manylinux_2_28_x86_64 wheel",
    )
    yield (
        "torch-other-version",
        lambda t, lk: _swap_torch(t, f"torch @ {_cpu_url('2.0.0')} \\\n    {HASH}"),
        "is not the uv.lock torch version",
    )
    yield (
        "torch-as-pin",
        lambda t, lk: _swap_torch(t, f"torch=={_torch_version(lk)} \\\n    {HASH}"),
        "torch is not a direct URL to a CPU wheel",
    )
    yield (
        "torch-foreign-host",
        lambda t, lk: _swap_torch(
            t, f"torch @ {_cpu_url(_torch_version(lk), 'example.org')} \\\n    {HASH}"
        ),
        "does not come from an https pytorch.org host",
    )
    yield (
        "torch-lookalike-host",
        lambda t, lk: _swap_torch(
            t,
            f"torch @ {_cpu_url(_torch_version(lk), 'evilpytorch.org')} \\\n    {HASH}",
        ),
        "does not come from an https pytorch.org host",
    )
    yield (
        "pin-dropped",
        lambda t, lk: _drop_entry(t, _first_pin(t).name),
        "is in uv.lock's full extra but not in the file",
    )
    yield (
        "pin-outside-full",
        lambda t, lk: t + _outside_full(lk),
        "black is in the file but not in uv.lock's full extra",
    )
    yield (
        "torch-twice",
        lambda t, lk: t + f"torch @ {_cpu_url(_torch_version(lk))} \\\n    {HASH}\n",
        "expected exactly one torch requirement, found 2",
    )
    yield (
        "version-not-locked",
        lambda t, lk: t + f"{_first_pin(t).name}==0.0.0.1 \\\n    {HASH}\n",
        "==0.0.0.1 is not a uv.lock version",
    )
    yield (
        "hash-not-locked",
        lambda t, lk: t.replace(_first_pin(t).hashes[0], "0" * 64, 1),
        "has a hash uv.lock does not list",
    )
    yield (
        "pin-without-hash",
        lambda t, lk: t + f"{_first_pin(t).name}=={_first_pin(t).version}\n",
        "has no --hash",
    )


MUTATIONS = list(_mutations())


class TestRemovalProofs:
    @pytest.mark.parametrize(
        "mutate,message", [m[1:] for m in MUTATIONS], ids=[m[0] for m in MUTATIONS]
    )
    def test_each_rule_fails_on_an_injected_violation(
        self, text, lock, mutate, message
    ):
        mutated = mutate(text, lock)
        assert mutated != text
        problems = cr.verify_requirements(mutated, lock)
        assert any(message in problem for problem in problems), problems


class TestGpuNames:
    @pytest.mark.parametrize(
        "name",
        [
            "nvidia-cublas",
            "nvidia_cudnn_cu13",
            "cuda-bindings",
            "cuda-python",
            "triton",
            "triton-windows",
            "pytorch-triton",
            "pytorch-triton-rocm",
            "cupy-cuda12x",
            "jax-cuda12-plugin",
            "flash-attn-cu12",
            "onnxruntime-gpu",
            "faiss-gpu",
        ],
    )
    def test_a_gpu_distribution_is_named(self, name):
        assert cr.is_gpu_distribution(name)

    @pytest.mark.parametrize(
        "name",
        ["torch", "faiss-cpu", "sentence-transformers", "tritonclient-x", "cudaq"],
    )
    def test_a_cpu_distribution_is_not(self, name):
        assert not cr.is_gpu_distribution(name)

    def test_the_lock_closure_of_full_is_the_file_plus_the_left_out_gpu_set(
        self, text, lock
    ):
        closure = cr.lock_closure(lock)
        names = {entry.key for entry in cr.parse_requirements(text)}
        assert names == {name for name in closure if not cr.is_gpu_distribution(name)}
        assert {"torch", "triton"} <= closure


class TestParse:
    def test_continuation_lines_markers_and_urls(self):
        entries = cr.parse_requirements(
            "# comment\n"
            "a-b==1.0 ; sys_platform == 'linux' \\\n"
            "    --hash=sha256:" + "1" * 64 + " \\\n"
            "    --hash=sha256:" + "2" * 64 + "\n"
            "torch @ https://x/whl/cpu/torch-1%2Bcpu.whl \\\n"
            "    --hash=sha256:" + "3" * 64 + "\n"
        )
        assert [(e.name, e.version, e.url, e.marker) for e in entries] == [
            ("a-b", "1.0", None, "sys_platform == 'linux'"),
            ("torch", None, "https://x/whl/cpu/torch-1%2Bcpu.whl", None),
        ]
        assert entries[0].hashes == ["1" * 64, "2" * 64]

    def test_an_unparseable_line_is_an_error(self):
        with pytest.raises(ValueError, match="unparseable requirement line"):
            cr.parse_requirements("-r other.txt\n")


INDEX = (
    "<html><body>"
    '<a href="https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp311-'
    "cp311-manylinux_2_28_x86_64.whl#sha256=" + "b" * 64 + '">x</a>'
    '<a href="/whl/cpu/torch-2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl'
    "#sha256=" + "c" * 64 + '">y</a>'
    "</body></html>"
)


class TestSelectTorchWheel:
    def test_it_picks_the_target_wheel_and_its_hash(self):
        url, digest = cr.select_torch_wheel(INDEX, "2.13.0")
        assert url.endswith("torch-2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl")
        assert digest == "b" * 64

    def test_a_relative_link_is_resolved_against_the_index(self):
        page = INDEX.replace("https://download-r2.pytorch.org/whl/cpu/", "")
        url, _ = cr.select_torch_wheel(page, "2.13.0")
        assert url.startswith(cr.PYTORCH_CPU_INDEX)

    @pytest.mark.parametrize(
        "page,message",
        [
            (
                INDEX.replace("2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64", "z"),
                "found 0",
            ),
            (INDEX + INDEX, "found 2"),
            (INDEX.replace("#sha256=" + "b" * 64, ""), "without a sha256 fragment"),
        ],
    )
    def test_anything_but_one_hashed_link_is_an_error(self, page, message):
        with pytest.raises(ValueError, match=message):
            cr.select_torch_wheel(page, "2.13.0")


def _fake_repo(tmp_path: Path, torch_version: str = "2.13.0") -> Path:
    lock = f"""\
version = 1
[[package]]
name = "predictive-maintenance-mcp"
version = "0.13.0"
source = {{ editable = "." }}
dependencies = [{{ name = "alpha" }}]

[package.optional-dependencies]
full = [
    {{ name = "torch" }},
    {{ name = "nvidia-cublas" }},
    {{ name = "triton" }},
    {{ name = "cuda-toolkit" }},
]

[[package]]
name = "alpha"
version = "1.0"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [{{ url = "https://x/alpha.whl", hash = "sha256:{'1' * 64}" }}]

[[package]]
name = "torch"
version = "{torch_version}"
source = {{ registry = "https://pypi.org/simple" }}
wheels = [{{ url = "https://x/torch.whl", hash = "sha256:{'2' * 64}" }}]
"""
    (tmp_path / "uv.lock").write_text(lock, encoding="utf-8")
    return tmp_path


EXPORT = (
    "alpha==1.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    "nvidia-cublas==13.1.1.3 ; sys_platform == 'linux' \\\n    --hash=sha256:"
    + "4" * 64
    + "\n"
    "torch==2.13.0 \\\n    --hash=sha256:" + "2" * 64 + "\n"
    "triton==3.7.1 ; sys_platform == 'linux' \\\n    --hash=sha256:" + "5" * 64 + "\n"
    "cuda-toolkit==13.0.3.0 \\\n    --hash=sha256:" + "6" * 64 + "\n"
)


def _generate(root: Path, export: str = EXPORT) -> str:
    return cr.generate(
        root,
        export=lambda _: export,
        fetch=lambda _: INDEX,
        blob_id=lambda *_: "f" * 40,
    )


class TestGenerate:
    def test_it_drops_gpu_pins_and_swaps_torch(self, tmp_path):
        root = _fake_repo(tmp_path)
        text = _generate(root)
        entries = cr.parse_requirements(text)
        assert [e.name for e in entries] == ["alpha", "torch"]
        assert entries[1].hashes == ["b" * 64]
        assert "(git blob " + "f" * 40 + ")" in text
        assert (
            "Left out (3): cuda-toolkit==13.0.3.0, nvidia-cublas==13.1.1.3, triton==3.7.1."
            in text
        )
        assert cr.verify_requirements(text, cr.load_lock(root / "uv.lock")) == []

    def test_an_export_that_disagrees_with_the_lock_is_an_error(self, tmp_path):
        root = _fake_repo(tmp_path, torch_version="2.12.0")
        with pytest.raises(
            ValueError, match="uv export pins torch 2.13.0, uv.lock has 2.12.0"
        ):
            _generate(root)

    def test_an_export_without_torch_is_an_error(self, tmp_path):
        root = _fake_repo(tmp_path)
        with pytest.raises(ValueError, match="exactly one torch pin"):
            _generate(root, EXPORT.replace("torch==", "notorch=="))


TORCH_URL = _cpu_url("2.13.0")
REQS = (
    "alpha==1.0 \\\n    --hash=sha256:" + "1" * 64 + "\n"
    "colorama==0.4.6 ; sys_platform == 'win32' \\\n    --hash=sha256:" + "8" * 64 + "\n"
    "setuptools==84.1.0 \\\n    --hash=sha256:" + "7" * 64 + "\n"
    "sphinx==9.0.4 ; python_full_version < '3.12' \\\n    --hash=sha256:"
    + "9" * 64
    + "\n"
    "sphinx==9.1.0 ; python_full_version >= '3.12' \\\n    --hash=sha256:"
    + "d" * 64
    + "\n"
    f"torch @ {TORCH_URL} \\\n    --hash=sha256:" + "b" * 64 + "\n"
)
FREEZE = (
    "alpha==1.0\npip==25.3\npredictive-maintenance-mcp @ file:///app\n"
    "setuptools==84.1.0\nsphinx==9.0.4\n"
    f"torch @ {TORCH_URL}#sha256=" + "b" * 64 + "\n"
)
BASE = "pip==25.3\nsetuptools==65.5.1\n"


class TestFreeze:
    def test_the_file_the_project_and_base_pip_pass(self):
        assert cr.verify_freeze(FREEZE, REQS, BASE) == []

    def test_a_marker_entry_may_be_absent_or_any_listed_fork(self):
        assert cr.verify_freeze(FREEZE.replace("sphinx==9.0.4\n", ""), REQS, BASE) == []
        assert cr.verify_freeze(FREEZE.replace("9.0.4", "9.1.0"), REQS, BASE) == []

    def test_a_marker_only_pin_may_stay_at_the_base_image_version(self):
        reqs = REQS + "wheel==0.48.0 ; python_full_version >= '3.12' \\\n"
        reqs += "    --hash=sha256:" + "f" * 64 + "\n"
        freeze = FREEZE + "wheel==0.46.3\n"
        assert cr.verify_freeze(freeze, reqs, BASE + "wheel==0.46.3\n") == []
        problems = cr.verify_freeze(freeze, reqs, BASE + "wheel==0.45.0\n")
        assert any("wheel is installed as 0.46.3" in p for p in problems), problems

    @pytest.mark.parametrize(
        "freeze,base,message",
        [
            (
                FREEZE + "extra==1\n",
                BASE,
                "extra is installed but is neither in the file",
            ),
            (FREEZE, "pip==24.0\n", "pip is installed but is neither in the file"),
            (FREEZE, None, "pip is installed but is neither in the file"),
            (
                FREEZE.replace(
                    f"torch @ {TORCH_URL}#sha256=" + "b" * 64, "torch==2.13.0"
                ),
                BASE,
                f"torch is installed as 2.13.0, the file pins {TORCH_URL}",
            ),
            (
                FREEZE.replace("=" + "b" * 64, "=" + "e" * 64),
                BASE,
                "torch is installed as",
            ),
            (
                FREEZE.replace("alpha==1.0\n", ""),
                BASE,
                "alpha is in the file but is not installed",
            ),
            (
                FREEZE.replace("sphinx==9.0.4", "sphinx==8.0.0"),
                BASE,
                "sphinx is installed as 8.0.0",
            ),
            (
                FREEZE + "nvidia-nccl-cu13==2.29.7\n",
                BASE,
                "nvidia-nccl-cu13 is an NVIDIA, CUDA or Triton",
            ),
        ],
    )
    def test_anything_else_fails(self, freeze, base, message):
        problems = cr.verify_freeze(freeze, REQS, base)
        assert any(message in problem for problem in problems), problems

    def test_the_cli_reports_pass_and_fail(self, tmp_path, capsys):
        (tmp_path / "docker").mkdir()
        (tmp_path / cr.REQUIREMENTS_RELPATH).write_text(REQS, encoding="utf-8")
        freeze, base = tmp_path / "freeze.txt", tmp_path / "base.txt"
        freeze.write_text(FREEZE, encoding="utf-8")
        base.write_text(BASE, encoding="utf-8")
        args = ["check-freeze", "--repo-root", str(tmp_path), "--freeze", str(freeze)]
        assert cr.main(args + ["--base", str(base)]) == 0
        assert cr.main(args) == 1
        assert "pip is installed but is neither" in capsys.readouterr().out


DIGEST = re.compile(r"^FROM python:3\.11-slim@sha256:([0-9a-f]{64})\b", re.MULTILINE)


class TestDockerfile:
    @pytest.fixture(scope="class")
    def dockerfile(self) -> str:
        return (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    def test_both_stages_use_the_same_digest_pinned_base(self, dockerfile):
        froms = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
        assert len(froms) == 2
        assert len(set(DIGEST.findall(dockerfile))) == 1
        assert len(DIGEST.findall(dockerfile)) == 2

    def test_the_variant_argument_defaults_to_todays_install(self, dockerfile):
        head, _, builder = dockerfile.partition("FROM ")
        assert "ARG TORCH_VARIANT=default" in head
        assert "\nARG TORCH_VARIANT\n" in builder

    def test_the_cpu_variant_installs_only_the_hashed_file(self, dockerfile):
        for command in (
            "pip install --no-cache-dir --require-hashes --no-deps -r docker/requirements-cpu.txt",
            "pip install --no-cache-dir --no-deps --no-build-isolation .",
            "pip check",
            "COPY docker/requirements-cpu.txt ./docker/requirements-cpu.txt",
        ):
            assert command in dockerfile

    def test_the_runtime_stage_replaces_the_base_site_packages(self, dockerfile):
        """Merging the builder's site-packages into the base image's left a
        second pip, setuptools, wheel or packaging dist-info of another version
        (seen in both the CPU and the default image before this line)."""
        site = "/usr/local/lib/python3.11/site-packages"
        assert f"RUN rm -rf {site}\nCOPY --from=builder {site} {site}" in dockerfile

    def test_the_default_variant_keeps_todays_install(self, dockerfile):
        assert 'pip install --no-cache-dir ".[full]" uvicorn' in dockerfile

    def test_an_unknown_variant_stops_the_build(self, dockerfile):
        assert "TORCH_VARIANT must be cpu or default" in dockerfile
        assert "exit 1" in dockerfile

    def test_the_build_context_lets_the_file_in(self):
        lines = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        assert "!docker/requirements-cpu.txt" in [line.strip() for line in lines]


class TestImageWorkflow:
    @pytest.fixture(scope="class")
    def runs(self) -> list:
        steps = load_workflows()["docker-image.yml"]["jobs"]["image"]["steps"]
        return [str(step.get("run", "")) for step in steps]

    def test_every_image_build_is_the_cpu_variant(self, runs):
        builds = [
            line
            for run in runs
            for line in run.splitlines()
            if re.search(r"docker build .*-t pmm-image-check(-2)? ", line)
        ]
        assert len(builds) == 2
        assert all("--build-arg TORCH_VARIANT=cpu" in line for line in builds)

    def test_the_image_check_applies_the_cpu_rule(self, runs):
        joined = "\n".join(runs)
        assert (
            "artifact-licensing --image - --cpu-only --variant docker-image " in joined
        )

    def test_the_dependency_file_and_the_installed_packages_are_checked(self, runs):
        joined = "\n".join(runs)
        assert "python -m tools.cpu_requirements check\n" in joined + "\n"
        assert (
            "python -m tools.cpu_requirements check-freeze --freeze "
            "results/pip-freeze.txt --base results/base-freeze.txt"
        ) in joined
        assert "freeze --all" in joined

    def test_a_planted_cuda_library_must_fail_the_cpu_rule(self, runs):
        (step,) = [run for run in runs if "gpu-bypass" in run]
        assert "--cpu-only" in step
        assert 'test "$code" -eq 1' in step
        assert 'grep -q "image-gpu-component" gpu-bypass.txt' in step

    def test_the_container_must_report_healthy(self, runs):
        (step,) = [run for run in runs if "pmm-f1" in run]
        assert 'test "$status" = healthy' in step
