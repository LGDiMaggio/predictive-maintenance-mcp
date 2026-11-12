"""The plugin version moves whenever the plugin's content moves (audit A-P-4).

The marketplace installs the plugin from ``./plugin`` on ``main`` and pins it by
the version in ``plugin/.claude-plugin/plugin.json``; a content change under an
unchanged version does not reach users who already have that version.
``tests/fixtures/plugin_content_digest.json`` records, for each plugin version,
a digest of the plugin tree. The test fails when the tree no longer matches the
digest recorded for the current manifest version: bump the version in both
manifests (``tests/test_version_alignment.py`` keeps them equal) and append an
entry with the new digest, which the failure message prints.

Limit: an existing entry edited in place instead of a new entry appended is
caught only in review; history entries are append-only by convention.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / "plugin"
MANIFEST = PLUGIN_DIR / ".claude-plugin" / "plugin.json"
DIGESTS = Path(__file__).resolve().parent / "fixtures" / "plugin_content_digest.json"
_SKIPPED_PARTS = {"__pycache__"}
#: Operating-system and editor files that are never plugin content.
_SKIPPED_NAMES = {"desktop.ini", "Thumbs.db", ".DS_Store"}
_SKIPPED_SUFFIXES = (".pyc", ".swp", ".bak", "~")


def plugin_digest(plugin_dir: Path = PLUGIN_DIR) -> str:
    """SHA-256 over every file under *plugin_dir*: its relative path and its
    bytes with CRLF read as LF, so a Windows checkout gives the same digest.

    Files are hashed in the order of their relative POSIX paths compared as
    plain, case-sensitive strings. Sorting ``Path`` objects would not do:
    they compare case-insensitively on Windows and case-sensitively elsewhere.
    """
    digest = hashlib.sha256()
    files = sorted(
        (
            p
            for p in plugin_dir.rglob("*")
            if p.is_file()
            and not _SKIPPED_PARTS & set(p.parts)
            and p.name not in _SKIPPED_NAMES
            and not p.name.endswith(_SKIPPED_SUFFIXES)
        ),
        key=lambda p: p.relative_to(plugin_dir).as_posix(),
    )
    for path in files:
        digest.update(path.relative_to(plugin_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _history() -> list[dict[str, str]]:
    return json.loads(DIGESTS.read_text(encoding="utf-8"))["versions"]


def test_plugin_content_matches_the_digest_of_its_version():
    version = json.loads(MANIFEST.read_text(encoding="utf-8"))["version"]
    current = plugin_digest()
    recorded = {entry["version"]: entry["digest"] for entry in _history()}
    assert version in recorded, (
        f"plugin version {version} has no entry in {DIGESTS.name}; append "
        f'{{"version": "{version}", "digest": "{current}"}}'
    )
    assert recorded[version] == current, (
        f"the plugin content changed but its version is still {version}: bump the "
        "version in plugin/.claude-plugin/plugin.json and "
        ".claude-plugin/marketplace.json, then append "
        f'{{"version": "<new version>", "digest": "{current}"}} to {DIGESTS.name}'
    )


def test_digest_history_is_append_only_in_shape():
    history = _history()
    assert history, "no digest recorded"
    versions = [entry["version"] for entry in history]
    assert versions == sorted(versions, key=_version_key), versions
    assert len(set(versions)) == len(versions), versions
    digests = [entry["digest"] for entry in history]
    assert len(set(digests)) == len(digests), "two versions share one digest"
    current = json.loads(MANIFEST.read_text(encoding="utf-8"))["version"]
    assert versions[-1] == current, (
        f"the last recorded version {versions[-1]} is not the manifest version "
        f"{current}"
    )


class TestDigestControls:
    def _tree(self, tmp_path: Path) -> Path:
        root = tmp_path / "plugin"
        (root / "skills" / "a").mkdir(parents=True)
        (root / "skills" / "a" / "SKILL.md").write_bytes(b"line one\nline two\n")
        (root / "README.md").write_bytes(b"readme\n")
        return root

    def test_a_content_change_changes_the_digest(self, tmp_path):
        root = self._tree(tmp_path)
        before = plugin_digest(root)
        (root / "skills" / "a" / "SKILL.md").write_bytes(b"line one\nline 2\n")
        assert plugin_digest(root) != before

    def test_a_new_file_or_a_rename_changes_the_digest(self, tmp_path):
        root = self._tree(tmp_path)
        before = plugin_digest(root)
        (root / "skills" / "a" / "SKILL.md").rename(root / "skills" / "a" / "S.md")
        renamed = plugin_digest(root)
        assert renamed != before
        (root / "NEW.md").write_bytes(b"")
        assert plugin_digest(root) != renamed

    def test_files_are_ordered_case_sensitively_on_every_platform(self, tmp_path):
        """README.md sorts before agents/ as a plain string; a Windows Path sort
        puts it after, which made the recorded digest platform-dependent."""
        root = tmp_path / "plugin"
        (root / "agents").mkdir(parents=True)
        (root / "agents" / "x.md").write_bytes(b"agent\n")
        (root / "README.md").write_bytes(b"readme\n")
        expected = hashlib.sha256()
        for relative, data in (("README.md", b"readme\n"), ("agents/x.md", b"agent\n")):
            expected.update(relative.encode("utf-8") + b"\0" + data + b"\0")
        assert plugin_digest(root) == expected.hexdigest()

    def test_line_endings_do_not_change_the_digest(self, tmp_path):
        root = self._tree(tmp_path)
        before = plugin_digest(root)
        (root / "skills" / "a" / "SKILL.md").write_bytes(b"line one\r\nline two\r\n")
        assert plugin_digest(root) == before

    @pytest.mark.parametrize(
        "junk",
        [
            "__pycache__/x.cpython-311.pyc",
            "m.pyc",
            "skills/desktop.ini",
            ".DS_Store",
            "README.md~",
            "skills/a/SKILL.md.swp",
        ],
    )
    def test_bytecode_is_ignored(self, tmp_path, junk):
        root = self._tree(tmp_path)
        before = plugin_digest(root)
        target = root / junk
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x00")
        assert plugin_digest(root) == before
