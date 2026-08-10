"""CWRU diagnostic-accuracy benchmark package (maintainer tooling).

This package is not server surface: it exercises the deterministic
diagnostic pipeline as a library against the CWRU Bearing Data Center
dataset. The vendored record tables and their blind/label access views
live in :mod:`benchmarks.cwru.records`; later units add the downloader,
importer, runner, and scorer, dispatched via ``python -m benchmarks.cwru``.
"""
