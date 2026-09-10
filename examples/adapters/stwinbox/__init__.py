"""Reference adapter for STWIN.box acquisitions (example, not server surface).

The package exists so the test suite can import the adapter in-process
(``from examples.adapters.stwinbox import stwinbox_to_measurement``) through
the repository root on ``sys.path``, the same way ``benchmarks.cwru`` is
imported. The adapter itself imports nothing from the server.
"""
