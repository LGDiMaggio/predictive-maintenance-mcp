"""Sphinx configuration for predictive-maintenance-mcp."""

import os
import sys

# Add src to path so autodoc can find modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

project = "predictive-maintenance-mcp"
copyright = "2026, Luigi Di Maggio"
author = "Luigi Di Maggio"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = []

# Suppress warnings for missing imports (scipy, sklearn, etc.)
autodoc_mock_imports = [
    "mcp",
    "plotly",
    "sklearn",
    "faiss",
    "sentence_transformers",
]
