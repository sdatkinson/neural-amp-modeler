# Configuration file for the Sphinx documentation builder.
#
# Build locally
# (e.g. https://readthedocs.org/projects/neural-amp-modeler/builds/23551748/)
#
# $ python -m sphinx -T -b html -d _build/doctrees -D language=en . ./html

# -- Project information

project = "neural-amp-modeler"
copyright = "2024 Steven Atkinson"
author = "Steven Atkinson"

# TODO update this automatically from nam.__version__!
release = "0.12"
version = "0.12.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
