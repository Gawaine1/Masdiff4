# Configuration file for the Sphinx documentation builder.

project = 'MASDiff'
copyright = '2024, MASDiff Contributors'
author = 'MASDiff Contributors'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# autodoc settings
autodoc_mock_imports = [
    'torch', 'ray', 'traci', 'sumolib',
]
