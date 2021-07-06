# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# import sphinx_readable_theme
# import sphinx_bootstrap_theme
# import sphinx_rtd_theme

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'AD-PEPS'
copyright = '2021, Boris Ponsioen'
author = 'Boris Ponsioen'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        'sphinx.ext.autodoc',
        'sphinx.ext.autosummary',
        'sphinx_autodoc_typehints',
        'sphinxarg.ext',
        'sphinx.ext.viewcode',
        'sphinx.ext.napoleon',
        # 'sphinx_autodoc_typehints',
        # 'sphinx_autodoc_napoleon_typehints',
        # 'sphinx_rtd_theme',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True
autoclass_content = "both"
add_module_names = False
napoleon_attr_annotations = True

autodoc_typehints = "description"

autodoc_type_aliases = {'Tensor_like': 'adpeps.types.TensorType'}

# -- Options for HTML output -------------------------------------------------

# html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
# html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'readable'
# html_theme = 'bootstrap'
# html_theme = "sphinx_rtd_theme"
# html_theme = 'sphinx_material'
# # Material theme options (see theme.conf for more information)
# html_theme_options = {
#     'globaltoc_collapse': False,
#     # If True, show hidden TOC entries
#     'globaltoc_includehidden': True,
# }
html_theme = "pydata_sphinx_theme"
html_theme_options = {
  "show_prev_next": False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
