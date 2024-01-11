# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import datetime
import longitudinal_tomography

# -- Project information -----------------------------------------------------

project = 'longitudinal_tomography'
copyright = '2020, Christoffer Hjertø Grindheim'
author = 'Christoffer Hjertø Grindheim'

copyright = "{0}, CERN".format(datetime.datetime.now().year)

# The full version, including alpha/beta/rc tags
version = longitudinal_tomography.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['acc_py_sphinx.theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'bizstyle'
# html_theme = 'haiku'
# html_theme = 'nature'
# html_theme = 'traditional'
# html_theme = 'agogo'
# html_theme = 'scrolls'
# html_theme = 'sphinxdoc'
# html_theme = 'classic'
# html_theme = 'alabaster'
# html_theme = 'pyramid'

html_theme = 'acc_py'

html_show_sphinx = False
html_show_sourcelink = True


# -- Options for sphinx.ext.autosummary

autosummary_generate = True
autosummary_imported_members = True


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
