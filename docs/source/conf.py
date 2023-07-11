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
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
print(sys.path)
from recommonmark.parser import CommonMarkParser
import sphinx_rtd_theme
import sphinx_sizzle_theme,sphinx_pdj_theme

# -- Project information -----------------------------------------------------

project = 'PyComplexHeatmap'
copyright = '2022, Wubin Ding'
author = 'Wubin Ding'

# The full version, including alpha/beta/rc tags
release = '1.5.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'recommonmark',
    'sphinx.ext.napoleon',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

from recommonmark.parser import CommonMarkParser
source_parsers = {'.md': CommonMarkParser}
source_suffix = ['.rst', '.md']

master_doc='index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style='sphinx'
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme' #Read the Docs; pip install --upgrade sphinx-rtd-theme
# documentation: https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
# html_theme = 'sphinx_pdj_theme' #pip install sphinx_sizzle_theme
html_theme_path=[sphinx_rtd_theme.get_html_theme_path()]
# html_theme_path=[sphinx_pdj_theme.get_html_theme_path()]
html_theme_options = {
    'analytics_id': 'G-VRB2NBWG05',
    'collapse_navigation':False,
    'globaltoc_collapse':False,
    'globaltoc_maxdepth':3,
    'collapse_navigation': False,
    'display_version': True,
    'sidebarwidth': 200, #sidebarwidth
    'navigation_depth': 6}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html'
    ]
}

html_context = {
  'display_github': True,
  'github_user': 'DingWB',
  'github_repo': 'PyComplexHeatmap',
  'github_version': 'main/docs/source/',
}

htmlhelp_basename = 'PyComplexHeatmapDoc'

latex_documents = [
    (master_doc, 'PyComplexHeatmap.tex', 'PyComplexHeatmap Documentation',
     'Wubin Ding', 'manual'),
]

# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'PyComplexHeatmap', 'PyComplexHeatmap Documentation',
     [author], 1)
]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'PyComplexHeatmap', 'PyComplexHeatmap Documentation',
     author, 'PyComplexHeatmap', 'One line description of project.',
     'Miscellaneous'),
]

# googleanalytics
# googleanalytics_id = 'G-F99YH1DGPY'


# Change the width of content, add the following to the css
html_css_files = [
    'css/custom.css',
]
# .wy-nav-content {
#     max-width: 75% !important;
# }