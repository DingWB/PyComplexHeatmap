# For developers
```shell
pip install sphinx sphinx-autobuild sphinx-rtd-theme pandoc nbsphinx sphinx_pdj_theme sphinx_sizzle_theme recommonmark readthedocs-sphinx-search
conda install conda-forge::pandoc

mkdir -p docs && cd docs
sphinx-quickstart
# Separate source and build directories (y/n) [n]: y
# Project name: adataviz

# vim source/conf.py
# add *.rst

# cd docs
# vim index.html: <meta http-equiv="refresh" content="0; url=./build/html/index.html" />
cd docs
rm -rf build
ln -s $HOME/Projects/Github/PyComplexHeatmap/notebooks source/notebooks
sphinx-apidoc -e -o source -f ../../PyComplexHeatmap
make html
rm -rf source/notebooks
cd ..
ls
ls docs

vim .nojekyll #create empty file
```