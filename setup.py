# -*- coding: utf-8 -*-
"""
Created on Thu may 5 11:27:17 2022

@author: DingWB
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
# print(long_description)

# release new version
setup(
    name="PyComplexHeatmap",
	use_scm_version={'version_scheme': 'post-release',"local_scheme": "no-local-version"},
	setup_requires=['setuptools_scm'],
	description="A Python package to plot complex heatmap",
    # long_description="#PyComplexHeatmap\n##Documentation:https://dingwb.github.io/PyComplexHeatmap/build/html/index.html",
    # long_description_content_type='text/markdown',
    author="Wubin Ding",
    author_email="ding.wu.bin.gm@gmail.com",
    url="https://github.com/DingWB/PyComplexHeatmap",
    packages=["PyComplexHeatmap"],  # src
    install_requires=["matplotlib","numpy","pandas>=1.3.5", "scipy","palettable"],
    include_package_data=True,
)
# rm -rf dist && rm -rf PyComplexHeatmap/PyComplexHeatmap.egg-info/
# python setup.py sdist bdist_wheel
# twine upload dist/*
