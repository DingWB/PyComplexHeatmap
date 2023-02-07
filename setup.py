# -*- coding: utf-8 -*-
"""
Created on Thu may 5 11:27:17 2022

@author: DingWB
"""

try:
    from setuptools import setup,find_packages
except ImportError:
    from distutils.core import setup,find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
# print(long_description)

setup(
   name='PyComplexHeatmap',
   version='1.2.2',
   description='A Python package to plot complex heatmap',
   # long_description="#PyComplexHeatmap\n##Documentation:https://dingwb.github.io/PyComplexHeatmap/build/html/index.html",
   # long_description_content_type='text/markdown',
   author='Wubin Ding',
   author_email='ding.wu.bin.gm@gmail.com',
   url="https://github.com/DingWB/PyComplexHeatmap",
   # packages=['PyComplexHeatmap'],
   package_dir={'':'PyComplexHeatmap'},
   packages=find_packages('PyComplexHeatmap'),
   # install_requires=['matplotlib>=3.3.1','pandas'],
   data_files=[('Lib/site-packages/PyComplexHeatmap/data',['data/mammal_array.pkl'])]
   #scripts=['scripts/PyComplexHeatmap'],
)
