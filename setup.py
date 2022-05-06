# -*- coding: utf-8 -*-
"""
Created on Thu may 5 11:27:17 2022

@author: DingWB
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
   name='pyclustermap',
   version='1.0',
   description='A Python package to plot complex heatmap',
   author='Wubin Ding',
   author_email='ding.wu.bin.gm@gmail.com',
   url="https://github.com/DingWB/pyclustermap",
   packages=['pyclustermap'],
   install_requires=['matplotlib>=3.4.3','pandas'],
   #scripts=['scripts/pyclustermap'],
)