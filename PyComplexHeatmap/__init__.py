#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from .clustermap import (heatmap, ClusterMapPlotter, composite,
                         DendrogramPlotter)
from .oncoPrint import oncoprint, oncoPrintPlotter
from .annotations import *
from .dotHeatmap import *
from .colors import *
from .utils import set_default_style

# __all__=['*']
from ._version import version as __version__
# __version__ = "1.6.5"

_ROOT = os.path.abspath(os.path.dirname(__file__))
