#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from .clustermap import heatmap,ClusterMapPlotter,composite
from .oncoPrint import oncoprint,oncoPrintPlotter
from .annotations import *
from .dotHeatmap import *
from .colors import *
from .tools import *
# from .example import *

#__all__=['*']
__version__='1.4'

_ROOT = os.path.abspath(os.path.dirname(__file__))
