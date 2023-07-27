#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from .clustermap import heatmap,ClusterMapPlotter,composite
from .oncoPrint import oncoprint,oncoPrintPlotter
from .annotations import *
from .dotHeatmap import *
from .colors import *
from .tools import *
from .utils import set_default_style

#__all__=['*']
__version__='1.5.3'

_ROOT = os.path.abspath(os.path.dirname(__file__))
