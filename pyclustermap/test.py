# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import os, sys
import pandas as pd
import matplotlib.pylab as plt
plt.rcParams['figure.dpi']=150
plt.rcParams['savefig.dpi']=300

sys.path.append(os.path.expanduser("~/Scripts/python/pyclustermap"))
import pyclustermap
pyclustermap.clustermap_example0()