# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, CSS4_COLORS
import random
from .utils import _calculate_luminance
import pandas as pd
default_cmaps = matplotlib.pyplot.colormaps()
import colorsys
import numpy as np
from matplotlib import colors as mcolors
named_colors=mcolors.CSS4_COLORS
named_colors.update(mcolors.BASE_COLORS) #kes is color string, values are hex
# named_colors.update(mcolors.TABLEAU_COLORS)
named_colors.update(mcolors.CSS4_COLORS)
for k in named_colors:
	if isinstance(named_colors[k],tuple):
		named_colors[k]=mcolors.to_hex(named_colors[k])

def register_cmap(c):
	try:
		plt.register_cmap(cmap=c)
	except:
		matplotlib.colormaps.register(c, force=True)

def define_cmap():
	all_cmaps = matplotlib.pyplot.colormaps()
	# if "binarize" not in all_cmaps:
	#     c = LinearSegmentedColormap.from_list(
	#         "binarize", [(0, 'lightgray'), (1, "black")]
	#     )
	if "exp1" not in all_cmaps:
		c = LinearSegmentedColormap.from_list(
			"exp1", [(0, "blue"), (0.5, "yellow"), (1, "red")]
		)
		register_cmap(c)
	if "exp2" not in all_cmaps:
		c = LinearSegmentedColormap.from_list(
			"exp2", [(0, "#4A5EA0"), (0.5, "#F9FFB2"), (1, "#A42A26")]
		)
		register_cmap(c)
	if "meth1" not in all_cmaps:
		c = LinearSegmentedColormap.from_list(
			"meth1",
			[
				(0, "#282972"),
				(0.2, "#36C4F2"),
				(0.4, "#C1D62F"),
				(0.6, "#F9AD17"),
				(0.8, "#EC3024"),
				(1, "#7D1416"),
			],
		)
		register_cmap(c)
	if "meth2" not in all_cmaps:
		c = LinearSegmentedColormap.from_list(
			"meth2",
			[
				(0, "#4B5DC6"),
				(0.1, "#439CFE"),
				(0.3, "#55B849"),
				(0.5, "gold"),
				(0.7, "darkorange"),
				(0.9, "tomato"),
				(1, "red"),
			],
		)
		register_cmap(c)
	if "diverging1" not in all_cmaps:
		c = LinearSegmentedColormap.from_list(
			"diverging1", [(0, "#67a9cf"), (0.5, "#f7f7f7"), (1, "#ef8a62")]
		)
		register_cmap(c)

	if "nature3_1" not in all_cmaps:
		colors = ['#BF1D2D','#262626','#293890']
		c = ListedColormap(colors,"nature3_1")
		register_cmap(c)

	if "nature3_2" not in all_cmaps:
		colors = ['#F1B656','#397FC7','#040676']
		c = ListedColormap(colors,"nature3_2")
		register_cmap(c)

	if "nature3_3" not in all_cmaps:
		colors = ['#2C91E0','#3ABF99','#F0A73A']
		c = ListedColormap(colors,"nature3_3")
		register_cmap(c)

	if "nature4_1" not in all_cmaps:
		colors = ['#DE582B','#1868B2','#018A67','#F3A332']
		c = ListedColormap(colors,"nature4_1")
		register_cmap(c)

	if "nature4_2" not in all_cmaps:
		colors = ['#32037D','#7C1A97','#C94E65','#D9995B']
		c = ListedColormap(colors,"nature4_2")
		register_cmap(c)

	if "nature4_3" not in all_cmaps:
		colors = ['#015493','#019092','#999999','#F4A99B']
		c = ListedColormap(colors,"nature4_3")
		register_cmap(c)

	if "nature4_4" not in all_cmaps:
		colors = ['#2F2D54','#9193B4','#BD9AAD','#E8D2B3']
		c = ListedColormap(colors,"nature4_4")
		register_cmap(c)

	if "nature5_1" not in all_cmaps:
		colors = ['#60966D','#5B3660','#FFC839','#E90F44','#63ADEE']
		c = ListedColormap(colors,"nature5_1")
		register_cmap(c)

	if "nature5_2" not in all_cmaps:
		colors = ['#ABC6E4','#C39398','#FCDABA','#A7D2BA','#D0CADE']
		c = ListedColormap(colors,"nature5_2")
		register_cmap(c)

	if "nature6_1" not in all_cmaps:
		colors = ['#C72228','#F98F34','#0C4E9B','#F5867F','#FFBC80','#6B98C4']
		c = ListedColormap(colors,"nature6_1")
		register_cmap(c)

	if "nature6_2" not in all_cmaps:
		colors = ['#3B549D','#D3272B','#6BBC46','#9EAAD1','#F29091','#B4DEA2']
		c = ListedColormap(colors,"nature6_2")
		register_cmap(c)

	if "random50" not in all_cmaps:
		colors = []
		for c in CSS4_COLORS:
			l = _calculate_luminance(c)
			if l > 0.25 and l < 0.8:
				colors.append(c)
		# colors=[c for c in CSS4_COLORS.keys() if c not in ["white", "snow"] and "gray" not in c]
		c = ListedColormap(
			random.sample(colors,50),
			"random50",
		)
		register_cmap(c)

	if "random100" not in all_cmaps:
		# colors = []
		# for c in CSS4_COLORS:
		# 	l = _calculate_luminance(c)
		# 	if l > 0.25 and l < 0.8:
		# 		colors.append(c)
		colors=[c for c in CSS4_COLORS.keys() if c not in ["white", "snow"] and "gray" not in c]
		c = ListedColormap(
			random.sample(colors,100),
			"random100",
		)
		register_cmap(c)

	if "parula" not in all_cmaps:
		cm_data = [
			[0.2081, 0.1663, 0.5292],
			[0.2116238095, 0.1897809524, 0.5776761905],
			[0.212252381, 0.2137714286, 0.6269714286],
			[0.2081, 0.2386, 0.6770857143],
			[0.1959047619, 0.2644571429, 0.7279],
			[0.1707285714, 0.2919380952, 0.779247619],
			[0.1252714286, 0.3242428571, 0.8302714286],
			[0.0591333333, 0.3598333333, 0.8683333333],
			[0.0116952381, 0.3875095238, 0.8819571429],
			[0.0059571429, 0.4086142857, 0.8828428571],
			[0.0165142857, 0.4266, 0.8786333333],
			[0.032852381, 0.4430428571, 0.8719571429],
			[0.0498142857, 0.4585714286, 0.8640571429],
			[0.0629333333, 0.4736904762, 0.8554380952],
			[0.0722666667, 0.4886666667, 0.8467],
			[0.0779428571, 0.5039857143, 0.8383714286],
			[0.079347619, 0.5200238095, 0.8311809524],
			[0.0749428571, 0.5375428571, 0.8262714286],
			[0.0640571429, 0.5569857143, 0.8239571429],
			[0.0487714286, 0.5772238095, 0.8228285714],
			[0.0343428571, 0.5965809524, 0.819852381],
			[0.0265, 0.6137, 0.8135],
			[0.0238904762, 0.6286619048, 0.8037619048],
			[0.0230904762, 0.6417857143, 0.7912666667],
			[0.0227714286, 0.6534857143, 0.7767571429],
			[0.0266619048, 0.6641952381, 0.7607190476],
			[0.0383714286, 0.6742714286, 0.743552381],
			[0.0589714286, 0.6837571429, 0.7253857143],
			[0.0843, 0.6928333333, 0.7061666667],
			[0.1132952381, 0.7015, 0.6858571429],
			[0.1452714286, 0.7097571429, 0.6646285714],
			[0.1801333333, 0.7176571429, 0.6424333333],
			[0.2178285714, 0.7250428571, 0.6192619048],
			[0.2586428571, 0.7317142857, 0.5954285714],
			[0.3021714286, 0.7376047619, 0.5711857143],
			[0.3481666667, 0.7424333333, 0.5472666667],
			[0.3952571429, 0.7459, 0.5244428571],
			[0.4420095238, 0.7480809524, 0.5033142857],
			[0.4871238095, 0.7490619048, 0.4839761905],
			[0.5300285714, 0.7491142857, 0.4661142857],
			[0.5708571429, 0.7485190476, 0.4493904762],
			[0.609852381, 0.7473142857, 0.4336857143],
			[0.6473, 0.7456, 0.4188],
			[0.6834190476, 0.7434761905, 0.4044333333],
			[0.7184095238, 0.7411333333, 0.3904761905],
			[0.7524857143, 0.7384, 0.3768142857],
			[0.7858428571, 0.7355666667, 0.3632714286],
			[0.8185047619, 0.7327333333, 0.3497904762],
			[0.8506571429, 0.7299, 0.3360285714],
			[0.8824333333, 0.7274333333, 0.3217],
			[0.9139333333, 0.7257857143, 0.3062761905],
			[0.9449571429, 0.7261142857, 0.2886428571],
			[0.9738952381, 0.7313952381, 0.266647619],
			[0.9937714286, 0.7454571429, 0.240347619],
			[0.9990428571, 0.7653142857, 0.2164142857],
			[0.9955333333, 0.7860571429, 0.196652381],
			[0.988, 0.8066, 0.1793666667],
			[0.9788571429, 0.8271428571, 0.1633142857],
			[0.9697, 0.8481380952, 0.147452381],
			[0.9625857143, 0.8705142857, 0.1309],
			[0.9588714286, 0.8949, 0.1132428571],
			[0.9598238095, 0.9218333333, 0.0948380952],
			[0.9661, 0.9514428571, 0.0755333333],
			[0.9763, 0.9831, 0.0538],
		]
		c = LinearSegmentedColormap.from_list("parula", cm_data)
		register_cmap(c)

def get_palettable_colors():
	"""
	category: sequential, diverging, qualitative
	"""
	R=[]
	for category in ['sequential','diverging','qualitative']:
		for source in dir(palettable):
			color_source=getattr(palettable,source)
			if not hasattr(color_source,category):
				continue
			color_category=getattr(color_source,category)
			color_names=[]
			for name in dir(color_category):
				color=getattr(color_category,name)
				if not hasattr(color,'mpl_colors'):
					continue
				R.append([category,color.name,color.type,color.number,color.colors,color.hex_colors,color.mpl_colors,color.mpl_colormap])
	df=pd.DataFrame(R,columns=['category','name','type','number','colors','hex_colors','mpl_colors','mpl_colormap'])
	df.insert(1,'prefix',df.name.apply(lambda x:x.split('_')[0]))
	df.sort_values(['prefix','number'],ascending=[True,False],inplace=True)
	max_n = df.groupby('prefix').number.max().to_dict()
	df.insert(5, 'max_n', df.prefix.map(max_n))
	df = df.loc[df.number == df.max_n]
	df['Name'] = df.name.apply(lambda x: x.split('_')[0] if not x.endswith('_r') else x.split('_')[0] + '_r')
	mpl_colormaps = []
	for cmap, prefix in zip(df.mpl_colormap.tolist(), df.Name.tolist()):
		cmap.name = prefix
		mpl_colormaps.append(cmap)
	df.mpl_colormap = mpl_colormaps
	df = df.loc[~ df.Name.isin(default_cmaps)]
	return df

def register_palettable():
	df=get_palettable_colors()
	all_cmaps = matplotlib.pyplot.colormaps()

	# register qualitative colormap
	df1=df.loc[df.category == 'qualitative']
	for colors,name in zip (df1.mpl_colors.tolist(),df1.Name.tolist()):
		if name in all_cmaps:
			continue
		c = ListedColormap(colors,name)
		register_cmap(c)

	# register sequential and diverging colormap
	df1 = df.loc[df.category != 'qualitative']
	for colors,name in zip (df1.mpl_colors.tolist(),df1.Name.tolist()):
		if name in all_cmaps:
			continue
		c = LinearSegmentedColormap.from_list(name, colors)
		register_cmap(c)

def identify_color_format(color):
	if color in named_colors:
		return 'named_color'
	elif isinstance(color, str) and color.startswith('#'):
		if len(color) == 7 or len(color) == 4:
			return 'HEX'
	elif isinstance(color, tuple) and len(color) == 3:
		if all(isinstance(val, int) and 0 <= val <= 255 for val in color):
			return 'RGB (integer)'
		elif all(isinstance(val, float) and 0 <= val <= 1 for val in color):
			return 'RGB (float)'
	elif isinstance(color, tuple) and len(color) == 3:
		h, l, s = color
		if (isinstance(h, float) and 0 <= h <= 1) or (isinstance(h, int) and 0 <= h <= 360):
			if isinstance(l, float) and 0 <= l <= 1 and isinstance(s, float) and 0 <= s <= 1:
				return 'HLS'
	return 'Unknown'

def interpolate_colors(color1, color2, num_colors, fmt='RGB'):
	if fmt == 'HLS':
		color1 = colorsys.hls_to_rgb(*color1)
		color2 = colorsys.hls_to_rgb(*color2)
	elif fmt != 'RGB':
		color1 = mcolors.to_rgb(color1)
		color2 = mcolors.to_rgb(color2)
	color1 = np.array(color1)
	color2 = np.array(color2)
	colors = [tuple(color1 + (color2 - color1) * i / (num_colors - 1)) for i in range(num_colors)]
	return colors


def generate_quantive_colors(color, n=5, white=(0.9, 0.9, 0.9), black=(0.3, 0.3, 0.3), ret_fmt='HEX'):
	fmt = identify_color_format(color)
	if not fmt.startswith('RGB'):
		color = mcolors.to_rgb(color)
	light_colors = interpolate_colors(white, color, n // 2 + 2)[1:-1]  # from white to color
	dark_colors = interpolate_colors(color, black, n // 2 + 2)[1:-1]  # from color to black;white=(1,1,1),black=(0,0,0)
	if len(light_colors) + len(dark_colors) == n:
		results = light_colors + dark_colors
	else:
		results = light_colors + [color] + dark_colors
	if ret_fmt == 'RGB':
		return results
	elif ret_fmt == 'HEX':
		return [mcolors.to_hex(c) for c in results]

try:
	define_cmap()
except:
	print("Register cmap failed!")

try:
	import palettable
	register_palettable()
except:
	print("Warning: Please install palettable to enable palettable colormaps")

