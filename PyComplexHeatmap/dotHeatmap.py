# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from .utils import mm2inch, plot_legend_list, despine, get_colormap
from .clustermap import ClusterMapPlotter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


# =============================================================================
def scale(values, vmin=None, vmax=None):
	if vmin is None:
		vmin = np.nanmin(values)
	if vmax is None:
		vmax = np.nanmax(values)
	if vmax == vmin:
		return [1 for j in values]
	delta = vmax - vmin
	return [(j - vmin) / delta for j in values]


# =============================================================================
def dotHeatmap2d(
	data,
	hue=None,
	vmin=None,
	vmax=None,
	ax=None,
	colors=None,
	cmap=None,
	max_s=None,
	spines=False,
	**kwargs
):
	"""
	Plot dot heatmap using a dataframe matrix as input.

	Parameters
	----------
	data : pd.DataFrame
		input matrix (pandas.DataFrame)
	hue : pd.DataFrame
		hue to control the colors and cmap of the dot.
	vmin : float
		minimal size for the dot.
	vmax : float
		maximal size for the dot
	ax : ax
		ax
	colors : dict
		colors to control the dot, keys should be the value in hue. if colors is a str, then colors will overwrite
		the parameter `c`.
	cmap : str of dict
		control the colormap of the dot, if cmap is a dict, keys should be the values from hue dataframe.
		If `cmap` is a str (such as 'Set1'), the parameter `colors` will overwrite the colors of dots.
		If `cmap` was a dict, then this paramter will overwrite the `colors`, and colors can only control the
		colors for markers.
	s : int, float, or dataframe
		control the sizes of dot.
	c : dataframe, or str
		control the colors of dots.
	marker : str, dataframe or dict
		when marker is a dict, hue must not be None, and keys are categorical values from hue, values should be marker.
	kwargs : dict
		such as s,c,marker, s,marker and colors can also be pandas.DataFrame.
		other kwargs passed to plt.scatter

	Returns
	-------
	axes:
	"""
	# print(locals())
	row_labels = data.index.tolist()
	col_labels = data.columns.tolist()
	# print(data.sort_index())
	data = data.stack().reset_index()
	data.columns = ["Row", "Col", "Value"]
	if ax is None:
		ax = plt.gca()
	df = data["Col"].apply(lambda j: col_labels.index(j) + 1).to_frame(name="X")
	df["Y"] = data["Row"].apply(lambda j: row_labels.index(j) + 1)
	df["Value"] = data.Value.values
	del data

	if max_s is None: #passed from DotClustermapPlotter, not None
		#The unit of size for the s parameter is squared points. This means
		# that the area of the marker is specified in points squared.
		# A point in this context is a unit of measure in typography,
		# equal to 1/72 of an inch. Therefore, if you specify s=100,
		# each marker's area will be 100 points squared, not its width or height.
		w, h = (
			ax.get_window_extent().width / ax.figure.dpi,
			ax.get_window_extent().height / ax.figure.dpi,
		) #unit is inch
		r = min(w * 72 / len(col_labels), h * 72 / len(row_labels))
		# r is the minimal of width and height for each scatter point, unit is point.
		max_s = r**2
	# s
	s = kwargs.pop("s", None)
	# print(s is None,vmin,vmax)
	if s is None:
		df["S"] = scale(df["Value"].abs().values,vmin=vmin, vmax=vmax)
	else:
		if isinstance(s, pd.DataFrame): # s is already normalized globally
			s = s.reindex(index=row_labels, columns=col_labels).stack().reset_index()
			s.columns = ["Row", "Col", "Value"]
			# df["S"] = scale(s.Value.abs().values) #scale to 0-1
			df['S'] = s.Value.values
		elif isinstance(s, (int, float)):
			df["S"] = s

	# hue
	if not hue is None:  # hue is a dataframe
		hue = hue.reindex(index=row_labels, columns=col_labels).stack().reset_index()
		hue.columns = ["Row", "Col", "Value"]
		df.insert(2, "Hue", hue.Value.values)
	# marker
	marker = kwargs.pop("marker", "o")
	if isinstance(marker, pd.DataFrame):
		marker = (
			marker.reindex(index=row_labels, columns=col_labels).stack().reset_index()
		)
		marker.columns = ["Row", "Col", "Value"]
		df["Markers"] = marker.Value.values
	elif isinstance(marker, str):
		df["Markers"] = marker
	elif isinstance(marker, dict):  # keys are values from hue, values should be marker.
		if hue is None:
			raise ValueError("when marker is a dict, hue must not be None")
		df["Markers"] = df.Hue.map(marker)
	else:
		raise ValueError("marker must be string, dataframe or dict")

	# colors
	c_ready = False
	if not colors is None and isinstance(colors, str):
		df["C"] = colors
		c_ready = True
	elif "c" in kwargs:  # c: dataframe or color, optional
		c = kwargs.pop("c")
		if isinstance(c, pd.DataFrame):
			c = c.reindex(index=row_labels, columns=col_labels).stack().reset_index()
			c.columns = ["Row", "Col", "Value"]
			df["C"] = c.Value.values
		else:  # str
			df["C"] = c
		c_ready = True
	elif hue is None:
		df["C"] = df.S.tolist()  # scale(data['Value'].values,vmin=vmin, vmax=vmax)
		kwargs.setdefault("cmap", cmap)
		c_ready = True
	elif not hue is None and isinstance(cmap, str):
		color_dict = {}  # keys are categorical values from hue, values are colors.
		if colors is None:  # using cmap
			col_list = df["Hue"].value_counts().index.tolist()
			for c in col_list:
				color_dict[c] = matplotlib.colors.to_hex(
					get_colormap(cmap)(col_list.index(c))
				)
		elif type(colors) == dict:
			color_dict = colors
		elif type(colors) == str:
			col_list = df["Hue"].value_counts().index.tolist()
			for c in col_list:
				color_dict[c] = colors
		else:
			raise ValueError("colors must be string or dict")

		df["C"] = df["Hue"].map(color_dict)
		c_ready = True
	kwargs.setdefault(
		"norm", matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
	)

	# print(f"max_s: {max_s}",df.S.min(),df.S.max()) #min and max S should be 0,1
	if c_ready and type(cmap) == str:
		kwargs["cmap"] = cmap
		for mk in df.Markers.unique():
			# df1 = df.query("Markers==@mk").copy()
			df1 = df.loc[df.Markers==mk].copy()
			if df1.shape[0] == 0:
				continue
			kwargs["marker"] = mk
			ax.scatter(
				x=df1.X.values,
				y=df1.Y.values,
				s=df1.S * max_s,
				c=df1.C.values,
				**kwargs
			)  # vmax=vmax,vmin=vmin,
	elif type(cmap) == dict and not hue is None:
		for h in cmap:  # key are hue, values are cmap
			# df1 = df.query("Hue==@h").copy()
			df1 = df.loc[df.Hue==h].copy()
			if df1.shape[0] == 0:
				continue
			kwargs["cmap"] = cmap[h]
			for mk in df1.Markers.unique():
				# df2 = df1.query("Markers==@mk").copy()
				df2 = df1.loc[df1.Markers==mk].copy()
				kwargs["marker"] = mk
				ax.scatter(
					x=df2.X.values,
					y=df2.Y.values,
					s=df2.S * max_s,
					c=df2.C.values,
					**kwargs
				)  #
	else:
		raise ValueError("cmap must be string or dict")

	ax.set_ylim([0.5, len(row_labels) + 0.5])
	ax.set_xlim(0.5, len(col_labels) + 0.5)
	y_locater = list(range(1, len(row_labels) + 1))
	x_locater = list(range(1, len(col_labels) + 1))
	ax.yaxis.set_major_locator(plt.FixedLocator(y_locater))
	ax.yaxis.set_minor_locator(plt.FixedLocator(np.array(y_locater) - 0.5))
	ax.xaxis.set_major_locator(plt.FixedLocator(x_locater))
	ax.xaxis.set_minor_locator(plt.FixedLocator(np.array(x_locater) - 0.5))
	ax.invert_yaxis()  # axis=1: left -> right, axis=0: bottom -> top.
	ax.set_yticklabels(row_labels)
	ax.set_xticklabels(col_labels)
	if not spines:
		despine(ax=ax, left=True, bottom=True, right=True, top=True)
		# for side in ["top", "right", "left", "bottom"]:
		#     ax.spines[side].set_visible(False)
	return ax


# =============================================================================
class DotClustermapPlotter(ClusterMapPlotter):
	"""
	DotClustermap (Heatmap) plotter, inherited from ClusterMapPlotter.
	Plot dot heatmap (clustermap) with annotation and legends.

	Parameters
	----------
	data : dataframe
		pandas dataframe or numpy array.
	x: str
		The column name in data.columns to be shown on the columns of heatmap / clustermap.
	y : str
		The column name in data.columns to be shown on the rows of heatmap / clustermap.
	value : str
		The column name in data.columns to control the sizes, or color of scatter (dot).
	hue : str, optional.
		The column name in data.columns to control the color, cmap, markers of scatter (dot).
	s : str or int, optional.
		The column name in data.columns to control the size of scatter (dot). If `s` is None,
		`value` will be used to control the sizes of dot. This parameter will overwrite value.
	c : str, optional.
		The column name in data.columns to control the color of scatter (dot).
		`c` can also be one color str, such as 'red'. If `c` is not given, colors of the dot
		will be determined by `cmap` or `colors`.
	marker :str or dict, optional.
		Please go to: https://matplotlib.org/stable/api/markers_api.html to see all available markers.
		Such as '.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_',
		default marker is 'o'.
		If marker is a string, it should be a marker to control the markers of scatter (dot).
		marker could also be a name of the column from data.columns.tolist()
		If marker is a dict, the keys should be the values from data[hue].values, and values should be marker.
	colors :dict.
		Keys should be the values from data[hue].values, and values should be color.
		It will be only used to control the colors of markers in figure legend.
	cmap :str or dict, optional.
		If cmap is a dict, the keys should be the values from data[hue].values, and values should be cmap.
		If cmap is a string, it should be colormap, such as 'Set1'.
	color_legend_kws: dict
		legend_kws passed to plot_color_dict_legend
	cmap_legend_kws: dict
		legend_kws passed to plot_cmap_legend
	dot_legend_kws: dict
		legend_kws passed to plot_marker_legend
	value_na : float or int
		used to fill na for data.pivot_table(index=self.y,columns=self.x,values=self.value,aggfunc=self.aggfunc).fillna(self.value_na)
	hue_na :
		float, str or int
		used to fill na for data.pivot_table(index=self.y,columns=self.x,values=self.hue,aggfunc=self.aggfunc).fillna(self.hue_na)
	s_na :
		floator int
		used to fill na for data.pivot_table(index=self.y,columns=self.x,values=self.s,aggfunc=self.aggfunc).fillna(self.s_na)
	c_na : float, int or str
		used to fill na for data.pivot_table(index=self.y,columns=self.x,values=self.c,aggfunc=self.aggfunc).fillna(self.c_na)
	aggfunc : function
		when there are multiple values for the same x and y, using aggfunc (default is np.mean) to aggregate them.
		aggfunc will be called in data.pivot(index=y,columns=x,values=value,aggfunc=aggfunc)
	spines: bool
		Whether show spines of the axes or not [False]
	max_s: float
		max size of the dot in scatter, default is None, will be inferred automatically.
	alpha: float [0,1]
		coefficient to scale the size of dot in figure legend, valid for marker and dot in legend.
	kwargs :dict
		Other kwargs passed to ClusterMapPlotter and dotHeatmap2d, such as max_s, vmin, vmax.

	Returns
	-------
	DotClustermapPlotter.
	"""
	def __init__(
		self,
		data=None,
		x=None,
		y=None,
		value=None,
		hue=None,
		s=None,
		c=None,
		marker="o",
		alpha=1,
		color_legend_kws={},
		cmap_legend_kws={},
		dot_legend_kws={},
		dot_legend_marker="o",
		aggfunc=np.mean,
		value_na=0,
		hue_na="NA",
		s_na=0,
		c_na=0,
		spines=False,
		max_s=None,
		**kwargs
	):
		kwargs["data"] = data
		self.x = x
		self.y = y
		self.value = value
		self.hue = hue
		self.s = s
		self.c = c
		self.marker = marker
		self.alpha = alpha
		self.aggfunc = aggfunc
		self.value_na = value_na
		self.hue_na = hue_na
		self.s_na = s_na
		self.c_na = c_na
		self.color_legend_kws = color_legend_kws
		self.cmap_legend_kws = cmap_legend_kws
		self.spines = spines
		self.dot_legend_kws = dot_legend_kws
		self.dot_legend_marker=dot_legend_marker
		self.max_s=max_s

		super().__init__(**kwargs)

	def format_data(self, data, mask=None, z_score=None, standard_scale=None):
		# self.data=data
		data2d = data.pivot_table(
			index=self.y, columns=self.x, values=self.value, aggfunc=self.aggfunc
		).fillna(self.value_na)
		# hue
		if not self.hue is None:
			self.kwargs["hue"] = data.pivot(
				index=self.y, columns=self.x, values=self.hue
			).fillna(self.hue_na)
		# s
		if not self.s is None:
			if isinstance(self.s, (int, float)):
				self.kwargs["s"] = self.s
				self.smax=self.s
				self.smin=None
			elif isinstance(self.s, str):
				self.kwargs["s"] = data.pivot_table(
					index=self.y, columns=self.x, values=self.s, aggfunc=self.aggfunc
				).fillna(self.s_na)
				self.smin = np.nanmin(self.kwargs["s"].values)
				self.smax = np.nanmax(self.kwargs["s"].values)
			elif isinstance(self.s,pd.Series):
				self.kwargs["s"] = data.assign(GivenS=self.s).pivot_table(
					index=self.y, columns=self.x, values='GivenS', aggfunc=self.aggfunc
				).fillna(self.s_na)
				self.smin = np.nanmin(self.kwargs["s"].values)
				self.smax = np.nanmax(self.kwargs["s"].values)
			else:
				raise ValueError("s must be a str, int or float!")

			if not self.smin is None: #s is a dataframe, perform standard normalization.
				delta=self.smax-self.smin
				self.kwargs["s"]=self.kwargs["s"].applymap(lambda x:(x-self.smin)/delta)

		# c
		if not self.c is None:
			if isinstance(self.c,pd.Series):
				self.kwargs["s"] = data.assign(GivenC=self.s).pivot_table(
					index=self.y, columns=self.x, values='GivenC', aggfunc=self.aggfunc
				).fillna(self.c_na)

			elif type(self.c)==str and self.c in data.columns:  # column name from data.columns
				self.kwargs["c"] = data.pivot_table(
					index=self.y, columns=self.x, values=self.c, aggfunc=self.aggfunc
				).fillna(self.c_na)

			elif type(self.c) == str:  # color, such as 'red'
				self.kwargs["c"] = self.c
			else:
				raise ValueError(
					"c must be a str: color or column name from data.columns"
				)
		# marker
		if not self.marker is None:
			if not isinstance(self.marker, (str, dict)):
				raise ValueError("marker must be a str or dict")
			if (
				isinstance(self.marker, str) and self.marker in data.columns
			):  # column name from data.columns
				self.kwargs["marker"] = data.pivot(
					index=self.y, columns=self.x, values=self.marker
				)
			elif isinstance(self.marker, str):  # color, such as 'red'
				self.kwargs["marker"] = self.marker
			else:
				if self.hue is None:
					raise ValueError("when marker is a dict, hue must not be None")
				self.kwargs["marker"] = self.marker

		if 'vmin' not in self.kwargs:
			self.vmin = np.nanmin(data2d.values)
			self.kwargs.setdefault("vmin", self.vmin)
		if 'vmax' not in self.kwargs:
			self.vmax = np.nanmax(data2d.values)
			self.kwargs.setdefault("vmax", self.vmax)

		if z_score is not None and standard_scale is not None:
			raise ValueError(
				"Cannot perform both z-scoring and standard-scaling on data"
			)
		if z_score is not None:
			data2d = self.z_score(data2d, z_score)
		if standard_scale is not None:
			data2d = self.standard_scale(data2d, standard_scale)
		return data2d

	def plot_matrix(self, row_order, col_order):
		if self.verbose >= 1:
			print("Plotting matrix..")
		nrows = len(row_order)
		ncols = len(col_order)

		ratio=self.kwargs.pop('ratio',None)
		if not ratio is None:
			print("Warning: ratio is deprecated, please use max_s instead")
		if self.max_s is None:
			self.max_s = ratio

		if self.max_s is None:
			# The unit of size for the s parameter is squared points. This means
			# that the area of the marker is specified in points squared.
			# A point in this context is a unit of measure in typography,
			# equal to 1/72 of an inch. Therefore, if you specify s=100,
			# each marker's area will be 100 points squared, not its width or height.
			w, h = (
				self.ax_heatmap.get_window_extent().width / self.ax_heatmap.figure.dpi,
				self.ax_heatmap.get_window_extent().height / self.ax_heatmap.figure.dpi,
			) # unit is inch
			r = min(w * 72 / self.data2d.shape[1], h * 72 / self.data2d.shape[0])
			# r is the minimal of width and height for each scatter point, unit is point.
			max_s = r ** 2
			if self.verbose >= 1:
				print(f"Inferred max_s (max size of scatter point) is: {max_s}")
		else:
			max_s = self.max_s
			if self.verbose >= 1:
				print(f"Using user provided max_s: {max_s}")
		self.kwargs['max_s'] = max_s
		self.col_split_gap_pixel = self.col_split_gap * mm2inch * self.ax.figure.dpi
		self.wspace = (
			(self.col_split_gap_pixel * ncols)
			/ (
				self.ax_heatmap.get_window_extent().width
				+ self.col_split_gap_pixel - self.col_split_gap_pixel * ncols
			)
		)
		self.row_split_gap_pixel = self.row_split_gap * mm2inch * self.ax.figure.dpi
		self.hspace = (
			(self.row_split_gap_pixel * nrows)
			/ (
				self.ax_heatmap.get_window_extent().height
				+ self.row_split_gap_pixel - self.row_split_gap_pixel * nrows
			)
		)
		self.heatmap_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
			nrows,
			ncols,
			hspace=self.hspace,
			wspace=self.wspace,
			subplot_spec=self.gs[1, 1],
			height_ratios=[len(rows) for rows in row_order],
			width_ratios=[len(cols) for cols in col_order],
		)
		self.heatmap_axes = np.empty(shape=(nrows, ncols), dtype=object)
		# if nrows > 1 or ncols > 1:
		self.ax_heatmap.set_axis_off()
		for i, rows in enumerate(row_order):
			for j, cols in enumerate(col_order):
				ax1 = self.ax_heatmap.figure.add_subplot(
					self.heatmap_gs[i, j],
					sharex=self.heatmap_axes[0, j],
					sharey=self.heatmap_axes[i, 0],
				)
				# ax1.set_xlim([0, len(rows)])
				# ax1.set_ylim([0, len(cols)])
				kwargs = self.kwargs.copy()
				# print(kwargs)
				dotHeatmap2d(
					self.data2d.loc[rows, cols],
					cmap=kwargs.pop("cmap", self.cmap),
					ax=ax1,
					spines=self.spines,
					**kwargs
				)
				self.heatmap_axes[i, j] = ax1
				ax1.yaxis.label.set_visible(False)
				ax1.xaxis.label.set_visible(False)
				ax1.tick_params(
					which="both",
					left=False,
					right=False,
					labelleft=False,
					labelright=False,
					top=False,
					bottom=False,
					labeltop=False,
					labelbottom=False,
				)

	def collect_legends(self):
		if self.verbose >= 1:
			print("Collecting legends..")
		self.legend_list = []
		self.label_max_width = 0
		for annotation in [
			self.top_annotation,
			self.bottom_annotation,
			self.left_annotation,
			self.right_annotation,
		]:
			if not annotation is None:
				annotation.collect_legends()
				if annotation.plot_legend and len(annotation.legend_list) > 0:
					self.legend_list.extend(annotation.legend_list)
				# print(annotation.label_max_width,self.label_max_width)
				if annotation.label_max_width > self.label_max_width:
					self.label_max_width = annotation.label_max_width
		if self.legend:
			if (isinstance(self.cmap, str) and not self.hue is None and self.c is None):  #
				color_dict = {}
				col_list = self.kwargs["hue"].unstack().value_counts().index.tolist()
				# print(col_list,self.kwargs['hue'])
				for c in col_list:
					color_dict[c] = matplotlib.colors.to_hex(
						get_colormap(self.cmap)(col_list.index(c))
					)
				self.legend_list.append(
					[
						color_dict,
						self.hue,
						self.color_legend_kws,
						len(color_dict),
						"color_dict",
					]
				)
			cmap = self.cmap
			c = self.kwargs.get("c", None)
			cmap_legend_kws = self.cmap_legend_kws.copy()
			# cmap_legend_kws["vmax"] = self.kwargs.get('vmax',1)
			# cmap_legend_kws["vmin"] = self.kwargs.get('vmin',0)
			cmap_legend_kws.setdefault("vmin", self.kwargs.get('vmin'))  # round(vmin, 2))
			cmap_legend_kws.setdefault("vmax", self.kwargs.get('vmax'))  # round(vmax, 2))
			if (
				not cmap is None
				and type(cmap) == str
				and not c is None
				and type(c) != str
			):
				# print(cmap_legend_kws)
				self.legend_list.append([cmap, self.value, cmap_legend_kws, 4, "cmap"])
			if type(cmap) == dict:
				for k in cmap:
					self.legend_list.append([cmap[k], k, cmap_legend_kws, 4, "cmap"])
			marker = self.kwargs.get("marker", None)
			# ax = self.heatmap_axes[0, 0]
			# w, h = (
			#     ax.get_window_extent().width / ax.figure.dpi,
			#     ax.get_window_extent().height / ax.figure.dpi,
			# )
			# r = min(w * 72 / len(self.col_order[0]), h * 72 / len(self.row_order[0]))
			max_s=self.kwargs['max_s']
			if type(marker) == dict and not self.hue is None:
				self.legend_list.append(
					[
						(marker, self.kwargs.get("colors", None), np.sqrt(max_s) * self.alpha),
						self.hue,
						self.dot_legend_kws,
						len(marker),
						"markers",
					] #size of s in scatter equal to marker_size**2
				)  # markersize is r*0.8
			# dot size legend:
			if type(self.s) == str:
				# s=self.kwargs.get('s',None)
				# colors=self.kwargs.get('colors',None)
				markers1 = {}
				ms = {}
				for f in [1, 0.8, 0.6, 0.4, 0.2]:
					k = str(round(f * self.smax, 2))
					markers1[k] = self.dot_legend_marker
					ms[k] = f  * np.sqrt(max_s) * self.alpha
					# ms[k] = np.sqrt(f * max_s * self.alpha)
				title = self.s if not self.s is None else self.value
				self.legend_list.append(
					[
						(markers1, None, ms),
						title,
						self.dot_legend_kws,
						len(markers1),
						"markers",
					]
				)
			heatmap_label_max_width = (
				max([label.get_window_extent().width for label in self.yticklabels])
				if len(self.yticklabels) > 0
				else 0
			)
			if (
				heatmap_label_max_width >= self.label_max_width
				or self.legend_anchor == "ax_heatmap"
			):
				self.label_max_width = heatmap_label_max_width * 1.1
			if len(self.legend_list) > 1:
				self.legend_list = sorted(self.legend_list, key=lambda x: x[3])

	def post_processing(self):
		if not self.spines:
			for ax in self.heatmap_axes.ravel():
				despine(ax=ax, left=True, bottom=True, right=True, top=True)


if __name__ == "__main__":
	pass
