# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from .utils import mm2inch
from .utils import (
    _calculate_luminance,
    cluster_labels,
    plot_legend_list,
    define_cmap,
    get_colormap
)
from .clustermap import plot_heatmap,heatmap
# -----------------------------------------------------------------------------
class AnnotationBase():
    """
    Base class for annotation objects.

    Parameters
    ----------
    df: dataframe
        a pandas series or dataframe (only one column).
    cmap: str
        colormap, such as Set1, Dark2, bwr, Reds, jet, hsv, rainbow and so on. Please see
        https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html for more information, or run
        matplotlib.pyplot.colormaps() to see all availabel cmap.
        default cmap is 'auto', it would be determined based on the dtype for each columns of df.
    colors: dict, list or str
        a dict or list (for boxplot, barplot) or str.
        If colors is a dict: keys should be exactly the same as df.iloc[:,0].unique(),
        values for the dict should be colors (color names or HEX color).
        If  colors is a list, then the length of this list should be equal to df.iloc[:,0].nunique()
        If colors is a string, means all values in df.iloc[:,0].unique() share the same color.
    height: float
        height (if axis=1) / width (if axis=0) for the annotation size.
    legend: bool
        whether to plot legend for this annotation when legends are plotted or
        plot legend with HeatmapAnnotation.plot_legends().
    legend_kws: dict
        vmax, vmin and other kws passed to plt.legend, such as title, prop, fontsize, labelcolor,
        markscale, frameon, framealpha, fancybox, shadow, facecolor, edgecolor, mode and so on, for more
        arguments, pleast type ?plt.legend. There is an additional parameter `color_text` (default is True),
        which would set the color of the text to the same color as legend marker. if one set
        `legend_kws={'color_text':False}`, then, black would be the default color for the text.
        If the user want to use a custom color instead of black (such as blue), please set
        legend_kws={'color_text':False,'labelcolor':'blue'}.
    plot_kws: dict
        other plot kws passed to annotation.plot, such as rotation, rotation_mode, ha, va,
        annotation_clip, arrowprops and matplotlib.text.Text for anno_label. For example, in anno_simple,
        there is also kws: vmin and vmax, if one want to change the range, please try:
        anno_simple(df_box.Gene1,vmin=0,vmax=1,legend_kws={'vmin':0,'vmax':1}).

    Returns
    ----------
    Class AnnotationBase.
    """

    def __init__(self, df=None, cmap='auto', colors=None,
                 height=None, legend=None, legend_kws=None, **plot_kws):
        self._check_df(df)
        self.label = None
        self.ylim = None
        self.color_dict = None
        self.nrows = self._n_rows()
        self.ncols = self._n_cols()
        self.height = self._height(height)
        self._type_specific_params()
        self.legend = legend
        self.legend_kws = legend_kws if not legend_kws is None else {}
        self._set_default_plot_kws(plot_kws)

        if colors is None:
            self._check_cmap(cmap)
            self._calculate_colors()  # modify self.plot_data, self.color_dict (each col is a dict)
        else:
            self._check_colors(colors)
            self._calculate_cmap()  # modify self.plot_data, self.color_dict (each col is a dict)
        self.plot_data = self.df.copy()

    def _check_df(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise TypeError("df must be a pandas DataFrame or Series.")

    def _n_rows(self):
        return self.df.shape[0]

    def _n_cols(self):
        return self.df.shape[1]

    def _height(self, height):
        return 3 * self.ncols if height is None else height

    def _set_default_plot_kws(self, plot_kws):
        self.plot_kws = {} if plot_kws is None else plot_kws
        self.plot_kws.setdefault('zorder', 10)

    def update_plot_kws(self, plot_kws):
        self.plot_kws.update(plot_kws)

    def set_label(self, label):
        self.label = label

    def set_legend(self, legend):
        if self.legend is None:
            self.legend = legend

    def set_axes_kws(self, subplot_ax):
        # ax.set_xticks(ticks=np.arange(1, self.nrows + 1, 1), labels=self.plot_data.index.tolist())
        if self.axis == 1:
            if self.ticklabels_side == 'left':
                subplot_ax.yaxis.tick_left()
            elif self.ticklabels_side == 'right':
                subplot_ax.yaxis.tick_right()
            subplot_ax.yaxis.set_label_position(self.label_side)
            subplot_ax.yaxis.label.update(self.label_kws)
            # ax.yaxis.labelpad = self.labelpad
            subplot_ax.xaxis.set_visible(False)
            subplot_ax.yaxis.label.set_visible(False)
        else:  # axis=0, row annotation
            if self.ticklabels_side == 'top':
                subplot_ax.xaxis.tick_top()
            elif self.ticklabels_side == 'bottom':
                subplot_ax.xaxis.tick_bottom()
            subplot_ax.xaxis.set_label_position(self.label_side)
            subplot_ax.xaxis.label.update(self.label_kws)
            subplot_ax.xaxis.set_tick_params(self.ticklabels_kws)
            # ax.yaxis.labelpad = self.labelpad
            subplot_ax.yaxis.set_visible(False)
            subplot_ax.xaxis.label.set_visible(False)
            # subplot_ax.invert_yaxis()

    def _check_cmap(self, cmap):
        if cmap == 'auto':
            col = self.df.columns.tolist()[0]
            if self.df.dtypes[col] == object:
                if self.df[col].nunique() <= 10:
                    self.cmap = 'Set1'
                elif self.df[col].nunique() <= 20:
                    self.cmap = 'tab20'
                else:
                    self.cmap = 'cmap50'
            elif self.df.dtypes[col] == float or self.df.dtypes[col] == int:
                self.cmap = 'jet'
            else:
                raise TypeError("Can not assign cmap for column %s, please specify cmap" % col)
        elif type(cmap) == str:
            self.cmap = cmap
        else:
            raise TypeError("Unknow data type for cmap!")
        if get_colormap(self.cmap).N == 256:  # then heatmap will automatically calculate vmin and vmax
            try:
                self.plot_kws.setdefault('vmax', np.nanmax(self.df.values))
                self.plot_kws.setdefault('vmin', np.nanmin(self.df.values))
            except:
                pass

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.color_dict = {}
        col = self.df.columns.tolist()[0]
        if get_colormap(self.cmap).N < 256 or self.df.dtypes[col] == object:
            cc_list = self.df[col].value_counts().index.tolist()  # sorted by value counts
            self.df[col] = self.df[col].map({v: cc_list.index(v) for v in cc_list})
            for v in cc_list:
                color = get_colormap(self.cmap)(cc_list.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            self.color_dict = {v: get_colormap(self.cmap)(v) for v in self.df[col].values}
        self.colors = None

    def _check_colors(self, colors):
        if isinstance(colors, str):
            colors = {label: colors for label in self.df.iloc[:, 0].unique()}
        if isinstance(colors, list):
            assert len(colors) == self.df.iloc[:, 0].nunique()
            colors = {label: color for label, color in zip(self.df.iloc[:, 0].unique(), colors)}
        if not isinstance(colors, dict):
            raise TypeError("colors must be a dict!")
        if len(colors) >= self.df.iloc[:, 0].nunique():
            self.colors = colors
        else:
            raise TypeError("The length of `colors` is not consistent with the shape of the input data")

    def _calculate_cmap(self):
        self.color_dict = self.colors
        col = self.df.columns.tolist()[0]
        cc_list = list(self.color_dict.keys())  # column values
        self.df[col] = self.df[col].map({v: cc_list.index(v) for v in cc_list})
        self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])
        self.plot_kws.setdefault('vmax', get_colormap(self.cmap).N-1)
        self.plot_kws.setdefault('vmin', 0)

    def _type_specific_params(self):
        pass

    def reorder(self, idx):  # Before plotting, df needs to be reordered according to the new clustered order.
        """
        idx: list
        """
        # n_overlap = len(set(self.df.index.tolist()) & set(idx))
        # if n_overlap == 0:
        #     raise ValueError("The input idx is not consistent with the df.index")
        # else:
        self.plot_data = self.df.reindex(idx)  #
        self.plot_data.fillna(np.nan, inplace=True)
        self.nrows = self.plot_data.shape[0]
        # self._set_default_plot_kws(self.plot_kws)

    def get_label_width(self):
        return self.ax.yaxis.label.get_window_extent(renderer=self.ax.figure.canvas.get_renderer()).width  #

    def get_ticklabel_width(self):
        yticklabels = self.ax.yaxis.get_ticklabels()
        if len(yticklabels) == 0:
            return 0
        else:
            return max([label.get_window_extent(renderer=self.ax.figure.canvas.get_renderer()).width
                        for label in self.ax.yaxis.get_ticklabels()])

    def get_max_label_width(self):
        return max([self.get_label_width(), self.get_ticklabel_width()])
# =============================================================================
class anno_simple(AnnotationBase):
    """
    Annotate simple annotation, categorical or continuous variables.
    """

    def __init__(self, df=None, cmap='auto', colors=None, add_text=False,
                 majority=True, text_kws=None, height=None, legend=True,
                 legend_kws=None, **plot_kws):
        self.add_text = add_text
        self.majority = majority
        self.text_kws = text_kws if not text_kws is None else {}
        self.plot_kws = plot_kws
        # print(self.plot_kws)
        legend_kws = legend_kws if not legend_kws is None else {}
        if 'vmax' in plot_kws:
            legend_kws.setdefault('vmax', plot_kws.get('vmax'))
        if 'vmin' in plot_kws:
            legend_kws.setdefault('vmin', plot_kws.get('vmin'))
        super().__init__(df=df, cmap=cmap, colors=colors,
                         height=height, legend=legend, legend_kws=legend_kws, **plot_kws)

    def _set_default_plot_kws(self, plot_kws):
        self.plot_kws = {} if plot_kws is None else plot_kws
        self.plot_kws.setdefault('zorder', 10)
        self.text_kws.setdefault('zorder', 16)
        self.text_kws.setdefault('ha', 'center')
        self.text_kws.setdefault('va', 'center')

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.color_dict = {}
        col = self.df.columns.tolist()[0]
        if get_colormap(self.cmap).N < 256:
            cc_list = self.df[col].value_counts().index.tolist()  # sorted by value counts
            for v in cc_list:
                color = get_colormap(self.cmap)(cc_list.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            cc_list = None
            self.color_dict = {v: get_colormap(self.cmap)(v) for v in self.df[col].values}
        self.cc_list = cc_list
        self.colors = None

    def _calculate_cmap(self):
        self.color_dict = self.colors
        col = self.df.columns.tolist()[0]
        cc_list = list(self.color_dict.keys())  # column values
        self.cc_list = cc_list
        self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])

    def plot(self, ax=None, axis=1, subplot_spec=None, label_kws={},
             ticklabels_kws={}):  # add self.gs,self.fig,self.ax,self.axes
        if hasattr(self.cmap,'N'):
            vmax=self.cmap.N-1
        elif type(self.cmap)==str:
            vmax=get_colormap(self.cmap).N-1
        else:
            vmax=len(self.color_dict)-1
        self.plot_kws.setdefault('vmax',vmax)  # get_colormap(self.cmap).N
        self.plot_kws.setdefault('vmin', 0)
        if self.cc_list:
            mat = self.plot_data.iloc[:, 0].map({v: self.cc_list.index(v) for v in self.cc_list}).values
        else:
            mat = self.plot_data.values
        matrix = mat.reshape(1, -1) if axis == 1 else mat.reshape(-1, 1)
        # print(matrix)
        xlabel = None if axis == 1 else self.label
        ylabel = self.label if axis == 1 else None

        # ax1 = heatmap(matrix, cmap=self.cmap, cbar=False, ax=ax, xlabel=xlabel, ylabel=ylabel,
        #               xticklabels=False, yticklabels=False, **self.plot_kws)
        ax1 = plot_heatmap(matrix, cmap=self.cmap, ax=ax, xticklabels=False, yticklabels=False, **self.plot_kws)
        # if axis==0:
        #     ax1.invert_yaxis()
        ax.tick_params(axis='both', which='both',
                       left=False, right=False, top=False, bottom=False,
                       labeltop=False, labelbottom=False, labelleft=False, labelright=False)
        if self.add_text:
            if axis == 0:
                self.text_kws.setdefault('rotation', 90)
            labels, ticks = cluster_labels(self.plot_data.iloc[:, 0].values, np.arange(0.5, self.nrows, 1),
                                           self.majority)
            n = len(ticks)
            if axis == 1:
                x = ticks
                y = [0.5] * n
            else:
                y = ticks
                x = [0.5] * n
            s = ax.get_window_extent().height if axis == 1 else ax.get_window_extent().width
            self.text_kws.setdefault('fontsize', 72 * s * 0.8 / ax.figure.dpi)
            # fontsize = self.text_kws.pop('fontsize', 72 * s * 0.8 / ax.figure.dpi)
            color = self.text_kws.pop('color', None)
            for x0, y0, t in zip(x, y, labels):
                # print(t,self.color_dict)
                lum = _calculate_luminance(self.color_dict[t])
                if color is None:
                    text_color = "black" if lum > 0.408 else "white"
                else:
                    text_color = color
                # print(t,self.color_dict,text_color,color)
                self.text_kws.setdefault('color', text_color)
                ax.text(x0, y0, t, **self.text_kws)
        self.ax = ax
        self.fig = self.ax.figure
        return self.ax
# =============================================================================
class anno_label(AnnotationBase):
    """
    Add label and text annotations.

    Parameters
    ----------
    merge: bool
        whether to merge the same clusters into one and label only once.
    extend: bool
        whether to distribute all the labels extend to the all axis, figure or ax or False.
    frac: float
        fraction of the armA and armB.
    majority: bool
        If there are multiple group for one label, whether to annotate the label in the largest group. [True]
    adjust_color: bool
        When the luminance of the color is too high, use black color replace the original color. [True]
    luminance: float
        luminance values [0-1], used together with adjust_color, when the calculated luminance > luminance,
        the color will be replaced with black. [0.5]
    relpos: tuple
        relpos passed to arrowprops in plt.annotate, tuple (x,y) means the arrow start point position relative to the
         label. default is (0, 0) if self.side == 'top' else (0, 1) for columns labels, (1, 1) if self.side == 'left'
         else (0, 0) for rows labels.
    plot_kws: dict
        passed to plt.annotate, including annotation_clip, arrowprops and matplotlib.text.Text,
        more information about arrowprops could be found in
        matplotlib.patches.FancyArrowPatch

    Returns
    ----------
    Class AnnotationBase.

    """

    def __init__(self, df=None, cmap='auto', colors=None, merge=False, extend=False, frac=0.2,
                 majority=True, adjust_color=True, luminance=0.5,height=None, legend=False, legend_kws=None,
                 relpos=None, **plot_kws):
        super().__init__(df=df, cmap=cmap, colors=colors,
                         height=height, legend=legend, legend_kws=legend_kws, **plot_kws)
        self.merge = merge
        self.majority = majority
        self.adjust_color=adjust_color
        self.luminance = luminance
        self.extend = extend
        self.frac = frac
        self.relpos = relpos
        self.annotated_texts = []

    def _height(self, height):
        return 4 if height is None else height

    def set_side(self, side):
        self.side = side

    def set_plot_kws(self, axis):
        shrink = 1  # 1 * mm2inch * 72  # 1mm -> points
        if axis == 1:  # columns
            relpos = (0, 0) if self.side == 'top' else (0, 1)  # position to anchor, x: left -> right, y: down -> top
            rotation = 90 if self.side == 'top' else -90
            ha = 'left'
            va = 'center'
        else:
            relpos = (1, 1) if self.side == 'left' else (0, 0)  # (1, 1) if self.side == 'left' else (0, 0)
            rotation = 0
            ha = 'right' if self.side == 'left' else 'left'
            va = 'center'
        # relpos: The exact starting point position of the arrow is defined by relpos. It's a tuple of relative
        # coordinates of the text box, where (0, 0) is the lower left corner and (1, 1) is the upper right corner.
        # Values <0 and >1 are supported and specify points outside the text box. By default (0.5, 0.5) the starting
        # point is centered in the text box.
        self.plot_kws.setdefault('rotation', rotation)
        self.plot_kws.setdefault('ha', ha)
        self.plot_kws.setdefault('va', va)
        rp=relpos if self.relpos is None else self.relpos
        arrowprops = dict(arrowstyle="-", color="black",
                          shrinkA=shrink, shrinkB=shrink, relpos=rp,
                          patchA=None, patchB=None, connectionstyle=None)
        # arrow: ->, from text to point.
        # self.plot_kws.setdefault('transform_rotates_text', False)
        self.plot_kws.setdefault('arrowprops', arrowprops)
        self.plot_kws.setdefault('rotation_mode', 'anchor')

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.color_dict = {}
        col = self.df.columns.tolist()[0]
        if get_colormap(self.cmap).N < 256 or self.df.dtypes[col] == object:
            cc_list = self.df[col].value_counts().index.tolist()  # sorted by value counts
            for v in cc_list:
                color = get_colormap(self.cmap)(cc_list.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            self.color_dict = {v: get_colormap(self.cmap)(v) for v in self.df[col].values}
        self.colors = None

    def _calculate_cmap(self):
        self.color_dict = self.colors
        col = self.df.columns.tolist()[0]
        cc_list = list(self.color_dict.keys())  # column values
        self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])

    def plot(self, ax=None, axis=1, subplot_spec=None, label_kws={},
             ticklabels_kws={}):  # add self.gs,self.fig,self.ax,self.axes
        self.axis = axis
        ax_index = ax.figure.axes.index(ax)
        ax_n = len(ax.figure.axes)
        i = ax_index / ax_n
        if self.side is None:
            if axis == 1 and i <= 0.5:
                side = 'top'
            elif axis == 1:
                side = 'bottom'
            elif axis == 0 and i <= 0.5:
                side = 'left'
            else:
                side = 'right'
            self.side = side
        self.set_plot_kws(axis)
        if self.merge:  # merge the adjacent ticklabels with the same text to one, return labels and mean x coordinates.
            labels, ticks = cluster_labels(self.plot_data.iloc[:, 0].values, np.arange(0.5, self.nrows, 1),
                                           self.majority)
        else:
            labels = self.plot_data.iloc[:, 0].values
            ticks = np.arange(0.5, self.nrows, 1)
        # labels are the merged labels, ticks are the merged mean x coordinates.

        n = len(ticks)
        text_height = self.height * mm2inch * ax.figure.dpi  # convert height (mm) to inch and to pixels.
        # print(ax.figure.dpi,text_height)
        text_y = text_height
        if self.side == 'bottom' or self.side == 'left':
            # when axis=0, there is an invertion for the x axes.
            text_y = -1 * text_y
            # bottom, y coordinate for text should be smaller than arrow start position (offset pixels)
            # left, x coordinate for text should be smaller than arrow start pos.
        # x,y are the coordinates where arrow starting from and x1,y1 is the end coordinates (text).
        if axis == 1:
            ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
            x = ticks  # coordinate for the labels (text).
            y = [0] * n if self.side == 'top' else [1] * n  # position for line on axes
            if self.extend:
                extend_pos = np.linspace(0, 1, n + 1)
                x1 = [(extend_pos[i] + extend_pos[i - 1]) / 2 for i in range(1, n + 1)]
                y1 = [1 + text_y / ax.figure.get_window_extent().height] * n if self.side == 'top' else [
                                                                                                            text_y / ax.figure.get_window_extent().height] * n
            else:
                x1 = [0] * n
                y1 = [text_y] * n
        else:
            ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
            y = ticks
            x = [1] * n if self.side == 'left' else [0] * n
            if self.extend:
                extend_pos = np.linspace(0, 1, n + 1)
                y1 = [(extend_pos[i] + extend_pos[i - 1]) / 2 for i in range(1, n + 1)]
                x1 = [1 + text_y / ax.figure.get_window_extent().width] * n if self.side == 'right' else [
                                                                                                             0 + text_y / ax.figure.get_window_extent().width] * n
            else:
                y1 = [0] * n
                x1 = [text_y] * n
        # angleA is the angle for the data point (clockwise), B is for text.
        # https://matplotlib.org/stable/gallery/userdemo/connectionstyle_demo.html
        xycoords = ax.get_xaxis_transform() if axis == 1 else ax.get_yaxis_transform()
        # get_xaxis_transform: x is data coordinates,y is between [0,1]
        if self.extend:
            text_xycoords = ax.transAxes  # if self.extend=='ax' else ax.figure.transFigure #ax.transAxes,
        else:
            text_xycoords = 'offset pixels'
        if self.plot_kws['arrowprops']['connectionstyle'] is None:
            arm_height = text_height * self.frac
            rad = 2  # arm_height / 10
            if axis == 1 and self.side == 'top':
                angleA, angleB = (self.plot_kws['rotation'] - 180, 90)
            elif axis == 1 and self.side == 'bottom':
                angleA, angleB = (180 + self.plot_kws['rotation'], -90)
            elif axis == 0 and self.side == 'left':
                angleA, angleB = (self.plot_kws['rotation'], -180)
            else:
                angleA, angleB = (self.plot_kws['rotation'] - 180, 0)
            connectionstyle = f"arc,angleA={angleA},angleB={angleB},armA={arm_height},armB={arm_height},rad={rad}"
            self.plot_kws['arrowprops']['connectionstyle'] = connectionstyle
        hs = []
        ws = []
        # import pdb;
        # pdb.set_trace()
        for t, x_0, y_0, x_1, y_1 in zip(labels, x, y, x1, y1):
            if pd.isna(t):
                continue
            color = self.color_dict[t]
            if self.adjust_color:
                lum = _calculate_luminance(color)
                if lum > self.luminance:
                    color = 'black'
            self.plot_kws['arrowprops']['color'] = color
            annotated_text = ax.annotate(text=t, xy=(x_0, y_0), xytext=(x_1, y_1), xycoords=xycoords,
                                         textcoords=text_xycoords,
                                         color=color,
                                         **self.plot_kws)  # unit for shrinkA is point (1 point = 1/72 inches)
            self.annotated_texts.append(annotated_text)
        # _draw_figure(ax.figure)
        # print('anno_label:',self.label_width,ax.get_window_extent().height,ax.get_window_extent().width)
        ax.tick_params(axis='both', which='both',
                       left=False, right=False, top=False, bottom=False,
                       labeltop=False, labelbottom=False, labelleft=False, labelright=False)
        ax.set_axis_off()
        self.ax = ax
        self.fig = self.ax.figure
        return self.ax

    def get_ticklabel_width(self):
        hs = [text.get_window_extent().width for text in self.annotated_texts]
        if len(hs) == 0:
            return 0
        else:
            return max(hs)
# =============================================================================
class anno_boxplot(AnnotationBase):
    """
        annotate boxplots, all arguments are included in AnnotationBase,
        plot_kws for anno_boxplot include showfliers, edgecolor, grid, medianlinecolor
        width and other arguments passed to plt.boxplot.

    Parameters
    ----------
    """

    def _height(self, height):
        return 10 if height is None else height

    def _set_default_plot_kws(self, plot_kws):
        self.plot_kws = plot_kws if plot_kws is not None else {}
        self.plot_kws.setdefault('showfliers', False)
        self.plot_kws.setdefault('edgecolor', 'black')
        self.plot_kws.setdefault('medianlinecolor', 'black')
        self.plot_kws.setdefault('grid', True)
        self.plot_kws.setdefault('zorder', 10)
        self.plot_kws.setdefault('widths', 0.5)

    def _check_cmap(self, cmap):
        if cmap == 'auto':
            self.cmap = 'jet'
        elif type(cmap) == str:
            self.cmap = cmap
        else:
            raise TypeError("cmap for boxplot should be a string")

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.colors = None

    def _check_colors(self, colors):
        if type(colors) == str:
            self.colors = colors
        else:
            raise TypeError(
                "Boxplot only support one string as colors now, if more colors are wanted, cmap can be specified.")

    def _calculate_cmap(self):
        self.set_legend(False)
        self.cmap = None

    def _type_specific_params(self):
        gap = self.df.max().max() - self.df.min().min()
        self.ylim = [self.df.min().min() - 0.02 * gap, self.df.max().max() + 0.02 * gap]

    def plot(self, ax=None, axis=1, subplot_spec=None, label_kws={},
             ticklabels_kws={}):  # add self.gs,self.fig,self.ax,self.axes
        fig = ax.figure
        if self.colors is None:  # calculate colors based on cmap
            colors = [get_colormap(self.cmap)(self.plot_data.loc[sampleID].mean()) for sampleID in
                      self.plot_data.index.values]
        else:
            colors = [self.colors] * self.plot_data.shape[0]  # self.colors is a string
        # print(self.plot_kws)
        plot_kws = self.plot_kws.copy()
        edgecolor = plot_kws.pop('edgecolor')
        mlinecolor = plot_kws.pop('medianlinecolor')
        grid = plot_kws.pop('grid')
        # bp = ax.boxplot(self.plot_data.T.values, patch_artist=True,**self.plot_kws)
        if axis == 1:
            vert = True
        else:
            vert = False
        if axis == 1:
            ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
        else:
            ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
        # bp = self.plot_data.T.boxplot(ax=ax, patch_artist=True,vert=vert,return_type='dict',**self.plot_kws)
        bp = ax.boxplot(x=self.plot_data.T.values, positions=np.arange(0.5, self.nrows, 1), patch_artist=True,
                        vert=vert, **plot_kws)
        if grid:
            ax.grid(linestyle='--', zorder=-10)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
            box.set_edgecolor(edgecolor)
        for median_line in bp['medians']:
            median_line.set_color(mlinecolor)
        if axis == 1:
            ax.set_xlim(0, self.nrows)
            ax.tick_params(axis='both', which='both',
                           top=False, bottom=False, labeltop=False, labelbottom=False)
        else:
            ax.set_ylim(0, self.nrows)
            ax.tick_params(axis='both', which='both',
                           left=False, right=False, labelleft=False, labelright=False)
            ax.invert_xaxis()
        self.fig = fig
        self.ax = ax
        return self.ax
# =============================================================================
class anno_barplot(anno_boxplot):
    """
    Annotate barplot, all arguments are included in AnnotationBase,
        plot_kws for anno_boxplot include edgecolor, grid,
        and other arguments passed to plt.barplot.
    """

    def _set_default_plot_kws(self, plot_kws):
        self.plot_kws = plot_kws if plot_kws is not None else {}
        self.plot_kws.setdefault('edgecolor', 'black')
        self.plot_kws.setdefault('grid', True)
        self.plot_kws.setdefault('zorder', 10)
        # self.plot_kws.setdefault('width', 0.7)
        self.plot_kws.setdefault('align', 'center')

    def _check_cmap(self, cmap):
        if cmap == 'auto':
            if self.ncols == 1:
                self.cmap = 'jet'
            else:
                self.cmap = 'Set1'
        # print(cmap,self.cmap)
        else:
            self.cmap = cmap
        if self.ncols >= 2 and get_colormap(self.cmap).N >= 256:
            raise TypeError("cmap for stacked barplot should not be continuous, you should try: Set1, Dark2 and so on.")

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        col_list = self.df.columns.tolist()
        self.color_dict = {}
        if len(col_list) >= 2:  # more than two columns, colored by columns names
            self.colors = [get_colormap(self.cmap)(col_list.index(v)) for v in self.df.columns]
            for v, color in zip(col_list, self.colors):
                self.color_dict[v] = color
        else:  # only one column, colored by cols[0] values (float)
            # vmax, vmin = np.nanmax(self.df[col_list[0]].values), np.nanmin(self.df[col_list[0]].values)
            # delta = vmax - vmin
            # values = self.df[col_list[0]].fillna(np.nan).unique()
            self.cmap,normalize=define_cmap(self.df[col_list[0]].fillna(np.nan).values,
                                            vmin=None, vmax=None, cmap=self.cmap,
                                            center=None, robust=False,
                          na_col='white')
            # self.colors = {v: matplotlib.colors.rgb2hex(get_colormap(self.cmap)((v - vmin) / delta)) for v in values}
            self.colors = lambda v:matplotlib.colors.rgb2hex(self.cmap(normalize(v))) #a function
            self.color_dict = None

    def _check_colors(self, colors):
        if not isinstance(colors, (list, str)):
            raise TypeError("colors must be list of string for barplot if provided !")
        if type(colors) == str:
            self.colors = [colors] * self.nrows
        elif self.ncols > 1 and type(colors) == list and len(colors) == self.ncols:
            self.colors = colors
        # elif self.ncols == 1 and type(colors)==list and len(colors)!=self.nrows:
        #     raise ValueError("If there is only one column in df,colors must have the same length as df.index for barplot!")
        else:
            raise ValueError(
                "the length of colors is not correct, If there are more than one column in df,colors must have the same length as df.columns for barplot!")

    def _calculate_cmap(self):
        # cc_list = list(self.color_dict.keys())  # column values
        # self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])
        self.cmap = None
        self.set_legend(False)

    def _type_specific_params(self):
        if self.ncols > 1:
            Max = self.df.max().max()
            Min = self.df.min().min()
            self.stacked = True
        else:
            Max = self.df.values.max()
            Min = self.df.values.min()
            self.stacked = False
        gap = Max - Min
        self.ylim = [Min - 0.02 * gap, Max + 0.02 * gap]

    def plot(self, ax=None, axis=1, subplot_spec=None, label_kws={},
             ticklabels_kws={}):  # add self.gs,self.fig,self.ax,self.axes
        if ax is None:
            ax = plt.gca()
        fig = ax.figure
        plot_kws = self.plot_kws.copy()
        grid = plot_kws.pop('grid', False)
        if grid:
            ax.grid(linestyle='--', zorder=-10)
        # bar_ct = ax.bar(x=list(range(1, self.nrows + 1,1)),
        #                 height=self.plot_data.values,**self.plot_kws)
        if type(self.colors) == list:
            colors = self.colors
        else: #dict
            #bad_value_color = matplotlib.colors.rgb2hex(get_colormap(self.cmap).get_bad())
            # colors = [[self.colors.get(v, bad_value_color) for v in self.plot_data.iloc[:, 0].values]]
            colors = [[self.colors(v) for v in self.plot_data.iloc[:, 0].values]]
        base_coordinates = [0] * self.plot_data.shape[0]
        for col, color in zip(self.plot_data.columns, colors):
            if axis == 1:
                ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
                ax.bar(x=np.arange(0.5, self.nrows, 1), height=self.plot_data[col].values,
                       bottom=base_coordinates, color=color, **plot_kws)
            else:
                ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
                ax.barh(y=np.arange(0.5, self.nrows, 1), width=self.plot_data[col].values,
                        left=base_coordinates, color=color, **plot_kws)
            base_coordinates = self.plot_data[col].values + base_coordinates
        # for patch in ax.patches:
        #     patch.set_edgecolor(edgecolor)
        if axis == 0:
            ax.tick_params(axis='both', which='both',
                           left=False, right=False, labelleft=False, labelright=False)
            ax.invert_xaxis()
        else:
            ax.tick_params(axis='both', which='both',
                           top=False, bottom=False, labeltop=False, labelbottom=False)
        self.fig = fig
        self.ax = ax
        return self.ax
# =============================================================================
class anno_scatterplot(anno_barplot):
    """
    Annotate scatterplot, all arguments are included in AnnotationBase,
        plot_kws for anno_boxplot include linewidths, grid, edgecolors
        and other arguments passed to plt.scatter.
    """

    def _check_df(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame(name=df.name)
        if isinstance(df, pd.DataFrame) and df.shape[1] != 1:
            raise ValueError("df must have only 1 column for scatterplot")
        elif isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise TypeError("df must be a pandas DataFrame or Series.")

    def _set_default_plot_kws(self, plot_kws):
        self.plot_kws = plot_kws if plot_kws is not None else {}
        self.plot_kws.setdefault('grid', True)
        self.plot_kws.setdefault('zorder', 10)
        self.plot_kws.setdefault('linewidths', 0)
        self.plot_kws.setdefault('edgecolors', 'black')

    def _check_cmap(self, cmap):
        self.cmap = 'jet'
        if cmap == 'auto':
            pass
        elif type(cmap) == str:
            self.cmap = cmap
        else:
            raise TypeError("cmap for scatterplot should be a string")

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.colors = None

    def _check_colors(self, colors):
        if not isinstance(colors, str):
            raise TypeError("colors must be string for scatterplot, if more colors are neded, please try cmap!")
        self.colors = colors

    def _calculate_cmap(self):
        self.cmap = None
        self.set_legend(False)

    def _type_specific_params(self):
        Max = self.df.values.max()
        Min = self.df.values.min()
        self.gap = Max - Min
        self.ylim = [Min - 0.2 * self.gap, Max + 0.2 * self.gap]

    def plot(self, ax=None, axis=1, subplot_spec=None, label_kws={},
             ticklabels_kws={}):  # add self.gs,self.fig,self.ax,self.axes
        if ax is None:
            ax = plt.gca()
        fig = ax.figure
        plot_kws = self.plot_kws.copy()
        grid = plot_kws.pop('grid', False)
        if grid:
            ax.grid(linestyle='--', zorder=-10)
        values = self.plot_data.iloc[:, 0].values
        if self.colors is None:
            colors = values
        else:  # self.colors is a string
            colors = [self.colors] * self.plot_data.shape[0]
        if axis == 1:
            spu = ax.get_window_extent().height * 72 / self.gap / fig.dpi  # size per unit
        else:
            spu = ax.get_window_extent().width * 72 / self.gap / fig.dpi  # size per unit
        self.s = (values - values.min() + self.gap * 0.1) * spu  # fontsize
        if axis == 1:
            ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
            x = np.arange(0.5, self.nrows, 1)
            y = values
        else:
            ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
            y = np.arange(0.5, self.nrows, 1)
            x = values
        c = self.plot_kws.get('c', colors)
        s = self.plot_kws.get('s', self.s)
        scatter_ax = ax.scatter(x=x, y=y, c=c, s=s, cmap=self.cmap, **plot_kws)
        if axis == 0:
            ax.tick_params(axis='both', which='both',
                           left=False, right=False, labelleft=False, labelright=False)
            ax.invert_xaxis()
        else:
            ax.tick_params(axis='both', which='both',
                           top=False, bottom=False, labeltop=False, labelbottom=False)
        self.fig = fig
        self.ax = ax
        return self.ax
# =============================================================================
class HeatmapAnnotation():
    """
    Generate and plot heatmap annotations.

    Parameters
    ----------
    self : Class
        HeatmapAnnotation
    df :  dataframe
        a pandas dataframe, each column will be converted to one anno_simple class.
    axis : int
        1 for columns annotation, 0 for rows annotations.
    cmap : str
        colormap, such as Set1, Dark2, bwr, Reds, jet, hsv, rainbow and so on. Please see
        https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html for more information, or run
        matplotlib.pyplot.colormaps() to see all availabel cmap.
        default cmap is 'auto', it would be determined based on the dtype for each columns in df.
        If df is None, then there is no need to specify cmap, cmap and colors will only be used when
        df is provided.
        If cmap is a string, then all columns in the df would have the same cmap, cmap can also be
        a dict, keys are the column names from df, values should be cmap (matplotlib.pyplot.colormaps()).
    colors : dict
        a dict, keys are the column names of df, values are list, dict or string passed to
        AnnotationBase.__subclasses__(), including anno_simple, anno_boxplot,anno_label and anno_scatter.
        colors must have the same length as the df.columns, if colors is not provided (default), else,
        colors would be calculated based on the given cmap.
        If colors is given, then the cmap would be invalid.
    label_side : str
        top or bottom when axis=1 (columns annotation), left or right when axis=0 (rows annotations).
    label_kws : dict
        kws passed to the labels of the annotation labels (would be df.columns if df is given).
        such as alpha, color, fontsize, fontstyle, ha (horizontalalignment),
        va (verticalalignment), rotation, rotation_mode, visible, rasterized and so on.
        For more information, see plt.gca().yaxis.label.properties() or ax.yaxis.label.properties()
    ticklabels_kws : dict
        label_kws is for the label of annotation, ticklabels_kws is for the label (text) in anno_label,
        such as axis, which, direction, length, width,
        color, pad, labelsize, labelcolor, colors, zorder, bottom, top, left, right, labelbottom, labeltop,
        labelleft, labelright, labelrotation, grid_color, grid_linestyle and so on.
        For more information,see ?matplotlib.axes.Axes.tick_params
    plot_kws : dict
        kws passed to annotation functions, such as anno_simple, anno_label et.al.
    plot : bool
        whether to plot, when the annotation are included in clustermap, plot would be
        set to False automotially.
    legend : bool
        True or False, or dict (when df is no None), when legend is dict, keys are the
        columns of df.
    legend_side : str
        right or left
    legend_gap : float
        the vertical gap between two legends, default is 2 [mm]
    legend_width: float
        width of the legend, default is 4.5[mm]
    legend_hpad: float
        Horizonal space between heatmap and legend, default is 2 [mm].
    legend_vpad: float
        Vertical space between top of ax and legend, default is 2 [mm].
    orientation: str
        up or down, when axis=1
        left or right, when axis=0;
        When anno_label shows up in annotation, the orientation would be automatically be assigned according
        to the position of anno_label.
    wgap: float or int
        optional,  the space used to calculate wspace, default is [0.1] (mm),
        control the vertical gap between two annotations.
    hgap: float or int
        optional,  the space used to calculate hspace, default is [0.1] (mm),
        control the horizontal gap between two annotations.
    plot_legend : bool
        whether to plot legends.
    args : name-value pair
        key is the annotation label (name), values can be a pandas dataframe,
        series, or annotation such as
        anno_simple, anno_boxplot, anno_scatter, anno_label, or anno_barplot.

    Returns
    -------
    Class HeatmapAnnotation.

    """

    def __init__(self, df=None, axis=1, cmap='auto', colors=None, label_side=None, label_kws=None,
                 ticklabels_kws=None, plot_kws=None, plot=False, legend=True, legend_side='right',
                 legend_gap=5, legend_width=4.5, legend_hpad=2, legend_vpad=5, orientation=None,
                 wgap=0.1,hgap=0.1,plot_legend=True, rasterized=False, verbose=1, **args):
        if df is None and len(args) == 0:
            raise ValueError("Please specify either df or other args")
        if not df is None and len(args) > 0:
            raise ValueError("df and Name-value pairs can only be given one, not both.")
        if not df is None:
            self._check_df(df)
        else:
            self.df = None
        self.axis = axis
        self.verbose = verbose
        self.label_side = label_side
        self.plot_kws = plot_kws if not plot_kws is None else {}
        self.args = args
        self._check_legend(legend)
        self.legend_side = legend_side
        self.legend_gap = legend_gap
        self.wgap=wgap
        self.hgap = hgap
        self.legend_width = legend_width
        self.legend_hpad = legend_hpad
        self.legend_vpad = legend_vpad
        self.plot_legend = plot_legend
        self.rasterized = rasterized
        self.orientation = orientation
        self.plot = plot
        self.mapping_dict={'up':'top','down':'bottom','left':'left','right':'right'}
        if colors is None:
            self._check_cmap(cmap)
            self.colors = None
        else:
            self._check_colors(colors)
        self._process_data()
        self._heights()
        self._nrows()
        self.label_kws,self.ticklabels_kws = label_kws, ticklabels_kws
        if self.plot:
            self.plot_annotations()

    def _check_df(self, df):
        if type(df) == list or isinstance(df, np.ndarray):
            df = pd.Series(df).to_frame(name='df')
        elif isinstance(df, pd.Series):
            name = df.name if not df.name is None else 'df'
            df = df.to_frame(name=name)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("data type of df could not be recognized, should be a dataframe")
        self.df = df

    def _check_legend(self, legend):
        if type(legend) == bool:
            if not self.df is None:
                self.legend = {col: legend for col in self.df.columns}
            if len(self.args) > 0:
                # self.legend = collections.defaultdict(lambda: legend)
                self.legend = {arg: legend for arg in self.args}
        elif type(legend) == dict:
            self.legend = legend
            for arg in self.args:
                if arg not in self.legend:
                    self.legend[arg] = False
        else:
            raise TypeError("Unknow data type for legend!")

    def _check_cmap(self, cmap):
        if self.df is None:
            return
        self.cmap = {}
        if cmap == 'auto':
            for col in self.df.columns:
                if self.df.dtypes[col] == object:
                    if self.df[col].nunique() <= 10:
                        self.cmap[col] = 'Set1'
                    elif self.df[col].nunique() <= 20:
                        self.cmap[col] = 'tab20'
                    else:
                        self.cmap[col] = 'cmap50'
                elif self.df.dtypes[col] == float or self.df.dtypes[col] == int:
                    self.cmap[col] = 'jet'
                else:
                    raise TypeError("Can not assign cmap for column %s, please specify cmap" % col)
        elif type(cmap) == str:
            self.cmap = {col: cmap for col in self.df.columns}
        elif type(cmap) == list:
            if len(cmap) == 1:
                cmap = cmap * len(self.df.shape[1])
            if len(cmap) != self.df.shape[1]:
                raise ValueError("kind must have the same lengt with the number of columns with df")
            self.cmap = {col: c for col, c in zip(self.df.columns, cmap)}
        elif type(cmap) == dict:
            if len(cmap) != self.df.shape[1]:
                raise ValueError("kind must have the same length with number of columns with df")
            self.cmap = cmap
        else:
            raise TypeError("Unknow data type for cmap!")

    def _check_colors(self, colors):
        if self.df is None:
            return
        self.colors = {}
        if not isinstance(colors, dict):
            raise TypeError("colors must be a dict!")
        if len(colors) != self.df.shape[1]:
            raise ValueError("colors must have the same length as the df.columns!")
        self.colors = colors

    def _process_data(self):  # add self.annotations,self.names,self.labels
        self.annotations = []
        self.plot_kws["rasterized"] = self.rasterized
        if not self.df is None:
            for col in self.df.columns:
                plot_kws = self.plot_kws.copy()
                if self.colors is None:
                    plot_kws.setdefault("cmap", self.cmap[col])  #
                else:
                    plot_kws.setdefault("colors", self.colors[col])
                anno1 = anno_simple(self.df[col], legend=self.legend.get(col, False), **plot_kws)
                anno1.set_label(col)
                self.annotations.append(anno1)
        elif len(self.args) > 0:
            # print(self.args)
            self.labels = []
            for arg in self.args:
                # print(arg)
                ann = self.args[arg]
                if type(ann) == list or isinstance(ann, np.ndarray):
                    ann = pd.Series(ann).to_frame(name=arg)
                elif isinstance(ann, pd.Series):
                    ann = ann.to_frame(name=arg)
                if isinstance(ann, pd.DataFrame):
                    if ann.shape[1] > 1:
                        for col in ann.columns:
                            anno1 = anno_simple(ann[col], legend=self.legend.get(col, False), **self.plot_kws)
                            anno1.set_label(col)
                            self.annotations.append(anno1)
                    else:
                        anno1 = anno_simple(ann, **self.plot_kws)
                        anno1.set_label(arg)
                        anno1.set_legend(self.legend.get(arg, False))
                        self.annotations.append(anno1)
                if hasattr(ann, 'set_label') and AnnotationBase.__subclasscheck__(type(ann)):
                    self.annotations.append(ann)
                    ann.set_label(arg)
                    ann.set_legend(self.legend.get(arg, False))
                    if type(ann) == anno_label and self.orientation is None:
                        if self.axis == 1 and len(self.labels) == 0:
                            self.orientation = 'up'
                        elif self.axis == 1:
                            self.orientation = 'down'
                        elif self.axis == 0 and len(self.labels) == 0:
                            self.orientation = 'left'
                        elif self.axis == 0:
                            self.orientation = 'right'
                    if type(ann) == anno_label:
                        ann.set_side(self.mapping_dict[self.orientation])
                self.labels.append(arg)

    def _set_orentation(self,orientation):
        if self.orientation is None:
            self.orientation = orientation

    def _heights(self):
        self.heights = [ann.height for ann in self.annotations]

    def _nrows(self):
        self.nrows = [ann.nrows for ann in self.annotations]

    def _set_label_kws(self, label_kws, ticklabels_kws):
        if self.label_side in ['left', 'right'] and self.axis != 1:
            raise ValueError("For columns annotation, label_side must be left or right!")
        if self.label_side in ['top', 'bottom'] and self.axis != 0:
            raise ValueError("For row annotation, label_side must be top or bottom!")
        if self.orientation is None:
            if self.axis == 1:
                self.orientation = 'up'
            else:  # horizonal
                self.orientation = 'left'
        self.label_kws = {} if label_kws is None else label_kws
        self.ticklabels_kws = {} if ticklabels_kws is None else ticklabels_kws
        self.label_kws.setdefault("rotation_mode", 'anchor')
        if self.label_side is None:
            self.label_side = 'right' if self.axis == 1 else 'top'  # columns annotation, default ylabel is on the right
        ha, va = 'left', 'center'
        if self.orientation == 'left':
            rotation, labelrotation = 90, 90
            ha='right' if self.label_side=='bottom' else 'left'
        elif self.orientation == 'right':
            ha = 'right' if self.label_side=='top' else 'left'
            rotation, labelrotation = -90, -90
        else: #self.orientation == 'up':
            rotation, labelrotation = 0, 0
            ha = 'left' if self.label_side == 'right' else 'right'
        self.label_kws.setdefault('rotation', rotation)
        self.ticklabels_kws.setdefault('labelrotation', labelrotation)
        self.label_kws.setdefault('horizontalalignment', ha)
        self.label_kws.setdefault('verticalalignment', va)

        map_dict = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}
        self.ticklabels_side = map_dict[self.label_side]
        # label_kws: alpha,color,fontfamily,fontname,fontproperties,fontsize,fontstyle,fontweight,label,rasterized,
        # rotation,rotation_mode(default,anchor),visible, zorder,verticalalignment,horizontalalignment

    def set_axes_kws(self):
        if self.axis == 1 and self.label_side == 'left':
            self.ax.yaxis.tick_right()
            for i in range(self.axes.shape[0]):
                self.axes[i, 0].yaxis.set_visible(True)
                self.axes[i, 0].yaxis.label.set_visible(True)
                self.axes[i, 0].tick_params(axis='y', which='both', left=False, labelleft=False, right=False,
                                            labelright=False)
                self.axes[i, 0].set_ylabel(self.annotations[i].label)
                self.axes[i, 0].yaxis.set_label_position(self.label_side)
                self.axes[i, 0].yaxis.label.update(self.label_kws)
                # self.axes[i, -1].yaxis.tick_right()  # ticks
                if type(self.annotations[i]) != anno_simple:
                    self.axes[i, -1].yaxis.set_visible(True)
                    self.axes[i, -1].tick_params(axis='y', which='both', right=True, labelright=True)
                    self.axes[i, -1].yaxis.set_tick_params(**self.ticklabels_kws)
        elif self.axis == 1 and self.label_side == 'right':
            self.ax.yaxis.tick_left()
            for i in range(self.axes.shape[0]):
                self.axes[i, -1].yaxis.set_visible(True)
                self.axes[i, -1].yaxis.label.set_visible(True)
                self.axes[i, -1].tick_params(axis='y', which='both', left=False, labelleft=False, right=False,
                                             labelright=False)
                self.axes[i, -1].set_ylabel(self.annotations[i].label)
                self.axes[i, -1].yaxis.set_label_position(self.label_side)
                self.axes[i, -1].yaxis.label.update(self.label_kws)
                # self.axes[i, 0].yaxis.tick_left()  # ticks
                if type(self.annotations[i]) != anno_simple:
                    self.axes[i, 0].yaxis.set_visible(True)
                    self.axes[i, 0].tick_params(axis='y', which='both', left=True, labelleft=True)
                    self.axes[i, 0].yaxis.set_tick_params(**self.ticklabels_kws)
        elif self.axis == 0 and self.label_side == 'top':
            self.ax.xaxis.tick_bottom()
            for j in range(self.axes.shape[1]):
                self.axes[0, j].xaxis.set_visible(True)
                self.axes[0, j].xaxis.label.set_visible(True)
                self.axes[0, j].tick_params(axis='x', which='both', top=False, labeltop=False, bottom=False,
                                            labelbottom=False)
                self.axes[0, j].set_xlabel(self.annotations[j].label)
                self.axes[0, j].xaxis.set_label_position(self.label_side)
                self.axes[0, j].xaxis.label.update(self.label_kws)
                # self.axes[-1, j].xaxis.tick_bottom()  # ticks
                if type(self.annotations[j]) != anno_simple:
                    self.axes[-1, j].xaxis.set_visible(True)
                    self.axes[-1, j].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
                    self.axes[-1, j].xaxis.set_tick_params(**self.ticklabels_kws)
        elif self.axis == 0 and self.label_side == 'bottom':
            self.ax.xaxis.tick_top()
            for j in range(self.axes.shape[1]):
                self.axes[-1, j].xaxis.set_visible(True)
                self.axes[-1, j].xaxis.label.set_visible(True)
                self.axes[-1, j].tick_params(axis='x', which='both', top=False, labeltop=False, bottom=False,
                                             labelbottom=False)
                self.axes[-1, j].set_xlabel(self.annotations[j].label)
                self.axes[-1, j].xaxis.set_label_position(self.label_side)
                self.axes[-1, j].xaxis.label.update(self.label_kws)
                # self.axes[0, j].xaxis.tick_top()  # ticks
                if type(self.annotations[j]) != anno_simple:
                    self.axes[0, j].xaxis.set_visible(True)
                    self.axes[0, j].tick_params(axis='x', which='both', top=True, labeltop=True)
                    self.axes[0, j].xaxis.set_tick_params(**self.ticklabels_kws)

    def collect_legends(self):
        """
        Collect legends.
        Returns
        -------
        None
        """
        if self.verbose >= 1:
            print("Collecting annotation legends..")
        self.legend_list = []  # handles(dict) / cmap, title, kws
        for annotation in self.annotations:
            if not annotation.legend:
                continue
            legend_kws = annotation.legend_kws.copy()
            # print(annotation.cmap,annotation)
            if (annotation.cmap is None) or \
                (hasattr(annotation.cmap,'N') and annotation.cmap.N < 256) or \
                (type(annotation.cmap) ==str and get_colormap(annotation.cmap).N < 256):
                color_dict = annotation.color_dict
                if color_dict is None:
                    continue
                self.legend_list.append(
                    [annotation.color_dict, annotation.label, legend_kws, len(annotation.color_dict),'color_dict'])
            else:
                if annotation.df.shape[1] == 1:
                    array = annotation.df.iloc[:, 0].values
                else:
                    array = annotation.df.values
                vmax = np.nanmax(array)
                vmin = np.nanmin(array)
                # print(vmax,vmin,annotation)
                legend_kws.setdefault('vmin', round(vmin, 2))
                legend_kws.setdefault('vmax', round(vmax, 2))
                self.legend_list.append([annotation.cmap, annotation.label, legend_kws, 4,'cmap'])
        if len(self.legend_list) > 1:
            self.legend_list = sorted(self.legend_list, key=lambda x: x[3])
        if self.label_side == 'right':
            self.label_max_width = max([ann.get_max_label_width() for ann in self.annotations])
        else:
            self.label_max_width = max([ann.get_ticklabel_width() for ann in self.annotations])
        # self.label_max_height = max([ann.ax.yaxis.label.get_window_extent().height for ann in self.annotations])

    def plot_annotations(self, ax=None, subplot_spec=None, idxs=None,
                         wspace=None, hspace=None):
        """
        Plot annotations

        Parameters
        ----------
        ax : ax
            axes to plot the annotations.
        subplot_spec : ax.figure.add_gridspec
            object from ax.figure.add_gridspec or matplotlib.gridspec.GridSpecFromSubplotSpec.
        idxs : list
            index to reorder df and df of annotation class.
        wspace : float
            if wspace not is None, use wspace, else wspace would be calculated based on gap.
        hspace : float
            if hspace not is None, use hspace, else hspace would be calculated based on gap.

        Returns
        -------
        self.ax
        """
        # print(ax.figure.get_size_inches())
        self._set_label_kws(self.label_kws, self.ticklabels_kws)
        if self.verbose >= 1:
            print("Starting plotting HeatmapAnnotations")
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        if idxs is None:
            idxs = [self.annotations[0].plot_data.index.tolist()]
        if self.axis == 1:
            nrows = len(self.heights)
            ncols = len(idxs)
            height_ratios = self.heights
            width_ratios = [len(idx) for idx in idxs]
            wspace = self.wgap * mm2inch * self.ax.figure.dpi / (
                    self.ax.get_window_extent().width / ncols) if wspace is None else wspace  # 1mm=mm2inch inch
            hspace = self.hgap * mm2inch * self.ax.figure.dpi / (
                    self.ax.get_window_extent().height / nrows) if hspace is None else hspace  # fraction of height
        else:
            nrows = len(idxs)
            ncols = len(self.heights)
            width_ratios = self.heights
            height_ratios = [len(idx) for idx in idxs]
            hspace = self.hgap * mm2inch * self.ax.figure.dpi / (
                    self.ax.get_window_extent().height / nrows) if hspace is None else hspace
            wspace = self.wgap * mm2inch * self.ax.figure.dpi / (
                    self.ax.get_window_extent().width / ncols) if wspace is None else wspace  # The amount of width reserved for space between subplots, expressed as a fraction of the average axis width
        if subplot_spec is None:
            self.gs = self.ax.figure.add_gridspec(nrows, ncols, hspace=hspace, wspace=wspace,
                                                  height_ratios=height_ratios,
                                                  width_ratios=width_ratios)
        else: #this ax is a subplot of another bigger figure.
            self.gs = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows, ncols, hspace=hspace, wspace=wspace,
                                                                  subplot_spec=subplot_spec,
                                                                  height_ratios=height_ratios,
                                                                  width_ratios=width_ratios)
        self.axes = np.empty(shape=(nrows, ncols), dtype=object)
        self.fig = self.ax.figure
        self.ax.set_axis_off()
        # self.ax.margins(x=0, y=0)
        for j, idx in enumerate(idxs):
            for i, ann in enumerate(self.annotations):
                # print(i,j,ann.label)
                # axis=1: left -> right, axis=0: bottom -> top.
                # idx=idx[::-1] if self.axis==0 else idx #1 for cols, 0 for rows.
                ann.reorder(idx)
                gs = self.gs[i, j] if self.axis == 1 else self.gs[j, i]
                sharex = self.axes[0, j] if self.axis == 1 else self.axes[0, i]
                sharey = self.axes[i, 0] if self.axis == 1 else self.axes[j, 0]
                ax1 = self.ax.figure.add_subplot(gs, sharex=sharex, sharey=sharey)
                if self.axis == 1:
                    ax1.set_xlim([0, len(idx)])
                else:
                    ax1.set_ylim([0, len(idx)])
                ann.plot(ax=ax1, axis=self.axis, subplot_spec=gs)
                if self.axis == 1:
                    # ax1.yaxis.set_visible(False)
                    ax1.yaxis.label.set_visible(False)
                    ax1.tick_params(left=False, right=False, labelleft=False, labelright=False)
                    self.ax.spines['top'].set_visible(False)
                    self.ax.spines['bottom'].set_visible(False)
                    self.axes[i, j] = ax1
                    if self.orientation == 'down':
                        ax1.invert_yaxis()

                else:  # horizonal
                    if type(ann) != anno_simple:
                        ax1.invert_yaxis()  # 20230312 fix bug for inversed row order in anno_label.
                    ax1.xaxis.label.set_visible(False)
                    ax1.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
                    self.ax.spines['left'].set_visible(False)
                    self.ax.spines['right'].set_visible(False)
                    self.axes[j, i] = ax1
                    if self.orientation == 'right':
                        ax1.invert_xaxis()

        self.set_axes_kws()
        self.legend_list = None
        if self.plot and self.plot_legend:
            self.plot_legends(ax=self.ax)
        # _draw_figure(self.ax.figure)
        return self.ax

    def show_ticklabels(self, labels, **kwargs):
        ha, va = 'left', 'center'
        if self.axis == 1:
            ax = self.axes[-1, 0] if self.orientation == 'up' else self.axes[0, 0]
            rotation = -45 if self.orientation == 'up' else 45
            ax.xaxis.set_visible(True)
            ax.xaxis.label.set_visible(True)
            if self.orientation == 'up':
                ax.xaxis.set_ticks_position('bottom')
                ax.tick_params(axis='both', which='both', bottom=True, labelbottom=True)
            else:
                ax.xaxis.set_ticks_position('top')
                ax.tick_params(axis='both', which='both', top=True, labeltop=True)
        else:
            ax = self.axes[0, -1] if self.orientation == 'left' else self.axes[0, 0]
            rotation = 0
            ax.yaxis.set_visible(True)
            ax.yaxis.label.set_visible(True)
            if self.orientation == 'left':
                ax.yaxis.set_ticks_position('right')
                ax.tick_params(axis='both', which='both', right=True, labelright=True)
            else:
                ha = 'right'
                ax.yaxis.set_ticks_position('left')
                ax.tick_params(axis='both', which='both', left=True, labelleft=True)
        kwargs.setdefault('rotation', rotation)
        kwargs.setdefault('ha', ha)
        kwargs.setdefault('va', va)
        kwargs.setdefault('rotation_mode', 'anchor')
        if self.axis == 1:
            ax.set_xticklabels(labels, **kwargs)
        else:
            ax.set_yticklabels(labels, **kwargs)

    def plot_legends(self, ax=None):
        """
        Plot legends.
        Parameters
        ----------
        ax : axes for the plot, is ax is None, then ax=plt.figure()

        Returns
        -------
        None
        """
        if self.legend_list is None:
            self.collect_legends()
        if len(self.legend_list) > 0:
            # if the legend is on the right side
            space = self.label_max_width if (self.legend_side == 'right' and self.label_side == 'right') else 0
            legend_hpad = self.legend_hpad * mm2inch * self.ax.figure.dpi  # mm to inch to pixel
            self.legend_axes, self.cbars, self.boundry = \
                plot_legend_list(self.legend_list, ax=ax, space=space + legend_hpad,
                                 legend_side='right', gap=self.legend_gap,
                                 legend_width=self.legend_width, legend_vpad=self.legend_vpad)
# =============================================================================
