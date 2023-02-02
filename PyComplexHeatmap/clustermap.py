# !/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from scipy.cluster import hierarchy
import collections
import warnings
import copy
from .utils import (
    _check_mask,
    _calculate_luminance,
    despine,
    _draw_figure,
    axis_ticklabels_overlap,
    _index_to_label,
    _index_to_ticklabels,
    cluster_labels,
    plot_legend_list
)

mm2inch = 1/25.4

class heatmapPlotter:
    def __init__(self, data=None, vmin=None, vmax=None, cmap='bwr', center=None,
                 robust=True, annot=None, fmt='.2g',
                 annot_kws=None, cbar=True, cbar_kws=None,
                 xlabel=None, ylabel=None,
                 xticklabels=True, yticklabels=True,
                 mask=None, na_col='white'):
        """Initialize the plotting object."""
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)
        # Validate the mask and convert to DataFrame
        mask = _check_mask(data, mask)
        plot_data = np.ma.masked_where(np.asarray(mask), plot_data)
        # Get good names for the rows and columns
        xtickevery = 1
        if isinstance(xticklabels, int):
            xtickevery = xticklabels
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is True:
            xticklabels = _index_to_ticklabels(data.columns)
        elif xticklabels is False:
            xticklabels = []

        ytickevery = 1
        if isinstance(yticklabels, int):
            ytickevery = yticklabels
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is True:
            yticklabels = _index_to_ticklabels(data.index)
        elif yticklabels is False:
            yticklabels = []

        if not len(xticklabels):
            self.xticks = []
            self.xticklabels = []
        elif isinstance(xticklabels, str) and xticklabels == "auto":
            self.xticks = "auto"
            self.xticklabels = _index_to_ticklabels(data.columns)
        else:
            self.xticks, self.xticklabels = self._skip_ticks(xticklabels, xtickevery)

        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, str) and yticklabels == "auto":
            self.yticks = "auto"
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = self._skip_ticks(yticklabels, ytickevery)

        # Get good names for the axis labels
        xlabel = _index_to_label(data.columns) if xlabel is None else xlabel
        ylabel = _index_to_label(data.index) if ylabel is None else ylabel
        self.xlabel = xlabel if xlabel is not None else ""
        self.ylabel = ylabel if ylabel is not None else ""
        self.na_col = na_col

        # Determine good default values for the colormapping
        self._determine_cmap_params(plot_data, vmin, vmax, cmap, center, robust)

        # Sort out the annotations
        if annot is None or annot is False:
            annot = False
            annot_data = None
        else:
            if isinstance(annot, bool):
                annot_data = plot_data
            else:
                annot_data = np.asarray(annot)
                if annot_data.shape != plot_data.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)
            annot = True

        # Save other attributes to the object
        self.data = data
        self.plot_data = plot_data

        self.annot = annot
        self.annot_data = annot_data

        self.fmt = fmt
        self.annot_kws = {} if annot_kws is None else annot_kws.copy()
        self.cbar = cbar
        self.cbar_kws = {} if cbar_kws is None else cbar_kws.copy()
        self.cbar_kws.setdefault("aspect", 5)
        self.cbar_kws.setdefault("fraction", 0.08)
        self.cbar_kws.setdefault("shrink", 0.5)

    def _determine_cmap_params(self, plot_data, vmin, vmax, cmap, center, robust):
        """Use some heuristics to set good defaults for colorbar and range."""
        # plot_data is a np.ma.array instance
        calc_data = plot_data.astype(float).filled(np.nan)
        if vmin is None:
            if robust:
                vmin = np.nanpercentile(calc_data, 2)
            else:
                vmin = np.nanmin(calc_data)
        if vmax is None:
            if robust:
                vmax = np.nanpercentile(calc_data, 98)
            else:
                vmax = np.nanmax(calc_data)
        self.vmin, self.vmax = vmin, vmax

        # Choose default colormaps if not provided
        if cmap is None:
            if center is None:
                try:
                    self.cmap = matplotlib.cm.get_cmap('turbo').copy()
                except:
                    self.cmap = matplotlib.cm.get_cmap('turbo')
            else:
                try:
                    self.cmap = matplotlib.cm.get_cmap('exp1').copy()
                except:
                    self.cmap = matplotlib.cm.get_cmap('exp1')
        elif isinstance(cmap, str):
            try:
                self.cmap = matplotlib.cm.get_cmap(cmap).copy()
            except:
                self.cmap = matplotlib.cm.get_cmap(cmap)
        elif isinstance(cmap, list):
            self.cmap = matplotlib.colors.ListedColormap(cmap)
        else:
            self.cmap = cmap

        self.cmap.set_bad(color=self.na_col)  # set the color for NaN values
        # Recenter a divergent colormap
        if center is not None:
            # Copy bad values
            # in matplotlib<3.2 only masked values are honored with "bad" color spec
            # (see https://github.com/matplotlib/matplotlib/pull/14257)
            bad = self.cmap(np.ma.masked_invalid([np.nan]))[0]  # set the first color as the na_color
            # under/over values are set for sure when cmap extremes
            # do not map to the same color as +-inf
            under = self.cmap(-np.inf)
            over = self.cmap(np.inf)
            under_set = under != self.cmap(0)
            over_set = over != self.cmap(self.cmap.N - 1)

            vrange = max(vmax - center, center - vmin)
            normlize = matplotlib.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            self.cmap = matplotlib.colors.ListedColormap(self.cmap(cc))
            # self.cmap.set_bad(bad)
            if under_set:
                self.cmap.set_under(under)  # set the color of -np.inf as the color for low out-of-range values.
            if over_set:
                self.cmap.set_over(over)

    def _annotate_heatmap(self, ax, mesh):
        """Add textual labels with the value in each cell."""
        mesh.update_scalarmappable()
        height, width = self.annot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
        for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                       mesh.get_array(), mesh.get_facecolors(),
                                       self.annot_data.flat):
            if m is not np.ma.masked:
                lum = _calculate_luminance(color)
                text_color = ".15" if lum > .408 else "w"
                annotation = ("{:" + self.fmt + "}").format(val)
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kws)
                ax.text(x, y, annotation, **text_kwargs)

    def _skip_ticks(self, labels, tickevery):
        """Return ticks and labels at evenly spaced intervals."""
        n = len(labels)
        if tickevery == 0:
            ticks, labels = [], []
        elif tickevery == 1:
            ticks, labels = np.arange(n) + .5, labels
        else:
            start, end, step = 0, n, tickevery
            ticks = np.arange(start, end, step) + .5
            labels = labels[start:end:step]
        return ticks, labels

    def _auto_ticks(self, ax, labels, axis):
        """Determine ticks and ticklabels that minimize overlap."""
        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]
        axis = [ax.xaxis, ax.yaxis][axis]
        tick, = axis.set_ticks([0])
        fontsize = tick.label1.get_size()
        max_ticks = int(size // (fontsize / 72))
        if max_ticks < 1:
            return [], []
        tick_every = len(labels) // max_ticks + 1
        tick_every = 1 if tick_every == 0 else tick_every
        ticks, labels = self._skip_ticks(labels, tick_every)
        return ticks, labels

    def _set_axes_label(self, ax, xlabel_kws, xlabel_bbox_kws, ylabel_kws, ylabel_bbox_kws,
                        xlabel_side, ylabel_side, xlabel_pad, ylabel_pad):
        # xlabel_kws: alpha,color,fontfamily,fontname,fontproperties,fontsize,fontstyle,fontweight,label,rasterized,
        # rotation,rotation_mode(default,anchor),visible, zorder,verticalalignment,horizontalalignment
        if xlabel_kws is None:
            xlabel_kws = {}
        xlabel_kws.setdefault("verticalalignment", 'center')
        xlabel_kws.setdefault("horizontalalignment", 'center')
        ax.xaxis.label.update(
            xlabel_kws)  # print(ax.xaxis.label.properties()) or matplotlib.axis.XAxis.label.properties()
        # xlabel_bbox_kws:alpha,clip_box, clip_on,edgecolor,facecolor,fill,height,in_layout,label,linestyle,
        # linewidth,rasterized,visible,width
        if not xlabel_bbox_kws is None:
            ax.xaxis.label.set_bbox(xlabel_bbox_kws)  # ax.xaxis.label.get_bbox_patch().properties()

        if ylabel_kws is None:
            ylabel_kws = {}
        ylabel_kws.setdefault("horizontalalignment", 'center')  # left, right
        ylabel_kws.setdefault("verticalalignment", 'center')  # top', 'bottom', 'center', 'baseline', 'center_baseline'
        ax.yaxis.label.update(ylabel_kws)
        if not ylabel_bbox_kws is None:
            ax.yaxis.label.set_bbox(ylabel_bbox_kws)  # ax.xaxis.label.get_bbox_patch().properties()

        if xlabel_side:
            ax.xaxis.set_label_position(xlabel_side)
            # ax.xaxis.label.update_bbox_position_size(ax.figure.canvas.get_renderer())
        if ylabel_side:
            ax.yaxis.set_label_position(ylabel_side)

        # ax.xaxis.labelpad = 10 #0.12 * ax.figure.dpi  # 0.12 inches = 3mm
        # ax.yaxis.labelpad = 10 #0.12 * ax.figure.dpi

        # ax.figure.tight_layout(rect=[0, 0, 1, 1])
        _draw_figure(ax.figure)
        # set the xlabel bbox patch color and width, make the width equal to the width of ax.get_window_extent().width
        # xlabel_bb = ax.xaxis.label.get_bbox_patch()
        #
        # ylabel_bb = ax.yaxis.label.get_bbox_patch()
        # cid = ax.figure.canvas.mpl_connect('resize_event', on_resize)
        # cid2 = ax.figure.canvas.mpl_connect('draw_event', on_resize)

    def _set_tick_label(self, ax, xticklabels_side, yticklabels_side, xticklabels_kws, yticklabels_kws):
        # position, (0,0) is at the left top corner.
        if xticklabels_side == 'top':
            ax.xaxis.tick_top()
        elif xticklabels_side == 'bottom':
            ax.xaxis.tick_bottom()
        if yticklabels_side == 'left':
            ax.yaxis.tick_left()
        elif yticklabels_side == 'right':
            ax.yaxis.tick_right()
        # xticklabel_kwas: axis (x,y,both), which (major,minor,both),reset (True,False), direction (in, out, inout),
        # length, width, color (tick color), pad, labelsize, labelcolor, colors (for both tick and label),
        # zorder, bottom, top, left, right (bool), labelbottom, labeltop, labelleft,labelright (bool),
        # labelrotation,grid_color,grid_alpha,grid_linewidth,grid_linestyle; ?matplotlib.axes.Axes.tick_params
        if not xticklabels_kws is None:
            ax.xaxis.set_tick_params(**xticklabels_kws)
        else:
            xticklabels_kws = {}
        if not yticklabels_kws is None:
            ax.yaxis.set_tick_params(**yticklabels_kws)
        else:
            yticklabels_kws = {}

        ha = None
        if xticklabels_kws.get('rotation', 0) > 0 or xticklabels_kws.get('labelrotation', 0) > 0:
            if xticklabels_side == 'top':
                ha = 'left'
            else:
                ha = 'right'
        elif xticklabels_kws.get('rotation', 0) < 0 or xticklabels_kws.get('labelrotation', 0) < 0:
            if xticklabels_side == 'top':
                ha = 'right'
            else:
                ha = 'left'
        if not ha is None:
            plt.setp(ax.get_xticklabels(), rotation_mode='anchor', ha=ha)

    def plot(self, ax, cax, xlabel_kws, xlabel_bbox_kws, ylabel_kws, ylabel_bbox_kws,
             xlabel_side, ylabel_side, xlabel_pad, ylabel_pad, xticklabels_side, yticklabels_side,
             xticklabels_kws, yticklabels_kws, kws):
        """Draw the heatmap on the provided Axes."""
        # Remove all the Axes spines
        despine(ax=ax, left=True, bottom=True)
        # setting vmin/vmax in addition to norm is deprecated
        # so avoid setting if norm is set
        if "norm" not in kws:
            kws.setdefault("vmin", self.vmin)
            kws.setdefault("vmax", self.vmax)

        # Draw the heatmap
        mesh = ax.pcolormesh(self.plot_data, cmap=self.cmap, **kws)
        # Set the axis limits
        ax.set(xlim=(0, self.data.shape[1]), ylim=(0, self.data.shape[0]))
        # Invert the y axis to show the plot in matrix form
        ax.invert_yaxis() # from top to bottom

        # Possibly add a colorbar
        if self.cbar:
            cb = ax.figure.colorbar(mesh, cax, ax, **self.cbar_kws)
            cb.outline.set_linewidth(0)
            # If rasterized is passed to pcolormesh, also rasterize the
            # colorbar to avoid white lines on the PDF rendering
            if kws.get('rasterized', False):
                cb.solids.set_rasterized(True)

        # Add row and column labels
        if isinstance(self.xticks, str) and self.xticks == "auto":
            xticks, xticklabels = self._auto_ticks(ax, self.xticklabels, 0)
        else:
            xticks, xticklabels = self.xticks, self.xticklabels

        if isinstance(self.yticks, str) and self.yticks == "auto":
            yticks, yticklabels = self._auto_ticks(ax, self.yticklabels, 1)
        else:
            yticks, yticklabels = self.yticks, self.yticklabels

        ax.set(xticks=xticks, yticks=yticks)
        xtl = ax.set_xticklabels(xticklabels)
        ytl = ax.set_yticklabels(yticklabels, rotation="vertical")
        plt.setp(ytl, va="center")
        plt.setp(xtl, ha="center")

        # Possibly rotate them if they overlap
        _draw_figure(ax.figure)
        if axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        if axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")

        # Annotate the cells with the formatted values
        if self.annot:
            self._annotate_heatmap(ax, mesh)

        # Add the axis labels
        ax.set(xlabel=self.xlabel, ylabel=self.ylabel)
        # put set tick label in the front of set axes label.
        self._set_tick_label(ax, xticklabels_side, yticklabels_side, xticklabels_kws, yticklabels_kws)
        self._set_axes_label(ax, xlabel_kws, xlabel_bbox_kws, ylabel_kws, ylabel_bbox_kws,
                             xlabel_side, ylabel_side, xlabel_pad, ylabel_pad)

def heatmap(data, xlabel=None, ylabel=None, xlabel_side='bottom', ylabel_side='left',
            vmin=None, vmax=None, cmap=None, center=None, robust=False,
            cbar=True, cbar_kws=None, cbar_ax=None, square=False,
            xlabel_kws=None, ylabel_kws=None, xlabel_bbox_kws=None, ylabel_bbox_kws=None,
            xlabel_pad=None, ylabel_pad=None,
            xticklabels="auto", yticklabels="auto", xticklabels_side='bottom', yticklabels_side='left',
            xticklabels_kws=None, yticklabels_kws=None, mask=None, na_col='white', ax=None,
            annot=None, fmt=".2g", annot_kws=None, linewidths=0, linecolor="white",
            **kwargs):
    """
    Plot heatmap.

    Parameters
    ----------
    data: pandas dataframe
    xlabel / ylabel: True, False, or list of xlabels
    xlabel_side / ylabel_side: bottom or top
    vmax, vmin: the maximal and minimal values for cmap colorbar.
    center, robust: the same as seaborn.heatmap
    xlabel_kws / ylabel_kws: parameter from matplotlib.axis.XAxis.label.properties()

    """
    plotter = heatmapPlotter(data=data, vmin=vmin, vmax=vmax, cmap=cmap, center=center, robust=robust,
                             annot=annot, fmt=fmt, annot_kws=annot_kws, cbar=cbar, cbar_kws=cbar_kws,
                             xlabel=xlabel, ylabel=ylabel, xticklabels=xticklabels, yticklabels=yticklabels,
                             mask=mask, na_col=na_col)
    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor
    # Draw the plot and return the Axes
    if ax is None:
        ax = plt.gca()
    if square:
        ax.set_aspect("equal")
    if xlabel_pad is None:
        xlabel_pad = 0.3  # 30% of the size of mutation_size (fontsize)
    if ylabel_pad is None:
        ylabel_pad = 0.3  # 0.04 * ax.figure.dpi / 16.
    plotter.plot(ax, cbar_ax, xlabel_kws, xlabel_bbox_kws, ylabel_kws, ylabel_bbox_kws,
                 xlabel_side, ylabel_side, xlabel_pad, ylabel_pad, xticklabels_side, yticklabels_side,
                 xticklabels_kws, yticklabels_kws, kwargs)
    return ax

class AnnotationBase():
    """
    Base class for annotation objects.

    Parameters
    ----------
    df: a pandas series or dataframe (only one column).
    cmap: colormap, such as Set1, Dark2, bwr, Reds, jet, hsv, rainbow and so on. Please see
        https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html for more information, or run
        matplotlib.pyplot.colormaps() to see all availabel cmap.
        default cmap is 'auto', it would be determined based on the dtype for each columns of df.
    colors: a dict or list (for boxplot, barplot) or str.
        If colors is a dict: keys should be exactly the same as df.iloc[:,0].unique(),
        values for the dict should be colors (color names or HEX color).
        If  colors is a list, then the length of this list should be equal to df.iloc[:,0].nunique()
        If colors is a string, means all values in df.iloc[:,0].unique() share the same color.
    height: height (if axis=1) / width (if axis=0) for the annotation size.
    legend: whether to plot legend for this annotation when legends are plotted or
        plot legend with HeatmapAnnotation.plot_legends().
    legend_kws: vmax, vmin and other kws passed to plt.legend, such as title, prop, fontsize, labelcolor,
        markscale, frameon, framealpha, fancybox, shadow, facecolor, edgecolor, mode and so on, for more
        arguments, pleast type ?plt.legend. There is an additional parameter `color_text` (default is True),
        which would set the color of the text to the same color as legend marker. if one set
        `legend_kws={'color_text':False}`, then, black would be the default color for the text.
        If the user want to use a custom color instead of black (such as blue), please set
        legend_kws={'color_text':False,'labelcolor':'blue'}.
    plot_kws: other plot kws passed to annotation.plot, such as rotation, rotation_mode, ha, va,
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
        self.nrows = self._n_rows()
        self.ncols = self._n_cols()
        self.height = self._height(height)
        self._type_specific_params()
        self.legend = legend
        self.legend_kws = legend_kws if not legend_kws is None else {}
        self._set_default_plot_kws(plot_kws)

        if colors is None:
            self._check_cmap(cmap)  # add self.dtype, self.cmap (a dict)
            self._calculate_colors()  # modify self.plot_data, self.color_dict (each col is a dict)
            self.colors = None
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
        if plt.get_cmap(self.cmap).N == 256:  # then heatmap will automatically calculate vmin and vmax
            try:
                self.plot_kws.setdefault('vmax', np.nanmax(self.df.values))
                self.plot_kws.setdefault('vmin', np.nanmin(self.df.values))
            except:
                pass

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.color_dict = {}
        col = self.df.columns.tolist()[0]
        if plt.get_cmap(self.cmap).N < 256 or self.df.dtypes[col] == object:
            cc_list = self.df[col].value_counts().index.tolist()  # sorted by value counts
            self.df[col] = self.df[col].map({v: cc_list.index(v) for v in cc_list})
            for v in cc_list:
                color = plt.get_cmap(self.cmap)(cc_list.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            self.color_dict = {v: plt.get_cmap(self.cmap)(v) for v in self.df[col].values}

    def _check_colors(self, colors):
        if isinstance(colors,str):
            colors={label:colors for label in self.df.iloc[:,0].unique()}
        if isinstance(colors,list):
            assert len(colors) == self.df.iloc[:, 0].nunique()
            colors = {label: color for label,color in zip(self.df.iloc[:, 0].unique(),colors)}
        if not isinstance(colors, dict):
            raise TypeError("colors must be a dict!")
        if len(colors) >= self.df.iloc[:, 0].nunique():
            self.colors = colors
        else:
            raise TypeError("Unknown type of colors")

    def _calculate_cmap(self):
        self.color_dict = self.colors
        col = self.df.columns.tolist()[0]
        cc_list = list(self.color_dict.keys())  # column values
        self.df[col] = self.df[col].map({v: cc_list.index(v) for v in cc_list})
        self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])
        self.plot_kws.setdefault('vmax', plt.get_cmap(self.cmap).N)
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
        self.plot_data = self.df.reindex(idx) #
        self.plot_data.fillna(np.nan, inplace=True)
        self.nrows = self.plot_data.shape[0]
        # self._set_default_plot_kws(self.plot_kws)

    def get_label_width(self):
        return self.ax.yaxis.label.get_window_extent().width

    def get_ticklabel_width(self):
        yticklabels = self.ax.yaxis.get_ticklabels()
        if len(yticklabels) == 0:
            return 0
        else:
            return max([label.get_window_extent().width for label in self.ax.yaxis.get_ticklabels()])

    def get_max_label_width(self):
        return max([self.get_label_width(),self.get_ticklabel_width()])

class anno_simple(AnnotationBase):
    """
    Annotate simple annotation, categorical or continuous variables.
    """

    def __init__(self, df=None, cmap='auto', colors=None, add_text=False,
                 majority=True,text_kws=None, height=None, legend=True,
                 legend_kws=None,**plot_kws):
        self.add_text = add_text
        self.majority=majority
        self.text_kws = text_kws if not text_kws is None else {}
        self.plot_kws = plot_kws
        # print(self.plot_kws)
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
        if plt.get_cmap(self.cmap).N < 256:
            cc_list = self.df[col].value_counts().index.tolist()  # sorted by value counts
            for v in cc_list:
                color = plt.get_cmap(self.cmap)(cc_list.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            cc_list = None
            self.color_dict = {v: plt.get_cmap(self.cmap)(v) for v in self.df[col].values}
        self.cc_list = cc_list

    def _calculate_cmap(self):
        self.color_dict = self.colors
        col = self.df.columns.tolist()[0]
        cc_list = list(self.color_dict.keys())  # column values
        self.cc_list = cc_list
        self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])
        self.plot_kws.setdefault('vmax', plt.get_cmap(self.cmap).N)
        self.plot_kws.setdefault('vmin', 0)

    def plot(self, ax=None, axis=1, subplot_spec=None, label_kws={},
             ticklabels_kws={}):  # add self.gs,self.fig,self.ax,self.axes
        self.plot_kws.setdefault('vmax', plt.get_cmap(self.cmap).N)
        self.plot_kws.setdefault('vmin', 0)
        if self.cc_list:
            mat = self.plot_data.iloc[:, 0].map({v: self.cc_list.index(v) for v in self.cc_list}).values
        else:
            mat = self.plot_data.values
        matrix = mat.reshape(1, -1) if axis == 1 else mat.reshape(-1, 1)
        # print(matrix)
        xlabel = None if axis == 1 else self.label
        ylabel = self.label if axis == 1 else None

        # print(self.plot_kws)
        ax1 = heatmap(matrix, cmap=self.cmap, cbar=False, ax=ax, xlabel=xlabel, ylabel=ylabel,
                      xticklabels=False, yticklabels=False, **self.plot_kws)
        # if axis==0:
        #     ax1.invert_yaxis()
        ax.tick_params(axis='both', which='both',
                       left=False, right=False, top=False, bottom=False,
                       labeltop=False, labelbottom=False, labelleft=False, labelright=False)
        if self.add_text:
            if axis==0:
                self.text_kws.setdefault('rotation', 90)
            labels, ticks = cluster_labels(self.plot_data.iloc[:, 0].values, np.arange(0.5, self.nrows, 1),self.majority)
            n = len(ticks)
            if axis == 1:
                x = ticks
                y = [0.5] * n
            else:
                y = ticks
                x = [0.5] * n
            s = ax.get_window_extent().height if axis == 1 else ax.get_window_extent().width
            self.text_kws.setdefault('fontsize',72 * s * 0.8 / ax.figure.dpi)
            # fontsize = self.text_kws.pop('fontsize', 72 * s * 0.8 / ax.figure.dpi)
            color = self.text_kws.pop('color', None)
            for x0, y0, t in zip(x, y, labels):
                lum = _calculate_luminance(self.color_dict[t])
                text_color = "black" if lum > 0.408 else "white"
                # print(t,self.color_dict,text_color,color)
                self.text_kws.setdefault('color', text_color)
                ax.text(x0, y0, t,**self.text_kws)
        self.ax = ax
        self.fig = self.ax.figure
        return self.ax

class anno_label(AnnotationBase):
    """
    Add label and text annotations.

    Parameters
    ----------
    merge: whether to merge the same clusters into one and label only once.
    extend: whether to distribute all the labels extend to the all axis, figure or ax or False.
    frac: fraction of the armA and armB.
    plot_kws is passed to plt.annotate, including annotation_clip, arrowprops and matplotlib.text.Text,
        more information about arrowprops could be found in
        matplotlib.patches.FancyArrowPatch

    Returns
    ----------
    Class AnnotationBase.

    """

    def __init__(self, df=None, cmap='auto', colors=None, merge=False,extend=False,frac=0.2,
                 majority=True,height=None, legend=False, legend_kws=None, **plot_kws):
        super().__init__(df=df, cmap=cmap, colors=colors,
                         height=height, legend=legend, legend_kws=legend_kws, **plot_kws)
        self.merge = merge
        self.majority=majority
        self.extend = extend
        self.frac=frac
        self.annotated_texts = []

    def _height(self, height):
        return 5 if height is None else height

    def set_side(self, side):
        self.side = side

    def set_plot_kws(self, axis):
        shrink = 1  # 1 * mm2inch * 72  # 1mm -> points
        if axis == 1: #columns
            relpos = (0, 0) if self.side == 'top' else (0, 1) #position to anchor, x: left -> right, y: down -> top
            rotation = 90 if self.side == 'top' else -90
            ha = 'left'
            va = 'center'
        else:
            relpos = (1, 1) if self.side == 'left' else (0, 0) #(1, 1) if self.side == 'left' else (0, 0)
            rotation = 0
            ha = 'right' if self.side == 'left' else 'left'
            va = 'center'
        #relpos: The exact starting point position of the arrow is defined by relpos. It's a tuple of relative
        # coordinates of the text box, where (0, 0) is the lower left corner and (1, 1) is the upper right corner.
        # Values <0 and >1 are supported and specify points outside the text box. By default (0.5, 0.5) the starting
        # point is centered in the text box.
        self.plot_kws.setdefault('rotation', rotation)
        self.plot_kws.setdefault('ha', ha)
        self.plot_kws.setdefault('va', va)
        arrowprops = dict(arrowstyle="-", color="black",
                          shrinkA=shrink, shrinkB=shrink, relpos=relpos,
                          patchA=None, patchB=None, connectionstyle=None)
        #arrow: ->, from text to point.
        # self.plot_kws.setdefault('transform_rotates_text', False)
        self.plot_kws.setdefault('arrowprops', arrowprops)
        self.plot_kws.setdefault('rotation_mode', 'anchor')

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.color_dict = {}
        col = self.df.columns.tolist()[0]
        if plt.get_cmap(self.cmap).N < 256 or self.df.dtypes[col] == object:
            cc_list = self.df[col].value_counts().index.tolist()  # sorted by value counts
            for v in cc_list:
                color = plt.get_cmap(self.cmap)(cc_list.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            self.color_dict = {v: plt.get_cmap(self.cmap)(v) for v in self.df[col].values}

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
            labels, ticks = cluster_labels(self.plot_data.iloc[:, 0].values, np.arange(0.5, self.nrows, 1),self.majority)
        else:
            labels = self.plot_data.iloc[:, 0].values
            ticks = np.arange(0.5, self.nrows, 1)
        #labels are the merged labels, ticks are the merged mean x coordinates.

        n = len(ticks)
        text_height = self.height * mm2inch * ax.figure.dpi  # convert height (mm) to inch and to pixels.
        # print(ax.figure.dpi,text_height)
        text_y = text_height
        if self.side == 'bottom' or self.side == 'left':
            #when axis=0, there is an invertion for the x axes.
            text_y = -1 * text_y
            #bottom, y coordinate for text should be smaller than arrow start position (offset pixels)
            # left, x coordinate for text should be smaller than arrow start pos.
        #x,y are the coordinates where arrow starting from and x1,y1 is the end coordinates (text).
        if axis == 1:
            ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
            x = ticks #coordinate for the labels (text).
            y = [0] * n if self.side == 'top' else [1] * n #position for line on axes
            if self.extend:
                extend_pos=np.linspace(0, 1, n+1)
                x1 = [(extend_pos[i]+extend_pos[i-1])/2 for i in range(1,n+1)]
                y1 = [1+text_y/ax.figure.get_window_extent().height] * n if self.side=='top' else [text_y/ax.figure.get_window_extent().height] * n
            else:
                x1=[0]*n
                y1=[text_y] * n
        else:
            ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
            y = ticks
            x = [1] * n if self.side == 'left' else [0] * n
            if self.extend:
                extend_pos = np.linspace(0, 1, n + 1)
                y1 = [(extend_pos[i] + extend_pos[i - 1]) / 2 for i in range(1, n + 1)]
                x1 = [1+text_y/ax.figure.get_window_extent().width] * n if self.side=='right' else [0+text_y/ax.figure.get_window_extent().width] * n
            else:
                y1=[0] * n
                x1=[text_y] * n
        #angleA is the angle for the data point (clockwise), B is for text.
        #https://matplotlib.org/stable/gallery/userdemo/connectionstyle_demo.html
        xycoords = ax.get_xaxis_transform() if axis == 1 else ax.get_yaxis_transform()
        # get_xaxis_transform: x is data coordinates,y is between [0,1]
        if self.extend:
            text_xycoords=ax.transAxes #if self.extend=='ax' else ax.figure.transFigure #ax.transAxes,
        else:
            text_xycoords='offset pixels'
        if self.plot_kws['arrowprops']['connectionstyle'] is None:
            arm_height = text_height * self.frac
            rad = 2  # arm_height / 10
            if axis == 1 and self.side == 'top':
                angleA, angleB = (self.plot_kws['rotation'] - 180,90)
            elif axis == 1 and self.side == 'bottom':
                angleA, angleB = (180 + self.plot_kws['rotation'], -90)
            elif axis==0 and self.side=='left':
                angleA, angleB = (self.plot_kws['rotation'], -180)
            else:
                angleA, angleB = (self.plot_kws['rotation'] - 180,0)
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
            lum = _calculate_luminance(color)
            if lum > 0.408:
                color = 'black'
            self.plot_kws['arrowprops']['color'] = color
            annotated_text = ax.annotate(text=t, xy=(x_0, y_0), xytext=(x_1, y_1), xycoords=xycoords, textcoords=text_xycoords,
                              color=color, **self.plot_kws)  # unit for shrinkA is point (1 point = 1/72 inches)
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
        hs=[text.get_window_extent().width for text in self.annotated_texts]
        if len(hs) == 0:
            return 0
        else:
            return max(hs)

class anno_boxplot(AnnotationBase):
    """
    annotate boxplots, all arguments are included in AnnotationBase,
    plot_kws for anno_boxplot include showfliers, edgecolor, grid, medianlinecolor
        width and other arguments passed to plt.boxplot.
    """

    def _height(self, height):
        return 8 if height is None else height

    def _set_default_plot_kws(self, plot_kws):
        self.plot_kws = plot_kws if plot_kws is not None else {}
        self.plot_kws.setdefault('showfliers', False)
        self.plot_kws.setdefault('edgecolor', 'black')
        self.plot_kws.setdefault('medianlinecolor', 'black')
        self.plot_kws.setdefault('grid', True)
        self.plot_kws.setdefault('zorder', 10)
        self.plot_kws.setdefault('widths', 0.5)

    def _check_cmap(self, cmap):
        self.cmap = 'jet'
        if cmap == 'auto':
            self.cmap = 'jet'
        elif type(cmap) == str:
            self.cmap = cmap
        else:
            raise TypeError("cmap for boxplot should be a string")

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.color_dict = {}
        col = self.df.columns.tolist()[0]
        if plt.get_cmap(self.cmap).N < 256:
            cc_list = self.df[col].value_counts().index.tolist()  # sorted by value counts
            for v in cc_list:
                color = plt.get_cmap(self.cmap)(cc_list.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            self.color_dict = {v: plt.get_cmap(self.cmap)(v) for v in self.df[col].values}

    def _check_colors(self, colors):
        if type(colors) == str:
            self.colors = colors
        else:
            raise TypeError(
                "Boxplot only support one string as colors now, if more colors are wanted, cmap can be specified.")
        self.color_dict = None

    def _calculate_cmap(self):
        self.color_dict = None

    def _type_specific_params(self):
        gap = self.df.max().max() - self.df.min().min()
        self.ylim = [self.df.min().min() - 0.02 * gap, self.df.max().max() + 0.02 * gap]

    def plot(self, ax=None, axis=1, subplot_spec=None, label_kws={},
             ticklabels_kws={}):  # add self.gs,self.fig,self.ax,self.axes
        fig = ax.figure
        if self.colors is None:  # calculate colors based on cmap
            colors = [plt.get_cmap(self.cmap)(self.plot_data.loc[sampleID].mean()) for sampleID in
                      self.plot_data.index.values]
        else:
            colors = [self.colors] * self.plot_data.shape[0]  # self.colors is a string
        # print(self.plot_kws)
        plot_kws=self.plot_kws.copy()
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

    def _check_cmap(self, cmap):
        self.cmap = 'jet' if self.ncols == 1 else 'Set1'
        if cmap == 'auto':
            pass
        elif type(cmap) == str:
            self.cmap = cmap
        else:
            raise TypeError("cmap for boxplot should be a string")

    def _calculate_colors(self):  # add self.color_dict (each col is a dict)
        self.color_dict = {}
        cols = self.df.columns.tolist()
        if plt.get_cmap(self.cmap).N < 256:
            for v in cols:
                color = plt.get_cmap(self.cmap)(cols.index(v))
                self.color_dict[v] = color  # matplotlib.colors.to_hex(color)
        else:  # float
            self.color_dict = {v: plt.get_cmap(self.cmap)(cols.index(v)) for v in cols}

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
        self.color_dict = {}
        for v, c in zip(self.df.columns.tolist(), self.colors):
            self.color_dict[v] = c

    def _calculate_cmap(self):
        cc_list = list(self.color_dict.keys())  # column values
        self.cmap = matplotlib.colors.ListedColormap([self.color_dict[k] for k in cc_list])

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
        if self.colors is None:
            col_list = self.plot_data.columns.tolist()
            colors = [plt.get_cmap(self.cmap)(col_list.index(v)) for v in self.plot_data.columns]
        else:  # self.colors is a list, length equal to the plot_data.shape[1]
            colors = self.colors
        plot_kws=self.plot_kws.copy()
        grid = plot_kws.pop('grid',False)
        if grid:
            ax.grid(linestyle='--', zorder=-10)
        # bar_ct = ax.bar(x=list(range(1, self.nrows + 1,1)),
        #                 height=self.plot_data.values,**self.plot_kws)
        for col, color in zip(self.plot_data.columns, colors):
            if axis == 1:
                ax.set_xticks(ticks=np.arange(0.5, self.nrows, 1))
                ax.bar(x=np.arange(0.5, self.nrows, 1), height=self.plot_data[col].values, color=color, **plot_kws)
            else:
                ax.set_yticks(ticks=np.arange(0.5, self.nrows, 1))
                ax.barh(y=np.arange(0.5, self.nrows, 1), width=self.plot_data[col].values, color=color, **plot_kws)
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
            raise TypeError("cmap for boxplot should be a string")

    def _check_colors(self, colors):
        if not isinstance(colors, str):
            raise TypeError("colors must be string for scatterplot, if more colors are neded, please try cmap!")
        self.colors = colors
        self.color_dict = None

    def _calculate_cmap(self):
        self._check_cmap('auto')

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
        plot_kws=self.plot_kws.copy()
        grid = plot_kws.pop('grid',False)
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

class HeatmapAnnotation():
    """
    Generate and plot heatmap annotations.

    Parameters
    ----------
    self : Class HeatmapAnnotation
    df :  a pandas dataframe, each column will be converted to one anno_simple class.
    axis : 1 for columns annotation, 0 for rows annotations.
    cmap : colormap, such as Set1, Dark2, bwr, Reds, jet, hsv, rainbow and so on. Please see
        https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html for more information, or run
        matplotlib.pyplot.colormaps() to see all availabel cmap.
        default cmap is 'auto', it would be determined based on the dtype for each columns in df.
        If df is None, then there is no need to specify cmap, cmap and colors will only be used when
        df is provided.
        If cmap is a string, then all columns in the df would have the same cmap, cmap can also be
        a dict, keys are the column names from df, values should be cmap (matplotlib.pyplot.colormaps()).
    colors : a dict, keys are the column names of df, values are list, dict or string passed to
        AnnotationBase.__subclasses__(), including anno_simple, anno_boxplot,anno_label and anno_scatter.
        colors must have the same length as the df.columns, if colors is not provided (default), else,
        colors would be calculated based on the given cmap.
        If colors is given, then the cmap would be invalid.
    label_side : top or bottom when axis=1 (columns annotation), left or right when axis=0 (rows annotations).
    label_kws :kws passed to the labels of the annotation labels (would be df.columns if df is given).
        such as alpha, color, fontsize, fontstyle, ha (horizontalalignment),
        va (verticalalignment), rotation, rotation_mode, visible, rasterized and so on.
        For more information, see plt.gca().yaxis.label.properties() or ax.yaxis.label.properties()
    ticklabels_kws : label_kws is for the label of annotation, ticklabels_kws is for the label (text) in anno_label,
        such as axis, which, direction, length, width,
        color, pad, labelsize, labelcolor, colors, zorder, bottom, top, left, right, labelbottom, labeltop,
        labelleft, labelright, labelrotation, grid_color, grid_linestyle and so on.
        For more information,see ?matplotlib.axes.Axes.tick_params
    plot_kws : kws passed to annotation functions, such as anno_simple, anno_label et.al.
    plot : whether to plot, when the annotation are included in clustermap, plot would be
        set to False automotially.
    legend : True or False, or dict (when df is no None), when legend is dict, keys are the
        columns of df.
    legend_side : right or left
    legend_gap : the vertical gap between two legends, default is 2 [mm]
    legend_width: width of the legend, default is 4.5[mm]
    legend_hpad: Horizonal space between heatmap and legend, default is 2 [mm].
    legend_vpad: Vertical space between top of ax and legend, default is 2 [mm].
    plot_legend : whether to plot legends.
    args : name-value pair, key is the annotation label (name), values can be a pandas dataframe,
        series, or annotation such as
        anno_simple, anno_boxplot, anno_scatter, anno_label, or anno_barplot.

    Returns
    -------
    Class HeatmapAnnotation.

    """
    def __init__(self, df=None, axis=1, cmap='auto', colors=None, label_side=None, label_kws=None,
                 ticklabels_kws=None, plot_kws=None, plot=False, legend=True, legend_side='right',
                 legend_gap=2, legend_width=4.5,legend_hpad=2,legend_vpad=5,
                 plot_legend=True,rasterized=False,verbose=1,**args):
        if df is None and len(args) == 0:
            raise ValueError("Please specify either df or other args")
        if not df is None and len(args) > 0:
            raise ValueError("df and Name-value pairs can only be given one, not both.")
        if not df is None:
            self._check_df(df)
        else:
            self.df = None
        self.axis = axis
        self.verbose=verbose
        self.label_side = label_side
        self._set_label_kws(label_kws, ticklabels_kws)
        self.plot_kws = plot_kws if not plot_kws is None else {}
        self.args = args
        self._check_legend(legend)
        self.legend_side = legend_side
        self.legend_gap = legend_gap
        self.legend_width = legend_width
        self.legend_hpad = legend_hpad
        self.legend_vpad = legend_vpad
        self.plot_legend = plot_legend
        self.rasterized=rasterized
        self.plot = plot
        if colors is None:
            self._check_cmap(cmap)
            self.colors = None
        else:
            self._check_colors(colors)
        self._process_data()
        self._heights()
        self._nrows()
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
            if len(self.args)>0:
                # self.legend = collections.defaultdict(lambda: legend)
                self.legend = {arg: legend for arg in self.args}
        elif type(legend) == dict:
            self.legend = legend
            for arg in self.args:
                if arg not in self.legend:
                    self.legend[arg]=False
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
        self.plot_kws["rasterized"]=self.rasterized
        if not self.df is None:
            for col in self.df.columns:
                plot_kws = self.plot_kws.copy()
                if self.colors is None:
                    plot_kws.setdefault("cmap", self.cmap[col]) #
                else:
                    plot_kws.setdefault("colors", self.colors[col])
                anno1 = anno_simple(self.df[col], legend=self.legend.get(col,False), **plot_kws)
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
                            anno1 = anno_simple(ann[col], legend=self.legend.get(col,False), **self.plot_kws)
                            anno1.set_label(col)
                            self.annotations.append(anno1)
                    else:
                        anno1 = anno_simple(ann,**self.plot_kws)
                        anno1.set_label(arg)
                        anno1.set_legend(self.legend.get(arg,False))
                        self.annotations.append(anno1)
                if hasattr(ann, 'set_label') and AnnotationBase.__subclasscheck__(type(ann)):
                    self.annotations.append(ann)
                    ann.set_label(arg)
                    ann.set_legend(self.legend.get(arg,False))
                    if type(ann) == anno_label:
                        if self.axis == 1 and len(self.labels) == 0:
                            ann.set_side('top')
                        elif self.axis == 1:
                            ann.set_side('bottom')
                        elif self.axis == 0 and len(self.labels) == 0:
                            ann.set_side('left')
                        elif self.axis == 0:
                            ann.set_side('right')
                self.labels.append(arg)

    def _heights(self):
        self.heights = [ann.height for ann in self.annotations]

    def _nrows(self):
        self.nrows = [ann.nrows for ann in self.annotations]

    def _set_label_kws(self, label_kws, ticklabels_kws):
        self.label_kws = {} if label_kws is None else label_kws
        self.ticklabels_kws = {} if ticklabels_kws is None else ticklabels_kws
        self.label_kws.setdefault("rotation_mode",'anchor')
        if self.label_side is None:
            self.label_side = 'right' if self.axis == 1 else 'top'  # columns annotation, default ylabel is on the right
        # ha = 'left' if self.label_side == 'left' else 'left' if self.label_side == 'right' else 'center'
        # va = 'bottom' if self.label_side == 'top' else 'top' if self.label_side == 'bottom' else 'center'
        ha,va='left','center'
        self.label_kws.setdefault('horizontalalignment', ha)
        self.label_kws.setdefault('verticalalignment', va)
        if self.label_side in ['left', 'right'] and self.axis != 1:
            raise ValueError("For columns annotation, label_side must be left or right!")
        if self.label_side in ['top', 'bottom'] and self.axis != 0:
            raise ValueError("For row annotation, label_side must be top or bottom!")

        map_dict = {'right': 'left', 'left': 'right', 'top': 'bottom', 'bottom': 'top'}
        self.ticklabels_side = map_dict[self.label_side]

        # self.label_kws.setdefault('fontsize', self.height * mm2inch * 72)
        if self.axis == 1:
            self.label_kws.setdefault('rotation', 0)
        else:
            self.label_kws.setdefault('rotation', 90)
            self.ticklabels_kws.setdefault('labelrotation', 90)
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
            legend_kws = annotation.legend_kws.copy()
            if annotation.legend:
                if plt.get_cmap(annotation.cmap).N < 256:
                    color_dict = annotation.color_dict
                    if color_dict is None:
                        continue
                    self.legend_list.append(
                        [annotation.color_dict, annotation.label, legend_kws, len(annotation.color_dict)])
                else:
                    if annotation.df.shape[1] == 1:
                        array = annotation.df.iloc[:, 0].values
                    else:
                        array = annotation.df.values
                    vmax = np.nanmax(array[array != np.inf])
                    vmin = np.nanmin(array[array != -np.inf])
                    legend_kws.setdefault('vmin', round(vmin, 2))
                    legend_kws.setdefault('vmax', round(vmax, 2))
                    self.legend_list.append([annotation.cmap, annotation.label, legend_kws, 4])
        if len(self.legend_list) > 1:
            self.legend_list = sorted(self.legend_list, key=lambda x: x[3])
        self.label_max_width = max([ann.get_max_label_width() for ann in self.annotations])
        # self.label_max_height = max([ann.ax.yaxis.label.get_window_extent().height for ann in self.annotations])

    def plot_annotations(self, ax=None, subplot_spec=None, idxs=None, gap=0.5,
                         wspace=None, hspace=None):
        """
        Parameters
        ----------
        ax : axes to plot the annotations.
        subplot_spec : object from ax.figure.add_gridspec or matplotlib.gridspec.GridSpecFromSubplotSpec.
        idxs : index to reorder df and df of annotation class.
        gap : gap to calculate wspace and hspace for gridspec.
        wspace : if wspace not is None, use wspace, else wspace would be calculated based on gap.
        hspace : if hspace not is None, use hspace, else hspace would be calculated based on gap.

        Returns
        -------
        self.ax
        """
        # print(ax.figure.get_size_inches())
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
            wspace = gap * mm2inch * self.ax.figure.dpi / (
                    self.ax.get_window_extent().width / nrows) if wspace is None else wspace  # 1mm=mm2inch inch
            hspace = 0
        else:
            nrows = len(idxs)
            ncols = len(self.heights)
            width_ratios = self.heights
            height_ratios = [len(idx) for idx in idxs]
            hspace = gap * mm2inch * self.ax.figure.dpi / (
                        self.ax.get_window_extent().height / ncols) if hspace is None else hspace
            wspace = 0
        if subplot_spec is None:
            self.gs = self.ax.figure.add_gridspec(nrows, ncols, hspace=hspace, wspace=wspace,
                                                  height_ratios=height_ratios,
                                                  width_ratios=width_ratios)
        else:
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
                #axis=1: left -> right, axis=0: bottom -> top.
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
                else:
                    ax1.invert_yaxis() # fix bug for inversed row order, invert yaxis for rows annotations.
                    ax1.xaxis.label.set_visible(False)
                    ax1.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False)
                    self.ax.spines['left'].set_visible(False)
                    self.ax.spines['right'].set_visible(False)
                    self.axes[j, i] = ax1
        self.set_axes_kws()
        self.legend_list = None
        if self.plot and self.plot_legend:
            self.plot_legends(ax=self.ax)
        # _draw_figure(self.ax.figure)
        return self.ax

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
            #if the legend is on the right side
            space = self.label_max_width if (self.legend_side=='right' and self.label_side=='right') else 0
            legend_hpad= self.legend_hpad * mm2inch * self.ax.figure.dpi #mm to inch to pixel
            self.legend_axes, self.cbars,self.boundry = \
                plot_legend_list(self.legend_list, ax=ax, space=space + legend_hpad,
                                 legend_side='right',gap=self.legend_gap,
                                 legend_width=self.legend_width,legend_vpad=self.legend_vpad)

class DendrogramPlotter(object):
    def __init__(self, data, linkage, metric, method, axis, label, rotate, dendrogram_kws=None):
        """Plot a dendrogram of the relationships between the columns of data
        """
        self.axis = axis
        if self.axis == 1:  # default 1, columns, when calculating dendrogram, each row is a point.
            data = data.T
        self.check_array(data)
        self.shape = self.data.shape
        self.metric = metric
        self.method = method
        self.label = label
        self.rotate = rotate
        self.dendrogram_kws = dendrogram_kws if not dendrogram_kws is None else {}
        if linkage is None:
            self.linkage = self.calculated_linkage
        else:
            self.linkage = linkage
        self.dendrogram = self.calculate_dendrogram()
        # Dendrogram ends are always at multiples of 5, who knows why
        ticks = np.arange(self.data.shape[0]) + 0.5  # xticklabels

        if self.label:
            ticklabels = _index_to_ticklabels(self.data.index)
            ticklabels = [ticklabels[i] for i in self.reordered_ind]
            if self.rotate:  # horizonal
                self.xticks = []
                self.yticks = ticks
                self.xticklabels = []

                self.yticklabels = ticklabels
                self.ylabel = _index_to_label(self.data.index)
                self.xlabel = ''
            else:  # vertical
                self.xticks = ticks
                self.yticks = []
                self.xticklabels = ticklabels
                self.yticklabels = []
                self.ylabel = ''
                self.xlabel = _index_to_label(self.data.index)
        else:
            self.xticks, self.yticks = [], []
            self.yticklabels, self.xticklabels = [], []
            self.xlabel, self.ylabel = '', ''

        self.dependent_coord = np.array(self.dendrogram['dcoord'])
        self.independent_coord = np.array(self.dendrogram['icoord']) / 10

    def check_array(self, data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        # To avoid missing values and infinite values and further error, remove missing values
        # nrow = data.shape[0]
        # keep_col = data.apply(np.isfinite).sum() == nrow
        # if keep_col.sum() < 3:
        #     raise ValueError("There are too many missing values or infinite values")
        # data = data.loc[:, keep_col[keep_col].index.tolist()]
        if data.isna().sum().sum() > 0:
            data = data.apply(lambda x: x.fillna(x.median()),axis=1)
        self.data = data
        self.array = data.values

    def _calculate_linkage_scipy(self):  # linkage is calculated by columns
        # print(type(self.array),self.method,self.metric)
        linkage = hierarchy.linkage(self.array, method=self.method, metric=self.metric)
        return linkage  # array is a distance matrix?

    def _calculate_linkage_fastcluster(self):
        import fastcluster
        # Fastcluster has a memory-saving vectorized version, but only
        # with certain linkage methods, and mostly with euclidean metric
        # vector_methods = ('single', 'centroid', 'median', 'ward')
        euclidean_methods = ('centroid', 'median', 'ward')
        euclidean = self.metric == 'euclidean' and self.method in euclidean_methods
        if euclidean or self.method == 'single':
            return fastcluster.linkage_vector(self.array, method=self.method, metric=self.metric)
        else:
            linkage = fastcluster.linkage(self.array, method=self.method, metric=self.metric)
            return linkage

    @property
    def calculated_linkage(self):
        try:
            return self._calculate_linkage_fastcluster()
        except ImportError:
            if np.product(self.shape) >= 1000:
                msg = ("Clustering large matrix with scipy. Installing "
                       "`fastcluster` may give better performance.")
                warnings.warn(msg)
        return self._calculate_linkage_scipy()

    def calculate_dendrogram(self):  # Z (linkage) shape = (n,4), then dendrogram icoord shape = (n,4)
        return hierarchy.dendrogram(self.linkage, no_plot=True, labels=self.data.index.tolist(),
                                    get_leaves=True, **self.dendrogram_kws)  # color_threshold=-np.inf,

    @property
    def reordered_ind(self):
        """Indices of the matrix, reordered by the dendrogram"""
        return self.dendrogram['leaves']  # idx of the matrix

    def plot(self, ax, tree_kws):
        """Plots a dendrogram of the similarities between data on the axes
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object upon which the dendrogram is plotted
        """
        tree_kws = {} if tree_kws is None else tree_kws
        tree_kws.setdefault("linewidth", .5)
        tree_kws.setdefault("colors", None)
        # tree_kws.setdefault("colors", tree_kws.pop("color", (.2, .2, .2)))
        if self.rotate and self.axis == 0:  # 0 is rows, 1 is columns (default)
            coords = zip(self.dependent_coord, self.independent_coord)  # independent is icoord (x), horizontal
        else:
            coords = zip(self.independent_coord, self.dependent_coord)  # vertical
        # lines = LineCollection([list(zip(x,y)) for x,y in coords], **tree_kws)  #
        # ax.add_collection(lines)
        colors = tree_kws.pop('colors')
        if colors is None:
            # colors=self.dendrogram['leaves_color_list']
            colors = ['black'] * len(self.dendrogram['ivl'])
        for (x, y), color in zip(coords, colors):
            ax.plot(x, y, color=color, **tree_kws)
        number_of_leaves = len(self.reordered_ind)
        max_dependent_coord = max(map(max, self.dependent_coord))  # max y

        if self.rotate:  # horizontal
            ax.yaxis.set_ticks_position('right')
            # Constants 10 and 1.05 come from
            # `scipy.cluster.hierarchy._plot_dendrogram`
            ax.set_ylim(0, number_of_leaves)
            # ax.set_xlim(0, max_dependent_coord * 1.05)
            ax.set_xlim(0, max_dependent_coord)
            ax.invert_xaxis()
            ax.invert_yaxis()
        else:  # vertical
            # Constants 10 and 1.05 come from
            # `scipy.cluster.hierarchy._plot_dendrogram`
            ax.set_xlim(0, number_of_leaves)
            ax.set_ylim(0, max_dependent_coord)
        despine(ax=ax, bottom=True, left=True)
        ax.set(xticks=self.xticks, yticks=self.yticks,
               xlabel=self.xlabel, ylabel=self.ylabel)
        xtl = ax.set_xticklabels(self.xticklabels)
        ytl = ax.set_yticklabels(self.yticklabels, rotation='vertical')
        # Force a draw of the plot to avoid matplotlib window error
        # _draw_figure(ax.figure)
        if len(ytl) > 0 and axis_ticklabels_overlap(ytl):
            plt.setp(ytl, rotation="horizontal")
        if len(xtl) > 0 and axis_ticklabels_overlap(xtl):
            plt.setp(xtl, rotation="vertical")
        return self

class ClusterMapPlotter():
    """
    Clustermap (Heatmap) plotter.
    Plot heatmap / clustermap with annotation and legends.

    Parameters
    ----------
    data : pandas dataframe or numpy array.
    z_score : whether to perform z score scale, either 0 for rows or 1 for columns, after scale,
        value range would be from -1 to 1.
    standard_scale : either 0 for rows or 1 for columns, after scale,value range would be from 0 to 1.
    top_annotation : annotation: class of HeatmapAnnotation.
    bottom_annotation : the same as top_annotation.
    left_annotation :the same as top_annotation.
    right_annotation :the same as top_annotation.
    row_cluster :whether to perform cluster on rows/columns.
    col_cluster :whether to perform cluster on rows/columns.
    row_cluster_method :cluster method for row/columns linkage, such single, complete, average,weighted,
        centroid, median, ward. see scipy.cluster.hierarchy.linkage or
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) for detail.
    row_cluster_metric : Pairwise distances between observations in n-dimensional space for row/columns,
        such euclidean, minkowski, cityblock, seuclidean, cosine, correlation, hamming, jaccard,
        chebyshev, canberra, braycurtis, mahalanobis, kulsinski et.al.
        centroid, median, ward. see scipy.cluster.hierarchy.linkage or
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
    col_cluster_method :same as row_cluster_method
    col_cluster_metric :same as row_cluster_metric
    show_rownames :True (default) or False, whether to show row ticklabels.
    show_colnames : True of False, same as show_rownames.
    row_names_side :right or left.
    col_names_side :top or bottom.
    row_dendrogram :True or False, whether to show dendrogram.
    col_dendrogram :True or False, whether to show dendrogram.
    row_dendrogram_size :int, default is 10mm.
    col_dendrogram_size :int, default is 10mm.
    row_split :int (number of cluster for hierarchical clustering) or pd.Series or pd.DataFrame,
        used to split rows or rows into subplots.
    col_split :int or pd.Series or pd.DataFrame, used to split rows or columns into subplots.
    dendrogram_kws :kws passed to hierarchy.dendrogram.
    tree_kws :kws passed to DendrogramPlotter.plot()
    row_split_gap :default are 0.5 and 0.2 mm for row and col.
    col_split_gap :default are 0.5 and 0.2 mm for row and col.
    mask :mask the data in heatmap, the cell with missing values of infinite values will be masked automatically.
    subplot_gap :the gap between subplots, default is 1mm.
    legend :True or False, whether to plot heatmap legend, determined by cmap.
    legend_kws :kws passed to plot legend. If one want to change the outline color and linewidth of cbar:
        for cbar in cm.cbars:
            if isinstance(cbar,matplotlib.colorbar.Colorbar):
                cbar.outline.set_color('white')
                cbar.outline.set_linewidth(2)
                cbar.dividers.set_color('red')
                cbar.dividers.set_linewidth(2)
    plot :whether to plot or not.
    plot_legend :True or False, whether to plot legend, if False, legends can be plot with
        ClusterMapPlotter.plot_legends()
    legend_anchor :str, ax_heatmap or ax, the ax to which legend anchor.
    legend_gap :the columns gap between different legends.
    legend_width: width of the legend, default is 4.5[mm]
    legend_hpad: Horizonal space between the heatmap and legend, default is 2 [mm].
    legend_vpad: Vertical space between the top of legend_anchor and legend, default is 5 [mm].
    legend_side :right of left.
    cmap :default is 'jet', the colormap for heatmap colorbar, see plt.colormaps().
    label :the title (label) that will be shown in heatmap colorbar legend.
    xticklabels_kws :xticklabels or yticklabels kws, such as axis, which, direction, length, width,
        color, pad, labelsize, labelcolor, colors, zorder, bottom, top, left, right, labelbottom, labeltop,
        labelleft, labelright, labelrotation, grid_color, grid_linestyle and so on.
        For more information,see ?matplotlib.axes.Axes.tick_params or ?ax.tick_params.
    yticklabels_kws :the same as xticklabels_kws.
    rasterized :default is False, when the number of rows * number of cols > 100000, rasterized would be suggested
        to be True, otherwise the plot would be very slow.
    heatmap_kws :kws passed to heatmap.

    Returns
    -------
    Class ClusterMapPlotter.
    """
    def __init__(self, data, z_score=None, standard_scale=None,
                 top_annotation=None, bottom_annotation=None, left_annotation=None, right_annotation=None,
                 row_cluster=True, col_cluster=True, row_cluster_method='average', row_cluster_metric='correlation',
                 col_cluster_method='average', col_cluster_metric='correlation',
                 show_rownames=False, show_colnames=False, row_names_side='right', col_names_side='bottom',
                 row_dendrogram=False, col_dendrogram=False, row_dendrogram_size=10, col_dendrogram_size=10,
                 row_split=None, col_split=None, dendrogram_kws=None, tree_kws=None,
                 row_split_order=None,col_split_order=None,row_split_gap=0.5, col_split_gap=0.2, mask=None,
                 subplot_gap=1, legend=True, legend_kws=None,plot=True, plot_legend=True,
                 legend_anchor='auto', legend_gap=4,legend_width=4.5,legend_hpad=2,legend_vpad=5,
                 legend_side='right', cmap='jet', label=None, xticklabels_kws=None, yticklabels_kws=None,
                 rasterized=False,legend_delta_x=None,verbose=1,**heatmap_kws):
        self.heatmap_kws = heatmap_kws if not heatmap_kws is None else {}
        self.data2d = self.format_data(data, z_score, standard_scale)
        self.verbose=verbose
        self.mask = _check_mask(self.data2d, mask)
        self._define_kws(xticklabels_kws, yticklabels_kws)
        self.top_annotation = top_annotation
        self.bottom_annotation = bottom_annotation
        self.left_annotation = left_annotation
        self.right_annotation = right_annotation
        self.row_dendrogram_size = row_dendrogram_size
        self.col_dendrogram_size = col_dendrogram_size
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster
        self.row_cluster_method = row_cluster_method
        self.row_cluster_metric = row_cluster_metric
        self.col_cluster_method = col_cluster_method
        self.col_cluster_metric = col_cluster_metric
        self.show_rownames = show_rownames
        self.show_colnames = show_colnames
        self.row_names_side = row_names_side
        self.col_names_side = col_names_side
        self.row_dendrogram = row_dendrogram
        self.col_dendrogram = col_dendrogram
        self.subplot_gap = subplot_gap
        self.dendrogram_kws = dendrogram_kws
        self.tree_kws = {} if tree_kws is None else tree_kws
        self.row_split = row_split
        self.col_split = col_split
        self.row_split_gap = row_split_gap
        self.col_split_gap = col_split_gap
        self.row_split_order=row_split_order
        self.col_split_order = col_split_order
        self.rasterized = rasterized
        self.legend = legend
        self.legend_kws = legend_kws if not legend_kws is None else {}
        self.legend_side = legend_side
        self.cmap = cmap
        self.label = label if not label is None else 'heatmap'
        self.legend_gap = legend_gap
        self.legend_width = legend_width
        self.legend_hpad = legend_hpad
        self.legend_vpad = legend_vpad
        self.legend_anchor = legend_anchor
        self.legend_delta_x=legend_delta_x
        if plot:
            self.plot()
            if plot_legend:
                if legend_anchor=='auto':
                    if not self.right_annotation is None and self.legend_side=='right':
                        legend_anchor='ax'
                    else:
                        legend_anchor='ax_heatmap'
                if legend_anchor == 'ax_heatmap':
                    self.plot_legends(ax=self.ax_heatmap)
                else:
                    self.plot_legends(ax=self.ax)

    def _define_kws(self, xticklabels_kws, yticklabels_kws):
        self.yticklabels_kws = {} if yticklabels_kws is None else yticklabels_kws
        # self.yticklabels_kws.setdefault('labelrotation', 0)
        self.xticklabels_kws = {} if xticklabels_kws is None else xticklabels_kws
        # self.xticklabels_kws.setdefault('labelrotation', 90)

    def format_data(self, data, z_score=None, standard_scale=None):
        data2d = data.copy()
        self.heatmap_kws.setdefault('vmin', np.nanmin(data.values))
        self.heatmap_kws.setdefault('vmax', np.nanmax(data.values))
        if z_score is not None and standard_scale is not None:
            raise ValueError('Cannot perform both z-scoring and standard-scaling on data')
        if z_score is not None:
            data2d = self.z_score(data, z_score)
        if standard_scale is not None:
            data2d = self.standard_scale(data, standard_scale)
        return data2d

    def _define_gs_ratio(self):
        self.top_heights = []
        self.bottom_heights = []
        self.left_widths = []
        self.right_widths = []
        if self.col_dendrogram:
            self.top_heights.append(self.col_dendrogram_size * mm2inch * self.ax.figure.dpi)
        if self.row_dendrogram:
            self.left_widths.append(self.row_dendrogram_size * mm2inch * self.ax.figure.dpi)
        if not self.top_annotation is None:
            self.top_heights.append(sum(self.top_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.top_heights.append(0)
        if not self.left_annotation is None:
            self.left_widths.append(sum(self.left_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.left_widths.append(0)
        if not self.bottom_annotation is None:
            self.bottom_heights.append(sum(self.bottom_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.bottom_heights.append(0)
        if not self.right_annotation is None:
            self.right_widths.append(sum(self.right_annotation.heights) * mm2inch * self.ax.figure.dpi)
        else:
            self.right_widths.append(0)
        heatmap_h = self.ax.get_window_extent().height - sum(self.top_heights) - sum(self.bottom_heights)
        heatmap_w = self.ax.get_window_extent().width - sum(self.left_widths) - sum(self.right_widths)
        self.heights = [sum(self.top_heights), heatmap_h, sum(self.bottom_heights)]
        self.widths = [sum(self.left_widths), heatmap_w, sum(self.right_widths)]

    def _define_axes(self, subplot_spec=None):
        wspace = self.subplot_gap * mm2inch * self.ax.figure.dpi / (self.ax.get_window_extent().width / 3)
        hspace = self.subplot_gap * mm2inch * self.ax.figure.dpi / (self.ax.get_window_extent().height / 3)

        if subplot_spec is None:
            self.gs = self.ax.figure.add_gridspec(3, 3, width_ratios=self.widths, height_ratios=self.heights,
                                                  wspace=wspace, hspace=hspace)
        else:
            self.gs = matplotlib.gridspec.GridSpecFromSubplotSpec(3, 3, width_ratios=self.widths,
                                                                  height_ratios=self.heights,
                                                                  wspace=wspace, hspace=hspace,
                                                                  subplot_spec=subplot_spec)

        #left -> right, top -> bottom
        self.ax_heatmap = self.ax.figure.add_subplot(self.gs[1, 1])
        self.ax_top = self.ax.figure.add_subplot(self.gs[0, 1], sharex=self.ax_heatmap)
        self.ax_bottom = self.ax.figure.add_subplot(self.gs[2, 1], sharex=self.ax_heatmap)
        self.ax_left = self.ax.figure.add_subplot(self.gs[1, 0], sharey=self.ax_heatmap)
        self.ax_right = self.ax.figure.add_subplot(self.gs[1, 2], sharey=self.ax_heatmap)
        self.ax_heatmap.set_xlim([0, self.data2d.shape[1]])
        self.ax_heatmap.set_ylim([0, self.data2d.shape[0]])
        self.ax.yaxis.label.set_visible(False)
        self.ax_heatmap.yaxis.set_visible(False)
        self.ax_heatmap.xaxis.set_visible(False)
        self.ax.tick_params(axis='both', which='both',
                            left=False, right=False, labelleft=False, labelright=False,
                            top=False, bottom=False, labeltop=False, labelbottom=False)
        self.ax_heatmap.tick_params(axis='both', which='both',
                                    left=False, right=False, top=False, bottom=False,
                                    labeltop=False, labelbottom=False, labelleft=False, labelright=False)
        self.ax.set_axis_off()
        # self.ax.figure.subplots_adjust(left=left,right=right,top=top,bottom=bottom)
        # self.gs.figure.subplots_adjust(0,0,1,1,hspace=0.1,wspace=0)
        # self.ax.margins(0.03, 0.03)
        # _draw_figure(self.ax.figure)
        # self.gs.tight_layout(ax.figure,h_pad=0.0,w_pad=0,pad=0)

    def _define_top_axes(self):
        self.top_gs = None
        if self.top_annotation is None and self.col_dendrogram:
            self.ax_col_dendrogram = self.ax_top
            self.ax_top_annotation = None
        elif self.top_annotation is None and not self.col_dendrogram:
            self.ax_top_annotation = None
            self.ax_col_dendrogram = None
        elif self.col_dendrogram:
            self.top_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, hspace=0, wspace=0,
                                                                      subplot_spec=self.gs[0, 1],
                                                                      height_ratios=[self.col_dendrogram_size,
                                                                                     sum(self.top_annotation.heights)])
            self.ax_top_annotation = self.ax_top.figure.add_subplot(self.top_gs[1, 0])
            self.ax_col_dendrogram = self.ax_top.figure.add_subplot(self.top_gs[0, 0])
        else:
            self.ax_top_annotation = self.ax_top
            self.ax_col_dendrogram = None
        self.ax_top.set_axis_off()

    def _define_left_axes(self):
        self.left_gs = None
        if self.left_annotation is None and self.row_dendrogram:
            self.ax_row_dendrogram = self.ax_left
            self.ax_left_annotation = None
        elif self.left_annotation is None and not self.row_dendrogram:
            self.ax_left_annotation = None
            self.ax_row_dendrogram = None
        elif self.row_dendrogram:
            self.left_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, hspace=0, wspace=0,
                                                                       subplot_spec=self.gs[1, 0],
                                                                       width_ratios=[self.row_dendrogram_size,
                                                                                     sum(self.left_annotation.heights)])
            self.ax_left_annotation = self.ax_left.figure.add_subplot(self.left_gs[0, 1])
            self.ax_row_dendrogram = self.ax_left.figure.add_subplot(self.left_gs[0, 0])
            self.ax_row_dendrogram.set_axis_off()
        else:
            self.ax_left_annotation = self.ax_left
            self.ax_row_dendrogram = None
        self.ax_left.set_axis_off()

    def _define_bottom_axes(self):
        if self.bottom_annotation is None:
            self.ax_bottom_annotation = None
        else:
            self.ax_bottom_annotation = self.ax_bottom
        self.ax_bottom.set_axis_off()

    def _define_right_axes(self):
        if self.right_annotation is None:
            self.ax_right_annotation = None
        else:
            self.ax_right_annotation = self.ax_right
        self.ax_right.set_axis_off()

    @staticmethod
    def z_score(data2d, axis=1):
        """
        Standarize the mean and variance of the data axis

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        normalized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        if axis == 1:
            z_scored = data2d
        else:
            z_scored = data2d.T

        z_scored = (z_scored - z_scored.mean()) / z_scored.std()
        if axis == 1:
            return z_scored
        else:
            return z_scored.T

    @staticmethod
    def standard_scale(data2d, axis=1):
        """
        Divide the data by the difference between the max and min

        Parameters
        ----------
        data2d : pandas.DataFrame
            Data to normalize
        axis : int
            Which axis to normalize across. If 0, normalize across rows, if 1,
            normalize across columns.

        Returns
        -------
        standardized : pandas.DataFrame
            Noramlized data with a mean of 0 and variance of 1 across the
            specified axis.

        """
        # Normalize these values to range from 0 to 1
        if axis == 1:
            standardized = data2d
        else:
            standardized = data2d.T

        subtract = standardized.min()
        standardized = (standardized - subtract) / (
                standardized.max() - standardized.min())
        if axis == 1:
            return standardized
        else:
            return standardized.T

    def calculate_row_dendrograms(self, data):
        if self.row_cluster:
            self.dendrogram_row = DendrogramPlotter(data, linkage=None, axis=0,
                                                    metric=self.row_cluster_metric, method=self.row_cluster_method,
                                                    label=False, rotate=True, dendrogram_kws=self.dendrogram_kws)
        if not self.ax_row_dendrogram is None:
            self.ax_row_dendrogram.set_axis_off()
        # despine(ax=self.ax_row_dendrogram, bottom=True, left=True, top=True, right=True)
        # self.ax_col_dendrogram.spines['top'].set_visible(False)

    def calculate_col_dendrograms(self, data):
        if self.col_cluster:
            self.dendrogram_col = DendrogramPlotter(data, linkage=None, axis=1,
                                                    metric=self.col_cluster_metric, method=self.col_cluster_method,
                                                    label=False, rotate=False, dendrogram_kws=self.dendrogram_kws)
            # self.dendrogram_col.plot(ax=self.ax_col_dendrogram)
        # despine(ax=self.ax_col_dendrogram, bottom=True, left=True, top=True, right=True)
        if not self.ax_col_dendrogram is None:
            self.ax_col_dendrogram.set_axis_off()

    def _reorder_rows(self):
        if self.verbose >= 1:
            print("Reordering rows..")
        if self.row_split is None and self.row_cluster:
            self.calculate_row_dendrograms(self.data2d)  # xind=self.dendrogram_row.reordered_ind
            self.row_order = [self.dendrogram_row.dendrogram['ivl']]  # self.data2d.iloc[:, xind].columns.tolist()
            return None
        elif isinstance(self.row_split, int) and self.row_cluster:
            self.calculate_row_dendrograms(self.data2d)
            self.row_clusters = pd.Series(hierarchy.fcluster(self.dendrogram_row.linkage, t=self.row_split,
                                                             criterion='maxclust'),
                                          index=self.data2d.index.tolist()).to_frame(name='cluster')\
                .groupby('cluster').apply(lambda x: x.index.tolist()).to_dict()
            #index=self.dendrogram_row.dendrogram['ivl']).to_frame(name='cluster')

        elif isinstance(self.row_split, (pd.Series, pd.DataFrame)):
            if isinstance(self.row_split, pd.Series):
                self.row_split = self.row_split.to_frame(name=self.row_split.name)
            cols = self.row_split.columns.tolist()
            row_clusters = self.row_split.groupby(cols).apply(lambda x: x.index.tolist())
            if self.row_split_order:
                row_clusters=row_clusters.loc[self.row_split_order]
            self.row_clusters=row_clusters.to_dict()
        elif not self.row_cluster:
            self.row_order = [self.data2d.index.tolist()]
            return None
        else:
            raise TypeError("row_split must be integar or dataframe or series")

        self.row_order = []
        self.dendrogram_rows = []
        for i, cluster in enumerate(self.row_clusters):
            rows = self.row_clusters[cluster]
            if len(rows) <= 1:
                self.row_order.append(rows)
                self.dendrogram_rows.append(None)
                continue
            if self.row_cluster:
                self.calculate_row_dendrograms(self.data2d.loc[rows])
                self.dendrogram_rows.append(self.dendrogram_row)
                self.row_order.append(self.dendrogram_row.dendrogram['ivl'])
            else:
                self.row_order.append(rows)

    def _reorder_cols(self):
        if self.verbose >= 1:
            print("Reordering cols..")
        if self.col_split is None and self.col_cluster:
            self.calculate_col_dendrograms(self.data2d)
            self.col_order = [self.dendrogram_col.dendrogram['ivl']]  # self.data2d.iloc[:, xind].columns.tolist()
            return None
        elif isinstance(self.col_split, int) and self.col_cluster:
            self.calculate_col_dendrograms(self.data2d)
            self.col_clusters = pd.Series(hierarchy.fcluster(self.dendrogram_col.linkage, t=self.col_split,
                                                             criterion='maxclust'),
                                          index=self.data2d.columns.tolist()).to_frame(name='cluster')\
                .groupby('cluster').apply(lambda x: x.index.tolist()).to_dict()
            #index=self.dendrogram_col.dendrogram['ivl']).to_frame(name='cluster')

        elif isinstance(self.col_split, (pd.Series, pd.DataFrame)):
            if isinstance(self.col_split, pd.Series):
                self.col_split = self.col_split.to_frame(name=self.col_split.name)
            cols = self.col_split.columns.tolist()
            col_clusters = self.col_split.groupby(cols).apply(lambda x: x.index.tolist())
            if self.col_split_order:
                col_clusters=col_clusters.loc[self.col_split_order]
            self.col_clusters=col_clusters.to_dict()
        elif not self.col_cluster:
            self.col_order = [self.data2d.columns.tolist()]
            return None
        else:
            raise TypeError("row_split must be integar or dataframe or series")

        self.col_order = []
        self.dendrogram_cols = []
        for i, cluster in enumerate(self.col_clusters):
            cols = self.col_clusters[cluster]
            if len(cols) <= 1:
                self.col_order.append(cols)
                self.dendrogram_cols.append(None)
                continue
            if self.col_cluster:
                self.calculate_col_dendrograms(self.data2d.loc[:, cols])
                self.dendrogram_cols.append(self.dendrogram_col)
                self.col_order.append(self.dendrogram_col.dendrogram['ivl'])
            else:
                self.col_order.append(cols)

    def plot_dendrograms(self, row_order, col_order):
        rcmap = self.tree_kws.pop('row_cmap', None)
        ccmap = self.tree_kws.pop('col_cmap', None)
        tree_kws = self.tree_kws.copy()

        if self.row_cluster and self.row_dendrogram:
            if self.left_annotation is None:
                gs = self.gs[1, 0]
            else:
                gs = self.left_gs[0, 0]
            self.row_dendrogram_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(len(row_order), 1, hspace=self.hspace,
                                                                                 wspace=0, subplot_spec=gs,
                                                                                 height_ratios=[len(rows) for rows
                                                                                                in row_order])
            self.ax_row_dendrogram_axes = []
            for i in range(len(row_order)):
                ax1 = self.ax_row_dendrogram.figure.add_subplot(self.row_dendrogram_gs[i, 0])
                ax1.set_axis_off()
                self.ax_row_dendrogram_axes.append(ax1)

            try:
                if rcmap is None:
                    colors = ['black'] * len(self.dendrogram_rows)
                else:
                    colors = [plt.get_cmap(rcmap)(i) for i in range(len(self.dendrogram_rows))]
                for ax_row_dendrogram, dendrogram_row, color in zip(self.ax_row_dendrogram_axes, self.dendrogram_rows,
                                                                    colors):
                    if dendrogram_row is None:
                        continue
                    tree_kws['colors'] = [color] * len(dendrogram_row.dendrogram['ivl'])
                    dendrogram_row.plot(ax=ax_row_dendrogram, tree_kws=tree_kws)
            except:
                self.dendrogram_row.plot(ax=self.ax_row_dendrogram, tree_kws=self.tree_kws)

        if self.col_cluster and self.col_dendrogram:
            if self.top_annotation is None:
                gs = self.gs[0, 1]
            else:
                gs = self.top_gs[0, 0]
            self.col_dendrogram_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, len(col_order), hspace=0,
                                                                                 wspace=self.wspace, subplot_spec=gs,
                                                                                 width_ratios=[len(cols) for cols
                                                                                               in col_order])
            self.ax_col_dendrogram_axes = []
            for i in range(len(col_order)):
                ax1 = self.ax_col_dendrogram.figure.add_subplot(self.col_dendrogram_gs[0, i])
                ax1.set_axis_off()
                self.ax_col_dendrogram_axes.append(ax1)

            try:
                if ccmap is None:
                    colors = ['black'] * len(self.dendrogram_cols)
                else:
                    colors = [plt.get_cmap(ccmap)(i) for i in range(len(self.dendrogram_cols))]
                for ax_col_dendrogram, dendrogram_col, color in zip(self.ax_col_dendrogram_axes, self.dendrogram_cols,
                                                                    colors):
                    if dendrogram_col is None:
                        continue
                    tree_kws['colors'] = [color] * len(dendrogram_col.dendrogram['ivl'])
                    dendrogram_col.plot(ax=ax_col_dendrogram, tree_kws=tree_kws)
            except:
                self.dendrogram_col.plot(ax=self.ax_col_dendrogram, tree_kws=self.tree_kws)

    def plot_matrix(self, row_order, col_order):
        if self.verbose >= 1:
            print("Plotting matrix..")
        nrows = len(row_order)
        ncols = len(col_order)
        self.wspace = self.col_split_gap * mm2inch * self.ax.figure.dpi / (
                self.ax_heatmap.get_window_extent().width / nrows)  # 1mm=mm2inch inch
        self.hspace = self.row_split_gap * mm2inch * self.ax.figure.dpi / (
                self.ax_heatmap.get_window_extent().height / ncols)
        self.heatmap_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows, ncols, hspace=self.hspace,
                                                                      wspace=self.wspace,
                                                                      subplot_spec=self.gs[1, 1],
                                                                      height_ratios=[len(rows) for rows in row_order],
                                                                      width_ratios=[len(cols) for cols in col_order])

        annot = self.heatmap_kws.pop("annot", None)
        if annot is None or annot is False:
            pass
        else:
            if isinstance(annot, bool):
                annot_data = self.data2d
            else:
                annot_data = annot.copy()
                if annot_data.shape != self.data2d.shape:
                    err = "`data` and `annot` must have same shape."
                    raise ValueError(err)

        self.heatmap_axes = np.empty(shape=(nrows, ncols), dtype=object)
        # if nrows > 1 or ncols > 1:
        self.ax_heatmap.set_axis_off()
        for i, rows in enumerate(row_order):
            for j, cols in enumerate(col_order):
                # print(i,j)
                ax1 = self.ax_heatmap.figure.add_subplot(self.heatmap_gs[i, j],
                                                        sharex=self.heatmap_axes[0, j],
                                                        sharey=self.heatmap_axes[i, 0])
                ax1.set_xlim([0, len(rows)])
                ax1.set_ylim([0, len(cols)])
                annot1 = None if annot is None else annot_data.loc[rows, cols]
                heatmap(self.data2d.loc[rows, cols], ax=ax1, cbar=False, cmap=self.cmap,
                        cbar_kws=None, mask=self.mask.loc[rows, cols], rasterized=self.rasterized,
                        xticklabels='auto', yticklabels='auto', annot=annot1, **self.heatmap_kws)
                self.heatmap_axes[i, j] = ax1
                ax1.yaxis.label.set_visible(False)
                ax1.xaxis.label.set_visible(False)
                ax1.tick_params(left=False, right=False, labelleft=False, labelright=False,
                                top=False, bottom=False, labeltop=False, labelbottom=False)

    def set_axes_labels_kws(self):
        # ax.set_xticks(ticks=np.arange(1, self.nrows + 1, 1), labels=self.plot_data.index.tolist())
        self.ax_heatmap.yaxis.set_tick_params(**self.yticklabels_kws)
        self.ax_heatmap.xaxis.set_tick_params(**self.xticklabels_kws)
        # self.ax_heatmap.tick_params(axis='both', which='both',
        #                             left=False, right=False, top=False, bottom=False)
        self.yticklabels = []
        self.xticklabels = []
        if (self.show_rownames and self.left_annotation is None and not self.row_dendrogram) \
                and ((not self.right_annotation is None) or (
                self.right_annotation is None and self.row_names_side == 'left')):  # tick left
            self.yticklabels_kws.setdefault('labelrotation', 0)
            for i in range(self.heatmap_axes.shape[0]):
                self.heatmap_axes[i, 0].yaxis.set_visible(True)
                self.heatmap_axes[i, 0].tick_params(axis='y', which='both', left=False, labelleft=True)
                self.heatmap_axes[i, 0].yaxis.set_tick_params(**self.yticklabels_kws)  # **self.ticklabels_kws
                plt.setp(self.heatmap_axes[i, 0].get_yticklabels(), rotation_mode='anchor',
                         ha='right', va='center')
                self.yticklabels.extend(self.heatmap_axes[i, 0].get_yticklabels())
        elif self.show_rownames and self.right_annotation is None:  # tick right
            self.yticklabels_kws.setdefault('labelrotation', 0)
            for i in range(self.heatmap_axes.shape[0]):
                self.heatmap_axes[i, -1].yaxis.tick_right()  # set_ticks_position('right')
                self.heatmap_axes[i, -1].yaxis.set_visible(True)
                self.heatmap_axes[i, -1].tick_params(axis='y', which='both', right=False, labelright=True)
                self.heatmap_axes[i, -1].yaxis.set_tick_params(**self.yticklabels_kws)
                plt.setp(self.heatmap_axes[i, -1].get_yticklabels(), rotation_mode='anchor',
                         ha='left', va='center')
                self.yticklabels.extend(self.heatmap_axes[i, -1].get_yticklabels())
        if self.show_colnames and self.top_annotation is None and not self.col_dendrogram and \
                ((not self.bottom_annotation is None) or (
                        self.bottom_annotation is None and self.col_names_side == 'top')):
            self.xticklabels_kws.setdefault('labelrotation', 90)
            for j in range(self.heatmap_axes.shape[1]):
                self.heatmap_axes[0, j].xaxis.tick_top()  # ticks
                self.heatmap_axes[0, j].xaxis.set_visible(True)
                self.heatmap_axes[0, j].tick_params(axis='x', which='both', top=False, labeltop=True)
                self.heatmap_axes[0, j].xaxis.set_tick_params(**self.xticklabels_kws)
                plt.setp(self.heatmap_axes[0, j].get_xticklabels(), rotation_mode = 'anchor',
                         ha = 'left',va='center') #rotation=90,ha=left is bottom, va is horizonal
                self.xticklabels.extend(self.heatmap_axes[0, j].get_xticklabels())
        elif self.show_colnames and self.bottom_annotation is None:  # tick bottom
            self.xticklabels_kws.setdefault('labelrotation', -90)
            for j in range(self.heatmap_axes.shape[1]):
                self.heatmap_axes[-1, j].xaxis.tick_bottom()  # ticks
                self.heatmap_axes[-1, j].xaxis.set_visible(True)
                self.heatmap_axes[-1, j].tick_params(axis='x', which='both', bottom=False, labelbottom=True)
                self.heatmap_axes[-1, j].xaxis.set_tick_params(**self.xticklabels_kws)
                plt.setp(self.heatmap_axes[-1, j].get_xticklabels(), rotation_mode='anchor',
                         ha='left', va='center')
                self.xticklabels.extend(self.heatmap_axes[-1, j].get_xticklabels())
        # self.ax.figure.subplots_adjust(left=0.03, right=2, bottom=0.03, top=0.97)
        # self.ax.margins(x=0.1,y=0.1)
        # tight_params = dict(h_pad=.1, w_pad=.1)
        # self.ax.figure.tight_layout(**tight_params)
        # _draw_figure(self.ax.figure)

    def collect_legends(self):
        if self.verbose >= 1:
            print("Collecting legends..")
        self.legend_list = []
        self.label_max_width = 0
        for annotation in [self.top_annotation, self.bottom_annotation, self.left_annotation, self.right_annotation]:
            if not annotation is None:
                annotation.collect_legends()
                if annotation.plot_legend and len(annotation.legend_list) > 0:
                    self.legend_list.extend(annotation.legend_list)
                # print(annotation.label_max_width,self.label_max_width)
                if annotation.label_max_width > self.label_max_width:
                    self.label_max_width = annotation.label_max_width
        if self.legend:
            vmax = self.heatmap_kws.get('vmax',np.nanmax(self.data2d[self.data2d != np.inf]))
            vmin = self.heatmap_kws.get('vmin',np.nanmin(self.data2d[self.data2d != -np.inf]))
            self.legend_kws.setdefault('vmin', round(vmin, 2))
            self.legend_kws.setdefault('vmax', round(vmax, 2))
            self.legend_list.append([self.cmap, self.label, self.legend_kws, 4])
            heatmap_label_max_width = max([label.get_window_extent().width for label in self.yticklabels]) if len(
                self.yticklabels) > 0 else 0
            # heatmap_label_max_height = max([label.get_window_extent().height for label in self.yticklabels]) if len(
            #     self.yticklabels) > 0 else 0
            if heatmap_label_max_width >= self.label_max_width or self.legend_anchor == 'ax_heatmap':
                self.label_max_width = heatmap_label_max_width * 1.1
            if len(self.legend_list) > 1:
                self.legend_list = sorted(self.legend_list, key=lambda x: x[3])

    def plot_legends(self, ax=None):
        if self.verbose >= 1:
            print("Plotting legends..")
        if len(self.legend_list) > 0:
            if self.legend_side == 'right' and not self.right_annotation is None:
                space = self.label_max_width
            elif self.legend_side == 'right' and self.show_rownames and self.row_names_side=='right':
                space = self.label_max_width
            else:
                space=0
            # if self.right_annotation:
            #     space+=sum(self.right_widths)
            legend_hpad = self.legend_hpad * mm2inch * self.ax.figure.dpi
            self.legend_axes, self.cbars,self.boundry = \
                plot_legend_list(self.legend_list, ax=ax, space=space + legend_hpad,
                                  legend_side=self.legend_side, gap=self.legend_gap,
                                  delta_x=self.legend_delta_x,legend_width=self.legend_width,
                                 legend_vpad=self.legend_vpad)

    def plot(self, ax=None, subplot_spec=None, row_order=None, col_order=None):
        if self.verbose >= 1:
            print("Starting plotting..")
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self._define_gs_ratio()
        self._define_axes(subplot_spec)
        self._define_top_axes()
        self._define_left_axes()
        self._define_bottom_axes()
        self._define_right_axes()
        if row_order is None:
            if self.verbose >= 1:
                print("Starting calculating row orders..")
            self._reorder_rows()
            row_order = self.row_order
        if col_order is None:
            if self.verbose >= 1:
                print("Starting calculating col orders..")
            self._reorder_cols()
            col_order = self.col_order
        self.plot_matrix(row_order=row_order, col_order=col_order)
        if not self.top_annotation is None:
            gs = self.gs[0, 1] if not self.col_dendrogram else self.top_gs[1, 0]
            self.top_annotation.plot_annotations(ax=self.ax_top_annotation, subplot_spec=gs,
                                                 idxs=col_order, wspace=self.wspace)
        if not self.bottom_annotation is None:
            self.bottom_annotation.plot_annotations(ax=self.ax_bottom_annotation, subplot_spec=self.gs[2, 1],
                                                    idxs=col_order, wspace=self.wspace)
        if not self.left_annotation is None:
            gs = self.gs[1, 0] if not self.row_dendrogram else self.left_gs[0, 1]
            self.left_annotation.plot_annotations(ax=self.ax_left_annotation, subplot_spec=gs,
                                                  idxs=row_order, hspace=self.hspace)
        if not self.right_annotation is None:
            self.right_annotation.plot_annotations(ax=self.ax_right_annotation, subplot_spec=self.gs[1, 2],
                                                   idxs=row_order, hspace=self.hspace)
        if self.row_cluster or self.col_cluster:
            if self.row_dendrogram or self.col_dendrogram:
                self.plot_dendrograms(row_order, col_order)
        self.set_axes_labels_kws()
        self.collect_legends()
        # _draw_figure(self.ax_heatmap.figure)
        return self.ax

    def tight_layout(self, **tight_params):
        tight_params = dict(h_pad=.02, w_pad=.02) if tight_params is None else tight_params
        left = 0
        right = 1
        if self.legend and self.legend_side == 'right':
            right = self.boundry
        elif self.legend and self.legend_side == 'left':
            left = self.boundry
        tight_params.setdefault("rect", [left, 0, right, 1])
        self.ax.figure.tight_layout(**tight_params)

    def set_height(self, fig, height):
        matplotlib.figure.Figure.set_figheight(fig, height)  # convert mm to inches

    def set_width(self, fig, width):
        matplotlib.figure.Figure.set_figwidth(fig, width)  # convert mm to inches

def composite(cmlist=None, main=0, ax=None, axis=1, row_gap=15, col_gap=15,
              legend_side='right', legend_gap=3, legend_y=0.8, legend_hpad=None,
              legend_width=4.5):
    """
    Assemble multiple ClusterMapPlotter objects vertically or horizontally together.

    Parameters
    ----------
    cmlist: a list of ClusterMapPlotter (with plot=False).
    axis: 1 for columns (align the cmlist horizontally), 0 for rows (vertically).
    main: use which as main ClusterMapPlotter, will influence row/col order. main is the index
        of cmlist.
    row/col_gap, the row or columns gap between subplots, unit is mm [15].
    legend_side: right,left [right].
    legend_gap, row gap between two legends, unit is mm.

    Returns
    -------
    legend_axes

    """
    if ax is None:
        ax = plt.gca()
    n = len(cmlist)
    wspace, hspace = 0, 0
    if axis == 1:  # horizontally
        wspace = col_gap * mm2inch * ax.figure.dpi / (ax.get_window_extent().width / n)
        nrows = 1
        ncols = n
        width_ratios = [cm.data2d.shape[1] for cm in cmlist]
        height_ratios = None
    else:  # vertically
        hspace = row_gap * mm2inch * ax.figure.dpi / (ax.get_window_extent().height / n)
        nrows = n
        ncols = 1
        width_ratios = None
        height_ratios = [cm.data2d.shape[0] for cm in cmlist]
    gs = ax.figure.add_gridspec(nrows, ncols, width_ratios=width_ratios,
                                height_ratios=height_ratios,
                                wspace=wspace, hspace=hspace)
    axes = []
    for i, cm in enumerate(cmlist):
        sharex = axes[0] if axis == 0 and i > 0 else None
        sharey = axes[0] if axis == 1 and i > 0 else None
        gs1 = gs[i, 0] if axis == 0 else gs[0, i]
        ax1 = ax.figure.add_subplot(gs1, sharex=sharex, sharey=sharey)
        ax1.set_axis_off()
        axes.append(ax1)
    cm_1 = cmlist[main]
    ax1 = axes[main]
    gs1 = gs[main, 0] if axis == 0 else gs[0, main]
    cm_1.plot(ax=ax1, subplot_spec=gs1, row_order=None, col_order=None)
    legend_list = cm_1.legend_list
    legend_names = [L[1] for L in legend_list]
    label_max_width = ax.figure.get_window_extent().width * cm_1.label_max_width / cm_1.ax.figure.get_window_extent().width
    for i, cm in enumerate(cmlist):
        if i == main:
            continue
        gs1 = gs[i, 0] if axis == 0 else gs[0, i]
        cm.plot(ax=axes[i], subplot_spec=gs1, row_order=cm_1.row_order, col_order=cm_1.col_order)
        for L in cm.legend_list:
            if L[1] not in legend_names:
                legend_names.append(L[1])
                legend_list.append(L)
        w = ax.figure.get_window_extent().width * cm.label_max_width / cm.ax.figure.get_window_extent().width
        if w > label_max_width:
            label_max_width = w
    if len(legend_list) == 0:
        return None
    legend_list = sorted(legend_list, key=lambda x: x[3])
    if legend_hpad is None:
        space = col_gap * mm2inch * ax.figure.dpi + label_max_width
    else:
        space = legend_hpad * ax.figure.dpi / 72
    legend_axes, cbars,boundry = \
        plot_legend_list(legend_list, ax=ax, space=space,
                        legend_side=legend_side, gap=legend_gap,
                         y0=legend_y,legend_width=legend_width)
    ax.set_axis_off()
    # import pdb;
    # pdb.set_trace()
    return legend_axes

if __name__ == "__main__":
    pass
