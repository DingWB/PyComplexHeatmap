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
from .utils import mm2inch

from .utils import (
    _check_mask,
    _calculate_luminance,
    despine,
    _draw_figure,
    axis_ticklabels_overlap,
    _skip_ticks,
    _auto_ticks,
    _index_to_label,
    _index_to_ticklabels,
    plot_legend_list,
    get_colormap
)
# =============================================================================
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
            self.xticks, self.xticklabels = _skip_ticks(xticklabels, xtickevery)

        if not len(yticklabels):
            self.yticks = []
            self.yticklabels = []
        elif isinstance(yticklabels, str) and yticklabels == "auto":
            self.yticks = "auto"
            self.yticklabels = _index_to_ticklabels(data.index)
        else:
            self.yticks, self.yticklabels = _skip_ticks(yticklabels, ytickevery)

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
                    self.cmap = get_colormap('turbo').copy()
                except:
                    self.cmap = get_colormap('turbo')
            else:
                try:
                    self.cmap = get_colormap('exp1').copy()
                except:
                    self.cmap = get_colormap('exp1')
        elif isinstance(cmap, str):
            try:
                self.cmap = get_colormap(cmap).copy()
            except:
                self.cmap = get_colormap(cmap)
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
            xticks, xticklabels = _auto_ticks(ax, self.xticklabels, 0)
        else:
            xticks, xticklabels = self.xticks, self.xticklabels

        if isinstance(self.yticks, str) and self.yticks == "auto":
            yticks, yticklabels = _auto_ticks(ax, self.yticklabels, 1)
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
# =============================================================================
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
    data: dataframe
        pandas dataframe
    xlabel / ylabel: bool
        True, False, or list of xlabels
    xlabel_side / ylabel_side: str
        bottom or top
    vmax, vmin: float
        the maximal and minimal values for cmap colorbar.
    center, robust:
        the same as seaborn.heatmap
    xlabel_kws / ylabel_kws:
        parameter from matplotlib.axis.XAxis.label.properties()

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
# =============================================================================
def plot_heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
            annot=None, fmt=".2g",annot_kws=None,
            xticklabels=True, yticklabels=True, mask=None, na_col='white', ax=None,
            linewidths=0, linecolor="white",**kwargs):
    """
    Plot heatmap.
        heatmap(self.data2d.loc[rows, cols], ax=ax1,cmap=self.cmap,
                        mask=self.mask.loc[rows, cols], rasterized=self.rasterized,
                        xticklabels='auto', yticklabels='auto', annot=annot1, **self.kwargs)
    Parameters
    ----------
    data: dataframe
        pandas dataframe
    vmax, vmin: float
        the maximal and minimal values for cmap colorbar.
    center, robust:
        the same as seaborn.heatmap
    annot: bool
        whether to add annotation for values
    fmt: str
        annotation format.
    anno_kws: dict
        passed to ax.text
    xticklabels,yticklabels: bool
        whether to show ticklabels

    """

    if isinstance(data, pd.DataFrame):
        plot_data = data.values
    else:
        plot_data = np.asarray(data)
        data = pd.DataFrame(plot_data)
    # Validate the mask and convert to DataFrame
    mask = _check_mask(data, mask)
    plot_data = np.ma.masked_where(np.asarray(mask), plot_data)
    # Get good names for the rows and columns
    if xticklabels is False:
        xticks = []
        xticklabels = []
    else:
        xticks = "auto"
        xticklabels = _index_to_ticklabels(data.columns)

    if yticklabels is False:
        yticks = []
        yticklabels = []
    else:
        yticks = "auto"
        yticklabels = _index_to_ticklabels(data.index)

    # Determine good default values for the colormapping
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

    # Choose default colormaps if not provided
    if isinstance(cmap, str):
        try:
            cmap = get_colormap(cmap).copy()
        except:
            cmap = get_colormap(cmap)

    cmap.set_bad(color=na_col)  # set the color for NaN values
    # Recenter a divergent colormap
    if center is not None:
        # bad = cmap(np.ma.masked_invalid([np.nan]))[0]  # set the first color as the na_color
        under = cmap(-np.inf)
        over = cmap(np.inf)
        under_set = under != cmap(0)
        over_set = over != cmap(cmap.N - 1)

        vrange = max(vmax - center, center - vmin)
        normlize = matplotlib.colors.Normalize(center - vrange, center + vrange)
        cmin, cmax = normlize([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap = matplotlib.colors.ListedColormap(cmap(cc))
        # self.cmap.set_bad(bad)
        if under_set:
            cmap.set_under(under)  # set the color of -np.inf as the color for low out-of-range values.
        if over_set:
            cmap.set_over(over)

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

    if annot_kws is None:
        annot_kws = {}

    # Add the pcolormesh kwargs here
    kwargs["linewidths"] = linewidths
    kwargs["edgecolor"] = linecolor

    # Draw the plot and return the Axes
    despine(ax=ax, left=True, bottom=True)
    if "norm" not in kwargs:
        kwargs.setdefault("vmin", vmin)
        kwargs.setdefault("vmax", vmax)

    # Draw the heatmap
    mesh = ax.pcolormesh(plot_data, cmap=cmap, **kwargs)
    # Set the axis limits
    ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0]))
    # Invert the y axis to show the plot in matrix form
    ax.invert_yaxis()  # from top to bottom

    # Add row and column labels
    if isinstance(xticks, str) and xticks == "auto":
        xticks, xticklabels = _auto_ticks(ax, xticklabels, 0)

    if isinstance(yticks, str) and yticks == "auto":
        yticks, yticklabels = _auto_ticks(ax, yticklabels, 1)

    ax.set(xticks=xticks, yticks=yticks)
    xtl = ax.set_xticklabels(xticklabels)
    ytl = ax.set_yticklabels(yticklabels, rotation="vertical")

    _draw_figure(ax.figure)
    if axis_ticklabels_overlap(xtl):
        plt.setp(xtl, rotation="vertical")
    if axis_ticklabels_overlap(ytl):
        plt.setp(ytl, rotation="horizontal")

    # Annotate the cells with the formatted values
    if annot:
        mesh.update_scalarmappable()
        height, width = annot_data.shape
        xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
        for x, y, m, color, val in zip(xpos.flat, ypos.flat,
                                       mesh.get_array(), mesh.get_facecolors(),
                                       annot_data.flat):
            if m is not np.ma.masked:
                lum = _calculate_luminance(color)
                text_color = ".15" if lum > .408 else "w"
                annotation = ("{:" + fmt + "}").format(val)
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(annot_kws)
                ax.text(x, y, annotation, **text_kwargs)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return ax
# =============================================================================
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
        # if self.axis==0: #TODO
        #     ax.invert_yaxis()  # 20230227 fix the bug for inverse order of row dendrogram

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
# =============================================================================
class ClusterMapPlotter():
    """
    Clustermap (Heatmap) plotter.
    Plot heatmap / clustermap with annotation and legends.

    Parameters
    ----------
    data : dataframe
        pandas dataframe or numpy array.
    z_score : int
        whether to perform z score scale, either 0 for rows or 1 for columns, after scale,
        value range would be from -1 to 1.
    standard_scale : int
        either 0 for rows or 1 for columns, after scale,value range would be from 0 to 1.
    top_annotation : annotation: class of HeatmapAnnotation.
    bottom_annotation : class AnnotationBase
        the same as top_annotation.
    left_annotation :class AnnotationBase
        the same as top_annotation.
    right_annotation :class AnnotationBase
        the same as top_annotation.
    row_cluster :bool
        whether to perform cluster on rows/columns.
    col_cluster :bool
        whether to perform cluster on rows/columns.
    row_cluster_method :str
        cluster method for row/columns linkage, such single, complete, average,weighted,
        centroid, median, ward. see scipy.cluster.hierarchy.linkage or
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) for detail.
    row_cluster_metric : str
        Pairwise distances between observations in n-dimensional space for row/columns,
        such euclidean, minkowski, cityblock, seuclidean, cosine, correlation, hamming, jaccard,
        chebyshev, canberra, braycurtis, mahalanobis, kulsinski et.al.
        centroid, median, ward. see scipy.cluster.hierarchy.linkage or
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html
    col_cluster_method :str
        same as row_cluster_method
    col_cluster_metric :str
        same as row_cluster_metric
    show_rownames :bool
        True (default) or False, whether to show row ticklabels.
    show_colnames : bool
        True of False, same as show_rownames.
    row_names_side :str
        right or left.
    col_names_side :str
        top or bottom.
    row_dendrogram :bool
        True or False, whether to show dendrogram.
    col_dendrogram :bool
        True or False, whether to show dendrogram.
    row_dendrogram_size :int
        default is 10mm.
    col_dendrogram_size :int
        default is 10mm.
    row_split :int or pd.Series or pd.DataFrame
        number of cluster for hierarchical clustering or pd.Series or pd.DataFrame,
        used to split rows or rows into subplots.
    col_split :int or pd.Series or pd.DataFrame
        int or pd.Series or pd.DataFrame, used to split rows or columns into subplots.
    dendrogram_kws :dict
        kws passed to hierarchy.dendrogram.
    tree_kws :dict
        kws passed to DendrogramPlotter.plot()
    row_split_gap :float
        default are 0.5 and 0.2 mm for row and col.
    col_split_gap :float
        default are 0.5 and 0.2 mm for row and col.
    mask :dataframe or array
        mask the data in heatmap, the cell with missing values of infinite values will be masked automatically.
    subplot_gap :float
        the gap between subplots, default is 1mm.
    legend :bool
        True or False, whether to plot heatmap legend, determined by cmap.
    legend_kws :dict
        kws passed to plot legend. If one want to change the outline color and linewidth of cbar:
        ```
        for cbar in cm.cbars:
            if isinstance(cbar,matplotlib.colorbar.Colorbar):
                cbar.outline.set_color('white')
                cbar.outline.set_linewidth(2)
                cbar.dividers.set_color('red')
                cbar.dividers.set_linewidth(2)
        ```
    plot :bool
        whether to plot or not.
    plot_legend :bool
        True or False, whether to plot legend, if False, legends can be plot with
        ClusterMapPlotter.plot_legends()
    legend_anchor :str
        ax_heatmap or ax, the ax to which legend anchor.
    legend_gap :float
        the columns gap between different legends.
    legend_width: float [mm]
        width of the legend, default is None (infer from data automatically)
    legend_hpad: float
        Horizonal space between the heatmap and legend, default is 2 [mm].
    legend_vpad: float
        Vertical space between the top of legend_anchor and legend, default is 5 [mm].
    legend_side :str
        right of left.
    cmap :str
        default is 'jet', the colormap for heatmap colorbar, see plt.colormaps().
    label :str
        the title (label) that will be shown in heatmap colorbar legend.
    xticklabels_kws :dict
        xticklabels or yticklabels kws, such as axis, which, direction, length, width,
        color, pad, labelsize, labelcolor, colors, zorder, bottom, top, left, right, labelbottom, labeltop,
        labelleft, labelright, labelrotation, grid_color, grid_linestyle and so on.
        For more information,see ?matplotlib.axes.Axes.tick_params or ?ax.tick_params.
    yticklabels_kws :dict
        the same as xticklabels_kws.
    rasterized :bool
        default is False, when the number of rows * number of cols > 100000, rasterized would be suggested
        to be True, otherwise the plot would be very slow.
    kwargs :kws passed to heatmap.

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
                 row_split_order=None, col_split_order=None, row_split_gap=0.5, col_split_gap=0.2, mask=None,
                 subplot_gap=1, legend=True, legend_kws=None, plot=True, plot_legend=True,
                 legend_anchor='auto', legend_gap=7, legend_width=None, legend_hpad=1, legend_vpad=5,
                 legend_side='right', cmap='jet', label=None, xticklabels_kws=None, yticklabels_kws=None,
                 rasterized=False, legend_delta_x=None, verbose=1, **kwargs):
        self.kwargs = kwargs if not kwargs is None else {}
        self.data2d = self.format_data(data, mask, z_score, standard_scale)
        self.verbose=verbose
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

        self.post_processing()

    def _define_kws(self, xticklabels_kws, yticklabels_kws):
        self.yticklabels_kws = {} if yticklabels_kws is None else yticklabels_kws
        # self.yticklabels_kws.setdefault('labelrotation', 0)
        self.xticklabels_kws = {} if xticklabels_kws is None else xticklabels_kws
        # self.xticklabels_kws.setdefault('labelrotation', 90)

    def format_data(self, data, mask=None, z_score=None, standard_scale=None):
        data2d = data.copy()
        self.kwargs.setdefault('vmin', np.nanmin(data.values))
        self.kwargs.setdefault('vmax', np.nanmax(data.values))
        if z_score is not None and standard_scale is not None:
            raise ValueError('Cannot perform both z-scoring and standard-scaling on data')
        if z_score is not None:
            data2d = self.z_score(data, z_score)
        if standard_scale is not None:
            data2d = self.standard_scale(data, standard_scale)
        self.mask = _check_mask(data2d, mask)
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
            if self.row_split_order is None:
                if len(cols)==1:
                    # calculate row_split_order using the mean across all samples in this group of
                    # values of mean values across all samples
                    self.row_split_order = row_clusters.apply(lambda x: self.data2d.loc[x].mean(axis=1).mean())\
                        .sort_values(ascending=False).index.tolist()
                else:
                    self.row_split_order=row_clusters.sort_index().index.tolist()
            self.row_clusters = row_clusters.loc[self.row_split_order].to_dict()
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
            if self.col_split_order is None:
                if len(cols)==1:
                    # calculate col_split_order using the mean across all samples in this group of
                    # values of mean values across all samples
                    self.col_split_order = col_clusters.apply(lambda x: self.data2d.loc[:,x].mean().mean())\
                        .sort_values(ascending=False).index.tolist()
                else:
                    self.col_split_order=col_clusters.sort_index().index.tolist()
            self.col_clusters = col_clusters.loc[self.col_split_order].to_dict()
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
                    colors = [get_colormap(rcmap)(i) for i in range(len(self.dendrogram_rows))]
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
                    colors = [get_colormap(ccmap)(i) for i in range(len(self.dendrogram_cols))]
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
                self.ax_heatmap.get_window_extent().width / ncols)  # 1mm=mm2inch inch
        self.hspace = self.row_split_gap * mm2inch * self.ax.figure.dpi / (
                self.ax_heatmap.get_window_extent().height / nrows) #height
        self.heatmap_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows, ncols, hspace=self.hspace,
                                                                      wspace=self.wspace,
                                                                      subplot_spec=self.gs[1, 1],
                                                                      height_ratios=[len(rows) for rows in row_order],
                                                                      width_ratios=[len(cols) for cols in col_order])

        annot = self.kwargs.pop("annot", None)
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
                # heatmap(self.data2d.loc[rows, cols], ax=ax1, cbar=False, cmap=self.cmap,
                #         cbar_kws=None, mask=self.mask.loc[rows, cols], rasterized=self.rasterized,
                #         xticklabels='auto', yticklabels='auto', annot=annot1, **self.kwargs)
                plot_heatmap(self.data2d.loc[rows, cols], ax=ax1, cmap=self.cmap,
                        mask=self.mask.loc[rows, cols], rasterized=self.rasterized,
                        xticklabels='auto', yticklabels='auto', annot=annot1, **self.kwargs)
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
            self.row_names_side='left'
            self.yticklabels_kws.setdefault('labelrotation', 0)
            for i in range(self.heatmap_axes.shape[0]):
                self.heatmap_axes[i, 0].yaxis.set_visible(True)
                self.heatmap_axes[i, 0].tick_params(axis='y', which='both', left=False, labelleft=True)
                self.heatmap_axes[i, 0].yaxis.set_tick_params(**self.yticklabels_kws)  # **self.ticklabels_kws
                plt.setp(self.heatmap_axes[i, 0].get_yticklabels(), rotation_mode='anchor',
                         ha='right', va='center')
                self.yticklabels.extend(self.heatmap_axes[i, 0].get_yticklabels())
        elif self.show_rownames and self.right_annotation is None:  # tick right
            self.row_names_side = 'right'
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
            vmax = self.kwargs.get('vmax', np.nanmax(self.data2d[self.data2d != np.inf]))
            vmin = self.kwargs.get('vmin', np.nanmin(self.data2d[self.data2d != -np.inf]))
            self.legend_kws.setdefault('vmin', round(vmin, 2))
            self.legend_kws.setdefault('vmax', round(vmax, 2))
            self.legend_list.append([self.cmap, self.label, self.legend_kws, 4,'cmap'])
            heatmap_label_max_width = max([label.get_window_extent().width for label in self.yticklabels]) if len(
                self.yticklabels) > 0 and self.row_names_side=='right' else 0
            # heatmap_label_max_height = max([label.get_window_extent().height for label in self.yticklabels]) if len(
            #     self.yticklabels) > 0 else 0
            if heatmap_label_max_width >= self.label_max_width or self.legend_anchor == 'ax_heatmap':
                self.label_max_width = heatmap_label_max_width #* 1.1
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
            self.top_annotation._set_orentation('up')
            self.top_annotation.plot_annotations(ax=self.ax_top_annotation, subplot_spec=gs,
                                                 idxs=col_order, wspace=self.wspace)
        if not self.bottom_annotation is None:
            self.bottom_annotation._set_orentation('down')
            self.bottom_annotation.plot_annotations(ax=self.ax_bottom_annotation, subplot_spec=self.gs[2, 1],
                                                    idxs=col_order, wspace=self.wspace)
        if not self.left_annotation is None:
            gs = self.gs[1, 0] if not self.row_dendrogram else self.left_gs[0, 1]
            self.left_annotation._set_orentation('left')
            self.left_annotation.plot_annotations(ax=self.ax_left_annotation, subplot_spec=gs,
                                                  idxs=row_order, hspace=self.hspace)
        if not self.right_annotation is None:
            self.right_annotation._set_orentation('right')
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

    def post_processing(self):
        pass
# =============================================================================
def composite(cmlist=None, main=0, ax=None, axis=1, row_gap=15, col_gap=15,
              legend_side='right', legend_gap=5, legend_y=0.8, legend_hpad=None,
              legend_width=None, width_ratios=None, height_ratios=None):
    """
    Assemble multiple ClusterMapPlotter objects vertically or horizontally together.

    Parameters
    ----------
    cmlist: list
        a list of ClusterMapPlotter (with plot=False).
    axis: int
        1 for columns (align the cmlist horizontally), 0 for rows (vertically).
    main: int
        use which as main ClusterMapPlotter, will influence row/col order. main is the index
        of cmlist.
    row/col_gap: float
        the row or columns gap between subplots, unit is mm [15].
    legend_side: str
        right,left [right].
    legend_gap: float
        row gap between two legends, unit is mm.
    legend_width: float
        default is None, will be estimated automatically
    width_ratios: list
        a list of width, values can be either float or int.
    height_ratios: list
        a list of height, values can be either float or int.

    Returns
    -------
    tuple:
        ax,legend_axes

    """
    if ax is None:
        ax = plt.gca()
    n = len(cmlist)
    wspace, hspace = 0, 0
    if axis == 1:  # horizontally
        wspace = col_gap * mm2inch * ax.figure.dpi / (ax.get_window_extent().width / n)
        nrows = 1
        ncols = n
        width_ratios = [cm.data2d.shape[1] for cm in cmlist] if width_ratios is None else width_ratios
        height_ratios = None
    else:  # vertically
        hspace = row_gap * mm2inch * ax.figure.dpi / (ax.get_window_extent().height / n)
        nrows = n
        ncols = 1
        width_ratios = None
        height_ratios = [cm.data2d.shape[0] for cm in cmlist] if height_ratios is None else height_ratios
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
        if axis==1: #composite horizontally, have the same row order
            col_order=None
            row_order=cm_1.row_order
        else: # vertically, have the same col order
            row_order=None
            col_order=cm_1.col_order
        cm.plot(ax=axes[i], subplot_spec=gs1, row_order=row_order, col_order=col_order)
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
    return ax,legend_axes
# =============================================================================
if __name__ == "__main__":
    pass
