# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""Utility functions, for internal use."""
import numpy as np
import pandas as pd
import collections
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def _check_mask(data, mask):
    """

    Ensure that data and mask are compatible and add missing values and infinite values.
    Values will be plotted for cells where ``mask`` is ``False``.
    ``data`` is expected to be a DataFrame; ``mask`` can be an array or
    a DataFrame.
    """
    if mask is None:
        mask = np.zeros(data.shape, bool)
    if isinstance(mask, np.ndarray):
        if mask.shape != data.shape:
            raise ValueError("Mask must have the same shape as data.")
        mask = pd.DataFrame(mask, index=data.index, columns=data.columns, dtype=bool)
    elif isinstance(mask, pd.DataFrame):
        if not mask.index.equals(data.index) and mask.columns.equals(data.columns):
            err = "Mask must have the same index and columns as data."
            raise ValueError(err)

    # Add any cells with missing values or infinite values to the mask
    mask = mask | pd.isnull(data) | np.logical_not(np.isfinite(data))
    return mask

def _calculate_luminance(color):
    """
    Calculate the relative luminance of a color according to W3C standards

    Parameters
    ----------
    color : matplotlib color or sequence of matplotlib colors
        Hex code, rgb-tuple, or html color name.
    Returns
    -------
    luminance : float(s) between 0 and 1

    """
    rgb = matplotlib.colors.colorConverter.to_rgba_array(color)[:, :3]
    rgb = np.where(rgb <= .03928, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lum = rgb.dot([.2126, .7152, .0722])
    try:
        return lum.item()
    except ValueError:
        return lum

def despine(fig=None, ax=None, top=True, right=True, left=False,
            bottom=False):
    """
    Remove the top and right spines from plot(s).

    Parameters
    ----------
    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.

    Returns
    -------
    None

    """
    if fig is None and ax is None:
        axes = plt.gcf().axes
    elif fig is not None:
        axes = fig.axes
    elif ax is not None:
        axes = [ax]

    for ax_i in axes:
        for side in ["top", "right", "left", "bottom"]:
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
        if left and not right: #remove left, keep right
            maj_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.minorTicks)
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.minorTicks)
            ax_i.xaxis.set_ticks_position("top")
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)

def _draw_figure(fig):
    """
    Force draw of a matplotlib figure, accounting for back-compat.

    """
    # See https://github.com/matplotlib/matplotlib/issues/19197 for context
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass

def axis_ticklabels_overlap(labels):
    """
    Return a boolean for whether the list of ticklabels have overlaps.

    Parameters
    ----------
    labels : list of matplotlib ticklabels
    Returns
    -------
    overlap : boolean
        True if any of the labels overlap.

    """
    if not labels:
        return False
    try:
        bboxes = [l.get_window_extent() for l in labels]
        overlaps = [b.count_overlaps(bboxes) for b in bboxes]
        return max(overlaps) > 1
    except RuntimeError:
        # Issue on macos backend raises an error in the above code
        return False

def to_utf8(obj):
    """
    Return a string representing a Python object.
    Strings (i.e. type ``str``) are returned unchanged.
    Byte strings (i.e. type ``bytes``) are returned as UTF-8-decoded strings.
    For other objects, the method ``__str__()`` is called, and the result is
    returned as a string.

    Parameters
    ----------
    obj : object
        Any Python object
    Returns
    -------
    s : str
        UTF-8-decoded string representation of ``obj``

    """
    if isinstance(obj, str):
        return obj
    try:
        return obj.decode(encoding="utf-8")
    except AttributeError:  # obj is not bytes-like
        return str(obj)

def _index_to_label(index):
    """
    Convert a pandas index or multiindex to an axis label.

    """
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name

def _index_to_ticklabels(index):
    """
    Convert a pandas index or multiindex into ticklabels.

    """
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values

def cluster_labels(labels,xticks):
    """
    Merge the adjacent labels into one.

    Parameters
    ----------
    labels : a list of labels.
    xticks : a list of x or y ticks coordinates.

    Returns
    -------
    labels,ticks: merged labels and ticks coordinates.

    """
    clusters_x = collections.defaultdict(list)
    clusters_labels = {}
    scanned_labels = ''
    i = 0
    for label, x in zip(labels, xticks):
        if label != scanned_labels:
            scanned_labels = label
            i += 1
            clusters_labels[i] = scanned_labels
        clusters_x[i].append(x)
    labels = [clusters_labels[i] for i in clusters_x]
    x = [np.mean(clusters_x[i]) for i in clusters_x]
    return labels, x

def plot_color_dict_legend(D=None, ax=None, title=None, color_text=True,
                           label_side='right',kws=None):
    """
    plot legned for color dict

    Parameters
    ----------
    D: a dict, key is categorical variable, values are colors.
    ax: axes to plot the legend.
    title: title of legend.
    color_text: whether to change the color of text based on the color in D.
    label_side: right of left.
    kws: kws passed to plt.legend.

    Returns
    -------
    ax.legend

    """
    lgd_kws=kws.copy() if not kws is None else {} #bbox_to_anchor=(x,-0.05)
    lgd_kws.setdefault("frameon",True)
    lgd_kws.setdefault("ncol", 1)
    lgd_kws['loc'] = 'upper left'
    lgd_kws['bbox_transform'] = ax.figure.transFigure
    lgd_kws.setdefault('borderpad',0.1 * 0.0394 * 72)  # 0.1mm
    lgd_kws.setdefault('markerscale',1)
    lgd_kws.setdefault('handleheight',1)  # font size, units is points
    lgd_kws.setdefault('handlelength',1)  # font size, units is points
    lgd_kws.setdefault('borderaxespad',0)
    lgd_kws.setdefault('handletextpad',0.1)
    lgd_kws.setdefault('labelspacing',0.1)  # gap height between two Patches,  0.05*0.0394*72
    lgd_kws.setdefault('columnspacing', 1)
    if label_side=='left':
        lgd_kws.setdefault("markerfirst", False)
        align = 'right'
    else:
        lgd_kws.setdefault("markerfirst", True)
        align='left'

    availabel_height=ax.figure.get_window_extent().height * kws['bbox_to_anchor'][1]
    if ax is None:
        ax=plt.gca()
    l = [mpatches.Patch(color=c, label=l) for l, c in D.items()] #kws:?mpatches.Patch; rasterized=True
    L = ax.legend(handles=l, title=title,**lgd_kws)
    ax.figure.canvas.draw()
    while L.get_window_extent().height > availabel_height:
        print("Incresing ncol")
        lgd_kws['ncol']+=1
        L = ax.legend(handles=l, title=title, **kws)
        ax.figure.canvas.draw()
        if lgd_kws['ncol']>=3:
            return None
    L._legend_box.align = align
    if color_text:
        for text in L.get_texts():
            try:
                lum = _calculate_luminance(D[text.get_text()])
                text_color = 'black' if lum > 0.408 else D[text.get_text()]
                text.set_color(text_color)
            except:
                pass
    ax.add_artist(L)
    ax.grid(False)
    # ax.axis("off")
    # print(availabel_height,L.get_window_extent().height)
    return L

def plot_cmap_legend(cax=None,ax=None,cmap='turbo',label=None,kws=None,label_side='right'):
    """
    Plot legend for cmap.

    Parameters
    ----------
    cax : axes to plot legend.
    ax :  axes to anchor.
    cmap : turbo, hsv, Set1, Dark2, Paired, Accent,tab20,exp1,exp2,meth1,meth2
    label : title for legend.
    kws : kws passed to plt.colorbar.
    label_side : right or left.

    Returns
    -------
    cbar: axes of legend

    """
    label='' if label is None else label
    cbar_kws={} if kws is None else kws.copy()
    cbar_kws.setdefault("aspect",3)
    cbar_kws.setdefault("fraction", 1)
    cbar_kws.setdefault("shrink", 1)
    cbar_kws.setdefault("pad", 0)
    vmax=cbar_kws.pop('vmax',1)
    vmin=cbar_kws.pop('vmin',0)
    # print(vmin,vmax,'vmax,vmin')
    cax.set_ylim([vmin,vmax])
    cbar_kws.setdefault("ticks",[vmin,(vmax+vmin)/2,vmax])
    m = plt.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
        cmap=cmap)
    cax.yaxis.set_label_position(label_side)
    cax.yaxis.set_ticks_position(label_side)
    cbar=ax.figure.colorbar(m,cax=cax,label=label,**cbar_kws) #use_gridspec=True
    # cax.spines['top'].set_visible(False)
    # cax.spines['bottom'].set_visible(False)
    # f = cbar.ax.get_window_extent().height / cax.get_window_extent().height
    return cbar

def plot_legend_list(legend_list=None,ax=None,space=0,legend_side='right',
                     y0=None,gap=2):
    """
    Plot all lengends for a given legend_list.

    Parameters
    ----------
    legend_list : a list including [handles(dict) / cmap, title, legend_kws]
    ax :axes to plot.
    space : unit is pixels.
    legend_side :right, or left
    y0 : the initate coordinate of y for the legend.
    gap : gap between legends, default is 2mm.

    Returns
    -------
    legend_axes,boundry:

    """
    if ax is None:
        print("No ax was provided, using plt.gca()")
        ax=plt.gca()
        ax.set_axis_off()
        left=ax.get_position().x0+ax.yaxis.labelpad*2/ax.figure.get_window_extent().width
    else:
        pad = (space+ax.yaxis.labelpad*2*ax.figure.dpi / 72) / ax.figure.get_window_extent().width #labelpad unit is points
        left=ax.get_position().x1 + pad
        # print(ax.get_position(),space,pad,left)
    width=4.5*0.0394*ax.figure.dpi / ax.figure.get_window_extent().width
    if legend_side=='right':
        ax_legend=ax.figure.add_axes([left,ax.get_position().y0,width,ax.get_position().height]) #left, bottom, width, height
    # print(ax.get_position(),ax_legend.get_position())
    legend_axes=[ax_legend]
    leg_pos = ax_legend.get_position()
    y = leg_pos.y1 if y0 is None else y0
    max_width=0
    h_gap=round(gap*0.0394*ax.figure.dpi/ax.figure.get_window_extent().height,2) #2mm height gap between two legends
    for i,legend in enumerate(legend_list):
        color, title, legend_kws, n = legend
        ax1 = legend_axes[-1]
        ax1.set_axis_off()
        # print(i,legend)
        color_text=legend_kws.pop("color_text",True)
        if type(color)==str: # a cmap, plot colorbar
            f=round(15*0.0394*ax.figure.dpi / ax.figure.get_window_extent().height,2) #15 mm
            if y-f < 0: #add a new column of axes to plot legends
                left_pos=ax1.get_position()
                pad=(max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                ax2=ax.figure.add_axes([left_pos.x0+pad, ax.get_position().y0, left_pos.width, ax.get_position().height])
                legend_axes.append(ax2)
                ax1=legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y=leg_pos.y1 if y0 is None else y0
                max_width = 0
            y_cax_to_figure=y-f
            width=leg_pos.width
            cax=ax.figure.add_axes([leg_pos.x0,y_cax_to_figure,width,f])
            cbar=plot_cmap_legend(ax=ax,cax=cax,cmap=color,label=title,label_side=legend_side,kws=legend_kws)
            cbar_width=cbar.ax.get_window_extent().width
            if cbar_width > max_width:
                max_width=cbar_width
            # print(cax.get_position(),cbar.ax.get_position())
        else:
            # print("color_dict",leg_pos.x0,y)
            legend_kws['bbox_to_anchor']=(leg_pos.x0,y) #x, y, width, height #kws['bbox_transform'] = ax.figure.transFigure
            # ax1.scatter(leg_pos.x0,y,s=6,color='red',zorder=20,transform=ax1.figure.transFigure)
            # print("color_dict",ax1.get_position(),leg_pos)
            L=plot_color_dict_legend(D=color, ax=ax1,title=title, label_side=legend_side,
                                     color_text=color_text,kws=legend_kws)
            L_width=L.get_window_extent().width
            if L_width > max_width:
                max_width=L_width
            if L is None:
                print("Legend too long, generating a new column..")
                pad = (max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                left_pos = ax1.get_position()
                ax2 = ax.figure.add_axes([left_pos.x0 + pad, ax.get_position().y0, left_pos.width, ax.get_position().height])
                legend_axes.append(ax2)
                ax1 = legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y=leg_pos.y1 if y0 is None else y0
                max_width=0
                continue
            f = L.get_window_extent().height / ax.figure.get_window_extent().height
        y = y - f - h_gap
    if legend_side=='right':
        boundry=ax1.get_position().y1+max_width / ax.figure.get_window_extent().width
    else:
        boundry = ax1.get_position().y0 - max_width / ax.figure.get_window_extent().width
    return legend_axes,boundry
