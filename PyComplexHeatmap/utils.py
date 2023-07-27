# -*- coding: utf-8 -*-
# !/usr/bin/env python3
"""Utility functions, for internal use."""
import numpy as np
import pandas as pd
import collections
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
mm2inch=1/25.4
# =============================================================================
def set_default_style():
    from matplotlib import rcParams
    D={
        # 'font.family':['sans serif'], #'serif',
        # 'mathtext.fontset':'dejavuserif',
        # 'font.sans-serif':['Arial'],
        'pdf.fonttype':42,

        # Remove legend frame
        'legend.frameon': True,
        'legend.fontsize': 10,

        # Savefig
        'figure.dpi': 100,
        'savefig.bbox': 'tight',
        'savefig.dpi':300,
        'savefig.pad_inches': 0.05
    }
    
    rcParams.update(D)
# =============================================================================
def get_colormap(cmap):
    try:
        return plt.colormaps.get(cmap) # matplotlib >= 3.5.1?
    except:
        return plt.get_cmap(cmap) # matplotlib <=3.4.3?

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
# =============================================================================
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
# =============================================================================
def define_cmap(plot_data, vmin=None, vmax=None, cmap=None, center=None, robust=True,
                          na_col='white'):
    """Use some heuristics to set good defaults for colorbar and range."""
    # plot_data is a np.ma.array instance
    # plot_data=np.ma.masked_where(np.asarray(plot_data), plot_data)
    # calc_data = plot_data.astype(float).filled(np.nan)
    if vmin is None:
        if robust:
            vmin = np.nanpercentile(plot_data, 2)
        else:
            vmin = np.nanmin(plot_data)
    if vmax is None:
        if robust:
            vmax = np.nanpercentile(plot_data, 98)
        else:
            vmax = np.nanmax(plot_data)

    # Choose default colormaps if not provided
    if cmap is None:
        if center is None:
            cmap = 'jet'
        else:
            cmap = 'exp1'
    if isinstance(cmap, str):
        cmap1 = matplotlib.cm.get_cmap(cmap).copy()
    elif isinstance(cmap, list):
        cmap1 = matplotlib.colors.ListedColormap(cmap)
    else:
        cmap1 = cmap

    cmap1.set_bad(color=na_col)  # set the color for NaN values
    # Recenter a divergent colormap
    if center is not None:
        # bad = cmap1(np.ma.masked_invalid([np.nan]))[0]  # set the first color as the na_color
        under = cmap1(-np.inf)
        over = cmap1(np.inf)
        under_set = under != cmap1(0)
        over_set = over != cmap1(cmap1.N - 1)

        vrange = max(vmax - center, center - vmin)
        normalize = matplotlib.colors.Normalize(center - vrange, center + vrange)
        cmin, cmax = normalize([vmin, vmax])
        cc = np.linspace(cmin, cmax, 256)
        cmap1 = matplotlib.colors.ListedColormap(cmap1(cc))
        # cmap1.set_bad(bad)
        if under_set:
            cmap1.set_under(under)  # set the color of -np.inf as the color for low out-of-range values.
        if over_set:
            cmap1.set_over(over)
    else:
        normalize = matplotlib.colors.Normalize(vmin, vmax)
    return cmap1,normalize
# =============================================================================
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
# =============================================================================
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
# =============================================================================
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
# =============================================================================
# =============================================================================
def _skip_ticks(labels, tickevery):
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
# =============================================================================
def _auto_ticks(ax, labels, axis):
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
    ticks, labels =_skip_ticks(labels, tick_every)
    return ticks, labels
# =============================================================================
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
# =============================================================================
def _index_to_label(index):
    """
    Convert a pandas index or multiindex to an axis label.

    """
    if isinstance(index, pd.MultiIndex):
        return "-".join(map(to_utf8, index.names))
    else:
        return index.name
# =============================================================================
def _index_to_ticklabels(index):
    """
    Convert a pandas index or multiindex into ticklabels.

    """
    if isinstance(index, pd.MultiIndex):
        return ["-".join(map(to_utf8, i)) for i in index.values]
    else:
        return index.values
# =============================================================================
def cluster_labels(labels=None,xticks=None,majority=True):
    """
    Merge the adjacent labels into one.

    Parameters
    ----------
    labels : a list of labels.
    xticks : a list of x or y ticks coordinates.
    majority: if majority=True, keep the labels with the largest clusters.

    Returns
    -------
    labels,ticks: merged labels and ticks coordinates.

    Examples
    -------
    labels=['A','A','B','B','A','C','C','B','B','B','C']
    xticks=list(range(len(labels)))
    new_labels,x=cluster_labels(labels,xticks)

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
    if majority:
        cluster_size=collections.defaultdict(int)
        largest_cluster={}
        for i in clusters_labels:
            if len(clusters_x[i]) > cluster_size[clusters_labels[i]]:
                cluster_size[clusters_labels[i]]=len(clusters_x[i])
                largest_cluster[clusters_labels[i]]=i
        labels = [clusters_labels[i] for i in clusters_x if i==largest_cluster[clusters_labels[i]]]
        x = [np.mean(clusters_x[i]) for i in clusters_x if i==largest_cluster[clusters_labels[i]]]
        return labels, x

    labels = [clusters_labels[i] for i in clusters_x]
    x = [np.mean(clusters_x[i]) for i in clusters_x]
    return labels, x
# =============================================================================
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
    if ax is None:
        ax=plt.gca()
    lgd_kws=kws.copy() if not kws is None else {} #bbox_to_anchor=(x,-0.05)
    lgd_kws.setdefault("frameon",True)
    lgd_kws.setdefault("ncol", 1)
    lgd_kws['loc'] = 'upper left'
    lgd_kws['bbox_transform'] = ax.figure.transFigure
    lgd_kws.setdefault('borderpad',0.1 * mm2inch * 72)  # 0.1mm
    lgd_kws.setdefault('markerscale',1)
    lgd_kws.setdefault('handleheight',1)  # font size, units is points
    lgd_kws.setdefault('handlelength',1)  # font size, units is points
    lgd_kws.setdefault('borderaxespad',0.1) #The pad between the axes and legend border, in font-size units.
    lgd_kws.setdefault('handletextpad',0.3) #The pad between the legend handle and text, in font-size units.
    lgd_kws.setdefault('labelspacing',0.1)  # gap height between two Patches,  0.05*mm2inch*72
    lgd_kws.setdefault('columnspacing', 1)
    lgd_kws.setdefault('bbox_to_anchor',(0,1))
    if label_side=='left':
        lgd_kws.setdefault("markerfirst", False)
        align = 'right'
    else:
        lgd_kws.setdefault("markerfirst", True)
        align='left'

    availabel_height=ax.figure.get_window_extent().height * lgd_kws['bbox_to_anchor'][1]
    l = [mpatches.Patch(color=c, label=l) for l, c in D.items()] #kws:?mpatches.Patch; rasterized=True
    L = ax.legend(handles=l, title=title,**lgd_kws)
    ax.figure.canvas.draw()
    while L.get_window_extent().height > availabel_height:
        # ax.cla()
        print("Incresing ncol")
        lgd_kws['ncol']+=1
        if lgd_kws['ncol']>=3:
            print("More than 3 cols is not supported")
            L.remove()
            return None
        L = ax.legend(handles=l, title=title, **lgd_kws)
        ax.figure.canvas.draw()
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
    # print(availabel_height,L.get_window_extent().height)
    return L
# =============================================================================
def plot_cmap_legend(cax=None,ax=None,cmap='turbo',label=None,kws=None,label_side='right'):
    """
    Plot legend for cmap.

    Parameters
    ----------
    cax : Axes into which the colorbar will be drawn.
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
    cbar_kws.setdefault('label',label)
    # cbar_kws.setdefault("aspect",3)
    cbar_kws.setdefault("orientation","vertical")
    # cbar_kws.setdefault("use_gridspec", True)
    # cbar_kws.setdefault("location", "bottom")
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
    cbar=ax.figure.colorbar(m,cax=cax,**cbar_kws) #use_gridspec=True
    # cbar.outline.set_color('white')
    # cbar.outline.set_linewidth(2)
    # cbar.dividers.set_color('red')
    # cbar.dividers.set_linewidth(2)
    # ax.figure.tight_layout(rect=[cax.get_position().x0, 0, cax.get_position().x1, 1],h_pad=0,w_pad=0) #
    # cax.spines['top'].set_visible(False)
    # cax.spines['bottom'].set_visible(False)
    # f = cbar.ax.get_window_extent().height / cax.get_window_extent().height
    return cbar
# =============================================================================
def plot_marker_legend(obj=None, ax=None, title=None, color_text=True,
                       label_side='right',kws=None):
    """
    plot legned for different marker

    Parameters
    ----------
    D: a dict, key is categorical variable, values are marker.
    ax: axes to plot the legend.
    title: title of legend.
    color_text: whether to change the color of text based on the color in D.
    label_side: right of left.
    kws: kws passed to plt.legend.

    Returns
    -------
    ax.legend

    """
    if ax is None:
        ax=plt.gca()
    markers,colors,ms = obj
    # markers = {'A': 'o', 'B': 's', 'C': 'D'}
    # color_dict = {'A': 'red', 'B': 'blue', 'C': 'green'}
    if colors is None:
        colors='black'
    elif type(colors)==dict:
        color_dict=colors
    if type(colors)==str:
        color_dict = {}
        for k in markers:
            color_dict[k]=colors

    lgd_kws=kws.copy() if not kws is None else {} #bbox_to_anchor=(x,-0.05)
    lgd_kws.setdefault("frameon", True)
    lgd_kws.setdefault("ncol", 1)
    lgd_kws['loc'] = 'upper left'
    lgd_kws['bbox_transform'] = ax.figure.transFigure
    lgd_kws.setdefault('borderpad', 0.1 * mm2inch * 72)  # 0.1mm
    if ms is None:
        s=lgd_kws.pop('markersize',10)
        ms_dict={} #key is label (markers.keys), values is markersize.
        for k in markers:
            ms_dict[k]=s
    elif type(ms) != dict:
        ms_dict = {}
        for k in markers:
            ms_dict[k] = ms
    else:
        ms_dict=ms

    lgd_kws.setdefault('markerscale', 1)
    lgd_kws.setdefault('handleheight', 1)  # font size, units is points
    lgd_kws.setdefault('handlelength', 1)  # font size, units is points
    lgd_kws.setdefault('borderaxespad', 0.1)  # The pad between the axes and legend border, in font-size units.
    lgd_kws.setdefault('handletextpad', 0.3)  # The pad between the legend handle and text, in font-size units.
    lgd_kws.setdefault('labelspacing', 0.5)  # gap height between two Patches,  0.05*mm2inch*72
    lgd_kws.setdefault('columnspacing', 1)
    lgd_kws.setdefault('bbox_to_anchor', (0, 1))
    if label_side=='left':
        lgd_kws.setdefault("markerfirst", False)
        align = 'right'
    else:
        lgd_kws.setdefault("markerfirst", True)
        align='left'

    availabel_height = ax.figure.get_window_extent().height * lgd_kws['bbox_to_anchor'][1]
    # print(ms_dict,markers)
    L = [mlines.Line2D([], [], color=color_dict.get(l,'black'), marker=m, linestyle='None',
                          markersize=ms_dict.get(l,10), label=l)
         for l, m in markers.items()] #kws:?mpatches.Patch; rasterized=True
    ms=lgd_kws.pop('markersize',10)
    Lgd = ax.legend(handles=L, title=title,**lgd_kws)
    ax.figure.canvas.draw()
    while Lgd.get_window_extent().height > availabel_height:
        print("Incresing ncol")
        lgd_kws['ncol']+=1
        if lgd_kws['ncol']>=3:
            print("More than 3 cols is not supported")
            Lgd.remove()
            return None
        Lgd = ax.legend(handles=L, title=title, **lgd_kws)
        ax.figure.canvas.draw()
    Lgd._legend_box.align = align
    if color_text:
        for text in Lgd.get_texts():
            try:
                lum = _calculate_luminance(color_dict[text.get_text()])
                text_color = 'black' if lum > 0.408 else color_dict[text.get_text()]
                text.set_color(text_color)
            except:
                pass
    ax.add_artist(Lgd)
    ax.grid(False)
    # print(availabel_height,L.get_window_extent().height)
    return Lgd
# =============================================================================
def cal_legend_width(legend_list):
    lgd_w=4.5
    legend_width=0
    for lgd in legend_list:
        obj, title, legend_kws, n, lgd_t = lgd
        if lgd_t=='color_dict':
            max_text_len=max(len(str(title)),max([len(str(k)) for k in obj]))
            fontsize = legend_kws.get('fontsize',plt.rcParams['legend.fontsize'])
            lgd_w=max_text_len * fontsize * 0.65 / 72 / mm2inch #point to inches to mm. in average, width = height * 0.6
        elif lgd_t=="markers":
            max_text_len = len(str(title))
            fontsize = legend_kws.get('fontsize', plt.rcParams['legend.fontsize'])
            lgd_w = max_text_len * fontsize * 0.65 / 72 / mm2inch
        if legend_width < lgd_w:
            legend_width=lgd_w
    return legend_width

def plot_legend_list(legend_list=None,ax=None,space=0,legend_side='right',
                     y0=None,gap=2,delta_x=None,legend_width=None,legend_vpad=5,
                     cmap_width=4.5):
    """
    Plot all lengends for a given legend_list.

    Parameters
    ----------
    legend_list : a list including [handles(dict) / cmap / markers dict, title, legend_kws, height, legend_type]
    ax :axes to plot.
    space : unit is pixels.
    legend_side :right, or left
    y0 : the initate coordinate of y for the legend.
    gap : gap between legends, default is 2mm.
    legend_width: width of the legend, default is 4.5mm.

    Returns
    -------
    legend_axes,boundry:

    """
    if ax is None:
        print("No ax was provided, using plt.gca()")
        ax=plt.gca()
        ax.set_axis_off()
        left=ax.get_position().x0+ax.yaxis.labelpad*2/ax.figure.get_window_extent().width if delta_x is None else ax.get_position().x0+delta_x
    else:
        #labelpad: Spacing in points, pad is the fraction relative to x1.
        pad = (space+ax.yaxis.labelpad*1.2*ax.figure.dpi / 72) / ax.figure.get_window_extent().width if delta_x is None else delta_x #labelpad unit is points
        left=ax.get_position().x1 + pad
    if legend_width is None:
        legend_width=cal_legend_width(legend_list) + 2.5 #base width for color rectangle is set to 2.5 mm
        # print(f"Estimated legend width: {legend_width} mm")
    legend_width=legend_width*mm2inch*ax.figure.dpi / ax.figure.get_window_extent().width #mm to px to fraction
    cmap_width = cmap_width * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().width  # mm to px to fraction
    if legend_side=='right':
        ax_legend=ax.figure.add_axes([left,ax.get_position().y0,legend_width,ax.get_position().height]) #left, bottom, width, height
    legend_axes=[ax_legend]
    cbars=[]
    leg_pos = ax_legend.get_position() #left bototm: x0,y0; top right: x1,y1

    # y is the bottom position of the first legend (from top to the bottom)
    y = leg_pos.y1 - legend_vpad*mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
    lgd_col_max_width=0 #the maximum width of all legends in one column
    v_gap=round(gap*mm2inch*ax.figure.dpi/ax.figure.get_window_extent().height,2) #2mm vertically height gap between two legends
    i=0
    while i <= len(legend_list)-1:
        obj, title, legend_kws, n, lgd_t = legend_list[i]
        ax1 = legend_axes[-1] #ax for the legend on the right
        ax1.set_axis_off()
        color_text=legend_kws.pop("color_text",True)
        if lgd_t=='cmap': #type(obj)==str: # a cmap, plot colorbar
            f = 15 * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height  # 15 mm
            if y-f < 0: #add a new column of axes to plot legends
                offset=(lgd_col_max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                ax2=ax.figure.add_axes(rect=[ax1.get_position().x0+offset, ax.get_position().y0, cmap_width, ax.get_position().height]) #left_pos.width
                legend_axes.append(ax2)
                ax1=legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y = leg_pos.y1 - legend_vpad * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
                lgd_col_max_width = 0

            cax=ax1.figure.add_axes(rect=[leg_pos.x0,y-f,cmap_width,f],
                                   xmargin=0,ymargin=0) #unit is fractions of figure width and height
            # [i.set_linewidth(0.5) for i in cax.spines.values()]
            cax.figure.subplots_adjust(bottom=0) #wspace=0, hspace=0
            #https://matplotlib.org/stable/api/figure_api.html
            #[left, bottom, width, height],sharex=True,anchor=(0,0),frame_on=False.
            cbar=plot_cmap_legend(ax=ax1,cax=cax,cmap=obj,label=title,label_side=legend_side,kws=legend_kws)
            cbar_width=cbar.ax.get_window_extent().width
            cbars.append(cbar)
            if cbar_width > lgd_col_max_width:
                lgd_col_max_width=cbar_width
        elif lgd_t == 'color_dict':
            # print(obj, title, legend_kws)
            legend_kws['bbox_to_anchor']=(leg_pos.x0,y) #lower left position of the box.
            #x, y, width, height #kws['bbox_transform'] = ax.figure.transFigure
            # ax1.scatter(leg_pos.x0,y,s=6,color='red',zorder=20,transform=ax1.figure.transFigure)
            L = plot_color_dict_legend(D=obj, ax=ax1, title=title, label_side=legend_side,
                                       color_text=color_text, kws=legend_kws)
            if L is None:
                print("Legend too long, generating a new column..")
                pad = (lgd_col_max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                left_pos = ax1.get_position()
                ax2 = ax.figure.add_axes([left_pos.x0 + pad, ax.get_position().y0, left_pos.width, ax.get_position().height])
                legend_axes.append(ax2)
                ax1 = legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y = leg_pos.y1 - legend_vpad * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
                legend_kws['bbox_to_anchor'] = (leg_pos.x0, y)
                L = plot_color_dict_legend(D=obj, ax=ax1, title=title, label_side=legend_side,
                                           color_text=color_text, kws=legend_kws)
                lgd_col_max_width = 0
            L_width = L.get_window_extent().width
            if L_width > lgd_col_max_width:
                lgd_col_max_width = L_width
            f = L.get_window_extent().height / ax.figure.get_window_extent().height
            cbars.append(L)
        elif lgd_t == 'markers':
            legend_kws['bbox_to_anchor'] = (leg_pos.x0, y)  # lower left position of the box.
            L = plot_marker_legend(obj=obj, ax=ax1, title=title, label_side=legend_side,
                                   color_text=color_text, kws=legend_kws) #obj is a tuple: markers and colors
            if L is None:
                print("Legend too long, generating a new column..")
                pad = (lgd_col_max_width + ax.yaxis.labelpad * 2) / ax.figure.get_window_extent().width
                left_pos = ax1.get_position()
                ax2 = ax.figure.add_axes(
                    [left_pos.x0 + pad, ax.get_position().y0, left_pos.width, ax.get_position().height])
                legend_axes.append(ax2)
                ax1 = legend_axes[-1]
                ax1.set_axis_off()
                leg_pos = ax1.get_position()
                y = leg_pos.y1 - legend_vpad * mm2inch * ax.figure.dpi / ax.figure.get_window_extent().height if y0 is None else y0
                legend_kws['bbox_to_anchor'] = (leg_pos.x0, y)
                L = plot_marker_legend(obj=obj, ax=ax1, title=title, label_side=legend_side,
                                           color_text=color_text, kws=legend_kws)
                lgd_col_max_width = 0
            L_width = L.get_window_extent().width
            if L_width > lgd_col_max_width:
                lgd_col_max_width = L_width
            f = L.get_window_extent().height / ax.figure.get_window_extent().height
            cbars.append(L)

        y = y - f - v_gap
        i+=1

    if legend_side=='right':
        boundry=ax1.get_position().y1+lgd_col_max_width / ax.figure.get_window_extent().width
    else:
        boundry = ax1.get_position().y0 - lgd_col_max_width / ax.figure.get_window_extent().width
    return legend_axes,cbars,boundry
# =============================================================================
set_default_style()