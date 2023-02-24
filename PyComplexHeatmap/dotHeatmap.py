# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from .utils import mm2inch,plot_legend_list
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# =============================================================================
def dotHeatmap(data=None, x=None, y=None, value=None, hue=None,
               row_order=None,col_order=None,
               colors=None, cmap='Set1', ax=None,
               show_rownames=True, show_colnames=True,
               plot_legend=True,legend_side='right',label_side='left',
               legend_hpad=3,legend_width=4.5,legend_vpad=5,
               legend_gap=5,color_legend_kws={},
               dot_legend_kws={},cmap_legend_kws={},**kwargs):
    marker = kwargs.get('marker', 'o')  # https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
    color_dict=None
    def scale(values):
        min_s, max_s = np.nanmin(values), np.nanmax(values)
        if min_s == max_s:
            return [1 for j in values]
        return [(j - min_s) / (max_s - min_s) for j in values]

    # fig, ax = plt.subplots(figsize=(6,6))
    if ax is None:
        ax = plt.gca()
    col_order = data[x].unique().tolist() if col_order is None else col_order
    row_order = data[y].unique().tolist() if row_order is None else row_order
    df = data[x].apply(lambda j: col_order.index(j) + 1).to_frame(name='X')
    df['Y'] = data[y].apply(lambda j: row_order.index(j) + 1)
    # df=data.pivot_table(index=x,columns=y,values=value,aggfunc=np.mean)

    w, h = ax.get_window_extent().width / ax.figure.dpi, ax.get_window_extent().height / ax.figure.dpi
    r = min(w * 72 / len(col_order), h * 72 / len(row_order))

    df['S'] = kwargs.pop('s', scale(data[value].values))

    # print(kwargs)
    if type(cmap) != dict and type(marker) == str:
        if 'c' in kwargs:
            df['C'] = kwargs.pop('c')
        elif not hue is None:
            if colors is None:  #
                color_dict = {}
                col_list = data[hue].value_counts().index.tolist()
                for c in col_list:
                    color_dict[c] = matplotlib.colors.to_hex(plt.get_cmap(cmap)(col_list.index(c)))
            elif type(colors) == dict:
                color_dict = colors
            elif type(colors) == str:
                col_list = data[hue].value_counts().index.tolist()
                for c in col_list:
                    color_dict[c] = colors
            else:
                raise ValueError('colors must be string or dict')

            df['C'] = data[hue].map(color_dict)
        else:
            df['C'] = scale(data[value].values)
            # kwargs.setdefault('norm',matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
            kwargs.setdefault('cmap', cmap)
        ax.scatter(x=df.X.values, y=df.Y.values, s=df.S * (r ** 2),
                   c=df.C.values, **kwargs)  #
    elif type(cmap) == dict and not hue is None:
        for h in cmap:  # key are hue, values are cmap
            df1 = df.loc[data[hue] == h]
            # df1['C']=kwargs.pop('c',scale(df1.S.values))
            df1.insert(0, 'C', kwargs.pop('c', scale(df1.S.values)))
            kwargs['cmap'] = cmap[h]
            if type(marker) == dict:
                kwargs['marker'] = marker[h]
            ax.scatter(x=df1.X.values, y=df1.Y.values, s=df1.S * (r ** 2),
                       c=df1.C.values, **kwargs)  #
    else:
        raise ValueError('cmap must be string or dict')

    ax.set_ylim([0.5, len(row_order) + 0.5])
    ax.set_xlim(0.5, len(col_order) + 0.5)
    y_locater = list(range(1, len(row_order) + 1))
    x_locater = list(range(1, len(col_order) + 1))
    ax.yaxis.set_major_locator(plt.FixedLocator(y_locater))
    ax.yaxis.set_minor_locator(plt.FixedLocator(np.array(y_locater) - 0.5))
    ax.xaxis.set_major_locator(plt.FixedLocator(x_locater))
    ax.xaxis.set_minor_locator(plt.FixedLocator(np.array(x_locater) - 0.5))
    #    ax.yaxis.set_major_formatter(plt.FixedFormatter(df.index.values))
    #    majorLocator = MultipleLocator(1)
    #    ax.yaxis.set_major_locator(majorLocator)
    if show_colnames:
        ax.set_yticklabels(row_order)
    if show_rownames:
        ax.set_xticklabels(col_order, rotation=-90)
    # ax.grid(which='minor',color='white',linestyle='-',alpha=0.6)
    ax.tick_params(axis='both', which='both',
                   left=False, right=False, labelleft=True, labelright=False,
                   top=False, bottom=False, labeltop=False, labelbottom=True)
    if plot_legend:
        legend_list=[]
        if not color_dict is None and not hue is None:
            legend_list.append([color_dict, hue, color_legend_kws, len(color_dict), 'color_dict'])
        if type(cmap)==str and colors is None:
            legend_list.append([cmap, value, cmap_legend_kws, 4, 'cmap'])
        if type(cmap)==dict:
            for k in cmap:
                legend_list.append([cmap[k], k, cmap_legend_kws, 4, 'cmap'])
        if type(marker)==dict and not hue is None:
            legend_list.append([(marker,colors,r*0.8), hue, dot_legend_kws, len(marker), 'markers']) #markersize is r*0.8
        # dot size legend:
        if 's' not in kwargs:
            max_s=np.nanmax(data[value].values)
            markers1={}
            ms={}
            for f in [1,0.8,0.6,0.4,0.2]:
                k=str(round(f*max_s,2))
                markers1[k]='o'
                ms[k]=f * r
            legend_list.append([(markers1, colors, ms), value, dot_legend_kws, len(markers1), 'markers'])
        label_width=ax.yaxis.label.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
        yticklabels = ax.yaxis.get_ticklabels()
        if len(yticklabels) == 0:
            ticklabel_width=0
        else:
            ticklabel_width=max([label.get_window_extent(renderer=ax.figure.canvas.get_renderer()).width
                 for label in ax.yaxis.get_ticklabels()])
        if len(legend_list) > 0:
            space = max([label_width,ticklabel_width]) if (legend_side == 'right' and label_side == 'right') else 0
            legend_hpad = legend_hpad * mm2inch * ax.figure.dpi  # mm to inch to pixel
            legend_axes, cbars, boundry = plot_legend_list(legend_list, ax=ax, space=space + legend_hpad,
                                 legend_side='right', gap=legend_gap,
                                 legend_width=legend_width, legend_vpad=legend_vpad)
    return ax
# =============================================================================
if __name__ == "__main__":
    pass