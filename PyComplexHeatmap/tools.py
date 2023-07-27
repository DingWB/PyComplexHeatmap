# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import os, sys
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from .utils import get_colormap

def tbarplot(df=None,x=None,y=None,hue=None,hue_order=None,palette='Set1',figsize=(4,6),
             outname='test.pdf',title=''):
    """
    Plot barplot with text on the top of bar.

    Parameters
    ----------
    df : dataframe
    x : str
    y : str
    hue : str
    hue_order : list
        order for hue
    palette : str
        palette
    figsize : tuple
        figsize
    outname : path
        output pdf filename
    title : str
        title

    Returns
    -------

    """
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x=x, y=y, hue=hue,
                     hue_order=hue_order, palette=palette, dodge=False)
    yticklabels = ax.axes.yaxis.get_majorticklabels()
    for tick in yticklabels: #add text on the top of bar
        pos = tick.get_position()
        t = tick.get_text()
        ax.text(x=0.02, y=pos[1] + 0.125, s=t, fontdict={'color': 'black'},
                verticalalignment="center", horizontalalignment='left',
                transform=ax.get_yaxis_transform())
    ax.set_yticklabels([])
    ax.grid(axis='x', which='major')
    plt.title(title)
    ax.figure.tight_layout()
    plt.savefig(outname, bbox_inches='tight')

def dotplot(df=None,x=None,y=None,hue=None,hue_order=None,
            size=10,color='blue',cmap='Set1',
            title='',figsize=(3,3.5),outname='test.pdf'):
    """
    Plot dot plot for enrichment analysis and other kind of usage.

    Parameters
    ----------
    df : dataframe
    x : str
    y : str
    hue : str
        hue
    hue_order : list
        order for hue.
    size : str or float
        column names used for size or float.
    color : str
        color
    cmap : str
        cmap
    title : str
        title
    figsize : tuple
        figsize
    outname : path
        output name

    Returns
    -------

    """
    if not hue is None:
        hue_order=df[hue].unique().tolist() if hue_order is None else hue_order
        color_dict={h:get_colormap(cmap)(hue_order.index(h)) for h in hue_order}
    else:
        color_dict=None
        hue_order=None
    N=list(range(1,df.shape[0]+1))
    s_min = np.nanmin(df[size].values)
    delta_s = np.nanmax(df[size].values) - s_min
    fig, ax = plt.subplots(figsize=figsize)
    w, h = ax.get_window_extent().width / ax.figure.dpi, ax.get_window_extent().height / ax.figure.dpi
    r = min(w * 72 / len(df.shape[1]), h * 72 / len(df.shape[0]))
    if not hue_order is None:
        for c in hue_order:
          idx = np.where(df[hue].values == c)[0]
          if type(size) == str:
              s = (df.iloc[idx][size] - s_min) / delta_s * r**2
          else:
              s = size
          color1 = color_dict[c] if not color_dict is None else color
          ax.scatter(x=df.iloc[idx][x].tolist(), y=[N[i] for i in idx],
                     s=s, color=color1, label=c)
    else:
        s = df[size].apply(lambda x:(x-s_min) / delta_s) * (r**2) if type(size) == str else size
        ax.scatter(x=df[x].tolist(), y=N,
                   s=s, color=color, label=None)

    ax.set_ylim([0,len(N)+0.8])
    ax.set_xlabel(x)
    ax.yaxis.set_major_locator(plt.FixedLocator(N))
    ax.yaxis.set_major_formatter(plt.FixedFormatter(df[y].tolist()))
    ax.set_yticklabels(labels=df[y].tolist())
    ax.set_title(title)
    ax.grid(color='gray',linestyle='--',alpha=0.5)
    ax.tick_params(left=False,bottom=True,which='both')
    lgnd=ax.legend(loc='best',scatterpoints=1,numpoints=1,handletextpad=0.1,
                   labelspacing=0.3,  # Vertical space between labels
                   fontsize=10,markerscale=1,frameon=True) #scatteryoffsets=[0.5],
    try:
        # lgnd.legendHandles[0]._sizes = [10,10,10]
        # lgnd.legendHandles[1]._sizes = [10,10,10]
        for m in lgnd.legendHandles:
            m._legmarker.set_markersize(15)
            m._sizes = [10]
    except:
        pass
    fig.tight_layout()
    fig.savefig(outname)

def volcano_plot(data=None,x='log2(Fold change)',y='-log10(adjp)',
                 outname='output',title=None,hue='DEG',label='gene_name',
                 size='Size',sizes=(10, 80),xlabel='log2(Fold change)',ylabel='-log10(adjusted pvalue)',
                 rasterized=True,hue_order=['Upregulated','Not.Sig','Downregulated'],
                  colors=['red', 'grey', 'blue'],topn=10,x_unit=0.5,y_unit=5,
                  figsize=(5,7),xlabel_rotate=90):
    """
    Plot volcano plot

    Parameters
    ----------
    data :
    x :
    y :
    outname :
    title :
    hue :
    label :
    size :
    sizes :
    xlabel :
    ylabel :
    rasterized :
    hue_order :
    colors :
    topn :
    x_unit :
    y_unit :
    figsize :
    xlabel_rotate :

    Returns
    -------

    """
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(data=data, x=x, y=y, hue=hue,
                         linewidths=0.2, size=size, sizes=sizes,
                         hue_order=hue_order,
                         palette=sns.xkcd_palette(colors),
                         legend='full', rasterized=rasterized,
                         edgecolor='none')
    handles, labels = ax.get_legend_handles_labels()
    l = plt.legend(handles[:4], labels[:4], loc='upper left', bbox_to_anchor=(1, 0.7),
                   markerscale=2)
    ax.xaxis.set_major_locator(MultipleLocator(x_unit))
    ax.yaxis.set_major_locator(MultipleLocator(y_unit))
    plt.axis(np.array([data[x].min() - x_unit, data[x].max() + x_unit, 0, data[y].max() + y_unit]))
    ax.tick_params(axis='both', width=1.5)
    plt.setp(ax.get_xticklabels(), rotation=xlabel_rotate)  #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Annotate
    for i, (x0, y0,t) in data.loc[data[hue] == hue_order[0],
                          [x, y,label]].sort_values(y,ascending=False).head(topn).iterrows():
        # print(t,x0,y0)
        ax.annotate(text=t, xy=(x0, y0), xytext=(-0.3, 3), annotation_clip=False,
                    color=colors[0],
                    textcoords='offset points')  # connectionstyle='arc3,rad=0.5',arrowprops=dict(arrowstyle='-',lw=0.5,color='black')
    for i, (x0, y0,t) in data.loc[data[hue] == hue_order[2],
                          [x, y,label]].sort_values(y,ascending=False).head(topn).iterrows():
        # print(t,x0,y0)
        ax.annotate(text=t, xy=(x0, y0), xytext=(0.1, 3), annotation_clip=False,
                    color=colors[2], textcoords='offset points')  # arrowprops=dict(arrowstyle='-',lw=0.5,color='black')
    plt.tight_layout()
    plt.savefig(outname)
