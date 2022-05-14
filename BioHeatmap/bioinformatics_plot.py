# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import os, sys
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np

def volcano_plot(data=None,x='log2(Fold change)',y='-log10(adjp)',
                 outname='output',title=None,hue='DEG',label='gene_name',
                 size='Size',sizes=(10, 80),xlabel='log2(Fold change)',ylabel='-log10(adjusted pvalue)',
                 rasterized=True,hue_order=['Upregulated','Not.Sig','Downregulated'],
                  colors=['red', 'grey', 'blue'],topn=10,x_unit=0.5,y_unit=5,
                  figsize=(5,7),xlabel_rotate=90):
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