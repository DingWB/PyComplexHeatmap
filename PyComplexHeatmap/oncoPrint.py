# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib
import matplotlib.pylab as plt
from .utils import _draw_figure,mm2inch,plot_legend_list,despine,_index_to_ticklabels,get_colormap
from .clustermap import ClusterMapPlotter
from .annotations import HeatmapAnnotation, anno_barplot
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# =============================================================================
def oncoprint(data,ax=None,colors=None, cmap='Set1',nvar=None,
              aspect=None,bgcolor='whitesmoke',row_gap=1,
              xticklabels_kws=None,subplot_spec=None,
              yticklabels_kws=None,**plot_kws):
    """
    Plot dot heatmap using a dataframe matrix as input.

    Parameters
    ----------
    data : pd.DataFrame
        input matrix (pandas.DataFrame), for each element in this dataframe, the value should be a list.
    ax : ax
        ax
    colors : dict
        colors to control the dot, keys should be the value in hue. if colors is a str, then colors will overwrite
        the parameter `c`.
    cmap : str of dict
        control the colormap of the dot, if cmap is a dict, keys should be the values from hue dataframe.
        If `cmap` is a str (such as 'Set1'), the parameter `colors` will overwrite the colors of dots.
        If `cmap` wisas a dict, then this paramter will overwrite the `colors`, and colors can only control the
        colors for markers.
    bgcolor: str
        background color, default is whitesmoke (#CCCCCC)
    kwargs : dict
        such as s,c,marker, s,marker and colors can also be pandas.DataFrame.
        other kwargs passed to plt.scatter

    Returns
    -------
    ax,axes:
    """
    if ax is None:
        ax = plt.gca()
    nrows, ncols = data.shape
    if nvar is None:
        nvar=data.iloc[:,0].apply(lambda x:len(x)).max()
    if colors is None:
        colors=[get_colormap(cmap)(i) for i in range(nvar)]
    plot_kws.setdefault('width',0.7)
    plot_kws.setdefault('align', 'center')
    rowticklabels=_index_to_ticklabels(data.index)
    colticklabels=_index_to_ticklabels(data.columns)
    ax.set_ylim(0, nrows)
    y_locater = list(np.arange(0.5, nrows, 1))
    ax.yaxis.set_major_locator(plt.FixedLocator(y_locater))
    ax.set_yticklabels(rowticklabels)
    ax.invert_yaxis()

    ax.set_xlim(0, ncols)
    x_locater = list(np.arange(0.5, ncols, 1))
    ax.xaxis.set_major_locator(plt.FixedLocator(x_locater))
    ax.set_xticklabels(colticklabels)
    ax.tick_params(axis='both', which='both',
                   left=False, right=False, labelleft=False, labelright=False,
                   top=False, bottom=False, labeltop=False, labelbottom=False)
    xticklabels_kws = {} if xticklabels_kws is None else xticklabels_kws
    yticklabels_kws = {} if yticklabels_kws is None else yticklabels_kws
    xticklabels_kws.setdefault('labelrotation', -90)
    xticklabels_kws.setdefault('labelbottom', True)
    yticklabels_kws.setdefault('labelleft', True)
    ax.xaxis.set_tick_params(**xticklabels_kws)
    ax.yaxis.set_tick_params(**yticklabels_kws)
    despine(ax=ax, left=True, bottom=True, right=True, top=True)

    hspace = row_gap * mm2inch * ax.figure.dpi / (ax.get_window_extent().height / nrows)
    if subplot_spec is None:
        gs = ax.figure.add_gridspec(nrows, 1, hspace=hspace)
    else:
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows, 1, hspace=hspace,
                                                     subplot_spec=subplot_spec)
    axes = np.empty(shape=(nrows, 1), dtype=object)
    # ax.set_axis_off()
    for i in range(nrows):
        df=pd.DataFrame(data.iloc[i,:].values.tolist()).apply(lambda x:x/x.sum(),axis=1).fillna(0)
        ax1 = ax.figure.add_subplot(gs[i, 0], sharex=ax)
        ax1.set_ylim([0,1])
        ax1.yaxis.set_major_locator(plt.FixedLocator([0.5]))
        ax1.set_yticklabels([rowticklabels[i]])
        if not aspect is None:
            ax1.set_aspect(aspect)
        base_coordinates = [0] * ncols
        x=np.arange(0.5, ncols, 1)
        for j,color in zip(range(df.shape[1]),colors):
            ax1.bar(x=x, height=df.iloc[:,j].values,
                   bottom=base_coordinates, color=color, **plot_kws)
            base_coordinates = df.iloc[:,j].values + base_coordinates
        #plot background colors
        df=df.loc[df.sum(axis=1)==0]
        ax1.bar(x=df.index.values+0.5, height=[1]*df.shape[0],
                color=bgcolor, **plot_kws)
        # ax1.tick_params(axis='both', which='both',
        #                 left=False, right=False, labelleft=False, labelright=False,
        #                 top=False, bottom=False, labeltop=False, labelbottom=False)
        # despine(ax=ax1, left=True, bottom=True, right=True, top=True)
        ax1.set_axis_off()
        axes[i,0]=ax1
        #secondary y tick labels: axes2 = axes1.twinx() #mirror
    _draw_figure(ax.figure)
    return ax,axes
# =============================================================================
class oncoPrintPlotter(ClusterMapPlotter):
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
    values : str or list
        The column names in data.columns to control the sizes, or color of scatter (dot).
    colors :list.
        colors for each column in values
    cmap :str or dict, optional.
        If cmap is a dict, the keys should be the values from data[hue].values, and values should be cmap.
        If cmap is a string, it should be colormap, such as 'Set1'.
    bgcolor: str
        background color, default is whitesmoke (or lightgray)
    color_legend_kws: dict
        legend_kws passed to plot_color_dict_legend
    cmap_legend_kws: dict
        legend_kws passed to plot_cmap_legend
    kwargs :dict
        Other kwargs passed to ClusterMapPlotter and oncoprint.

    Returns
    -------
    oncoPrintPlotter.
    """

    def __init__(self, data=None, x=None, y=None, values=None, cmap='Set1',colors=None,
                 aspect=None, bgcolor='whitesmoke',row_gap=0.8,
                 color_legend_kws={},cmap_legend_kws={},
                 remove_empty_rows=True,remove_empty_columns=True,
                 **kwargs):
        kwargs['data']=data
        self.x=x
        self.y=y
        self.values=values if type(values)==list else [values]
        self.cmap=cmap
        self.colors=colors
        self.aspect=aspect
        self.bgcolor=bgcolor
        self.row_gap=row_gap
        self.color_legend_kws=color_legend_kws
        self.cmap_legend_kws=cmap_legend_kws
        self.remove_empty_rows=remove_empty_rows
        self.remove_empty_cols=remove_empty_columns

        super().__init__(**kwargs)

    def format_data(self, data, mask=None,z_score=None, standard_scale=None):
        data2d = data.assign(VALUE=data.loc[:, self.values].apply(lambda x: x.tolist(), axis=1))\
            .pivot(index=self.y,columns=self.x,values='VALUE')
        df_sum=data2d.applymap(np.sum)
        if self.remove_empty_rows:
            data2d=data2d.loc[df_sum.sum(axis=1)>0]
        if self.remove_empty_cols:
            data2d=data2d.loc[:,df_sum.sum(axis=0)>0]
        row_vc = data2d.apply(lambda x: x.apply(np.array).sum(), axis=1)
        self.row_vc = pd.DataFrame(row_vc.tolist(), index=row_vc.index.tolist(), columns=self.values)
        self.col_vc = data2d.apply(lambda x: x.apply(np.array).sum(), axis=0).T
        self.col_vc.columns=self.values
        if self.colors is None:
            self.colors = [get_colormap(self.cmap)(i) for i in range(len(self.values))]
        self.color_dict = {}
        for label, color in zip(self.values, self.colors):
            self.color_dict[label] = color
        return data2d

    def _reorder_rows(self):
        if self.verbose >= 1:
            print("Reordering rows..")
        if self.row_split is None and self.row_cluster:
            self.row_order = [self.row_vc.sum(axis=1).sort_values(ascending=False).index.tolist()]
            return None
        elif isinstance(self.row_split, int) and self.row_cluster:
            self.calculate_row_dendrograms(self.data2d.applymap(lambda x:np.sum(x)))
            self.row_clusters = pd.Series(hierarchy.fcluster(self.dendrogram_row.linkage, t=self.row_split,
                                                             criterion='maxclust'),
                                          index=self.data2d.index.tolist()).to_frame(name='cluster')\
                .groupby('cluster').apply(lambda x: x.index.tolist()).to_dict()
            #index=self.dendrogram_row.dendrogram['ivl']).to_frame(name='cluster')

        elif isinstance(self.row_split, (pd.Series, pd.DataFrame)):
            if isinstance(self.row_split, pd.Series):
                self.row_split = self.row_split.to_frame(name=self.row_split.name).loc[self.data2d.index.tolist()]
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
                row_order=self.row_vc.loc[rows].sum(axis=1).sort_values(ascending=False).index.tolist()
                self.row_order.append(row_order)
            else:
                self.row_order.append(rows)

    def get_samples_order(self,data,row_order):
        """
        data is a dataframe, row_order is a list ([[],[]]).
        """
        nrows = data.shape[0]
        row_orders = list(np.array(row_order).flatten())
        # https://gist.github.com/armish/564a65ab874a770e2c26
        col_order = data.apply(
            lambda x: x.apply(lambda y: 0 if np.sum(y) == 0 else 2 ** (nrows - row_orders.index(x.name) - 1)), axis=1) \
            .sum().sort_values(ascending=False).index.tolist()
        return col_order

    def _reorder_cols(self):
        if self.verbose >= 1:
            print("Reordering cols..")
        if self.col_split is None and self.col_cluster:
            col_order=self.get_samples_order(self.data2d,self.row_order)
            self.col_order = [col_order]  # self.data2d.iloc[:, xind].columns.tolist()
            return None
        elif isinstance(self.col_split, int) and self.col_cluster:
            self.calculate_col_dendrograms(self.data2d.applymap(lambda x:np.sum(x)))
            self.col_clusters = pd.Series(hierarchy.fcluster(self.dendrogram_col.linkage, t=self.col_split,
                                                             criterion='maxclust'),
                                          index=self.data2d.columns.tolist()).to_frame(name='cluster')\
                .groupby('cluster').apply(lambda x: x.index.tolist()).to_dict()
            #index=self.dendrogram_col.dendrogram['ivl']).to_frame(name='cluster')

        elif isinstance(self.col_split, (pd.Series, pd.DataFrame)):
            if isinstance(self.col_split, pd.Series):
                self.col_split = self.col_split.to_frame(name=self.col_split.name).loc[self.data2d.columns.tolist()]
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
                col_order=self.get_samples_order(self.data2d.loc[:,cols],self.row_order)
                self.col_order.append(col_order)
            else:
                self.col_order.append(cols)

    def plot_matrix(self, row_order, col_order):
        if self.verbose >= 1:
            print("Plotting matrix..")
        nrows = len(row_order)
        ncols = len(col_order)
        self.wspace = self.col_split_gap * mm2inch * self.ax.figure.dpi / (
                self.ax_heatmap.get_window_extent().width / ncols)  # 1mm=mm2inch inch
        self.hspace = self.row_split_gap * mm2inch * self.ax.figure.dpi / (
                self.ax_heatmap.get_window_extent().height / nrows)
        self.heatmap_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(nrows, ncols, hspace=self.hspace,
                                                                      wspace=self.wspace,
                                                                      subplot_spec=self.gs[1, 1],
                                                                      height_ratios=[len(rows) for rows in row_order],
                                                                      width_ratios=[len(cols) for cols in col_order])
        self.heatmap_axes = np.empty(shape=(nrows, ncols), dtype=object)
        self.heatmap_subaxes = np.empty(shape=(nrows, ncols), dtype=object)
        self.ax_heatmap.set_axis_off()
        for i, rows in enumerate(row_order):
            for j, cols in enumerate(col_order):
                ax1 = self.ax_heatmap.figure.add_subplot(self.heatmap_gs[i, j],
                                                        sharex=self.heatmap_axes[0, j],
                                                        sharey=self.heatmap_axes[i, 0])
                kwargs=self.kwargs.copy()
                ax2,axes=oncoprint(self.data2d.loc[rows, cols],colors=self.colors,nvar=len(self.values),
                          aspect=self.aspect, bgcolor=self.bgcolor, row_gap=self.row_gap,
                         cmap=kwargs.pop('cmap',self.cmap),ax=ax1,subplot_spec=self.heatmap_gs[i, j],
                         **kwargs)
                self.heatmap_axes[i, j] = ax1
                self.heatmap_subaxes[i,j]=axes
                ax1.yaxis.label.set_visible(False)
                ax1.xaxis.label.set_visible(False)
                ax1.tick_params(which='both',left=False, right=False, labelleft=False, labelright=False,
                                top=False, bottom=False, labeltop=False, labelbottom=False)

    def add_default_annotations(self):
        if self.top_annotation is None:
            self.top_annotation=HeatmapAnnotation(axis=1,
                Col=anno_barplot(self.col_vc,colors=self.colors,legend=False),
                verbose=0, label_side='left', label_kws={'horizontalalignment': 'right'})
        else:
            col_ann = anno_barplot(self.col_vc, colors=self.colors, legend=False)
            col_ann.set_label('Col')
            self.top_annotation.annotations.append(col_ann)
        if self.right_annotation is None:
            self.right_annotation = HeatmapAnnotation(axis=0,
                Row=anno_barplot(self.row_vc,colors=self.colors,legend=False),
                verbose=0, label_side='top', label_kws={'horizontalalignment': 'right'})
        else:
            row_ann = anno_barplot(self.row_vc, colors=self.colors, legend=False)
            row_ann.set_label('Row')
            self.right_annotation=[row_ann]+self.right_annotation.annotations

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
            self.legend_list.append(
                [self.color_dict, self.label, self.color_legend_kws, len(self.color_dict), 'color_dict'])
            heatmap_label_max_width = max([label.get_window_extent().width for label in self.yticklabels]) if len(
                self.yticklabels) > 0 and self.row_names_side=='right' else 0
            if heatmap_label_max_width >= self.label_max_width or self.legend_anchor == 'ax_heatmap':
                self.label_max_width = heatmap_label_max_width #* 1.1
            if len(self.legend_list) > 1:
                self.legend_list = sorted(self.legend_list, key=lambda x: x[3])

    def post_processing(self):
        pass

if __name__ == "__main__":
    pass