# !/usr/bin/env python3
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import PyComplexHeatmap
from .clustermap import *
import pandas as pd
import numpy as np
import collections
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def clustermap_example0():
    df = pd.DataFrame(['AAAA1'] * 5 + ['BBBBB2'] * 5, columns=['AB'])
    df['CD'] = ['C'] * 3 + ['D'] * 3 + ['G'] * 4
    df['EF'] = ['E'] * 6 + ['F'] * 2 + ['H'] * 2
    df['F'] = np.random.normal(0, 1, 10)
    df.index = ['sample' + str(i) for i in range(1, df.shape[0] + 1)]
    df_box = pd.DataFrame(np.random.randn(10, 4), columns=['Gene' + str(i) for i in range(1, 5)])
    df_box.index = ['sample' + str(i) for i in range(1, df_box.shape[0] + 1)]
    df_bar = pd.DataFrame(np.random.uniform(0, 10, (10, 2)), columns=['TMB1', 'TMB2'])
    df_bar.index = ['sample' + str(i) for i in range(1, df_box.shape[0] + 1)]
    df_scatter = pd.DataFrame(np.random.uniform(0, 10, 10), columns=['Scatter'])
    df_scatter.index = ['sample' + str(i) for i in range(1, df_box.shape[0] + 1)]
    df_heatmap = pd.DataFrame(np.random.randn(50, 10), columns=['sample' + str(i) for i in range(1, 11)])
    df_heatmap.index = ["Fea" + str(i) for i in range(1, df_heatmap.shape[0] + 1)]
    df_heatmap.iloc[1, 2] = np.nan

    # plt.figure(figsize=(6, 12))
    # row_ha = HeatmapAnnotation(label=anno_label(df.AB, merge=True),
    #                            AB=anno_simple(df.AB,add_text=True),axis=1,
    #                            CD=anno_simple(df.CD, colors={'C': 'red', 'D': 'yellow', 'G': 'green'},add_text=True),
    #                            Exp=anno_boxplot(df_box, cmap='turbo',
    #                                             # plot_kws={'edgecolor': 'black',
    #                                             #           'medianlinecolor': 'red'}
    #                                             ),
    #                            Scatter=anno_scatterplot(df_scatter), TMB_bar=anno_barplot(df_bar),
    #                            )
    # cm = ClusterMapPlotter(data=df_heatmap, top_annotation=row_ha, col_split=2, row_split=3, col_split_gap=0.5,
    #                      row_split_gap=1,col_dendrogram=False,plot=True,
    #                      tree_kws={'col_cmap': 'Set1', 'row_cmap': 'Dark2'})
    # # cm.plot_legend(ax=ax1)
    # # CC.ax_col_dendrogram.cla()
    # plt.savefig("/Users/dingw1/Desktop/20220412_test.pdf", bbox_inches='tight')
    # plt.show()

    plt.figure(figsize=(6, 4))
    row_ha = HeatmapAnnotation(label=anno_label(df.AB, merge=True),
                                AB=anno_simple(df.AB,add_text=True,legend=True), axis=1,
                                CD=anno_simple(df.CD, colors={'C': 'red', 'D': 'gray', 'G': 'yellow'},
                                               add_text=True,legend=True),
                                Exp=anno_boxplot(df_box, cmap='turbo',legend=True),
                                Scatter=anno_scatterplot(df_scatter), TMB_bar=anno_barplot(df_bar,legend=True),
                               plot=True,legend=True,plot_legend=True,
                               legend_gap=5
                                )
    # row_ha.plot_legend(row_ha.ax)
    plt.savefig("/Users/dingw1/Desktop/20220412_test_text.pdf", bbox_inches='tight')
    plt.show()

    plt.figure()
    row_ha.plot_legends()
    plt.savefig("/Users/dingw1/Desktop/20220412_test_text_legend.pdf",bbox_inches='tight')
    plt.show()

def clustermap_example1():
    import pickle
    f = open(os.path.join(PyComplexHeatmap._ROOT, 'data', "influence_of_snp_on_beta.pickle"), 'rb')
    data=pickle.load(f)
    f.close()
    beta,snp,df_row,df_col,col_colors_dict,row_colors_dict=data

    row_ha = HeatmapAnnotation(Target=anno_simple(df_row.Target,colors=row_colors_dict['Target'],rasterized=True),
                               Group=anno_simple(df_row.Group,colors=row_colors_dict['Group'],rasterized=True),
                               axis=0)
    col_ha= HeatmapAnnotation(label=anno_label(df_col.Strain,merge=True,rotation=15),
                              Strain=anno_simple(df_col.Strain,add_text=True),
                              Tissue=df_col.Tissue,Sex=df_col.Sex,axis=1) #df=df_col.loc[:,['Strain','Tissue','Sex']]
    plt.figure(figsize=(6, 10))
    cm = ClusterMapPlotter(data=beta, top_annotation=col_ha, left_annotation=row_ha,
                         show_rownames=False,show_colnames=False,
                         row_dendrogram=False,col_dendrogram=False,
                         row_split=df_row.loc[:, ['Target', 'Group']],
                         col_split=df_col['Strain'],cmap='parula',
                         rasterized=True,row_split_gap=1,legend=True,
                         tree_kws={'col_cmap':'Set1'}) #
    # cm = cm.plot()
    # cm.ax_heatmap.figure.tight_layout()
    # plt.tight_layout()
    plt.savefig("/Users/dingw1/Desktop/20220428_test.pdf", bbox_inches='tight')
    plt.show()

def get_kycg_example_data():
    import pandas as pd
    filelist = ['Clark2018_Argelaguet2019_neg.kycg', 'Clark2018_Argelaguet2019_pos.kycg',
                'GSE140493_Luo2022_hg38_neg.kycg', 'GSE140493_Luo2022_hg38_pos.kycg']
    ds = ['Clark2018_Argelaguet2019', 'Clark2018_Argelaguet2019', 'Luo2022', 'Luo2022']
    cg = ['Negative', 'Positive', 'Negative', 'Positive']
    data = ''
    for file, d, c in zip(filelist, ds, cg):
        df = pd.read_csv(file, sep='\t')
        df['Category'] = df['V5'].apply(
            lambda x: x.split('/')[-1].replace('.cg.gz', '').replace('.cg', '').split('~')[0])
        df['Term'] = df['V5'].apply(lambda x: x.split('/')[-1].replace('.cg.gz', '').replace('.cg', ''))
        df['Term'] = df.Term.apply(lambda x: x.split('~')[1] if '~' in x else x)
        df = df.loc[df.Term != 'NA', ['Term', 'p_val_enr', 'p_val_dep', 'odds_ratio', 'Category']]
        df['SampleID'] = d
        df['CpGType'] = c
        df1 = df.loc[(df['p_val_enr'] <= 0.05) & (df['odds_ratio'].apply(lambda x: np.log2(float(x))).abs() >= 1)]
        df2 = df.loc[(df['p_val_dep'] <= 0.05) & (df['odds_ratio'].apply(lambda x: np.log2(float(x))).abs() >= 1)]
        df1['pvalue'] = df1.p_val_enr
        df1['EnrichType'] = 'Enrich'
        df2['pvalue'] = df2.p_val_dep
        df2['EnrichType'] = 'Depletion'
        df1.drop(['p_val_enr', 'p_val_dep'], axis=1, inplace=True)
        df2.drop(['p_val_enr', 'p_val_dep'], axis=1, inplace=True)
        df = pd.concat([df1, df2])
        if type(data) == str:
            data = df.copy()
        else:
            data = pd.concat([data, df])

    data['-log10(Pval)'] = df['pvalue'].apply(lambda x: -np.log10(x + 1e-26))
    return data