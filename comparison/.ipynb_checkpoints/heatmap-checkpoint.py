import os,sys
import matplotlib.pylab as plt
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi']=300
from PyComplexHeatmap import *

# Read data
beta = pd.read_csv("beta.csv",sep="\t",index_col=0)
df_row = beta = pd.read_csv("df_row.csv",sep="\t",index_col=0)
df_col = beta = pd.read_csv("df_col.csv",sep="\t",index_col=0)

# define color mapping
tissue_col <- structure(c("#00E5FF","#6CBF00","#007F19","#FF0000"),names=c('Frontal Lobe Brain','Liver','Tail','Spleen'))
strain_col <- structure(c('#66AA9F','#8A6699','#D8A49C'),names=c('CAST_EiJ','MOLF_EiJ','PWK_PhJ'))

target_col <- structure(c('yellowgreen','orangered'), names = c(0,1))
snp_col <- structure(c('gray','black'), names = c(0,1))
group_col <- structure(c('darkorange','skyblue','red','wheat','green','darkgray'),names=c('Artificial high meth. reading','Artificial low meth. reading','G-R','No Effect','R-G','Suboptimal hybridization'))


row_ha = HeatmapAnnotation(Target=anno_simple(df_row.Target,colors=row_colors_dict['Target'],rasterized=True),
                           Group=anno_simple(df_row.Group,colors=row_colors_dict['Group'],rasterized=True),
                           axis=0)
col_ha= HeatmapAnnotation(label=anno_label(df_col.Strain,merge=True,rotation=15),
                          Strain=anno_simple(df_col.Strain,add_text=True),
                          Tissue=df_col.Tissue,Sex=df_col.Sex,
                          axis=1)
plt.figure(figsize=(5, 8))
cm = ClusterMapPlotter(data=beta, top_annotation=col_ha, left_annotation=row_ha,
                     show_rownames=False,show_colnames=False,
                     row_dendrogram=False,col_dendrogram=False,
                     row_split=df_row.loc[:, ['Target', 'Group']],
                     col_split=df_col['Strain'],cmap='parula',
                     rasterized=True,row_split_gap=1,legend=True,legend_anchor='ax_heatmap',legend_vpad=5)
cm.ax.set_title("Beta",y=1.03,fontdict={'fontweight':'bold'})
#plt.savefig("clustermap.pdf", bbox_inches='tight')
plt.show()