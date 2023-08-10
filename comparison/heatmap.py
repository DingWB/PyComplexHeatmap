import os, sys
import matplotlib.pylab as plt

plt.rcParams["figure.dpi"] = 100
plt.rcParams["savefig.dpi"] = 300
# sys.path.append(os.path.expanduser("~/Projects/Github/PyComplexHeatmap/"))
from PyComplexHeatmap import *

# Read data
beta = pd.read_csv("beta.csv", sep="\t", index_col=0)
df_row = pd.read_csv("df_row.csv", sep="\t", index_col=0)
df_col = pd.read_csv("df_col.csv", sep="\t", index_col=0)

# define color mapping
tissue_col = {
    t: c
    for t, c in zip(
        ["Frontal Lobe Brain", "Liver", "Tail", "Spleen"],
        ["#00E5FF", "#6CBF00", "#007F19", "#FF0000"],
    )
}
strain_col = {
    t: c
    for t, c in zip(
        ["CAST_EiJ", "MOLF_EiJ", "PWK_PhJ"], ["#66AA9F", "#8A6699", "#D8A49C"]
    )
}
target_col = {t: c for t, c in zip([0, 1], ["yellowgreen", "orangered"])}
group_col = {
    t: c
    for t, c in zip(
        [
            "Artificial high meth. reading",
            "Artificial low meth. reading",
            "G-R",
            "No Effect",
            "R-G",
            "Suboptimal hybridization",
        ],
        ["darkorange", "skyblue", "red", "wheat", "green", "darkgray"],
    )
}

row_ha = HeatmapAnnotation(
    Target=anno_simple(df_row.Target, colors=target_col, rasterized=True),
    Group=anno_simple(df_row.Group, colors=group_col, rasterized=True),
    axis=0,
    verbose=0,
)
col_ha = HeatmapAnnotation(  # label=anno_label(df_col.Strain,colors=strain_col,merge=True,rotation=15),
    Strain=anno_simple(df_col.Strain, colors=strain_col, add_text=True),
    Tissue=anno_simple(df_col.Tissue, colors=tissue_col),
    axis=1,
    verbose=0,
)
plt.figure(figsize=(4, 8))
cm = ClusterMapPlotter(
    data=beta,
    top_annotation=col_ha,
    left_annotation=row_ha,
    show_rownames=False,
    show_colnames=False,
    row_dendrogram=False,
    col_dendrogram=False,
    row_split=df_row.loc[:, ["Target", "Group"]],
    col_split=df_col["Strain"],
    cmap="parula",
    rasterized=True,
    row_split_gap=0,
    legend=True,
    legend_anchor="ax_heatmap",
    legend_vpad=5,
    label="beta",
    verbose=0,
)
cm.ax.set_title("Beta", y=1.03, fontdict={"fontweight": "bold"})
plt.savefig("PyComplexHeatmap.pdf", bbox_inches="tight")
