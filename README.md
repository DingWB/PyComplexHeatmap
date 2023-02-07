# PyComplexHeatmap
PyComplexHeatmap is a Python package to plot complex heatmap (clustermap). Please click [here](https://dingwb.github.io/PyComplexHeatmap) for documentation.

## Documentation:
[https://dingwb.github.io/PyComplexHeatmap](https://dingwb.github.io/PyComplexHeatmap)

## Dependencies:
- matplotlib>=3.4.3
- numpy
- pandas
```
pip install --ignore-install matplotlib==3.5.1 numpy==1.20.3 pandas==1.4.1
```

## **Installation**
1. **Install the developmental version directly from github**:
```
pip install git+https://github.com/DingWB/PyComplexHeatmap
```
if you have installed it previously and want to update it, please run 
`pip uninstall PyComplexHeatmap`
and install from github again
OR
```
git clone https://github.com/DingWB/PyComplexHeatmap
cd PyComplexHeatmap
python setup.py install
```
2. **Install with pip**:
```shell
pip install --ignore-installed PyComplexHeatmap
```

## **Usage**
### **1. Simple Guide To Get started.**
```
from PyComplexHeatmap import *

#Generate example dataset (random)
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

plt.figure(figsize=(6, 12))
col_ha = HeatmapAnnotation(label=anno_label(df.AB, merge=True,rotation=15),
                           AB=anno_simple(df.AB,add_text=True),axis=1,
                           CD=anno_simple(df.CD,add_text=True,colors={'C':'red','D':'green','G':'blue'},
                                            legend_kws={'frameon':False}),
                           Exp=anno_boxplot(df_box, cmap='turbo'),
                           Gene1=anno_simple(df_box.Gene1,vmin=0,vmax=1,legend_kws={'vmin':0,'vmax':1}),
                           Scatter=anno_scatterplot(df_scatter), 
                           TMB_bar=anno_barplot(df_bar,legend_kws={'color_text':False,'labelcolor':'blue'}),
                           )
cm = ClusterMapPlotter(data=df_heatmap, top_annotation=col_ha, col_split=2, row_split=3, col_split_gap=0.5,
                     row_split_gap=1,label='values',row_dendrogram=True,show_rownames=True,show_colnames=True,
                     tree_kws={'row_cmap': 'Dark2'},legend_gap=8,legend_width=6)
# cm.ax_heatmap.set_axis_off()
plt.show()
```
### Example output
![image](docs/images/1.png)
![image](docs/images/2.png)
![image](docs/images/3.png)
![image](docs/images/4.png)
![image](docs/images/5.png)

## **More Examples**
https://github.com/DingWB/PyComplexHeatmap/blob/main/notebooks/examples.ipynb
Or
https://dingwb.github.io/PyComplexHeatmap/build/html/documentation.html
