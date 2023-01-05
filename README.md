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
1. **Install from github**:
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
pip install PyComplexHeatmap (TODO)
```

![image](https://github.com/DingWB/PyComplexHeatmap/blob/main/docs/images/output_19_1.png)
![image](https://github.com/DingWB/PyComplexHeatmap/blob/main/docs/images/output_19_1.png)
![image](https://github.com/DingWB/PyComplexHeatmap/blob/main/docs/images/output_21_1.png)
![image](https://github.com/DingWB/PyComplexHeatmap/blob/main/docs/images/output_23_1.png)
![image](https://github.com/DingWB/PyComplexHeatmap/blob/main/docs/images/output_25_1.png)

## **Usage**
### **1. Simple Guide To Get started.**
```
from PyComplexHeatmap import *

# Generate fake dataset
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

# Define annotation and plot
row_ha = HeatmapAnnotation(label=anno_label(df.AB, merge=True),
                            AB=anno_simple(df.AB,add_text=True),axis=1,
                            CD=anno_simple(df.CD, colors={'C': 'red', 'D': 'yellow', 'G': 'green'},add_text=True),
                            Exp=anno_boxplot(df_box, cmap='turbo'),
                            Scatter=anno_scatterplot(df_scatter), TMB_bar=anno_barplot(df_bar))
plt.figure(figsize=(6, 12))                         
cm = ClusterMapPlotter(data=df_heatmap, top_annotation=row_ha, col_split=2, 
                       row_split=3, col_split_gap=0.5,row_split_gap=1,
                      tree_kws={'col_cmap': 'Set1', 'row_cmap': 'Dark2'})
plt.savefig("heatmap.pdf", bbox_inches='tight')
plt.show()
```

## **Examples**
https://github.com/DingWB/PyComplexHeatmap/blob/main/notebooks/examples.ipynb
