# Benchmark Dataset
We use the dataset obtained from PMID: 36617464 as test dataset to compare the performance between ComplexHeatmap and PyComplexHeatmap.
<br>
This dataset include 29,827 rows and 28 columns. Here, we generated the same heatmap using the same clustering method and metric to compare the processing time and memory usage between these two packages.

# Comparison Result
| Package Name     | Processing Time (s) | Memory (kb) |
| ---------------- | ------------------- | ----------- |
| ComplexHeatmap   | 40.21               | 3,366,768   |
| PyComplexHeatmap | 22.57               | 1,037,944   |

# Outputs
<table>
    <tr>
        <th>ComplexHeatmap</th>
        <th>PyComplexHeatmap</th>
  </tr>
    <tr style="height: 800px">
        <td style="width:33%; background-color:white;text-align:center; vertical-align:middle">
            <a href="heatmap.R">
                <img src="ComplexHeatmap.png" title="ComplexHeatmap" align="center" width="375px">
            </a>
        </td>
        <td style="width:33%; background-color:white;text-align:center; vertical-align:middle">
            <a href="heatmap.py">
                <img src="PyComplexHeatmap.png" title="PyComplexHeatmap" align="center" width="375px">
            </a>
        </td>
    </tr>
</table>