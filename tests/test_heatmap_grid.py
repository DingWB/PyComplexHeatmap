import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import QuadMesh

from PyComplexHeatmap import heatmap
from PyComplexHeatmap.clustermap import plot_heatmap


class HeatmapGridTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def assert_grid_is_hidden(self, ax):
        ax.figure.canvas.draw()
        gridlines = [*ax.get_xgridlines(), *ax.get_ygridlines()]
        self.assertFalse(any(line.get_visible() for line in gridlines))

    def assert_cell_borders_remain(self, ax):
        meshes = [
            collection
            for collection in ax.collections
            if isinstance(collection, QuadMesh)
        ]
        self.assertEqual(len(meshes), 1)
        self.assertTrue(np.any(np.asarray(meshes[0].get_linewidths()) > 0))

    def test_heatmap_disables_inherited_axis_grid(self):
        with matplotlib.rc_context({"axes.grid": True}):
            _, ax = plt.subplots()
            heatmap(
                np.array([[1, 2], [3, 4]]),
                ax=ax,
                cbar=False,
                linewidths=0.5,
                linecolor="black",
            )

        self.assert_grid_is_hidden(ax)
        self.assert_cell_borders_remain(ax)

    def test_plot_heatmap_disables_inherited_axis_grid(self):
        with matplotlib.rc_context({"axes.grid": True}):
            _, ax = plt.subplots()
            plot_heatmap(
                np.array([[1, 2], [3, 4]]),
                ax=ax,
                cmap="viridis",
                linewidths=0.5,
                linecolor="black",
            )

        self.assert_grid_is_hidden(ax)
        self.assert_cell_borders_remain(ax)


if __name__ == "__main__":
    unittest.main()
