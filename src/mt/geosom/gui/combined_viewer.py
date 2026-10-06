# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
CombinedViewer: everything about a trained SOM in one window -- the map of the whole attribute vectors
at the top, and one map per attribute in a grid below it, all kept in step (needs matplotlib).

    from mt.geosom.gui import CombinedViewer

    CombinedViewer(som, data, labels=labels, feature_names=names).show()

This is what explore() and `python -m mt.geosom` open by default (layout='combined').

    window.map      the map of the whole attribute vectors, with its layers and neuron inspector
                    (SOMViewer for a GeoSOM, PlaneSOMViewer for a PlaneSOM)
    window.matrix   one map per attribute (ComponentMatrixViewer / PlaneComponentMatrixViewer)

Kept in step: dragging any map rotates every map (GeoSOM); clicking a neuron on any map selects it on all
of them; switching the projection in either part switches both.  Keys act on the part of the window under
the mouse pointer (the rotation they cause is then followed by the other part).
"""
from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mt.geosom.GeoSOM import GeoSOM
from mt.geosom.gui._figure import save_figure
from mt.geosom.gui.inspector import LAYER_DISTANCE
from mt.geosom.gui.link import link_views
from mt.geosom.PlaneSOM import PlaneSOM


class CombinedViewer:
    """
    One window: the map of a trained GeoSOM or PlaneSOM on top, its component maps in a grid below,
    linked (rotation, selection, projection).

    :param som: a trained GeoSOM or PlaneSOM
    :param data, labels, feature_names: optional; data enables hits and sample dots, labels the class map
    :param layer, component, show_samples, node_labels, sample_names, label_size: for the top map, as for SOMViewer
    :param matrix_mode: 'difference' (default) or 'value', for the component maps
    :param include_total, shared_scale, ncols: for the component maps, as for ComponentMatrixViewer
                                               (ncols default: whatever makes the maps largest)
    :param projection, view: (GeoSOM) map projection and initial (lat, lon) centre in degrees
    :param link: keep the two parts in step (default True)
    :param map_height: fraction of the window height given to the top map (default 0.48)
    :param figsize: window size in inches (default (16, 9.5))
    :param title: window title (default: describes the SOM)
    """

    def __init__(self, som, data=None, labels: Sequence | None = None, feature_names: Sequence[str] | None = None,
                 *, layer: str = LAYER_DISTANCE, component: int = 0, show_samples: bool | None = None,
                 node_labels: str | None = None, sample_names: Sequence | None = None,
                 label_size: float | None = None, matrix_mode: str = 'difference', include_total: bool = True,
                 shared_scale: bool = False, ncols: int | None = None, projection: str = 'Equal Earth',
                 view: Sequence[float] | None = None, link: bool = True, map_height: float = 0.48,
                 figsize=(16, 9.5), title: str | None = None):
        if not 0.2 <= map_height <= 0.8:
            raise ValueError('map_height must be between 0.2 and 0.8')
        self.som = som
        self.fig = plt.figure(figsize=figsize)
        top, bottom = self.fig.subfigures(2, 1, height_ratios=[map_height, 1.0 - map_height])
        self.map_area, self.matrix_area = top, bottom
        map_options = dict(labels=labels, feature_names=feature_names, layer=layer, component=component,
                           show_samples=show_samples, node_labels=node_labels, sample_names=sample_names,
                           label_size=label_size, fig=top)
        matrix_options = dict(mode=matrix_mode, include_total=include_total, shared_scale=shared_scale,
                              ncols=ncols, data=data, labels=labels, fig=bottom)
        if isinstance(som, GeoSOM):
            from mt.geosom.gui.matrix_viewer import ComponentMatrixViewer
            from mt.geosom.gui.som_viewer import SOMViewer
            self.map = SOMViewer(som, data, projection=projection, view=view, **map_options)
            self.matrix = ComponentMatrixViewer(som, feature_names, projection=projection, view=view,
                                                **matrix_options)
            kind = f'GeoSOM: {som.n_nodes} neurons on GeodesicDome({som.dome.frequency})'
        elif isinstance(som, PlaneSOM):
            from mt.geosom.gui.plane_viewer import PlaneComponentMatrixViewer, PlaneSOMViewer
            self.map = PlaneSOMViewer(som, data, **map_options)
            self.matrix = PlaneComponentMatrixViewer(som, feature_names, **matrix_options)
            kind = f'PlaneSOM: {som.row} x {som.col} {som.lattice.name.lower()} grid'
        else:
            raise TypeError(f'CombinedViewer shows a GeoSOM or a PlaneSOM, not {type(som).__name__}')
        self.map.window = self.matrix.window = self
        self.linked = bool(link)
        if self.linked:
            link_views(self.map, self.matrix)

        # a thin line between the two parts
        y = 1.0 - map_height
        self.fig.add_artist(Line2D([0.01, 0.99], [y, y], transform=self.fig.transFigure, color='#bbbbbb',
                                   lw=0.8, zorder=10))
        manager = self.fig.canvas.manager
        if manager is not None:
            manager.set_window_title(title or f'mt.geosom -- {kind}, {som.dim} attributes')

        # while one part is dragged, the other part follows by blitting (fast) instead of redrawing the window
        self._follower = None
        if self.linked and hasattr(self.map, 'follow_drag') and hasattr(self.matrix, 'follow_drag'):
            self.fig.canvas.mpl_connect('button_press_event', self._on_press)
            self.fig.canvas.mpl_connect('button_release_event', self._on_release)

    # ================================================================ public API
    @property
    def views(self):
        """(map, matrix)"""
        return self.map, self.matrix

    def show(self):
        plt.show()

    def save(self, path, **kwargs):
        """Saves the whole window to an image file (window.map.save / window.matrix.save save one part)."""
        save_figure(self.fig, path, **kwargs)

    @property
    def selected(self):
        return self.map.selected

    def select_node(self, node: int | None):
        """Selects a neuron on every map (None clears)."""
        self.map.select_node(node)
        if not self.linked:
            self.matrix.select_node(node)

    def __getattr__(self, name):
        # rotation and projection (GeoSOM): set_view, rotate, set_rotation, reset, set_projection, rotation, ...
        # go to the top map, which the matrix follows
        if name in ('map', 'matrix') or name.startswith('_'):
            raise AttributeError(name)
        return getattr(self.map, name)

    # ========================================================== event handlers
    def _on_press(self, event):
        if event.button != 1 or event.inaxes is None or event.dblclick:
            return
        if event.inaxes is getattr(self.map, 'ax', None):
            self._follower = self.matrix
        elif any(event.inaxes is p['ax'] for p in self.matrix.panels):
            self._follower = self.map
        else:
            return
        self._follower.follow_drag(True)

    def _on_release(self, _event):
        if self._follower is not None:
            follower, self._follower = self._follower, None
            follower.follow_drag(False)
