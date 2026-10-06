# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
SOMViewer: explore a trained GeoSOM on a rotatable map of the sphere.

    from mt.geosom.GeoSOM import GeoSOM
    from mt.geosom.gui import SOMViewer

    som = GeoSOM(8).train(data)
    SOMViewer(som, data, labels=labels, feature_names=names).show()

Built on mt.geodesicdome.interactive.ProjectionViewer, so the map rotates the same way.  The layers
and the neuron inspector are described in mt.geosom.gui.inspector.

Mouse and keyboard
    drag                 rotate the sphere            click   inspect the neuron under the pointer
    [ / ]                previous / next attribute    s       show or hide the samples
    plus ProjectionViewer's keys: arrows, , . (roll), r (reset), p (projection), g (grid), e (edges)

Given `fig=` (a matplotlib Figure or SubFigure) it draws into that instead of opening a window of its
own; CombinedViewer uses this to put the map and the component matrix in one window.  Keys then act on
the viewer under the mouse pointer.
"""
from collections.abc import Callable, Sequence

import numpy as np
from mt.geodesicdome.interactive.viewer import ProjectionViewer

from mt.geosom.GeoSOM import GeoSOM
from mt.geosom.gui._figure import figure_redirected, owns_event, save_figure
from mt.geosom.gui.inspector import LAYER_DISTANCE, SOMInspector


class SOMViewer(SOMInspector, ProjectionViewer):
    """
    Interactive, rotatable map of a trained GeoSOM.

    :param som: a trained GeoSOM
    :param data: optional (samples, attributes) data, for hits, sample markers and classes
    :param labels: optional label per sample (class map and marker colours)
    :param feature_names: optional name per attribute
    :param layer: the layer to show first (default 'Neighbour distance')
    :param component: the attribute shown first by the Component layers
    :param show_samples: draw each sample as a dot at its best matching unit (default: when data is given)
    :param node_labels: start with neuron labels on, in this style: 'majority', 'all', 'counts' or 'first'
                        (default None: off). The 'neuron labels' check box, `l`, and show_node_labels() turn them
                        on and off; the 'labels:' button, `L` and set_node_labels() change the style
    :param sample_names: optional text per sample for the labels instead of `labels` (e.g. sample names);
                         given on its own, labels start on in the 'first' style
    :param label_size: font size of the neuron labels in points (default 7); A-/A+ buttons, - / + keys and
                       set_label_size() change it later
    :param projection, view, graticule, edges, figsize, title: as for ProjectionViewer ('' for no title)
    :param fig: optional matplotlib Figure or SubFigure to draw into (default: a new window of `figsize`)
    """

    def __init__(self, som: GeoSOM, data=None, labels: Sequence | None = None,
                 feature_names: Sequence[str] | None = None, *, layer: str = LAYER_DISTANCE,
                 component: int = 0, show_samples: bool | None = None, node_labels: str | None = None,
                 sample_names: Sequence | None = None, label_size: float | None = None, projection=None,
                 view=None, graticule: bool = True, edges: bool | None = False,
                 figsize=(16, 8), title: str | None = None, fig=None):
        if not isinstance(som, GeoSOM):
            raise TypeError('SOMViewer shows a GeoSOM; use PlaneSOMViewer for a PlaneSOM')
        self._init_inspector(som, data, labels, feature_names, component, show_samples, sample_names,
                             label_size)
        if title is None:
            title = (f'GeoSOM: {som.n_nodes} neurons on GeodesicDome({som.dome.frequency}), '
                     f'{som.dim} attributes each')
        self._given_fig = fig
        self._following = False                   # True while another view in the same window is dragged
        self._projection_callbacks: list[Callable] = []
        ProjectionViewer.__init__(self, som.dome, projection, colors='position', edges=edges,
                                  graticule=graticule, view=view, title=title, figsize=figsize)
        if fig is None and self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title('mt.geosom SOMViewer')
        if not np.array_equal(self.map.faces, som.faces):
            raise RuntimeError('the map and the SOM disagree about the dome faces')
        self._build_inspector()
        box = self.ax.get_position()                       # ProjectionViewer's status line: under the map
        self._info.set_x(box.x0 + box.width / 2)

        # sample dots, jittered around their best matching unit and kept on the sphere
        if self.data is not None:
            pos = som.points[self.bmu]
            jitter = np.random.default_rng(0).normal(scale=0.28 * som.ring_length, size=pos.shape)
            jitter -= (jitter * pos).sum(1, keepdims=True) * pos             # tangent to the sphere
            pos = pos + jitter
            self._sample_xyz = pos / np.linalg.norm(pos, axis=1, keepdims=True)
            self._samples = self.ax.scatter(*self._project(self._sample_xyz).T, s=7, c=self._sample_colours(),
                                            edgecolors='white', linewidths=0.3, zorder=4)
            self._samples.set_clip_path(self._outline)
        else:
            self._samples = None
        self._ready = True
        self.on_rotate(lambda _viewer: self._update_overlays())
        self.set_layer(layer)
        if node_labels is not None or sample_names is not None:
            self.set_node_labels(node_labels or self.node_label_mode)

    # ================================================================ public API
    def node_at(self, x: float, y: float) -> int:
        """The neuron nearest to map coordinates (x, y) in the current view."""
        xy = self._project(self.som.points)
        return int(np.argmin(((xy - [x, y]) ** 2).sum(axis=1)))

    def save(self, path, **kwargs):
        """Saves the viewer to an image file (in a shared window: only this viewer's part of it)."""
        save_figure(self.fig, path, **kwargs)

    def on_projection(self, callback: Callable[['SOMViewer'], None]):
        """Registers callback(viewer), called when the map projection changes."""
        self._projection_callbacks.append(callback)

    def follow_drag(self, on: bool):
        """
        Another view in the same window is being dragged (on=True) or was released (False).  While on,
        rotations from the link are drawn quickly by blitting just this map, instead of redrawing the window.
        """
        self._following = bool(on)
        if not on and self._background is not None and self._drag_from is None:
            self._background = None
            self.update()                                    # full quality again

    def update(self):
        if getattr(self, '_following', False) and self._background is None and self._drag_from is None:
            self._background = self._grab_background()
        super().update()

    # ======================================================= ProjectionViewer
    def _build_figure(self, title, figsize):
        with figure_redirected(self._given_fig):             # ProjectionViewer draws into the given figure
            super()._build_figure(title, figsize)

    def _redraw_outline(self):
        super()._redraw_outline()
        for callback in list(getattr(self, '_projection_callbacks', ())):
            callback(self)

    def _on_key(self, event):
        if owns_event(self.fig, event):
            super()._on_key(event)

    # ============================================================ SOMInspector
    def _show_layer_colours(self, values, cmap):
        rgba, norm = self._face_rgba(values, cmap)
        self.set_colors(rgba)
        return norm

    def _node_xy(self, nodes):
        return self._project(self.som.points[nodes])

    def _node_position_text(self, i):
        lat = np.degrees(np.arcsin(self.som.points[i, 2]))
        lon = np.degrees(np.arctan2(self.som.points[i, 1], self.som.points[i, 0]))
        return f'lat {lat:+.1f}  lon {lon:+.1f}'

    # ================================================================ overlays
    def _overlay_artists(self):
        return [a for a in (self._samples, self._marker, self._marker_dot) if a is not None] + self._label_texts

    def _project(self, xyz: np.ndarray) -> np.ndarray:
        p = xyz @ self.rotation.T
        lat = np.arcsin(np.clip(p[:, 2], -1.0, 1.0))
        lon = np.arctan2(p[:, 1], p[:, 0])
        return self.map.project(lat, lon)

    def _update_overlays(self):
        if not self._ready:
            return
        if self._samples is not None:
            self._samples.set_offsets(self._project(self._sample_xyz))
            self._samples.set_visible(self.show_samples)
        self._position_node_labels()
        if self.selected is not None:
            xy = self._project(self.som.points[[self.selected]])
            for m in (self._marker, self._marker_dot):
                m.set_data(xy[:, 0], xy[:, 1])
                m.set_visible(True)
        else:
            self._marker.set_visible(False)
            self._marker_dot.set_visible(False)
        if self._background is not None:                     # mid-drag: blit the overlays too
            for artist in self._overlay_artists():
                self.ax.draw_artist(artist)
            self.fig.canvas.blit(self.ax.bbox)

    def _grab_background(self):
        # the cached background must not contain the overlays, which move while dragging
        overlays = self._overlay_artists() if self._ready else []
        visible = [a.get_visible() for a in overlays]
        for a in overlays:
            a.set_visible(False)
        try:
            return super()._grab_background()
        finally:
            for a, v in zip(overlays, visible, strict=True):
                a.set_visible(v)
