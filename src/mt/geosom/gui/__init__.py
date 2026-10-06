# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
mt.geosom.gui -- interactive visualisation of trained SOMs (needs matplotlib:
`pip install "geosom[interactive]"`).

    from mt.geosom.gui import explore, CombinedViewer, SOMViewer, ComponentMatrixViewer, link_views

Spherical maps (GeoSOM), rotatable with the mouse
    SOMViewer                     one map: neighbour distance between neurons, U-matrix, component
                                  planes, per-attribute differences, PCA colour, hits, classes;
                                  click a neuron to see its attribute vector
    ComponentMatrixViewer         one synchronised map per attribute, in a grid

Flat maps (PlaneSOM: hexagonal or rectilinear, plane or torus)
    PlaneSOMViewer                the same layers and inspector as SOMViewer
    PlaneComponentMatrixViewer    one map per attribute, in a grid

Both kinds
    CombinedViewer                the single map on top and the component maps below, in one window,
                                  rotating (spheres) and selecting together
    explore(som, data, ...)       opens the right views for the SOM, linked: by default one CombinedViewer
                                  window (layout='separate': two windows)
    train_and_explore(data, ...)  trains a GeoSOM or PlaneSOM first
    link_views(a, b, ...)         keeps viewers rotating (spheres), selecting and switching projection together

Modules: som_viewer, matrix_viewer, plane_viewer, combined_viewer, inspector (shared layers and side panel),
link, explorer (also `python -m mt.geosom`).
"""
_EXPORTS = {
    'SOMViewer': 'mt.geosom.gui.som_viewer',
    'ComponentMatrixViewer': 'mt.geosom.gui.matrix_viewer',
    'PlaneSOMViewer': 'mt.geosom.gui.plane_viewer',
    'PlaneComponentMatrixViewer': 'mt.geosom.gui.plane_viewer',
    'CombinedViewer': 'mt.geosom.gui.combined_viewer',
    'link_views': 'mt.geosom.gui.link',
    'explore': 'mt.geosom.gui.explorer',
    'train_and_explore': 'mt.geosom.gui.explorer',
    'Views': 'mt.geosom.gui.explorer',
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    # imported on first use, so `import mt.geosom.gui` itself does not load matplotlib
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
