# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""CombinedViewer (the map and the component matrix in one window), driven headlessly (Agg)."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402
from mt.geodesicdome.interactive import rotation as rot  # noqa: E402

from mt.geosom import datasets  # noqa: E402
from mt.geosom.GeoSOM import GeoSOM  # noqa: E402
from mt.geosom.gui import CombinedViewer, explore  # noqa: E402
from mt.geosom.gui.explorer import main  # noqa: E402
from mt.geosom.PlaneSOM import PlaneSOM  # noqa: E402


@pytest.fixture(scope="module")
def trained():
    ds = datasets.clusters(n_clusters=3, dim=4, per_cluster=20)
    x = datasets.standardise(ds.data)
    return GeoSOM(5, seed=0).train(x, epochs=5), x, ds.labels, ds.feature_names


def _mouse(canvas, name, x, y, dblclick=False):
    canvas.callbacks.process(name, MouseEvent(name, canvas, x, y, button=1, dblclick=dblclick))


def _key(canvas, key, x, y):
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key, x=x, y=y))


def _inside(subfigure):
    """A pixel position inside a part of the window, away from its widgets (its centre)."""
    b = subfigure.bbox
    return b.x0 + b.width / 2, b.y0 + b.height / 2


def test_explore_opens_one_window_by_default(trained):
    som, x, labels, names = trained
    views = explore(som, x, labels, names, select=3, show=False)
    window = views.window
    assert isinstance(window, CombinedViewer)
    assert views.map.fig.figure is window.fig and views.matrix.fig.figure is window.fig
    assert views.map.fig.bbox.y0 > views.matrix.fig.bbox.y0          # the map above, the matrix below
    assert views.map.selected == 3 and views.matrix.selected == 3
    assert len(views.matrix.panels) == som.dim + 1

    separate = explore(som, x, labels, names, layout="separate", show=False)
    assert separate.window is None and separate.map.fig is not separate.matrix.fig
    with pytest.raises(ValueError):
        explore(som, x, layout="side-by-side", show=False)


def test_everything_stays_in_step(trained):
    som, x, labels, names = trained
    w = CombinedViewer(som, x, labels, names)
    w.map.rotate(d_lon=25)
    assert np.allclose(w.matrix.rotation, w.map.rotation)
    w.matrix.rotate(d_lat=-10)
    assert np.allclose(w.matrix.rotation, w.map.rotation)
    w.matrix.select_node(7)
    assert w.map.selected == 7 and w.selected == 7
    w.select_node(None)
    assert w.matrix.selected is None
    w.map.set_projection("Wagner VI")
    assert w.matrix.projection_name == "Wagner VI"
    w.matrix.set_projection("Kavrayskiy VII")
    assert w.map.projection_name == "Kavrayskiy VII"
    w.set_view(10, 40)                                   # passed on to the top map, which the matrix follows
    assert np.allclose(w.matrix.rotation, w.map.rotation)


def test_dragging_a_component_map_rotates_the_top_map(trained):
    som, x, labels, names = trained
    w = CombinedViewer(som, x, labels, names)
    canvas = w.fig.canvas
    panel = w.matrix.panels[2]["ax"]
    px, py = panel.transData.transform([0.0, 0.0])
    _mouse(canvas, "button_press_event", px, py)
    _mouse(canvas, "motion_notify_event", px + 15, py)
    _mouse(canvas, "motion_notify_event", px + 30, py + 5)
    assert w.map._following and np.allclose(w.map.rotation, w.matrix.rotation)
    _mouse(canvas, "button_release_event", px + 30, py + 5)
    assert not w.map._following and w.map._background is None
    assert abs(w.map.centre[1]) > 5 and np.allclose(w.map.rotation, w.matrix.rotation)

    # and the other way round
    mx, my = w.map.ax.transData.transform([0.0, 0.0])
    _mouse(canvas, "button_press_event", mx, my)
    _mouse(canvas, "motion_notify_event", mx - 40, my)
    _mouse(canvas, "button_release_event", mx - 40, my)
    assert not w.matrix._following and np.allclose(w.map.rotation, w.matrix.rotation)


def test_clicks_select_everywhere(trained):
    som, x, labels, names = trained
    w = CombinedViewer(som, x, labels, names)
    canvas = w.fig.canvas
    px, py = w.matrix.panels[1]["ax"].transData.transform([0.0, 0.0])
    _mouse(canvas, "button_press_event", px, py)
    _mouse(canvas, "button_release_event", px, py)
    assert w.matrix.selected is not None and w.map.selected == w.matrix.selected


def test_keys_go_to_the_part_under_the_pointer(trained):
    som, x, labels, names = trained
    w = CombinedViewer(som, x, labels, names)
    canvas = w.fig.canvas
    before = w.map.rotation.copy()
    _key(canvas, "right", *_inside(w.matrix_area))              # rotated once (not once per part)
    assert np.allclose(w.map.rotation, rot.drag_rotation(np.radians(5.0), 0.0) @ before)
    assert np.allclose(w.matrix.rotation, w.map.rotation)

    _key(canvas, "v", *_inside(w.map_area))                     # 'v' means nothing to the top map
    assert w.matrix.mode == "difference"
    _key(canvas, "v", *_inside(w.matrix_area))
    assert w.matrix.mode == "value"
    _key(canvas, "]", *_inside(w.matrix_area))                  # '[' / ']' belong to the top map
    assert w.map.layer == "Neighbour distance"
    _key(canvas, "]", *_inside(w.map_area))
    assert w.map.layer == "Component" and w.map.component == 1
    _key(canvas, "p", *_inside(w.matrix_area))                  # projection: switched in both
    assert w.map.projection_name == w.matrix.projection_name == "Kavrayskiy VII"


def test_saving(trained, tmp_path):
    som, x, labels, names = trained
    views = explore(som, x, labels, names, show=False)
    views.save(tmp_path / "all.png", dpi=50)
    views.map.save(tmp_path / "map.png", dpi=50)
    views.matrix.save(tmp_path / "matrix.png", dpi=50)
    import matplotlib.image as mpimg
    whole, top, bottom = (mpimg.imread(tmp_path / f"{n}.png") for n in ("all", "map", "matrix"))
    assert top.shape[1] == bottom.shape[1] == whole.shape[1]
    assert abs(top.shape[0] + bottom.shape[0] - whole.shape[0]) <= 2


def test_plane_som_in_one_window():
    ds = datasets.clusters(n_clusters=3, dim=3, per_cluster=15)
    x = datasets.standardise(ds.data)
    som = PlaneSOM(8, 10, seed=0).train(x, epochs=5)
    views = explore(som, x, ds.labels, ds.feature_names, select=4, show=False)
    assert views.window is not None and views.map.fig.figure is views.matrix.fig.figure
    assert views.map.selected == 4 and views.matrix.selected == 4
    views.matrix.select_node(11)
    assert views.map.selected == 11
    canvas = views.window.fig.canvas
    _key(canvas, "v", *_inside(views.window.map_area))
    assert views.matrix.mode == "difference"
    _key(canvas, "v", *_inside(views.window.matrix_area))
    assert views.matrix.mode == "value"


def test_command_line_layouts(tmp_path):
    main(["--freq", "3", "--epochs", "2", "--save", str(tmp_path / "c"), "--no-gui"])
    assert (tmp_path / "c.png").stat().st_size > 0 and (tmp_path / "c_map.png").stat().st_size > 0
    views = main(["--freq", "3", "--epochs", "2", "--layout", "separate", "--save", str(tmp_path / "s"), "--no-gui"])
    assert views.window is None and not (tmp_path / "s.png").exists()
    assert (tmp_path / "s_matrix.png").stat().st_size > 0
