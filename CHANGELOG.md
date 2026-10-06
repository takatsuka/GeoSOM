# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **`LineSOM`**: a Self-Organising Map on a finite one-dimensional line lattice.
- **`CombinedViewer`** (`mt.geosom.gui`): the map of the whole attribute vectors at the top and the component maps in
  a grid below, in **one window**, all in step: dragging any map rotates every map, a neuron clicked on any map is
  selected on all of them, and the projection is shared. While one part is dragged the other follows by blitting, so
  rotating stays smooth. Keys act on the part under the mouse pointer. Works for `GeoSOM` and `PlaneSOM`.
- `explore(..., layout='combined' | 'separate')` and `--layout` on the command line; `Views.window` and
  `Views.save(path)` (the whole window); `--save PREFIX` also writes `PREFIX.png`, the whole window.
- Every viewer (`SOMViewer`, `ComponentMatrixViewer`, `PlaneSOMViewer`, `PlaneComponentMatrixViewer`) takes `fig=`,
  a matplotlib `Figure` or `SubFigure` to draw into instead of opening a window; `save()` then saves just its part.
- `link_views(..., projection=True)` also keeps the map projection in step (`SOMViewer`, `ComponentMatrixViewer`
  have `on_projection()`).

### Changed
- **Renamed to GeoSOM.** The repository is now `takatsuka/GeoSOM` and the PyPI distribution is `geosom`
  (`pip install geosom`); the command-line tool is `geosom`. The import name is now `mt.geosom` (was `mt.geodesicsom`): change `from mt.geodesicsom import ...` to `from mt.geosom import ...`.
- The class `GeodesicSOM` is now `GeoSOM`, in the module `GeoSOM.py` (was `GeodesicSOM.py`): `from mt.geosom.GeoSOM import GeoSOM`.
- Depends on `geodesicdomes` (the renamed mtgeodesicdome, repository `takatsuka/GeodesicDome`).
- `setup_env.sh`: the environment is `.venv` in the repository (`$GEOSOM_VENV`); a sibling checkout is looked for in
  `../GeodesicDome` (`$GEODESICDOME_DIR`).
- **`explore()` and `python -m mt.geosom` now open one combined window by default** (was two windows);
  `layout='separate'` / `--layout separate` gives the old behaviour.
- The neuron inspector's layout keeps fixed distances from the top of the figure, so it also fits a shorter map
  (as in the combined window); its attribute list stops where the space ends (the bar chart shows them all).
- `title=''` gives a viewer no title (`None` is still the default description).

## [1.1.2] — 2026-10-01

### No major change
- touched.

### Added
- **Example notebooks** in `examples/notebooks/`: one Jupyter notebook per example script, split into explained
  steps, with ipywidgets controls under every viewer (layer, attribute, selected neuron, view centre, ...) and a
  "Try it" section to retrain with other settings. With `ipympl` the viewers are live in the notebook (drag,
  click, the on-figure buttons and keys). New `notebooks` extra (`pip install -e ".[notebooks]"`), included in `all`
  and so in `setup_env.sh`.
- **Kohonen's animals** as sample data (`animals`: 16 animals × 13 binary attributes, from Ritter & Kohonen,
  "Self-organizing semantic maps", 1989). `datasets.animals(symbols=True)` sets it up as the paper's semantic map;
  `examples/05_kohonen_animals.py` trains it and labels every neuron with the animals it wins. `--data animals` on
  the command line (not standardised, since the attributes are 0/1).

### Fixed
- A class legend with more than six classes no longer runs into the projection buttons: it wraps into columns.

## [1.1.0] — 2026-10-01

### Added
- **GPU and multi-core training.** `GeoSOM`, `PlaneSOM` (and any `LatticeSOM`) take `backend=`
  (`'auto'` by default: an NVIDIA GPU through torch or CuPy, the Apple Silicon GPU through torch/MPS, otherwise every
  CPU core). `som.to('cpu')` switches device; `MTGEODESIC_BACKEND` sets the default. Best-matching-unit search,
  `distances`, quantisation/topographic error and the neighbourhood products of batch and online training run on the
  device, in blocks sized to its memory. Results are unchanged (float64 on the CPU; float32 by default on a GPU).
- `mt.geosom.geometry`: `SphereDistance`, `PlaneDistance`, `DenseDistance` — distances between neurons computed
  on the device when needed.

### Changed
- The `(neurons × neurons)` distance matrix is no longer built up front: rows are computed on the device when needed
  and the whole matrix is kept only when it fits (`node_distance` is still available and is built on first use).
  Maps far larger than before now train in bounded memory — frequency 64 (40 962 neurons) needs under 1 GB instead of
  more than 13 GB.
- Batch training finds the BMUs and the previous epoch's quantisation error in one pass over the data.
- `topographic_error` is vectorised.
- Needs geodesicdomes ≥ 1.3.0.

## [1.0.0] — 2026-09-29

First release.

### Added
- **`GeoSOM`**: a spherical Self-Organising Map whose neurons are the points of a geodesic dome
  (geodesicdomes). Batch and online training, linear (PCA) and random initialisation, missing values (NaN) ignored
  in the best-matching-unit search and the updates, hits, node labels (`node_labels`, `bmu_labels`), quantisation and
  topographic error. Weights
  are also stored on every dome vertex (`vertex.data`), seam copies included.
- **`PlaneSOM`**: the same SOM on a flat hexagonal or rectilinear grid, with borders or as a torus, with distances
  measured on the grid (wrapping round on a torus).
- **`LatticeSOM`** (`mt.geosom.lattice_som`): the training and measurements shared by both.
- Distances between the attribute vectors of neighbouring neurons: `edge_distance()`, `face_distance()` (per triangle
  or square), `u_matrix()`, and per attribute `face_component_difference()` / `face_component_value()`.
- **`mt.geosom.gui`** (needs matplotlib):
  - `SOMViewer`: a rotatable map of a trained GeoSOM, built on GeodesicDome's `ProjectionViewer`, with layers
    (neighbour distance, U-matrix, component planes, per-attribute differences, PCA colour, hits, classes), sample
    markers that rotate with the sphere, and click-to-inspect neurons.
  - `ComponentMatrixViewer`: one rotatable map per attribute, in a grid; dragging any map rotates them all; click
    marks a neuron on every map; double-click opens that attribute in a linked `SOMViewer`.
  - `PlaneSOMViewer` and `PlaneComponentMatrixViewer`: the same two views for a `PlaneSOM`.
  - `link_views(...)`: keeps viewers rotating and selecting together.
  - Neuron labels: `node_labels=` / `set_node_labels(mode, sample_names=None)` on `SOMViewer` and `PlaneSOMViewer`
    write on each neuron the label(s) of the samples it is the best matching unit of (majority, all, counts, or your
    own text per sample); they rotate with the sphere; `--node-labels` on the command line.
  - On/off switches in both single-map viewers: *sample dots* and *neuron labels* check boxes and a *labels:* style
    button below the colour bar; keys `s`, `l` (on/off) and `L` (next style); `show_node_labels()`,
    `toggle_node_labels()`, `next_label_style()`, `set_show_samples()`.
  - Label size: *A−* / *A+* buttons, keys `-` / `+`, `set_label_size(points)`, `label_size=` (viewers and
    `explore()`), `--label-size` on the command line.
  - `explore()` opens the two views that suit a trained SOM, linked; `train_and_explore()` trains first.
- `python -m mt.geosom` and the `geosom` command: train and explore from a terminal
  (`--data`, `--lattice`, `--topology`, `--save`, ...); `main.py` does the same in the repository.
- `mt.geosom.datasets`: `clusters()`, `colours()`, `read_csv()`, `standardise()`, and sample data shipped with the
  package (`load_sample()`, `samples()`, `sample_path()`, `load()`): `clusters` (synthetic), `iris`, `penguins` (CC0)
  and `wine` (CC BY 4.0), with sources and licences in `mt/geosom/data/README.md`.
- `SOM` abstract base class, `Neuron` (a weight vector with a missing-value-aware distance), `util.MISSING`.
- Examples `01_plane_som.py`, `02_geosom_interactive.py`, `03_component_matrix.py`, `04_explore_all.py`, with
  screenshots and animations in `examples/output/`.
- User guide `docs/index.md`; tests in `tests/`; `setup_env.sh` for a complete development environment (it installs a
  sibling `../GeodesicDome` checkout in editable mode).
- `src/` layout and `pyproject.toml`; `mt` is a namespace package shared with `mt.geodesicdome`.
- GitHub Actions workflows for tests (Linux, macOS, Windows; Python 3.10–3.14) and PyPI publishing.
- Licensed under **AGPL-3.0-or-later** with an attribution term (see `NOTICE`), like GeodesicDome.
