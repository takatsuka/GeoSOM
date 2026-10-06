# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
LineSOM: a Self-Organising Map on a one-dimensional line lattice.

    import numpy as np
    from mt.geosom.LineSOM import LineSOM

    data = np.random.rand(500, 6)
    som = LineSOM(length=100)
    som.train(data, epochs=20)

The line has two endpoints and connects each interior neuron to its immediate neighbours.
Training and measurements are shared with GeoSOM and PlaneSOM through
mt.geosom.lattice_som.LatticeSOM.
"""

import numpy as np
from mt.geodesicdome.backend import Backend
from mt.geodesicdome.manifold import Manifold
from mt.geodesicdome.vertex import Vertex
from numpy import ndarray

from mt.geosom.geometry import LineDistance
from mt.geosom.lattice_som import LatticeSOM


class _LineGrid(Manifold):
    """Minimal finite path lattice implementing GeodesicDome's Manifold interface."""

    def __init__(self, length: int):
        super().__init__()
        self.vertices = [Vertex(i, 0) for i in range(length)]
        for i, vertex in enumerate(self.vertices):
            vertex.id = i
            vertex.manifold = self
            vertex.coord = np.array([float(i), 0.0, 0.0])

    def get_all_vertices(self) -> list[Vertex]:
        return self.vertices

    def get_faces(self) -> list[Vertex]:
        return []

    def get_number_of_vertices_per_face(self) -> int:
        return 2

    def get_vertex_at(self, x, y) -> Vertex | None:
        if y == 0 and 0 <= x < len(self.vertices) and int(x) == x:
            return self.vertices[int(x)]
        return None

    def get_neighbours(self, vertex: Vertex, visit_same_vertex: bool = False) -> list[Vertex]:
        del visit_same_vertex
        vertex.visited = True
        adjacent = []
        for i in (vertex.id - 1, vertex.id + 1):
            if 0 <= i < len(self.vertices) and not self.vertices[i].visited:
                self.vertices[i].visited = True
                adjacent.append(self.vertices[i])
        return adjacent


class LineSOM(LatticeSOM):
    """
    SOM on a finite line of `length` neurons.

    :param length: number of neurons (at least 2)
    :param dim: number of attributes (optional; taken from the data at initialisation)
    :param seed: seed or numpy Generator for random initialisation and online training
    :param backend: compute device -- None/'auto' (GPU if available, else every CPU core), 'cuda',
                    'mps', 'cupy', 'cpu', 'numpy' or a mt.geodesicdome.backend.Backend

    Neurons are ordered from 0 to `length - 1`; only consecutive neurons are neighbours.
    """

    def __init__(self, length: int, dim: int | None = None, seed=None, backend: str | Backend | None = None):
        if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
            raise TypeError('length must be an integer')
        if length < 2:
            raise ValueError('a LineSOM needs at least 2 neurons')
        length = int(length)
        super().__init__(_LineGrid(length), seed, backend)
        self.length = length
        self.dim = int(dim or 0)
        self.positions: ndarray = np.arange(length, dtype=float)[:, None]
        edges = np.column_stack([np.arange(length - 1), np.arange(1, length)])
        self._set_lattice(np.empty((0, 2), dtype=int), LineDistance(self.positions[:, 0]),
                          index_map=np.arange(length), init_coords=np.linspace(-1.0, 1.0, length)[:, None],
                          edges=edges)

    def __repr__(self) -> str:
        return f'LineSOM(length={self.length}, dim={self.dim})'
