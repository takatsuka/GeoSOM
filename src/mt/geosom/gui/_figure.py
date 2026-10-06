# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2022-2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Helpers for viewers that draw either into a window of their own (a matplotlib Figure) or into part
of a shared window (a SubFigure, as in CombinedViewer).  Not used on its own.
"""
import contextlib
import math

import matplotlib.pyplot as plt


def new_figure(fig, figsize, window_title: str):
    """`fig` itself when one is given (a Figure or SubFigure to draw into), otherwise a new window."""
    if fig is not None:
        return fig
    fig = plt.figure(figsize=figsize)
    manager = fig.canvas.manager
    if manager is not None:
        manager.set_window_title(window_title)
    return fig


def is_part(fig) -> bool:
    """True for a SubFigure: part of a window shared with other viewers."""
    return fig.figure is not fig


def root(fig):
    """The window's Figure (the figure itself, or the Figure a SubFigure belongs to)."""
    return fig.figure


def size_inches(fig):
    """(width, height) in inches of a Figure or SubFigure."""
    return fig.bbox.width / fig.dpi, fig.bbox.height / fig.dpi


def owns_event(fig, event) -> bool:
    """
    Whether a mouse or key event is meant for the viewer drawn in `fig`: always for a whole window;
    in a shared window, when the pointer is over this viewer's part of it.
    """
    if not is_part(fig):
        return True
    return event.x is not None and event.y is not None and fig.bbox.contains(event.x, event.y)


def save_figure(fig, path, **kwargs):
    """Saves a Figure, or just the area of a SubFigure (unless bbox_inches is given)."""
    if not is_part(fig):
        fig.savefig(path, **kwargs)
        return
    whole = root(fig)
    kwargs.setdefault('bbox_inches', fig.bbox.transformed(whole.dpi_scale_trans.inverted()))
    whole.savefig(path, **kwargs)


@contextlib.contextmanager
def figure_redirected(fig):
    """
    While active, `plt.figure(...)` returns `fig` instead of opening a window: lets a viewer class
    from another library (ProjectionViewer) build itself inside a given Figure or SubFigure.
    """
    if fig is None:
        yield
        return
    original = plt.figure
    plt.figure = lambda *args, **kwargs: fig
    try:
        yield
    finally:
        plt.figure = original


def fit_columns(n_panels: int, width: float, height: float, aspect: float, title_height: float = 0.2) -> int:
    """
    The number of columns that makes `n_panels` maps of the given aspect (width / height) largest in a
    width x height (inches) area, with the panel layout of the matrix viewers.
    """
    best_size, best_cols = -math.inf, 1
    for cols in range(1, max(1, n_panels) + 1):
        rows = math.ceil(n_panels / cols)
        size = min(0.86 * width / cols / aspect, 0.84 * height / rows - title_height)
        if size > best_size + 1e-9:
            best_size, best_cols = size, cols
    return best_cols
