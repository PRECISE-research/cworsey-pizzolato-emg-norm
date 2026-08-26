"""Small publication-style plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt


def despine(ax: plt.Axes) -> None:
    """Retain only left and bottom axes for a restrained scientific style."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
