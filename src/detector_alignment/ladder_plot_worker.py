from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ladder_viz import PreparedIndividualResidualPlot


def render_individual_sensor_plot(
    plot: PreparedIndividualResidualPlot,
    dpi: int,
    point_size: float,
) -> str:
    """Render one sensor PNG in a process-owned Agg Matplotlib context."""
    os.environ["MPLBACKEND"] = "Agg"

    from .ladder_viz import render_prepared_individual_residual

    path = render_prepared_individual_residual(
        plot,
        dpi=dpi,
        point_size=point_size,
    )
    return str(path)
