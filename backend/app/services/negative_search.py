import numpy as np

from app.geospatial.grid import ProbabilityGrid, cells_in_polygon
from app.models.heatmap import HeatmapCellDelta

EPSILON = 1e-8
LOG_THRESHOLD = 1e-6


def apply_negative_search(
    grid: ProbabilityGrid,
    polygon_geojson: dict,
    pod: float,
) -> list[HeatmapCellDelta]:
    pod = float(np.clip(pod, 0.01, 0.99))
    mask_cells = cells_in_polygon(grid, polygon_geojson)
    if not mask_cells:
        return []

    probs = grid.probabilities.copy()
    factor = 1.0 - pod
    log_factor = np.log(factor)

    changed: list[HeatmapCellDelta] = []

    for row, col in mask_cells:
        old = probs[row, col]
        if old < LOG_THRESHOLD and old > 0:
            new = max(np.exp(np.log(old) + log_factor), EPSILON)
        else:
            new = max(old * factor, EPSILON)
        probs[row, col] = new
        if abs(new - old) > 1e-10:
            changed.append(HeatmapCellDelta(row=row, col=col, probability=new))

    total = probs.sum()
    if total < EPSILON:
        probs.fill(EPSILON)
        total = probs.sum()

    probs /= total
    grid.probabilities = probs

    # Recompute changed cells with final normalized values
    result: list[HeatmapCellDelta] = []
    for row, col in mask_cells:
        result.append(HeatmapCellDelta(row=row, col=col, probability=float(probs[row, col])))

    # Also include unsearched cells whose probability increased due to renormalization
    mask_set = set(mask_cells)
    for row in range(grid.rows):
        for col in range(grid.cols):
            if (row, col) not in mask_set:
                result.append(HeatmapCellDelta(row=row, col=col, probability=float(probs[row, col])))

    return result
