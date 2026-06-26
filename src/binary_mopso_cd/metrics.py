from __future__ import annotations

from binary_mopso_cd.entities import Solution


COSINE_DISTANCE_UPPER_BOUND = 2.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalized_objective_point(solution: Solution) -> tuple[float, float] | None:
    if solution.objectives is None:
        return None
    fidelity = clamp((solution.objectives.f1 + 1.0) / 2.0, 0.0, 1.0)
    diversity = clamp(solution.objectives.f2 / COSINE_DISTANCE_UPPER_BOUND, 0.0, 1.0)
    return fidelity, diversity


def pareto_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    front = []
    for point in unique:
        if not any(
            other[0] >= point[0]
            and other[1] >= point[1]
            and (other[0] > point[0] or other[1] > point[1])
            for other in unique
        ):
            front.append(point)
    return sorted(front, key=lambda item: item[0])


def calculate_hypervolume(points: list[tuple[float, float]]) -> float | None:
    front = pareto_points(points)
    if not front:
        return None
    collapsed: dict[float, float] = {}
    for x_value, y_value in front:
        collapsed[x_value] = max(collapsed.get(x_value, 0.0), y_value)
    hypervolume = 0.0
    previous_x = 0.0
    for x_value, y_value in sorted(collapsed.items()):
        if x_value > previous_x:
            hypervolume += (x_value - previous_x) * y_value
            previous_x = x_value
    return clamp(hypervolume, 0.0, 1.0)


def archive_metrics(solutions: list[Solution]) -> dict[str, float | None]:
    points = [point for solution in solutions for point in [normalized_objective_point(solution)] if point is not None]
    return {
        "hypervolume": calculate_hypervolume(points),
    }
