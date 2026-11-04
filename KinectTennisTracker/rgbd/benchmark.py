from math import acos, degrees, dist, sqrt
from statistics import mean
from time import perf_counter

from .detection import BallDetector
from .simulation import FPS, SCENARIOS, ground_truth, simulate
from .tracking import Ballistic, ConstantVelocity, HORIZONS, Pipeline, Tracker, assign


def evaluate(scenario: str, seed: int = 7):
    models = {"cv": ConstantVelocity()}
    if scenario == "ballistic":
        models["ballistic"] = Ballistic()
    runs = {}
    for name, model in models.items():
        pipeline = Pipeline(BallDetector(stride=1), Tracker(model))
        rows = []
        positions, angles, runtimes = [], [], []
        forecasts = {h: [] for h in HORIZONS}
        hold = {h: [] for h in HORIZONS}
        forecast_opportunities = dict.fromkeys(HORIZONS, 0)
        matched = false_positives = misses = switches = coasting = 0
        last_ids = {}
        for index, (frame, truth) in enumerate(simulate(scenario, seed)):
            start = perf_counter()
            measurements, estimates = pipeline.step(frame)
            runtimes.append((perf_counter() - start) * 1000)
            ids = list(truth)
            for h in HORIZONS:
                if index + round(h * FPS) < SCENARIOS[scenario]:
                    forecast_opportunities[h] += len(ids)
            costs = [[dist(e.xyz, truth[i].xyz) for e in estimates] for i in ids]
            pairs = assign([[d if d <= .25 else float("inf") for d in row] for row in costs])
            matched += len(pairs)
            misses += len(ids) - len(pairs)
            false_positives += len(estimates) - len(pairs)
            for row, col in pairs:
                object_id = ids[row]
                estimate, state = estimates[col], truth[object_id]
                positions.append(costs[row][col])
                coasting += int(not estimate.observed)
                switches += int(object_id in last_ids and estimate.id != last_ids[object_id])
                last_ids[object_id] = estimate.id
                speed = dist(state.velocity, (0, 0, 0))
                if estimate.direction is not None and speed >= .08:
                    dot = sum(a * b for a, b in zip(estimate.direction, state.velocity)) / speed
                    angles.append(degrees(acos(max(-1.0, min(1.0, dot)))))
                for h, point in zip(HORIZONS, estimate.forecasts):
                    future_index = index + round(h * FPS)
                    if future_index < SCENARIOS[scenario]:
                        target = ground_truth(scenario, future_index)[object_id].xyz
                        forecasts[h].append(dist(point, target))
                        hold[h].append(dist(estimate.xyz, target))
            rows.append((frame.depth_ms, measurements, estimates))
        runs[name] = ({
            "frames": len(rows),
            "matched": matched, "misses": misses, "false_positives": false_positives,
            "id_switches": switches, "matched_coasting": coasting,
            "position_rmse_m": sqrt(mean(d**2 for d in positions)) if positions else None,
            "mean_direction_error_deg": mean(angles) if angles else None,
            "direction_samples": len(angles),
            "pipeline_mean_ms": mean(runtimes),
            "pipeline_p95_ms": sorted(runtimes)[(95 * len(rows) + 99) // 100 - 1],
            "forecast": {str(h): {
                "mean_error_m": mean(forecasts[h]) if forecasts[h] else None,
                "hold_mean_error_m": mean(hold[h]) if hold[h] else None,
                "samples": len(forecasts[h]), "eligible": forecast_opportunities[h],
            } for h in HORIZONS},
        }, rows)
    return runs
