# benchmarks/benchmark_cpp_vs_python.py

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from statistics import mean, stdev

import csv

import solitonkit as sk
import solitonkit.viz as viz


def time_function(fn, repeats: int = 20) -> tuple[float, float]:
    """
    Return mean and std runtime in seconds.
    """

    times = []

    for _ in range(repeats):
        start = perf_counter()
        fn()
        end = perf_counter()

        times.append(end - start)

    if len(times) == 1:
        return times[0], 0.0

    return mean(times), stdev(times)


def python_total_energy(field_array, spacing: float) -> float:
    density = viz.energy_density(
        field_array,
        spacing=spacing,
    )

    return float(density.sum() * spacing * spacing)


def run_single_benchmark(
    nx: int,
    ny: int,
    spacing: float,
    repeats: int,
) -> dict[str, float | int]:
    field = sk.make_skyrmion_field(
        nx,
        ny,
        spacing=spacing,
        radius=4.0,
    )

    field_array = sk.to_numpy(field)

    cpp_energy = sk.total_energy(field)
    py_energy = python_total_energy(field_array, spacing)

    cpp_charge = sk.topological_charge(field)
    py_charge = viz.topological_charge(field_array, spacing=spacing)

    cpp_energy_mean, cpp_energy_std = time_function(
        lambda: sk.total_energy(field),
        repeats=repeats,
    )

    py_energy_mean, py_energy_std = time_function(
        lambda: python_total_energy(field_array, spacing),
        repeats=repeats,
    )

    cpp_charge_mean, cpp_charge_std = time_function(
        lambda: sk.topological_charge(field),
        repeats=repeats,
    )

    py_charge_mean, py_charge_std = time_function(
        lambda: viz.topological_charge(field_array, spacing=spacing),
        repeats=repeats,
    )

    return {
        "nx": nx,
        "ny": ny,
        "points": nx * ny,

        "cpp_energy": cpp_energy,
        "python_energy": py_energy,
        "energy_abs_diff": abs(cpp_energy - py_energy),

        "cpp_charge": cpp_charge,
        "python_charge": py_charge,
        "charge_abs_diff": abs(cpp_charge - py_charge),

        "cpp_energy_time_mean": cpp_energy_mean,
        "cpp_energy_time_std": cpp_energy_std,
        "python_energy_time_mean": py_energy_mean,
        "python_energy_time_std": py_energy_std,
        "energy_speedup": py_energy_mean / cpp_energy_mean,

        "cpp_charge_time_mean": cpp_charge_mean,
        "cpp_charge_time_std": cpp_charge_std,
        "python_charge_time_mean": py_charge_mean,
        "python_charge_time_std": py_charge_std,
        "charge_speedup": py_charge_mean / cpp_charge_mean,
    }


def main() -> None:
    spacing = 0.2
    repeats = 30

    sizes = [
        64,
        128,
        256,
        512,
        1024,
        2048
    ]

    results = []

    for size in sizes:
        print(f"Running benchmark for {size} x {size}...")

        result = run_single_benchmark(
            nx=size,
            ny=size,
            spacing=spacing,
            repeats=repeats,
        )

        results.append(result)

        print(
            f"  Energy speedup: {result['energy_speedup']:.2f}x | "
            f"Charge speedup: {result['charge_speedup']:.2f}x"
        )

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "benchmark_cpp_vs_python.csv"

    fieldnames = list(results[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("Benchmark results:")
    print("-" * 80)

    for result in results:
        print(
            f"{result['nx']}x{result['ny']} | "
            f"E C++ {result['cpp_energy_time_mean']:.6f}s | "
            f"E Python {result['python_energy_time_mean']:.6f}s | "
            f"E speedup {result['energy_speedup']:.2f}x | "
            f"Q C++ {result['cpp_charge_time_mean']:.6f}s | "
            f"Q Python {result['python_charge_time_mean']:.6f}s | "
            f"Q speedup {result['charge_speedup']:.2f}x"
        )

    print()
    print("Saved CSV to:", output_path)


if __name__ == "__main__":
    main()