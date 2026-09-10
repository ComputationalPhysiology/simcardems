#!/usr/bin/env python3
"""Generate an ellipsoid cardiac dataset with Latin-hypercube sampling.

The script intentionally runs cases sequentially.  FEniCS may use MPI internally,
and launching several cases in one Python process is usually less predictable than
parallelising invocations of this script at the scheduler level.
"""

import argparse
import copy
import csv
import json
import math
import random
import shutil
import subprocess
import traceback
from pathlib import Path

TARGET_PREFIXES = ("config.", "geometry.", "stimulus.")


def latin_hypercube(n_samples, n_dimensions, seed):
    """Return a reproducible Latin hypercube in the half-open unit cube."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = random.Random(seed)
    points = [[0.0] * n_dimensions for _ in range(n_samples)]
    for column in range(n_dimensions):
        strata = list(range(n_samples))
        rng.shuffle(strata)
        for row, stratum in enumerate(strata):
            points[row][column] = (stratum + rng.random()) / n_samples
    return points


def sample_value(unit_value, definition):
    """Map a unit-cube value to a parameter definition."""
    kind = definition.get("distribution", "uniform")
    low = definition["min"]
    high = definition["max"]
    if high < low:
        raise ValueError(f"Invalid range [{low}, {high}]")

    if kind == "uniform":
        value = low + unit_value * (high - low)
    elif kind == "loguniform":
        if low <= 0 or high <= 0:
            raise ValueError("loguniform bounds must be positive")
        value = math.exp(math.log(low) + unit_value * (math.log(high) - math.log(low)))
    elif kind == "integer":
        # Each integer, including both endpoints, occupies an equal interval.
        value = min(int(low) + int(unit_value * (int(high) - int(low) + 1)), int(high))
    else:
        raise ValueError(f"Unknown distribution {kind!r}")
    return value


def build_samples(spec):
    definitions = spec["parameters"]
    for target in definitions:
        if not target.startswith(TARGET_PREFIXES):
            raise ValueError(
                f"Unsupported target {target!r}; use config.*, geometry.*, or stimulus.*",
            )

    cube = latin_hypercube(spec["n_samples"], len(definitions), spec.get("seed", 0))
    samples = []
    for row in cube:
        samples.append(
            {
                target: sample_value(unit_value, definition)
                for target, unit_value, definition in zip(definitions, row, definitions.values())
            },
        )
    return samples


def expanded_case(spec, sampled):
    case = {
        "config": dict(spec.get("base_config", {})),
        "geometry": dict(spec.get("geometry", {})),
        "stimulus": dict(spec.get("stimulus", {})),
    }
    for target, value in sampled.items():
        section, name = target.split(".", 1)
        case[section][name] = value
    if case["geometry"].get("type") == "generated_lv":
        geometry = case["geometry"]
        geometry["r_long_epi"] = geometry["r_long_endo"] + geometry["wall_thickness_long"]
        geometry["r_short_epi"] = geometry["r_short_endo"] + geometry["wall_thickness_short"]
    return case


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_manifest(path, samples, statuses):
    parameter_names = list(samples[0]) if samples else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "status", *parameter_names])
        writer.writeheader()
        for index, sample in enumerate(samples):
            writer.writerow(
                {"sample_id": index, "status": statuses[index], **sample},
            )


def generate_lv_geometry(geometry, case_dir):
    """Create a valid LV mesh and analytic fibers for one sampled case."""
    geometry_dir = case_dir / "geometry"
    mesh_path = geometry_dir / "lv_ellipsoid.h5"
    schema_path = geometry_dir / "lv_ellipsoid.json"
    if mesh_path.is_file() and schema_path.is_file():
        return mesh_path, schema_path

    executable = shutil.which("cardiac-geometries")
    if executable is None:
        raise RuntimeError(
            "cardiac-geometries was not found. Run inside the SimCardEMS/FEniCS "
            "environment with the cardiac-geometries package installed.",
        )

    if geometry["r_long_epi"] <= geometry["r_long_endo"]:
        raise ValueError("The outer long radius must exceed the inner long radius")
    if geometry["r_short_epi"] <= geometry["r_short_endo"]:
        raise ValueError("The outer short radius must exceed the inner short radius")

    command = [
        executable,
        "create-lv-ellipsoid",
        str(geometry_dir),
        "--r-long-endo",
        str(geometry["r_long_endo"]),
        "--r-long-epi",
        str(geometry["r_long_epi"]),
        "--r-short-endo",
        str(geometry["r_short_endo"]),
        "--r-short-epi",
        str(geometry["r_short_epi"]),
        "--psize-ref",
        str(geometry.get("psize_ref", 3.0)),
        "--create-fibers",
        "--fiber-angle-endo",
        str(geometry.get("fiber_angle_endo", -60.0)),
        "--fiber-angle-epi",
        str(geometry.get("fiber_angle_epi", 60.0)),
        "--fiber-space",
        str(geometry.get("fiber_space", "Quadrature_3")),
    ]
    subprocess.run(command, check=True)
    if not mesh_path.is_file() or not schema_path.is_file():
        raise RuntimeError(f"Geometry generation did not produce {mesh_path} and {schema_path}")
    return mesh_path, schema_path


def run_case(case, case_dir):
    # Heavy imports are delayed so --dry-run can generate a design anywhere.
    import dolfin
    import simcardems

    stimulus = case["stimulus"]

    def stimulus_domain(mesh):
        domain = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
        domain.set_all(0)
        mode = stimulus.get("mode", "corner")
        if mode == "corner":
            lengths = {
                "lx": float(stimulus.get("lx", stimulus.get("L", 1.5))),
                "ly": float(stimulus.get("ly", stimulus.get("L", 1.5))),
                "lz": float(stimulus.get("lz", stimulus.get("L", 1.5))),
            }
            region = dolfin.CompiledSubDomain(
                "x[0] <= lx + DOLFIN_EPS && x[1] <= ly + DOLFIN_EPS && x[2] <= lz + DOLFIN_EPS",
                **lengths,
            )
        elif mode == "x_min_layer":
            threshold = float(mesh.coordinates()[:, 0].min()) + float(stimulus.get("depth", 2.0))
            region = dolfin.CompiledSubDomain(
                "x[0] <= threshold + DOLFIN_EPS",
                threshold=threshold,
            )
        else:
            raise ValueError(f"Unknown stimulus mode {mode!r}")
        region.mark(domain, 1)
        return simcardems.geometry.StimulusDomain(domain=domain, marker=1)

    geometry_values = dict(case["geometry"])
    geometry_type = geometry_values.pop("type", "slab")
    source_geometry_path = None
    source_schema_path = None
    if geometry_type == "slab":
        geometry = simcardems.slabgeometry.SlabGeometry(
            parameters=geometry_values,
            stimulus_domain=stimulus_domain,
        )
    elif geometry_type == "file":
        geometry_path = geometry_values.pop("path")
        schema_path = geometry_values.pop("schema_path", None)
        source_geometry_path = geometry_path
        source_schema_path = schema_path
        if geometry_values:
            raise ValueError(f"Unknown file geometry options: {sorted(geometry_values)}")
        geometry = simcardems.geometry.load_geometry(
            mesh_path=geometry_path,
            schema_path=schema_path,
            stimulus_domain=stimulus_domain,
        )
    elif geometry_type == "generated_lv":
        geometry_path, schema_path = generate_lv_geometry(geometry_values, case_dir)
        source_geometry_path = geometry_path
        source_schema_path = schema_path
        geometry = simcardems.geometry.load_geometry(
            mesh_path=geometry_path,
            schema_path=schema_path,
            stimulus_domain=stimulus_domain,
        )
    else:
        raise ValueError(f"Unknown geometry type {geometry_type!r}")
    config_values = dict(case["config"])
    config_values["outdir"] = case_dir
    if source_geometry_path is not None:
        config_values["geometry_path"] = source_geometry_path
        config_values["geometry_schema_path"] = source_schema_path
    config = simcardems.Config(**config_values)
    coupling = simcardems.models.em_model.setup_EM_model_from_config(
        config=config,
        geometry=geometry,
    )
    runner = simcardems.Runner.from_models(config=config, coupling=coupling)
    runner.solve(
        T=config.T,
        save_freq=config.save_freq,
        show_progress_bar=config.show_progress_bar,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON experiment specification")
    parser.add_argument("--dry-run", action="store_true", help="Write the design but run no simulations")
    parser.add_argument("--resume", action="store_true", help="Skip cases containing results.h5 and state.h5")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed simulation")
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.config.read_text())
    spec = copy.deepcopy(spec)
    geometry_spec = spec.get("geometry", {})
    if geometry_spec.get("type") == "file":
        for key in ("path", "schema_path"):
            if key in geometry_spec:
                value = Path(geometry_spec[key])
                if not value.is_absolute():
                    geometry_spec[key] = str(args.config.resolve().parent / value)
    samples = build_samples(spec)
    output = Path(spec.get("output", "lhs_dataset"))
    if not output.is_absolute():
        output = args.config.resolve().parent / output
    output.mkdir(parents=True, exist_ok=True)

    write_json(output / "experiment.json", spec)
    statuses = ["pending"] * len(samples)
    manifest = output / "manifest.csv"

    for index, sampled in enumerate(samples):
        case_dir = output / f"sample_{index:04d}"
        case_dir.mkdir(exist_ok=True)
        case = expanded_case(spec, sampled)
        write_json(case_dir / "case.json", case)

        if args.dry_run:
            statuses[index] = "designed"
        elif args.resume and (case_dir / "results.h5").is_file() and (case_dir / "state.h5").is_file():
            statuses[index] = "complete"
        else:
            statuses[index] = "running"
            write_manifest(manifest, samples, statuses)
            try:
                run_case(case, case_dir)
                statuses[index] = "complete"
            except Exception:
                statuses[index] = "failed"
                (case_dir / "error.txt").write_text(traceback.format_exc())
                if args.fail_fast:
                    write_manifest(manifest, samples, statuses)
                    raise
        write_manifest(manifest, samples, statuses)

    print(f"Dataset design and status: {manifest}")


if __name__ == "__main__":
    main()
