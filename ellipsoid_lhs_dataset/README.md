# Ellipsoid Latin-hypercube dataset

This folder contains the complete experiment definition and generator for a
left-ventricular ellipsoid dataset. Each Latin-hypercube sample gets a newly
generated ventricular mesh and analytic fiber field. Inner long/short radii
and positive wall thicknesses are sampled independently; outer radii are
derived, preventing invalid negative wall thickness. Every artifact is written
under `results/`.

The default design also samples conservative multipliers around the repository's
baseline physics: longitudinal and transverse conductivity (0.8-1.2), stimulus
amplitude (0.9-1.1), stimulus duration (1.5-2.5 ms), and passive myocardial
stiffness (0.8-1.2). These are baseline sensitivity ranges, not population-level
clinical reference intervals.

Preview and validate the sampled design without FEniCS:

```bash
python ellipsoid_lhs_dataset/generate_dataset.py \
  ellipsoid_lhs_dataset/config.json --dry-run
```

Run the simulations in the project's FEniCS environment:

```bash
python ellipsoid_lhs_dataset/generate_dataset.py \
  ellipsoid_lhs_dataset/config.json
```

Resume an interrupted run:

```bash
python ellipsoid_lhs_dataset/generate_dataset.py \
  ellipsoid_lhs_dataset/config.json --resume
```

Render a geometry-only preflight GIF from a dry run or completed dataset:

```bash
python ellipsoid_lhs_dataset/render_geometry_preview.py \
  ellipsoid_lhs_dataset/results
```

This preview shows the sampled parametric shapes. It is not a substitute for
voltage/displacement fields from completed FEniCS simulations.

The generated `results/` folder contains the resolved experiment, a CSV
manifest, and one directory per sampled simulation. Each sample directory has
its exact `case.json`, generated `geometry/`, simulation output, restart state,
or an `error.txt` when a run fails.
