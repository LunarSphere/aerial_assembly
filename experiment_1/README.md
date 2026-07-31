# Actuatorless Aerial Gripper Simulation

This repository is a mechanics-focused MuJoCo model of a rigid, pose-commanded
aerial gripper picking up a free stackable block with seven colliding cables,
pressing the block's pegs into four compliant washers, and unloading the cables.
It intentionally omits the airframe, propellers, aerodynamics, battery, and
flight controller.

The simulation is not an animation: the payload has a free joint, cable loads
come from flex/contact mechanics, and no pickup or full-cycle attachment is
welded, parented, or toggled. The washer-only insertion and pullout experiments
use a clearly named mocap test fixture to prescribe the bench-test peg motion.

> **Calibration warning:** masses, elastic properties, yarn slack,
> friction, numerical regularization, force limits, and trajectories are
> engineering starting assumptions. They have not been validated against a
> physical prototype. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Current validation status

The asset pipeline, inextensible-cable tests, cable ramp test,
compliant-washer insertion, washer pullout, and placement fixture pass
deterministically. The default block and washer assembly now comes from
`ghast_0_new/urdf/ghast_0.urdf`, including its deeper J-shaped teeth and four
explicit fixed-joint washer poses. The nominal pickup still fails honestly:
the 0.6 mm yarn holds length to numerical precision but leaves the J pockets
during take-up before lifting the 25 g block by the required 20 mm.
Consequently release and full-cycle tests remain slow expected regressions
rather than being made to pass with a hidden attachment.

## Installation

Python is managed exclusively by [UV](https://docs.astral.sh/uv/). The project
pins Python 3.12 in `.python-version`; do not install dependencies into or run
the operating system's base Python.

```bash
uv sync --extra dev
```

Place the four legacy STL files and concept PDF in `assets/raw/` using these
exact names:

```text
GR_0.stl
BB_0.stl
SLW_0.stl
Ghast_0.stl
Autonomous System Design V0.pdf
```

Raw assets are read-only inputs. Generated normalized meshes, hulls, diagnostics,
and the cache manifest go to `assets/processed/`.

The default configuration additionally reads the fixed assembly at
`ghast_0_new/urdf/ghast_0.urdf`. Set `paths.assembly_urdf=null` to return to
the legacy standalone block and disconnected target-assembly inputs.

## Quick start

```bash
uv run aerial-gripper-sim inspect-assets --config configs/default.yaml
uv run aerial-gripper-sim preprocess --config configs/default.yaml
uv run aerial-gripper-sim run --scenario single_string_ramp_test
uv run aerial-gripper-sim run --scenario seven_string_pickup
uv run aerial-gripper-sim run --scenario washer_insertion_test
uv run aerial-gripper-sim run --scenario washer_pullout_test
uv run aerial-gripper-sim run --scenario placement_only
uv run aerial-gripper-sim run --scenario release_only
uv run aerial-gripper-sim run --scenario full_cycle
```

Use the interactive official viewer or record headlessly:

```bash
uv run aerial-gripper-sim run --scenario seven_string_pickup --viewer
MUJOCO_GL=egl uv run aerial-gripper-sim run \
  --scenario full_cycle --record outputs/full_cycle.mp4
```

Run the optional tetrahedral washer backend:

```bash
uv run aerial-gripper-sim run --scenario washer_insertion_test \
  --set washer.mode=deformable_flex
```

Run deterministic sweeps:

```bash
uv run aerial-gripper-sim sweep \
  --config configs/default.yaml --output outputs/sweep --trials 12
```

Every meaningful scalar can be overridden without editing YAML:

```bash
uv run aerial-gripper-sim run --scenario washer_pullout_test \
  --set payload.mass_kg=0.030 \
  --set washer.effective_stiffness_nm_rad=0.008
```

## Tests

The normal suite excludes expensive simulations:

```bash
uv run pytest -q
```

The deformable, pickup, release, and full-cycle checks are retained under the
`slow` marker. Known physics regressions report `XFAIL`, not false success:

```bash
uv run pytest -q -m slow
```

## Outputs and interpretation

Each run directory contains `results.json`, resolved parameters, state
transitions, CSV/Parquet telemetry, compiled `model.xml`, and `summary.png`.
Failures additionally contain the last valid `qpos`, `qvel`, time, model,
configuration, and telemetry tail.

Important pass criteria are saved with every run:

- pickup: at least 20 mm lift, 0.5 s supported hold, and bounded tilt;
- cable integrity: endpoint error below 0.05 mm and axial strain near numerical
  zero throughout take-up;
- placement: all four recovered washer locations exceed 3 mm insertion, remain
  within pose tolerance, and do not exceed the configured fixture force;
- pullout: retention force above payload weight is measurable;
- release: payload contacts remain clear for 0.25 s and separation increases.

Preprocessing generates:

- `assets/processed/anchor_detection.png`;
- `assets/processed/collision_proxy_overlay.png`;
- `assets/processed/asset_inspection.json`;
- `assets/processed/manifest.json`.

## Configuration map

`configs/default.yaml` covers units, solver/contact settings, payload inertia
scaling, cable material and numerical mass, both washer backends, smooth
trajectory speeds, force caps, collision decomposition, output settings,
acceptance thresholds, and deterministic perturbations. Scenario YAML files
provide compact starting points; CLI overrides always remain typed and
validated.

Detailed design notes:

- [Physics model](docs/physics_model.md)
- [Geometry pipeline](docs/geometry_pipeline.md)
- [Controller](docs/controller.md)
- [Validation and calibration](docs/validation.md)
- [Limitations](docs/limitations.md)
