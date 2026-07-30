# Codex task: actuatorless aerial gripper physics simulation

You are working in a new repository that already contains these input files in `assets/raw/`:

- `GR_0.stl` - rigid arch-shaped gripper frame. It contains the holes for the strings, but no string geometry.
- `BB_0.stl` - rigid stackable payload block with asymmetric ramp-and-hook pickup grooves and four lower pegs.
- `SLW_0.stl` - self-locking star washer / wedge insert.
- `Ghast_0.stl` - reference target assembly containing one block plus four washer instances as disconnected mesh components.
- `Autonomous System Design V0.pdf` - concept drawings and the intended pickup, placement, unloading, and release sequence.

Build a complete, runnable, documented Python physics-simulation repository. Do not stop after planning or scaffolding. Implement the system, run the tests and example scenarios, diagnose failures, and iterate until the acceptance criteria at the end are met.

## 1. Scope and objective

Simulate only the mechanical interaction needed to pick up, carry, place, and release the block. Do not simulate the Crazyflie airframe, motors, propellers, aerodynamics, battery, or flight controller.

Represent the gripper as a kinematically commanded rigid body whose 6-DoF pose is prescribed by a controller/state machine. The gripper trajectory stands in for the drone motion. The simulation must calculate the actual contact forces, string tensions, payload motion, peg insertion forces, washer deformation, and release behavior.

The nominal sequence is:

1. Start above a free payload block resting on a plane.
2. Descend until the parallel strings contact the block's pickup ramps.
3. Translate in the engagement direction so the strings slide down the asymmetric ramps and settle beneath the hooks.
4. Ascend and verify that the block lifts and remains supported by the strings.
5. Move the gripper and payload over a fixed target block.
6. Descend and align the carried block's four lower pegs with the four self-locking washers in the target block.
7. Press downward until the pegs are inserted and retained by the washer fingers.
8. Lower slightly to remove string tension and create slack.
9. Translate backward or diagonally away so the strings slide out of the hook grooves.
10. Retreat and verify that the placed block remains attached to the target while the strings are fully released.

The same codebase must support pickup-only, placement-only, release-only, and full-cycle scenarios.

## 2. Required simulation stack

Use Python and MuJoCo as the primary simulator. Target the currently installed stable MuJoCo 3.x package and declare a compatible version range in `pyproject.toml`. Use the official Python bindings and viewer.

Use:

- `mujoco` for dynamics, contact, deformable flex objects, rendering, and the interactive viewer.
- `trimesh`, `numpy`, and `scipy` for STL inspection, transforms, connected-component analysis, sectioning, and geometry checks.
- `pyyaml` or equivalent for human-readable configuration.
- `matplotlib` and `pandas` for plots and result tables.
- `pytest` for tests.
- `imageio` or `ffmpeg` integration for optional MP4 export.
- A reproducible convex-decomposition tool such as CoACD/VHACD for collision proxies, with a documented fallback.
- `gmsh`/`meshio` or another deterministic tetrahedral meshing path for the deformable washer.

Do not silently replace the requested physics with animation. Strings must have collision geometry and physical tension. The payload must be a free rigid body. Success must emerge from contact and mechanics, not from welding, teleporting, parenting, or manually toggling attachment states.

## 3. Facts already observed from the supplied assets

Treat these as validation targets, not as substitutes for runtime asset inspection:

- STL coordinates are in millimeters. Convert to SI units with a scale of `0.001`.
- `GR_0.stl` is approximately `106.472 x 20.000 x 44.960 mm`.
- `BB_0.stl` is approximately `35 x 40 x 48 mm`.
- `SLW_0.stl` is approximately `9.0 mm` in outer diameter and `1.2 mm` thick.
- `Ghast_0.stl` contains five disconnected watertight components: one large block and four washer-sized components.
- The gripper has seven matching circular string holes in each lower arm. The holes are about `1.5 mm` in diameter.
- In the raw `GR_0.stl` frame, a useful fallback estimate for the seven left-arm hole centers is:
  - `x ~= -49 to -52 mm`
  - `z ~= -32.65 mm`
  - `y ~= [-17.70, -15.20, -12.70, -10.20, -7.70, -5.20, -2.70] mm`
  - the right-arm centers should be detected as the corresponding symmetric set.

Implement automatic detection and use the values above only as a checked fallback. Fail loudly with a clear diagnostic if the detected count is not seven or if left/right holes cannot be paired.

## 4. Repository layout

Create at least this structure:

```text
.
├── pyproject.toml
├── README.md
├── ASSUMPTIONS.md
├── assets/
│   ├── raw/
│   └── processed/
├── configs/
│   ├── default.yaml
│   ├── pickup_only.yaml
│   ├── placement_only.yaml
│   └── full_cycle.yaml
├── docs/
│   ├── physics_model.md
│   ├── geometry_pipeline.md
│   ├── controller.md
│   ├── validation.md
│   └── limitations.md
├── src/aerial_gripper_sim/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── geometry.py
│   ├── collision_proxies.py
│   ├── string_model.py
│   ├── washer_model.py
│   ├── scene_builder.py
│   ├── controller.py
│   ├── sensors.py
│   ├── metrics.py
│   ├── recorder.py
│   ├── visualization.py
│   └── sweeps.py
├── scripts/
│   ├── inspect_assets.py
│   ├── preprocess_assets.py
│   ├── run_scenario.py
│   └── run_sweep.py
└── tests/
    ├── test_assets.py
    ├── test_anchor_detection.py
    ├── test_target_assembly.py
    ├── test_string_physics.py
    ├── test_washer_physics.py
    ├── test_pickup.py
    ├── test_placement.py
    ├── test_release.py
    └── test_full_cycle.py
```

Small changes to names are acceptable, but preserve the same separation of concerns.

## 5. Geometry preprocessing and validation

Implement a cached preprocessing pipeline keyed by file hashes and configuration values.

### 5.1 Asset inspection

For every STL, report and save to JSON:

- bounds, extents, centroid, volume, surface area, face count, vertex count
- watertightness and connected-component count
- principal axes and candidate up direction
- inferred unit scale
- warnings for degenerate triangles, duplicate faces, inverted normals, or nonmanifold edges

Never overwrite the raw files.

### 5.2 Coordinate normalization

Create explicit transforms from each raw STL frame into a common simulation frame:

- `+Z` is up.
- The gripper span is along `X`.
- The seven strings run primarily along `X`.
- The engagement/release translation is along a configurable horizontal axis, nominally `Y`.
- The payload block begins level on the ground plane.

Store all transforms in the processed-asset manifest and show them in the viewer with debug axes.

### 5.3 String-hole detection

Detect the seven hole pairs from the gripper mesh instead of hard-coding them blindly. A suitable approach is:

1. Identify the two lower arm regions by position and symmetry.
2. Slice each arm with planes normal to the span axis.
3. Extract closed boundary loops.
4. Fit circles and filter by expected diameter, lower-arm height, spacing, and symmetry.
5. Pair left and right centers by nearest `y,z` coordinates.
6. Choose anchor sites at the inner mouths of the holes, not at the outer arm surfaces.

Save a diagnostic image showing the mesh, detected centers, hole indices, and left/right pair lines. Add a unit test that checks there are exactly seven pairs and approximately uniform spacing.

### 5.4 Target assembly extraction

Use `Ghast_0.stl` as a reference assembly only. Split it into connected components, identify the largest component as the target block and the four small equal-size components as washers, and recover the four washer transforms relative to the target block.

Build the physical target from:

- one rigid `BB_0.stl` instance, fixed to the world; and
- four independent `SLW_0.stl` washer models positioned using the recovered transforms.

Do not simulate the entire `Ghast_0.stl` as one rigid body, because the washers need independent compliant/deformable behavior.

### 5.5 Collision representation

Use the original STL meshes for visualization. Build separate collision representations.

The asymmetric hook undercuts are essential. A single convex hull is unacceptable because it will fill the hook cavities and make the pickup mechanism meaningless.

Implement this priority order:

1. High-resolution convex decomposition into multiple collision geoms that preserves the ramp surfaces and undercuts.
2. If available and stable, an SDF collision representation for the block.
3. A documented analytic/procedural proxy assembled from convex pieces as a final fallback.

Add an automated validation that samples points/rays in the groove region and confirms that the collision proxy has open ramp entrances and retained undercut cavities. Render a side-by-side visual/collision overlay for inspection.

## 6. Physics models

### 6.1 Gripper

- Model `GR_0.stl` as a rigid body with no free joint.
- Command it through a MuJoCo mocap body or an equivalent stable kinematic mechanism.
- The gripper itself can have collision disabled except where frame/block collision is relevant; make this configurable.
- Add named sites at all fourteen string anchors and at the gripper reference frame.
- Log the reaction force/torque required to follow the commanded trajectory. This is the mechanical load the drone would need to provide, even though the drone is not simulated.

### 6.2 Payload and target blocks

- The carried payload is a free 6-DoF rigid body using `BB_0.stl` visual geometry and validated collision proxies.
- The target block is fixed to the world.
- Payload mass and inertia must be configurable. Default to a nominal `0.025 kg`, but compute inertia from the mesh and rescale it to the configured mass.
- Include gravity and a ground plane.
- Do not infer mass directly from STL volume unless an explicit density mode is selected.

### 6.3 Strings

Create seven independent, parallel, physically colliding strings between the paired gripper holes.

Preferred implementation:

- one-dimensional MuJoCo flex objects with capsule-like collision radius;
- both endpoints pinned to the corresponding gripper anchor sites/body;
- enough segments to wrap around the ramp and hook without excessive faceting;
- very low bending resistance;
- axial stiffness, damping, density, radius, friction, and pretension configurable;
- the string must buckle or go slack instead of supporting compression.

If direct 1-D flex endpoint pinning or pretension is unreliable, implement a segmented capsule cable with ball joints and explicit endpoint constraints as a fallback. Keep both backends behind the same `StringModel` interface.

Pretension must be physical. Implement it by shortening the string rest edge lengths relative to anchor separation, or by a deterministic initialization sequence that starts with closer anchors and expands to the nominal span. Record the achieved steady-state tension before the pickup begins.

Do not use noncolliding tendons as the only string representation. They may be used only as auxiliary tension sensors or stabilizers.

Required string telemetry per time step:

- total and per-segment tension estimate
- endpoint reaction forces
- contact count and contact impulses with the payload
- minimum distance to each hook region
- total string length, rest length, and slack estimate

### 6.4 Flexible self-locking washer

Yes, the washer should be modeled as a flexible rubber-like part. Implement two fidelity modes:

#### Mode A: `deformable_flex`

- Tetrahedralize `SLW_0.stl` into a 3-D mesh suitable for MuJoCo flex.
- Pin or strongly constrain the outer annulus to the target block.
- Leave the inward star fingers free to bend as a peg enters.
- Enable contact between the peg, washer, and target block.
- Use a configurable elastic material with Young's modulus, Poisson ratio, damping, density, and contact friction.
- Use a near-incompressible but numerically stable starting value such as `poisson_ratio = 0.45`, not an extreme value that destabilizes the solver.
- Document clearly that this is a linear-elastic approximation to rubber-like behavior unless a validated nonlinear constitutive model is added.

#### Mode B: `compliant_fingers`

- Use a rigid outer ring plus eight inward fingers represented as small rigid or flexural elements with rotational/translational springs and damping.
- Match the STL finger geometry and rest angles as closely as practical.
- This mode is the deterministic CI fallback and should expose an effective stiffness and friction that can be calibrated to Mode A or to physical tests.

Both modes must support insertion and pull-out tests. The full-cycle demo should run with `compliant_fingers` by default for reliability, while a separate documented command must run the deformable mode.

The washer retention should emerge from finger deformation, normal force, geometry, and friction. Do not create a one-way constraint or manually lock the peg after insertion.

## 7. Contact, solver, and numerical stability

Expose the following in YAML:

- timestep and integrator
- solver type and iteration count
- contact margin/gap
- contact impedance/regularization parameters
- friction coefficients for string-block, peg-washer, block-block, and block-ground pairs
- collision bitmasks
- string segment count and radius
- deformable-mesh resolution
- damping values

Use conservative defaults suitable for millimeter-scale geometry, and explain the scaling rationale in `docs/physics_model.md`.

Implement checks for:

- initial interpenetration
- contact tunneling
- exploding kinetic energy
- NaN/Inf state
- excessive constraint violation
- solver nonconvergence indicators

Abort a run with a useful diagnostic and write the last valid state and config to a failure bundle.

## 8. Controller and state machine

Implement a deterministic state machine. Use named states and log every transition:

```text
RESET
SETTLE
APPROACH_PAYLOAD
DESCEND_TO_RAMPS
ENGAGE_FORWARD
TENSION_CHECK
LIFT
HOLD_TEST
TRANSPORT_TO_TARGET
ALIGN_PEGS
PRESS_INSERT
VERIFY_RETENTION
LOWER_FOR_SLACK
RELEASE_TRANSLATE
VERIFY_RELEASE
RETREAT
DONE
FAIL
```

The controller commands a smooth gripper pose trajectory with bounded velocity, acceleration, and jerk. Use quintic interpolation or an equivalent smooth profile.

Event/transition conditions must use simulated measurements, not only elapsed time. Examples:

- ramp contact detected from string-block contacts
- engagement confirmed from upward load sharing across multiple strings
- payload lift confirmed from payload COM rise and bounded relative pose
- target contact detected from peg/washer contact
- insertion confirmed from peg depth and washer reaction force
- slack confirmed from total string tension below a threshold
- release confirmed from zero or near-zero string-block contact and increasing separation

Every state needs:

- entry action
- command generation
- timeout
- success condition
- failure condition
- telemetry fields

The engagement direction and release vector must be configurable. Add an optional calibration routine that evaluates `+Y` and `-Y` engagement directions and chooses the one that produces higher retention in the supplied geometry. Allow diagonal release vectors.

## 9. Force-limited kinematic motion

Although the gripper pose is commanded, calculate the reaction wrench required to impose that pose. Add configurable safety limits such as:

- maximum downward force
- maximum upward lift force
- maximum horizontal force
- maximum torque

Use nominal starting limits only; label them as unvalidated simulation assumptions. If a command would exceed a limit, slow or stop the trajectory and report a controller failure instead of continuing with infinite kinematic authority.

Provide force-control-like behavior during `PRESS_INSERT`: advance downward slowly until either the insertion-depth criterion is met or the downward-force limit is reached.

## 10. Configuration

Create typed configuration dataclasses and YAML files. The default config should include at least:

```yaml
units:
  stl_to_m: 0.001

simulation:
  timestep_s: 0.00025
  duration_limit_s: 30.0
  seed: 1
  headless: true

payload:
  mass_kg: 0.025

strings:
  count: 7
  backend: flex
  radius_m: 0.00030
  segments_per_string: 48
  pretension_n: 0.10
  axial_stiffness_n_per_m: 1200.0
  damping_n_s_per_m: 0.5
  density_kg_per_m3: 1100.0
  friction: [0.5, 0.01, 0.001]

washer:
  mode: compliant_fingers
  young_modulus_pa: 8.0e6
  poisson_ratio: 0.45
  damping: 0.02
  density_kg_per_m3: 1200.0
  peg_friction: [0.8, 0.02, 0.002]

controller:
  engagement_axis: y
  engagement_sign: auto
  approach_speed_m_s: 0.02
  engagement_speed_m_s: 0.01
  lift_speed_m_s: 0.015
  press_speed_m_s: 0.003
  release_speed_m_s: 0.01
  max_downward_force_n: 1.0
  max_upward_force_n: 1.0
  max_horizontal_force_n: 0.5
```

These numbers are starting values for a runnable model, not experimentally validated material or vehicle data. State this prominently. Make all meaningful values overrideable from the CLI.

## 11. Scenarios and experiments

Implement these named scenarios:

1. `asset_inspection`
2. `single_string_ramp_test`
3. `seven_string_pickup`
4. `washer_insertion_test`
5. `washer_pullout_test`
6. `placement_only`
7. `release_only`
8. `full_cycle`
9. `robustness_sweep`

### Robustness sweep

Run parameter sweeps or Monte Carlo trials over:

- payload X/Y offset and yaw error
- gripper height error
- forward translation distance
- engagement speed
- release direction and angle
- string pretension
- string radius and axial stiffness
- string/block friction
- payload mass
- washer stiffness and friction
- peg/washer lateral misalignment
- press-force limit

Write one row per run to CSV/Parquet and produce success-rate heatmaps and sensitivity plots. Use deterministic seeds.

## 12. Metrics and success criteria

Compute and save at least:

- pickup success boolean
- number of strings carrying load
- peak and mean tension per string
- load-sharing imbalance
- peak gripper reaction wrench
- payload COM trajectory
- payload roll/pitch/yaw during lift
- minimum lift height
- hold duration before drop/slip
- peg insertion depth for all four pegs
- peak insertion force
- washer finger displacement/strain proxy
- pull-out/retention force
- placed-block residual pose error
- release time
- remaining string-block contacts after release
- target disturbance during release
- total simulation wall time and real-time factor

Use geometry-derived thresholds where possible. Example acceptance logic:

- Pickup: payload rises at least `20 mm`, remains supported for at least `0.5 s`, and does not exceed configured tilt/slip limits.
- Placement: all four pegs pass the washer entry plane by a configurable fraction of the available peg length, without exceeding the force limit.
- Release: all string-block contacts clear for at least `0.25 s`, gripper separation increases, and the placed block moves less than a configured translation/rotation tolerance.

Do not hard-code a pass result. Save the exact threshold values with every run.

## 13. Visualization and outputs

Provide:

- an interactive MuJoCo viewer command
- a headless renderer that writes MP4 or image sequences
- optional slow-motion playback
- debug rendering for collision proxies, string anchor sites, string segment indices, contact points/normals, peg axes, washer constraints, and controller target pose
- time-series plots for state, forces, string tensions, payload height/attitude, insertion depth, and contact count
- a final summary figure for each scenario
- a machine-readable `results.json`

The viewer should visibly distinguish:

- original visual meshes
- collision proxies
- strings
- flexible/compliant washer parts
- current state-machine state
- success/failure status

## 14. Command-line interface

Support commands equivalent to:

```bash
python -m aerial_gripper_sim.cli inspect-assets --config configs/default.yaml
python -m aerial_gripper_sim.cli preprocess --config configs/default.yaml
python -m aerial_gripper_sim.cli run --scenario seven_string_pickup --viewer
python -m aerial_gripper_sim.cli run --scenario full_cycle --record outputs/full_cycle.mp4
python -m aerial_gripper_sim.cli run --scenario washer_insertion_test --set washer.mode=deformable_flex
python -m aerial_gripper_sim.cli sweep --config configs/default.yaml --output outputs/sweep
pytest -q
```

Provide a `--set section.key=value` override mechanism and clear `--help` output.

## 15. Testing requirements

All tests must run headlessly. Mark expensive deformable and full-cycle tests as `slow`, but include at least one fast deterministic integration test in the default test run.

Required tests:

### Geometry tests

- all raw files load
- extents are within tolerance of the observed values
- `BB_0.stl`, `SLW_0.stl`, and `GR_0.stl` are watertight
- `Ghast_0.stl` splits into one block plus four washers
- recovered washer transforms are symmetric and nonoverlapping
- seven string-hole pairs are detected and ordered consistently
- collision proxy preserves hook cavities

### String tests

- a free string sags under gravity
- a slack string carries negligible compression
- pretension reaches the configured value within tolerance
- a string sliding on a ramp generates plausible normal/friction forces
- no string segment tunnels through a thin test hook at nominal timestep

### Washer tests

- peg insertion deflects fingers and increases reaction force
- force drops or stabilizes after passing the finger tips
- pulling the peg out requires a measurable retention force
- increasing stiffness increases insertion/retention force monotonically over a small controlled sweep
- outer washer rim remains attached to the target

### Scenario tests

- pickup works from the nominal initial pose
- placement aligns and inserts all four pegs
- lowering reduces total string tension
- reverse/diagonal translation clears the strings
- full cycle completes without NaNs or manual attachment

Use tolerances instead of exact floating-point trajectories. Save regression metrics for debugging, but avoid brittle pixel-based tests.

## 16. Documentation

The README must contain:

- what is simulated and what is intentionally omitted
- installation instructions
- asset placement instructions
- quick-start commands
- screenshots or generated figures
- configuration overview
- interpretation of success/failure metrics
- limitations and next validation steps

`ASSUMPTIONS.md` must list every assumption made because dimensions/material data were unavailable, including:

- payload mass
- string material, diameter, stiffness, damping, and pretension
- friction coefficients
- washer elastic properties
- gripper force limits
- exact initial alignment and trajectory distances

`docs/validation.md` must explain how to replace assumptions with measured data from bench tests, including suggested measurements for string force-extension, ramp friction, washer insertion force, and peg pull-out force.

## 17. Engineering quality

- Use type hints throughout.
- Use structured logging.
- Keep simulation construction separate from controller logic.
- Avoid global mutable state.
- Validate config values with helpful errors.
- Use deterministic random seeds.
- Cache processed assets but rebuild them when inputs/config change.
- Add concise comments explaining non-obvious physics and coordinate transforms.
- Do not leave critical-path TODOs, placeholder pass statements, fake data, or mocked success values.

If a requested MuJoCo feature is unavailable in the installed version, implement the documented fallback rather than stopping. Record the fallback in the run metadata.

## 18. Acceptance criteria

The task is complete only when all of the following are true:

1. A fresh environment can install the package from `pyproject.toml`.
2. Asset inspection and preprocessing run from the CLI.
3. The nominal seven-string pickup scenario runs and produces force/tension plots plus `results.json`.
4. The washer insertion and pull-out scenarios run in `compliant_fingers` mode.
5. A documented deformable-washer example is implemented and can be invoked, even if it is marked slow.
6. The nominal full-cycle scenario runs from pickup through release without manually attaching the block.
7. `pytest -q` passes for the default test set.
8. Slow tests are runnable with `pytest -m slow`.
9. The README contains exact commands and describes any remaining numerical limitations honestly.
10. The final Codex response summarizes files created, commands executed, test results, scenario metrics, fallbacks used, and any assumptions that still need physical calibration.

Begin by inspecting the repository and assets, then implement the full system. Continue working until the acceptance criteria are satisfied.
