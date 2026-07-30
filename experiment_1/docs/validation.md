# Validation and calibration

## Reproducible software checks

From a clean UV environment:

```bash
uv sync --extra dev
uv run aerial-gripper-sim inspect-assets --config configs/default.yaml
uv run aerial-gripper-sim preprocess --config configs/default.yaml
uv run pytest -q
uv run pytest -q -m slow
```

The implemented default suite checks STL dimensions/topology, automatic
anchors, target symmetry, collision decomposition, cable pretension/sag/slack,
single-ramp contact, rigid-finger insertion, measurable pullout force, and
placement. Slow checks compile/step the tetrahedral backend and retain known
pickup/release/full-cycle regressions as strict `XFAIL`.

Representative results from the implementation session:

- default pytest: 12 passed, 4 deselected;
- compliant pullout: 3.056 mm insertion and 0.288 N retention above weight;
- compliant insertion calibration: 4.44 mm insertion and 0.00687 peak finger
  strain proxy;
- full-span 24-segment pickup: seven loaded cables, 4.07 N total final cable
  tension, 1.03 mm maximum lift, failure against the 20 mm threshold;
- release calibration: middle cable segments remain caught after a 40 mm
  diagonal translation.

## Replacing assumptions with bench data

### Cable force-extension

Clamp a representative cable at the simulated span, apply several known loads,
and record elongation and hysteresis. Fit axial stiffness and damping from slow
ramps and step relaxation. Measure pretension after installation with an inline
load cell.

### Ramp friction and capture

Pull a cable coupon across a printed ramp at several normal loads and speeds.
Fit Coulomb friction separately for clean and worn surfaces. High-speed video
of entry and lift will show whether the cable occupies the modeled undercut and
whether the critical-feature proxy has the correct lip depth.

### Washer insertion

Use a force gauge or materials tester to press one production peg through one
washer at the configured speed. Record force versus depth, unload, and repeat.
Fit finger stiffness/damping and contact friction to the full curve, not only
the peak.

### Pullout retention

After controlled insertion, pull vertically and with expected lateral error.
Record breakaway force, steady pullout force, permanent set, and cycle count.
Set the acceptance threshold from the worst required payload/load case with an
explicit factor of safety.

### Rigid-body properties

Weigh the printed block and estimate inertia using a bifilar pendulum or CAD
material assignment. Measure washer and cable mass separately so numerical
mass scaling can be reduced or replaced by a smaller stable timestep.

Commit calibrated YAML separately from `default.yaml`, rerun all scenarios and
Monte Carlo sweeps, and compare force-depth/tension curves before accepting any
success rate as predictive.
