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

The implemented default suite checks STL and URDF mesh dimensions/topology,
URDF fixed-joint washer transforms, automatic anchors, target symmetry,
collision decomposition, cable sag/slack,
inextensible length, endpoint attachment accuracy,
single-ramp contact, rigid-finger insertion, measurable pullout force, and
placement. Slow checks compile/step the tetrahedral backend and retain known
pickup/release/full-cycle regressions as strict `XFAIL`.

Representative results from the implementation session:

- default pytest: 12 passed, 4 deselected;
- compliant pullout: 3.056 mm insertion and 0.288 N retention above weight;
- compliant insertion calibration: 4.44 mm insertion and 0.00687 peak finger
  strain proxy;
- legacy-block, full-resolution 48-segment inextensible pickup: five cables
  entered measured tooth cavities, 2.52 mm maximum payload lift, 3.06 µm peak
  endpoint error, and `8.75e-9` peak absolute axial strain; yarn ultimately
  escaped before the 20 mm threshold;
- new-URDF reduced 16-segment pickup: four cables entered J pockets, 0.558 mm
  maximum payload lift, 2.29 µm peak endpoint error, and `8.27e-9` peak
  absolute axial strain before the cables cleared the block;
- release calibration: middle cable segments remain caught after a 40 mm
  diagonal translation.

The new URDF J profile was sectioned at the block midpoint. Its first tooth has
a rounded return nose near `(y, z) = (-6.30, 45.68) mm`. Minimum normal
clearance to the opposing ramp is approximately 1.609 mm, consistent with the
1.6 mm CAD dimension, while the internal pocket dimension is approximately
1.134 mm. A 0.6 mm yarn therefore fits within the pocket. Void-probe validation
found and removed twelve CoACD pieces that bridged empty J space across seven
X sections. Short
engagement-offset trials with the corrected proxy still allowed physical
escape, so pickup is not yet reported as successful.

## Replacing assumptions with bench data

### Yarn length and damping

Measure cut length, installed chord, initial slack, diameter, mass per length,
and bending damping. The default cable backend deliberately assumes negligible
axial elongation; a measured non-negligible force-extension curve would justify
using and calibrating the diagnostic flex backend.

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
