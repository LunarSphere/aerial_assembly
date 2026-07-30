# Assumptions requiring physical calibration

None of the following values came from a measured prototype unless explicitly
identified as geometry recovered from the supplied STL files.

## Geometry and coordinates

- STL coordinates are millimetres and are uniformly scaled by `0.001`.
- `+Z` is up, the cable span is `X`, and nominal engagement is `+Y`.
- Mesh bounding boxes define the gripper top datum and payload ground datum.
- The four washer centers and orientations are registered from the disconnected
  `Ghast_0.stl` components. Their centers are more trustworthy than vertex-wise
  ICP because the reference washers were re-tessellated.
- Thin hook lips are reinforced with a full-span analytic collision overlay
  derived from block sections because convex-decomposition seams are too coarse
  at the 1–2 mm feature scale.

## Payload

- Nominal mass is 0.025 kg.
- Inertia comes from uniform mesh volume and is rescaled to that mass.
- Density mode is optional and disabled by default.
- Initial pose is level and centered unless a perturbation is configured.

## Cables

- Radius: 0.30 mm.
- Density: 1100 kg/m³.
- Axial stiffness: 1200 N/m.
- Edge damping: 0.5 N·s/m.
- Pretension: 0.10 N per cable.
- Friction: `[0.5, 0.01, 0.001]`.
- Forty-eight segments are used by default.
- A 50× numerical cable-mass scale is required for stable sub-millimetre flex
  contact at the nominal timestep. Reported physical density is not changed.
- Equality edge constraints provide stable axial response; tension telemetry
  uses the configured physical stiffness and rejects compression.

## Washers

- Compliant mode uses eight rectangular inward fingers, 0.006 N·m/rad hinge
  stiffness, and 0.0004 N·m·s/rad damping.
- Nominal material values are 8 MPa Young's modulus, 0.45 Poisson ratio,
  1200 kg/m³ density, and `[0.8, 0.02, 0.002]` contact friction.
- The tetrahedral backend uses TetGen when the discrete Gmsh surface cannot be
  filled. MuJoCo trilinear deformation retains the full collision mesh while
  using 24 DOFs per washer.
- Trilinear control springs are derived as `E*V/(8*d²)`.
- A 1000× numerical washer mass scale stabilizes the reduced model. It must not
  be interpreted as physical washer mass.

## Contact and control

- Contact impedance, a 0.25 ms timestep, Newton solver, and 100 iterations are
  numerical starting points.
- Payload-ground, block, cable, and washer friction values are unmeasured.
- The 30 N insertion/horizontal test-fixture cap, 1 N upward cap, and 0.08 N·m
  torque cap are unvalidated. The insertion cap is high because the current
  rigid block proxies generate transient fixture forces near 25 N.
- Nominal motion distances are 32 mm descent, 7 mm engagement, 10 mm ramp
  follow-down, 25 mm lift, 18 mm insertion, 5 mm slack lowering, and 25 mm
  release translation.
- `+Y` is selected from the concept/section orientation. Both signs were
  simulated; neither currently meets the pickup threshold.
- The fixed target has exactly zero disturbance by construction.

## Success thresholds

- Pickup: 20 mm lift, 0.5 s hold, 20° maximum roll/pitch.
- Placement: 3 mm minimum across all four registered washer centers, 3 mm
  lateral error, and 5° rotation error.
- Release: zero cable-payload contact for 0.25 s.
- Pullout: at least 0.02 N above payload weight.
