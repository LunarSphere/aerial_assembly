# Limitations

- The nominal seven-cable pickup does not yet meet acceptance. Contact is
  now evaluated with an inextensible cable and the new URDF J-hook faces.
  After removing twelve convex hulls that incorrectly filled the J pockets,
  four cables occupy three distinct pockets but still leave during take-up.
  The fourth pocket remains unoccupied with the current seven-string routing;
  engagement trajectory and cable/hook pitch compatibility remain under
  calibration.
- Full-cycle completion is therefore blocked at pickup. No weld or hidden
  state toggle was introduced to conceal this.
- Release-only initialization reaches the washer entry plane without
  interpenetration, but cable middle segments become frictionally caught during
  translation. This is a tracked slow expected regression.
- CoACD reaches the configured 64-hull limit and reports residual concavity.
  The hook feature therefore uses an STL-section-derived overlay for the
  exposed slopes, stems, and return undersides. Proxy thickness lies inside
  the measured solid.
- The collision validation uses volume/nonconvexity metrics and visual overlays;
  a denser signed-distance ray test should be added before manufacturing
  decisions.
- Compliant fingers approximate the detailed star washer with eight boxes and
  hinge springs.
- Deformable washers use reduced trilinear deformation and numerical mass
  scaling. They retain high-resolution tetrahedral collision, but they are not
  a nonlinear hyperelastic rubber model.
- Payload-block collision proxies generate insertion-fixture transients near
  25 N, requiring an unvalidated 30 N default test cap.
- `segmented` is a deprecated alias for the composite ball-joint cable.
  The older extensible `flex` backend remains available only for comparison.
- The interactive viewer does not draw textual controller status inside the
  scene; status remains in logs/results.
- Target disturbance is zero because the target is fixed to world.
- Rendering requires a working OpenGL backend. On headless Linux, use
  `MUJOCO_GL=egl`.
