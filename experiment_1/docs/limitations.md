# Limitations

- The nominal seven-cable pickup does not yet meet acceptance. Contact is
  sustained by the full-span undercut proxy, but the payload oscillates around
  a 1 mm lift instead of rising 20 mm.
- Full-cycle completion is therefore blocked at pickup. No weld or hidden
  state toggle was introduced to conceal this.
- Release-only initialization reaches the washer entry plane without
  interpenetration, but cable middle segments become frictionally caught during
  translation. This is a tracked slow expected regression.
- CoACD reaches the configured 64-hull limit and reports residual concavity.
  The thin hook feature therefore depends on a section-derived analytic
  overlay.
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
- The segmented cable fallback shares MuJoCo's flex topology with fewer
  segments; it is not yet an independently implemented ball-joint capsule
  chain.
- The interactive viewer does not draw textual controller status inside the
  scene; status remains in logs/results.
- Target disturbance is zero because the target is fixed to world.
- Rendering requires a working OpenGL backend. On headless Linux, use
  `MUJOCO_GL=egl`.
