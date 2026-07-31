# Controller

`ScenarioController` follows explicit routes through:

```text
RESET → SETTLE → APPROACH_PAYLOAD → DESCEND_TO_RAMPS
→ ENGAGE_FORWARD → POSITION_UNDERCUT → SEAT_UNDERCUT
→ VERIFY_CAPTURE → TAKE_UP_SLACK → LIFT → HOLD_TEST
→ TRANSPORT_TO_TARGET → ALIGN_PEGS → PRESS_INSERT
→ VERIFY_RETENTION → LOWER_FOR_SLACK → RELEASE_TRANSLATE
→ VERIFY_RELEASE → RETREAT → DONE
```

Scenarios select subsets of this route. Every transition is timestamped with a
reason; every state has a trajectory duration and timeout.

Motion uses quintic interpolation. Progress is integrated monotonically using
the measured wrench-dependent speed scale, so a force spike cannot make the
target move backward. Reaching a wall above a force cap stops progress and
eventually produces a useful timeout.

Measurement gates include:

- cable-payload contact for descent and engagement;
- measured cable occupancy in at least four tooth cavities before take-up;
- verified pre-lift pocket occupancy, followed during slack take-up by at least
  three taut load paths, continued payload contact, aggregate support tension,
  measurable payload lift, endpoint accuracy, and bounded axial strain;
- payload COM rise for lift/hold;
- minimum of all four insertion depths and reaction force for retention;
- low excess tension or already-clear contact for slack;
- continuous zero contact for release.

The forward positioning phase consumes lateral slack while the yarn remains
low. The externally driven rise then begins from that forward-biased pose.
Take-up is slow and separately force-limited; the commanded motion never
creates a hidden attachment.

Two-block routes use `controller.transport_lift_m` rather than the shorter
pickup-demonstration lift. Its default 55 mm rise clears the 48 mm target block
before horizontal transport.

During `pick_and_place` insertion, the solid gripper's lower press surface is
collision-enabled after alignment. This gives the externally driven gripper a
real contact path for downward insertion force; the tension-only strings are
not treated as if they could push the payload.

The engagement axis/sign and arbitrary 3-D release vector are configurable.
`auto` currently records the geometry-derived `+Y` choice. Calibration runs
with both signs are documented in validation; a production calibration policy
should evaluate a larger parameter neighborhood, not only sign.
