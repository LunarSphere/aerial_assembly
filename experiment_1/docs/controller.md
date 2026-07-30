# Controller

`ScenarioController` follows explicit routes through:

```text
RESET → SETTLE → APPROACH_PAYLOAD → DESCEND_TO_RAMPS
→ ENGAGE_FORWARD → TENSION_CHECK → LIFT → HOLD_TEST
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
- at least three loaded cables, tension, and contact for tension check;
- payload COM rise for lift/hold;
- minimum of all four insertion depths and reaction force for retention;
- low excess tension or already-clear contact for slack;
- continuous zero contact for release.

The engagement axis/sign and arbitrary 3-D release vector are configurable.
`auto` currently records the geometry-derived `+Y` choice. Calibration runs
with both signs are documented in validation; a production calibration policy
should evaluate a larger parameter neighborhood, not only sign.
