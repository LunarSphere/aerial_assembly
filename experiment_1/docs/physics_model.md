# Physics model

## Rigid bodies

The payload is a free six-DOF `BB_0` body. Its mesh-derived inertia is rescaled
to the configured mass. The target is a fixed independent block. The gripper
is driven by a mocap target through a stiff weld constraint; the free body on
the driven side exists so `qfrc_constraint` reports the wrench required to
follow the command. That wrench is the load a future vehicle would have to
provide.

Washer insertion and pullout scenarios use the same mocap-weld arrangement as
a virtual materials-test fixture. This fixture is never used to claim pickup.

## Cables

Each of seven cables is a MuJoCo 1-D flex with colliding capsule elements and
both end vertices pinned in the gripper frame. Rest edges are shortened by
`pretension / axial_stiffness`. Equality edge constraints provide a stable
implicit axial response, while telemetry estimates:

```text
tension = max(0, k * (current_length - rest_length) + c * edge_velocity)
```

The `max(0, ...)` enforces zero compressive load. Contact impulses come from
`mj_contactForce`; hook distance is evaluated in the payload frame.

The numerical cable mass is 50 times material mass. Without this explicit
regularization, a 0.3 mm cable produces sub-microgram nodes and unstable contact
accelerations at 0.25 ms.

## Washer modes

`compliant_fingers` is the deterministic default. A fixed polygonal outer ring
supports eight inward rigid fingers with hinge stiffness and damping. Peg
contact deflects these fingers; there is no one-way lock.

`deformable_flex` loads a first-order tetrahedral Gmsh 4.1 mesh. Gmsh preserves
the STL surface but cannot fill this multiply connected thin topology, so the
documented deterministic fallback is quality-controlled TetGen. Singular
slivers below `1e-16 m³` are removed, connectivity is compacted, and one node
block is written for MuJoCo.

Full per-vertex 8 MPa FEM requires impractical microsecond steps at this scale.
The runnable mode therefore uses MuJoCo's trilinear parameterization: the full
tetrahedral mesh remains active for collision, while eight control points
provide 24 deformation DOFs. Control springs use `E*V/(8*d²)` and the configured
damping. This is a linear-elastic surrogate, not a validated rubber law.

## Contact and solver scaling

The default 0.25 ms timestep resolves a 0.3 mm cable moving at the configured
centimetre-per-second speeds. Newton iterations, soft contact reference times,
small margins, and no contact gap are exposed in YAML. Separate bitmasks allow
cable-payload, payload-washer, and ground-payload contacts while preventing
washer-target self-intersection.

Safety checks stop on nonfinite state, MuJoCo warnings, excessive kinetic
energy, more than 1.5 mm contact penetration, or excessive mocap constraint
error.
