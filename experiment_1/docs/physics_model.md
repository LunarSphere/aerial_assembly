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

Each of seven yarns is a MuJoCo composite `cable`: a chain of colliding capsule
bodies and ball joints. This topology is exactly inextensible at the joint
level, unlike the optional `flex` diagnostic backend. The first end is rooted
in the gripper body and the second is connected to its named anchor site with a
stiff equality constraint.

```text
nominal length = anchor chord + configured slack
axial strain = (reconstructed capsule-chain length - nominal length)
```

Initial vertices follow a sine sag whose polyline length is solved to the
requested value. There is no nominal pretension in cable mode: the chain
buckles under compression, becomes taut during ascent, and transmits load only
through joints/contact. Endpoint reaction is read from the equality constraint.
Contact impulses come from `mj_contactForce`; hook occupancy is evaluated
against the STL-section-derived tooth cavities in the moving payload frame.

Telemetry records nominal/current length, slack, axial strain, endpoint error,
endpoint reaction, payload contacts, and capture state. The default numerical
mass scale is 1, so configured density is the physical density.

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

The hook collision overlay places measured slope, vertical-wall, and lip
surfaces at the STL section coordinates. Proxy depth extends inward into solid
material so compliant contacts cannot tunnel through a paper-thin shell.

Safety checks stop on nonfinite state, MuJoCo warnings, excessive kinetic
energy, more than 1.5 mm contact penetration, or excessive mocap constraint
error. The take-up gate separately rejects excessive cable endpoint or length
error.
