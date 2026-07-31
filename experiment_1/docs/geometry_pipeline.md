# Geometry pipeline

`GeometryPipeline` hashes all four raw STL files, the configured assembly URDF,
every mesh referenced by that URDF, scale, and collision settings. Pipeline
schema changes also invalidate the manifest. Source files are never
overwritten.

Inspection records raw and merged topology, bounds, extents, centroid, volume,
area, counts, watertightness, connected components, principal axes, inferred
scale, and mesh warnings.

Normalization uses explicit 4×4 transforms:

- gripper origin: XY center and top Z;
- block origin: XY center and bottom Z;
- washer origin: bounding-box center;
- scale: 0.001 m/mm.

## Cable anchors

Planes normal to X slice each lower gripper arm at its inner mouth. Closed YZ
loops are circle-fitted and filtered to 1.30–1.70 mm diameter and low residual.
The seven centers on each side are sorted by Y and paired by Y/Z distance.
Nonuniform spacing or an ambiguous count is fatal. A checked seven-center
fallback exists only for sectioning failure and is labeled in the manifest.

## Target recovery

By default, the fixed-joint graph in `ghast_0_new/urdf/ghast_0.urdf` supplies
one block mesh and four washer instances. `package://` mesh paths, visual
origins, mesh scales, joint origins, and RPY rotations are resolved into the
block frame. The assembly is rejected unless it is fixed, connected, contains
one unambiguous large mesh and four equal smaller meshes, and the washer mesh
matches the legacy `SLW_0.stl` within tolerance.

With `paths.assembly_urdf=null`, the legacy `Ghast_0.stl` fallback splits into
five watertight components and recovers washer poses by registration. Both
paths produce the same approximately 19.0 × 28.385 mm washer grid.

## Collision geometry

Visual STL meshes have collision disabled. CoACD produces up to 64 convex
pieces. Validation rejects a single hull, excessive source-volume error, or a
proxy that fills too much of the global hull. A voxel-box fallback preserves
openings when CoACD fails.

The sawtooth slopes, stems, and thin hook returns are reinforced by analytic
full-span boxes derived from X-normal sections. The URDF block selects a
separate profile for its deeper J features. Exposed faces coincide with the
STL and thickness extends inward into solid material. These remain ordinary
payload collision geoms and create no constraint or attachment. The generated
orthographic overlay must be inspected after geometry or decomposition
changes.

For the URDF profile, central-section void probes reject complete CoACD hulls
that bridge any measured J cavity. The exact ramp, rounded valley, stem, upper
arm, rounded nose, and return surfaces replace those rejected hulls. This is
required because a good global proxy-volume score did not prevent six convex
pieces at the center section—and twelve across the full hook span—from
occupying empty hook space.
