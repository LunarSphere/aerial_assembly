# Geometry pipeline

`GeometryPipeline` hashes all four raw STL files plus scale and collision
settings. Pipeline schema changes also invalidate the manifest. Raw files are
never overwritten.

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

`Ghast_0` splits into five watertight components. Volume identifies the block;
the four equal small components are washers. ICP recovers orientation and
component bounding-box centers recover robust placement. The resulting grid is
approximately 19.0 × 28.336 mm.

## Collision geometry

Visual STL meshes have collision disabled. CoACD produces up to 64 convex
pieces. Validation rejects a single hull, excessive source-volume error, or a
proxy that fills too much of the global hull. A voxel-box fallback preserves
openings when CoACD fails.

The thin extruded sawtooth hook lips are reinforced by analytic full-span boxes
derived from X-normal sections. These remain ordinary payload collision geoms;
they create no constraint or attachment. The generated orthographic overlay
must be inspected after geometry or decomposition changes.
