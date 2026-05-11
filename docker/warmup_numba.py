"""
Pre-compile all numba JIT functions used during Aubo data generation.
Run once during Docker build so containers start without recompilation.
"""
import numpy as np

# ── geometrout.transform ──────────────────────────────────────────────────────
from geometrout.transform import SE3, SO3
q_id = np.array([1.0, 0.0, 0.0, 0.0])
p0 = np.zeros(3)
se3 = SE3(p0, q_id)
so3 = SO3(q_id)
SO3.from_rpy(0.1, 0.2, 0.3)
SO3.from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.5)
SO3.from_matrix(np.eye(3))
_ = so3.matrix
_ = so3.rpy
_ = se3.matrix
_ = se3.xyz
_ = se3.inverse
se3b = SE3(np.array([0.1, 0.0, 0.0]), q_id)
_ = se3 * se3b
SE3.interpolate(se3, se3b, 0.5)

# ── geometrout.primitive ──────────────────────────────────────────────────────
from geometrout.primitive import Cuboid, Cylinder, Sphere, CuboidArray, CylinderArray
cub = Cuboid(np.zeros(3), np.ones(3) * 0.1, q_id)
_ = cub.corners
_ = cub.sdf(np.array([0.5, 0.0, 0.0]))
_ = cub.sample_surface(4)
cyl = Cylinder(np.zeros(3), 0.05, 0.1, q_id)
_ = cyl.sdf(np.array([0.5, 0.0, 0.0]))
sph = Sphere(np.zeros(3), 0.05)
_ = sph.sdf(np.array([0.5, 0.0, 0.0]))
ca = CuboidArray([cub, cub])
_ = ca.scene_sdf(np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]))
ya = CylinderArray([cyl, cyl])
_ = ya.scene_sdf(np.array([[0.5, 0.0, 0.0]]))

# ── geometrout.maths ──────────────────────────────────────────────────────────
from geometrout.maths import transform_in_place
pts = np.random.rand(4, 3)
transform_in_place(pts, np.eye(4))

# ── robofin kinematics (numba) ────────────────────────────────────────────────
from robofin.kinematics.numba_aubo import aubo_arm_link_fk, aubo_eef_link_fk
poses = aubo_arm_link_fk(np.zeros(6), np.eye(4))
aubo_arm_link_fk(np.array([0.1, 0.2, -0.3, 0.4, -0.5, 0.6]), np.eye(4))
aubo_eef_link_fk(0.0, np.eye(4))

print("Numba precompile done")
