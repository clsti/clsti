"""Small math helpers shared by the robots, planner and renderer."""
import numpy as np
import mujoco


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mat2quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=float).reshape(9))
    return q


def quat2mat(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q):
    R = quat2mat(q)
    return np.arctan2(R[1, 0], R[0, 0])


def frame_z(zdir, xhint):
    """Rotation whose z axis points along zdir and whose x axis is as close to xhint as possible."""
    z = unit(np.asarray(zdir, dtype=float))
    x = np.asarray(xhint, dtype=float) - np.dot(xhint, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
        x = x - np.dot(x, z) * z
    x = unit(x)
    return np.column_stack([x, np.cross(z, x), z])


def slerp(q0, q1, s):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    d = np.dot(q0, q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)


def smooth(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


def smoother(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (s * (6 * s - 15) + 10)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def lerp_angle(a, b, s):
    return a + wrap(b - a) * s


def box_extent(R, half, direction):
    """Half extent of an oriented box along a unit direction."""
    return float(np.sum(np.abs(R.T @ direction) * half))


def two_link_ik(root, target, l1, l2, pole):
    """Return the middle joint position of a two-segment chain reaching for target."""
    d_vec = target - root
    d = np.linalg.norm(d_vec)
    d = np.clip(d, 1e-4, (l1 + l2) * 0.9995)
    u = unit(d_vec)
    v = pole - np.dot(pole, u) * u
    v = unit(v) if np.linalg.norm(v) > 1e-6 else unit(np.cross(u, [0, 0, 1.0]))
    cos_a = np.clip((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1)
    a = np.arccos(cos_a)
    return root + l1 * (np.cos(a) * u + np.sin(a) * v)
