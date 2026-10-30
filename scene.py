"""MJCF scene: a concrete lab floor, the calendar printed on a rubber mat, one physical block per active day,
and the two robots (see robots.py)."""
import os
import random

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

PITCH = 0.10             # cell spacing (m)
HALF = 0.042             # half width of a block
HEIGHT = {1: 0.14, 2: 0.18, 3: 0.22, 4: 0.26}
BEVEL = 0.006
# painted-wood greens, light to dark (busier days are darker and taller)
LEVEL_RGB = [(62, 64, 67), (176, 201, 141), (118, 165, 94), (64, 126, 66), (33, 84, 49)]
MAT_EXTENT = (-0.34, 5.42, -0.26, 0.80)        # x0, x1, y0, y1 of the printed mat
QUALITY = dict(shadow=8192, samples=8)          # lowered with --fast (software rendering in CI)


def cell_xy(col, row):
    return np.array([col * PITCH, (6 - row) * PITCH])


def _fmt(v):
    return " ".join(f"{x:.5g}" for x in np.ravel(v))


def bevel_box_mesh(hx, hy, hz, b):
    """Chamfered box as an explicit triangle list (outward winding)."""
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts += [
                    (sx * hx, sy * (hy - b), sz * (hz - b)),
                    (sx * (hx - b), sy * hy, sz * (hz - b)),
                    (sx * (hx - b), sy * (hy - b), sz * hz),
                ]
    pts = np.array(pts)
    hull = ConvexHull(pts)
    faces = []
    for simplex, eq in zip(hull.simplices, hull.equations):
        a, b_, c = pts[simplex]
        if np.dot(np.cross(b_ - a, c - a), eq[:3]) < 0:
            simplex = simplex[[0, 2, 1]]
        faces.append(simplex)
    return pts, np.array(faces)


def build_spec(cells, total, width, height, texdir, hkey, qkey):
    import robots
    import textures

    os.makedirs(texdir, exist_ok=True)
    floor_png = os.path.abspath(os.path.join(texdir, "concrete.png"))
    mat_png = os.path.abspath(os.path.join(texdir, "mat.png"))
    if not os.path.exists(floor_png):
        textures.concrete(floor_png)
    textures.mat(mat_png, cells, total, MAT_EXTENT, PITCH, HALF, LEVEL_RGB[1:])

    x0, x1, y0, y1 = MAT_EXTENT
    out = []
    w = out.append
    w('<mujoco model="commit-robots">')
    w('<compiler angle="radian" autolimits="true"/>')
    w('<option timestep="0.001" integrator="implicitfast" cone="elliptic"/>')
    w('<size memory="96M"/>')
    w('<statistic center="2.7 0.4 0.4" extent="5"/>')
    w('<visual>')
    w(f'<global offwidth="{width}" offheight="{height}" fovy="20"/>')
    w(f'<quality shadowsize="{QUALITY["shadow"]}" offsamples="{QUALITY["samples"]}"/>')
    w('<headlight ambient=".20 .20 .21" diffuse=".10 .10 .10" specular="0 0 0"/>')
    w('<map znear="0.1" zfar="80" shadowclip="1.6" shadowscale="0.6" haze="0"/>')
    w('</visual>')

    w('<asset>')
    w('<texture name="sky" type="skybox" builtin="flat" rgb1=".5 .5 .5" rgb2=".5 .5 .5" width="16" height="16"/>')
    w(f'<texture name="concrete" type="2d" file="{floor_png}"/>')
    w(f'<texture name="mat" type="2d" file="{mat_png}"/>')
    w('<material name="floor" texture="concrete" texrepeat="0.45 0.45" texuniform="true" '
      'specular="0.08" shininess="0.15" reflectance="0.015"/>')
    w('<material name="mat" texture="mat" specular="0.05" shininess="0.05"/>')
    # the printed mat: one flat textured quad
    verts = [x0, y0, 0.0015, x1, y0, 0.0015, x1, y1, 0.0015, x0, y1, 0.0015]
    w(f'<mesh name="mat" vertex="{_fmt(verts)}" texcoord="0 1 1 1 1 0 0 0" face="0 1 2 0 2 3" '
      'inertia="shell"/>')
    for lvl in range(1, 5):
        pts, faces = bevel_box_mesh(HALF, HALF, HEIGHT[lvl] / 2, BEVEL)
        w(f'<mesh name="blk{lvl}" vertex="{_fmt(pts)}" face="{_fmt(faces)}"/>')
    rng = random.Random(7)
    active = [c for c in cells if c.level > 0]
    for i, c in enumerate(active):
        base = np.array(LEVEL_RGB[c.level]) / 255
        tint = np.clip(base * rng.uniform(0.93, 1.06) + rng.uniform(-0.015, 0.015), 0, 1)
        w(f'<material name="m{i}" rgba="{_fmt(tint)} 1" specular="0.18" shininess="0.25"/>')
    w('</asset>')

    w('<worldbody>')
    w('<light name="key" directional="true" pos="0 -3 6" dir="0.42 0.55 -0.72" diffuse=".68 .63 .56" '
      'specular=".25 .23 .20" castshadow="true"/>')
    w('<light name="fill" directional="true" pos="0 0 6" dir="-0.55 -0.15 -0.82" diffuse=".20 .24 .30" '
      'specular="0 0 0" castshadow="false"/>')
    # soft overhead light
    w('<light name="top" directional="true" pos="1 0 3.5" dir="0 0 -1" diffuse=".7 .7 .7" specular=".3 .3 .3" '
      'castshadow="true"/>')
    w('<geom name="floor" type="plane" size="40 40 .1" material="floor" contype="1" conaffinity="1" '
      'friction=".7 .01 .0001" condim="4"/>')
    w('<geom name="mat" type="mesh" mesh="mat" material="mat" contype="0" conaffinity="0"/>')
    for i, c in enumerate(active):
        x, y = cell_xy(c.col, c.row)
        h = HEIGHT[c.level]
        w(f'<body name="b{i}" pos="{x:.4f} {y:.4f} {h / 2:.4f}">')
        w('<freejoint/>')
        w(f'<geom type="box" size="{HALF} {HALF} {h / 2:.4f}" group="3" density="500" contype="1" '
          'conaffinity="3" condim="4" friction=".65 .01 .0001" solref=".004 1"/>')
        w(f'<geom type="mesh" mesh="blk{c.level}" material="m{i}" contype="0" conaffinity="0" density="0"/>')
        w('</body>')
    w('</worldbody>')
    w('</mujoco>')

    spec = mujoco.MjSpec.from_string("\n".join(out))
    robots.attach(spec, hkey, qkey)
    return spec
