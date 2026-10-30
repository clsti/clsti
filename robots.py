"""Real robot models for any profile in profiles.py: model preparation (sole and hand sites, wheels for the
wheeled quadrupeds, grippers), body measurements for the motion planners, and damped-least-squares IK that
turns the planners' targets into joint angles."""
import functools
import math
import os

import mujoco
import numpy as np

import profiles as PR
from common import mat2quat, quat2mat, rot_y, rot_z, slerp, unit, wrap

HERE = os.path.dirname(os.path.abspath(__file__))
MENAGERIE = os.path.join(HERE, "assets", "menagerie")
SOURCES = dict(menagerie=MENAGERIE, extra=os.path.join(HERE, "assets", "extra"))


def body(spec, name):
    """spec.body() that also works for specs imported from URDF (whose name lookup can come up empty)."""
    b = spec.body(name)
    return b if b is not None else next(x for x in spec.bodies if x.name == name)


def load_spec(P):
    """MjSpec of a robot from MuJoCo Menagerie or assets/extra (MJCF or URDF).  URDF: ROS package:// paths
    are resolved, glTF visuals are swapped for the OBJ copies made by tools/gltf2obj.py, visual geoms are kept
    and empty placeholder geoms dropped.  Robots modelled with a fixed base get a free joint."""
    path = os.path.join(SOURCES[P.get("src", "menagerie")], P["xml"])
    if path.endswith(".urdf"):
        import re
        src = open(path, encoding="utf-8").read()
        for pkg, rel in P.get("packages", {}).items():
            src = src.replace(f"package://{pkg}/", os.path.join(SOURCES[P["src"]], rel).replace("\\", "/") + "/")
        src = src.replace('.gltf"', '.obj"')
        base = os.path.dirname(path).replace("\\", "/")
        src = re.sub(r'filename="(?![A-Za-z]:|/)([^"]+)"', lambda mt: f'filename="{base}/{mt.group(1)}"', src)
        src = re.sub(r"<mujoco>.*?</mujoco>", "", src, flags=re.S)        # mesh paths are absolute: drop meshdir
        src = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler discardvisual="false" balanceinertia="true" '
                     r'fusestatic="false"/></mujoco>', src, count=1)
        spec = mujoco.MjSpec.from_string(src)
        for g in list(spec.geoms):
            if g.type != mujoco.mjtGeom.mjGEOM_MESH and not np.asarray(g.size)[0] > 0:
                spec.delete(g)
    else:
        spec = mujoco.MjSpec.from_file(path)
    if P.get("gripper"):
        mount_robotiq(spec, P)
    root = body(spec, P["root"])
    if not any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in root.joints):
        root.add_freejoint()
    return spec


ROBOTIQ = os.path.join(SOURCES["extra"], "robotiq", "robotiq_3f_gripper_visualization", "meshes",
                       "robotiq-3f-gripper_articulated", "visual")


def robotiq_3f_spec():
    """The articulated Robotiq 3-finger gripper of ROS-Industrial (robotiq_3f_gripper_visualization, BSD-2),
    expanded from its xacro: palm, fingers 1 and 2 (spread joint + 3 joints) and the fixed middle finger."""
    mesh = lambda n: f'<mesh filename="{ROBOTIQ}/{n}.obj"/>'.replace("\\", "/")
    vis = lambda n, xyz="0 0 0", rpy="0 0 0": (f'<visual><origin xyz="{xyz}" rpy="{rpy}"/><geometry>{mesh(n)}'
                                               f'</geometry></visual>')
    inert = '<inertial><mass value="0.1"/><inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>'
    links, joints = [f'<link name="palm">{inert}{vis("palm")}</link>'], []
    for f in ("finger_1_", "finger_2_", "finger_middle_"):
        links += [f'<link name="{f}link_0">{inert}{vis("link_0", "0.020 0 0")}</link>',
                  f'<link name="{f}link_1">{inert}{vis("link_1", "0.050 -.028 0", "0 0 -0.52")}</link>',
                  f'<link name="{f}link_2">{inert}{vis("link_2", "0.039 0 0.0075")}</link>',
                  f'<link name="{f}link_3">{inert}{vis("link_3", "0 0 0", "0 0 0.52")}</link>']
        for k, parent, xyz, rpy, lo, hi in ((1, 0, "0.020 0 0", "0 0 0", 0.0495, 1.2218),
                                            (2, 1, "0.050 -.028 0", "0 0 -0.52", 0.0, 1.5708),
                                            (3, 2, "0.039 0 0", "0 0 0", -1.2217, -0.0523)):
            joints.append(f'<joint name="{f}joint_{k}" type="revolute"><parent link="{f}link_{parent}"/>'
                          f'<child link="{f}link_{k}"/><axis xyz="0 0 1"/><origin xyz="{xyz}" rpy="{rpy}"/>'
                          f'<limit lower="{lo}" upper="{hi}" effort="100" velocity="100"/></joint>')
    for f, xyz, lo, hi in (("finger_1_", "-0.0455 0.0214 0.036", -0.1784, 0.192),
                           ("finger_2_", "-0.0455 0.0214 -0.036", -0.192, 0.1784)):
        joints.append(f'<joint name="palm_{f}joint" type="revolute"><parent link="palm"/><child link="{f}link_0"/>'
                      f'<axis xyz="0 1 0"/><origin xyz="{xyz}" rpy="0 3.1416 -1.57"/>'
                      f'<limit lower="{lo}" upper="{hi}" effort="100" velocity="100"/></joint>')
    joints.append('<joint name="palm_finger_middle_joint" type="fixed"><parent link="palm"/>'
                  '<child link="finger_middle_link_0"/><origin xyz="0.0455 0.0214 0" rpy="0 0 1.57"/></joint>')
    urdf = ('<robot name="robotiq_3f"><mujoco><compiler discardvisual="false" fusestatic="false"/></mujoco>'
            + "".join(links) + "".join(joints) + "</robot>")
    return mujoco.MjSpec.from_string(urdf)


def mount_robotiq(spec, P):
    """Replace a robot's gripper placeholder (Atlas: a convex hull of the open gripper) by the articulated
    Robotiq 3-finger gripper; its palm frame sits `pos` out from the mounting body, fingers along +y."""
    G = P["gripper"]
    for side in PR.SIDES:
        b = body(spec, PR.name(G["body"], side))
        for g in list(b.geoms):
            spec.delete(g)
        frame = b.add_frame(pos=G["pos"])
        rq = robotiq_3f_spec()
        frame.attach_body(body(rq, "palm"), f"{side['s']}_rq_", "")
    for b in spec.bodies:
        if b.name.startswith(("left_rq_", "right_rq_")):
            for g in b.geoms:
                g.contype = g.conaffinity = 0
                g.rgba = G["rgba"]


def _strip(spec):
    """Drop keyframes, actuators, sensors and the models' own lights and floors (the scene brings those)."""
    for light in list(spec.lights):
        spec.delete(light)
    for g in list(spec.worldbody.geoms):     # floors and markers that some models bring along
        spec.delete(g)
    for k in list(spec.keys):
        spec.delete(k)
    for a in list(spec.actuators):
        spec.delete(a)
    for s in list(spec.sensors):
        spec.delete(s)


def _obstacles(spec):
    """Collision shapes become obstacles for flying blocks (contype 2), visuals never collide."""
    for g in spec.geoms:
        g.contype, g.conaffinity = (2, 0) if (g.contype or g.conaffinity) else (0, 0)


def _world_verts(m, d, body):
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * m.geom_size[g]
            pts.append(c @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(pts)


def root_slices(m, root_body, nq, nv):
    """qpos / qvel slices of an attached robot (its joints are contiguous, starting at the free joint)."""
    j = m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root_body)]
    a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
    return slice(a, a + nq), slice(v, v + nv)


# ----------------------------------------------------------------------------- humanoids
def stand_qpos(m, P, z=1.0):
    q = m.qpos0.copy()
    q[0:3], q[3:7] = (0, 0, z), (1, 0, 0, 0)
    for i, side in enumerate(PR.SIDES):
        for tmpl, v in P["stand"].items():
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, PR.name(tmpl, side))
            q[m.jnt_qposadr[j]] = v * PR.mirror_sign(m.jnt_axis[j], i == 1)
    return q


@functools.lru_cache(maxsize=None)
def humanoid_parts(key):
    """Site placements and body measurements of a humanoid in its standing pose (root frame, feet on 0)."""
    P = PR.HUMANOIDS[key]
    m = load_spec(P).compile()
    d = mujoco.MjData(m)
    d.qpos[:] = stand_qpos(m, P)
    mujoco.mj_kinematics(m, d)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    soles, sole_sites, foot_half = [], [], None
    for side in PR.SIDES:
        fb = bid(PR.name(P["foot"], side))
        v = _world_verts(m, d, fb)
        low = v[v[:, 2] < v[:, 2].min() + 0.006]
        c = np.array([(low[:, 0].min() + low[:, 0].max()) / 2, (low[:, 1].min() + low[:, 1].max()) / 2,
                      v[:, 2].min()])
        Rf = d.xmat[fb].reshape(3, 3)
        sole_sites.append((Rf.T @ (c - d.xpos[fb]), mat2quat(Rf.T)))
        soles.append(c)
        foot_half = ((low[:, 0].max() - low[:, 0].min()) / 2 + 0.01, (low[:, 1].max() - low[:, 1].min()) / 2 + 0.008)
    zs = 1.0 - min(s[2] for s in soles)                       # standing root height
    root = d.xpos[bid(P["root"])] + np.array([0, 0, zs - 1.0])
    rel = lambda p: p + np.array([0, 0, zs - 1.0]) - root      # world (standing) -> root-relative

    H = P["hand"]
    palm_sites = []
    for i, side in enumerate(PR.SIDES):
        mir = np.array([1, -1 if i and H.get("mirror", True) else 1, 1])
        pos = np.array(H["site"]) * mir
        rot = np.array(H["rot_right" if i and "rot_right" in H else "rot"], float) if H.get("rot") else np.eye(3)
        grasp = pos + rot @ (np.array(H["grasp"]) * mir) if "grasp" in H else pos
        palm_sites.append((pos, mat2quat(rot), grasp))
    sh, el, wr, palm = [], [], [], []
    for i, side in enumerate(PR.SIDES):
        sh.append(rel(d.xanchor[jid(PR.name(P["arm"][0], side))]))
        el.append(rel(d.xanchor[jid(PR.name(P["elbow"], side))]))
        hb = bid(PR.name(H["body"], side))
        wr.append(rel(d.xpos[hb]))
        palm.append(rel(d.xpos[hb] + d.xmat[hb].reshape(3, 3) @ palm_sites[i][0]))
    hip = rel(d.xanchor[jid(PR.name(P["leg"][0], PR.SIDES[0]))])
    yaw_j = [jid(PR.name(n, PR.SIDES[0])) for n in P["leg"]]
    yaw_j = [j for j in yaw_j if abs(m.jnt_axis[j][2]) > 0.5][0]          # hip yaw: limits the turn per step
    hip_yaw = float(min(abs(m.jnt_range[yaw_j][0]), abs(m.jnt_range[yaw_j][1])))
    waist = 0.0
    if "pitch" in P["waist"]:
        waist = rel(d.xanchor[jid(P["waist"]["pitch"][0])])[2]
    geo = dict(
        z_stand=zs, waist=waist, shoulder=sh[0] - np.array([0, 0, waist]),
        l_uarm=float(np.linalg.norm(el[0] - sh[0])), l_farm=float(np.linalg.norm(wr[0] - el[0])),
        grip=float(np.linalg.norm(palm[0] - wr[0])), rest_hand=[p - s for p, s in zip(palm, sh)],
        foot_w=float(abs(rel(soles[0])[1])), foot_half=foot_half, leg=float(hip[2] + zs),
        head=float(d.xpos[:, 2].max() - d.xpos[bid(P["root"])][2]),     # highest body origin above the root
        hip_yaw=hip_yaw,
    )
    return dict(sole_sites=sole_sites, palm_sites=palm_sites, geo=geo)


def humanoid_spec(key):
    P = PR.HUMANOIDS[key]
    parts = humanoid_parts(key)
    spec = load_spec(P)
    _strip(spec)
    _obstacles(spec)
    for i, side in enumerate(PR.SIDES):
        pos, quat = parts["sole_sites"][i]
        body(spec, PR.name(P["foot"], side)).add_site(name=f"ik_{side['s']}_sole", pos=pos, quat=quat, size=[0.01, 0, 0])
        pos, quat, grasp = parts["palm_sites"][i]
        hb = body(spec, PR.name(P["hand"]["body"], side))
        hb.add_site(name=f"ik_{side['s']}_palm", pos=pos, quat=quat, size=[0.01, 0, 0])
        hb.add_site(name=f"ik_{side['s']}_grasp", pos=grasp, size=[0.01, 0, 0])
    return spec


@functools.lru_cache(maxsize=None)
def humanoid_motion(key):
    """Walking height and squat that this robot's legs can really do (checked with the leg IK against its
    joint limits): the highest pelvis at which a full stride with a lifted swing foot still fits, and the
    deepest squat (pelvis drop, hips back, torso lean) that keeps both soles flat."""
    ik = HumanoidIK(key)
    g = ik.geo
    k = g["leg"] / 0.66
    stride, lift, fw = 0.23 * k, 0.06 * k, g["foot_w"]
    has_pitch = "pitch" in ik.waist

    def worst(z, lean, back, feet):
        d = ik.d
        wp = float(np.clip(0.4 * lean, -0.5, 0.5)) if has_pitch else 0.0
        d.qpos[0:3] = (-back, 0.0, z)
        d.qpos[3:7] = mat2quat(rot_y(lean - wp))
        ik._set(ik.waist, "pitch", wp)
        err = 0.0
        for i, side in enumerate(PR.SIDES):
            ch = ik.chains[(side["s"], "leg")]
            d.qpos[ch["qadr"]] = ik.q0[ch["qadr"]]
            pos, pitch = feet[i]
            ik._solve(ch, pos, rot_y(pitch), w_rot=0.6, iters=80, secondary=len(ch["dof"]) < 6)
            mujoco.mj_kinematics(ik.m, d)
            err = max(err, float(np.linalg.norm(d.site_xpos[ch["site"]] - pos)))
        return err

    walk = [[(np.array([-stride / 2, fw, 0]), 0.0), (np.array([stride / 2, -fw, lift]), 0.18)],
            [(np.array([stride / 2, fw, lift]), -0.18), (np.array([-stride / 2, -fw, 0]), 0.0)],
            [(np.array([stride / 2, fw, 0]), 0.0), (np.array([-stride / 2, -fw, 0]), 0.0)]]
    z0 = g["z_stand"] - 0.12 * k
    for dz in np.arange(0.014, 0.12, 0.006) * k:
        if all(worst(g["z_stand"] - dz, 0.0, 0.0, f) < 0.003 for f in walk):
            z0 = g["z_stand"] - dz
            break
    flat = [(np.array([0, fw, 0]), 0.0), (np.array([0, -fw, 0]), 0.0)]
    best = None
    for drop in (0.52, 0.48, 0.42, 0.36, 0.30, 0.24, 0.18):
        for lean in (1.3, 1.1, 0.9, 0.7, 0.5, 0.3):
            for back in (0.16, 0.10, 0.04, -0.02):
                D, B = drop * z0, back * z0
                if any(worst(z0 - D * c, lean * c, B * c, flat) > 0.004 for c in (1.0, 0.6, 0.3)):
                    continue
                # lower, more forward shoulders reach the floor more easily
                Rt = rot_y(lean)
                sh = np.array([-B, 0.0, z0 - D + g["waist"]]) + Rt @ g["shoulder"]
                score = sh[2] - 0.6 * sh[0]
                if best is None or score < best[0]:
                    best = (score, (B, D, lean))
    return dict(z0=z0, crouch=best[1] if best else (0.04 * z0, 0.18 * z0, 0.7))


class HumanoidIK:
    """Leg IK on the sole sites (6-DoF pose; position first for 5-DoF legs), arm IK on the palm sites with
    the hand orientation in the null space, or as a hard task while grasping.  A posture term in the
    remaining null space keeps elbows low and close to the body and the wrists straight."""
    ARM_W_ROT = 0.5
    GRASP_W_ROT = 0.3

    def __init__(self, key):
        self.key, P = key, PR.HUMANOIDS[key]
        self.P = P
        self.m = humanoid_spec(key).compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        self.geo = humanoid_parts(key)["geo"]
        self.q0 = stand_qpos(m, P, self.geo["z_stand"])
        self.d.qpos[:] = self.q0
        self.chains = {}
        for side in PR.SIDES:
            for kind, names, site in (("leg", P["leg"], "sole"), ("arm", P["arm"], "palm")):
                j = [jid(PR.name(n, side)) for n in names]
                self.chains[(side["s"], kind)] = dict(
                    qadr=np.array([m.jnt_qposadr[i] for i in j]), dof=np.array([m.jnt_dofadr[i] for i in j]),
                    lo=m.jnt_range[j, 0], hi=m.jnt_range[j, 1], site=sid(f"ik_{side['s']}_{site}"))
            ch = self.chains[(side["s"], "arm")]
            e = jid(PR.name(P["elbow"], side))
            k = list(ch["qadr"]).index(m.jnt_qposadr[e])
            ch.update(elbow_jnt=e, elbow_body=m.jnt_bodyid[e], wrist=np.arange(len(ch["qadr"])) > k)
        self.waist = {k: (m.jnt_qposadr[jid(n)], s, m.jnt_range[jid(n)]) for k, (n, s) in P["waist"].items()}
        self.neck = {k: (m.jnt_qposadr[jid(n)], s, m.jnt_range[jid(n)]) for k, (n, s) in (P["neck"] or {}).items()}
        self.fingers = None
        if P.get("fingers"):
            F = P["fingers"]
            self.fingers = {side["s"]: (np.array([m.jnt_qposadr[jid(PR.name(n, side))] for n in F["joints"]]),
                                        np.array(F["mirror"], float) ** (1 if i else 0))
                            for i, side in enumerate(PR.SIDES)}
            self.shapes = {k: np.array(F[k]) for k in ("rest", "open", "grip")}
        self.ball = P["hand"]["kind"] == "ball"
        mujoco.mj_kinematics(m, self.d)
        # hanging-hand orientation in the robot frame
        self.hand_rest = {s["s"]: self.d.site_xmat[self.chains[(s["s"], "arm")]["site"]].reshape(3, 3).copy()
                          for s in PR.SIDES}
        self.grasp_sites = [sid(f"ik_{s['s']}_grasp") for s in PR.SIDES]
        self.q_carry = None
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def _posture(self, chain, out_dir):
        """Null-space gradient: elbow down and in, wrist joints back to neutral."""
        m, d = self.m, self.d
        mujoco.mj_jac(m, d, self.jacp, None, d.xanchor[chain["elbow_jnt"]], chain["elbow_body"])
        Je = self.jacp[:, chain["dof"]]
        g = -0.25 * Je.T @ (np.array([0, 0, 1.0]) + 0.8 * out_dir)
        q = d.qpos[chain["qadr"]]
        g[chain["wrist"]] += -0.15 * q[chain["wrist"]]
        # never push a joint into its limit (that would block the reaching task)
        room = np.where(g < 0, q - chain["lo"], chain["hi"] - q)
        return g * np.clip(room / 0.15, 0.0, 1.0)

    def _solve(self, chain, pos, R=None, w_rot=1.0, iters=40, secondary=False, posture=None):
        """Damped least squares. With secondary=True the orientation is only tracked in the null space
        of the position task (redundant arms), so reaching always wins over hand alignment."""
        m, d = self.m, self.d
        n = len(chain["dof"])
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            s = chain["site"]
            e = pos - d.site_xpos[s]
            mujoco.mj_jacSite(m, d, self.jacp, self.jacr, s)
            J = self.jacp[:, chain["dof"]]
            er = None
            if R is not None:
                Rc = d.site_xmat[s].reshape(3, 3)
                er = 0.5 * sum(np.cross(Rc[:, i], R[:, i]) for i in range(3))
            g = posture(chain) if posture is not None else None
            if secondary:
                Jp_pinv = J.T @ np.linalg.inv(J @ J.T + 2e-3 * np.eye(3))
                dq = Jp_pinv @ e
                N = np.eye(n) - Jp_pinv @ J
                sec = np.zeros(n)
                if er is not None:
                    Jr = self.jacr[:, chain["dof"]]
                    sec += Jr.T @ np.linalg.solve(Jr @ Jr.T + 1e-2 * np.eye(3), w_rot * er)
                if g is not None:
                    sec += g
                dq += N @ sec
                if g is None and np.linalg.norm(e) < 2e-4 and (er is None or np.linalg.norm(er) < 0.02):
                    break
            else:
                if er is not None:
                    e = np.concatenate([e, w_rot * er])
                    J = np.vstack([J, w_rot * self.jacr[:, chain["dof"]]])
                if g is None and np.linalg.norm(e) < 2e-4:
                    break
                Jpinv = J.T @ np.linalg.inv(J @ J.T + 2e-3 * np.eye(len(e)))
                dq = Jpinv @ e
                if g is not None:
                    dq += (np.eye(n) - Jpinv @ J) @ g
            dq *= min(1.0, 0.25 / (np.abs(dq).max() + 1e-12))
            d.qpos[chain["qadr"]] = np.clip(d.qpos[chain["qadr"]] + dq, chain["lo"], chain["hi"])

    def _carry_seed(self):
        """Arm joints for a typical carry, found once from many starts: the fallback seed when the hanging-arm
        seed gets stuck at a joint limit (wide-shouldered robots reaching in to a narrow block)."""
        import itertools
        import humanoid as H
        from common import yaw_quat
        d, q = self.d, {}
        st = H.HState(0.0, 0.0, 0.0)
        g = H.Grasp(yaw_quat(0.0), np.array([0.042, 0.042, 0.09]), 0.0)
        pos, quat = H.carry_pose(st, g)
        palms, frames, _ = H.hand_frames(pos, quat, g, st)
        keep = d.qpos.copy()
        d.qpos[:] = self.q0
        d.qpos[0:3] = st.pelvis
        for i, side in enumerate(PR.SIDES):
            ch = self.chains[(side["s"], "arm")]
            best = None
            for seed in itertools.product((0.0, 0.8, -0.8), (-0.4, 0.4), (-0.6, -1.4, 0.6, 1.4)):
                d.qpos[ch["qadr"]] = self.q0[ch["qadr"]]
                d.qpos[ch["qadr"][[0, 1]]] = seed[:2]
                d.qpos[ch["qadr"][list(ch["qadr"]).index(self.m.jnt_qposadr[ch["elbow_jnt"]])]] = seed[2]
                d.qpos[ch["qadr"]] = np.clip(d.qpos[ch["qadr"]], ch["lo"], ch["hi"])
                self._solve(ch, palms[i], frames[i], w_rot=self.GRASP_W_ROT, iters=80)
                mujoco.mj_kinematics(self.m, d)
                e = np.linalg.norm(d.site_xpos[ch["site"]] - palms[i])
                if best is None or e < best[0]:
                    best = (e, d.qpos[ch["qadr"]].copy())
            q[side["s"]] = best[1]
        d.qpos[:] = keep
        return q

    def _set(self, joints, key, value):
        if key in joints:
            adr, sg, rng = joints[key]
            self.d.qpos[adr] = np.clip(sg * value, rng[0], rng[1])

    def solve(self, st):
        import humanoid as H
        d = self.d
        has_pitch = "pitch" in self.waist
        waist_pitch = float(np.clip(0.4 * st.lean, -0.5, 0.5)) if has_pitch else 0.0
        d.qpos[0:3] = st.pelvis
        d.qpos[3:7] = mat2quat(rot_z(st.yaw) @ rot_y(st.lean - waist_pitch))
        look_yaw = look_pitch = 0.0
        if st.look is not None:
            v = st.look - (st.pelvis + np.array([0, 0, self.geo["head"]]))
            look_yaw = float(np.clip(wrap(math.atan2(v[1], v[0]) - st.yaw), -1.2, 1.2))
            look_pitch = float(np.clip(math.atan2(-v[2], np.linalg.norm(v[:2])) - st.lean, -0.3, 0.5))
        if self.neck:
            self._set(self.neck, "yaw", 0.8 * look_yaw)
            self._set(self.neck, "pitch", look_pitch)
            self._set(self.waist, "yaw", st.twist + 0.2 * look_yaw)
        else:
            self._set(self.waist, "yaw", st.twist + 0.6 * float(np.clip(look_yaw, -0.9, 0.9)))
        self._set(self.waist, "pitch", waist_pitch)
        self._set(self.waist, "roll", 0.0)

        for i, side in enumerate(PR.SIDES):
            fpos, fyaw, fpitch = st.feet[i]
            ch = self.chains[(side["s"], "leg")]
            d.qpos[ch["qadr"]] = self.q0[ch["qadr"]]           # bent-knee seed every call
            self._solve(ch, fpos, rot_z(fyaw) @ rot_y(fpitch), w_rot=0.6, iters=60, secondary=len(ch["dof"]) < 6)

        mujoco.mj_kinematics(self.m, d)
        relaxed = H.relaxed_grips(st)
        Ry = rot_z(st.yaw)
        for i, side in enumerate(PR.SIDES):
            s = side["s"]
            target = st.hands[i] if st.hands[i] is not None else relaxed[i]
            sh = H.shoulders(st)[i]
            out_dir = Ry[:, 1] * (1 if i == 0 else -1)
            chain = self.chains[(s, "arm")]
            d.qpos[chain["qadr"]] = self.q0[chain["qadr"]]     # same seed every call: no frame-to-frame flicker
            posture = lambda ch, o=out_dir: self._posture(ch, o)
            if self.ball:
                self._solve(chain, target, iters=50, posture=posture)
            else:
                v = Ry.T @ (target - sh)
                phi = math.atan2(v[0], -v[2])                 # swing the hand along the arm's pointing direction
                R = Ry @ rot_y(-phi) @ self.hand_rest[s]
                w, Rg = st.hand_w[i], st.hand_R[i]
                if Rg is not None and w > 0:                   # blend towards the grasp orientation
                    R = quat2mat(slerp(mat2quat(R), mat2quat(Rg), w))
                self._solve(chain, target, R, w_rot=self.ARM_W_ROT, secondary=True, posture=posture)
                if Rg is not None and w > 0:                   # grasping: the hand orientation is a hard task,
                    keep = d.qpos[chain["qadr"]].copy()        # unless that costs reach (joint limits)
                    self._solve(chain, target, R, w_rot=self.GRASP_W_ROT * w, iters=40, posture=posture)
                    mujoco.mj_kinematics(self.m, d)
                    if np.linalg.norm(d.site_xpos[chain["site"]] - target) > 0.005:
                        d.qpos[chain["qadr"]] = keep
            mujoco.mj_kinematics(self.m, d)
            if np.linalg.norm(d.site_xpos[chain["site"]] - target) > 0.005:   # stuck: retry from a carry pose
                if self.q_carry is None:
                    self.q_carry = self._carry_seed()
                    mujoco.mj_kinematics(self.m, d)
                d.qpos[chain["qadr"]] = self.q_carry[s]
                if self.ball:
                    self._solve(chain, target, iters=60, posture=posture)
                else:
                    self._solve(chain, target, R, w_rot=self.ARM_W_ROT, secondary=True, iters=60, posture=posture)
            mujoco.mj_kinematics(self.m, d)
            if np.linalg.norm(d.site_xpos[chain["site"]] - target) > 0.002:   # posture must never cost reach
                self._solve(chain, target, None if self.ball else R, w_rot=self.ARM_W_ROT, secondary=True, iters=30)
            if self.fingers is not None:
                adr, sg = self.fingers[s]
                sh_ = self.shapes
                shape = sh_["rest"] + (sh_["open"] - sh_["rest"]) * st.hand_w[i]
                d.qpos[adr] = sg * (shape + (sh_["grip"] - shape) * st.grip)
        return d.qpos.copy()


# ----------------------------------------------------------------------------- quadrupeds
WHEEL_MATERIALS = dict(rubber=([0.04, 0.04, 0.045, 1], 0.15, 0.2), rim=([0.62, 0.64, 0.66, 1], 0.8, 0.8),
                       motor=([0.12, 0.12, 0.13, 1], 0.5, 0.6))


def _subtree(m, root):
    out = []
    for b in range(m.nbody):
        a = b
        while a > root:
            a = m.body_parentid[a]
        if a == root:
            out.append(b)
    return out


@functools.lru_cache(maxsize=None)
def quadruped_legs(key):
    """Per leg, in the lower-leg frame: foot centre (foot sphere, or the hub of a built-in wheel) and
    its radius, outward lateral axis, up axis."""
    Q = PR.QUADRUPEDS[key]
    W = Q["wheel"] or {}
    m = load_spec(Q).compile()
    d = mujoco.MjData(m)
    d.qpos[3:7] = (1, 0, 0, 0)                 # qpos0 may carry a yaw (ANYmal's does)
    mujoco.mj_kinematics(m, d)
    Ry = rot_z(Q["yaw_offset"])
    out = []
    for leg in Q["legs"]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, Q["shank"].format(l=leg))
        Rs = d.xmat[b].reshape(3, 3)
        if W.get("native"):
            wb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, W["body"].format(l=leg))
            r = W.get("radius") or max(np.linalg.norm(_world_verts(m, d, wb) - d.xpos[wb], axis=1).max() * 0.98, 0.05)
            centre = d.xpos[wb]
        else:
            sub = _subtree(m, b)
            spheres = [g for g in range(m.ngeom) if m.geom_bodyid[g] in sub and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            g = min(spheres, key=lambda g: d.geom_xpos[g][2])
            r, centre = float(m.geom_size[g][0]), d.geom_xpos[g]
        lateral = Ry[:, 1] * (1 if _is_left(leg) else -1)
        out.append(dict(foot=Rs.T @ (centre - d.xpos[b]), radius=float(r),
                        out=Rs.T @ lateral, up=Rs.T @ np.array([0, 0, 1.0])))
    return out


@functools.lru_cache(maxsize=None)
def quadruped_nominal(key):
    """Ride height, stance (x, y of the feet / wheels) and wheel or foot radius.  "auto" in the profile:
    measured from the standing seed pose."""
    Q = PR.QUADRUPEDS[key]
    r = (Q["wheel"] or {}).get("radius") if Q["wheel"] and not Q["wheel"].get("native") else quadruped_legs(key)[0]["radius"]
    if Q["ride_z"] != "auto" and Q["stance"] != "auto":
        return Q["ride_z"], tuple(Q["stance"]), r
    ik = QuadrupedIK(key)
    ik.d.qpos[:] = ik.seed
    ik.d.qpos[0:3] = (0, 0, 1.0)
    mujoco.mj_kinematics(ik.m, ik.d)
    pts = np.array([ik._point(c) for c in ik.legs])
    ride = 1.0 - pts[:, 2].mean() + r
    stance = (float(np.abs(pts[:, 0]).mean()), float(np.abs(pts[:, 1]).mean()))
    return (ride if Q["ride_z"] == "auto" else Q["ride_z"]), (stance if Q["stance"] == "auto" else tuple(Q["stance"])), r


def quadruped_spec(key):
    Q = PR.QUADRUPEDS[key]
    legs = quadruped_legs(key)
    spec = load_spec(Q)
    _strip(spec)
    _obstacles(spec)
    W = Q["wheel"]
    native = bool(W and W.get("native"))
    if not W:
        for g in spec.geoms:            # stepping feet push blocks aside instead of catapulting them
            if g.contype:
                g.solref = [0.03, 1.0]
    if W and not native:
        for g in spec.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and g.contype:   # the point feet sit inside the hubs
                g.contype, g.conaffinity = 0, 0
        for n, (rgba, spc, shn) in WHEEL_MATERIALS.items():
            spec.add_material(name=n, rgba=rgba, specular=spc, shininess=shn)
    for leg, L in zip(Q["legs"], legs):
        if native:                       # the model has its own wheels
            continue
        shank = body(spec, Q["shank"].format(l=leg))
        foot = L["foot"]
        if not W:
            shank.add_site(name=f"{leg}_foot", pos=foot, size=[0.01, 0, 0])
            continue
        R, Wd = W["radius"], W["width"]
        # wheel frame: y along the axle (outward), z up when the leg is straight
        ay = unit(L["out"])
        az = unit(L["up"] - ay * (L["up"] @ ay))
        wb = shank.add_body(name=f"{leg}_wheel", pos=foot + ay * W["lateral"],
                            quat=mat2quat(np.column_stack([np.cross(ay, az), ay, az])))
        wb.add_joint(name=f"{leg}_wheel_joint", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0])
        q_axle = [math.sqrt(0.5), math.sqrt(0.5), 0, 0]
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[R, Wd / 2, 0], quat=q_axle,
                    material="rubber", contype=2, conaffinity=0, mass=0.8)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.62 * R, Wd / 2 + 0.002, 0],
                    quat=q_axle, material="motor", contype=0, conaffinity=0, mass=0)
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.3 * R, Wd / 2 + 0.006, 0],
                    quat=q_axle, material="rim", contype=0, conaffinity=0, mass=0)
        for k in range(5):
            a = 2 * math.pi * k / 5
            wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.16 * R, Wd / 2 + 0.004, 0.012],
                        pos=[0.42 * R * math.cos(a), 0, 0.42 * R * math.sin(a)],
                        quat=[math.cos(-a / 2), 0, math.sin(-a / 2), 0], material="rim",
                        contype=0, conaffinity=0, mass=0)
    return spec


def _is_left(leg):
    return leg.upper() in ("FL", "RL", "HL", "LF", "LH")


class QuadrupedIK:
    """Position IK per leg on the wheel centre (wheeled) or the foot centre (walking)."""

    def __init__(self, key):
        self.key, Q = key, PR.QUADRUPEDS[key]
        self.Q = Q
        self.m = quadruped_spec(key).compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.q_off = mat2quat(rot_z(Q["yaw_offset"]))
        self.d.qpos[2], self.d.qpos[3] = 0.5, 1
        self.legs = []
        for leg in Q["legs"]:
            j = [jid(n.format(l=leg)) for n in Q["joints"]]
            c = dict(qadr=np.array([m.jnt_qposadr[i] for i in j]), dof=np.array([m.jnt_dofadr[i] for i in j]),
                     lo=m.jnt_range[j, 0], hi=m.jnt_range[j, 1])
            if Q["wheel"] and Q["wheel"].get("native"):
                c.update(body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, Q["wheel"]["body"].format(l=leg)),
                         spin=m.jnt_qposadr[jid(Q["wheel"]["joint"].format(l=leg))])
            elif Q["wheel"]:
                c.update(body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_wheel"),
                         spin=m.jnt_qposadr[jid(f"{leg}_wheel_joint")])
            else:
                c.update(site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot"))
            self.legs.append(c)
        self.seed = self._seed()
        self.d.qpos[:] = self.seed
        mujoco.mj_kinematics(m, self.d)
        self.knee_side = [self._knee_side(c) for c in self.legs]
        self.d.qpos[7:] = self.seed[7:]
        self.jacp = np.zeros((3, m.nv))

    def _point(self, c):
        return self.d.xpos[c["body"]] if "body" in c else self.d.site_xpos[c["site"]]

    def _seed(self):
        """Standing joint angles. Without a given seed, pick the knee direction per leg like the real robot:
        front knees bend backwards, hind knees forwards (ANYmal's X stance)."""
        m, d, Q = self.m, self.d, self.Q
        q = d.qpos.copy()
        q[3:7] = self.q_off
        if Q["seed"] is not None:
            for c in self.legs:
                q[c["qadr"]] = Q["seed"]
            return q
        fwd = rot_z(Q["yaw_offset"])[:, 0]
        for k, c in enumerate(self.legs):
            best = None
            for a in (0.6, -0.6):
                d.qpos[:] = q
                d.qpos[c["qadr"]] = [0.0, a, -2 * a]
                mujoco.mj_kinematics(m, d)
                knee = d.xanchor[self._jnt_of(c["qadr"][2])] - d.xanchor[self._jnt_of(c["qadr"][1])]
                score = -knee @ fwd if (k < 2 or Q.get("knees") == "back") else knee @ fwd
                if best is None or score > best[0]:
                    best = (score, a)
            q[c["qadr"]] = [0.0, best[1], -2 * best[1]]
        return q

    def _knee_side(self, c):
        """Which side of the hip-foot line the knee is on, along the body's forward axis."""
        d = self.d
        mid = 0.5 * (d.xanchor[self._jnt_of(c["qadr"][1])] + self._point(c))
        return np.sign((d.xanchor[self._jnt_of(c["qadr"][2])] - mid) @ d.xmat[1].reshape(3, 3)[:, 0])

    def _jnt_of(self, qadr):
        return int(np.flatnonzero(self.m.jnt_qposadr == qadr)[0])

    def solve(self, pos, quat, targets, spins=None, iters=30):
        m, d = self.m, self.d
        d.qpos[0:3] = pos
        d.qpos[3:7] = mujoco_mul(quat, self.q_off)
        for c, side, target in zip(self.legs, self.knee_side, targets):
            self._leg(c, target, iters)
            mujoco.mj_kinematics(m, d)
            if self._knee_side(c) != side:          # knee went through the straight leg: start again from the seed
                d.qpos[c["qadr"]] = self.seed[c["qadr"]]
                self._leg(c, target, 2 * iters)
        if spins is not None and self.Q["wheel"]:
            for c, sp in zip(self.legs, spins):
                d.qpos[c["spin"]] = sp
        return d.qpos.copy()

    def _leg(self, c, target, iters):
        m, d = self.m, self.d
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e = target - self._point(c)
            if np.linalg.norm(e) < 2e-4:
                break
            if "body" in c:
                mujoco.mj_jacBody(m, d, self.jacp, None, c["body"])
            else:
                mujoco.mj_jacSite(m, d, self.jacp, None, c["site"])
            J = self.jacp[:, c["dof"]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), e)
            dq *= min(1.0, 0.25 / (np.abs(dq).max() + 1e-12))
            d.qpos[c["qadr"]] = np.clip(d.qpos[c["qadr"]] + dq, c["lo"], c["hi"])


def mujoco_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def attach(scene_spec, hkey, qkey):
    """Attach both robots to the scene: body prefixes h_ (humanoid) and q_ (quadruped)."""
    for child, prefix in ((humanoid_spec(hkey), "h_"), (quadruped_spec(qkey), "q_")):
        frame = scene_spec.worldbody.add_frame()
        scene_spec.attach(child, prefix=prefix, frame=frame)
