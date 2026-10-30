"""Humanoid motion planning: footstep-planned walking (stance feet never slide), squatting pick & place.
Produces HState targets (pelvis, feet, hands, lean, grip); robots.HumanoidIK turns them into joint angles.
Body proportions come from the robot model: call configure(key) before planning."""
import copy
import math

import mujoco
import numpy as np

import profiles as PR
from common import lerp_angle, mat2quat, quat2mat, rot_y, rot_z, slerp, smooth, smoother, unit, wrap, yaw_quat
from scene import HALF

UP = np.array([0.0, 0.0, 1.0])
ALPHA_CARRY, ALPHA_LOW = 0.35, 1.40   # fingers below horizontal: carrying vs. reaching down to the floor
GRIP_TOP = 0.05                       # hands hold a block this far below its top (clear of the neighbours)
MAX_TURN = 0.55                       # yaw change per footstep (rad); configure() adapts it to the hip yaw range
SHUFFLE = 0.7                         # shorter moves: side-step / shuffle while turning

# set by configure()
KEY = None
Z0 = WAIST = L_UARM = L_FARM = GRIP = REACH = FOOT_W = STRIDE = STEP_T = TURN_T = LIFT = None
SHOULDER = REST_HAND = FOOT_HALF = CROUCH = CARRY_C = FACE = GRASP_OFF = HAND_PARTS = None
HAND_KIND = BALL_R = PICK_D = PLACE_D = HAND_ALPHA = HAND_TILT = None


def configure(key):
    """Adopt the proportions of one humanoid (measured from its model in the standing pose)."""
    import robots
    global KEY, Z0, WAIST, L_UARM, L_FARM, GRIP, REACH, FOOT_W, STRIDE, STEP_T, TURN_T, LIFT
    global SHOULDER, REST_HAND, FOOT_HALF, CROUCH, CARRY_C, FACE, GRASP_OFF, HAND_PARTS
    global HAND_KIND, BALL_R, PICK_D, PLACE_D, MAX_TURN, HAND_ALPHA, HAND_TILT
    if KEY == key:
        return
    KEY = key
    g, hand = robots.humanoid_parts(key)["geo"], PR.HUMANOIDS[key]["hand"]
    k = g["leg"] / 0.66                                       # leg length relative to the G1
    motion = robots.humanoid_motion(key)
    Z0 = motion["z0"]
    WAIST = g["waist"]
    SHOULDER = np.array(g["shoulder"])
    L_UARM, L_FARM, GRIP = g["l_uarm"], g["l_farm"], g["grip"]
    HAND_KIND = hand["kind"]
    HAND_ALPHA = hand.get("alpha")                            # grippers close horizontally at any height
    HAND_TILT = hand.get("tilt", 0.0)                         # grippers come from above near the floor
    BALL_R = hand.get("radius", 0.0)
    if HAND_KIND == "ball":
        GRASP_OFF = (np.zeros(3), np.zeros(3))
        HAND_PARTS = []
        contact = GRIP + BALL_R
    else:
        gl = np.array(hand["grasp"])
        GRASP_OFF = (gl, gl * np.array([1, -1, 1]))
        HAND_PARTS = hand["parts"]
        contact = GRIP + gl[0]
    REACH = L_UARM + L_FARM + GRIP
    reach_c = L_UARM + L_FARM + contact
    REST_HAND = [np.array(v) for v in g["rest_hand"]]
    FOOT_W, FOOT_HALF = g["foot_w"], g["foot_half"]
    STRIDE, STEP_T, TURN_T, LIFT = 0.23 * k, 0.38 * math.sqrt(k), 0.32 * math.sqrt(k), 0.06 * k
    MAX_TURN = min(0.5, 0.6 * g["hip_yaw"])     # the pelvis lags behind the feet in a turn: keep a margin
    MAX_TURN = PR.HUMANOIDS[key].get("max_turn", MAX_TURN)
    CROUCH = motion["crouch"]                                 # hips back, pelvis drop, torso lean at c = 1
    # carried block: forearms roughly level, elbows low, block in front of the belly
    sh = np.array([SHOULDER[0], 0.0, WAIST + SHOULDER[2]])
    CARRY_C = sh + np.array([0.46 * Z0, 0.0, -0.35 * Z0])         # scaled by body height (G1: 35 cm out)
    FACE = np.array([0.16 * k, -0.02, WAIST + SHOULDER[2] + 0.18 * k])
    PICK_D, PLACE_D = 0.20 * REACH / 0.478, 0.25 * REACH / 0.478
    # robots that squat with a steep forward lean: stand back so the hands hang under the shoulders instead of
    # reaching back between the knees (elbows up in front of the head)
    st = HState(0.0, 0.0, 0.0)
    apply_crouch(st, 0.9, np.zeros(2), 0.0)
    under = np.mean([sh[0] for sh in shoulders(st)]) - 0.042 + 0.03
    PICK_D = max(PICK_D, under)
    PLACE_D = max(PLACE_D, PICK_D + 0.05)
    CARRY_C = _reachable_carry(key, CARRY_C, reach_c)


_CARRY = {}


def _reachable_carry(key, default, reach):
    """The carry position the arms can really hold (wide-shouldered robots may need the block further out or
    higher): the default first, then nearby alternatives, checked with the arm IK."""
    if key in _CARRY:
        return _CARRY[key]
    import robots
    ik = robots.HumanoidIK(key)
    st = HState(0.0, 0.0, 0.0)
    g = Grasp(yaw_quat(0.0), np.array([HALF, HALF, 0.09]), 0.0)
    best = (np.inf, default)
    for dx in (0.0, -0.08, -0.16, -0.24, 0.06, 0.12):            # default first, then closer to the body
        for dz in (0.0, 0.06, -0.06, 0.12):
            c = default + np.array([dx, 0.0, dz]) * reach / 0.56
            quat = mat2quat(g.rrel)
            pos = st.pelvis + c - quat2mat(quat)[:, 2] * g.u
            palms, frames, _ = hand_frames(pos, quat, g, st)
            st.hands, st.hand_R, st.hand_w, st.held = palms, frames, [1.0, 1.0], None
            ik.solve(st)
            mujoco.mj_kinematics(ik.m, ik.d)
            err = max(np.linalg.norm(ik.d.site_xpos[ik.chains[(sd["s"], "arm")]["site"]] - palms[i])
                      for i, sd in enumerate(PR.SIDES))
            if err < 0.005:
                _CARRY[key] = c
                return c
            if err < best[0]:
                best = (err, c)
    _CARRY[key] = best[1]
    return best[1]


def side(i):
    return 1.0 if i == 0 else -1.0


def lateral(yaw):
    return np.array([-math.sin(yaw), math.cos(yaw)])


class HState:
    """Everything needed to pose the robot at one instant."""

    def __init__(self, cx, cy, yaw):
        self.yaw = yaw
        self.pelvis = np.array([cx, cy, Z0])
        self.lean = 0.0
        self.twist = 0.0
        self.feet = [[np.array([cx, cy, 0.0]) + np.append(side(i) * FOOT_W * lateral(yaw), 0), yaw, 0.0]
                     for i in (0, 1)]
        self.hands = [None, None]       # palm-site targets
        self.hand_R = [None, None]      # hand orientation targets (grasping)
        self.hand_w = [0.0, 0.0]        # blend from the free-arm pose to hand_R / open hand
        self.look = None
        self.held = None        # (block id, pos, quat)
        self.grip = 0.0         # finger closure 0..1

    def free_hands(self):
        self.hands, self.hand_R, self.hand_w, self.held, self.grip = [None, None], [None, None], [0.0, 0.0], None, 0.0

    def center(self):
        return (self.feet[0][0][:2] + self.feet[1][0][:2]) / 2

    def clone(self):
        return copy.deepcopy(self)


def apply_crouch(st, c, center, yaw):
    """Squat: drop the pelvis, push the hips back and lean the torso forward."""
    back, drop, lean = CROUCH
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    st.pelvis = np.array([*(center - back * c * fwd), Z0 - drop * c])
    st.lean = lean * c
    st.yaw = yaw


def torso_frame(st):
    Rp = rot_z(st.yaw)
    Rt = Rp @ rot_z(st.twist) @ rot_y(st.lean)
    waist = st.pelvis + np.array([0, 0, WAIST])
    return Rp, Rt, waist


def shoulders(st):
    _, Rt, waist = torso_frame(st)
    return [waist + Rt @ (SHOULDER * np.array([1, side(i), 1])) for i in (0, 1)]


def relaxed_grips(st):
    Rp, _, _ = torso_frame(st)
    fwd = Rp[:, 0]
    swing = np.dot(st.feet[0][0] - st.feet[1][0], fwd)
    return [sh + Rp @ REST_HAND[i] - fwd * 0.55 * swing * side(i) for i, sh in enumerate(shoulders(st))]


class Grasp:
    """Where the hands hold a block, fixed in the block frame: the two faces of block axis `axis` (x or y),
    `u` above the block centre along its long axis.  `sgn` picks which face is the left hand's.
    `rrel` is the upright carry orientation relative to the pelvis."""

    def __init__(self, quat, half, yaw):
        R0, lat = quat2mat(quat), rot_z(yaw)[:, 1]
        self.axis = int(np.argmax(np.abs(R0[:, :2].T @ lat)))
        self.sgn = 1.0 if R0[:, self.axis] @ lat >= 0 else -1.0
        top = BALL_R + 0.008 if HAND_KIND == "ball" else GRIP_TOP     # ball hands press close to the top
        self.u = max(0.0, half[2] - top)
        self.half = np.asarray(half, float)
        cols = [None, None, UP]
        cols[self.axis] = lat * self.sgn
        if self.axis == 0:
            cols[1] = np.cross(UP, cols[0])
        else:
            cols[0] = np.cross(cols[1], UP)
        self.rrel = rot_z(yaw).T @ np.column_stack(cols)

    def contacts(self, pos, quat):
        """Contact centres (left, right) and the left face's outward normal."""
        R = quat2mat(quat)
        n = R[:, self.axis] * self.sgn
        mid = pos + R[:, 2] * self.u
        return [mid + n * self.half[self.axis], mid - n * self.half[self.axis]], n


def carry_pose(st, grasp):
    """Held block pose: contacts at CARRY_C in front of the belly, upright, orientation grasp.rrel."""
    Rp = rot_z(st.yaw)
    R = Rp @ grasp.rrel
    return st.pelvis + Rp @ CARRY_C - R[:, 2] * grasp.u, mat2quat(R)


def finger_alpha(contact_z):
    """Fingers point down when the hands are near the floor, forward (forearm level) at carry height."""
    z_carry = Z0 + CARRY_C[2]
    s = smooth((contact_z - 0.25 * z_carry) / (0.75 * z_carry))
    return ALPHA_LOW + (ALPHA_CARRY - ALPHA_LOW) * s


def hand_frames(pos, quat, grasp, st, gap=0.0, lift=0.0, alpha=None):
    """Palm-site targets and hand orientations for the grasp on a block at (pos, quat).  The fingers lie in
    the face plane, pointing forward and down by `alpha`.  gap / lift move the hands out of contact.
    Also returns (contact, finger direction, outward direction) per hand for collision checks."""
    cs, n = grasp.contacts(pos, quat)
    fwd = rot_z(st.yaw)[:, 0]
    palms, frames, contacts = [], [], []
    for i, c in enumerate(cs):
        c = c + side(i) * n * gap + UP * lift
        a = alpha if alpha is not None else HAND_ALPHA if HAND_ALPHA is not None else finger_alpha(c[2])
        f = math.cos(a) * fwd - math.sin(a) * UP
        f = unit(f - n * (f @ n))
        if HAND_KIND == "ball":
            palms.append(c + side(i) * n * BALL_R)
            frames.append(None)
        else:
            nh = n
            if HAND_TILT:                  # tip the approach down about the closing axis, more near the floor
                t = HAND_TILT * (finger_alpha(c[2]) - ALPHA_CARRY) / (ALPHA_LOW - ALPHA_CARRY)
                nh = math.cos(t) * n + side(i) * math.sin(t) * np.cross(f, n)
            Rh = np.column_stack([f, nh, np.cross(f, nh)])
            palms.append(c - Rh @ GRASP_OFF[i])
            frames.append(Rh)
        contacts.append((c, f, n * side(i)))
    return palms, frames, contacts


def hand_points(contacts, shoulders_):
    """Sample points filling both hands (forearms for ball-handed robots), for collision checks."""
    pts = []
    for i, (c, f, o) in enumerate(contacts):
        if HAND_KIND == "ball":
            b = c + o * BALL_R
            arm = unit(shoulders_[i] - b)
            for t in np.linspace(0, L_FARM + GRIP - 0.02, 6):
                for dz in (-0.7, 0.0, 0.7):
                    pts.append(b + arm * t + np.cross(arm, o) * dz * BALL_R)
            continue
        z = np.cross(f, o * side(i))
        for x0, x1, o0, o1, z0, z1 in HAND_PARTS:
            for x in np.linspace(x0, x1, 4):
                for oo in np.linspace(o0 + 0.003, o1, 3):
                    for zz in np.linspace(z0, z1, 3):
                        pts.append(c + f * x + o * oo + z * zz)
    return np.array(pts)


def set_hands(st, palms, frames, w=1.0, grip=1.0):
    st.hands, st.hand_R, st.hand_w, st.grip = list(palms), list(frames), [w, w], grip


def hold(st, bid, pos, quat, grasp):
    """Hands on the block; st.held also keeps the planned contact midpoint, so the renderer can place the
    block between the hands exactly where the IK put them."""
    palms, frames, contacts = hand_frames(pos, quat, grasp, st)
    st.held = (bid, pos, quat, (contacts[0][0] + contacts[1][0]) / 2)
    set_hands(st, palms, frames)


def solve_crouch(center, yaw, palms_fn):
    """Smallest squat for which both palm targets are inside the arm workspace."""
    reach = 0.95 * REACH
    st = HState(center[0], center[1], yaw)
    for c in np.linspace(0, 1, 41):
        apply_crouch(st, c, center, yaw)
        if all(np.linalg.norm(g - sh) <= reach for g, sh in zip(palms_fn(st), shoulders(st))):
            return c
    return 1.0


# ----------------------------------------------------------------------------- motion segments
class Segment:
    t0 = t1 = 0.0
    end = None

    def eval(self, t):
        raise NotImplementedError


class Idle(Segment):
    def __init__(self, t0, t1, base, look_fn=None, gestures=()):
        self.t0, self.t1, self.base, self.look_fn, self.gestures = t0, t1, base, look_fn, gestures
        self.end = base

    def eval(self, t):
        st = self.base.clone()
        b = math.sin(2 * math.pi * (t - self.t0) / 3.4)
        st.lean += 0.012 * b
        st.pelvis = st.pelvis + np.array([0, 0, 0.003 * b])
        if self.look_fn is not None:
            st.look = self.look_fn(t)
        for g in self.gestures:
            g(t, st)
        return st


class Facepalm:
    """Right hand to the face, slump, slow head shake (a waist twist, so robots without a neck can do it too)."""

    def __init__(self, t0, hold=1.6):
        self.t0, self.hold = t0, hold

    def __call__(self, t, st):
        u = t - self.t0
        w = smoother(u / 0.45) * (1 - smoother((u - 0.45 - self.hold) / 0.5))
        if w <= 0:
            return
        Rp = rot_z(st.yaw)
        relaxed = relaxed_grips(st)[1]
        face = st.pelvis + Rp @ FACE
        st.hands[1] = relaxed + (face - relaxed) * w
        st.lean += 0.14 * w
        shake = smooth((u - 0.5) / 0.3) * (1 - smooth((u - 0.4 - self.hold) / 0.3))
        st.twist += 0.22 * shake * math.sin(2 * math.pi * 1.6 * (u - 0.5))
        st.look = None



def plan_footsteps(st, path, end_yaw):
    """Alternating footsteps along a 2D path. The robot turns while it walks (blending from its start
    heading to the path direction and on to the requested end heading) instead of stopping to pivot."""
    cur = [[f[0][:2].copy(), f[1]] for f in st.feet]
    center = (cur[0][0] + cur[1][0]) / 2
    yaw0 = st.yaw
    steps = []

    def put(i, p, y):
        dur = STEP_T if np.linalg.norm(p - cur[i][0]) > 0.5 * STRIDE else TURN_T
        steps.append((i, p.copy(), y, dur))
        cur[i] = [p.copy(), y]

    path = np.asarray(path, float)
    path = np.vstack([center, path[1:]]) if len(path) >= 2 else center[None]
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6])
    path = path[keep]
    s_acc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    total = s_acc[-1]

    if total < 0.03:                                    # turn on the spot
        d = wrap(end_yaw - yaw0)
        if abs(d) < 0.05 and all(abs(wrap(c[1] - end_yaw)) < 0.05 for c in cur):
            return steps
        n = max(1, math.ceil(abs(d) / MAX_TURN))
        i = 0 if d > 0 else 1
        for k in range(1, n + 1):
            yk = yaw0 + d * k / n
            put(i, center + side(i) * FOOT_W * lateral(yk), yk)
            i = 1 - i
        put(i, center + side(i) * FOOT_W * lateral(end_yaw), end_yaw)
        return steps

    def at(s):
        return np.array([np.interp(s, s_acc, path[:, 0]), np.interp(s, s_acc, path[:, 1])])

    def heading(s):
        d = at(min(total, s + 0.12)) - at(max(0.0, s - 0.12))
        return math.atan2(d[1], d[0])

    # continuous heading along the path, on the branch nearest the start heading, so a turn of almost
    # 180 degrees keeps one direction instead of flipping from step to step
    hs = np.linspace(0, total, 60)
    head_u = np.unwrap([heading(s) for s in hs])
    head_u += yaw0 + wrap(head_u[0] - yaw0) - head_u[0]
    end_u = head_u[-1] + wrap(end_yaw - head_u[-1])

    # turn into / out of the path heading over a stretch long enough for the robot's turn per step
    stride0 = STRIDE if total >= SHUFFLE else 0.6 * STRIDE
    rate = 0.8 * MAX_TURN / stride0 * (hs[1] - hs[0])          # corners: heading changes no faster than that
    for i in range(1, len(head_u)):
        head_u[i] = np.clip(head_u[i], head_u[i - 1] - rate, head_u[i - 1] + rate)
    end_u = head_u[-1] + wrap(end_yaw - head_u[-1])
    l_in = max(0.45, 2.0 * abs(head_u[0] - yaw0) / MAX_TURN * stride0)
    l_out = max(0.45, 2.0 * abs(end_u - head_u[-1]) / MAX_TURN * stride0)
    if l_in + l_out > total:
        l_in, l_out = l_in * total / (l_in + l_out), l_out * total / (l_in + l_out)

    def yaw_at(s):
        if total < SHUFFLE:
            return yaw0 + wrap(end_yaw - yaw0) * smooth(s / total)
        y = yaw0 + (np.interp(s, hs, head_u) - yaw0) * smooth(s / l_in)
        return end_u + (y - end_u) * smooth((total - s) / l_out)

    samples = np.linspace(0, total, 40)
    turn_total = sum(abs(wrap(yaw_at(b) - yaw_at(a))) for a, b in zip(samples[:-1], samples[1:]))
    stride = STRIDE if total >= SHUFFLE else 0.6 * STRIDE
    n = max(2, math.ceil(total / stride), math.ceil(1.8 * turn_total / MAX_TURN))   # smooth turns peak ~1.5x

    s1 = total / n                                      # lead with the foot on the side we move/turn to
    lead = np.dot(at(s1) - center, lateral(yaw0)) + 0.15 * wrap(yaw_at(s1) - yaw0)
    i = 0 if lead >= 0 else 1
    for k in range(1, n + 1):
        s = total * k / n
        yk = yaw_at(s)
        put(i, at(s) + side(i) * FOOT_W * lateral(yk), yk)
        i = 1 - i
    put(i, at(total) + side(i) * FOOT_W * lateral(end_yaw), end_yaw)
    return steps


def _hermite(times, values, t):
    k = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
    t0, t1 = times[k], times[k + 1]
    h = t1 - t0
    s = np.clip((t - t0) / h, 0, 1)

    def tangent(j):
        if j == 0 or j == len(times) - 1:
            return values[j] * 0
        return (values[j + 1] - values[j - 1]) / (times[j + 1] - times[j - 1])

    h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
    h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * values[k] + h10 * h * tangent(k) + h01 * values[k + 1] + h11 * h * tangent(k + 1)


class Walk(Segment):
    def __init__(self, t0, start, steps, carry=None):
        self.t0, self.start, self.steps, self.carry = t0, start, steps, carry
        self.times = np.concatenate([[t0], t0 + np.cumsum([s[3] for s in steps])])
        self.t1 = self.times[-1]
        cfg = [[f[0][:2].copy(), f[1]] for f in start.feet]
        self.cfgs = [copy.deepcopy(cfg)]
        for i, p, y, _ in steps:
            cfg[i] = [p.copy(), y]
            self.cfgs.append(copy.deepcopy(cfg))
        mids = [(c[0][0] + c[1][0]) / 2 for c in self.cfgs]
        yaws = [math.atan2(math.sin(c[0][1]) + math.sin(c[1][1]), math.cos(c[0][1]) + math.cos(c[1][1]))
                for c in self.cfgs]
        yaws = np.unwrap(yaws)
        self.mid_vals = [np.array([m[0], m[1], y]) for m, y in zip(mids, yaws)]
        self.end = self.eval(self.t1)

    def eval(self, t):
        t = float(np.clip(t, self.t0, self.t1))
        k = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.steps) - 1))
        ta, tb = self.times[k], self.times[k + 1]
        ph = (t - ta) / (tb - ta)
        before, after = self.cfgs[k], self.cfgs[k + 1]
        moving = self.steps[k][0]
        stance = 1 - moving
        s = np.clip((ph - 0.11) / 0.78, 0, 1)

        st = self.start.clone()
        st.lean, st.twist = 0.0, 0.0
        st.free_hands()
        p0, p1 = before[moving][0], after[moving][0]
        dist = np.linalg.norm(p1 - p0)
        lift = (LIFT if dist > 0.35 * STRIDE else 0.6 * LIFT) * math.sin(math.pi * s)
        st.feet[moving] = [np.array([*(p0 + (p1 - p0) * smoother(s)), lift]),
                           lerp_angle(before[moving][1], after[moving][1], smoother(s)),
                           0.18 * math.sin(2 * math.pi * s) * min(1.0, dist / 0.15)]
        st.feet[stance] = [np.array([*before[stance][0], 0.0]), before[stance][1], 0.0]

        mv = _hermite(self.times, self.mid_vals, t)
        env = min(1.0, (t - self.t0) / STEP_T, (self.t1 - t) / STEP_T)
        sway_dir = unit(before[stance][0] - mv[:2])
        xy = mv[:2] + sway_dir * 0.02 * math.sin(math.pi * ph) * env
        z = Z0 - 0.014 * (1 - math.sin(math.pi * s)) * env
        st.yaw = float(mv[2])
        st.pelvis = np.array([xy[0], xy[1], z])
        st.twist = -0.08 * np.dot(st.feet[0][0][:2] - st.feet[1][0][:2],
                                  [math.cos(st.yaw), math.sin(st.yaw)])
        st.look = np.array([*(xy + 1.3 * np.array([math.cos(st.yaw), math.sin(st.yaw)])), 0.0])
        if self.carry is not None:
            bid, grasp = self.carry
            hold(st, bid, *carry_pose(st, grasp), grasp)
        return st


def _lerp_pts(a, b, s):
    return [p + (q - p) * s for p, q in zip(a, b)]


class Pick(Segment):
    """Reach with open hands to just outside two opposite side faces, close in, squeeze, lift.  While lifting,
    the block pivots between the hands into the upright carry pose."""
    T_REACH, T_GRIP, T_LIFT = 0.65, 0.22, 0.7

    def __init__(self, t0, base, bid, pos0, quat0, half):
        self.t0, self.base, self.bid = t0, base, bid
        self.pos0, self.quat0, self.half = np.asarray(pos0), np.asarray(quat0), half
        self.t_grasp = t0 + self.T_REACH + self.T_GRIP
        self.t1 = self.t_grasp + self.T_LIFT
        self.center, self.yaw = base.center(), base.yaw
        self.g = Grasp(self.quat0, half, self.yaw)
        frames = lambda st, **kw: hand_frames(self.pos0, self.quat0, self.g, st, **kw)
        self.c_star = solve_crouch(self.center, self.yaw, lambda st: frames(st)[0])
        st = base.clone()
        apply_crouch(st, self.c_star, self.center, self.yaw)
        self.grasp, self.R, _ = frames(st)
        self.pre = frames(st, gap=0.05, lift=0.04)[0]
        self.relaxed = relaxed_grips(base)
        self.carry_pos, self.carry_q = carry_pose(base, self.g)
        self.end = self.eval(self.t1)

    def eval(self, t):
        u = t - self.t0
        st = self.base.clone()
        st.look = self.pos0.copy()
        if u < self.T_REACH:
            s = smoother(u / self.T_REACH)
            apply_crouch(st, self.c_star * s, self.center, self.yaw)
            set_hands(st, _lerp_pts(self.relaxed, self.pre, s), self.R, w=s, grip=0.0)
        elif u < self.T_REACH + self.T_GRIP:
            v = (u - self.T_REACH) / self.T_GRIP
            apply_crouch(st, self.c_star, self.center, self.yaw)
            set_hands(st, _lerp_pts(self.pre, self.grasp, smoother(v / 0.7)), self.R,
                      grip=smoother((v - 0.4) / 0.6))
        else:
            s = smoother((u - self.T_REACH - self.T_GRIP) / self.T_LIFT)
            apply_crouch(st, self.c_star * (1 - s), self.center, self.yaw)
            pos = self.pos0 + (self.carry_pos - self.pos0) * s + UP * 0.05 * math.sin(math.pi * s)
            q = slerp(self.quat0, self.carry_q, smooth(min(1.0, 1.3 * s)))
            hold(st, self.bid, pos, q, self.g)
            st.look = pos
        return st


class Place(Segment):
    """Lower the block onto its cell, loosen the fingers, then move the hands out and back."""
    T_MOVE, T_SET, T_OPEN, T_BACK = 0.75, 0.3, 0.18, 0.55

    def __init__(self, t0, base, bid, grasp, target_pos, target_yaw):
        self.t0, self.base, self.bid, self.g = t0, base, bid, grasp
        self.target = np.asarray(target_pos)
        self.center, self.yaw = base.center(), base.yaw
        self.carry_pos, self.carry_q = carry_pose(base, grasp)
        # the square block looks the same every 90 deg: settle on the yaw closest to how it is held
        cands = [yaw_quat(target_yaw + k * math.pi / 2) for k in range(4)]
        self.q_t = max(cands, key=lambda q: abs(q @ self.carry_q))
        self.t_release = t0 + self.T_MOVE + self.T_SET
        self.t1 = self.t_release + self.T_OPEN + self.T_BACK
        self.above = self.target + UP * 0.05
        frames = lambda st, **kw: hand_frames(self.target, self.q_t, grasp, st, **kw)
        self.c_star = solve_crouch(self.center, self.yaw, lambda st: frames(st)[0])
        st = base.clone()
        apply_crouch(st, self.c_star, self.center, self.yaw)
        self.grasp, self.R, _ = frames(st)
        self.open = frames(st, gap=0.05, lift=0.03)[0]
        self.end = self.eval(self.t1)

    def eval(self, t):
        u = t - self.t0
        st = self.base.clone()
        st.look = self.target.copy()
        if u < self.T_MOVE + self.T_SET:
            if u < self.T_MOVE:
                s = smoother(u / self.T_MOVE)
                apply_crouch(st, self.c_star * s, self.center, self.yaw)
                pos = self.carry_pos + (self.above - self.carry_pos) * s
                q = slerp(self.carry_q, self.q_t, s)
            else:
                s = smooth((u - self.T_MOVE) / self.T_SET)
                apply_crouch(st, self.c_star, self.center, self.yaw)
                pos, q = self.above + (self.target - self.above) * s, self.q_t
            hold(st, self.bid, pos, q, self.g)
        elif u < self.T_MOVE + self.T_SET + self.T_OPEN:
            v = (u - self.T_MOVE - self.T_SET) / self.T_OPEN
            apply_crouch(st, self.c_star, self.center, self.yaw)
            set_hands(st, _lerp_pts(self.grasp, self.open, smoother((v - 0.35) / 0.65)), self.R,
                      grip=1 - smoother(v / 0.5))
        else:
            s = smoother((u - self.T_MOVE - self.T_SET - self.T_OPEN) / self.T_BACK)
            apply_crouch(st, self.c_star * (1 - s), self.center, self.yaw)
            set_hands(st, _lerp_pts(self.open, relaxed_grips(st), s), self.R, w=1 - s, grip=0.0)
            if s >= 1:
                st.free_hands()
        return st


class Timeline:
    def __init__(self, first):
        self.segs = [first]

    @property
    def t_end(self):
        return self.segs[-1].t1

    @property
    def state(self):
        st = self.segs[-1].end.clone()
        st.free_hands()
        st.look = None
        apply_crouch(st, 0.0, st.center(), st.yaw)
        return st

    def add(self, seg):
        self.segs.append(seg)
        return seg

    def walk(self, path, end_yaw, carry=None):
        self.walk_steps(plan_footsteps(self.state, path, end_yaw), carry)

    def walk_steps(self, steps, carry=None):
        if steps:
            self.add(Walk(self.t_end, self.state, steps, carry))

    def eval(self, t):
        starts = [s.t0 for s in self.segs]
        k = max(0, int(np.searchsorted(starts, t, side="right")) - 1)
        return self.segs[k].eval(min(t, self.segs[k].t1))
