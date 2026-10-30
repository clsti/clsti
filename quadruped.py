"""Quadruped choreography, sampled at 240 Hz and solved to joint angles with IK.

Wheeled (Go2-W, B2-W, M20, ANYmal on wheels): drive through the calendar, brake, spin a donut, backflip,
play-bow at the humanoid, drive off.  Walking (Spot, A2, X30, ANYmal): trot through the calendar, hop round
180 degrees, dance, play-bow, trot off.  The walking gait places footholds from the body trajectory (stance feet never slide)."""
import math

import mujoco
import numpy as np

import profiles as PR
from common import mat2quat, rot_x, rot_y, rot_z, slerp, smooth, smoother, wrap
from robots import QuadrupedIK, quadruped_legs, quadruped_nominal

HZ = 240
DT = 1.0 / HZ
G = 9.81
PARK = np.array([-40.0, -40.0])
RIDE, WALK, DANCE = 0, 1, 2          # per-sample foot modes (walking robots)


def catmull_rom(points, samples=120):
    p = np.asarray(points, float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for s in np.linspace(0, 1, samples, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3))
    out.append(p[-2])
    return np.array(out)


def make(key, t_start, humanoid_xy):
    return (Walker if PR.QUADRUPEDS[key]["wheel"] is None else Quadruped)(key, t_start, humanoid_xy)


class Quadruped:
    def __init__(self, key, t_start, humanoid_xy):
        self.key, self.t_start = key, t_start
        Q = PR.QUADRUPEDS[key]
        self.label = Q["label"]
        self.ride_z, (sx, sy), r = quadruped_nominal(key)
        self.stance = [(sx, sy), (sx, -sy), (-sx, sy), (-sx, -sy)]     # FL FR HL HR foot / wheel offsets
        self.wheel_r = r if Q["wheel"] else 0.0
        self.S = []                       # samples: x y yaw z roll pitch air tuck mode
        self.events = {}
        self._build(np.asarray(humanoid_xy, float))
        self.events = {k: t_start + v for k, v in self.events.items()}
        S = np.array(self.S)
        self.t = t_start + np.arange(len(S)) * DT
        self.t_end = self.t[-1]
        self.xy, self.yaw, self.z = S[:, :2], S[:, 2], S[:, 3]
        self.roll, self.pitch, self.air, self.tuck, self.mode = S[:, 4], S[:, 5], S[:, 6], S[:, 7], S[:, 8]
        self._solve()

    # ------------------------------------------------------------------ choreography
    mode_default = RIDE

    def push(self, x, y, yaw, z=None, roll=0.0, pitch=0.0, air=0.0, tuck=0.0, mode=None):
        self.S.append([x, y, yaw, self.ride_z if z is None else z, roll, pitch, air, tuck,
                       self.mode_default if mode is None else mode])

    @property
    def last(self):
        return self.S[-1]

    def drive(self, pts, v0, v1, v_max=2.3, a_lat=2.6, a_lon=3.5, lean=1.0):
        path = catmull_rom(pts)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s_raw = np.concatenate([[0], np.cumsum(seg)])
        s = np.linspace(0, s_raw[-1], max(2, int(s_raw[-1] / 0.004)))
        xy = np.column_stack([np.interp(s, s_raw, path[:, 0]), np.interp(s, s_raw, path[:, 1])])
        d = np.gradient(xy, s, axis=0)
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        if self.S:
            heading += round((self.last[2] - heading[0]) / (2 * math.pi)) * 2 * math.pi
        kappa = np.convolve(np.gradient(heading, s), np.ones(15) / 15, mode="same")
        v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)))
        v[0], v[-1] = min(v[0], v0), min(v[-1], v1)
        ds = s[1] - s[0]
        for i in range(1, len(v)):
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_lon * ds))
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_lon * ds))
        pos, vel = 0.0, max(v0, 0.05)
        while pos < s[-1]:
            target = max(np.interp(pos, s, v), 0.05)
            acc = (target - vel) / DT
            vel = target
            k = np.interp(pos, s, kappa)
            self.push(np.interp(pos, s, xy[:, 0]), np.interp(pos, s, xy[:, 1]), np.interp(pos, s, heading),
                      roll=float(np.clip(-0.035 * lean * vel * vel * k, -0.09, 0.09)),
                      pitch=float(np.clip(-0.022 * lean * acc, -0.08, 0.08)))
            pos += vel * DT
        return self

    def hold(self, dur, fn=None, mode=RIDE):
        x, y, yaw, z, roll, pitch, *_ = self.last
        n = int(dur * HZ)
        for i in range(n):
            u = i / HZ
            dz, dr, dp = fn(u) if fn else (0, 0, 0)
            decay = math.exp(-6 * u)
            self.push(x, y, yaw, self.ride_z + dz, roll * decay + dr, pitch * decay + dp, mode=mode)

    def turn(self, yaw_to, dur, extra=0.0):
        x, y, yaw, *_ = self.last
        yaw_to = yaw + wrap(yaw_to - yaw) + extra
        n = int(dur * HZ)
        for i in range(1, n + 1):
            s = smoother(i / n)
            self.push(x, y, yaw + (yaw_to - yaw) * s, roll=0.04 * math.sin(math.pi * s) * np.sign(yaw_to - yaw))

    def backflip(self, flight=0.78, drift=0.22):
        x, y, yaw, *_ = self.last
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        z0 = self.ride_z
        z_top = z0 + 0.4 * z0
        for i in range(int(0.22 * HZ)):                         # load the legs
            s = smoother(i / (0.22 * HZ))
            self.push(x, y, yaw, z0 - 0.17 * z0 * s, pitch=0.06 * s)
        for i in range(int(0.14 * HZ)):                         # explosive push, nose comes up
            s = smooth(i / (0.14 * HZ))
            self.push(x, y, yaw, z0 * 0.83 + (z_top - z0 * 0.83) * s, pitch=0.06 - 0.41 * s)
        self.events["takeoff"] = len(self.S) * DT
        v0 = G * flight / 2
        n = int(flight * HZ)
        for i in range(n + 1):
            tau = i / HZ
            s = tau / flight
            p = x + -fwd[0] * drift * s, y + -fwd[1] * drift * s
            spin = s + 0.12 * math.sin(2 * math.pi * s) / (2 * math.pi)   # tuck speeds up the middle
            air = min(1.0, tau / 0.05, (flight - tau) / 0.05)
            self.push(p[0], p[1], yaw, z_top + v0 * tau - 0.5 * G * tau * tau,
                      pitch=-0.35 - (2 * math.pi - 0.35) * spin, air=max(air, 0.0),
                      tuck=math.sin(math.pi * s))
        self.events["land"] = len(self.S) * DT
        xl, yl = self.last[0], self.last[1]
        for i in range(int(0.55 * HZ)):                         # absorb the landing
            u = i / HZ
            self.push(xl, yl, yaw, z0 + (z_top - z0) * math.exp(-9 * u) * math.cos(11 * u),
                      pitch=0.07 * math.exp(-6 * u) * math.sin(13 * u))

    def bow(self, hxy, dur=1.5):
        x, y = self.last[0], self.last[1]
        self.turn(math.atan2(hxy[1] - y, hxy[0] - x), 0.6)      # look at the humanoid
        self.events["bow"] = len(self.S) * DT
        z = self.ride_z

        def fn(u):
            e = smooth(u / 0.3) * (1 - smooth((u - dur + 0.3) / 0.3))
            return -0.1 * z * e, 0.10 * e * math.sin(2 * math.pi * 3.2 * u), 0.22 * e

        self.hold(dur, fn)

    def _build(self, hxy):
        # slalom through the calendar, carve a U-turn into the open floor in front of it
        weave = [(x, 0.06 if k % 2 == 0 else 0.54) for k, x in enumerate(np.arange(0.2, 4.8, 0.95))]
        self.drive([(-2.0, 0.30), (-0.7, 0.30), *weave, (5.0, 0.30), (5.65, 0.05), (5.6, -0.55),
                    (5.1, -0.80), (4.65, -0.82)], v0=2.3, v1=0.0, a_lat=5.0)
        self.hold(0.3)
        x, y, yaw, *_ = self.last
        self.turn(yaw, 1.1, extra=2 * math.pi)                  # a full skid-steer donut
        self.hold(0.25)
        self.backflip()
        self.bow(hxy)
        x, y = self.last[0], self.last[1]
        exit_pts = [(x, y), (3.6, -0.9), (0.5, -0.92), (-2.3, -0.92)]
        self.turn(math.atan2(exit_pts[1][1] - y, exit_pts[1][0] - x), 0.55)
        self.drive(exit_pts, v0=0.0, v1=2.3, v_max=2.4)

    # ------------------------------------------------------------------ joint-space tables
    def _body(self, k):
        x, y, yaw, z = self.xy[k, 0], self.xy[k, 1], self.yaw[k], self.z[k]
        return np.array([x, y, z]), rot_z(yaw) @ rot_y(self.pitch[k]) @ rot_x(self.roll[k])

    def _targets(self):
        """Wheel centres per sample: on the ground under the stance, or tucked to the body in flight."""
        h = self.ride_z - self.wheel_r
        out = []
        for k in range(len(self.t)):
            p, Rb = self._body(k)
            Ry = rot_z(self.yaw[k])
            a, tuck = self.air[k], self.tuck[k]
            wk = []
            for wx, wy in self.stance:
                ground = np.array([p[0], p[1], 0]) + Ry @ np.array([wx, wy, 0]) + np.array([0, 0, self.wheel_r])
                body = p + Rb @ np.array([wx * 0.9, wy * 0.9, -1.6 * h + 0.8 * h * tuck])
                wk.append(ground * (1 - a) + body * a)
            out.append(wk)
        return out

    def _solve(self):
        ik = QuadrupedIK(self.key)
        n = len(self.t)
        targets = self._targets()
        spins, rate, prev = np.zeros(4), np.zeros(4), None
        q = []
        for k in range(n):
            p, Rb = self._body(k)
            wheels = targets[k]
            if self.wheel_r and prev is not None:
                fwd = np.array([math.cos(self.yaw[k]), math.sin(self.yaw[k]), 0])
                for j in range(4):
                    if self.air[k] < 0.5:
                        rate[j] = np.dot(wheels[j] - prev[j], fwd) / self.wheel_r
                    else:
                        rate[j] *= 0.985
                    spins[j] += rate[j]
            prev = wheels
            q.append(ik.solve(p, mat2quat(Rb), wheels, spins))
        self.q = np.array(q)
        self.qv = np.zeros((n, ik.m.nv))
        for k in range(n - 1):
            mujoco.mj_differentiatePos(ik.m, self.qv[k], DT, self.q[k], self.q[k + 1])
        self.park = self.q[0].copy()
        self.park[0:2] = PARK

    def active(self, t):
        return self.t_start <= t <= self.t_end

    def _k(self, t):
        f = (t - self.t_start) * HZ
        k = int(np.clip(math.floor(f), 0, len(self.t) - 2))
        return k, float(np.clip(f - k, 0, 1))

    def qpos(self, t):
        if not self.active(t):
            return self.park.copy()
        k, f = self._k(t)
        a, b = self.q[k], self.q[k + 1]
        out = a + (b - a) * f
        out[3:7] = slerp(a[3:7], b[3:7], f)
        return out

    def qvel(self, t):
        if not self.active(t):
            return np.zeros(self.qv.shape[1])
        return self.qv[self._k(t)[0]]

    def position(self, t):
        if not self.active(t):
            return np.array([*PARK, self.ride_z])
        k, f = self._k(t)
        return np.array([*(self.xy[k] + (self.xy[k + 1] - self.xy[k]) * f), self.z[k]])


class Walker(Quadruped):
    """Walking quadrupeds: trotting gait instead of wheels."""
    mode_default = WALK
    DUTY = 0.55
    PHASE = (0.0, 0.5, 0.5, 0.0)           # diagonal pairs FL+HR / FR+HL

    def hop_turn(self, flight=0.42):
        """Crouch, jump, turn round 180 degrees in the air, land."""
        x, y, yaw, *_ = self.last
        z0 = self.ride_z
        for i in range(int(0.25 * HZ)):
            s = smoother(i / (0.25 * HZ))
            self.push(x, y, yaw, z0 - 0.12 * s, pitch=0.05 * s, mode=RIDE)
        for i in range(int(0.12 * HZ)):
            s = smooth(i / (0.12 * HZ))
            self.push(x, y, yaw, z0 - 0.12 + 0.2 * s, pitch=0.05 - 0.05 * s, mode=RIDE)
        self.events["takeoff"] = len(self.S) * DT
        zt, v0 = z0 + 0.08, G * flight / 2
        n = int(flight * HZ)
        for i in range(n + 1):
            tau = i / HZ
            s = smoother(tau / flight)
            air = min(1.0, tau / 0.04, (flight - tau) / 0.04)
            self.push(x, y, yaw + math.pi * s, zt + v0 * tau - 0.5 * G * tau * tau, air=max(air, 0.0),
                      tuck=math.sin(math.pi * tau / flight), mode=RIDE)
        self.events["land"] = len(self.S) * DT
        for i in range(int(0.45 * HZ)):
            u = i / HZ
            self.push(x, y, yaw + math.pi, z0 + 0.08 * math.exp(-10 * u) * math.cos(12 * u) - 0.06 * math.exp(-8 * u),
                      mode=RIDE)

    def dance(self, dur=2.2):
        """Stepping in place to a beat, body bouncing and swaying."""

        def fn(u):
            e = smooth(u / 0.3) * (1 - smooth((u - dur + 0.3) / 0.3))
            beat = 2 * math.pi * 2.2 * u
            return -0.035 * e * (1 + math.cos(2 * beat)) / 2, 0.08 * e * math.sin(beat), 0.05 * e * math.sin(2 * beat)

        self.hold(dur, fn, mode=DANCE)

    def _build(self, hxy):
        weave = [(x, 0.06 if k % 2 == 0 else 0.54) for k, x in enumerate(np.arange(0.2, 4.8, 0.95))]
        self.drive([(-1.4, 0.30), (-0.6, 0.30), *weave, (5.0, 0.30), (5.55, 0.05), (5.5, -0.5),
                    (5.1, -0.75), (4.7, -0.78)], v0=1.2, v1=0.0, v_max=1.5, a_lat=1.6, a_lon=1.4, lean=0.5)
        self.hold(0.4, mode=WALK)
        self.hop_turn()
        self.hold(0.2, mode=RIDE)
        self.dance()
        self.bow(hxy)
        x, y = self.last[0], self.last[1]
        exit_pts = [(x, y), (3.6, -0.9), (0.5, -0.92), (-2.0, -0.92)]
        self.turn(math.atan2(exit_pts[1][1] - y, exit_pts[1][0] - x), 0.8)
        self.drive(exit_pts, v0=0.0, v1=1.5, v_max=1.6, a_lon=1.4, lean=0.5)

    def turn(self, yaw_to, dur, extra=0.0):
        super().turn(yaw_to, max(dur, abs(wrap(yaw_to - self.last[2]) + extra) / 1.6))

    def _targets(self):
        """Footholds from a trot: each foot lifts off on its gait phase and lands where its hip will be
        half a stance later, so feet never slide.  Standing still, a foot only steps if it is out of place."""
        n = len(self.t)
        r = quadruped_legs(self.key)[0]["radius"]
        feet = []
        cur = None
        phase = 0.45                           # the second diagonal pair lifts off right away
        swing = [None] * 4                     # (t0, t1, start, target, height) per leg
        for k in range(n):
            p, Rb = self._body(k)
            Ry = rot_z(self.yaw[k])
            nominal = [np.array([p[0], p[1], 0.0]) + Ry @ np.array([sx, sy, 0.0]) for sx, sy in self.stance]
            if cur is None:                    # entering at speed: each foot planted where its hip is halfway
                cur = []                       # to the foot's first lift-off
                for j, (sx, sy) in enumerate(self.stance):
                    k_mid = min(n - 1, int(0.5 * (1 - (phase + self.PHASE[j]) % 1.0) * 0.45 * HZ))
                    pm, _ = self._body(k_mid)
                    cur.append(np.array([pm[0], pm[1], 0.0]) + rot_z(self.yaw[k_mid]) @ np.array([sx, sy, 0.0]))
            v = np.linalg.norm(self.xy[min(k + 1, n - 1)] - self.xy[k]) * HZ
            mode = self.mode[k]
            T = float(np.clip(0.62 - 0.14 * v, 0.40, 0.62)) if mode != DANCE else 1 / 2.2
            phase += DT / T
            t = k * DT
            out = []
            for j in range(4):
                if self.air[k] > 0.0:          # airborne: feet tucked under the body
                    body = p + Rb @ np.array([self.stance[j][0] * 0.9, self.stance[j][1] * 0.9, -0.3 + 0.12 * self.tuck[k]])
                    ground = cur[j]
                    cur[j] = nominal[j].copy() if self.air[k] > 0.5 else cur[j]
                    swing[j] = None
                    out.append(ground * (1 - self.air[k]) + body * self.air[k] + np.array([0, 0, r]) * (1 - self.air[k]))
                    continue
                ph = (phase + self.PHASE[j]) % 1.0
                prev_ph = (phase - DT / T + self.PHASE[j]) % 1.0
                if swing[j] is None and mode != RIDE and ph < prev_ph:          # lift-off instant
                    t_sw = (1 - self.DUTY) * T
                    k_mid = min(n - 1, k + int((t_sw + 0.5 * self.DUTY * T) * HZ))
                    pm, _ = self._body(k_mid)
                    target = np.array([pm[0], pm[1], 0.0]) + rot_z(self.yaw[k_mid]) @ np.array([*self.stance[j], 0.0])
                    if mode == DANCE or np.linalg.norm(target - cur[j]) > 0.02:
                        swing[j] = (t, t + t_sw, cur[j].copy(), target, 0.07 if mode == DANCE else 0.11)
                if swing[j] is not None:
                    t0, t1, a, b, hgt = swing[j]
                    s = min(1.0, (t - t0) / (t1 - t0))
                    f = a + (b - a) * smoother(s)
                    f[2] = hgt * math.sin(math.pi * s)
                    if s >= 1.0:
                        cur[j], swing[j] = b.copy(), None
                    out.append(f + np.array([0, 0, r]))
                else:
                    out.append(cur[j] + np.array([0, 0, r]))
            feet.append(out)
        return feet
