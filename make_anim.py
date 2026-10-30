"""commit-robots: a humanoid robot rebuilds your GitHub contribution calendar after a quadruped crashes
through it.  Physics: MuJoCo.  Output: looping GIF / WebP / MP4.

    python make_anim.py                       # sample data, robots of the day
    python make_anim.py --user <login>        # real data, needs GITHUB_TOKEN
    python make_anim.py --preview             # a few stills only
"""
import argparse
import datetime
import math
import os
import pickle
import shutil
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import contrib
import humanoid as H
import profiles as PR
import quadruped
import robots
import scene
from common import slerp, smooth
from rebuild import Planner, Rebuild

W, HGT, FPS = 960, 420, 20
CAM = dict(lookat=(3.05, 0.48, 0.25), distance=9.3, azimuth=90.0, elevation=-50.0)
WATCH = (5.75, 1.15, -math.pi / 2)            # where the humanoid stands while the quadruped drives
T_QUAD = 0.9
REC_DT = 1 / 120
GAP_SCREEN = (2.0, 5.0)                       # screen seconds for the two time-lapse stretches
CLOCK0 = 14 * 3600 + 2 * 60                   # wall clock shown in the camera overlay (14:02:00)
MAX_BLUR = 8                                  # motion-blur sub-frames in the time-lapse (4 with --fast)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def block_qadr(m, nb):
    return [m.jnt_qposadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"b{i}")]]
            for i in range(nb)]


class Rig:
    """Compiled scene + qpos/qvel slices of both robots + the humanoid's IK solver."""

    def __init__(self, cells, total, texdir, hkey, qkey):
        H.configure(hkey)
        self.m = scene.build_spec(cells, total, W, HGT, texdir, hkey, qkey).compile()
        self.d = mujoco.MjData(self.m)
        self.hum = robots.HumanoidIK(hkey)
        self.h_q, self.h_v = robots.root_slices(self.m, "h_" + PR.HUMANOIDS[hkey]["root"],
                                                self.hum.m.nq, self.hum.m.nv)
        dog = robots.quadruped_spec(qkey).compile()
        self.q_q, self.q_v = robots.root_slices(self.m, "q_" + PR.QUADRUPEDS[qkey]["root"], dog.nq, dog.nv)


# ----------------------------------------------------------------------------- physics
def simulate(cells, total, texdir, hkey, qkey):
    active = [c for c in cells if c.level > 0]
    rig = Rig(cells, total, texdir, hkey, qkey)
    m, d = rig.m, rig.d
    nb = len(active)
    qadr = block_qadr(m, nb)
    vadr = [m.jnt_dofadr[m.body_jntadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"b{i}")]]
            for i in range(nb)]
    quad = quadruped.make(qkey, T_QUAD, WATCH[:2])
    h_idle = rig.hum.solve(H.HState(*WATCH))
    mujoco.mj_forward(m, d)

    log(f"physics: {nb} blocks, {PR.HUMANOIDS[hkey]['label']} + {quad.label}, quadruped {quad.t_start:.1f}s - {quad.t_end:.1f}s, "
        f"backflip lands at {quad.events['land']:.1f}s")
    rec_t, rec_q = [], []
    t, next_rec, calm = 0.0, 0.0, 0.0
    wall = time.time()
    while True:
        # both robots are kinematic: overwrite their state every step (the quadruped's velocity is
        # set too, so contacts push the blocks the way a real 60 kg robot would)
        d.qpos[rig.q_q], d.qvel[rig.q_v] = quad.qpos(t), quad.qvel(t)
        d.qpos[rig.h_q], d.qvel[rig.h_v] = h_idle, 0.0
        mujoco.mj_step(m, d)
        t += m.opt.timestep
        if t >= next_rec - 1e-9:
            rec_t.append(t)
            rec_q.append(np.array([d.qpos[a:a + 7] for a in qadr]))
            next_rec += REC_DT
        if t > quad.t_end + 0.3:
            vmax = max(np.abs(d.qvel[v:v + 6]).max() for v in vadr)
            calm = calm + m.opt.timestep if vmax < 0.08 else 0.0
            if calm > 0.25 or t > quad.t_end + 1.5:
                break
    log(f"physics: {t:.2f}s simulated in {time.time() - wall:.1f}s wall")
    return dict(cells=cells, total=total, rec_t=np.array(rec_t), rec_q=np.array(rec_q), quad=quad,
                hkey=hkey, qkey=qkey)


# ----------------------------------------------------------------------------- humanoid plan
class QuadWatcher:
    """Look target for the idle humanoid: follow the quadruped while it is around."""

    def __init__(self, quad, fwd_pt):
        self.quad, self.fwd_pt = quad, fwd_pt

    def __call__(self, t):
        q = self.quad
        w = smooth((t - q.t_start - 0.4) / 0.8) * (1 - smooth((t - q.t_end + 0.8) / 1.2))
        target = q.position(min(max(t, q.t_start), q.t_end - 0.01))
        return self.fwd_pt + (target - self.fwd_pt) * w


def plan(phys):
    cells, rec_q, rec_t, quad = phys["cells"], phys["rec_q"], phys["rec_t"], phys["quad"]
    H.configure(phys["hkey"])
    active = [c for c in cells if c.level > 0]
    targets = [dict(pos=np.array([*scene.cell_xy(c.col, c.row), scene.HEIGHT[c.level] / 2]),
                    level=c.level, col=c.col) for c in active]
    blocks = [dict(level=c.level, half=np.array([scene.HALF, scene.HALF, scene.HEIGHT[c.level] / 2]),
                   pos=rec_q[-1][i][:3].copy(), quat=rec_q[-1][i][3:].copy() / np.linalg.norm(rec_q[-1][i][3:]))
              for i, c in enumerate(active)]
    t_phys_end = rec_t[-1]
    t_rb = t_phys_end + 0.2
    fwd_pt = np.array([WATCH[0] + 3 * math.cos(WATCH[2]), WATCH[1] + 3 * math.sin(WATCH[2]), 0.85])
    tl = H.Timeline(H.Idle(0.0, t_rb, H.HState(*WATCH), QuadWatcher(quad, fwd_pt),
                           gestures=[H.Facepalm(quad.events["land"] + 0.35)]))
    rb = Rebuild(targets, blocks, tl, log)
    rb.run()
    tl.walk(Planner().path(rb.polys(), tl.state.center(), WATCH[:2]), WATCH[2])
    tl.add(H.Idle(tl.t_end, tl.t_end + 1.7, tl.state))
    picks = [s for s in tl.segs if isinstance(s, H.Pick)]
    places = [s for s in tl.segs if isinstance(s, H.Place)]
    return dict(phys, tl=tl, events=rb.events, t_phys_end=t_phys_end, t_rb=t_rb,
                cycles=[(p.t0, q.t1) for p, q in zip(picks, places)], t_end=tl.t_end)


# ----------------------------------------------------------------------------- time remapping
def _gap_speed(a, b, target):
    """Exponential ramp up / plateau / ramp down so that sim time [a, b] lasts `target` screen seconds."""
    span = b - a
    if span <= target:
        return (lambda t: 1.0), 1.0
    R = min(1.2, target / 3)
    rl = lambda n: R * math.log(n) * n / (n - 1)
    lo, hi = 1.001, 5000.0
    for _ in range(100):
        n = math.sqrt(lo * hi)
        mid = span - 2 * rl(n)
        if mid < 0 or 2 * R + mid / n < target:
            hi = n
        else:
            lo = n
    n, r = lo, rl(lo)

    def speed(t):
        if t < a + r:
            return n ** ((t - a) / r)
        if t > b - r:
            return n ** ((b - t) / r)
        return n

    return speed, n


def time_map(sim):
    cyc = sim["cycles"]
    n_first, n_last = min(2, len(cyc)), min(3, len(cyc))      # placements shown in real time
    windows = [(0.0, sim["t_rb"] + 1.2),
               (cyc[0][0] - 0.8, cyc[n_first - 1][1] + 0.2),
               (cyc[-n_last][0] - 0.8, sim["t_end"])]
    targets = list(GAP_SCREEN)
    if sim["t_end"] - cyc[-1][1] > 7.0:                 # long walk home: fast-forward it too
        windows[2] = (windows[2][0], cyc[-1][1] + 0.6)
        windows.append((sim["t_end"] - 1.8, sim["t_end"]))
        targets.append(1.6)
    gaps = []
    for (_, a1), (b0, _), target in zip(windows[:-1], windows[1:], targets):
        fn, n = _gap_speed(a1, b0, target)
        gaps.append((a1, b0, fn, n))

    def speed(t):
        for a, b, fn, _ in gaps:
            if a < t < b:
                return fn(t)
        return 1.0

    ts = np.arange(0, sim["t_end"] + 1e-9, 0.002)
    v = np.array([speed(t) for t in ts])
    tau = np.concatenate([[0], np.cumsum(0.002 / (0.5 * (v[1:] + v[:-1])))])
    n_frames = int(tau[-1] * FPS)
    frame_t = np.interp(np.arange(n_frames) / FPS, tau, ts)
    log(f"timeline: {sim['t_end']:.0f}s simulated -> {n_frames / FPS:.1f}s animation; real-time windows "
        + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in windows)
        + "; time-lapse " + ", ".join(f"{g[3]:.0f}x" for g in gaps)
        + f"; real time for the first {n_first} and the last {n_last} placements")
    return frame_t, np.array([speed(t) for t in frame_t])


# ----------------------------------------------------------------------------- state at time t
class World:
    def __init__(self, sim, texdir):
        self.sim = sim
        self.rig = Rig(sim["cells"], sim["total"], texdir, sim["hkey"], sim["qkey"])
        self.m, self.d = self.rig.m, self.rig.d
        self.nb = sim["rec_q"].shape[1]
        self.qadr = block_qadr(self.m, self.nb)
        self.ev_t = {i: np.array([e[0] for e in ev]) for i, ev in sim["events"].items()}

    def set_time(self, t):
        s, d = self.sim, self.d
        st = s["tl"].eval(t)
        hum = self.rig.hum
        d.qpos[self.rig.h_q] = hum.solve(st)
        mujoco.mj_kinematics(hum.m, hum.d)
        if t <= s["t_phys_end"]:
            rt = s["rec_t"]
            k = int(np.clip(np.searchsorted(rt, t) - 1, 0, len(rt) - 2))
            f = float(np.clip((t - rt[k]) / (rt[k + 1] - rt[k]), 0, 1))
            qa, qb = s["rec_q"][k], s["rec_q"][k + 1]
            for i, a in enumerate(self.qadr):
                d.qpos[a:a + 3] = qa[i, :3] + (qb[i, :3] - qa[i, :3]) * f
                d.qpos[a + 3:a + 7] = slerp(qa[i, 3:], qb[i, 3:], f)
        else:
            for i, a in enumerate(self.qadr):
                ev = s["events"][i]
                j = int(np.searchsorted(self.ev_t[i], t, side="right")) - 1
                kind, pose = ev[j][1], ev[j][2]
                if kind == "held":
                    if st.held is not None and st.held[0] == i:
                        # the block sits between the hands as solved, not as planned: no gaps, no drift
                        pose = (st.held[1] + hum.d.site_xpos[hum.grasp_sites].mean(0) - st.held[3], st.held[2])
                    else:
                        pose = ev[j - 1][2]
                d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pose
        d.qpos[self.rig.q_q] = s["quad"].qpos(t)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)


def _vignette():
    y, x = np.mgrid[0:HGT, 0:W]
    r = np.hypot((x - W / 2) / (W / 2), (y - HGT / 2) / (HGT / 2)) / math.sqrt(2)
    return (1 - 0.32 * r ** 2.2).astype(np.float32)[..., None]


class Painter:
    def __init__(self, world):
        self.world = world
        self.r = mujoco.Renderer(world.m, HGT, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = CAM["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = CAM["distance"], CAM["azimuth"], CAM["elevation"]
        self.vig = _vignette()

    def shot(self, t):
        self.world.set_time(t)
        self.r.update_scene(self.world.d, camera=self.cam)
        return self.r.render().astype(np.float32)

    def frame(self, t, v):
        span = v / FPS
        if span < 0.12:
            img = self.shot(t)
        else:
            k = int(min(MAX_BLUR, math.ceil(span / 0.3)))
            img = sum(self.shot(t + ((j + 0.5) / k - 0.5) * span) for j in range(k)) / k
        return self.grade(img)

    def grade(self, img):
        x = img / 255.0
        lum = x @ np.array([0.299, 0.587, 0.114], np.float32)
        x = 0.5 + (x - 0.5) * 1.08                                   # a touch of contrast
        x[..., 0] += 0.018 * lum - 0.006                             # warm highlights
        x[..., 2] += 0.022 * (1 - lum) - 0.004                       # cool shadows
        x *= self.vig
        return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


# ----------------------------------------------------------------------------- camera overlay
def mono(size):
    for n in ("consola.ttf", "DejaVuSansMono.ttf"):
        for dd in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", ""):
            try:
                return ImageFont.truetype(os.path.join(dd, n) if dd else n, size)
            except OSError:
                pass
    return ImageFont.load_default(size)


class Overlay:
    """Security-camera style: camera name, date and clock bottom left; a fast-forward badge top right
    during the time-lapse."""

    def __init__(self, date):
        self.f = mono(13)
        self.date = date

    def _text(self, dr, xy, s, alpha=255):
        dr.text((xy[0] + 1, xy[1] + 1), s, font=self.f, fill=(0, 0, 0, int(alpha * 0.55)))
        dr.text(xy, s, font=self.f, fill=(242, 242, 236, alpha))

    def draw(self, img, t, v):
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        c = CLOCK0 + t
        stamp = f"CAM 2   {self.date}  {int(c // 3600) % 24:02d}:{int(c // 60) % 60:02d}:{c % 60:05.2f}"
        self._text(dr, (14, HGT - 24), stamp, 215)
        a = float(smooth((v - 1.3) / 3.0))
        if a > 0.02:
            label = f"{v:5.0f}\u00d7  TIME-LAPSE"
            x = W - 16 - dr.textlength(label, font=self.f)
            for k in range(2):
                ox = x - 22 + k * 8
                dr.polygon([(ox + 1, 15), (ox + 8, 20), (ox + 1, 25)], fill=(0, 0, 0, int(140 * a)))
                dr.polygon([(ox, 14), (ox + 7, 19), (ox, 24)], fill=(242, 242, 236, int(255 * a)))
            self._text(dr, (x, 12), label, int(255 * a))
        return np.asarray(im)


# ----------------------------------------------------------------------------- main
GIF_MAX_MB = 8.0           # GitHub's image proxy does not reliably serve bigger animations
# (width, fps, colours) tried in order until the GIF fits
GIF_STEPS = [(720, 15, 96), (720, 12, 96), (640, 12, 80), (560, 10, 64)]
WEBP_STEPS = [(720, 20, 70), (720, 15, 60), (640, 12, 55)]


def encode(frames_dir, out_base):
    ff = shutil.which("ffmpeg")
    if not ff:
        log("ffmpeg not found: frames are in " + frames_dir)
        return
    src = ["-framerate", str(FPS), "-i", os.path.join(frames_dir, "f%05d.png")]
    run = lambda args: subprocess.run([ff, "-y", "-loglevel", "error", *args], check=True)
    mb = lambda path: os.path.getsize(path) / 1e6
    run(src + ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", out_base + ".mp4"])
    for w, fps, q in WEBP_STEPS:
        run(src + ["-vf", f"fps={fps},scale={w}:-2:flags=lanczos", "-c:v", "libwebp_anim", "-lossless", "0",
                   "-q:v", str(q), "-compression_level", "6", "-loop", "0", out_base + ".webp"])
        if mb(out_base + ".webp") <= GIF_MAX_MB:
            break
    for w, fps, colours in GIF_STEPS:
        run(src + ["-vf", f"fps={fps},scale={w}:-2:flags=lanczos,split[a][b];"
                          f"[a]palettegen=max_colors={colours}:stats_mode=diff[p];"
                          "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                   "-loop", "0", out_base + ".gif"])
        log(f"gif {w}px {fps} fps {colours} colours: {mb(out_base + '.gif'):.1f} MB")
        if mb(out_base + ".gif") <= GIF_MAX_MB:
            break
    for ext in (".gif", ".webp", ".mp4"):
        log(f"wrote {out_base + ext} ({mb(out_base + ext):.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="GitHub login (needs GITHUB_TOKEN); default: sample data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--preview", action="store_true", help="render a handful of stills only")
    ap.add_argument("--reuse", action="store_true", help="reuse the cached physics run")
    ap.add_argument("--reuse-plan", action="store_true", help="also reuse the cached rebuild plan")
    ap.add_argument("--frames-only", action="store_true", help="skip encoding")
    ap.add_argument("--resume", action="store_true", help="keep already rendered frames")
    ap.add_argument("--humanoid", choices=list(PR.HUMANOIDS), help="default: the robot of the day")
    ap.add_argument("--quadruped", choices=list(PR.QUADRUPEDS), help="default: the robot of the day")
    ap.add_argument("--date", type=datetime.date.fromisoformat, default=datetime.date.today(),
                    help="YYYY-MM-DD: use the robots of that day (default: today)")
    ap.add_argument("--stage", choices=["all", "plan"], default="all", help="plan: stop after physics + plan")
    ap.add_argument("--part", default="0/1", help="i/n: render every n-th frame starting at i (parallel jobs)")
    ap.add_argument("--encode-only", action="store_true", help="encode the frames already in <out>/frames")
    ap.add_argument("--fast", action="store_true", help="lighter shadows, anti-aliasing and motion blur")
    ap.add_argument("--sample-seed", type=int, default=11, help=argparse.SUPPRESS)       # testing
    ap.add_argument("--sample-density", type=float, default=1.0, help=argparse.SUPPRESS)
    args = ap.parse_args()
    fdir = os.path.join(args.out, "frames")
    if args.encode_only:
        encode(fdir, os.path.join(args.out, "commit-robots"))
        return
    if args.fast:
        global MAX_BLUR
        scene.QUALITY.update(shadow=4096, samples=4)
        MAX_BLUR = 4
    os.makedirs(args.out, exist_ok=True)
    texdir = os.path.join(args.out, "textures")
    cache = os.path.join(args.out, "physics.pkl")

    phys = None
    if args.reuse and os.path.exists(cache):
        with open(cache, "rb") as f:
            phys = pickle.load(f)
    # robots of the day, unless given, or taken from the cache (parallel jobs must agree across midnight)
    hkey, qkey = PR.pair_for(args.date)
    if phys is not None and not (args.humanoid or args.quadruped):
        hkey, qkey = phys["hkey"], phys["qkey"]
    hkey, qkey = args.humanoid or hkey, args.quadruped or qkey
    log(f"robots: {PR.HUMANOIDS[hkey]['label']} and {PR.QUADRUPEDS[qkey]['label']}")
    if phys is not None:
        if (phys.get("hkey"), phys.get("qkey")) != (hkey, qkey):
            log("cached physics is for other robots: simulating again")
            phys, args.reuse_plan = None, False
    if phys is None:
        if args.user:
            cells, total = contrib.fetch_calendar(args.user, os.environ["GITHUB_TOKEN"])
        else:
            cells, total = contrib.sample_calendar(seed=args.sample_seed, density=args.sample_density)
        phys = simulate(cells, total, texdir, hkey, qkey)
        phys["date"] = datetime.date.today().isoformat()
        with open(cache, "wb") as f:
            pickle.dump(phys, f)
    plan_cache = os.path.join(args.out, "sim.pkl")
    if args.reuse_plan and os.path.exists(plan_cache):
        with open(plan_cache, "rb") as f:
            sim = pickle.load(f)
    else:
        sim = plan(phys)
        with open(plan_cache, "wb") as f:
            pickle.dump(sim, f)
    frame_t, frame_v = time_map(sim)
    if args.stage == "plan":
        return

    world = World(sim, texdir)
    painter, overlay = Painter(world), Overlay(sim.get("date", datetime.date.today().isoformat()))
    if args.preview:
        q = sim["quad"]
        picks = {"start": 0.0, "crash": q.t_start + 2.6, "spin": q.events["takeoff"] - 0.5,
                 "flip": q.events["takeoff"] + 0.35, "bow": q.events["bow"] + 0.8,
                 "debris": sim["t_rb"], "first": sim["cycles"][0][0] + 1.0,
                 "lapse": sim["cycles"][len(sim["cycles"]) // 2][0], "last": sim["cycles"][-1][0] + 1.2,
                 "end": sim["t_end"] - 0.1}
        for name, t in picks.items():
            k = int(np.clip(np.searchsorted(frame_t, t), 0, len(frame_t) - 1))
            img = overlay.draw(painter.frame(frame_t[k], frame_v[k]), frame_t[k], frame_v[k])
            Image.fromarray(img).save(os.path.join(args.out, f"preview-{name}.png"))
        log("previews written")
        return
    part, parts = map(int, args.part.split("/"))
    if not args.resume and parts == 1:
        shutil.rmtree(fdir, ignore_errors=True)
    os.makedirs(fdir, exist_ok=True)
    wall = time.time()
    for k, (t, v) in enumerate(zip(frame_t, frame_v)):
        if k % parts != part:
            continue
        path = os.path.join(fdir, f"f{k:05d}.png")
        if args.resume and os.path.exists(path):
            continue
        img = overlay.draw(painter.frame(t, v), t, v)
        Image.fromarray(img).save(path, compress_level=1)
        if k % 100 == 0:
            log(f"frame {k}/{len(frame_t)}")
    log(f"rendered part {part}/{parts} of {len(frame_t)} frames in {time.time() - wall:.0f}s")
    if not args.frames_only and parts == 1:
        encode(fdir, os.path.join(args.out, "commit-robots"))


if __name__ == "__main__":
    main()
