"""Task + motion planning for the humanoid that rebuilds the contribution grid from the debris."""
import math

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

import humanoid as H
from common import quat2mat, quat_yaw, rot_z, wrap, yaw_quat
from scene import HALF

RES = 0.04
X0, Y0, X1, Y1 = -4.0, -3.5, 10.0, 4.5
GRID_BAND = (-0.15, 5.35, -0.15, 0.75)       # x0, x1, y0, y1 around the calendar


# ------------------------------------------------------------------ 2D geometry
def hull2d(pts):
    pts = sorted(map(tuple, np.round(pts, 6)))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])          # counter-clockwise


def box_footprint(pos, quat, half):
    R = quat2mat(quat)
    corners = [pos + R @ (np.array([sx, sy, sz]) * half)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return hull2d(np.array(corners)[:, :2])


def rect(cx, cy, yaw, hx, hy):
    R = rot_z(yaw)[:2, :2]
    return np.array([[cx, cy] + R @ [sx * hx, sy * hy] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def overlap(P, Q, margin=0.0):
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            n /= np.linalg.norm(n) + 1e-12
            a, b = P @ n, Q @ n
            if a.max() + margin < b.min() or b.max() + margin < a.min():
                return False
    return True


def feet_polys(center, yaw):
    lat = H.lateral(yaw)
    return [rect(*(center + s * H.FOOT_W * lat), yaw, *H.FOOT_HALF) for s in (1, -1)]


# ------------------------------------------------------------------ grid path planning
class Planner:
    def __init__(self):
        self.nx = int((X1 - X0) / RES)
        self.ny = int((Y1 - Y0) / RES)
        xs = X0 + (np.arange(self.nx) + 0.5) * RES
        ys = Y0 + (np.arange(self.ny) + 0.5) * RES
        self.gx, self.gy = np.meshgrid(xs, ys, indexing="ij")

    def idx(self, p):
        return (int(np.clip((p[0] - X0) / RES, 0, self.nx - 1)),
                int(np.clip((p[1] - Y0) / RES, 0, self.ny - 1)))

    def raster(self, polys, r):
        blocked = np.zeros((self.nx, self.ny), bool)
        for poly in polys:
            lo, hi = poly.min(0) - r, poly.max(0) + r
            i0, j0 = self.idx(lo)
            i1, j1 = self.idx(hi)
            px = self.gx[i0:i1 + 1, j0:j1 + 1]
            py = self.gy[i0:i1 + 1, j0:j1 + 1]
            inside = np.ones(px.shape, bool)
            dmin = np.full(px.shape, np.inf)
            for k in range(len(poly)):
                a, b = poly[k], poly[(k + 1) % len(poly)]
                e = b - a
                inside &= (e[0] * (py - a[1]) - e[1] * (px - a[0])) >= 0
                t = np.clip(((px - a[0]) * e[0] + (py - a[1]) * e[1]) / (e @ e + 1e-12), 0, 1)
                dmin = np.minimum(dmin, np.hypot(px - a[0] - t * e[0], py - a[1] - t * e[1]))
            blocked[i0:i1 + 1, j0:j1 + 1] |= inside | (dmin <= r)
        return blocked

    def cost_field(self, polys, win):
        """Walking cost per grid cell of the window: 1 where the whole body (both feet) stays clear of the
        blocks with a margin, rising through the margin, and steep where a foot would touch a block."""
        (i0, i1), (j0, j1) = win
        occ = self.raster(polys, 0.0)[i0:i1, j0:j1]
        d = ndimage.distance_transform_edt(~occ) * RES - 0.5 * RES      # cell centre to the nearest block
        body = H.FOOT_W + H.FOOT_HALF[1]
        margin = 0.06
        c = np.ones(d.shape)
        mid = (d < body + margin) & (d >= body)
        c[mid] = 1 + 8 * (body + margin - d[mid]) / margin
        c[d < body] = 60 + 300 * (body - d[d < body]) / body
        return c

    def path(self, polys, a, b):
        """Shortest path weighted by the cost field: keeps the body clear of the blocks where it can and
        otherwise squeezes through where the least is in the way."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        (ia, ja), (ib, jb) = self.idx(a), self.idx(b)
        pad = int(1.5 / RES)
        win = ((max(0, min(ia, ib) - pad), min(self.nx, max(ia, ib) + pad + 1)),
               (max(0, min(ja, jb) - pad), min(self.ny, max(ja, jb) + pad + 1)))
        (i0, i1), (j0, j1) = win
        near = [p for p in polys if (p.max(0) >= (X0 + i0 * RES - 0.5, Y0 + j0 * RES - 0.5)).all()
                and (p.min(0) <= (X0 + i1 * RES + 0.5, Y0 + j1 * RES + 0.5)).all()]
        c = self.cost_field(near, win)
        lo, hi = np.array([X0 + i0 * RES, Y0 + j0 * RES]), np.array([i1 - i0 - 1, j1 - j0 - 1])

        def cell(p):
            k = np.clip(((np.asarray(p) - lo) / RES).astype(int), 0, hi)
            return tuple(k.T)

        def seg_cost(p, q):                       # worst cell on the straight segment p-q
            n = max(2, int(np.linalg.norm(q - p) / (RES * 0.5)))
            return c[cell(p + np.outer(np.linspace(0, 1, n), q - p))].max()

        ends = max(c[cell(a)], c[cell(b)])
        if seg_cost(a, b) <= max(1.0, ends):
            return [a, b]
        nx, ny = c.shape
        ids = np.arange(nx * ny).reshape(nx, ny)
        rows, cols, w = [], [], []
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            sa = ids[max(0, -dx):nx - max(0, dx), max(0, -dy):ny - max(0, dy)]
            sb = ids[max(0, dx):nx - max(0, -dx) or None, max(0, dy):ny - max(0, -dy) or None]
            ca = c[max(0, -dx):nx - max(0, dx), max(0, -dy):ny - max(0, dy)]
            cb = c[max(0, dx):nx - max(0, -dx) or None, max(0, dy):ny - max(0, -dy) or None]
            ww = (0.5 * (ca + cb) * math.hypot(dx, dy)).ravel()
            rows += [sa.ravel(), sb.ravel()]
            cols += [sb.ravel(), sa.ravel()]
            w += [ww, ww]
        g = sparse.csr_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
        s, t = ids[cell(a)], ids[cell(b)]
        _, pred = csgraph.dijkstra(g, indices=s, return_predecessors=True)
        raw, n = [], t
        while n != s and n >= 0:
            raw.append(n)
            n = pred[n]
        raw = [np.array([X0 + (i0 + k // ny + 0.5) * RES, Y0 + (j0 + k % ny + 0.5) * RES]) for k in raw[::-1][:-1]]
        raw = [a] + raw + [b]
        worst = c[cell(np.array(raw))]
        out, i = [raw[0]], 0
        while i < len(raw) - 1:                      # string pulling, never through costlier cells
            j = len(raw) - 1
            while j > i + 1 and seg_cost(raw[i], raw[j]) > max(1.0, max(worst[i:j + 1])):
                j -= 1
            out.append(raw[j])
            i = j
        return out


# ------------------------------------------------------------------ the rebuild task
class Rebuild:
    def __init__(self, targets, blocks, timeline, log=print):
        """targets: list of dict(pos, level, col); blocks: list of dict(level, half, pos, quat)."""
        self.targets = targets
        self.blocks = blocks
        self.tl = timeline
        self.log = log
        self.planner = Planner()
        self.events = {i: [] for i in range(len(blocks))}
        self.filled = [False] * len(targets)
        self.debris = set(range(len(blocks)))
        self.reserved = set()
        self.walk_hits = self.walk_steps = 0
        self.strict = False
        self.squares = [rect(t["pos"][0], t["pos"][1], 0, HALF - 0.002, HALF - 0.002) for t in targets]
        for i, b in enumerate(blocks):
            b["poly"] = box_footprint(b["pos"], b["quat"], b["half"])
            self.events[i].append((-np.inf, "static", (b["pos"].copy(), b["quat"].copy())))
        # blocks that survived the crash in (almost exactly) their own cell
        for i, b in enumerate(blocks):
            t = targets[i]
            tilt = math.acos(np.clip(quat2mat(b["quat"])[2, 2], -1, 1))
            if np.linalg.norm(b["pos"][:2] - t["pos"][:2]) < 0.008 and tilt < 0.06:
                self._settle(i, i)
                snap = (t["pos"].copy(), yaw_quat(round(quat_yaw(b["quat"]) / (math.pi / 2)) * math.pi / 2))
                self.events[i] = [(-np.inf, "static", snap)]
        self.log(f"rebuild: {len(self.debris)} blocks displaced, {len(blocks) - len(self.debris)} survived")

    # -- helpers
    def _settle(self, bid, cell):
        self.filled[cell] = True
        self.debris.discard(bid)
        b = self.blocks[bid]
        b["pos"] = self.targets[cell]["pos"].copy()
        b["quat"] = yaw_quat(0.0)
        b["poly"] = rect(b["pos"][0], b["pos"][1], 0, HALF, HALF)

    def polys(self, exclude=()):
        return [b["poly"] for i, b in enumerate(self.blocks) if i not in exclude and b.get("poly") is not None]

    def covered(self, bid):
        b = self.blocks[bid]
        for j in self.debris:
            if j != bid:
                o = self.blocks[j]
                if o["pos"][2] > b["pos"][2] + 0.03 and overlap(o["poly"], b["poly"]):
                    return True
        return False

    def cover_of(self, bid):
        b = self.blocks[bid]
        tops = [j for j in self.debris if j != bid and self.blocks[j]["pos"][2] > b["pos"][2] + 0.03
                and overlap(self.blocks[j]["poly"], b["poly"])]
        return max(tops, key=lambda j: self.blocks[j]["pos"][2])

    def obstructors(self, cell):
        return [j for j in self.debris if overlap(self.blocks[j]["poly"], self.squares[cell], 0.003)]

    def hand_clash(self, pos, quat, half, center, psi, exclude=()):
        """How badly the hands would hit other blocks when grasping a block at (pos, quat) from a stand at
        `center` facing `psi`: number of hand sample points inside other blocks, in contact and on approach.
        Also returns the blocks that are hit."""
        st = H.HState(center[0], center[1], psi)
        H.apply_crouch(st, 0.6, center, psi)
        g = H.Grasp(quat, half, psi)
        pts = np.vstack([H.hand_points(H.hand_frames(pos, quat, g, st, gap=gap, lift=lift)[2], H.shoulders(st))
                         for gap, lift in ((0.0, 0.0), (0.05, 0.04))])
        n, hit = 0, []
        for j, b in enumerate(self.blocks):
            if j in exclude or b.get("poly") is None or np.linalg.norm(b["pos"][:2] - pos[:2]) > 0.45:
                continue
            loc = (pts - b["pos"]) @ quat2mat(b["quat"])
            k = int(np.all(np.abs(loc) < b["half"] + 0.002, axis=1).sum())
            if k:
                n += k
                hit.append(j)
        return n, hit

    def _ext(self, b):
        R = quat2mat(b["quat"])
        return max(abs(R.T @ np.array([1, 0, 0])) @ b["half"], abs(R.T @ np.array([0, 1, 0])) @ b["half"])

    def pick_clash(self, j):
        """Least hand clash over 8 approach directions for picking block j, and the blocks in the way."""
        b = self.blocks[j]
        best = (np.inf, [])
        for psi in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            c = b["pos"][:2] - (H.PICK_D + self._ext(b)) * np.array([math.cos(psi), math.sin(psi)])
            n, hit = self.hand_clash(b["pos"], b["quat"], b["half"], c, psi, (j,))
            if n < best[0]:
                best = (n, hit)
        return best

    def place_clash(self, bid, cell):
        """Least hand clash over the four grid-square approach directions for setting bid onto cell."""
        pos, b = self.targets[cell]["pos"], self.blocks[bid]
        out = np.inf
        for psi in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            c = pos[:2] - H.PLACE_D * np.array([math.cos(psi), math.sin(psi)])
            out = min(out, self.hand_clash(pos, yaw_quat(0.0), b["half"], c, psi, (bid,))[0])
        return out

    def unblock(self, j, depth=2):
        """If the hands cannot get at block j from any side, clear one of the debris blocks in the way first."""
        for _ in range(depth):
            n, hit = self.pick_clash(j)
            hit = [h for h in hit if h in self.debris and h != j and not self.covered(h)
                   and h not in self.reserved]                      # finale blocks stay where they lie
            if n <= 6 or not hit:
                break
            j = min(hit, key=lambda h: np.linalg.norm(self.blocks[h]["pos"][:2] - self.blocks[j]["pos"][:2]))
        return j

    def stand_pose(self, point, d0, exclude=(), align=None, n=5):
        """The n best collision-free spots at distance ~d0 from `point`, facing it (best first).
        `align(psi, c)` adds a cost."""
        st = self.tl.state
        here, yaw = st.center(), st.yaw
        obstacles = self.polys(exclude)
        local = [o for o in obstacles if np.linalg.norm(o.mean(0) - point) < 1.0]
        centers = np.array([o.mean(0) for o in local]).reshape(-1, 2)
        for margin in (0.02, 0.0):
            best = []                                            # (score, c, psi), sorted
            for d in (d0, d0 + 0.05, d0 + 0.1):
                for psi in np.linspace(-math.pi, math.pi, 32, endpoint=False):
                    c = point - d * np.array([math.cos(psi), math.sin(psi)])
                    if not (X0 + 0.5 < c[0] < X1 - 0.5 and Y0 + 0.5 < c[1] < Y1 - 0.5):
                        continue
                    worst = best[-1][0] if len(best) == n else np.inf
                    score = np.linalg.norm(c - here) + 0.12 * abs(wrap(psi - yaw)) + 0.3 * (d - d0)
                    if score >= worst:                       # the heading cost only adds: prune first
                        continue
                    if align is not None:
                        score += align(psi, c)
                    if score >= worst:
                        continue
                    feet = feet_polys(c, psi)
                    near = [local[k] for k in np.flatnonzero(np.hypot(*(centers - c).T) < 0.4)]
                    if any(overlap(f, o, margin) for f in feet for o in near):
                        continue
                    best = sorted(best + [(score, c, psi)], key=lambda e: e[0])[:n]
            if best:
                return [(c, psi) for _, c, psi in best]
        psi = math.atan2(point[1] - here[1], point[0] - here[0])
        return [(point - d0 * np.array([math.cos(psi), math.sin(psi)]), psi)]

    def foot_hits(self, steps, polys):
        """Footsteps that land on a block or swing through one."""
        if not polys:
            return 0
        centers = np.array([o.mean(0) for o in polys])
        cfg = [f[0][:2].copy() for f in self.tl.state.feet]
        hits = 0
        for i, p, y, _ in steps:
            a, b = cfg[i], p[:2]
            for f in (0.5, 1.0):
                q = a + (b - a) * f
                foot = rect(*q, y, *H.FOOT_HALF)
                if any(overlap(foot, polys[k]) for k in np.flatnonzero(np.hypot(*(centers - q).T) < 0.3)):
                    hits += 1
                    break
            cfg[i] = b.copy()
        return hits

    def go(self, stands, exclude=(), carry=None):
        """Walk to the first of the candidate spots whose footsteps stay clear of the blocks (else to the one
        with the fewest footsteps on blocks)."""
        polys = self.polys(exclude)
        best = None
        for k, (c, psi) in enumerate(stands):
            path = self.planner.path(polys, self.tl.state.center(), c)
            steps = H.plan_footsteps(self.tl.state, path, psi)
            hits = self.foot_hits(steps, polys)
            if best is None or hits < best[0]:
                best = (hits, steps)
            if hits == 0:
                break
        self.walk_hits += best[0]
        self.walk_steps += len(best[1])
        self.tl.walk_steps(best[1], carry)

    def pick(self, bid):
        b = self.blocks[bid]
        ext = self._ext(b)
        R = quat2mat(b["quat"])

        def align(psi, c):
            # face the block so that two of its long side faces point left/right (the hands go there),
            # from a side where the hands do not run into neighbouring blocks
            square = 1.0 - np.abs(R[:, :2].T @ rot_z(psi)[:, 1]).max()
            return square + 0.03 * self.hand_clash(b["pos"], b["quat"], b["half"], c, psi, (bid,))[0]

        stand = self.stand_pose(b["pos"][:2], H.PICK_D + ext, exclude=(bid,), align=align,
                                n=12 if self.strict else 5)
        self.go(stand, exclude=(bid,))
        seg = self.tl.add(H.Pick(self.tl.t_end, self.tl.state, bid, b["pos"], b["quat"], b["half"]))
        b["grasp"] = seg.g
        self.events[bid].append((seg.t_grasp, "held", None))
        self.debris.discard(bid)
        b["poly"] = None

    def put(self, bid, pos, yaw):
        b = self.blocks[bid]
        q_goal = yaw_quat(yaw)

        def align(psi, c):
            # stand square to the grid (the held block then only needs a small yaw correction), on a side
            # where the hands fit between the neighbouring blocks
            square = 0.8 * abs(wrap(psi - yaw - round(wrap(psi - yaw) / (math.pi / 2)) * math.pi / 2))
            return square + 0.03 * self.hand_clash(pos, q_goal, b["half"], c, psi, (bid,))[0]

        stand = self.stand_pose(pos[:2], H.PLACE_D, exclude=(bid,), align=align,
                                n=12 if self.strict else 5)
        self.go(stand, exclude=(bid,), carry=(bid, b["grasp"]))
        seg = self.tl.add(H.Place(self.tl.t_end, self.tl.state, bid, b["grasp"], pos, yaw))
        q = seg.q_t
        self.events[bid].append((seg.t_release, "static", (np.array(pos, float), q)))
        b["pos"], b["quat"] = np.array(pos, float), q
        b["poly"] = box_footprint(b["pos"], q, b["half"])

    def set_aside(self, bid):
        b = self.blocks[bid]
        here = self.tl.state.center()
        x0, x1, y0, y1 = GRID_BAND
        best = None
        for r in np.arange(0.4, 2.5, 0.1):
            for a in np.linspace(-math.pi, math.pi, 24, endpoint=False):
                p = here + r * np.array([math.cos(a), math.sin(a)])
                if x0 - 0.1 < p[0] < x1 + 0.1 and y0 - 0.1 < p[1] < y1 + 0.1:
                    continue
                sq = rect(p[0], p[1], 0, HALF + 0.08, HALF + 0.08)
                if any(overlap(sq, o) for o in self.polys((bid,))):
                    continue
                best = p
                break
            if best is not None:
                break
        h = b["half"][2]
        self.put(bid, np.array([best[0], best[1], h]), 0.0)
        self.debris.add(bid)

    def reserve_finale(self, n):
        """Hold back n cells near the end of the calendar whose matching debris lies closest, so the
        real-time finale is a few short pick-and-place hops."""
        self.finale = {}
        last_col = max(t["col"] for t in self.targets)
        cands = [k for k, f in enumerate(self.filled) if not f and self.targets[k]["col"] >= last_col - 8]
        usable = [j for j in self.debris if not self.covered(j)
                  and not any(overlap(self.blocks[j]["poly"], sq, 0.003) for sq in self.squares)]
        anchor = None
        for _ in range(n):
            best = None
            for k in cands:
                if k in self.finale:
                    continue
                tgt = self.targets[k]["pos"][:2]
                for j in usable:
                    if self.blocks[j]["level"] != self.targets[k]["level"] or j in self.finale.values():
                        continue
                    cost = np.linalg.norm(self.blocks[j]["pos"][:2] - tgt)
                    if anchor is not None:
                        cost += 0.5 * np.linalg.norm(tgt - anchor)
                    if best is None or cost < best[0]:
                        best = (cost, k, j)
            if best is None:
                break
            self.finale[best[1]] = best[2]
            anchor = self.targets[best[1]]["pos"][:2] if anchor is None else anchor

    def run(self, finale=3):
        self.reserve_finale(finale)
        reserved = self.reserved = set(self.finale.values())
        steps = 0
        while not all(self.filled):
            steps += 1
            if steps > 4 * len(self.blocks) + 50:
                raise RuntimeError("rebuild did not converge")
            here = self.tl.state.center()
            open_cells = [k for k, f in enumerate(self.filled) if not f]
            regular = [k for k in open_cells if k not in self.finale]
            if regular:
                open_cells = regular
            front = min(self.targets[k]["col"] for k in open_cells)
            cands = [k for k in open_cells if self.targets[k]["col"] <= front + 3]

            options = []
            for k in cands:
                tgt = self.targets[k]["pos"][:2]
                obs = self.obstructors(k)
                if obs:
                    j = min(obs, key=lambda j: np.linalg.norm(self.blocks[j]["pos"][:2] - here))
                    while self.covered(j):
                        j = self.cover_of(j)
                    cost = np.linalg.norm(self.blocks[j]["pos"][:2] - here)
                    act = ("clear", j, None)
                else:
                    lvl = self.targets[k]["level"]
                    pool = [j for j in self.debris if self.blocks[j]["level"] == lvl]
                    if k in self.finale and self.finale[k] in self.debris:
                        pool = [self.finale[k]]
                    elif any(j not in reserved for j in pool):
                        pool = [j for j in pool if j not in reserved]
                    free = [j for j in pool if not self.covered(j)]
                    if free:
                        carry_w = 3.0 if steps <= 2 else 1.0     # open with short, watchable carries
                        j = min(free, key=lambda j: np.linalg.norm(self.blocks[j]["pos"][:2] - here)
                                + carry_w * np.linalg.norm(self.blocks[j]["pos"][:2] - tgt))
                        act = ("fill", j, k)
                    else:
                        j = self.cover_of(min(pool, key=lambda j: np.linalg.norm(self.blocks[j]["pos"][:2] - here)))
                        while self.covered(j):
                            j = self.cover_of(j)
                        act = ("clear", j, None)
                    cost = (np.linalg.norm(self.blocks[j]["pos"][:2] - here)
                            + (3.0 if steps <= 2 else 1.0) * np.linalg.norm(self.blocks[j]["pos"][:2] - tgt))
                cost += 0.25 * (self.targets[k]["col"] - front)
                options.append((cost, act))
            # the hand-room check is costly: evaluate it for the cheapest options only (it can only add cost)
            self.strict = steps <= 2 or not regular        # actions shown in real time: no feet on blocks
            ranked, n_try = [], 6 if self.strict else 1
            for cost, act in sorted(options, key=lambda o: o[0]):
                if len(ranked) >= n_try and cost >= ranked[n_try - 1][0]:
                    break
                if act[0] == "fill" and self.place_clash(act[1], act[2]) > 4:
                    cost += 0.8                       # hands would not fit between the neighbours yet
                ranked = sorted(ranked + [(cost, act)], key=lambda o: o[0])
            act = ranked[0][1]
            if self.strict and len(ranked) > 1:       # try the cheapest few, keep the first that stays clear
                best = None
                for _, cand in ranked[:n_try]:
                    snap = self._snapshot()
                    h0 = self.walk_hits
                    self._do(cand, open_cells, here)
                    hits = self.walk_hits - h0
                    self._restore(snap)
                    if best is None or hits < best[0]:
                        best = (hits, cand)
                    if hits == 0:
                        break
                act = best[1]
            self._do(act, open_cells, here)
        self.strict = False
        self.log(f"rebuild: done at t={self.tl.t_end:.1f}s after {steps} actions; {self.walk_hits} of "
                 f"{self.walk_steps} footsteps touch a block")

    def _do(self, act, open_cells, here):
        kind, bid, cell = act
        free = self.unblock(bid)
        if free != bid:                               # first move a block that is in the hands' way
            kind, bid, cell = "clear", free, None
        if kind == "clear":
            lvl = self.blocks[bid]["level"]
            dest = [k for k in open_cells if self.targets[k]["level"] == lvl
                    and not [j for j in self.obstructors(k) if j != bid]
                    and not overlap(self.blocks[bid]["poly"], self.squares[k], 0.003)]
            self.pick(bid)
            if dest:
                cell = min(dest, key=lambda k: (self.targets[k]["col"],
                                                np.linalg.norm(self.targets[k]["pos"][:2] - here)))
                self._place_cell(bid, cell)
            else:
                self.set_aside(bid)
        else:
            self.pick(bid)
            self._place_cell(bid, cell)

    def _snapshot(self):
        return (len(self.tl.segs), [dict(b) for b in self.blocks], list(self.filled), set(self.debris),
                {i: list(e) for i, e in self.events.items()}, self.walk_hits, self.walk_steps)

    def _restore(self, snap):
        n, blocks, filled, debris, events, self.walk_hits, self.walk_steps = snap
        del self.tl.segs[n:]
        self.blocks[:] = blocks
        self.filled, self.debris, self.events = filled, debris, events

    def _place_cell(self, bid, cell):
        pos = self.targets[cell]["pos"]
        self.put(bid, pos, 0.0)
        self._settle(bid, cell)
