"""Procedural textures: tileable concrete floor and the printed rubber mat that carries the calendar."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def _periodic_noise(n, power, rng):
    """Tileable fractal noise via spectral synthesis (1/f^power)."""
    f = np.fft.fftfreq(n)
    fx, fy = np.meshgrid(f, f)
    r = np.sqrt(fx ** 2 + fy ** 2)
    r[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) / r ** power
    spec[0, 0] = 0
    out = np.real(np.fft.ifft2(spec))
    return (out - out.mean()) / out.std()


def concrete(path, n=1024, seed=4):
    rng = np.random.default_rng(seed)
    base = np.array([104, 102, 98], float)
    big = _periodic_noise(n, 1.6, rng)
    fine = _periodic_noise(n, 0.6, rng)
    v = 1 + 0.045 * big + 0.03 * fine
    img = base[None, None, :] * v[..., None]
    specks = rng.random((n, n)) < 0.004
    img[specks] *= rng.uniform(0.75, 1.15, specks.sum())[:, None]
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6)).save(path)


def _font(size, bold=False):
    names = (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["arial.ttf", "DejaVuSans.ttf"])
    for n in names:
        for d in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", ""):
            try:
                return ImageFont.truetype(os.path.join(d, n) if d else n, size)
            except OSError:
                pass
    return ImageFont.load_default(size)


def mat(path, cells, total, extent, pitch, half, level_rgb, px_per_m=700, seed=5):
    """Rubber mat with the calendar printed on it. extent = (x0, x1, y0, y1) in metres."""
    x0, x1, y0, y1 = extent
    W, H = int((x1 - x0) * px_per_m), int((y1 - y0) * px_per_m)
    rng = np.random.default_rng(seed)
    base = np.array([52, 54, 56], float)
    grain = rng.normal(0, 1, (H // 4 + 1, W // 4 + 1))
    grain = np.array(Image.fromarray(grain.astype(np.float32)).resize((W, H), Image.BILINEAR))
    img = np.clip(base[None, None] * (1 + 0.035 * grain[..., None]), 0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)

    def P(x, y):  # metres -> pixels (image row 0 = far edge of the mat)
        return (x - x0) * px_per_m, (y1 - y) * px_per_m

    ink = (214, 212, 204)
    for c in cells:
        cx, cy = c.col * pitch, (6 - c.row) * pitch
        a, b = P(cx - half - 0.004, cy + half + 0.004), P(cx + half + 0.004, cy - half - 0.004)
        dr.rectangle([a, b], fill=(62, 64, 67), outline=(92, 94, 96), width=2)

    f_lab, f_title = _font(int(0.045 * px_per_m)), _font(int(0.06 * px_per_m), bold=True)
    seen = set()
    for c in cells:
        if c.row == 0 and c.date.day <= 7 and c.date.month not in seen and c.col < 52:
            seen.add(c.date.month)
            dr.text(P(c.col * pitch - half, 6 * pitch + half + 0.075), MONTHS[c.date.month - 1], font=f_lab, fill=ink)
    for row, name in ((1, "Mon"), (3, "Wed"), (5, "Fri")):
        dr.text(P(-0.22, (6 - row) * pitch + 0.028), name, font=f_lab, fill=ink)
    dr.text(P(-half, -half - 0.045), f"{total:,} contributions in the last year", font=f_title, fill=ink)
    # legend
    lx = 5.2 + half
    lw = dr.textlength("More", font=f_lab) / px_per_m
    dr.text(P(lx - lw, -half - 0.05), "More", font=f_lab, fill=ink)
    lx -= lw + 0.02
    for rgb in reversed(level_rgb):
        dr.rectangle([P(lx - 0.045, -half - 0.05), P(lx, -half - 0.095)], fill=tuple(rgb), outline=(92, 94, 96))
        lx -= 0.055
    lw = dr.textlength("Less", font=f_lab) / px_per_m
    dr.text(P(lx - lw - 0.01, -half - 0.05), "Less", font=f_lab, fill=ink)
    # slight print wear
    arr = np.asarray(im).astype(float)
    wear = rng.random(arr.shape[:2]) < 0.02
    arr[wear] = arr[wear] * 0.85 + base * 0.15
    Image.fromarray(arr.astype(np.uint8)).save(path)
