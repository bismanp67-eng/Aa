"""
Ellipse focus "satisfying" video generator  (v2).

Balls launch from one focus of an ellipse, bounce off the wall and ALL meet at
the other focus at the exact same instant (reflection property of the ellipse).
Every time they meet, the number of balls doubles: 2 -> 4 -> ... -> 1,048,576.

Output: vertical 1080x1920, 60 fps MP4 with generated stereo music.
Palette rule: NO pink / magenta / purple anywhere (hues limited to red..blue).

Usage:  python3 make_video.py [out.mp4]
Needs:  numpy, pillow, pycairo, ffmpeg
"""
import math
import os
import subprocess
import sys
import wave
from multiprocessing import Pool

import cairo
import numpy as np
from PIL import Image, ImageFilter

# ------------------------------------------------------------------ settings
W, H = 1080, 1920
FPS = 60
PHASES = 20                 # 2 -> 1,048,576
SPLAT_N = 2048              # above this, balls are rendered as a particle splat
START_BALLS = 2
TRAVEL = 2.30               # focus -> wall -> focus
HOLD = 0.38                 # pause at the meeting point
PERIOD = TRAVEL + HOLD
INTRO = 0.5
OUTRO = 3.8
TOTAL = INTRO + PHASES * PERIOD + OUTRO
NFRAMES = int(TOTAL * FPS)
OUT = sys.argv[1] if len(sys.argv) > 1 else "ellipse_balls.mp4"

CX, CY = W / 2, 1085.0
A, B = 610.0, 360.0         # semi-major (vertical), semi-minor
C = math.sqrt(A * A - B * B)
F_TOP = np.array([CX, CY - C])
F_BOT = np.array([CX, CY + C])

RING = (0.25, 0.95, 0.92)
GOLD = (1.0, 0.74, 0.22)
EASE_K = 0.88               # <1 keeps motion monotone; slow at both foci


# ------------------------------------------------------------------ colour
def hsv(h, s=0.9, v=1.0):
    h = (h % 1.0) * 6
    i = int(h) % 6
    f = h - int(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def ball_colour(i, n):
    # symmetric sweep red -> orange -> yellow -> green -> cyan -> blue and back.
    # max hue 0.61 (pure-ish blue) so nothing ever drifts into purple/pink.
    # evenly spread over a hand-picked palette so every colour gets equal space
    x = abs(2 * (i + 0.5) / n - 1) if n > 2 else i
    x = x * (len(PALETTE) - 1)
    j = min(int(x), len(PALETTE) - 2)
    f = x - j
    return tuple((1 - f) * a + f * b for a, b in zip(PALETTE[j], PALETTE[j + 1]))


def colours(n):
    if n <= 2:
        return np.array([ball_colour(i, n) for i in range(n)])
    x = np.abs(2 * (np.arange(n) + 0.5) / n - 1) * (len(PALETTE) - 1)
    P = np.array(PALETTE)
    return np.stack([np.interp(x, np.arange(len(P)), P[:, c]) for c in range(3)], 1)


PALETTE = [(1.00, 0.23, 0.19), (1.00, 0.55, 0.10), (1.00, 0.86, 0.15), (0.55, 1.00, 0.25),
           (0.15, 0.95, 0.55), (0.15, 0.90, 1.00), (0.20, 0.55, 1.00), (0.25, 0.35, 1.00)]


# ------------------------------------------------------------------ physics
def ease(u):
    return u - math.sin(2 * math.pi * u) / (2 * math.pi) * EASE_K


def ease_np(u):
    return u - np.sin(2 * np.pi * u) / (2 * np.pi) * EASE_K


def inv_ease(v):
    lo, hi = np.zeros_like(v), np.ones_like(v)
    for _ in range(40):
        m = (lo + hi) / 2
        big = ease_np(m) >= v
        hi = np.where(big, m, hi)
        lo = np.where(big, lo, m)
    return (lo + hi) / 2


class Phase:
    def __init__(self, k):
        self.k = k
        self.n = n = START_BALLS * 2 ** k
        self.src, self.dst = (F_BOT, F_TOP) if k % 2 == 0 else (F_TOP, F_BOT)
        off = 0.21 + math.pi / n * 0.5
        ang = off + 2 * np.pi * np.arange(n) / n
        self.dir = np.stack([np.cos(ang), np.sin(ang)], 1)
        # intersect ray with ellipse
        px, py = self.src[0] - CX, self.src[1] - CY
        dx, dy = self.dir[:, 0], self.dir[:, 1]
        a = dx * dx / B**2 + dy * dy / A**2
        b = 2 * (px * dx / B**2 + py * dy / A**2)
        c = px * px / B**2 + py * py / A**2 - 1
        t = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
        self.d1 = t
        self.p = self.src + self.dir * t[:, None]
        v = self.dst - self.p
        self.dir2 = v / np.linalg.norm(v, axis=1)[:, None]
        self.col = colours(n)
        self.t0 = INTRO + k * PERIOD
        self.meet = self.t0 + TRAVEL
        self.hit_t = self.t0 + inv_ease(self.d1 / (2 * A)) * TRAVEL
        self.r = 15.0 * n ** -0.17
        self.sub = (np.arange(n) if n <= DRAW_MAX else
                    np.sort(np.random.default_rng(k).choice(n, DRAW_MAX, replace=False)))

    def pos(self, t, sub=None):
        u = np.clip((t - self.t0) / TRAVEL, 0, 1)
        s = ease_np(u) * 2 * A
        d1, dr, p, dr2 = self.d1, self.dir, self.p, self.dir2
        if sub is not None:
            d1, dr, p, dr2 = d1[sub], dr[sub], p[sub], dr2[sub]
        s = np.full(len(d1), s)
        before = s <= d1
        p1 = self.src + dr * s[:, None]
        p2 = p + dr2 * (s - d1)[:, None]
        return np.where(before[:, None], p1, p2), s


DRAW_MAX = 131072
PH = [Phase(k) for k in range(PHASES)]
RNG = np.random.default_rng(7)
STARS = np.column_stack([RNG.uniform(0, W, 140), RNG.uniform(0, H, 140),
                         RNG.uniform(0.6, 2.0, 140), RNG.uniform(0.05, 0.35, 140),
                         RNG.uniform(0, 6.28, 140)])
SPARKS = [(RNG.uniform(0, 2 * np.pi, 70), RNG.uniform(120, 620, 70),
           RNG.integers(0, 10_000, 70)) for _ in range(PHASES)]


# ------------------------------------------------------------------ drawing
def ellipse_path(ctx, pad=0.0):
    ctx.save()
    ctx.translate(CX, CY)
    ctx.scale(B + pad, A + pad)
    ctx.arc(0, 0, 1, 0, 2 * math.pi)
    ctx.restore()


def text_center(ctx, parts, y, size, weight=cairo.FONT_WEIGHT_BOLD, family="Inter",
                scale=1.0, alpha=1.0):
    """parts = [(text, (r,g,b)), ...] drawn as one centred line."""
    ctx.save()
    ctx.select_font_face(family, cairo.FONT_SLANT_NORMAL, weight)
    ctx.set_font_size(size * scale)
    widths = [ctx.text_extents(t).x_advance for t, _ in parts]
    x = (W - sum(widths)) / 2
    for (t, col), w in zip(parts, widths):
        # soft shadow
        ctx.set_source_rgba(0, 0, 0, 0.7 * alpha)
        ctx.move_to(x + 3, y + 4)
        ctx.show_text(t)
        ctx.set_source_rgba(*col, alpha)
        ctx.move_to(x, y)
        ctx.show_text(t)
        x += w
    ctx.restore()


def draw_background(ctx, t):
    g = cairo.RadialGradient(CX, CY, 50, CX, CY, 1250)
    g.add_color_stop_rgb(0, 0.03, 0.06, 0.09)
    g.add_color_stop_rgb(1, 0.0, 0.0, 0.0)
    ctx.set_source(g)
    ctx.paint()
    for x, y, r, a, ph in STARS:
        tw = a * (0.6 + 0.4 * math.sin(t * 1.7 + ph))
        ctx.set_source_rgba(0.75, 0.9, 1.0, tw)
        ctx.arc(x, y, r, 0, 2 * math.pi)
        ctx.fill()
    # ellipse interior tint
    ellipse_path(ctx)
    g = cairo.RadialGradient(CX, CY, 0, CX, CY, A)
    g.add_color_stop_rgba(0, 0.02, 0.10, 0.12, 0.55)
    g.add_color_stop_rgba(1, 0.00, 0.03, 0.05, 0.25)
    ctx.set_source(g)
    ctx.fill()


def current(t):
    k = int(np.clip((t - INTRO) // PERIOD, 0, PHASES - 1))
    return k


def draw_scene(ctx, t):
    k = current(t)
    ph = PH[k]
    tau = t - ph.t0
    final = t >= PH[-1].meet + HOLD

    # ---- ring (with pulse on each meeting)
    since_meet = min([t - p.meet for p in PH if t >= p.meet] or [9])
    pulse = math.exp(-since_meet * 4)
    ctx.set_operator(cairo.OPERATOR_ADD)
    for w, a in ((26, 0.06), (14, 0.12), (6, 0.35 + 0.4 * pulse)):
        ellipse_path(ctx)
        ctx.set_line_width(w)
        ctx.set_source_rgba(*RING, a)
        ctx.stroke()
    ellipse_path(ctx)
    ctx.set_line_width(2.2)
    ctx.set_source_rgba(0.85, 1.0, 1.0, 0.95)
    ctx.stroke()

    # ---- wall flares (current + previous phase)
    for p in PH[max(0, k - 1):k + 1]:
        dt = t - p.hit_t
        m = (dt >= 0) & (dt < 0.7)
        if p.n > SPLAT_N or not m.any():
            continue
        fade = np.exp(-dt[m] * 6) * min(1.0, 2.2 / math.sqrt(p.n))
        rad = p.r * 1.2 + dt[m] * 40
        for (x, y), col, f, rr in zip(p.p[m], p.col[m], fade, rad):
            g = cairo.RadialGradient(x, y, 0, x, y, rr * 2.2)
            g.add_color_stop_rgba(0, *col, 0.9 * f)
            g.add_color_stop_rgba(1, *col, 0)
            ctx.set_source(g)
            ctx.arc(x, y, rr * 2.2, 0, 2 * math.pi)
            ctx.fill()

    if not final:
        moving = tau < TRAVEL
        hold_fade = 1.0 if moving else math.exp(-(tau - TRAVEL) * 9)
        pos, s = ph.pos(t)

        # ---- ray lines (thin, additive, dimmer when crowded)
        la = min(0.55, 1.1 / math.sqrt(ph.n)) * hold_fade
        if la > 0.005 and ph.n <= 1024:
            ctx.set_line_width(1.3 if ph.n < 128 else 0.9)
            for i in range(ph.n):
                cr, cg, cb = ph.col[i]
                ctx.set_source_rgba(cr, cg, cb, la)
                ctx.move_to(*ph.src)
                if s[i] > ph.d1[i]:
                    ctx.line_to(*ph.p[i])
                ctx.line_to(*pos[i])
                ctx.stroke()

        # ---- motion tails
        if moving and ph.n <= SPLAT_N:
            steps = 7
            hist = [ph.pos(t - j * 0.014)[0] for j in range(steps)]
            for j in range(steps - 1):
                a = 0.5 * (1 - j / steps)
                ctx.set_line_width(ph.r * 1.5 * (1 - j / steps))
                ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                for i in range(ph.n):
                    ctx.set_source_rgba(*ph.col[i], a)
                    ctx.move_to(*hist[j][i])
                    ctx.line_to(*hist[j + 1][i])
                    ctx.stroke()

        # ---- balls
        ctx.set_operator(cairo.OPERATOR_OVER)
        r = ph.r
        for (x, y), col in zip(pos if ph.n <= SPLAT_N else [], ph.col):
            g = cairo.RadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r)
            g.add_color_stop_rgb(0, 1, 1, 1)
            g.add_color_stop_rgb(0.35, *[0.5 + 0.5 * c for c in col])
            g.add_color_stop_rgb(1, *[0.75 * c for c in col])
            ctx.set_source(g)
            ctx.arc(x, y, r, 0, 2 * math.pi)
            ctx.fill()
        ctx.set_operator(cairo.OPERATOR_ADD)

    # ---- foci markers
    for f in (F_TOP, F_BOT):
        target = (f == PH[k].dst).all() and not final
        a = 0.55 + (0.45 * (0.5 + 0.5 * math.sin(t * 8)) if target else 0)
        ctx.set_source_rgba(*GOLD, a)
        ctx.set_line_width(4)
        ctx.arc(f[0], f[1], 17, 0, 2 * math.pi)
        ctx.stroke()

    # ---- meeting effects: flash, shockwave, sparks
    for p in PH:
        dt = t - p.meet
        if dt < -0.12 or dt > 1.6:
            continue
        x, y = p.dst
        if dt < 0:      # brightening right before impact
            f = (dt + 0.12) / 0.12
            g = cairo.RadialGradient(x, y, 0, x, y, 40)
            g.add_color_stop_rgba(0, 1, 1, 1, 0.6 * f)
            g.add_color_stop_rgba(1, 1, 1, 1, 0)
            ctx.set_source(g)
            ctx.arc(x, y, 40, 0, 2 * math.pi)
            ctx.fill()
            continue
        big = 1.0 + 0.06 * p.k
        fl = math.exp(-dt * 7)
        g = cairo.RadialGradient(x, y, 0, x, y, 140 * big)
        g.add_color_stop_rgba(0, 1, 1, 1, fl)
        g.add_color_stop_rgba(0.25, 1.0, 0.85, 0.45, 0.6 * fl)
        g.add_color_stop_rgba(1, 1.0, 0.6, 0.1, 0)
        ctx.set_source(g)
        ctx.arc(x, y, 140 * big, 0, 2 * math.pi)
        ctx.fill()
        for j, (spd, w) in enumerate(((900, 5), (600, 3))):
            rr = spd * dt * (1 - 0.3 * dt) * big
            a = max(0, 1 - dt / 1.1) ** 2
            ctx.set_line_width(w)
            ctx.set_source_rgba(*(RING if j == 0 else GOLD), a * 0.8)
            ctx.arc(x, y, rr, 0, 2 * math.pi)
            ctx.stroke()
        angs, spds, seeds = SPARKS[p.k]
        d = spds * (1 - np.exp(-dt * 3.5)) * big
        a = max(0, 1 - dt / 1.4)
        for an, dd, sd in zip(angs, d, seeds):
            sx, sy = x + math.cos(an) * dd, y + math.sin(an) * dd
            col = hsv(0.61 * ((sd % 100) / 100), 0.7, 1)
            ctx.set_source_rgba(*col, a)
            ctx.arc(sx, sy, 2.5 + 2 * a, 0, 2 * math.pi)
            ctx.fill()

    # ---- grand finale: last balls explode outward
    if final and PH[-1].n <= SPLAT_N:
        p = PH[-1]
        dt = t - (p.meet + HOLD)
        x, y = p.dst
        n = p.n
        rng = np.random.default_rng(3)
        ang = rng.uniform(0, 2 * np.pi, n)
        spd = rng.uniform(80, 900, n)
        d = spd * (1 - np.exp(-dt * 1.6))
        a = max(0.0, 1 - dt / 3.0)
        for an, dd, col in zip(ang, d, p.col):
            ctx.set_source_rgba(*col, a)
            ctx.arc(x + math.cos(an) * dd, y + math.sin(an) * dd * 1.4, 3.6, 0, 2 * math.pi)
            ctx.fill()
    ctx.set_operator(cairo.OPERATOR_OVER)


def draw_text(ctx, t):
    k = current(t)
    # count shown: doubles at each meeting
    n = START_BALLS
    last_meet = None
    for p in PH:
        if t >= p.meet:
            n = p.n * 2 if p.k < PHASES - 1 else p.n
            last_meet = p.meet
    n = PH[k].n if t < PH[k].meet else n
    pop = 1.0
    if last_meet is not None and t - last_meet < 0.35 and t < PH[-1].meet + 0.01 + 0.35:
        pop = 1.0 + 0.45 * math.exp(-(t - last_meet) * 10)

    title_a = min(1.0, t / 0.3)
    text_center(ctx, [("Will they ", (1, 1, 1)), ("ALL", GOLD), (" hit", (1, 1, 1))],
                205, 76, cairo.FONT_WEIGHT_BOLD, "Inter Display", alpha=title_a)
    text_center(ctx, [("the same point?", (1, 1, 1))], 295, 76,
                cairo.FONT_WEIGHT_BOLD, "Inter Display", alpha=title_a)
    text_center(ctx, [("balls: ", (0.45, 0.92, 0.95)), (f"{n:,}", GOLD)], 392, 54,
                scale=pop, alpha=title_a)

    end = PH[-1].meet + HOLD + 0.4
    if t > end:
        a = min(1.0, (t - end) / 0.4)
        sc = 1 + 0.3 * math.exp(-(t - end) * 8)
        text_center(ctx, [("YES. ", GOLD), ("Every single time.", (1, 1, 1))],
                    1800, 62, cairo.FONT_WEIGHT_BOLD, "Inter Display", scale=sc, alpha=a)


FIN_RNG = np.random.default_rng(3)
FIN_M = 30000   # particles shown in the final burst (the rest are too dense to see)
FIN_IDX = FIN_RNG.choice(PH[-1].n, FIN_M, replace=False)
FIN_ANG = FIN_RNG.uniform(0, 2 * np.pi, FIN_M).astype(np.float32)
FIN_SPD = (950 - 450 * FIN_RNG.uniform(0, 1, FIN_M) ** 3).astype(np.float32)


FIN_COL = colours(4096)[(FIN_ANG / (2 * np.pi) * 4096).astype(int) % 4096].astype(np.float32)


def splat_points(buf, xy, col, w):
    """additive splat of points into float HxWx3 buffer (one bincount per channel)"""
    x, y = xy[:, 0], xy[:, 1]
    ok = (x >= 0) & (x < W - 1) & (y >= 0) & (y < H - 1)
    w = (w[ok] if np.ndim(w) else np.float32(w))
    idx = y[ok].astype(np.int32) * W + x[ok].astype(np.int32)
    col = col[ok]
    flat = buf.reshape(-1, 3)
    for c in range(3):
        flat[:, c] += np.bincount(idx, weights=col[:, c] * w, minlength=W * H)


# at most this many points are drawn per sample; above it a fixed random subset
# is drawn with proportionally more weight (looks identical, ~10x faster)
DRAW_MAX = 131072


def splat_layer(t):
    """particle rendering for huge ball counts; returns float buffer or None"""
    k = current(t)
    ph = PH[k]
    final = t >= PH[-1].meet + HOLD
    buf = None
    if final and PH[-1].n > SPLAT_N:
        p = PH[-1]
        dt = t - (p.meet + HOLD)
        d = FIN_SPD * (1 - np.exp(-dt * 1.6))
        xy = np.stack([p.dst[0] + np.cos(FIN_ANG) * d, p.dst[1] + np.sin(FIN_ANG) * d * 1.4], 1)
        a = max(0.0, 1 - dt / 3.0)
        buf = np.zeros((H, W, 3), np.float32)
        splat_points(buf, xy, FIN_COL, 255 * a * 0.8)
        return buf
    if ph.n <= SPLAT_N:
        return None
    buf = np.zeros((H, W, 3), np.float32)
    tau = t - ph.t0
    base = 255 * min(1.0, 1200 / ph.n)
    if tau < TRAVEL:
        steps = 16
        sub = ph.sub
        xs, cs, ws = [], [], []
        for j in range(steps):
            xy, _ = ph.pos(t - j * 0.0035, sub)
            xs.append(xy)
            cs.append(ph.col[sub])
            ws.append(np.full(len(sub), base * ph.n / len(sub) * (1 - j / steps) ** 1.5, np.float32))
        splat_points(buf, np.concatenate(xs), np.concatenate(cs), np.concatenate(ws))
    else:
        splat_points(buf, ph.pos(t, ph.sub)[0], ph.col[ph.sub], base * ph.n / len(ph.sub))
    # wall flashes where balls bounce
    for p in PH[max(0, k - 1):k + 1]:
        if p.n <= SPLAT_N:
            continue
        dt = t - p.hit_t[p.sub]
        m = (dt >= 0) & (dt < 0.5)
        if m.any():
            sc = p.n / len(p.sub)
            splat_points(buf, p.p[p.sub][m], p.col[p.sub][m],
                         (255 * min(1.0, 8000 / p.n) * 1.5 * sc * np.exp(-dt[m] * 7)).astype(np.float32))
    return buf


def add_glow(arr, extra=None):
    """arr: HxWx3 uint8 (+ optional float layer) -> bloom-added uint8"""
    if extra is not None:
        # thicken the particle layer so a million dots read as a glowing wave
        e8 = Image.fromarray(np.minimum(extra / 3, 255).astype(np.uint8))
        thick = np.asarray(e8.filter(ImageFilter.GaussianBlur(3.5)), np.float32) * 3 * 2.2
        arr = np.minimum(arr.astype(np.float32) + np.minimum(extra, 255) + thick, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    s1 = img.resize((W // 4, H // 4), Image.BOX).filter(ImageFilter.GaussianBlur(4))
    s2 = img.resize((W // 12, H // 12), Image.BOX).filter(ImageFilter.GaussianBlur(4))
    g1 = np.asarray(s1.resize((W, H), Image.BILINEAR), dtype=np.float32)
    g2 = np.asarray(s2.resize((W, H), Image.BILINEAR), dtype=np.float32)
    out = arr.astype(np.float32) + g1 * 1.1 + g2 * 1.3
    # soft tone-map instead of hard clipping
    out = 255 * (1 - np.exp(-out / 230))
    return out.astype(np.uint8)


def surf_rgb(surf):
    buf = np.ndarray((H, W, 4), np.uint8, surf.get_data())
    return buf[:, :, [2, 1, 0]].copy()


def render(i):
    t = i / FPS
    surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
    ctx = cairo.Context(surf)
    draw_background(ctx, t)
    draw_scene(ctx, t)
    surf.flush()
    rgb = add_glow(surf_rgb(surf), splat_layer(t))
    # text drawn after bloom so it stays crisp
    surf2 = cairo.ImageSurface.create_for_data(
        np.dstack([rgb[:, :, 2], rgb[:, :, 1], rgb[:, :, 0],
                   np.full((H, W), 255, np.uint8)]).copy(), cairo.FORMAT_RGB24, W, H)
    ctx2 = cairo.Context(surf2)
    draw_text(ctx2, t)
    surf2.flush()
    return surf_rgb(surf2).tobytes()


# ------------------------------------------------------------------ audio
SR = 44100


def make_audio(path):
    N = int((TOTAL + 1) * SR)
    L = np.zeros(N, np.float32)
    R = np.zeros(N, np.float32)

    def put(t0, sig, pan=0.0):
        i0 = int(t0 * SR)
        if i0 >= N:
            return
        sig = sig[: N - i0]
        L[i0:i0 + len(sig)] += sig * math.sqrt(0.5 * (1 - pan))
        R[i0:i0 + len(sig)] += sig * math.sqrt(0.5 * (1 + pan))

    def marimba(f, dur=0.6, vol=0.2):
        tt = np.arange(int(dur * SR)) / SR
        s = (np.sin(2 * np.pi * f * tt) * np.exp(-tt * 7)
             + 0.35 * np.sin(2 * np.pi * f * 3.93 * tt) * np.exp(-tt * 22)
             + 0.12 * np.sin(2 * np.pi * f * 9.2 * tt) * np.exp(-tt * 45))
        s *= np.minimum(1, tt / 0.002)
        return (s * vol).astype(np.float32)

    def bell(f, dur=3.0, vol=0.3):
        tt = np.arange(int(dur * SR)) / SR
        s = np.zeros_like(tt)
        for h, a, d in ((1, 1, 1.6), (2, 0.5, 2.2), (2.76, 0.25, 3), (5.4, 0.12, 5), (0.5, 0.35, 1.2)):
            s += a * np.sin(2 * np.pi * f * h * tt) * np.exp(-tt * d)
        s *= np.minimum(1, tt / 0.003)
        return (s * vol).astype(np.float32)

    def thump(vol=0.6):
        tt = np.arange(int(0.6 * SR)) / SR
        f = 45 + 90 * np.exp(-tt * 18)
        ph = 2 * np.pi * np.cumsum(f) / SR
        return (np.sin(ph) * np.exp(-tt * 6) * vol).astype(np.float32)

    def riser(dur, f0, vol=0.08):
        tt = np.arange(int(dur * SR)) / SR
        f = f0 * (1 + 1.0 * (tt / dur) ** 2)
        ph = 2 * np.pi * np.cumsum(f) / SR
        env = (tt / dur) ** 2.5
        return ((np.sin(ph) + 0.4 * np.sin(2.01 * ph)) * env * vol).astype(np.float32)

    # C major pentatonic, several octaves
    penta = [0, 2, 4, 7, 9]
    notes = [261.63 * 2 ** ((o * 12 + s) / 12) for o in range(-1, 3) for s in penta]
    roots = [130.81 * 2 ** (s / 12) for s in [0, 2, 4, 7, 9, 12, 14, 16, 19, 21,
                                              24, 26, 28, 31, 33, 36, 38, 40, 43, 45]]

    for p in PH:
        # wall hits -> pentatonic arpeggio, pitch by vertical position, pan by x
        order = np.argsort(p.hit_t)
        bidx = (p.hit_t[order] / 0.03).astype(np.int64)
        _, first, counts = np.unique(bidx, return_index=True, return_counts=True)
        for f0, cnt in zip(first, counts):
            i = order[f0 + cnt // 2]
            idx = range(cnt)
            x, y = p.p[i]
            hgt = 1 - (y - (CY - A)) / (2 * A)
            note = notes[int(np.clip(hgt * (len(notes) - 1), 0, len(notes) - 1))]
            vol = 0.09 * (1 + 0.25 * math.log2(len(idx))) / (1 + 0.06 * p.k)
            put(p.hit_t[i], marimba(note, vol=vol), pan=(x - CX) / B * 0.7)
        put(p.meet - 0.9, riser(0.9, roots[p.k] * 2))
        # meeting: bell chord + thump, rising each phase
        r = roots[p.k]
        for f, v in ((r, 0.32), (r * 1.5, 0.2), (r * 2, 0.22), (r * 2.5, 0.1)):
            put(p.meet, bell(f, vol=v))
        put(p.meet, thump(0.55 + 0.015 * p.k))

    # finale chord
    fin = PH[-1].meet + HOLD
    for f in (261.63, 329.63, 392.0, 523.25, 659.26, 783.99):
        put(fin, bell(f, dur=4.0, vol=0.18))
    put(fin, thump(0.8))

    # simple stereo reverb (feedback combs)
    def comb(x, d, g):
        y = x.copy()
        for i in range(d, len(x), d):
            j = min(i + d, len(x))
            y[i:j] += g * y[i - d:j - d]
        return y

    def verb(x, ds):
        return sum(comb(x, int(d * SR), 0.55) for d in ds) / len(ds)

    L = L + 0.35 * verb(L, (0.0297, 0.0371, 0.0411, 0.0437))
    R = R + 0.35 * verb(R, (0.0313, 0.0353, 0.0429, 0.0451))
    st = np.stack([L, R], 1)
    st /= np.abs(st).max() * 1.12
    st = np.tanh(st * 1.2) / np.tanh(1.2)
    pcm = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ------------------------------------------------------------------ main
def main():
    wav = OUT + ".wav"
    make_audio(wav)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
           "-crf", "19", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
           "-shortest", OUT]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count()) as pool:
        for i, fr in enumerate(pool.imap(render, range(NFRAMES), chunksize=4)):
            p.stdin.write(fr)
            if i % 120 == 0:
                print(f"frame {i}/{NFRAMES}", flush=True)
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print("done ->", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--stills":
        for tt in map(float, sys.argv[3:]):
            Image.frombytes("RGB", (W, H), render(int(tt * FPS))).resize((540, 960)).save(
                f"{OUT}_{tt:.2f}.png")
    else:
        main()
