"""
Ellipse focus "satisfying" video generator.

Balls launch from one focus of an ellipse, bounce off the wall and ALL meet
at the other focus at the same instant (reflection property of the ellipse).
Every time they meet, the number of balls doubles: 2 -> 4 -> 8 -> ... -> 1024.

Output: vertical 1080x1920 MP4 with generated sound.
Usage:  python3 make_video.py [out.mp4]
"""
import math
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------- settings ----------------
W, H = 1080, 1920
FPS = 30
PHASES = 10            # 2,4,...,1024 balls
PHASE_TIME = 2.2       # seconds for one focus -> wall -> focus trip
INTRO = 1.0
OUTRO = 2.5
START_BALLS = 2
TITLE = "Will they all hit the same point?"
OUT = sys.argv[1] if len(sys.argv) > 1 else "ellipse_balls.mp4"

CX, CY = W / 2, H * 0.56
A = 600.0              # semi-major (vertical)
B = 340.0              # semi-minor (horizontal)
C = math.sqrt(A * A - B * B)
F_TOP = np.array([CX, CY - C])
F_BOT = np.array([CX, CY + C])

BG = (0, 0, 0)
RING = (60, 230, 220)
FOCUS_COL = (255, 170, 40)

try:
    FONT_T = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
    FONT_C = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 46)
except OSError:
    FONT_T = FONT_C = ImageFont.load_default()


def hsv(h):
    h = (h % 1.0) * 6
    i = int(h)
    f = h - i
    q, t = 1 - f, f
    r, g, b = [(1, t, 0), (q, 1, 0), (0, 1, t), (0, q, 1), (t, 0, 1), (1, 0, q)][i % 6]
    return int(r * 255), int(g * 255), int(b * 255)


def wall_hit(f, ang):
    """Ray from focus f at angle ang -> point on ellipse and distance."""
    dx, dy = math.cos(ang), math.sin(ang)
    px, py = f[0] - CX, f[1] - CY
    a = dx * dx / (B * B) + dy * dy / (A * A)
    b = 2 * (px * dx / (B * B) + py * dy / (A * A))
    c = px * px / (B * B) + py * py / (A * A) - 1
    t = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    return np.array([f[0] + dx * t, f[1] + dy * t]), t


def make_phase(k):
    n = START_BALLS * 2 ** k
    src, dst = (F_BOT, F_TOP) if k % 2 == 0 else (F_TOP, F_BOT)
    off = math.pi / n * 0.5 + 0.13
    balls = []
    for i in range(n):
        ang = off + 2 * math.pi * i / n
        p, d1 = wall_hit(src, ang)
        balls.append((ang, p, d1, hsv(i / n)))
    return n, src, dst, balls


PHASE_DATA = [make_phase(k) for k in range(PHASES)]
TOTAL = INTRO + PHASES * PHASE_TIME + OUTRO
NFRAMES = int(TOTAL * FPS)


def ellipse_bbox(pad=0):
    return [CX - B - pad, CY - A - pad, CX + B + pad, CY + A + pad]


def draw_frame(t):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # which phase
    tp = t - INTRO
    if tp < 0:
        k, u = 0, 0.0
    elif tp >= PHASES * PHASE_TIME:
        k, u = PHASES - 1, 1.0
    else:
        k = int(tp // PHASE_TIME)
        u = (tp - k * PHASE_TIME) / PHASE_TIME
    # ease so balls slow down near the focus for drama
    s = (u - math.sin(2 * math.pi * u) / (2 * math.pi) * 0.55) * 2 * A
    n, src, dst, balls = PHASE_DATA[k]
    shown_n = n
    if tp >= PHASES * PHASE_TIME:
        shown_n = n * 2  # final split

    # ring
    d.ellipse(ellipse_bbox(), outline=RING, width=6)

    # foci rings
    for f in (F_TOP, F_BOT):
        d.ellipse([f[0] - 16, f[1] - 16, f[0] + 16, f[1] + 16], outline=FOCUS_COL, width=4)

    lw = 2 if n <= 64 else 1
    r = max(5, int(16 / (1 + math.log2(n) * 0.35)))
    positions = []
    for ang, p, d1, col in balls:
        if s <= d1:
            pos = src + np.array([math.cos(ang), math.sin(ang)]) * s
            d.line([tuple(src), tuple(pos)], fill=tuple(c // 2 for c in col), width=lw)
        else:
            v = dst - p
            pos = p + v / np.linalg.norm(v) * (s - d1)
            dim = tuple(c // 3 for c in col)
            d.line([tuple(src), tuple(p)], fill=dim, width=lw)
            d.line([tuple(p), tuple(pos)], fill=tuple(c // 2 for c in col), width=lw)
        positions.append((pos, col))
    for pos, col in positions:
        d.ellipse([pos[0] - r, pos[1] - r, pos[0] + r, pos[1] + r], fill=col)

    # flash when they meet
    if u > 0.97 or (tp >= PHASES * PHASE_TIME):
        fr = 30 + 40 * (u - 0.97) / 0.03 if u <= 1 else 60
        d.ellipse([dst[0] - fr, dst[1] - fr, dst[0] + fr, dst[1] + fr], fill=(255, 255, 255))

    # glow: blur a downscaled copy and add it
    small = img.resize((W // 4, H // 4), Image.BILINEAR).filter(ImageFilter.GaussianBlur(6))
    glow = np.asarray(small.resize((W, H), Image.BILINEAR), dtype=np.uint16)
    out = np.minimum(np.asarray(img, dtype=np.uint16) + glow * 2, 255).astype(np.uint8)
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)

    # texts
    tw = d.textlength(TITLE, font=FONT_T)
    d.text(((W - tw) / 2, 230), TITLE, font=FONT_T, fill=(255, 255, 255))
    lab, num = "balls: ", str(shown_n)
    w1, w2 = d.textlength(lab, font=FONT_C), d.textlength(num, font=FONT_C)
    x0 = (W - w1 - w2) / 2
    d.text((x0, 310), lab, font=FONT_C, fill=(120, 230, 240))
    d.text((x0 + w1, 310), num, font=FONT_C, fill=(255, 200, 90))
    return img


def make_audio(path):
    sr = 44100
    audio = np.zeros(int(TOTAL * sr) + sr, dtype=np.float32)

    def add(t0, freq, dur, vol):
        i0 = int(t0 * sr)
        tt = np.arange(int(dur * sr)) / sr
        tone = np.sin(2 * np.pi * freq * tt) + 0.3 * np.sin(4 * np.pi * freq * tt)
        tone *= np.exp(-tt * 6) * vol
        audio[i0:i0 + len(tone)] += tone[: len(audio) - i0]

    scale = [261.6, 293.7, 329.6, 392.0, 440.0, 523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]
    for k in range(PHASES):
        n, src, dst, balls = PHASE_DATA[k]
        t_start = INTRO + k * PHASE_TIME
        # wall-hit ticks (sample at most 48 so it doesn't get muddy)
        step = max(1, n // 48)
        for ang, p, d1, col in balls[::step]:
            # invert the easing to find when s == d1
            target = d1 / (2 * A)
            lo, hi = 0.0, 1.0
            for _ in range(30):
                m = (lo + hi) / 2
                if m - math.sin(2 * math.pi * m) / (2 * math.pi) * 0.55 < target:
                    lo = m
                else:
                    hi = m
            f = 600 + 900 * (1 - d1 / (2 * A))
            add(t_start + lo * PHASE_TIME, f, 0.12, 0.05)
        # meeting chime, rising pitch each time
        f0 = scale[min(k, len(scale) - 1)]
        for h, v in ((1, 0.5), (1.5, 0.25), (2, 0.2)):
            add(t_start + PHASE_TIME, f0 * h, 1.2, v)
    audio /= max(1e-6, np.abs(audio).max()) * 1.1
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    wav = OUT + ".wav"
    make_audio(wav)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", wav,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", "-shortest", OUT,
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(NFRAMES):
        p.stdin.write(draw_frame(i / FPS).tobytes())
        if i % 60 == 0:
            print(f"frame {i}/{NFRAMES}", flush=True)
    p.stdin.close()
    p.wait()
    import os
    os.remove(wav)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
