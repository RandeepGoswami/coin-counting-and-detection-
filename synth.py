"""
synth.py - generates realistic-looking synthetic coin scenes WITH ground truth,
so the project runs and can be evaluated even without your own photos.
"""
import math
import random

import cv2
import numpy as np

PX_PER_MM = 4.2
# name, diameter_mm, BGR colour, optional BGR colour of a different centre disc (bimetallic coins)
CURRENCIES = {
    "INR": [
        ("50p",  19.0, (185, 185, 190), None),
        ("Rs1",  20.0, (190, 190, 195), None),
        ("Rs5",  23.0, (150, 175, 195), None),
        ("Rs2",  25.0, (185, 185, 190), None),
        ("Rs10", 27.0, (95, 160, 200), (195, 195, 200)),   # bimetallic: brass ring, steel centre
    ],
    "USD": [
        ("dime",    17.91, (185, 185, 190), None),
        ("penny",   19.05, (70, 120, 185), None),
        ("nickel",  21.21, (170, 175, 175), None),
        ("quarter", 24.26, (195, 195, 200), None),
    ],
}


def draw_coin(img, cx, cy, r, color, inner=None):
    cx, cy, r = int(cx), int(cy), int(r)
    # soft shadow
    sh = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(sh, (cx + r // 8, cy + r // 8), r, 255, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), r / 8).astype(np.float32) / 255.0
    img[:] = (img * (1 - 0.35 * sh[..., None])).astype(np.uint8)
    # body, rim, inner disc, "engraving"
    cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    if inner is not None:
        cv2.circle(img, (cx, cy), int(r * 0.62), inner, -1, cv2.LINE_AA)
    dark = tuple(int(c * 0.65) for c in color)
    light = tuple(min(255, int(c * 1.12)) for c in color)
    cv2.circle(img, (cx, cy), r, dark, max(2, r // 14), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r * 0.78), light, max(1, r // 20), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(r * 0.35), dark, max(1, r // 18), cv2.LINE_AA)
    cv2.line(img, (cx - r // 3, cy + r // 2), (cx + r // 3, cy + r // 2), dark, max(1, r // 16), cv2.LINE_AA)


def _background(w, h, light, rng):
    base = 225 if light else 40
    bg = np.full((h, w, 3), base, np.float32)
    noise = cv2.GaussianBlur(rng.normal(0, 6, (h, w)).astype(np.float32), (0, 0), 25)
    bg += noise[..., None] * 3
    xs = np.linspace(-1, 1, w)[None, :]
    bg += (xs * 10)[..., None]                      # mild lighting gradient
    return np.clip(bg, 0, 255).astype(np.uint8)


def make_scene(n_coins=12, size=(1000, 700), light_bg=True, allow_touch=True, seed=None, currency="INR"):
    """Returns image, ground_truth list[(x, y, r, name)]."""
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    types = CURRENCIES[currency]
    w, h = size
    img = _background(w, h, light_bg, rng)
    placed = []
    tries = 0
    while len(placed) < n_coins and tries < 5000:
        tries += 1
        name, d, col, inner = rnd.choice(types)
        r = d * PX_PER_MM / 2
        x, y = rnd.uniform(r + 10, w - r - 10), rnd.uniform(r + 10, h - r - 10)
        gap = -2 if allow_touch else 6
        if all(math.hypot(x - px, y - py) >= r + pr + gap for px, py, pr, _, _, _ in placed):
            placed.append((x, y, r, name, col, inner))
    for x, y, r, name, col, inner in placed:
        draw_coin(img, x, y, r, col, inner)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = rng.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img, [(x, y, r, name) for x, y, r, name, _, _ in placed]


def make_video(path, n_coins=8, frames=150, size=(960, 640), fps=25, seed=1, currency="INR"):
    """Coins drift and bounce around -> a test video for the tracker/counter."""
    rnd = random.Random(seed)
    w, h = size
    rng = np.random.default_rng(seed)
    bg = _background(w, h, True, rng)
    coins = []
    for _ in range(n_coins):
        name, d, col, inner = rnd.choice(CURRENCIES[currency])
        r = d * PX_PER_MM / 2
        for _ in range(500):
            x, y = rnd.uniform(r + 10, w - r - 10), rnd.uniform(r + 10, h - r - 10)
            if all(math.hypot(x - c["x"], y - c["y"]) > r + c["r"] + 8 for c in coins):
                break
        ang = rnd.uniform(0, 2 * math.pi)
        sp = rnd.uniform(1.5, 3.5)
        coins.append(dict(x=x, y=y, r=r, col=col, inner=inner, vx=sp * math.cos(ang), vy=sp * math.sin(ang)))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(frames):
        for c in coins:
            c["x"] += c["vx"]; c["y"] += c["vy"]
            if c["x"] < c["r"] + 5 or c["x"] > w - c["r"] - 5: c["vx"] *= -1
            if c["y"] < c["r"] + 5 or c["y"] > h - c["r"] - 5: c["vy"] *= -1
        for i, a in enumerate(coins):               # simple collision: swap velocities
            for b in coins[i + 1:]:
                if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < a["r"] + b["r"] + 4:
                    a["vx"], b["vx"] = b["vx"], a["vx"]
                    a["vy"], b["vy"] = b["vy"], a["vy"]
        f = bg.copy()
        for c in coins:
            draw_coin(f, c["x"], c["y"], c["r"], c["col"], c["inner"])
        vw.write(cv2.GaussianBlur(f, (3, 3), 0))
    vw.release()
    return n_coins
