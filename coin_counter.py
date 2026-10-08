"""
coin_counter.py - core library for coin detection and counting.

Pipeline (distance-transform method, default)
    1. Resize + grayscale + Gaussian blur          (noise reduction)
    2. Otsu threshold + Canny edge fill            (foreground mask, handles low contrast)
    3. Morphological open/close + hole filling     (clean binary mask)
    4. Distance transform                          (coin centre = peak, height = radius)
    5. Peak detection + non-max suppression        (separates touching coins)
    6. Validation: radius range + background ring  (reject non-coins)
    7. Optional: classify by diameter -> total value
Alternative method: Hough circle transform.
"""
import glob
import json
import math
import os
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Coin:
    x: float
    y: float
    r: float                 # radius in pixels
    name: str = ""
    value: float = 0.0
    id: int = -1


@dataclass
class Params:
    method: str = "distance"         # "distance" or "hough"
    work_width: int = 900            # image is resized to this width for processing
    blur: int = 7                    # Gaussian kernel size (odd)
    min_radius_frac: float = 0.02    # min coin radius as fraction of work_width
    max_radius_frac: float = 0.12    # max coin radius as fraction of work_width
    ring_bg_frac: float = 0.25       # min fraction of background around a candidate coin
    hough_param2: int = 35           # Hough accumulator threshold (lower = more circles)


@dataclass
class Result:
    coins: list
    annotated: np.ndarray
    mask: np.ndarray = None
    stages: dict = field(default_factory=dict)
    total_value: float = None

    @property
    def count(self):
        return len(self.coins)


# --------------------------------------------------------------------------- #
# Pre-processing
# --------------------------------------------------------------------------- #
def preprocess(image, p: Params):
    h, w = image.shape[:2]
    scale = p.work_width / w if w > p.work_width else 1.0
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else image.copy()
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    k = p.blur | 1
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    return small, gray, scale


def _background_is_light(gray):
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return np.median(border) > 127


def _fill_external(m):
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    return filled


def foreground_mask(gray):
    """
    Combine two cues so low-contrast (e.g. silver-on-white) coins are not lost:
      a) Otsu threshold with automatic polarity   -> dark / bright coins
      b) Canny edges -> close gaps -> fill        -> coins whose brightness ~ background
    Then clean up with morphology and fill holes.
    """
    flag = cv2.THRESH_BINARY_INV if _background_is_light(gray) else cv2.THRESH_BINARY
    _, otsu = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)

    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med * 0.35)), int(min(255, 1.33 * med * 0.35)) + 10)
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, ek, iterations=2)
    edge_mask = _fill_external(edge_mask)
    edge_mask = cv2.erode(edge_mask, ek, iterations=1)      # undo the dilation from closing

    m = cv2.bitwise_or(_fill_external(otsu), edge_mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return _fill_external(m)


# --------------------------------------------------------------------------- #
# Detection methods
# --------------------------------------------------------------------------- #
def detect_distance(gray, p: Params):
    """
    Distance-transform peak detection.
    Every pixel of the binary mask gets its distance to the nearest background pixel.
    The centre of a round coin is a peak of that map and the peak HEIGHT is the coin radius
    (largest inscribed circle). Touching coins still give separate peaks because the
    contact "neck" is narrow, so no explicit cutting of the blob is needed.
    """
    h, w = gray.shape
    min_r, max_r = p.min_radius_frac * w, p.max_radius_frac * w
    mask = foreground_mask(gray)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # 1) local maxima of the distance map (window ~ one small coin wide)
    k = int(2 * min_r * 0.8) | 1
    win = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    peaks = (dist == cv2.dilate(dist, win)) & (dist >= min_r * 0.8)
    ys, xs = np.nonzero(peaks)
    cands = sorted(zip(dist[ys, xs], xs, ys), reverse=True)      # biggest first

    # 2) non-maximum suppression (plateaus give several pixels per coin)
    kept = []
    for r, x, y in cands:
        if all(math.hypot(x - kx, y - ky) >= 0.9 * max(r, kr) for kr, kx, ky in kept):
            kept.append((r, x, y))

    # 3) validate: radius range + a ring just outside the circle must be mostly background
    #    (rejects peaks inside big non-circular blobs, tolerates touching neighbours)
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    coins = []
    for r, x, y in kept:
        r += 0.5                                                  # distance map underestimates by ~half a pixel
        if not (min_r <= r <= max_r):
            continue
        rx = np.clip((x + 1.12 * r * np.cos(ang)).astype(int), 0, w - 1)
        ry = np.clip((y + 1.12 * r * np.sin(ang)).astype(int), 0, h - 1)
        if np.mean(mask[ry, rx] == 0) >= p.ring_bg_frac:
            c = Coin(float(x), float(y), float(r))
            c.r = refine_radius(gray, c)
            coins.append(c)
    return coins, mask, dist


def refine_radius(gray, coin, lo=0.80, hi=1.10, n_rays=90):
    """
    Snap the radius to the real coin edge. The binary mask can be a few pixels too big
    (soft shadows), so sample brightness along rays from the centre and take the median
    radius of the strongest intensity change within [lo*r, hi*r].
    """
    h, w = gray.shape
    radii = np.arange(coin.r * lo, coin.r * hi, 0.5)
    ang = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    xs = np.clip(np.rint(coin.x + np.outer(np.cos(ang), radii)).astype(int), 0, w - 1)
    ys = np.clip(np.rint(coin.y + np.outer(np.sin(ang), radii)).astype(int), 0, h - 1)
    prof = gray[ys, xs].astype(np.float32)                # (rays, radii)
    grad = np.abs(np.diff(prof, axis=1))
    best = radii[np.argmax(grad, axis=1)] + 0.25
    strong = grad.max(axis=1) > 4                         # ignore rays with no visible edge
    if strong.sum() < n_rays * 0.3:
        return coin.r
    return float(np.median(best[strong]))


def detect_hough(gray, p: Params):
    h, w = gray.shape
    min_r, max_r = int(p.min_radius_frac * w), int(p.max_radius_frac * w)
    g = cv2.medianBlur(gray, 7)
    found = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 1.6,
                             param1=100, param2=p.hough_param2,
                             minRadius=min_r, maxRadius=max_r)
    coins = []
    if found is not None:
        for x, y, r in found[0]:         # already sorted by accumulator strength
            if all(math.hypot(x - c.x, y - c.y) > 0.7 * min(r, c.r) for c in coins):
                coins.append(Coin(float(x), float(y), float(r)))
    return coins, None, None


# --------------------------------------------------------------------------- #
# Coin classification (size based)
# --------------------------------------------------------------------------- #
def load_denominations(path):
    with open(path) as f:
        return json.load(f)


def available_currencies(folder=None):
    """Return {"INR": "coins_inr.json", ...} for every coins_*.json next to this file."""
    folder = folder or os.path.dirname(os.path.abspath(__file__))
    out = {}
    for path in sorted(glob.glob(os.path.join(folder, "coins_*.json"))):
        try:
            out[load_denominations(path)["currency"]] = path
        except (OSError, KeyError, ValueError):
            continue
    return out


def summarize(coins, denoms):
    """Per-denomination breakdown -> list of dicts (name, display, count, value, subtotal)."""
    meta = {c["name"]: c for c in denoms["coins"]}
    rows = []
    for name, m in meta.items():
        n = sum(1 for c in coins if c.name == name)
        if n:
            rows.append({"name": name, "display": m.get("display", name), "count": n,
                         "value": m["value"], "subtotal": n * m["value"]})
    unknown = sum(1 for c in coins if c.name == "?")
    if unknown:
        rows.append({"name": "?", "display": "Unrecognised", "count": unknown, "value": 0.0, "subtotal": 0.0})
    return rows


def estimate_scale(coins, denoms, ref_coin):
    """pixels-per-mm, assuming the LARGEST detected coin is `ref_coin`."""
    ref = next(c for c in denoms["coins"] if c["name"] == ref_coin)
    biggest = max(coins, key=lambda c: c.r)
    return (2 * biggest.r) / ref["diameter_mm"]


def classify(coins, denoms, px_per_mm, tolerance=0.07):
    for c in coins:
        d_mm = 2 * c.r / px_per_mm
        best = min(denoms["coins"], key=lambda k: abs(k["diameter_mm"] - d_mm))
        err = abs(best["diameter_mm"] - d_mm) / best["diameter_mm"]
        if err <= tolerance:
            c.name, c.value = best["name"], best["value"]
        else:
            c.name, c.value = "?", 0.0
    return sum(c.value for c in coins)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def annotate(image, coins, scale, total_value=None, currency=""):
    out = image.copy()
    inv = 1.0 / scale
    thick = max(2, int(round(image.shape[1] / 450)))
    fs = max(0.6, image.shape[1] / 1400)
    for i, c in enumerate(coins, 1):
        cx, cy, r = int(c.x * inv), int(c.y * inv), int(c.r * inv)
        cv2.circle(out, (cx, cy), r, (0, 255, 0), thick)
        cv2.circle(out, (cx, cy), max(2, thick), (0, 0, 255), -1)
        label = str(c.id if c.id >= 0 else i)
        if c.name:
            label += f":{c.name}"
        cv2.putText(out, label, (cx - r // 2, cy - r - 6), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8,
                    (255, 0, 0), thick, cv2.LINE_AA)
    text = f"Coins: {len(coins)}"
    if total_value is not None:
        text += f" | Total: {total_value:.2f} {currency}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, thick)
    cv2.rectangle(out, (0, 0), (tw + 24, th + 24), (0, 0, 0), -1)
    cv2.putText(out, text, (12, th + 12), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, (255, 255, 255), thick, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Main entry point for one image / frame
# --------------------------------------------------------------------------- #
def count_coins(image, params: Params = None, denoms=None, ref_coin=None, px_per_mm=None):
    p = params or Params()
    small, gray, scale = preprocess(image, p)
    if p.method == "hough":
        coins, mask, dist = detect_hough(gray, p)
    else:
        coins, mask, dist = detect_distance(gray, p)

    total = None
    currency = ""
    if denoms and coins and (px_per_mm or ref_coin):
        pm = px_per_mm or estimate_scale(coins, denoms, ref_coin)
        total = classify(coins, denoms, pm)
        currency = denoms.get("currency", "")

    # sort left-to-right, top-to-bottom for stable numbering
    coins.sort(key=lambda c: (round(c.y / (2 * c.r + 1)), c.x))
    result = Result(coins, annotate(image, coins, scale, total, currency), mask, total_value=total)
    result.stages = {"gray": gray, "mask": mask, "distance": dist, "small": small}
    return result


# --------------------------------------------------------------------------- #
# Video helpers: tracker + count smoothing
# --------------------------------------------------------------------------- #
class CentroidTracker:
    """Greedy nearest-centroid tracker so each coin keeps a persistent ID in video."""

    def __init__(self, max_dist=60, max_missing=8):
        self.tracks = {}            # id -> [x, y, missing]
        self.next_id = 1
        self.max_dist = max_dist
        self.max_missing = max_missing

    def update(self, coins):
        pairs = sorted(
            ((math.hypot(c.x - t[0], c.y - t[1]), ci, tid)
             for ci, c in enumerate(coins) for tid, t in self.tracks.items()),
            key=lambda a: a[0])
        used_c, used_t = set(), set()
        for d, ci, tid in pairs:
            if d > self.max_dist or ci in used_c or tid in used_t:
                continue
            coins[ci].id = tid
            self.tracks[tid] = [coins[ci].x, coins[ci].y, 0]
            used_c.add(ci); used_t.add(tid)
        for ci, c in enumerate(coins):
            if ci not in used_c:
                c.id = self.next_id
                self.tracks[self.next_id] = [c.x, c.y, 0]
                self.next_id += 1
        for tid in list(self.tracks):
            if tid not in used_t and tid not in {c.id for c in coins}:
                self.tracks[tid][2] += 1
                if self.tracks[tid][2] > self.max_missing:
                    del self.tracks[tid]
        return coins


class CountSmoother:
    """Median of the last N per-frame counts -> removes flicker."""

    def __init__(self, n=9):
        self.buf = deque(maxlen=n)

    def update(self, count):
        self.buf.append(count)
        return int(np.median(self.buf))


# --------------------------------------------------------------------------- #
# Debug montage (4-panel pipeline picture)
# --------------------------------------------------------------------------- #
def debug_montage(res):
    def to_bgr(img):
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    st = res.stages
    h, w = st["gray"].shape
    tiles = [("1 gray+blur", st["gray"]), ("2 binary mask", st["mask"]),
             ("3 distance transform", st["distance"]), ("4 result", res.annotated)]
    out = []
    for title, img in tiles:
        if img is None:
            img = np.zeros((h, w), np.uint8)
        t = cv2.resize(to_bgr(img), (w, h))
        cv2.putText(t, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        out.append(t)
    return np.vstack([np.hstack(out[:2]), np.hstack(out[2:])])


# --------------------------------------------------------------------------- #
# Video file processing (used by the Streamlit app)
# --------------------------------------------------------------------------- #
def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _to_h264(src, dst):
    """OpenCV writes mp4v, which browsers cannot play. Re-encode to H.264 if ffmpeg exists."""
    exe = _ffmpeg_exe()
    if exe:
        try:
            subprocess.run([exe, "-y", "-loglevel", "error", "-i", src, "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst],
                           check=True, timeout=300)
            os.remove(src)
            return True
        except Exception:
            pass
    os.replace(src, dst)
    return False


def process_video(src, dst, params=None, max_frames=300, max_width=640, progress=None):
    """Count coins in every frame of a video, draw tracked IDs + stable count, save to dst.

    Returns dict(frames, median, min, max, unique_ids, browser_ready).
    `progress` is an optional callback taking a float 0..1.
    """
    p = params or Params()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = max_width / W if W > max_width else 1.0
    ow, oh = max(2, int(W * s) // 2 * 2), max(2, int(H * s) // 2 * 2)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames, max_frames)
    raw = dst + ".raw.mp4"
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    tracker, smoother = CentroidTracker(max_dist=0.08 * min(ow, oh)), CountSmoother()
    counts = []
    while len(counts) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if s != 1.0:
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        res = count_coins(frame, p)
        tracker.update(res.coins)
        smooth = smoother.update(res.count)
        counts.append(res.count)
        scale = p.work_width / ow if ow > p.work_width else 1.0
        vis = annotate(frame, res.coins, scale)
        cv2.putText(vis, f"Stable count: {smooth}   unique IDs seen: {tracker.next_id - 1}",
                    (12, oh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(vis)
        if progress:
            progress(min(1.0, len(counts) / max(total, 1)))
    cap.release()
    vw.release()
    if not counts:
        raise ValueError("Video has no readable frames")
    ready = _to_h264(raw, dst)
    return {"frames": len(counts), "median": int(np.median(counts)), "min": min(counts),
            "max": max(counts), "unique_ids": tracker.next_id - 1, "browser_ready": ready}
