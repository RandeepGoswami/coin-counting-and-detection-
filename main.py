"""
main.py - command line interface

  python main.py demo                                  # generate test data + run everything
  python main.py image coins.jpg                       # count coins in a photo
  python main.py image coins.jpg --method hough        # use Hough circles instead
  python main.py image coins.jpg --ref-coin Rs10       # largest coin is a Rs 10 -> estimate total
  python main.py image coins.jpg --ref-coin quarter --coins-json coins_usd.json   # US coins
  python main.py video clip.mp4 --show                 # count in a video
  python main.py webcam                                # live counting (press q to quit)
"""
import argparse
import os
import sys

import cv2
import numpy as np

import coin_counter as cc
import synth


def build_params(a):
    return cc.Params(method=a.method, min_radius_frac=a.min_radius, max_radius_frac=a.max_radius,
                     ring_bg_frac=a.ring_bg)


debug_montage = cc.debug_montage


def run_image(a):
    img = cv2.imread(a.path)
    if img is None:
        sys.exit(f"Cannot read image: {a.path}")
    denoms = cc.load_denominations(a.coins_json) if a.ref_coin else None
    res = cc.count_coins(img, build_params(a), denoms, a.ref_coin)
    os.makedirs(a.out, exist_ok=True)
    base = os.path.splitext(os.path.basename(a.path))[0]
    out_path = os.path.join(a.out, f"{base}_{a.method}_result.jpg")
    cv2.imwrite(out_path, res.annotated)
    cv2.imwrite(os.path.join(a.out, f"{base}_{a.method}_pipeline.jpg"), debug_montage(res))
    print(f"Method: {a.method} | Coins detected: {res.count}")
    if res.total_value is not None:
        print(f"Estimated total value: {res.total_value:.2f}")
        for c in res.coins:
            print(f"  coin {c.x:.0f},{c.y:.0f}  d={2*c.r:.0f}px -> {c.name}")
    print(f"Saved: {out_path}")
    if a.show:
        cv2.imshow("Coin counter", res.annotated); cv2.waitKey(0)
    return res


def run_video(a, source=None, live=False):
    cap = cv2.VideoCapture(0 if live else a.path)
    if not cap.isOpened():
        sys.exit("Cannot open video source")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W, H = int(cap.get(3)), int(cap.get(4))
    os.makedirs(a.out, exist_ok=True)
    out_path = os.path.join(a.out, "webcam_result.mp4" if live else
                            os.path.splitext(os.path.basename(a.path))[0] + "_result.mp4")
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    params = build_params(a)
    tracker, smoother = cc.CentroidTracker(max_dist=0.08 * min(W, H)), cc.CountSmoother()
    counts, n = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = cc.count_coins(frame, params)
        # tracker works in the resized coordinates used by the detector
        tracker.update(res.coins)
        smooth = smoother.update(res.count)
        counts.append(res.count)
        # redraw with persistent IDs and the smoothed count
        scale = params.work_width / W if W > params.work_width else 1.0
        vis = cc.annotate(frame, res.coins, scale)
        cv2.putText(vis, f"Stable count: {smooth}   unique IDs seen: {tracker.next_id - 1}",
                    (12, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(vis)
        n += 1
        if a.show or live:
            cv2.imshow("Coin counter (q to quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release(); vw.release(); cv2.destroyAllWindows()
    print(f"Frames: {n} | median count: {int(np.median(counts))} | min/max: {min(counts)}/{max(counts)}")
    print(f"Saved: {out_path}")


def run_demo(a):
    os.makedirs("samples", exist_ok=True)
    img, gt = synth.make_scene(n_coins=14, light_bg=True, seed=7)
    cv2.imwrite("samples/demo_coins.jpg", img)
    img2, gt2 = synth.make_scene(n_coins=10, light_bg=False, seed=3)
    cv2.imwrite("samples/demo_coins_dark.jpg", img2)
    n = synth.make_video("samples/demo_video.mp4", n_coins=8)
    print(f"Ground truth: image1={len(gt)}  image2={len(gt2)}  video={n}\n")
    a.out = "output"
    for path, method in [("samples/demo_coins.jpg", "distance"), ("samples/demo_coins.jpg", "hough"),
                         ("samples/demo_coins_dark.jpg", "distance")]:
        a.path, a.method, a.ref_coin, a.show = path, method, "Rs10", False
        run_image(a); print()
    a.path, a.method = "samples/demo_video.mp4", "distance"
    run_video(a)


def main():
    ap = argparse.ArgumentParser(description="Coin detection and counting")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("image", "video", "webcam", "demo"):
        s = sub.add_parser(name)
        if name in ("image", "video"):
            s.add_argument("path")
        s.add_argument("--method", choices=["distance", "hough"], default="distance")
        s.add_argument("--ref-coin", help="name from the coins json that the LARGEST coin is (enables value estimate)")
        s.add_argument("--coins-json", default="coins_inr.json")
        s.add_argument("--min-radius", type=float, default=0.02, help="fraction of image width")
        s.add_argument("--max-radius", type=float, default=0.12, help="fraction of image width")
        s.add_argument("--ring-bg", type=float, default=0.25, help="min background fraction around a coin (lower = accept more crowded coins)")
        s.add_argument("--out", default="output")
        s.add_argument("--show", action="store_true", help="open a preview window")
    a = ap.parse_args()
    {"image": run_image, "video": run_video, "demo": run_demo,
     "webcam": lambda x: run_video(x, live=True)}[a.cmd](a)


if __name__ == "__main__":
    main()
