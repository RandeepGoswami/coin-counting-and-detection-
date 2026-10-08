"""
evaluate.py - measures accuracy on randomly generated scenes (ground truth known).
Gives you real numbers for the "Results" section of your report.

  python evaluate.py            # 40 scenes per setting
  python evaluate.py --n 100

NOTE: these are SYNTHETIC scenes. Also test on real photos and report both.
"""
import argparse
import math

import numpy as np

import coin_counter as cc
import synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--currency", default="INR", choices=["INR", "USD"])
    args = ap.parse_args()
    n = args.n
    denoms = cc.load_denominations(f"coins_{args.currency.lower()}.json")
    settings = [("light bg, separated", True, False), ("light bg, touching", True, True),
                ("dark bg, separated", False, False), ("dark bg, touching", False, True)]

    print(f"{'setting':22s} {'method':9s} {'exact%':>7s} {'MAE':>6s} {'type acc%':>10s}")
    for label, light, touch in settings:
        for method in ("distance", "hough"):
            exact, errs, ok, tot = 0, [], 0, 0
            for seed in range(n):
                img, gt = synth.make_scene(n_coins=6 + seed % 9, light_bg=light,
                                           allow_touch=touch, seed=seed, currency=args.currency)
                res = cc.count_coins(img, cc.Params(method=method))
                exact += res.count == len(gt)
                errs.append(abs(res.count - len(gt)))
                # denomination accuracy: use the true px/mm scale, work-image coordinates
                scale = 900 / img.shape[1] if img.shape[1] > 900 else 1.0
                cc.classify(res.coins, denoms, synth.PX_PER_MM * scale)
                for c in res.coins:
                    g = min(gt, key=lambda t: math.hypot(t[0] * scale - c.x, t[1] * scale - c.y))
                    if math.hypot(g[0] * scale - c.x, g[1] * scale - c.y) < g[2] * scale:   # matched
                        tot += 1
                        ok += c.name == g[3]
            print(f"{label:22s} {method:9s} {100*exact/n:7.1f} {np.mean(errs):6.2f} "
                  f"{100*ok/max(tot,1):10.1f}")


if __name__ == "__main__":
    main()
