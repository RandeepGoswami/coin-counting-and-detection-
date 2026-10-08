# 🪙 Coin Detection & Counting - Streamlit app

Classical computer-vision project (OpenCV + NumPy, no deep learning) that detects, counts and
optionally values coins in **photos, camera snapshots and videos**. Supports **Indian coins
(50p, Rs 1, Rs 2, Rs 5, Rs 10)** and US coins, with a web UI built in Streamlit.

## Deploy on Streamlit Community Cloud (free)
1. Create a new GitHub repository and upload **the contents of this folder** (so `app.py` and
   `requirements.txt` sit at the repo root).
2. Go to <https://share.streamlit.io> -> **Create app** -> choose your repo, branch `main`, main file `app.py`.
3. Click **Deploy**. First build takes a few minutes.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Command line (optional)
```bash
python main.py demo                                        # synthetic data + full run
python main.py image my_coins.jpg --ref-coin Rs10          # largest coin in photo is a Rs 10
python main.py image my_coins.jpg --method hough
python main.py image us.jpg --ref-coin quarter --coins-json coins_usd.json
python main.py video clip.mp4 --show
python main.py webcam                                      # local only (press q)
python evaluate.py --n 40 --currency INR                   # accuracy on synthetic scenes
```
> The CLI needs a display for `--show`/`webcam`; install `opencv-python` instead of the headless build for that.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit web app (upload / camera / video / demo tabs) |
| `coin_counter.py` | Core algorithms: preprocessing, 2 detectors, size classification, tracker, video processing |
| `main.py` | Command-line interface |
| `synth.py` | Synthetic coin scenes/videos with ground truth (INR + USD) |
| `evaluate.py` | Batch accuracy evaluation |
| `coins_inr.json`, `coins_usd.json` | Coin diameters and values - edit or add `coins_<name>.json` for another currency |
| `samples/` | Small demo image/video |
| `.streamlit/config.toml` | Upload size limit and theme |

## How it works (default method)
1. Resize, grayscale, Gaussian blur.
2. Foreground mask = Otsu threshold (auto light/dark) OR Canny edges filled - rescues silver coins on light backgrounds.
3. Morphological open/close + hole filling.
4. **Distance transform** - a round coin's centre is a peak and the peak height is its radius.
5. Peak detection + non-maximum suppression - touching coins still give separate peaks.
6. Validation (radius range, background ring) and radius refinement along rays.
7. Optional value estimate: the largest coin is declared a known denomination (sets px/mm), every coin is matched to the nearest diameter in the currency JSON (7% tolerance, else `?`).

Hough circles is included as a baseline. Videos use a centroid tracker (persistent IDs) and a 9-frame median filter for a flicker-free count.

## Indian coin sizes used
| Coin | Diameter |
|---|---|
| 50 paise | 19 mm |
| Rs 1 | 20 mm |
| Rs 5 | 23 mm |
| Rs 2 | 25 mm |
| Rs 10 | 27 mm |

These are nominal values for current circulation coins. Older series differ (for example the
old large Rs 1 coin), so check your coins and edit `coins_inr.json` if needed. The Rs 20 coin is
12-sided, not round, so it is not supported.

## Results (synthetic data, `python evaluate.py --currency INR`, 40 scenes per row)
| Setting | Distance exact-count | Hough exact-count | Denomination accuracy (distance) |
|---|---|---|---|
| light bg, separated | 100% | 55.0% | 74.9% |
| light bg, touching | 100% | 50.0% | 75.1% |
| dark bg, separated | 100% | 47.5% | 100% |
| dark bg, touching | 100% | 42.5% | 100% |

## Limitations
* Synthetic data is easy; real photos have glare, shadows and perspective - test on your own.
* **Denomination is size-based only.** Indian coins are close in size (Rs 1 vs 50p differ by ~5%, Rs 2 vs Rs 5 by ~8%), so identification is much less reliable than counting, especially on light backgrounds where shadows blur the edge. A colour cue (the bimetallic Rs 10) would be a good next step.
* Assumes a top-down view and mostly non-overlapping coins; stacked coins are under-counted.
* Streamlit Cloud cannot stream a live webcam, so the app uses camera snapshots; video is limited to 300 frames.
