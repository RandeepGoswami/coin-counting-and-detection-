"""
app.py - Streamlit web app for coin detection and counting (Indian + US coins).

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub -> https://share.streamlit.io -> pick app.py
"""
import os
import tempfile

import cv2
import numpy as np
import streamlit as st

import coin_counter as cc
import synth

st.set_page_config(page_title="Coin Counter", page_icon="🪙", layout="wide")

MAX_IMAGE_WIDTH = 1600          # larger uploads are shrunk first (speed / memory on free hosting)
CURRENCIES = cc.available_currencies()


@st.cache_data
def get_denoms(path):
    return cc.load_denominations(path)


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_upload(file):
    data = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > MAX_IMAGE_WIDTH:
        img = cv2.resize(img, None, fx=MAX_IMAGE_WIDTH / w, fy=MAX_IMAGE_WIDTH / w, interpolation=cv2.INTER_AREA)
    return img


# ----------------------------------------------------------------------------- #
# Sidebar settings
# ----------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")
currency = st.sidebar.selectbox("Currency", list(CURRENCIES), index=list(CURRENCIES).index("INR") if "INR" in CURRENCIES else 0)
denoms = get_denoms(CURRENCIES[currency])
labels = {c["name"]: c.get("display", c["name"]) for c in denoms["coins"]}

st.sidebar.markdown("**Estimate total value**")
ref_choice = st.sidebar.selectbox(
    "The LARGEST coin in the photo is…",
    ["(skip - just count)"] + [c["name"] for c in sorted(denoms["coins"], key=lambda c: c["diameter_mm"])],
    index=0, format_func=lambda n: n if n.startswith("(") else labels[n],
    help="Coins are identified by size, so the app needs one known coin as a ruler.")
ref_coin = None if ref_choice.startswith("(") else ref_choice

method = st.sidebar.radio("Detection method", ["distance", "hough"],
                          format_func=lambda m: "Distance transform (recommended)" if m == "distance" else "Hough circles (baseline)")
with st.sidebar.expander("Advanced tuning"):
    min_r = st.slider("Min coin radius (% of width)", 1.0, 10.0, 2.0, 0.5) / 100
    max_r = st.slider("Max coin radius (% of width)", 5.0, 25.0, 12.0, 0.5) / 100
    ring_bg = st.slider("Background around coin", 0.05, 0.60, 0.25, 0.05,
                        help="Lower = accept more crowded coins")
    show_pipeline = st.checkbox("Show processing pipeline", value=False)

params = cc.Params(method=method, min_radius_frac=min_r, max_radius_frac=max_r, ring_bg_frac=ring_bg)


# ----------------------------------------------------------------------------- #
# Shared result renderer
# ----------------------------------------------------------------------------- #
def show_result(img, key):
    with st.spinner("Detecting coins…"):
        res = cc.count_coins(img, params, denoms if ref_coin else None, ref_coin)
    left, right = st.columns([3, 2])
    left.image(to_rgb(res.annotated), caption="Detected coins", use_container_width=True)
    with right:
        c1, c2 = st.columns(2)
        c1.metric("Coins detected", res.count)
        if res.total_value is not None:
            c2.metric("Estimated total", f"{denoms['symbol']} {res.total_value:g}")
            rows = cc.summarize(res.coins, denoms)
            if rows:
                st.table({"Coin": [r["display"] for r in rows], "Count": [r["count"] for r in rows],
                          "Subtotal": [f"{r['subtotal']:g}" for r in rows]})
            if any(c.name == "?" for c in res.coins):
                st.warning("Some coins were not recognised - check the reference coin choice.")
        else:
            st.info("Pick the largest coin type in the sidebar to also get the total value.")
        ok, buf = cv2.imencode(".png", res.annotated)
        if ok:
            st.download_button("⬇️ Download annotated image", buf.tobytes(), "coins_result.png", "image/png", key=f"dl_{key}")
    if show_pipeline:
        st.image(to_rgb(cc.debug_montage(res)), caption="gray → mask → distance transform → result",
                 use_container_width=True)
    if res.count == 0:
        st.warning("No coins found. Try a plain background, top-down angle, or adjust the radius sliders.")


# ----------------------------------------------------------------------------- #
# Page
# ----------------------------------------------------------------------------- #
st.title("🪙 Coin Detection & Counting")
st.caption("Classical computer vision (OpenCV) - no deep learning. Works with Indian and US coins.")

tab_img, tab_cam, tab_vid, tab_demo = st.tabs(["📷 Upload image", "🤳 Camera", "🎞️ Video", "🧪 Demo"])

with tab_img:
    up = st.file_uploader("Upload a photo of coins", type=["jpg", "jpeg", "png", "webp"], key="img")
    if up:
        img = decode_upload(up)
        if img is None:
            st.error("Could not read that image.")
        else:
            show_result(img, "img")
    else:
        st.write("Tips: top-down photo, plain contrasting background, even light, coins mostly not overlapping.")

with tab_cam:
    shot = st.camera_input("Take a picture of the coins")
    if shot:
        img = decode_upload(shot)
        if img is not None:
            show_result(img, "cam")

with tab_vid:
    st.write("Upload a short clip (first 300 frames are processed). Each coin gets a persistent ID and the count is smoothed over 9 frames.")
    vid = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="vid")
    if vid and st.button("Process video", type="primary"):
        suffix = os.path.splitext(vid.name)[1] or ".mp4"
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in" + suffix), os.path.join(tmp, "out.mp4")
            with open(src, "wb") as f:
                f.write(vid.getbuffer())
            bar = st.progress(0.0, text="Processing frames…")
            try:
                stats = cc.process_video(src, dst, params, progress=lambda x: bar.progress(x, text="Processing frames…"))
            except ValueError as e:
                bar.empty()
                st.error(str(e))
            else:
                bar.empty()
                m = st.columns(4)
                m[0].metric("Frames", stats["frames"])
                m[1].metric("Median count", stats["median"])
                m[2].metric("Min / max", f"{stats['min']} / {stats['max']}")
                m[3].metric("Unique IDs", stats["unique_ids"])
                data = open(dst, "rb").read()
                st.video(data)
                st.download_button("⬇️ Download result video", data, "coins_result.mp4", "video/mp4")

with tab_demo:
    st.write("No coins handy? Generate a synthetic scene with known ground truth.")
    d1, d2, d3, d4 = st.columns(4)
    n = d1.slider("Number of coins", 3, 20, 10)
    light = d2.toggle("Light background", value=True)
    touch = d3.toggle("Allow touching coins", value=True)
    seed = d4.number_input("Seed", 0, 9999, 7)
    demo_cur = currency if currency in synth.CURRENCIES else "INR"
    scene, gt = synth.make_scene(n, light_bg=light, allow_touch=touch, seed=int(seed), currency=demo_cur)
    st.caption(f"Ground truth: {len(gt)} coins ({demo_cur})")
    show_result(scene, "demo")

st.divider()
st.caption("Coin type is estimated from size only, so for value estimation use a top-down photo and pick the correct "
           "reference coin. Counting is much more reliable than denomination.")
