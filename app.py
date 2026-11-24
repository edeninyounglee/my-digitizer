"""
NE111 Plot Digitizer
====================
A single-page Streamlit app for pulling x-y data out of an image of a graph.

Workflow
  1. Upload an image (png, jpg, gif, tiff; colour or greyscale).
  2. Calibrate: click two known points on the x-axis (X1, X2) and two on the
     y-axis (Y1, Y2), then type in their values. Linear or log axes are
     supported for each axis independently, and because the axes are defined
     by two points each, slightly rotated (or skewed) scans are handled too.
  3. Digitize: click on the data. Each dataset gets its own colour and name.
  4. Export: download a CSV in the format required by the handout.

Run locally with:   uv run streamlit run app.py
"""

import csv
import io
import math

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
CAL_KEYS = ["X1", "X2", "Y1", "Y2"]
CAL_HELP = {
    "X1": "a labelled tick on the x-axis (e.g. the left-most one)",
    "X2": "another labelled tick on the x-axis, far from X1",
    "Y1": "a labelled tick on the y-axis (e.g. the bottom one)",
    "Y2": "another labelled tick on the y-axis, far from Y1",
}
CAL_COLOR = (230, 120, 0)
# Distinct, colour-blind-friendly-ish palette for datasets
PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9A6324", "#000075", "#808000",
]


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
def load_image(file) -> Image.Image:
    """Open any common image (png/jpg/gif/tiff, colour or greyscale,
    8- or 16-bit, with or without transparency) and return an RGB image."""
    img = Image.open(file)
    try:
        img.seek(0)  # first frame of animated GIF / multi-page TIFF
    except EOFError:
        pass
    img = ImageOps.exif_transpose(img)  # honour camera/phone orientation

    # 16-bit / 32-bit / float greyscale (common in scientific TIFFs)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        arr = np.asarray(img, dtype=float)
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        arr = (arr - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8), "L")

    # Transparent images: put them on a white background
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)

    return img.convert("RGB")


# --------------------------------------------------------------------------
# Calibration maths
# --------------------------------------------------------------------------
class Calibration:
    """Maps between image pixels and data coordinates.

    The x-axis is the line through pixels X1, X2 and the y-axis the line
    through Y1, Y2. Any pixel P is written as  P = O + a*ex + b*ey  where O is
    where the two axis lines cross, ex = X2-X1 and ey = Y2-Y1. Because ex and
    ey are measured from the image itself, a rotated (or even sheared) scan is
    handled automatically. 'a' is then linearly mapped to x (or log10 x) and
    'b' to y (or log10 y).
    """

    def __init__(self, pts, vals, xlog=False, ylog=False):
        self.xlog, self.ylog = xlog, ylog
        X1, X2, Y1, Y2 = (np.asarray(pts[k], float) for k in CAL_KEYS)
        x1, x2, y1, y2 = (float(vals[k]) for k in CAL_KEYS)

        if xlog and (x1 <= 0 or x2 <= 0):
            raise ValueError("Log x-axis values must be greater than 0.")
        if ylog and (y1 <= 0 or y2 <= 0):
            raise ValueError("Log y-axis values must be greater than 0.")
        if x1 == x2:
            raise ValueError("X1 and X2 need different values.")
        if y1 == y2:
            raise ValueError("Y1 and Y2 need different values.")

        ex, ey = X2 - X1, Y2 - Y1
        if np.linalg.norm(ex) < 1 or np.linalg.norm(ey) < 1:
            raise ValueError("Calibration points on the same axis are too close together.")
        M = np.column_stack([ex, ey])
        if abs(np.linalg.det(M)) < 1e-6 * np.linalg.norm(ex) * np.linalg.norm(ey) * 10:
            raise ValueError("The x- and y-axis points are (nearly) parallel; check your clicks.")

        # Origin O = intersection of line X1 + s*ex and line Y1 + t*ey
        s, _t = np.linalg.solve(np.column_stack([ex, -ey]), Y1 - X1)
        self.O = X1 + s * ex
        self.M, self.Minv = M, np.linalg.inv(M)

        self.a1 = self._ab(X1)[0]
        self.a2 = self._ab(X2)[0]
        self.b1 = self._ab(Y1)[1]
        self.b2 = self._ab(Y2)[1]
        self.u1, self.u2 = self._f(x1, xlog), self._f(x2, xlog)
        self.v1, self.v2 = self._f(y1, ylog), self._f(y2, ylog)

        # Angle of the x-axis relative to horizontal (image y points down)
        self.rotation_deg = math.degrees(math.atan2(-ex[1], ex[0]))

    @staticmethod
    def _f(v, log):
        return math.log10(v) if log else v

    @staticmethod
    def _finv(u, log):
        return 10 ** u if log else u

    def _ab(self, P):
        return self.Minv @ (np.asarray(P, float) - self.O)

    def to_data(self, P):
        a, b = self._ab(P)
        u = self.u1 + (a - self.a1) / (self.a2 - self.a1) * (self.u2 - self.u1)
        v = self.v1 + (b - self.b1) / (self.b2 - self.b1) * (self.v2 - self.v1)
        return self._finv(u, self.xlog), self._finv(v, self.ylog)

    def to_pixel(self, x, y):
        u, v = self._f(x, self.xlog), self._f(y, self.ylog)
        a = self.a1 + (u - self.u1) / (self.u2 - self.u1) * (self.a2 - self.a1)
        b = self.b1 + (v - self.v1) / (self.v2 - self.v1) * (self.b2 - self.b1)
        P = self.O + self.M @ np.array([a, b])
        return float(P[0]), float(P[1])


# --------------------------------------------------------------------------
# CSV export (format from the project handout)
# --------------------------------------------------------------------------
def build_csv(datasets, cal, xlabel, ylabel, sort_x=False) -> str:
    """Row 1: dataset names. Row 2: axis labels. Then x, y pairs."""
    cols = []
    for ds in datasets:
        pts = [cal.to_data(p) for p in ds["points"]]
        if sort_x:
            pts.sort(key=lambda p: p[0])
        cols.append(pts)

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(sum(([ds["name"], ""] for ds in datasets), []))
    w.writerow([xlabel, ylabel] * len(datasets))
    n = max((len(c) for c in cols), default=0)
    for i in range(n):
        row = []
        for c in cols:
            row += [f"{c[i][0]:.6g}", f"{c[i][1]:.6g}"] if i < len(c) else ["", ""]
        w.writerow(row)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def get_font(size):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_view(img, box, disp_w, cal_pts, datasets, active_id, show_axes):
    """Crop `box` from the original image, scale it to disp_w wide and draw
    calibration points and data markers on top. Returns (image, scale)."""
    left, top, right, bottom = box
    s = disp_w / (right - left)
    disp_h = max(1, round((bottom - top) * s))
    view = img.crop(box).resize((disp_w, disp_h), Image.LANCZOS)
    d = ImageDraw.Draw(view)
    font = get_font(18)

    def to_view(p):
        return ((p[0] - left) * s, (p[1] - top) * s)

    # Axis lines through the calibration points
    if show_axes:
        for k1, k2 in (("X1", "X2"), ("Y1", "Y2")):
            if cal_pts.get(k1) and cal_pts.get(k2):
                d.line([to_view(cal_pts[k1]), to_view(cal_pts[k2])], fill=CAL_COLOR, width=1)

    # Calibration markers (crosshair + label)
    for k in CAL_KEYS:
        p = cal_pts.get(k)
        if p:
            x, y = to_view(p)
            r = 11
            d.line([(x - r, y), (x + r, y)], fill=CAL_COLOR, width=3)
            d.line([(x, y - r), (x, y + r)], fill=CAL_COLOR, width=3)
            d.text((x + 6, y - 20), k, fill=CAL_COLOR, font=font,
                   stroke_width=2, stroke_fill=(255, 255, 255))

    # Data points; the active dataset is drawn last and slightly larger
    for ds in sorted(datasets, key=lambda ds: ds["id"] == active_id):
        col = hex_to_rgb(ds["color"])
        r = 7 if ds["id"] == active_id else 5
        for p in ds["points"]:
            x, y = to_view(p)
            d.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=(255, 255, 255), width=2)
    return view, s


def zoom_box(W, H, zoom, cx, cy):
    """Crop rectangle for a zoom factor centred at fractions (cx, cy)."""
    cw, ch = W / zoom, H / zoom
    left = min(max(cx * W - cw / 2, 0), W - cw)
    top = min(max(cy * H - ch / 2, 0), H - ch)
    return (left, top, left + cw, top + ch)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main():
    import streamlit as st
    from streamlit_image_coordinates import streamlit_image_coordinates

    st.set_page_config(page_title="Plot Digitizer", page_icon="📈", layout="wide")
    ss = st.session_state

    # ---------- state helpers ----------
    def new_dataset(name=None):
        i = ss.next_id
        ss.next_id += 1
        ds = {"id": i, "name": name or f"Dataset {i}",
              "color": PALETTE[(i - 1) % len(PALETTE)], "points": []}
        ss.datasets.append(ds)
        ss.active_id = i
        ss.editor_ver += 1
        return ds

    def reset_for_image(img_id):
        ss.img_id = img_id
        ss.cal_pts = {k: None for k in CAL_KEYS}
        ss.datasets = []
        ss.next_id = 1
        ss.editor_ver = 0
        ss.last_click = None
        new_dataset()
        ss.pending_cal_target = "X1"
        ss.pending_mode = "Calibrate axes"

    def active_ds():
        for ds in ss.datasets:
            if ds["id"] == ss.active_id:
                return ds
        return ss.datasets[0]

    # Apply widget values queued by the previous run (must happen before
    # those widgets are created)
    for pend, key in (("pending_cal_target", "cal_target"), ("pending_mode", "mode")):
        if ss.get(pend):
            ss[key] = ss.pop(pend)

    # ---------- header ----------
    st.title("📈 Plot Digitizer")
    st.caption("Extract x-y data from an image of a graph: upload → calibrate the axes → "
               "click your data → download a CSV.")

    # ---------- sidebar: upload, axes, view ----------
    with st.sidebar:
        st.header("1 · Upload image")
        up = st.file_uploader("Graph image", type=["png", "jpg", "jpeg", "gif", "tif", "tiff", "bmp"],
                              help="Colour or greyscale PNG, JPG, GIF or TIFF.")

    if up is None:
        st.info("👈 Start by uploading an image of a plot in the sidebar "
                "(PNG, JPG, GIF or TIFF; colour or greyscale).")
        with st.expander("How does this work?", expanded=True):
            st.markdown(
                "1. **Upload** a picture or scan of a graph.\n"
                "2. **Calibrate**: click two labelled ticks on the x-axis (**X1**, **X2**) and two on "
                "the y-axis (**Y1**, **Y2**), then type what values they are. Tick *Log scale* for log "
                "axes. Rotated scans are fine: the axes are taken from your clicks.\n"
                "3. **Digitize**: switch to *Add data points* and click on each point of a curve. "
                "Make a new dataset for each curve.\n"
                "4. **Export**: name your axes and download the CSV."
            )
        st.stop()

    img_id = f"{up.name}-{up.size}"
    if ss.get("img_id") != img_id:
        try:
            ss.img = load_image(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that image: {e}")
            st.stop()
        reset_for_image(img_id)
    img = ss.img
    W, H = img.size

    with st.sidebar:
        st.caption(f"{up.name} · {W}×{H} px")

        st.header("2 · Axes")
        xlabel = st.text_input("X-axis label", value="x", key="xlabel")
        ylabel = st.text_input("Y-axis label", value="y", key="ylabel")

        st.markdown("**Calibration values**")
        c1, c2 = st.columns(2)
        raw = {
            "X1": c1.text_input("X1 value", "0", key="val_X1"),
            "X2": c2.text_input("X2 value", "1", key="val_X2"),
            "Y1": c1.text_input("Y1 value", "0", key="val_Y1"),
            "Y2": c2.text_input("Y2 value", "1", key="val_Y2"),
        }
        xlog = c1.checkbox("Log x-axis", key="xlog")
        ylog = c2.checkbox("Log y-axis", key="ylog")
        st.caption("Scientific notation like `1e-3` works.")

        st.header("View")
        disp_w = st.slider("Image sharpness (px)", 400, 1400, 700, 50,
                           help="The image always fits the page; higher = sharper but smaller markers.")
        zoom = st.select_slider("Zoom", options=[1, 1.5, 2, 3, 4, 6, 8], value=1,
                                help="Zoom in to click more precisely.")
        if zoom > 1:
            cx = st.slider("Pan left ↔ right", 0.0, 1.0, 0.5, 0.01)
            cy = st.slider("Pan top ↕ bottom", 0.0, 1.0, 0.5, 0.01)
        else:
            cx = cy = 0.5
        show_axes = st.checkbox("Show calibration axis lines", value=True)

    # ---------- calibration object ----------
    cal, cal_error = None, None
    if all(ss.cal_pts.values()):
        try:
            vals = {k: float(raw[k]) for k in CAL_KEYS}
            cal = Calibration(ss.cal_pts, vals, xlog, ylog)
        except ValueError as e:
            cal_error = str(e) if "could not convert" not in str(e) else \
                "Calibration values must be numbers."

    # ---------- main layout ----------
    left_col, right_col = st.columns([3, 2], gap="large")

    with right_col:
        # Step indicator
        n_cal = sum(v is not None for v in ss.cal_pts.values())
        n_pts = sum(len(ds["points"]) for ds in ss.datasets)
        st.markdown(
            "✅ Image &nbsp;·&nbsp; "
            f"{'✅' if cal else '⬜'} Calibrated ({n_cal}/4) &nbsp;·&nbsp; "
            f"{'✅' if n_pts else '⬜'} {n_pts} point(s)"
        )

        mode = st.radio("What does a click do?", ["Calibrate axes", "Add data points"],
                        key="mode", horizontal=True)

        if mode == "Calibrate axes":
            target = st.radio("Next click sets", CAL_KEYS, key="cal_target", horizontal=True,
                              format_func=lambda k: f"{k} {'✓' if ss.cal_pts[k] else ''}")
            st.info(f"Click on **{target}**: {CAL_HELP[target]}. "
                    f"Then enter its value in the sidebar.")
            if st.button("Reset calibration points"):
                ss.cal_pts = {k: None for k in CAL_KEYS}
                ss.pending_cal_target = "X1"
                st.rerun()
        else:
            if not cal:
                st.warning("Axes aren't calibrated yet. You can still click points, but values "
                           "will only appear once calibration is done.")

        if cal_error:
            st.error(cal_error)
        elif cal and abs(cal.rotation_deg) > 0.3:
            st.caption(f"↻ Detected axis rotation: {cal.rotation_deg:.1f}° (corrected automatically)")

        # ---- datasets ----
        st.subheader("Datasets")
        ids = [ds["id"] for ds in ss.datasets]
        names = {ds["id"]: ds["name"] for ds in ss.datasets}
        colors = {ds["id"]: ds["color"] for ds in ss.datasets}
        counts = {ds["id"]: len(ds["points"]) for ds in ss.datasets}
        sel = st.selectbox("Active dataset (clicks go here)", ids, index=ids.index(ss.active_id),
                           format_func=lambda i: f"● {names[i]} ({counts[i]} pts)")
        if sel != ss.active_id:
            ss.active_id = sel
            ss.editor_ver += 1
        ds = active_ds()
        st.markdown(f"<span style='color:{ds['color']};font-size:1.4em'>●</span> "
                    f"colour of **{ds['name']}**", unsafe_allow_html=True)

        new_name = st.text_input("Dataset name", value=ds["name"], key=f"name_{ds['id']}")
        if new_name.strip() and new_name != ds["name"]:
            ds["name"] = new_name.strip()
            st.rerun()

        b1, b2 = st.columns(2)
        b3, b4 = st.columns(2)
        if b1.button("➕ New dataset", use_container_width=True, help="Start a new dataset (new colour)"):
            new_dataset()
            ss.pending_mode = "Add data points"
            st.rerun()
        if b2.button("↶ Undo point", use_container_width=True, disabled=not ds["points"],
                     help="Remove the last point of this dataset"):
            ds["points"].pop()
            ss.editor_ver += 1
            st.rerun()
        if b3.button("🧹 Clear points", use_container_width=True, disabled=not ds["points"],
                     help="Remove all points of this dataset"):
            ds["points"].clear()
            ss.editor_ver += 1
            st.rerun()
        if b4.button("🗑 Delete dataset", use_container_width=True, help="Delete this dataset"):
            ss.datasets = [d for d in ss.datasets if d["id"] != ds["id"]]
            if not ss.datasets:
                new_dataset()
            ss.active_id = ss.datasets[-1]["id"]
            ss.editor_ver += 1
            st.rerun()

        # ---- live editable table (bonus) ----
        st.subheader(f"Points in “{ds['name']}”")
        if cal:
            data = [cal.to_data(p) for p in ds["points"]]
            df = pd.DataFrame(data, columns=["x", "y"], dtype=float)
            edited = st.data_editor(
                df, num_rows="dynamic", use_container_width=True, hide_index=False,
                key=f"editor_{ds['id']}_{ss.editor_ver}",
                column_config={
                    "x": st.column_config.NumberColumn(xlabel or "x", format="%.6g"),
                    "y": st.column_config.NumberColumn(ylabel or "y", format="%.6g"),
                },
            )
            st.caption("Edit a value, add a row, or select rows and delete them; "
                       "the marks on the plot follow.")
            edited = edited.reset_index(drop=True)
            if not edited.equals(df):
                new_pts, bad = [], 0
                for x, y in edited[["x", "y"]].itertuples(index=False):
                    if pd.isna(x) or pd.isna(y):
                        continue  # half-filled new row; wait for both values
                    try:
                        new_pts.append(cal.to_pixel(float(x), float(y)))
                    except (ValueError, OverflowError):
                        bad += 1
                complete = edited[["x", "y"]].notna().all(axis=1).all()
                if bad:
                    st.warning(f"{bad} row(s) can't be shown on a log axis (values must be > 0).")
                elif complete or len(new_pts) != len(ds["points"]):
                    ds["points"] = new_pts
                    ss.editor_ver += 1
                    st.rerun()
        else:
            st.caption("The table appears once the axes are calibrated.")

        # ---- export ----
        st.subheader("3 · Export")
        if not cal:
            st.caption("Calibrate the axes to enable the download.")
        elif n_pts == 0:
            st.caption("Click some data points to enable the download.")
        else:
            sort_x = st.checkbox("Sort each dataset by x", value=False)
            exportable = [d for d in ss.datasets if d["points"]]
            csv_text = build_csv(exportable, cal, xlabel, ylabel, sort_x)
            st.download_button("⬇️ Download CSV", csv_text, file_name="digitized_data.csv",
                               mime="text/csv", type="primary", use_container_width=True)
            with st.expander("Preview CSV"):
                st.code(csv_text, language=None)

    # ---------- image + clicks ----------
    with left_col:
        box = zoom_box(W, H, zoom, cx, cy)
        view, scale = render_view(img, box, disp_w, ss.cal_pts, ss.datasets,
                                  ss.active_id, show_axes)
        click = streamlit_image_coordinates(view, key="plot_click", cursor="crosshair",
                                            use_column_width="always")
        hint = ("Click to place " + ss.get("cal_target", "X1")) if mode == "Calibrate axes" \
            else f"Click to add a point to {active_ds()['name']}"
        st.caption(f"{hint} · zoom {zoom}×")

    # Process a new click (each click is processed exactly once)
    if click:
        sig = (click.get("x"), click.get("y"), click.get("unix_time"), img_id)
        if sig != ss.last_click:
            ss.last_click = sig
            # Account for the browser possibly displaying the image at a different size
            fx = view.width / click.get("width", view.width) if click.get("width") else 1
            fy = view.height / click.get("height", view.height) if click.get("height") else 1
            px = box[0] + click["x"] * fx / scale
            py = box[1] + click["y"] * fy / scale
            if mode == "Calibrate axes":
                tgt = ss.get("cal_target", "X1")
                ss.cal_pts[tgt] = (px, py)
                nxt = [k for k in CAL_KEYS if ss.cal_pts[k] is None]
                if nxt:
                    ss.pending_cal_target = nxt[0]
                else:
                    ss.pending_mode = "Add data points"
            else:
                active_ds()["points"].append((px, py))
                ss.editor_ver += 1
            st.rerun()


if __name__ == "__main__":
    main()
